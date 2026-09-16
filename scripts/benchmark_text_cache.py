#!/usr/bin/env python
"""Verify and benchmark the fixed-prompt SAM3 text-encoder cache.

Both phases run in one process on a single model instance:

* **uncached** - the SAM3 text encoder runs on every ``segment()`` call
  (original behavior, ``segmenter.use_text_cache = False``)
* **cached**   - the ``"license plate"`` text features are computed once by
  ``PlateSegmenter.load()`` and injected into the grounding pass

Checks that boxes, scores and masks are equivalent, then reports the one-time
cache build cost, first-run / steady-state latency, and peak XPU memory.

Usage::

    python scripts/benchmark_text_cache.py [IMAGE ...] [--runs 5] [--device xpu]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from lpnrecog.device import get_device, synchronize
from lpnrecog.segmenter import CACHED_PROMPTS, PlateSegmenter
from lpnrecog.types import PlateDetection

PROMPT = CACHED_PROMPTS[0]
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGES = (
    _REPO_ROOT / "assets" / "samples" / "guangdong_plate.jpg",
    _REPO_ROOT / "assets" / "samples" / "geely_kingkong.jpg",
)

_BOX_ATOL = 1e-2  # pixels
_SCORE_ATOL = 1e-4


def _timed(fn, device: str):
    synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    synchronize(device)
    return time.perf_counter() - t0, out


def _reset_peak_memory(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()
    elif device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_gb(device: str) -> float:
    if device.startswith("xpu"):
        return torch.xpu.max_memory_allocated() / 1e9
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def _compare(
    ref: Sequence[PlateDetection], cand: Sequence[PlateDetection]
) -> tuple[bool, str]:
    """Compare two detection lists; return (equivalent, report line)."""
    if len(ref) != len(cand):
        return False, f"count {len(ref)} != {len(cand)}"
    box_diff = 0.0
    score_diff = 0.0
    min_iou = 1.0
    for a, b in zip(ref, cand):
        box_diff = max(
            box_diff, float(np.abs(np.asarray(a.box) - np.asarray(b.box)).max())
        )
        score_diff = max(score_diff, abs(a.score - b.score))
        inter = np.logical_and(a.mask, b.mask).sum()
        union = np.logical_or(a.mask, b.mask).sum()
        min_iou = min(min_iou, float(inter) / float(union) if union else 1.0)
    ok = box_diff <= _BOX_ATOL and score_diff <= _SCORE_ATOL and min_iou >= 0.999
    return (
        ok,
        f"boxes max|d|={box_diff:.2e}px scores max|d|={score_diff:.2e} "
        f"masks min IoU={min_iou:.6f}",
    )


def _count_text_calls(segmenter: PlateSegmenter) -> dict:
    """Wrap the text encoder to count how often it actually runs."""
    original = segmenter._model.backbone.forward_text
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    segmenter._model.backbone.forward_text = counted
    return counter


def _benchmark_image(
    segmenter: PlateSegmenter, image: Path, runs: int, device: str, counter: dict
) -> bool:
    print(f"\n== {image.name} ==")

    segmenter.use_text_cache = False
    _reset_peak_memory(device)
    counter["n"] = 0
    first_uncached, ref = _timed(lambda: segmenter.segment(image), device)
    uncached_calls = counter["n"]
    peak_uncached = _peak_memory_gb(device)

    segmenter.use_text_cache = True
    _reset_peak_memory(device)
    counter["n"] = 0
    first_cached, cand = _timed(lambda: segmenter.segment(image), device)
    cached_calls = counter["n"]
    peak_cached = _peak_memory_gb(device)

    steady: dict[str, List[float]] = {"uncached": [], "cached": []}
    for _ in range(runs):
        for label, use_cache in (("cached", True), ("uncached", False)):
            segmenter.use_text_cache = use_cache
            elapsed, _ = _timed(lambda: segmenter.segment(image), device)
            steady[label].append(elapsed)

    equivalent, detail = _compare(ref, cand)
    med_u, med_c = (statistics.median(steady[k]) for k in ("uncached", "cached"))
    print(f"  detections: {len(ref)}")
    print(f"  equivalence [{'PASS' if equivalent else 'FAIL'}]: {detail}")
    print(
        f"  text-encoder calls during first run: "
        f"uncached {uncached_calls}, cached {cached_calls}"
    )
    print(f"  {'phase':<10} {'first-run':>10} {'steady (median)':>16}")
    print(f"  {'uncached':<10} {first_uncached:>9.3f}s {med_u:>15.3f}s")
    print(f"  {'cached':<10} {first_cached:>9.3f}s {med_c:>15.3f}s")
    print(f"  steady-state speedup: {med_u / med_c:.2f}x")
    if not np.isnan(peak_uncached):
        print(
            f"  peak device memory: uncached {peak_uncached:.2f} GB, "
            f"cached {peak_cached:.2f} GB"
        )
    return equivalent


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help=f"input images (default: {' '.join(p.name for p in _DEFAULT_IMAGES)})",
    )
    parser.add_argument("--runs", type=int, default=5, help="steady-state runs (default 5)")
    parser.add_argument("--device", default=None, help="xpu / cuda / cpu (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="SAM3 checkpoint path")
    parser.add_argument(
        "--fallback-zoom",
        action="store_true",
        help="enable the vehicle-crop zoom fallback (off by default for comparability)",
    )
    args = parser.parse_args(argv)

    images = [Path(p) for p in (args.images or _DEFAULT_IMAGES)]
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"error: image not found: {missing[0]}", file=sys.stderr)
        return 2

    device = get_device(args.device)
    print(
        f"device={device} prompt={PROMPT!r} runs={args.runs} "
        f"fallback_zoom={args.fallback_zoom}"
    )

    segmenter = PlateSegmenter(
        checkpoint=args.checkpoint,
        device=device,
        prompts=(PROMPT,),
        fallback_zoom=args.fallback_zoom,
    )
    t0 = time.perf_counter()
    segmenter.load()  # also builds the fixed-prompt cache
    print(f"model load incl. cache build: {time.perf_counter() - t0:.2f}s")
    cache_build, _ = _timed(lambda: segmenter._cache_text_prompt(PROMPT), device)
    print(f"one text-encoder pass (cost avoided per inference): {cache_build * 1e3:.1f} ms")

    # Warm up XPU kernels on a small dummy image so the per-image "first-run"
    # below is the first inference on that image, not one-time kernel compilation.
    warmup = Image.new("RGB", (320, 240))
    for use_cache in (False, True):
        segmenter.use_text_cache = use_cache
        segmenter.segment(warmup)

    counter = _count_text_calls(segmenter)
    all_ok = True
    for image in images:
        all_ok &= _benchmark_image(segmenter, image, args.runs, device, counter)

    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
