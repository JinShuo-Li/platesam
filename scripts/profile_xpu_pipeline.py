#!/usr/bin/env python
"""Stage-level XPU latency profiler and cache-equivalence check.

Every stage is timed with ``torch.xpu.synchronize()`` around accelerator work,
after warmup, and reported as the median over repeated runs. The fully cached
prompt path is also compared against the uncached one for exact output
equivalence (detection count, boxes, scores, mask IoU, OCR text).

Stages: image load, preprocess (to PIL), set_image (transform + ViT),
per-prompt grounding, mask upsampling, XPU->CPU extraction, shape filtering,
mask-IoU NMS, full segmentation, rectification, OCR, full pipeline.

Usage::

    python scripts/profile_xpu_pipeline.py --runs 5
    python scripts/profile_xpu_pipeline.py --mode single --verify-only
    python scripts/profile_xpu_pipeline.py assets/samples/byd_surui.jpg
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import lpnrecog.pipeline as pipeline_mod
import lpnrecog.segmenter as segmenter_mod
import sam3.model.sam3_image_processor as sam3_processor_mod
from lpnrecog.device import get_device, synchronize
from lpnrecog.pipeline import PlateRecognitionPipeline
from lpnrecog.segmenter import CACHED_PROMPTS, DEFAULT_PROMPTS, PlateSegmenter
from lpnrecog.types import PlateResult

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGES = (
    _REPO_ROOT / "assets" / "samples" / "guangdong_plate.jpg",
    _REPO_ROOT / "assets" / "samples" / "geely_kingkong.jpg",
)
_MODES: Dict[str, Tuple[str, ...]] = {
    "single": ("license plate",),
    "default": DEFAULT_PROMPTS,
}

_BOX_ATOL = 1e-2  # pixels
_SCORE_ATOL = 1e-4
_TEXT_SCORE_ATOL = 1e-5


class StageTimer:
    """Accumulates synchronized stage latencies for each pipeline run."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.runs: List[Dict[str, float]] = []
        self.counts: List[Dict[str, int]] = []
        self._current: Dict[str, float] = {}
        self._calls: Dict[str, int] = {}

    def reset(self) -> None:
        self._current = {}
        self._calls = {}

    def record(self, stage: str, dt: float) -> None:
        self._current[stage] = self._current.get(stage, 0.0) + dt
        self._calls[stage] = self._calls.get(stage, 0) + 1

    def call(self, stage: str, fn, *args, sync: bool = True, **kwargs):
        if sync:
            synchronize(self.device)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if sync:
            synchronize(self.device)
        self.record(stage, time.perf_counter() - t0)
        return out

    def wrap(self, obj, attr: str, stage: str, sync: bool = True) -> None:
        original = getattr(obj, attr)

        def wrapper(*args, **kwargs):
            return self.call(stage, original, *args, sync=sync, **kwargs)

        setattr(obj, attr, wrapper)

    def wrap_prompt(self, obj, attr: str) -> None:
        original = getattr(obj, attr)

        def wrapper(state, prompt, *args, **kwargs):
            return self.call(
                f"grounding:{prompt}", original, state, prompt, *args, sync=True, **kwargs
            )

        setattr(obj, attr, wrapper)

    def snapshot(self) -> None:
        self.runs.append(dict(self._current))
        self.counts.append(dict(self._calls))

    def median(self, stage: str) -> float:
        values = [run[stage] for run in self.runs if stage in run]
        return statistics.median(values) if values else float("nan")

    def median_calls(self, stage: str) -> int:
        values = [calls[stage] for calls in self.counts if stage in calls]
        return int(statistics.median(values)) if values else 0

    def stage_names(self) -> List[str]:
        names = {stage for run in self.runs for stage in run}
        return sorted(names, key=_stage_rank)


def _stage_rank(name: str) -> Tuple[int, str]:
    order = (
        "image load",
        "preprocess",
        "set_image",
        "grounding:",
        "mask upsample",
        "xpu->cpu",
        "filter",
        "NMS merge",
        "segment total",
        "rectify",
        "OCR",
        "pipeline total",
    )
    for i, prefix in enumerate(order):
        if name.startswith(prefix):
            return i, name
    return len(order), name


def install_profiling(
    timer: StageTimer, segmenter: PlateSegmenter, pipeline: PlateRecognitionPipeline
) -> None:
    timer.wrap_prompt(segmenter, "_set_text_prompt")
    timer.wrap(segmenter._processor, "set_image", "set_image (transform + ViT)")
    timer.wrap(segmenter, "_extract", "xpu->cpu extract")
    timer.wrap(segmenter, "_keep", "filter (shape)", sync=False)
    timer.wrap(segmenter, "_merge", "NMS merge (mask IoU)", sync=False)
    timer.wrap(segmenter, "segment", "segment total")
    timer.wrap(pipeline, "run", "pipeline total")
    timer.wrap(pipeline.ocr, "recognize_plate", "OCR", sync=False)

    original_to_pil = segmenter_mod._to_pil
    segmenter_mod._to_pil = lambda image: timer.call(
        "preprocess (to PIL)", original_to_pil, image, sync=False
    )

    original_imread = pipeline_mod._imread
    pipeline_mod._imread = lambda path: timer.call(
        "image load (imread)", original_imread, path, sync=False
    )

    original_rectify = pipeline_mod.rectify_plate
    pipeline_mod.rectify_plate = lambda *args, **kwargs: timer.call(
        "rectify", original_rectify, *args, sync=False, **kwargs
    )

    original_interpolate = sam3_processor_mod.interpolate
    sam3_processor_mod.interpolate = lambda *args, **kwargs: timer.call(
        "mask upsample", original_interpolate, *args, sync=True, **kwargs
    )


def print_profile(timer: StageTimer) -> None:
    total = timer.median("pipeline total")
    print(f"  {'stage':<36}{'median':>10}{'calls':>7}{'share':>8}")
    for stage in timer.stage_names():
        median = timer.median(stage)
        share = median / total * 100.0 if total else float("nan")
        print(
            f"  {stage:<36}{median * 1e3:>8.1f}ms"
            f"{timer.median_calls(stage):>7}{share:>7.1f}%"
        )


def _compare_results(
    ref: Sequence[PlateResult], cand: Sequence[PlateResult]
) -> Tuple[bool, str]:
    if len(ref) != len(cand):
        return False, f"count {len(ref)} != {len(cand)}"
    box_diff = score_diff = text_score_diff = 0.0
    min_iou = 1.0
    texts_match = True
    for a, b in zip(ref, cand):
        da, db = a.detection, b.detection
        box_diff = max(
            box_diff, float(np.abs(np.asarray(da.box) - np.asarray(db.box)).max())
        )
        score_diff = max(score_diff, abs(da.score - db.score))
        inter = np.logical_and(da.mask, db.mask).sum()
        union = np.logical_or(da.mask, db.mask).sum()
        min_iou = min(min_iou, float(inter) / float(union) if union else 1.0)
        texts_match = texts_match and a.text == b.text
        text_score_diff = max(text_score_diff, abs(a.text_score - b.text_score))
    ok = (
        box_diff <= _BOX_ATOL
        and score_diff <= _SCORE_ATOL
        and min_iou >= 0.999
        and texts_match
        and text_score_diff <= _TEXT_SCORE_ATOL
    )
    return (
        ok,
        f"count={len(cand)} boxes max|d|={box_diff:.2e}px "
        f"scores max|d|={score_diff:.2e} masks min IoU={min_iou:.6f} "
        f"OCR {'==' if texts_match else '!='} max|d|={text_score_diff:.2e}",
    )


def verify(
    pipeline: PlateRecognitionPipeline,
    image: Path,
    prompts: Sequence[str],
) -> Tuple[bool, str]:
    pipeline.segmenter.use_text_cache = False
    ref = pipeline.run(image, prompts=prompts)
    pipeline.segmenter.use_text_cache = True
    cand = pipeline.run(image, prompts=prompts)
    ok, detail = _compare_results(ref, cand)
    texts = ", ".join(f"{r.text}({r.text_score:.4f})" for r in cand) or "none"
    return ok, f"{detail}; OCR: {texts}"


def profile(
    timer: StageTimer,
    pipeline: PlateRecognitionPipeline,
    image: Path,
    prompts: Sequence[str],
    warmup: int,
    runs: int,
) -> None:
    pipeline.segmenter.use_text_cache = True
    for _ in range(warmup):
        pipeline.run(image, prompts=prompts)
    for _ in range(runs):
        timer.reset()
        pipeline.run(image, prompts=prompts)
        timer.snapshot()
    print_profile(timer)
    timer.runs.clear()
    timer.counts.clear()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help=f"input images (default: {' '.join(p.name for p in _DEFAULT_IMAGES)})",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "default", "both"),
        default="both",
        help="prompt mode to profile/verify (default: both)",
    )
    parser.add_argument("--runs", type=int, default=5, help="steady-state runs (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs (default 1)")
    parser.add_argument("--device", default=None, help="xpu / cuda / cpu (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="SAM3 checkpoint path")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare cached vs uncached outputs before profiling (default: on)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="skip latency profiling"
    )
    parser.add_argument(
        "--no-fallback-zoom",
        dest="fallback_zoom",
        action="store_false",
        help="disable the vehicle-crop zoom fallback",
    )
    args = parser.parse_args(argv)

    images = [Path(p) for p in (args.images or _DEFAULT_IMAGES)]
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"error: image not found: {missing[0]}", file=sys.stderr)
        return 2

    device = get_device(args.device)
    print(
        f"device={device} runs={args.runs} warmup={args.warmup} "
        f"fallback_zoom={args.fallback_zoom}"
    )

    segmenter = PlateSegmenter(
        checkpoint=args.checkpoint,
        device=device,
        prompts=DEFAULT_PROMPTS,
        fallback_zoom=args.fallback_zoom,
    )
    t0 = time.perf_counter()
    segmenter.load()
    print(
        f"model load incl. {len(CACHED_PROMPTS)} cached prompts "
        f"({', '.join(CACHED_PROMPTS)}): {time.perf_counter() - t0:.2f}s"
    )

    pipeline = PlateRecognitionPipeline(segmenter=segmenter)
    timer = StageTimer(device)
    install_profiling(timer, segmenter, pipeline)

    modes = ("single", "default") if args.mode == "both" else (args.mode,)
    all_ok = True
    for mode in modes:
        prompts = _MODES[mode]
        print(f"\n===== mode={mode} prompts={list(prompts)} =====")
        for image in images:
            print(f"\n== {image.name} ==")
            if args.verify:
                ok, detail = verify(pipeline, image, prompts)
                all_ok &= ok
                print(f"  verify [{'PASS' if ok else 'FAIL'}]: {detail}")
            if not args.verify_only:
                profile(timer, pipeline, image, prompts, args.warmup, args.runs)

    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
