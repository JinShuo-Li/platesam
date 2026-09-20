#!/usr/bin/env python
"""Eager vs torch.compile vision-backbone benchmark and equivalence check.

Uninstrumented end-to-end timing (no stage hooks). This laptop's CPU/GPU clocks
ramp slowly, so instead of measuring the eager and compiled implementations in
separate long phases, the script conditions the machine, compiles the vision
backbone in place, then alternates short eager/compiled measurement blocks and
reports the median over all blocks. Every measurement is synchronized with
``torch.xpu.synchronize()``.

Compiled outputs are compared against the eager references (vision features,
detections, OCR).

Usage::

    python scripts/benchmark_compile.py --runs 3 --cycles 3
    python scripts/benchmark_compile.py --target trunk --mode reduce-overhead
    python scripts/benchmark_compile.py --verify-only assets/samples/*.jpg
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from lpnrecog.device import bf16_autocast, get_device, synchronize
from lpnrecog.pipeline import PlateRecognitionPipeline
from lpnrecog.segmenter import DEFAULT_PROMPTS, PlateSegmenter
from lpnrecog.types import PlateDetection, PlateResult

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGE = _REPO_ROOT / "assets" / "samples" / "guangdong_plate.jpg"

# Compiled kernels reorder bf16 reductions, so outputs are equivalent rather
# than bit-identical: allow sub-pixel boxes, <1% scores, >0.999 mask IoU and
# an identical OCR string (score may drift with the crop pixels).
_BOX_ATOL = 1.0  # pixels
_SCORE_ATOL = 1e-2
_MASK_IOU_MIN = 0.999
_TEXT_SCORE_ATOL = 2e-2

_DEVICE = "cpu"
_STAGES = ("vision_ms", "segment_ms", "pipeline_ms")


def _timed(fn):
    synchronize(_DEVICE)
    t0 = time.perf_counter()
    out = fn()
    synchronize(_DEVICE)
    return time.perf_counter() - t0, out


def _median_runs(fn, runs: int) -> float:
    return statistics.median(_timed(fn)[0] for _ in range(runs))


def _flatten(obj, prefix: str = "") -> Dict[str, torch.Tensor]:
    items: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        items[prefix] = obj
    elif isinstance(obj, dict):
        for key, val in obj.items():
            items.update(_flatten(val, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            items.update(_flatten(val, f"{prefix}[{i}]"))
    return items


def _vision(segmenter: PlateSegmenter, tensor: torch.Tensor) -> Dict:
    with torch.inference_mode(), bf16_autocast(
        segmenter.device, enabled=segmenter.use_autocast
    ):
        return segmenter._model.backbone.forward_image(tensor)


def _peak_gb() -> float:
    if _DEVICE.startswith("xpu"):
        return torch.xpu.max_memory_allocated() / 1e9
    if _DEVICE.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def _reset_peak() -> None:
    if _DEVICE.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()
    elif _DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _measure_block(
    segmenter: PlateSegmenter,
    pipeline: PlateRecognitionPipeline,
    image: Path,
    x: torch.Tensor,
    runs: int,
) -> Dict[str, float]:
    return {
        "vision_ms": _median_runs(lambda: _vision(segmenter, x), runs) * 1e3,
        "segment_ms": _median_runs(lambda: segmenter.segment(image), runs) * 1e3,
        "pipeline_ms": _median_runs(lambda: pipeline.run(image), runs) * 1e3,
    }


def _references(
    segmenter: PlateSegmenter,
    pipeline: PlateRecognitionPipeline,
    x: torch.Tensor,
    images: Sequence[Path],
) -> Dict:
    with torch.inference_mode(), bf16_autocast(
        segmenter.device, enabled=segmenter.use_autocast
    ):
        vision = {
            k: v.detach().to("cpu") for k, v in _flatten(_vision(segmenter, x)).items()
        }
    return {
        "vision": vision,
        "segments": {str(p): segmenter.segment(p) for p in images},
        "pipelines": {str(p): pipeline.run(p) for p in images},
    }


def _compare_vision(
    ref: Dict[str, torch.Tensor], cand: Dict[str, torch.Tensor]
) -> Tuple[bool, str]:
    if set(ref) != set(cand):
        return False, f"keys differ: {set(ref) ^ set(cand)}"
    max_abs = mean_abs = 0.0
    exact = True
    for key in ref:
        a, b = ref[key].float(), cand[key].float()
        if a.shape != b.shape:
            return False, f"shape mismatch for {key}: {a.shape} vs {b.shape}"
        diff = (a - b).abs()
        max_abs = max(max_abs, float(diff.max()))
        mean_abs = max(mean_abs, float(diff.mean()))
        exact = exact and bool(torch.equal(a, b))
    return True, (
        f"tensors={len(ref)} exact={exact} "
        f"max|d|={max_abs:.3e} mean|d|={mean_abs:.3e} (bf16)"
    )


def _compare_detections(
    ref: Sequence[PlateDetection], cand: Sequence[PlateDetection]
) -> Tuple[bool, str]:
    if len(ref) != len(cand):
        return False, f"count {len(ref)} != {len(cand)}"
    box_diff = score_diff = mask_pixel_diff = 0.0
    min_iou = 1.0
    masks_equal = True
    for a, b in zip(ref, cand):
        box_diff = max(
            box_diff, float(np.abs(np.asarray(a.box) - np.asarray(b.box)).max())
        )
        score_diff = max(score_diff, abs(a.score - b.score))
        masks_equal = masks_equal and np.array_equal(a.mask, b.mask)
        mask_pixel_diff = max(
            mask_pixel_diff, float(np.logical_xor(a.mask, b.mask).mean())
        )
        inter = np.logical_and(a.mask, b.mask).sum()
        union = np.logical_or(a.mask, b.mask).sum()
        min_iou = min(min_iou, float(inter) / float(union) if union else 1.0)
    ok = (
        box_diff <= _BOX_ATOL
        and score_diff <= _SCORE_ATOL
        and min_iou >= _MASK_IOU_MIN
    )
    return ok, (
        f"count={len(cand)} boxes max|d|={box_diff:.3e}px "
        f"scores max|d|={score_diff:.3e} masks exact={masks_equal} "
        f"pixels changed={mask_pixel_diff * 100:.4f}% min IoU={min_iou:.6f}"
    )


def _compare_pipeline(
    ref: Sequence[PlateResult], cand: Sequence[PlateResult]
) -> Tuple[bool, str]:
    if len(ref) != len(cand):
        return False, f"plate count {len(ref)} != {len(cand)}"
    texts = []
    text_score_diff = 0.0
    texts_equal = True
    for a, b in zip(ref, cand):
        texts.append(f"{b.text}({b.text_score:.4f})")
        texts_equal = texts_equal and a.text == b.text
        text_score_diff = max(text_score_diff, abs(a.text_score - b.text_score))
    return texts_equal and text_score_diff <= _TEXT_SCORE_ATOL, (
        f"OCR {'==' if texts_equal else '!='} max|d|={text_score_diff:.3e}; "
        f"{', '.join(texts) or 'none'}"
    )


def _compile_target(segmenter: PlateSegmenter, target_name: str):
    target = {
        "trunk": segmenter._model.backbone.vision_backbone.trunk,
        "vision_backbone": segmenter._model.backbone.vision_backbone,
        "forward_image": segmenter._model.backbone,
    }[target_name]
    attr = "forward" if target_name != "forward_image" else "forward_image"
    original = getattr(target, attr)
    segmenter.compile_vision = True
    segmenter.compile_vision_mode = COMPILE_MODE
    segmenter.compile_vision_target = target_name
    segmenter._compile_vision()
    return target, attr, original, getattr(target, attr)


COMPILE_MODE = "default"


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _DEVICE, COMPILE_MODE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help=f"images (default: {_DEFAULT_IMAGE.name})",
    )
    parser.add_argument("--runs", type=int, default=3, help="runs per block (default 3)")
    parser.add_argument(
        "--cycles", type=int, default=3, help="eager/compiled blocks (default 3)"
    )
    parser.add_argument(
        "--condition", type=float, default=60.0, help="conditioning seconds (default 60)"
    )
    parser.add_argument("--device", default=None, help="xpu / cuda / cpu (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="SAM3 checkpoint path")
    parser.add_argument(
        "--target",
        choices=("trunk", "vision_backbone", "forward_image"),
        default="trunk",
        help="module to compile (default: trunk)",
    )
    parser.add_argument(
        "--mode", default="default", help="torch.compile mode (default: %(default)s)"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="skip latency measurements"
    )
    args = parser.parse_args(argv)

    images = [Path(p) for p in (args.images or [_DEFAULT_IMAGE])]
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"error: image not found: {missing[0]}", file=sys.stderr)
        return 2

    _DEVICE = get_device(args.device)
    COMPILE_MODE = args.mode
    measure = not args.verify_only
    latency_image = images[0]
    print(
        f"device={_DEVICE} target={args.target} mode={args.mode} "
        f"runs={args.runs} cycles={args.cycles} condition={args.condition:.0f}s"
    )

    segmenter = PlateSegmenter(
        checkpoint=args.checkpoint, device=_DEVICE, prompts=DEFAULT_PROMPTS
    )
    t0 = time.perf_counter()
    segmenter.load()
    print(f"model load incl. text cache: {time.perf_counter() - t0:.2f}s")
    pipeline = PlateRecognitionPipeline(segmenter=segmenter)
    pil = Image.open(latency_image).convert("RGB")
    x = segmenter.processor.transform(v2.functional.to_image(pil)).unsqueeze(0).to(_DEVICE)

    refs = _references(segmenter, pipeline, x, images)

    if measure:
        t_end = time.perf_counter() + args.condition
        while time.perf_counter() < t_end:
            pipeline.run(latency_image)

    target, attr, original, compiled = _compile_target(segmenter, args.target)
    first_vision_s, _ = _timed(lambda: _vision(segmenter, x))
    first_segment_s, _ = _timed(lambda: segmenter.segment(latency_image))
    first_pipeline_s, _ = _timed(lambda: pipeline.run(latency_image))
    cand_refs = _references(segmenter, pipeline, x, images)
    setattr(target, attr, original)

    if measure:
        _reset_peak()
        eager_blocks: List[Dict[str, float]] = []
        compiled_blocks: List[Dict[str, float]] = []
        for _ in range(args.cycles):
            setattr(target, attr, original)
            eager_blocks.append(
                _measure_block(segmenter, pipeline, latency_image, x, args.runs)
            )
            setattr(target, attr, compiled)
            compiled_blocks.append(
                _measure_block(segmenter, pipeline, latency_image, x, args.runs)
            )
        setattr(target, attr, original)
        peak = _peak_gb()

        def pooled(blocks, stage):
            return statistics.median(block[stage] for block in blocks)

        eager = {s: pooled(eager_blocks, s) for s in _STAGES}
        comp = {s: pooled(compiled_blocks, s) for s in _STAGES}
        # Adjacent eager/compiled blocks share the same machine state, so the
        # paired ratio per cycle is less drift-sensitive than block medians.
        paired = {
            s: statistics.median(
                e[s] / c[s] for e, c in zip(eager_blocks, compiled_blocks)
            )
            for s in _STAGES
        }
        print(
            f"\n== compiled [target={args.target}, mode={args.mode}] ==\n"
            f"  compile + first vision call: {first_vision_s:.2f}s | "
            f"first segment: {first_segment_s:.2f}s | "
            f"first pipeline: {first_pipeline_s:.2f}s"
        )
        for label, blocks in (("eager", eager_blocks), ("compiled", compiled_blocks)):
            per_cycle = " | ".join(
                f"{b['vision_ms']:.0f}/{b['segment_ms']:.0f}/{b['pipeline_ms']:.0f}ms"
                for b in blocks
            )
            print(f"  {label:<8} per cycle (vision/segment/pipeline): {per_cycle}")
        print(
            f"  median over {args.cycles} blocks: "
            f"eager vision {eager['vision_ms']:.1f}ms, segment {eager['segment_ms']:.1f}ms, "
            f"pipeline {eager['pipeline_ms']:.1f}ms"
        )
        print(
            f"  median over {args.cycles} blocks: "
            f"compiled vision {comp['vision_ms']:.1f}ms, segment {comp['segment_ms']:.1f}ms, "
            f"pipeline {comp['pipeline_ms']:.1f}ms"
        )
        print(
            f"  speedup (block medians): vision "
            f"{eager['vision_ms'] / comp['vision_ms']:.3f}x | segment "
            f"{eager['segment_ms'] / comp['segment_ms']:.3f}x | pipeline "
            f"{eager['pipeline_ms'] / comp['pipeline_ms']:.3f}x"
        )
        print(
            f"  speedup (paired per cycle, median): vision {paired['vision_ms']:.3f}x | "
            f"segment {paired['segment_ms']:.3f}x | pipeline {paired['pipeline_ms']:.3f}x"
        )
        print(f"  peak XPU memory: {peak:.2f}GB")

    print("\n== correctness (eager vs compiled) ==")
    vis_ok, vis_detail = _compare_vision(refs["vision"], cand_refs["vision"])
    print(f"  vision features [{'OK' if vis_ok else 'FAIL'}]: {vis_detail}")

    all_ok = vis_ok
    for path in [str(p) for p in images]:
        det_ok, det_detail = _compare_detections(
            refs["segments"][path], cand_refs["segments"][path]
        )
        pipe_ok, pipe_detail = _compare_pipeline(
            refs["pipelines"][path], cand_refs["pipelines"][path]
        )
        all_ok = all_ok and det_ok and pipe_ok
        print(f"  {Path(path).name} detections [{'OK' if det_ok else 'FAIL'}]: {det_detail}")
        print(f"  {Path(path).name} pipeline   [{'OK' if pipe_ok else 'FAIL'}]: {pipe_detail}")

    print(f"\nverdict: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
