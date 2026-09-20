#!/usr/bin/env python
"""Evaluate the plate pipeline on the frozen ``benchmark_data`` set.

Reads ``benchmark_data/ground_truth.csv``, runs the full pipeline (SAM3
segmentation + rectification + PaddleOCR) on every image, and reports
detection/recognition metrics against the ground truth: exact-match rate,
character accuracy, valid-format rate, and per-category/per-plate-type
breakdowns. ``--compare`` additionally re-runs the set with the compiled vision
backbone and checks that eager and compiled results agree.

Usage::

    python scripts/evaluate_benchmark.py
    python scripts/evaluate_benchmark.py --compile-vision
    python scripts/evaluate_benchmark.py --compare --output outputs/benchmark_eval
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from lpnrecog.device import get_device
from lpnrecog.ocr import is_valid_plate, normalize_plate_text
from lpnrecog.pipeline import PlateRecognitionPipeline
from lpnrecog.segmenter import DEFAULT_PROMPTS, PlateSegmenter

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = _REPO_ROOT / "benchmark_data"
_FIELDS = (
    "image",
    "category",
    "plate_type",
    "gt",
    "n_plates",
    "pred",
    "text_score",
    "det_score",
    "valid",
    "exact",
    "char_acc",
    "elapsed_s",
    "mask_sha256",
)


def _normalize(text: str) -> str:
    return normalize_plate_text(text)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _char_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return float("nan")
    return max(0.0, 1.0 - _edit_distance(gt, pred) / len(gt))


def _load_ground_truth(data_dir: Path) -> List[Dict[str, str]]:
    with (data_dir / "ground_truth.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run(pipe: PlateRecognitionPipeline, data_dir: Path, rows: Sequence[Dict[str, str]]):
    records: List[Dict[str, object]] = []
    for i, row in enumerate(rows, 1):
        image = data_dir / "images" / row["image"]
        t0 = time.perf_counter()
        results = pipe.run(image)
        elapsed = time.perf_counter() - t0
        gt = _normalize(row["plate"])
        top = results[0] if results else None
        pred = _normalize(top.text) if top else ""
        texts = [_normalize(r.text) for r in results]
        masks = [r.detection.mask for r in results]
        record = {
            "image": row["image"],
            "category": row["category"],
            "plate_type": row["plate_type"],
            "gt": gt,
            "n_plates": len(results),
            "pred": pred,
            "text_score": round(float(top.text_score), 4) if top else 0.0,
            "det_score": round(float(top.score), 4) if top else 0.0,
            "valid": bool(top and is_valid_plate(pred)),
            "exact": bool(pred) and pred == gt,
            "any_exact": gt in texts,
            "char_acc": _char_accuracy(gt, pred) if pred else 0.0,
            "elapsed_s": round(elapsed, 3),
            "mask_sha256": (
                hashlib.sha256(top.detection.mask.tobytes()).hexdigest() if top else ""
            ),
            "masks": masks,
        }
        records.append(record)
        status = "OK " if record["exact"] else "MISS"
        print(
            f"[{i:>2}/{len(rows)}] {row['image']} {status} "
            f"gt={gt} pred={pred or '-'} ({record['text_score']:.2f}) "
            f"{elapsed:.1f}s"
        )
    return records


def _summary(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    n = len(records)
    detected = [r for r in records if r["n_plates"] > 0]
    exact = [r for r in records if r["exact"]]
    any_exact = [r for r in records if r["any_exact"]]
    valid = [r for r in records if r["valid"]]

    def rate(part, whole=n):
        return f"{len(part)}/{whole} ({len(part) / whole * 100:.0f}%)"

    by_category: Dict[str, Dict[str, int]] = {}
    for r in records:
        stats = by_category.setdefault(r["category"], {"n": 0, "exact": 0})
        stats["n"] += 1
        stats["exact"] += int(r["exact"])
    by_type: Dict[str, Dict[str, int]] = {}
    for r in records:
        stats = by_type.setdefault(r["plate_type"], {"n": 0, "exact": 0})
        stats["n"] += 1
        stats["exact"] += int(r["exact"])

    elapsed = [r["elapsed_s"] for r in records]
    return {
        "images": n,
        "detected": rate(detected),
        "exact_top1": rate(exact),
        "exact_any": rate(any_exact),
        "valid_format": rate(valid),
        "mean_char_acc": statistics.mean(r["char_acc"] for r in records),
        "median_latency_s": statistics.median(elapsed),
        "total_latency_s": sum(elapsed),
        "by_category": by_category,
        "by_plate_type": by_type,
    }


def _print_summary(summary: Dict[str, object], records: Sequence[Dict[str, object]]) -> None:
    print("\n================ benchmark summary ================")
    print(f"images            : {summary['images']}")
    print(f"detected          : {summary['detected']}")
    print(f"exact match top-1 : {summary['exact_top1']}")
    print(f"exact match any   : {summary['exact_any']}")
    print(f"valid plate format: {summary['valid_format']}")
    print(f"mean char accuracy: {summary['mean_char_acc'] * 100:.1f}%")
    print(
        f"latency           : median {summary['median_latency_s']:.2f}s, "
        f"total {summary['total_latency_s']:.1f}s"
    )
    for label, key in (("by category", "by_category"), ("by plate type", "by_plate_type")):
        print(f"{label}:")
        for name, stats in summary[key].items():
            print(f"  {name:<10} {stats['exact']}/{stats['n']}")
    failures = [r for r in records if not r["exact"]]
    if failures:
        print("failures:")
        for r in failures:
            print(
                f"  {r['image']} [{r['category']}] gt={r['gt']} "
                f"pred={r['pred'] or '-'} ({r['text_score']:.2f})"
            )


def _write_outputs(output_dir: Path, records, summary, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"results_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    json_path = output_dir / f"summary_{tag}.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {csv_path} and {json_path}")


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def _compare(
    eager: Sequence[Dict[str, object]],
    compiled: Sequence[Dict[str, object]],
    output_dir: Optional[Path] = None,
) -> None:
    same_pred = same_exact = same_count = masks_exact = 0
    max_score_diff = 0.0
    text_mismatches = []
    count_mismatches = []
    ious: List[float] = []
    rows = []
    for a, b in zip(eager, compiled):
        same_pred += int(a["pred"] == b["pred"])
        same_exact += int(a["exact"] == b["exact"])
        max_score_diff = max(max_score_diff, abs(a["text_score"] - b["text_score"]))
        if a["pred"] != b["pred"]:
            text_mismatches.append((a["image"], a["pred"], b["pred"]))
        image_min_iou = None
        if a["n_plates"] == b["n_plates"]:
            same_count += 1
            image_ious = [_mask_iou(x, y) for x, y in zip(a["masks"], b["masks"])]
            image_min_iou = min(image_ious, default=1.0)
            ious.append(image_min_iou)
            masks_exact += int(
                all(np.array_equal(x, y) for x, y in zip(a["masks"], b["masks"]))
            )
        else:
            count_mismatches.append((a["image"], a["n_plates"], b["n_plates"]))
        rows.append(
            {
                "image": a["image"],
                "n_eager": a["n_plates"],
                "n_compiled": b["n_plates"],
                "pred_eager": a["pred"],
                "pred_compiled": b["pred"],
                "min_mask_iou": "" if image_min_iou is None else f"{image_min_iou:.6f}",
                "exact_eager": a["exact"],
                "exact_compiled": b["exact"],
            }
        )

    n = len(eager)
    print("\n============ eager vs compiled agreement ============")
    print(f"top-1 text identical  : {same_pred}/{n}")
    print(f"recognition agreement : {same_exact}/{n} (exact-match verdict unchanged)")
    print(f"detection count same  : {same_count}/{n}")
    print(f"mask sets bit-exact   : {masks_exact}/{same_count} (equal-count images)")
    if ious:
        print(
            f"mask IoU (equal count): min {min(ious):.6f}, "
            f"median {statistics.median(ious):.6f}"
        )
    print(f"max text-score drift  : {max_score_diff:.4f}")
    if count_mismatches:
        print("count mismatches:")
        for image, ne, nc in count_mismatches:
            print(f"  {image}: eager={ne} compiled={nc}")
    if text_mismatches:
        print("text mismatches:")
        for image, ep, cp in text_mismatches:
            print(f"  {image}: eager={ep or '-'} compiled={cp or '-'}")

    if output_dir is not None:
        path = output_dir / "eager_vs_compiled.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


def _compile_in_place(segmenter: PlateSegmenter):
    target = segmenter._model.backbone.vision_backbone.trunk
    original = target.forward
    segmenter.compile_vision = True
    segmenter.compile_vision_mode = "default"
    segmenter.compile_vision_target = "trunk"
    segmenter._compile_vision()
    return target, original


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "outputs" / "benchmark_eval")
    parser.add_argument("--device", default=None, help="xpu / cuda / cpu (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="SAM3 checkpoint path")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument(
        "--compile-vision",
        action="store_true",
        help="evaluate the compiled vision backbone instead of eager",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="evaluate eager then compiled and report agreement",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    if not (data_dir / "ground_truth.csv").exists():
        print(f"error: {data_dir}/ground_truth.csv not found", file=sys.stderr)
        return 2
    rows = _load_ground_truth(data_dir)
    if args.limit:
        rows = rows[: args.limit]

    device = get_device(args.device)
    print(
        f"device={device} images={len(rows)} "
        f"prompts={list(DEFAULT_PROMPTS)} compile_vision={args.compile_vision} "
        f"compare={args.compare}"
    )

    segmenter = PlateSegmenter(
        checkpoint=args.checkpoint,
        device=device,
        prompts=DEFAULT_PROMPTS,
        compile_vision=args.compile_vision,
    )
    pipeline = PlateRecognitionPipeline(segmenter=segmenter)

    eager_records = None
    if args.compare:
        eager_records = _run(pipeline, data_dir, rows)
        eager_summary = _summary(eager_records)
        _print_summary(eager_summary, eager_records)
        _write_outputs(args.output, eager_records, eager_summary, "eager")

        target, original = _compile_in_place(segmenter)
        print("\n-- compiled run (first image pays compilation) --")
        compiled_records = _run(pipeline, data_dir, rows)
        setattr(target, "forward", original)
        compiled_summary = _summary(compiled_records)
        _print_summary(compiled_summary, compiled_records)
        _write_outputs(args.output, compiled_records, compiled_summary, "compiled")
        _compare(eager_records, compiled_records, args.output)
        return 0

    records = _run(pipeline, data_dir, rows)
    summary = _summary(records)
    _print_summary(summary, records)
    tag = "compiled" if args.compile_vision else "eager"
    _write_outputs(args.output, records, summary, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
