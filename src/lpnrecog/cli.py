"""Command line entry point: ``python -m lpnrecog`` / ``lpnrecog``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .device import get_device
from .ocr import is_valid_plate
from .pipeline import PlateRecognitionPipeline
from .segmenter import DEFAULT_PROMPTS, PlateSegmenter
from .viz import draw_results

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lpnrecog",
        description="License plate recognition: SAM3 mask + perspective rectification + OCR",
    )
    parser.add_argument("-i", "--input", required=True, help="image file or directory")
    parser.add_argument("-o", "--output", default="outputs", help="output directory")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help=f"SAM3 text prompt (repeatable). Default: {', '.join(DEFAULT_PROMPTS)}",
    )
    parser.add_argument("--conf", type=float, default=0.4, help="detection score threshold")
    parser.add_argument("--device", default=None, help="xpu / cuda / cpu (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="SAM3 checkpoint path")
    parser.add_argument("--plate-size", default="440x140", help="rectified WxH, default 440x140")
    parser.add_argument("--no-viz", action="store_true", help="skip visualization output")
    parser.add_argument("--no-crops", action="store_true", help="skip saving plate crops")
    parser.add_argument("--json", default=None, help="results JSON path (default: <output>/results.json)")
    parser.add_argument(
        "--compile-vision",
        action="store_true",
        help="torch.compile the SAM3 vision backbone (first run pays compilation)",
    )
    parser.add_argument(
        "--compile-vision-mode", default="default", help="torch.compile mode"
    )
    parser.add_argument(
        "--compile-vision-target",
        default="trunk",
        choices=("trunk", "vision_backbone", "forward_image"),
        help="vision module to compile (default: trunk)",
    )
    return parser


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().replace(" ", "").split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size {text!r}, expected WxH") from exc


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    images = _iter_images(input_path)
    if not images:
        print(f"error: no images found under {input_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        (output_dir / "viz").mkdir(exist_ok=True)
    if not args.no_crops:
        (output_dir / "crops").mkdir(exist_ok=True)

    device = get_device(args.device)
    print(f"device={device} images={len(images)} prompts={args.prompt or list(DEFAULT_PROMPTS)}")

    segmenter = PlateSegmenter(
        checkpoint=args.checkpoint,
        device=device,
        prompts=tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS,
        confidence_threshold=args.conf,
        compile_vision=args.compile_vision,
        compile_vision_mode=args.compile_vision_mode,
        compile_vision_target=args.compile_vision_target,
    )
    pipeline = PlateRecognitionPipeline(
        segmenter=segmenter, out_size=_parse_size(args.plate_size)
    )

    all_results = []
    for image_path in images:
        t0 = time.time()
        results = pipeline.run_path(image_path)
        elapsed = time.time() - t0
        entry = {
            "image": str(image_path),
            "elapsed_s": round(elapsed, 2),
            "plates": [
                {
                    "box": [round(float(v), 1) for v in r.detection.box],
                    "det_score": round(r.score, 4),
                    "prompt": r.detection.prompt,
                    "text": r.text,
                    "text_score": round(r.text_score, 4),
                    "valid": is_valid_plate(r.text),
                }
                for r in results
            ],
        }
        all_results.append(entry)

        def _fmt(r) -> str:
            marker = "" if is_valid_plate(r.text) else ",invalid"
            return f"{r.text or '?'}({r.text_score:.2f}{marker})"

        summary = ", ".join(_fmt(r) for r in results) or "none"
        print(f"{image_path.name}: {len(results)} plate(s) in {elapsed:.1f}s -> {summary}")

        if not args.no_viz or not args.no_crops:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            if not args.no_viz:
                canvas = draw_results(image_bgr, results)
                cv2.imwrite(str(output_dir / "viz" / f"{image_path.stem}_viz.jpg"), canvas)
            if not args.no_crops:
                for idx, res in enumerate(results):
                    if res.crop is None:
                        continue
                    cv2.imwrite(
                        str(output_dir / "crops" / f"{image_path.stem}_plate{idx}.jpg"),
                        res.crop,
                    )

    json_path = Path(args.json) if args.json else output_dir / "results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"results written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
