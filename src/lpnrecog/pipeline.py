"""End-to-end plate recognition: SAM3 segmentation -> rectification -> OCR."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from .ocr import PlateOCR
from .rectify import DEFAULT_OUT_SIZE, rectify_plate
from .segmenter import PlateSegmenter
from .types import PlateResult


def _imread(path: Union[str, Path]) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


class PlateRecognitionPipeline:
    def __init__(
        self,
        segmenter: Optional[PlateSegmenter] = None,
        ocr: Optional[PlateOCR] = None,
        out_size: Sequence[int] = DEFAULT_OUT_SIZE,
        inset_ratio: float = 0.02,
    ) -> None:
        self.segmenter = segmenter or PlateSegmenter()
        self.ocr = ocr or PlateOCR()
        self.out_size = tuple(out_size)
        self.inset_ratio = inset_ratio

    def run(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompts: Optional[Sequence[str]] = None,
    ) -> List[PlateResult]:
        if isinstance(image, (str, Path)):
            image_bgr = _imread(image)
        elif isinstance(image, Image.Image):
            image_bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        else:
            image_bgr = np.asarray(image)
            if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
                image_bgr = image_bgr[:, :, ::-1].copy()  # RGB -> BGR
            elif image_bgr.ndim == 2:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        detections = self.segmenter.segment(image, prompts=prompts)

        results: List[PlateResult] = []
        for det in detections:
            crop, corners = rectify_plate(
                image_bgr,
                det.mask,
                out_size=self.out_size,
                inset_ratio=self.inset_ratio,
            )
            text, text_score = "", 0.0
            if crop is not None:
                text, text_score = self.ocr.recognize_plate(crop)
            results.append(
                PlateResult(
                    detection=det,
                    corners=corners,
                    crop=crop,
                    text=text,
                    text_score=text_score,
                )
            )
        results.sort(key=lambda r: r.detection.score, reverse=True)
        return results

    def run_path(
        self,
        path: Union[str, Path],
        prompts: Optional[Sequence[str]] = None,
    ) -> List[PlateResult]:
        return self.run(path, prompts=prompts)
