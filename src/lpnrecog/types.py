"""Shared data structures for the plate recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PlateDetection:
    """A license plate candidate produced by the SAM3 segmenter."""

    box: tuple[float, float, float, float]  # x0, y0, x1, y1 (original image pixels)
    score: float
    mask: np.ndarray  # bool mask, shape (H, W), original image size
    prompt: str = ""

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class PlateResult:
    """Final recognition result for one plate."""

    detection: PlateDetection
    corners: Optional[np.ndarray]  # (4, 2) float32, ordered tl, tr, br, bl
    crop: Optional[np.ndarray]  # rectified BGR crop, shape (out_h, out_w, 3)
    text: str = ""
    text_score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.detection.box

    @property
    def score(self) -> float:
        return self.detection.score
