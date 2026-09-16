"""Visualization helpers (BGR in / BGR out)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import PlateResult

_FONT_CANDIDATES = (
    "~/.local/share/fonts/NotoSansSC-Bold.otf",
    "~/.local/share/fonts/NotoSansSC-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def overlay_mask(
    image_bgr: np.ndarray, mask: np.ndarray, color: Sequence[int] = (0, 255, 0), alpha: float = 0.4
) -> np.ndarray:
    out = image_bgr.copy()
    m = mask.astype(bool)
    if m.shape != out.shape[:2]:
        m = cv2.resize(
            m.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    color_arr = np.array(color, dtype=np.uint8)
    out[m] = (out[m] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def draw_results(
    image_bgr: np.ndarray,
    results: Iterable[PlateResult],
    show_mask: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    canvas = image_bgr.copy()
    font = None

    for res in results:
        if show_mask:
            canvas = overlay_mask(canvas, res.detection.mask)
        x0, y0, x1, y1 = (int(round(v)) for v in res.detection.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 255), line_thickness)
        if res.corners is not None:
            pts = res.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, (0, 0, 255), line_thickness)

    # Text labels are drawn with PIL so that Chinese characters render correctly.
    pil = Image.fromarray(canvas[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    for res in results:
        label = res.text or "(no text)"
        if res.text_score > 0:
            label = f"{label} {res.text_score:.2f}"
        if font is None:
            font = _load_font(max(16, canvas.shape[0] // 40))
        x0, y0 = int(round(res.detection.box[0])), int(round(res.detection.box[1]))
        bbox = draw.textbbox((x0, y0), label, font=font)
        pad = 3
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(255, 255, 255),
        )
        draw.text((x0, y0), label, fill=(200, 0, 0), font=font)
    return np.asarray(pil)[:, :, ::-1].copy()
