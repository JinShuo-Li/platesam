"""Mask -> quadrilateral -> perspective rectification to the plate's real shape.

The standard Chinese single-row plate is 440 x 140 mm (aspect ~3.14:1); we warp
the segmented plate region to that canonical size so the OCR sees a frontal,
axis-aligned plate.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# 440 x 140 mm, the standard single-row (blue) plate size.
DEFAULT_OUT_SIZE: Tuple[int, int] = (440, 140)  # (width, height)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0])]
    left, right = x_sorted[:2], x_sorted[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def mask_to_corners(
    mask: np.ndarray,
    epsilon_ratio: float = 0.02,
    min_area: float = 16.0,
) -> Optional[np.ndarray]:
    """Extract the 4 corners of the largest connected component of ``mask``."""
    binary = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    peri = cv2.arcLength(contour, True)
    for eps in (epsilon_ratio, 0.03, 0.05):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    # Fall back to the minimum-area rotated rectangle.
    return order_points(cv2.boxPoints(cv2.minAreaRect(contour)))


def shrink_quad(corners: np.ndarray, ratio: float) -> np.ndarray:
    """Shrink a quad towards its centroid (drops plate frames/borders)."""
    if ratio <= 0:
        return corners
    centroid = corners.mean(axis=0, keepdims=True)
    return (centroid + (corners - centroid) * (1.0 - ratio)).astype(np.float32)


def perspective_rectify(
    image: np.ndarray,
    corners: np.ndarray,
    out_size: Sequence[int] = DEFAULT_OUT_SIZE,
    inset_ratio: float = 0.02,
) -> np.ndarray:
    """Warp the quad in ``image`` (BGR) to ``out_size`` = (width, height)."""
    out_w, out_h = int(out_size[0]), int(out_size[1])
    pts = shrink_quad(order_points(corners), inset_ratio)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC
    )


def rectified_long_side_is_width(corners: np.ndarray) -> bool:
    """True if the quad's long side is horizontal in the source image."""
    p = order_points(corners)
    top = np.linalg.norm(p[1] - p[0])
    bottom = np.linalg.norm(p[2] - p[3])
    left = np.linalg.norm(p[3] - p[0])
    right = np.linalg.norm(p[2] - p[1])
    return (top + bottom) >= (left + right)


def rectify_plate(
    image: np.ndarray,
    mask: np.ndarray,
    out_size: Sequence[int] = DEFAULT_OUT_SIZE,
    inset_ratio: float = 0.02,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Rectify the plate given its mask.

    Returns ``(crop, corners)``; ``crop`` is None if no valid quad was found.
    """
    corners = mask_to_corners(mask)
    if corners is None:
        return None, None
    crop = perspective_rectify(image, corners, out_size=out_size, inset_ratio=inset_ratio)
    if not rectified_long_side_is_width(corners):
        # Plate is ~90-degree rotated: rotate the warp so text is horizontal.
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop, corners
