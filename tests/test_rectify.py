import cv2
import numpy as np

from lpnrecog.rectify import (
    mask_to_corners,
    order_points,
    perspective_rectify,
    rectify_plate,
)


def _synthetic_plate(w=440, h=140):
    plate = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.rectangle(plate, (0, 0), (w - 1, h - 1), (150, 60, 0), -1)  # blue (BGR)
    cv2.putText(
        plate, "A12345", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 8
    )
    return plate


def _quad(h=140, w=440):
    return np.array(
        [[20, 30], [w + 20, 40], [w + 10, h + 30], [30, h + 20]], dtype=np.float32
    )


def test_order_points():
    pts = np.array([[10, 20], [100, 10], [110, 90], [5, 100]], dtype=np.float32)
    ordered = order_points(pts)
    assert np.allclose(ordered[0], [10, 20])
    assert np.allclose(ordered[1], [100, 10])
    assert np.allclose(ordered[2], [110, 90])
    assert np.allclose(ordered[3], [5, 100])


def test_mask_to_corners_recovers_quad():
    canvas = np.zeros((300, 600), dtype=np.uint8)
    quad = _quad().astype(np.int32)
    cv2.fillPoly(canvas, [quad], 1)
    corners = mask_to_corners(canvas.astype(bool))
    assert corners is not None
    assert corners.shape == (4, 2)
    # Every recovered corner should be close to one of the true quad corners.
    for corner in corners:
        dist = np.linalg.norm(quad.astype(np.float32) - corner, axis=1).min()
        assert dist < 4.0


def test_perspective_rectify_size_and_content():
    plate = _synthetic_plate()
    quad = _quad()
    src = np.zeros((300, 600, 3), dtype=np.uint8)
    src[30:170, 20:460] = plate

    warped = cv2.warpPerspective(
        plate,
        cv2.getPerspectiveTransform(
            np.array([[0, 0], [439, 0], [439, 139], [0, 139]], dtype=np.float32), quad
        ),
        (600, 300),
    )
    mask = np.zeros((300, 600), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 1)

    crop, corners = rectify_plate(warped, mask.astype(bool), inset_ratio=0.0)
    assert crop is not None
    assert corners is not None
    assert crop.shape[:2] == (140, 440)
    # After rectification the crop should mostly match the original plate layout.
    diff = np.abs(crop.astype(np.int16) - plate.astype(np.int16)).mean()
    assert diff < 60, diff


def test_rectify_rotated_plate_is_rotated_back():
    plate = _synthetic_plate()
    quad = _quad()
    warped = cv2.warpPerspective(
        plate,
        cv2.getPerspectiveTransform(
            np.array([[0, 0], [439, 0], [439, 139], [0, 139]], dtype=np.float32), quad
        ),
        (600, 300),
    )
    mask = np.zeros((300, 600), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 1)
    crop, _ = rectify_plate(warped, mask.astype(bool), inset_ratio=0.0)
    assert crop is not None and crop.shape[:2] == (140, 440)


def test_mask_to_corners_empty():
    assert mask_to_corners(np.zeros((50, 50), dtype=bool)) is None
