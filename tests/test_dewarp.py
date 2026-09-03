"""
Tests for YOLOv8 boundary detector and cubic geometric dewarper.
"""

import cv2
import numpy as np
import pytest

from camscan.dewarp import fit_cubic_polynomial, cubic_geometric_dewarp, YOLODewarpEngine
from camscan import scanner


def test_fit_cubic_polynomial():
    """Verify cubic polynomial fitting reproduces known curve parameters."""
    xs = np.linspace(0, 100, 20)
    # y = 2*x^3 - 5*x^2 + 3*x + 10
    ys = 2 * (xs**3) - 5 * (xs**2) + 3 * xs + 10
    poly = fit_cubic_polynomial(xs, ys)
    np.testing.assert_allclose(poly, [2, -5, 3, 10], rtol=1e-3, atol=1e-2)


def test_cubic_geometric_dewarp():
    """Verify classical geometric dewarping remapping on a curved test pattern."""
    W, H = 600, 400
    img = np.full((H, W, 3), 200, dtype=np.uint8)

    # Simulated curved top and bottom coordinates
    xs = np.linspace(50, 550, 40)
    ytop = 50 + 20 * np.sin(xs / 100.0)
    ybot = 350 + 20 * np.sin(xs / 100.0)

    top_points = np.column_stack([xs, ytop])
    bottom_points = np.column_stack([xs, ybot])

    dewarped = cubic_geometric_dewarp(
        image=img,
        top_points=top_points,
        bottom_points=bottom_points,
        output_width=500,
        output_height=300,
    )

    assert dewarped is not None
    assert dewarped.shape == (300, 500, 3)


@pytest.mark.integration
def test_yolo_dewarp_engine_sample():
    """Verify YOLOv8 boundary detection on actual sample image."""
    engine = YOLODewarpEngine()
    img = cv2.imread("tests/images/IMG_1842.jpg")
    assert img is not None

    dewarped, contour = engine.detect_and_dewarp(img)
    # YOLO should detect the document boundary and return dewarped image and corners
    assert dewarped is not None
    assert contour is not None
    assert contour.shape == (4, 2)
    assert dewarped.shape[0] > 100 and dewarped.shape[1] > 100


def test_original_contour_fallback_maintained():
    """Verify original OpenCV contour detector is intact and functional."""
    img = cv2.imread("tests/images/IMG_1842.jpg")
    result = scanner.main(img)
    assert result.contour is not None
    assert result.warped is not None
    assert result.contour.shape == (4, 2)
