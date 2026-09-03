"""
Module for YOLOv8-based document boundary detection and classical geometric dewarping
using cubic polynomial interpolation and remapping to flatten curved notebook pages.
"""

import logging
import typing as t

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def fit_cubic_polynomial(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Fit a cubic polynomial y = a*x^3 + b*x^2 + c*x + d.
    Falls back to linear if fewer than 4 unique points are available.
    """
    deg = 3 if len(np.unique(xs)) >= 4 else 1
    return np.polyfit(xs, ys, deg=deg)


def cubic_geometric_dewarp(
    image: np.ndarray,
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    output_width: t.Optional[int] = None,
    output_height: t.Optional[int] = None,
) -> np.ndarray:
    """
    Apply classical geometric correction using cubic polynomial interpolation and
    bicubic pixel remapping to flatten curved notebook pages.

    :param image: Input OpenCV image
    :param top_points: Array of points (x, y) along the top curved edge
    :param bottom_points: Array of points (x, y) along the bottom curved edge
    :param output_width: Desired rectified width (optional)
    :param output_height: Desired rectified height (optional)
    :return: Dewarped, flattened image
    """
    # Sort points left-to-right along x
    top_sorted = top_points[np.argsort(top_points[:, 0])]
    bot_sorted = bottom_points[np.argsort(bottom_points[:, 0])]

    p_top = fit_cubic_polynomial(top_sorted[:, 0], top_sorted[:, 1])
    p_bot = fit_cubic_polynomial(bot_sorted[:, 0], bot_sorted[:, 1])

    x_min = max(float(np.min(top_sorted[:, 0])), float(np.min(bot_sorted[:, 0])))
    x_max = min(float(np.max(top_sorted[:, 0])), float(np.max(bot_sorted[:, 0])))

    if x_max <= x_min + 10:
        x_min = float(min(np.min(top_sorted[:, 0]), np.min(bot_sorted[:, 0])))
        x_max = float(max(np.max(top_sorted[:, 0]), np.max(bot_sorted[:, 0])))

    if output_width is None:
        output_width = max(50, int(x_max - x_min))

    # Evaluate height across span
    sample_xs = np.linspace(x_min, x_max, 50)
    sample_yt = np.polyval(p_top, sample_xs)
    sample_yb = np.polyval(p_bot, sample_xs)
    avg_h = float(np.mean(np.abs(sample_yb - sample_yt)))

    if output_height is None:
        output_height = max(50, int(avg_h))

    # Generate 2D meshgrid of target rectified coordinates
    u = np.linspace(x_min, x_max, output_width, dtype=np.float32)
    v = np.linspace(0.0, 1.0, output_height, dtype=np.float32)

    uu, vv = np.meshgrid(u, v)

    # Compute source coordinates via cubic polynomial interpolation
    y_top_eval = np.polyval(p_top, uu)
    y_bot_eval = np.polyval(p_bot, uu)

    map_x = uu.astype(np.float32)
    map_y = ((1.0 - vv) * y_top_eval + vv * y_bot_eval).astype(np.float32)

    # Remap using bicubic interpolation
    dewarped = cv2.remap(
        src=image,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return dewarped


class YOLODewarpEngine:
    """
    Document boundary detector using YOLOv8 segmentation and classical geometric dewarper.
    """

    def __init__(self, model_name: str = "yolov8n-seg.pt"):
        self.model_name = model_name
        self._model = None

    def _init_model(self):
        if self._model is None:
            logger.info(f"Loading YOLOv8 model: {self.model_name}")
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)

    def detect_and_dewarp(
        self,
        image: np.ndarray,
        target_height: int = 640,
    ) -> tuple[t.Optional[np.ndarray], t.Optional[np.ndarray]]:
        """
        Detect document boundary polygon with YOLOv8 and apply cubic polynomial dewarping.

        :param image: Input OpenCV BGR image
        :param target_height: Height to scale image for fast YOLO inference
        :return: (dewarped_image, 4_corner_contour)
        """
        self._init_model()
        h, w = image.shape[:2]

        # Downscale for fast inference if needed
        scale = 1.0
        if h > target_height:
            scale = target_height / float(h)
            inference_img = cv2.resize(
                image, (int(w * scale), target_height), interpolation=cv2.INTER_AREA
            )
        else:
            inference_img = image

        results = self._model(inference_img, verbose=False, conf=0.2)
        if not results or results[0].masks is None or len(results[0].masks.xy) == 0:
            return None, None

        # Find mask with largest area
        masks_xy = results[0].masks.xy
        best_poly = max(masks_xy, key=lambda poly: cv2.contourArea(poly))

        # Check minimum area
        inf_h, inf_w = inference_img.shape[:2]
        if cv2.contourArea(best_poly) < 0.15 * (inf_h * inf_w):
            return None, None

        # Scale polygon coordinates back to original image
        orig_poly = (best_poly / scale).astype(np.float32)

        # Identify corners: TL, TR, BR, BL
        s = orig_poly[:, 0] + orig_poly[:, 1]
        tl_idx = int(np.argmin(s))
        br_idx = int(np.argmax(s))

        diff = orig_poly[:, 0] - orig_poly[:, 1]
        tr_idx = int(np.argmax(diff))
        bl_idx = int(np.argmin(diff))

        corners = np.array(
            [
                orig_poly[tl_idx],
                orig_poly[tr_idx],
                orig_poly[br_idx],
                orig_poly[bl_idx],
            ],
            dtype=np.int32,
        )

        tl = orig_poly[tl_idx]
        tr = orig_poly[tr_idx]
        bl = orig_poly[bl_idx]
        br = orig_poly[br_idx]

        # Split polygon into top and bottom segments
        # Top segment: points with y near the upper half of [tl, tr]
        x_min = min(tl[0], bl[0])
        x_max = max(tr[0], br[0])
        mid_y = (tl[1] + tr[1] + bl[1] + br[1]) / 4.0

        top_pts = []
        bot_pts = []

        for pt in orig_poly:
            if pt[1] < mid_y:
                top_pts.append(pt)
            else:
                bot_pts.append(pt)

        # Ensure endpoints are included
        top_pts.append(tl)
        top_pts.append(tr)
        bot_pts.append(bl)
        bot_pts.append(br)

        top_arr = np.array(top_pts)
        bot_arr = np.array(bot_pts)

        try:
            dewarped = cubic_geometric_dewarp(
                image=image,
                top_points=top_arr,
                bottom_points=bot_arr,
            )
            return dewarped, corners
        except Exception as e:
            logger.warning(f"Geometric dewarping failed: {e}")
            return None, corners
