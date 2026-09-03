"""
This module provides utility postprocessing functions for images.
"""

import cv2


def dummy(image: cv2.Mat) -> cv2.Mat:
    """
    Apply no processing whatsoever and simply return the image again.
    :param image: The input image
    :return: The original image with no modification
    """
    return image


def sharpen(image: cv2.Mat) -> cv2.Mat:
    """
    Apply a sharpening effect to the input image.
    :param image: The input image
    :return: The image with the effect applied
    """
    blurred = cv2.GaussianBlur(
        src=image,
        ksize=(0, 0),
        sigmaX=3,
    )
    sharpened = cv2.addWeighted(
        src1=image,
        alpha=1.5,
        src2=blurred,
        beta=-0.5,
        gamma=0,
    )
    return sharpened


def grayscale(image: cv2.Mat) -> cv2.Mat:
    """
    Convert the input image to grayscale.
    :param image: The input image
    :return: The image with the effect applied
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def black_and_white(image: cv2.Mat) -> cv2.Mat:
    """
    Convert the image to black and white (it looks like a pencil sketch).
    This is done by converting it to grayscale, applying a sharpening effect,
    and then an adaptive threshold.
    :param image: The input image
    :return: The image with the effect applied
    """
    gray = grayscale(image=image)
    sharpened = sharpen(image=gray)
    thresholded = cv2.adaptiveThreshold(
        src=sharpened,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
    return thresholded


def magic_color(image: cv2.Mat) -> cv2.Mat:
    """
    CamScanner / Google Lens Magic Color illumination algorithm:
    1. Background illumination estimation via morphological filter
    2. High-pass division to strip shadows and equalize paper to bright white
    3. Contrast stretching (CLAHE) to deepen black text
    4. Unsharp masking to make character glyphs sharp and crisp
    """
    if image is None:
        return image

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Illumination background estimation
    ksize = max(31, int(min(image.shape[:2]) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    bg = cv2.morphologyEx(l, cv2.MORPH_DILATE, kernel)
    bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

    # Shadow removal & paper illumination division
    norm = cv2.divide(l, bg, scale=255)

    # Adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # Text sharpening
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharp = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)

    out_lab = cv2.merge([sharp, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

