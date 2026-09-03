"""
Module for generating searchable PDFs with embedded images and invisible text layers.
"""

import logging
import typing as t

import cv2
import numpy as np
import pymupdf

if t.TYPE_CHECKING:
    from camscan.ocr import TextLine

logger = logging.getLogger(__name__)


def create_searchable_pdf(
    images: list[np.ndarray],
    ocr_results: t.Optional[list[list["TextLine"]]] = None,
    output_path: str = "output.pdf",
    jpeg_quality: int = 92,
) -> str:
    """
    Build a multi-page PDF where each page embeds the scanned image as the visible layer
    and recognized text lines as an invisible, selectable, and searchable text layer.

    :param images: List of OpenCV BGR images (one per page)
    :param ocr_results: Optional list of TextLine lists (one per page)
    :param output_path: Destination file path for the PDF
    :param jpeg_quality: Compression quality for the embedded background image
    :return: The output file path
    """
    if not images:
        raise ValueError("Cannot create PDF with no images.")

    doc = pymupdf.open()

    for idx, img in enumerate(images):
        h, w = img.shape[:2]
        page = doc.new_page(width=w, height=h)
        full_rect = pymupdf.Rect(0, 0, w, h)

        # Encode image to JPEG bytes
        success, encoded = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not success:
            raise RuntimeError(f"Failed to encode page {idx + 1} to JPEG.")

        page.insert_image(full_rect, stream=encoded.tobytes())

        # Embed invisible searchable text layer if OCR results exist for this page
        if ocr_results and idx < len(ocr_results) and ocr_results[idx]:
            for line in ocr_results[idx]:
                if not line.text or not line.text.strip():
                    continue

                xmin, ymin, xmax, ymax = line.box
                line_h = max(8, ymax - ymin)
                fontsize = max(6, int(line_h * 0.8))
                baseline_y = min(h - 1, ymax - int(line_h * 0.15))
                baseline_x = max(0, xmin)

                try:
                    # render_mode=3 is PDF standard for invisible text (neither stroke nor fill)
                    # Text is fully searchable and selectable without altering the visual appearance
                    page.insert_text(
                        point=(baseline_x, baseline_y),
                        text=line.text,
                        fontsize=fontsize,
                        render_mode=3,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to insert text '{line.text}' at ({baseline_x}, {baseline_y}): {e}"
                    )

    doc.save(output_path, deflate=True, garbage=3)
    doc.close()
    logger.info(f"Successfully wrote {len(images)}-page searchable PDF to {output_path}")
    return output_path
