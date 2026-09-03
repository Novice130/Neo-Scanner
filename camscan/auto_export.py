"""
Module for auto-exporting finalized student sessions to a watched folder (e.g. OneDrive).
"""

import logging
import os
from pathlib import Path
import typing as t

import cv2
import numpy as np

from camscan import session, pdf_builder

if t.TYPE_CHECKING:
    from camscan.ocr import BaseOCREngine

logger = logging.getLogger(__name__)

DEFAULT_WATCHED_FOLDER = str(Path.home() / "OneDrive" / "CamScan")


class AutoExporter:
    """
    Handles automatic export of finalized student scanning sessions
    to a designated watched directory (e.g., local OneDrive sync folder).
    """

    def __init__(
        self,
        watched_folder: str = DEFAULT_WATCHED_FOLDER,
        export_pdf: bool = True,
        export_separate_images: bool = False,
    ):
        self.watched_folder = os.path.expanduser(watched_folder)
        self.export_pdf = export_pdf
        self.export_separate_images = export_separate_images

    def set_watched_folder(self, folder_path: str):
        """Set and expand destination watched folder."""
        self.watched_folder = os.path.expanduser(folder_path.strip())

    def export_session(
        self,
        images: list[np.ndarray],
        student_tag: str,
        ocr_engine: t.Optional["BaseOCREngine"] = None,
        progress_callback: t.Optional[t.Callable[[str], None]] = None,
    ) -> dict[str, str]:
        """
        Export a finalized session's images to the watched folder.
        :param images: List of OpenCV BGR images
        :param student_tag: Student identifier or name
        :param ocr_engine: Optional OCR engine for searchable handwriting PDF
        :param progress_callback: Optional status callback
        :return: Dict with paths of exported artifacts
        """
        if not images:
            return {}

        os.makedirs(self.watched_folder, exist_ok=True)
        results = {}

        # 1. Export merged PDF
        if self.export_pdf:
            pdf_filename = session.generate_session_filename(
                student_tag=student_tag, ext="pdf"
            )
            pdf_path = os.path.join(self.watched_folder, pdf_filename)

            ocr_results = None
            if ocr_engine is not None:
                ocr_results = []
                for idx, img in enumerate(images):
                    if progress_callback:
                        progress_callback(
                            f"Auto-export: OCR on page {idx + 1}/{len(images)}..."
                        )
                    try:
                        lines = ocr_engine.recognize(img)
                    except Exception as e:
                        logger.warning(f"OCR failed on page {idx + 1}: {e}")
                        lines = []
                    ocr_results.append(lines)

            if progress_callback:
                progress_callback("Auto-export: Writing PDF to watched folder...")

            pdf_builder.create_searchable_pdf(
                images=images,
                ocr_results=ocr_results,
                output_path=pdf_path,
            )
            results["pdf"] = pdf_path
            logger.info(f"Auto-exported session PDF to {pdf_path}")

        # 2. Export separate page images if configured
        if self.export_separate_images:
            session_dir_name = session.generate_session_dirname(student_tag=student_tag)
            session_dir_path = os.path.join(self.watched_folder, session_dir_name)
            os.makedirs(session_dir_path, exist_ok=True)

            for idx, img in enumerate(images, start=1):
                img_name = f"{idx:03d}_{session_dir_name}.png"
                img_path = os.path.join(session_dir_path, img_name)
                cv2.imwrite(img_path, img)

            results["images_dir"] = session_dir_path
            logger.info(f"Auto-exported session images to {session_dir_path}")

        return results
