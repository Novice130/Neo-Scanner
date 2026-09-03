"""
Tests for auto-export to watched folder.
"""

import os
import cv2
import numpy as np
import pymupdf
import pytest

from camscan.auto_export import AutoExporter


def test_auto_export_session_pdf(tmp_path):
    watched_dir = str(tmp_path / "OneDrive" / "CamScan")
    exporter = AutoExporter(watched_folder=watched_dir, export_pdf=True, export_separate_images=True)

    img1 = np.full((400, 600, 3), 220, dtype=np.uint8)
    cv2.putText(img1, "Page 1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    img2 = np.full((400, 600, 3), 240, dtype=np.uint8)
    cv2.putText(img2, "Page 2", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    results = exporter.export_session(
        images=[img1, img2],
        student_tag="Student_99",
        ocr_engine=None,
    )

    assert "pdf" in results
    assert os.path.exists(results["pdf"])
    assert "Student_99" in results["pdf"]

    # Verify PDF content
    doc = pymupdf.open(results["pdf"])
    assert len(doc) == 2
    doc.close()

    assert "images_dir" in results
    assert os.path.isdir(results["images_dir"])
    images_in_dir = os.listdir(results["images_dir"])
    assert len(images_in_dir) == 2
