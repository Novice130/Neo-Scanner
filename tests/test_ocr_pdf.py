"""
Tests for OCR models and Searchable PDF generation.
"""

import numpy as np
import cv2
import pymupdf
import pytest

from camscan.ocr import TextLine, get_ocr_engine
from camscan.pdf_builder import create_searchable_pdf


def test_create_searchable_pdf(tmp_path):
    """Verify searchable PDF creation with PyMuPDF text layer."""
    # Create synthetic test page image
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(
        img, "Student Handwriting Test", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )

    text_lines = [
        TextLine(box=(50, 160, 450, 210), text="Student Handwriting Test"),
        TextLine(box=(50, 260, 350, 310), text="Page 1 Notes"),
    ]

    pdf_path = str(tmp_path / "test_output.pdf")
    create_searchable_pdf(
        images=[img],
        ocr_results=[text_lines],
        output_path=pdf_path,
    )

    # Verify that the PDF is searchable and text can be extracted
    doc = pymupdf.open(pdf_path)
    assert len(doc) == 1
    page_text = doc[0].get_text()
    doc.close()

    assert "Student Handwriting Test" in page_text
    assert "Page 1 Notes" in page_text


def test_ocr_engine_factory():
    """Verify get_ocr_engine factory maps modes correctly."""
    engine_handwriting = get_ocr_engine("PaddleOCR + TrOCR (Handwriting)")
    assert engine_handwriting is not None

    engine_none = get_ocr_engine("None")
    assert engine_none is None

    engine_llm = get_ocr_engine("Vision LLM API")
    assert engine_llm is not None


@pytest.mark.integration
def test_paddle_trocr_pipeline_integration(tmp_path):
    """Test full PaddleOCR line detection + TrOCR handwriting recognition on a synthesized handwriting patch."""
    engine = get_ocr_engine("PaddleOCR + TrOCR (Handwriting)")
    assert engine is not None

    # Create a test image with distinct text
    test_img = np.full((300, 600, 3), 255, dtype=np.uint8)
    cv2.putText(
        test_img,
        "MATH HOMEWORK",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3,
        lineType=cv2.LINE_AA,
    )

    lines = engine.recognize(test_img)
    assert isinstance(lines, list)
    assert len(lines) > 0

    # Build searchable PDF from it
    pdf_path = str(tmp_path / "handwriting_searchable.pdf")
    create_searchable_pdf(images=[test_img], ocr_results=[lines], output_path=pdf_path)

    doc = pymupdf.open(pdf_path)
    extracted = doc[0].get_text()
    doc.close()
    assert len(extracted.strip()) > 0

