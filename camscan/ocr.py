"""
OCR module providing handwriting-optimized transcription pipelines.
Supports:
1. Two-stage local pipeline: PaddleOCR for text line detection + TrOCR for handwriting recognition.
2. Vision LLM fallback: Cloud vision API transcription (e.g., Gemini).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import os
import typing as t

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TextLine:
    """Represents a recognized line of text with its bounding box."""
    # Bounding box in pixel coordinates: (x_min, y_min, x_max, y_max)
    box: tuple[int, int, int, int]
    text: str
    confidence: float = 1.0


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        progress_callback: t.Optional[t.Callable[[str], None]] = None,
    ) -> list[TextLine]:
        """
        Run OCR on an image and return recognized text lines with bounding boxes.
        :param image: OpenCV BGR image
        :param progress_callback: Optional callback for status updates
        :return: List of TextLine objects
        """
        pass


class PaddleTrOCREngine(BaseOCREngine):
    """
    Two-stage handwriting OCR pipeline:
    - Stage 1: PaddleOCR detects text line locations / bounding polygons.
    - Stage 2: Hugging Face TrOCR performs handwriting recognition on each line crop.
    """

    def __init__(
        self,
        trocr_model_name: str = "microsoft/trocr-small-handwritten",
        device: t.Optional[str] = None,
    ):
        self.trocr_model_name = trocr_model_name
        self.device = device
        self._paddle_ocr = None
        self._trocr_processor = None
        self._trocr_model = None

    def _init_models(self):
        """Lazy load the models on first use."""
        import torch

        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        if self._paddle_ocr is None:
            logger.info("Initializing PaddleOCR text detection pipeline...")
            from paddleocr import PaddleOCR

            # Initialize PaddleOCR
            self._paddle_ocr = PaddleOCR()

        if self._trocr_model is None or self._trocr_processor is None:
            logger.info(
                f"Initializing TrOCR ({self.trocr_model_name}) on {self.device}..."
            )
            from transformers import (
                AutoImageProcessor,
                XLMRobertaTokenizer,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )

            image_processor = AutoImageProcessor.from_pretrained(
                self.trocr_model_name
            )
            tokenizer = XLMRobertaTokenizer.from_pretrained(self.trocr_model_name)
            self._trocr_processor = TrOCRProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                self.trocr_model_name
            ).to(self.device)
            self._trocr_model.eval()

    def recognize(
        self,
        image: np.ndarray,
        progress_callback: t.Optional[t.Callable[[str], None]] = None,
    ) -> list[TextLine]:
        """
        Extract text lines using PaddleOCR detection and transcribe each with TrOCR.
        """
        self._init_models()
        import torch

        h, w = image.shape[:2]
        if progress_callback:
            progress_callback("Detecting text lines with PaddleOCR...")

        # Run PaddleOCR detection
        predict_results = list(self._paddle_ocr.predict(image))
        if not predict_results:
            return []

        ocr_res = predict_results[0]
        # In PaddleX OCRResult, rec_boxes or dt_polys contain the line bounds
        boxes = []
        if "rec_boxes" in ocr_res and ocr_res["rec_boxes"] is not None and len(ocr_res["rec_boxes"]) > 0:
            for b in ocr_res["rec_boxes"]:
                # [xmin, ymin, xmax, ymax]
                boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        elif "dt_polys" in ocr_res and ocr_res["dt_polys"] is not None:
            for poly in ocr_res["dt_polys"]:
                poly_arr = np.array(poly)
                xmin = int(np.min(poly_arr[:, 0]))
                xmax = int(np.max(poly_arr[:, 0]))
                ymin = int(np.min(poly_arr[:, 1]))
                ymax = int(np.max(poly_arr[:, 1]))
                boxes.append((xmin, ymin, xmax, ymax))

        # Sort boxes top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b[1], b[0]))

        text_lines: list[TextLine] = []
        total_boxes = len(boxes)
        logger.info(f"PaddleOCR detected {total_boxes} text line regions.")

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            if progress_callback and total_boxes > 0:
                progress_callback(
                    f"Transcribing handwriting {idx + 1}/{total_boxes} with TrOCR..."
                )

            # Clamp coordinates with slight padding
            pad_x = int(0.02 * (xmax - xmin))
            pad_y = int(0.05 * (ymax - ymin))
            c_xmin = max(0, xmin - pad_x)
            c_ymin = max(0, ymin - pad_y)
            c_xmax = min(w, xmax + pad_x)
            c_ymax = min(h, ymax + pad_y)

            # Skip degenerate boxes
            if c_xmax - c_xmin < 5 or c_ymax - c_ymin < 5:
                continue

            crop = pil_img.crop((c_xmin, c_ymin, c_xmax, c_ymax))

            try:
                pixel_values = self._trocr_processor(
                    images=crop, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.no_grad():
                    generated_ids = self._trocr_model.generate(
                        pixel_values, max_new_tokens=64
                    )
                text = self._trocr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

                if text:
                    text_lines.append(
                        TextLine(
                            box=(xmin, ymin, xmax, ymax),
                            text=text,
                            confidence=1.0,
                        )
                    )
            except Exception as e:
                logger.warning(f"TrOCR failed on crop {idx}: {e}")

        return text_lines


class VisionLLMOCREngine(BaseOCREngine):
    """
    Vision LLM fallback OCR engine using Gemini API.
    Sends full page image to the model and requests line-by-line transcription with bounding boxes.
    """

    def __init__(self, api_key: t.Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable or api_key parameter is required for VisionLLMOCREngine."
                )
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def recognize(
        self,
        image: np.ndarray,
        progress_callback: t.Optional[t.Callable[[str], None]] = None,
    ) -> list[TextLine]:
        if progress_callback:
            progress_callback("Sending page to Vision LLM for transcription...")

        client = self._get_client()
        h, w = image.shape[:2]

        # Encode image as JPEG bytes
        success, encoded_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not success:
            raise ValueError("Failed to encode image for Vision LLM.")

        image_bytes = encoded_img.tobytes()

        prompt = (
            "Analyze this handwritten document page. Transcribe each line of handwritten text "
            "and provide its approximate 2D bounding box in normalized coordinates [ymin, xmin, ymax, xmax] "
            "scaled to 1000 (0 to 1000).\n"
            "Return JSON only as an array of objects: "
            '[{"box_2d": [ymin, xmin, ymax, xmax], "text": "transcribed text"}]. '
            "Do not wrap in markdown or extra commentary."
        )

        from google.genai import types

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )

        response_text = response.text.strip()
        # Strip code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            items = json.loads(response_text)
        except Exception as e:
            logger.error(f"Failed to parse Vision LLM JSON: {e}\nResponse: {response_text}")
            # Fallback: treat entire response as single text line covering center
            return [TextLine(box=(0, 0, w, h), text=response.text.strip())]

        text_lines: list[TextLine] = []
        for item in items:
            box_2d = item.get("box_2d")
            text = item.get("text", "").strip()
            if not text:
                continue

            if box_2d and len(box_2d) == 4:
                ymin, xmin, ymax, xmax = box_2d
                # Denormalize from 1000 to pixel dimensions
                px_xmin = int(xmin * w / 1000.0)
                px_ymin = int(ymin * h / 1000.0)
                px_xmax = int(xmax * w / 1000.0)
                px_ymax = int(ymax * h / 1000.0)
                text_lines.append(
                    TextLine(
                        box=(px_xmin, px_ymin, px_xmax, px_ymax),
                        text=text,
                    )
                )
            else:
                text_lines.append(TextLine(box=(0, 0, w, h), text=text))

        return text_lines


_CACHED_ENGINES: dict[str, BaseOCREngine] = {}


def get_ocr_engine(engine_type: str) -> t.Optional[BaseOCREngine]:
    """
    Factory function returning an OCR engine instance by type name:
    - 'PaddleOCR + TrOCR (Handwriting)'
    - 'Vision LLM API'
    - 'None'
    """
    normalized = engine_type.strip().lower()

    if "trocr" in normalized or "paddle" in normalized or "handwriting" in normalized:
        if "paddle_trocr" not in _CACHED_ENGINES:
            _CACHED_ENGINES["paddle_trocr"] = PaddleTrOCREngine()
        return _CACHED_ENGINES["paddle_trocr"]

    if "vision" in normalized or "llm" in normalized or "gemini" in normalized:
        if "vision_llm" not in _CACHED_ENGINES:
            _CACHED_ENGINES["vision_llm"] = VisionLLMOCREngine()
        return _CACHED_ENGINES["vision_llm"]

    return None
