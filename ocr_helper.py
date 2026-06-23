from __future__ import annotations

import os
import re
from pathlib import Path
from PIL import Image

class OCRHelper:
    """
    A unified wrapper for text extraction supporting EasyOCR, PyTesseract,
    and a robust Mock OCR fallback for offline/development environments.
    """
    def __init__(self, engine_name: str | None = None) -> None:
        self.engine_name = (engine_name or os.getenv("OCR_ENGINE", "mock")).lower()
        self.reader = None

        if self.engine_name == "easyocr":
            try:
                import easyocr
                print("[OCR] Initializing EasyOCR Reader (English)...")
                self.reader = easyocr.Reader(["en"])
            except ImportError:
                print("[OCR] [WARNING] easyocr package not found. Falling back to Mock OCR.")
                self.engine_name = "mock"
            except Exception as e:
                print(f"[OCR] [WARNING] EasyOCR failed to load: {e}. Falling back to Mock OCR.")
                self.engine_name = "mock"

        elif self.engine_name == "pytesseract":
            try:
                import pytesseract
                # Test call to ensure pytesseract is installed and path is resolved
                pytesseract.get_tesseract_version()
                print("[OCR] Initialized PyTesseract OCR Engine.")
            except ImportError:
                print("[OCR] [WARNING] pytesseract package not found. Falling back to Mock OCR.")
                self.engine_name = "mock"
            except Exception as e:
                print(
                    f"[OCR] [WARNING] PyTesseract/Tesseract binary not found on this system: {e}. "
                    "Falling back to Mock OCR."
                )
                self.engine_name = "mock"

        if self.engine_name == "mock":
            print("[OCR] Running in Mock OCR mode (no external OCR library required).")

    def extract_text(self, image_path: str | Path, is_question_anchor: bool = False, block_idx: int = 1) -> str:
        """
        Extracts text from the image block crop.
        `is_question_anchor` and `block_idx` are used to generate realistic mock texts when in mock mode.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return ""

        if self.engine_name == "easyocr" and self.reader is not None:
            try:
                # Run EasyOCR
                results = self.reader.readtext(str(image_path), detail=0)
                text = " ".join(results).strip()
                return text
            except Exception as e:
                print(f"[OCR] EasyOCR run failed: {e}. Falling back to mock text extraction.")

        elif self.engine_name == "pytesseract":
            try:
                import pytesseract
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img).strip()
                # Clean up multiple newlines/whitespace
                text = re.sub(r'\s+', ' ', text)
                return text
            except Exception as e:
                print(f"[OCR] PyTesseract run failed: {e}. Falling back to mock text extraction.")

        # Default Mock OCR Fallback
        # Returns realistic text depending on whether it is a question label or student answer
        if is_question_anchor:
            # Generate a realistic question label e.g., "Q1" or "Question 1"
            return f"Question {block_idx}"
        else:
            return f"Student handwritten response content for block {block_idx}"
