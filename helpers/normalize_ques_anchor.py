from __future__ import annotations

import re


class QuestionAnchorNormalizer:
    """Parses and cleans raw OCR text extracted from question blocks to identify

    consistent, structured question anchor strings (e.g., Q1, 2.a).
    """

    def normalize(self, ocr_text: str) -> str:
        """Cleans and extracts structured question numbers or identifiers from raw text."""
        text = ocr_text.strip()
        if not text:
            return "Q_unknown"

        # Try to find patterns like "Q1", "Q.1", "Question 1", "1a"
        match = re.search(
            r"\b(Q(uestion)?\.?\s*\d+\s*[a-zA-Z]?)\b", text, re.IGNORECASE
        )
        if match:
            return match.group(1).replace(" ", "").upper()

        # Try to find starting numbers/alphabets with standard delimiters e.g. "1.", "a)"
        match_num = re.match(r"^(\d+[\.\-\s]+[a-zA-Z]?|^[a-zA-Z][\)\.])", text)
        if match_num:
            return match_num.group(1).strip(". - ) (")

        # Fallback to the first 2-3 words or a truncated version
        words = text.split()
        if len(words) <= 3:
            return text
        return " ".join(words[:2])
