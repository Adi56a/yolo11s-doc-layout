from __future__ import annotations

import shutil
from pathlib import Path
from typing import List
import cv2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class AnswerSheetSplitter:
    """Handles splitting student PDF answer sheets into high-quality page images,

    or copying raw input images for single-page processing.
    """

    def __init__(self, dpi: int = 200) -> None:
        self.dpi = dpi

    def split(self, file_path: str | Path, temp_dir: str | Path) -> List[Path]:
        """Splits PDF page-by-page into PNGs, or prepares an input image.

        Returns a list of Path objects pointing to page images in the temp_dir.
        """
        file_path = Path(file_path)
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        suffix = file_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            dest = temp_dir / "page_001.png"
            # Read and write via OpenCV to ensure clean layout pipeline compatibility
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Could not read input image: {file_path}")
            cv2.imwrite(str(dest), img)
            return [dest]

        if suffix == ".pdf":
            if fitz is None:
                raise ImportError(
                    "PyMuPDF (fitz) is required to process PDF files. "
                    "Please run 'pip install pymupdf' or input a single image file directly."
                )

            print(f"[PDF] Splitting PDF: {file_path}")
            doc = fitz.open(str(file_path))
            page_paths: List[Path] = []

            # Calculate zoom from DPI (72 DPI is PyMuPDF default)
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                out_path = temp_dir / f"page_{i:03d}.png"
                pix.save(str(out_path))
                page_paths.append(out_path)

            print(f"[PDF] Rendered {len(page_paths)} pages.")
            return page_paths

        raise ValueError(
            f"Unsupported file format '{suffix}'. Only PDFs and standard images are supported."
        )
