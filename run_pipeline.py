from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Load local helpers
sys.path.append(str(Path(__file__).parent.resolve()))
from ocr_helper import OCRHelper
from s3_helper import S3StorageHelper


def load_env(dotenv_path: str = ".env") -> None:
    """Loads environment variables from a local .env file if it exists."""
    path = Path(dotenv_path)
    if path.exists():
        print(f"[ENV] Loading variables from {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    # Remove surrounding quotes if present
                    val = val.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val
    else:
        print("[ENV] No .env file found. Using system environment variables.")


class MarksensePipeline:
    """
    Production-grade class-based pipeline for the Marksense document layout analysis.
    Coordinates PDF page rendering, YOLO block detection, crop generation,
    cloud/local storage, OCR text extraction, and question-to-answer mapping.
    """
    def __init__(self, env_path: str = ".env") -> None:
        load_env(env_path)
        
        # Initialize storage helper
        use_local = os.getenv("USE_LOCAL_STORAGE", "True").lower() == "true"
        local_dir = os.getenv("LOCAL_STORAGE_DIR", "outputs/local_s3_mock")
        bucket = os.getenv("S3_BUCKET_NAME", "edtech-b2b-media")
        region = os.getenv("AWS_REGION", "ap-south-1")
        
        self.storage = S3StorageHelper(
            bucket_name=bucket,
            region=region,
            use_local_storage=use_local,
            local_storage_dir=local_dir
        )

        # Initialize OCR helper
        ocr_engine = os.getenv("OCR_ENGINE", "mock")
        self.ocr = OCRHelper(engine_name=ocr_engine)

        # Initialize YOLO model weights and parameters
        self.weights_path = self.resolve_weights()
        self.conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
        self.model = self.load_model()

    def resolve_weights(self) -> Path:
        """Finds the YOLO model weights path with fallbacks."""
        configured_path = os.getenv("YOLO_WEIGHTS_PATH", "models/110-best.pt")
        path = Path(configured_path)
        if path.exists():
            return path

        # Fallbacks in script folder or project parent
        script_dir = Path(__file__).parent.resolve()
        fallbacks = [
            script_dir / "models" / "110-best.pt",
            script_dir / "models" / "110-best.onnx",
            script_dir / "models" / "best.pt",
            script_dir / "models" / "best.onnx",
            Path("models") / "110-best.pt",
            Path("models") / "best.pt"
        ]
        for f in fallbacks:
            if f.exists():
                return f

        raise FileNotFoundError(f"YOLO weights not found. Configured path was: {configured_path}")

    def load_model(self) -> Any:
        """Loads ONNX Runtime or PyTorch model based on file type."""
        suffix = self.weights_path.suffix.lower()
        print(f"[MODEL] Loading YOLO model weights from: {self.weights_path.resolve()}")
        
        if suffix == ".onnx":
            try:
                from onnx_inference import YOLOONNX
                return YOLOONNX(str(self.weights_path))
            except Exception as e:
                print(f"[MODEL] [ERROR] Failed to load ONNX model: {e}")
                # Try PyTorch as backup
                print("[MODEL] Retrying loading weights using PyTorch/Ultralytics...")
                from ultralytics import YOLO
                return YOLO(str(self.weights_path))
        else:
            from ultralytics import YOLO
            return YOLO(str(self.weights_path))

    def split_pdf_to_pages(self, pdf_path: Path, temp_dir: Path) -> List[Path]:
        """Converts a PDF file page-by-page to PNG images."""
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # If the input is already an image, copy it and return it as page 1
        suffix = pdf_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            dest = temp_dir / "page_001.png"
            shutil_img = cv2.imread(str(pdf_path))
            cv2.imwrite(str(dest), shutil_img)
            return [dest]

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required to process PDF files. "
                "Please run 'pip install pymupdf' or input a single image file directly."
            )

        print(f"[PDF] Splitting PDF: {pdf_path}")
        doc = fitz.open(str(pdf_path))
        page_paths: List[Path] = []
        
        # 200 DPI is standard for document block layout analysis
        zoom = 200 / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = temp_dir / f"page_{i:03d}.png"
            pix.save(str(out_path))
            page_paths.append(out_path)

        print(f"[PDF] Rendered {len(page_paths)} pages.")
        return page_paths

    def sort_blocks_reading_order(self, detections: List[Dict[str, Any]], y_threshold: int = 20) -> List[Dict[str, Any]]:
        """
        Sorts layout block detections based on Y reading coordinates (top-to-bottom)
        and then X coordinates (left-to-right) for blocks sharing a similar Y coordinate.
        """
        if not detections:
            return []

        # Sort primarily by y_min (top), then by x_min (left)
        sorted_by_y = sorted(detections, key=lambda d: (d["bbox_raw"][1], d["bbox_raw"][0]))

        grouped: List[List[Dict[str, Any]]] = []
        current_group = [sorted_by_y[0]]

        for det in sorted_by_y[1:]:
            # If the top Y distance is within threshold, group them in the same row
            if abs(det["bbox_raw"][1] - current_group[-1]["bbox_raw"][1]) <= y_threshold:
                current_group.append(det)
            else:
                grouped.append(current_group)
                current_group = [det]
        grouped.append(current_group)

        # Sort each grouped row left-to-right (by X coordinate)
        final_sorted = []
        for group in grouped:
            final_sorted.extend(sorted(group, key=lambda d: d["bbox_raw"][0]))

        return final_sorted

    def clean_anchor_name(self, ocr_text: str) -> str:
        """Parses OCR text of a question block to extract a clean question number (e.g. Q1, 1.a)."""
        text = ocr_text.strip()
        if not text:
            return "Q_unknown"

        # Try to find common patterns like "Q1", "Q.1", "Question 1", "1. ", "a) "
        match = re.search(r'\b(Q(uestion)?\.?\s*\d+\s*[a-zA-Z]?)\b', text, re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "").upper()

        match_num = re.match(r'^(\d+[\.\-\s]+[a-zA-Z]?|^[a-zA-Z][\)\.])', text)
        if match_num:
            return match_num.group(1).strip(". - ) (")

        # Fallback to the first 2-3 words or a truncated version of the string
        words = text.split()
        if len(words) <= 3:
            return text
        return " ".join(words[:2])

    def run(
        self,
        pdf_path: str | Path,
        school_name: str,
        academic_year: str,
        class_name: str,
        section: str,
        subject: str,
        assessment_id: str,
        student_id: str,
        marksense_uuid: str,
        question_paper_uuid: str,
        temp_working_dir: str = "outputs/temp"
    ) -> Dict[str, Any]:
        """Runs the end-to-end Marksense processing pipeline."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input file not found: {pdf_path}")

        # Construct S3 prefix path base
        base_prefix = f"{school_name}/{academic_year}/{class_name}/{section}/{subject}/{assessment_id}/students/{student_id}"
        
        # Temp directories
        working_path = Path(temp_working_dir)
        pages_temp_dir = working_path / "pages"
        crops_temp_dir = working_path / "crops"

        # 1. Split PDF to Page Images
        page_files = self.split_pdf_to_pages(pdf_path, pages_temp_dir)

        # 2. Upload Pages to S3 & Build Pages JSON
        pages_metadata: List[List[Any]] = []
        for i, page_file in enumerate(page_files, start=1):
            s3_key = f"{base_prefix}/answer_sheet_pages/P{i}.png"
            page_s3_url = self.storage.upload_file(page_file, s3_key)
            pages_metadata.append([i, page_s3_url])

        # 3. Detect Blocks & Crop Layout Regions Page-by-Page
        crops_temp_dir.mkdir(parents=True, exist_ok=True)
        global_block_counter = 1
        
        blocks_metadata: List[List[Any]] = []
        block_contents_metadata: List[List[Any]] = []
        answer_to_question_lookup: List[Dict[str, Any]] = []
        
        # Mapping logic variables
        current_question_anchor: str | None = None
        active_answers_list: List[Dict[str, Any]] = []

        class_names = self.model.names

        for page_no, page_file in enumerate(page_files, start=1):
            image = cv2.imread(str(page_file))
            if image is None:
                print(f"[PIPELINE] [WARNING] Could not read page image: {page_file}")
                continue
            h, w = image.shape[:2]

            # Run YOLO layout inference
            results = self.model.predict(source=str(page_file), conf=self.conf_threshold)
            result = results[0]
            boxes = result.boxes

            # Build box data dictionary for sorting
            detections_on_page: List[Dict[str, Any]] = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"Class_{cls_id}")

                detections_on_page.append({
                    "bbox_raw": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": class_name,
                    "conf": conf
                })

            # Sort detected blocks in reading order on this page
            sorted_detections = self.sort_blocks_reading_order(detections_on_page)

            for block_idx, det in enumerate(sorted_detections, start=1):
                x1, y1, x2, y2 = det["bbox_raw"]
                class_name = det["class_name"]

                # Ensure coordinates are within image boundaries
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)

                if cx2 <= cx1 or cy2 <= cy1:
                    continue  # Invalid bounding box sizes

                # Crop original page image
                crop_img = image[cy1:cy2, cx1:cx2]

                # Save Crop PNG file locally
                crop_filename = f"P{page_no}_B{block_idx}.png"
                local_crop_path = crops_temp_dir / crop_filename
                cv2.imwrite(str(local_crop_path), crop_img)

                # Upload crop to S3
                s3_key = f"{base_prefix}/answer_blocks/{crop_filename}"
                block_s3_url = self.storage.upload_file(local_crop_path, s3_key)

                # Check if it is a question anchor (question Class ID: 5, sub_question Class ID: 6)
                is_anchor = (det["class_id"] in {5, 6})

                # Run OCR text extraction
                ocr_text = self.ocr.extract_text(
                    local_crop_path,
                    is_question_anchor=is_anchor,
                    block_idx=global_block_counter
                )

                # Extract anchor name
                anchor_name = self.clean_anchor_name(ocr_text) if is_anchor else None

                # Build Bounding Box spec format: {x, y, w, h}
                bbox_spec = {
                    "x": cx1,
                    "y": cy1,
                    "w": cx2 - cx1,
                    "h": cy2 - cy1
                }

                # 3.1 & 3.2 Add to blocks list: [block_number, block_s3_url, block_type, bbox, is_question_anchor, anchor_name]
                blocks_metadata.append([
                    global_block_counter,
                    block_s3_url,
                    class_name,
                    bbox_spec,
                    is_anchor,
                    anchor_name
                ])

                # Add to contents list: [block_number, block_s3_url, block_content]
                block_contents_metadata.append([
                    global_block_counter,
                    block_s3_url,
                    ocr_text
                ])

                # 4. Lookup Mapping (mapping student answer to question paper)
                if is_anchor:
                    # If we had a previous active question anchor, save its answers
                    if current_question_anchor is not None and active_answers_list:
                        answer_to_question_lookup.append({
                            "question_anchor": current_question_anchor,
                            "answer_blocks": active_answers_list
                        })
                    
                    # Start new question mapping group
                    current_question_anchor = anchor_name
                    active_answers_list = []
                else:
                    # Standard student response block (e.g. text/diagram)
                    # Exclude empty blocks to keep the mapping clean
                    if class_name != "block_empty":
                        active_answers_list.append({
                            "block_number": global_block_counter,
                            "block_type": class_name,
                            "block_s3_url": block_s3_url
                        })

                global_block_counter += 1

        # Add the final mapping group
        if current_question_anchor is not None and active_answers_list:
            answer_to_question_lookup.append({
                "question_anchor": current_question_anchor,
                "answer_blocks": active_answers_list
            })

        # Clean up temporary page/crop files
        for p_file in page_files:
            if p_file.exists():
                p_file.unlink()
        for c_file in crops_temp_dir.iterdir():
            if c_file.is_file():
                c_file.unlink()
        
        # Remove empty temp folders
        try:
            pages_temp_dir.rmdir()
            crops_temp_dir.rmdir()
            working_path.rmdir()
        except OSError:
            pass

        # Compile final responses according to manager specs
        pages_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "answer_sheet_pages": pages_metadata
        }

        blocks_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "answer_sheet_blocks": blocks_metadata
        }

        contents_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "answer_sheet_blocks_content": block_contents_metadata
        }

        lookup_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "question_paper_uuid": question_paper_uuid,
            "student_answer_mapping": answer_to_question_lookup
        }

        return {
            "pages": pages_response,
            "blocks": blocks_response,
            "contents": contents_response,
            "lookup": lookup_response
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Marksense PDF Answer Sheet Processing Pipeline")
    parser.add_argument("--input", required=True, help="Path to input answer sheet PDF or page image")
    parser.add_argument("--school-name", default="scholars_home", help="School identifier folder name")
    parser.add_argument("--academic-year", default="2025-2026", help="Academic Year subfolder")
    parser.add_argument("--class", dest="class_name", default="9th", help="Class subfolder")
    parser.add_argument("--section", default="9th-A", help="Section subfolder")
    parser.add_argument("--subject", default="SCIENCE_CHEMISTRY", help="Subject subfolder")
    parser.add_argument("--assessment-id", default="Assessment1", help="Assessment ID subfolder")
    parser.add_argument("--student-id", default="11", help="Student UUID or Roll Number")
    parser.add_argument("--marksense-uuid", default="ms_456", help="Marksense Session ID")
    parser.add_argument("--question-paper-uuid", default="qp_789", help="Question Paper UUID")
    parser.add_argument("--output-dir", default="outputs", help="Directory where JSON results will be written")
    
    args = parser.parse_args()

    pipeline = MarksensePipeline()
    results = pipeline.run(
        pdf_path=args.input,
        school_name=args.school_name,
        academic_year=args.academic_year,
        class_name=args.class_name,
        section=args.section,
        subject=args.subject,
        assessment_id=args.assessment_id,
        student_id=args.student_id,
        marksense_uuid=args.marksense_uuid,
        question_paper_uuid=args.question_paper_uuid
    )

    # Save output JSONs to the output folder
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "res_pages.json", "w", encoding="utf-8") as f:
        json.dump(results["pages"], f, indent=2, ensure_ascii=False)
    
    with open(out_dir / "res_blocks.json", "w", encoding="utf-8") as f:
        json.dump(results["blocks"], f, indent=2, ensure_ascii=False)

    with open(out_dir / "res_contents.json", "w", encoding="utf-8") as f:
        json.dump(results["contents"], f, indent=2, ensure_ascii=False)

    with open(out_dir / "res_lookup.json", "w", encoding="utf-8") as f:
        json.dump(results["lookup"], f, indent=2, ensure_ascii=False)

    print("\n[SUCCESS] Pipeline completed successfully!")
    print(f"JSON outputs saved to: {out_dir.resolve()}")
    print("  - res_pages.json")
    print("  - res_blocks.json")
    print("  - res_contents.json")
    print("  - res_lookup.json")


if __name__ == "__main__":
    main()
