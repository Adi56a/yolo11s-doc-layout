from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

# Ensure helpers are discoverable
sys.path.append(str(Path(__file__).parent.resolve()))

from helpers.ans_sheet_split import AnswerSheetSplitter
from helpers.block_detection import BlockDetector
from helpers.block_ocr import BlockOCR
from helpers.block_store import BlockStore
from helpers.normalize_ques_anchor import QuestionAnchorNormalizer
from helpers.s3_helper import S3StorageHelper


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
                    val = val.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val
    else:
        print("[ENV] No .env file found. Using system environment variables.")


class MarksensePipeline:
    """Production-grade pipeline for the Marksense document layout analysis.

    Coordinates PDF page rendering, YOLO block detection, crop generation,
    cloud/local storage uploads, OCR text extraction, and question-to-answer mapping.
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
            local_storage_dir=local_dir,
        )

        # Initialize OCR helper
        ocr_engine = os.getenv("OCR_ENGINE", "mock")
        self.ocr = BlockOCR(engine_name=ocr_engine)

        # Initialize YOLO model weights and detector
        configured_weights = os.getenv("YOLO_WEIGHTS_PATH", "models/110-best.pt")
        conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))

        self.detector = BlockDetector(
            weights_path=configured_weights, conf_threshold=conf_threshold
        )

        # Initialize core utilities
        self.splitter = AnswerSheetSplitter(dpi=200)
        self.normalizer = QuestionAnchorNormalizer()

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
        output_dir: str = "outputs",
        temp_working_dir: str = "outputs/temp",
    ) -> Dict[str, Any]:
        """Runs the end-to-end Marksense processing pipeline."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input file not found: {pdf_path}")

        # Initialize output block store
        block_store = BlockStore(output_dir=output_dir)

        # Construct S3 prefix path base
        base_prefix = f"{school_name}/{academic_year}/{class_name}/{section}/{subject}/{assessment_id}/students/{student_id}"

        # 1. Split PDF to Page Images (Directly saved to outputs/given_images/)
        page_files = self.splitter.split(pdf_path, block_store.given_images_dir)

        # 2. Upload Pages to S3 & Build Pages JSON metadata
        pages_metadata: List[List[Any]] = []
        for i, page_file in enumerate(page_files, start=1):
            s3_key = f"{base_prefix}/answer_sheet_pages/P{i}.png"
            page_s3_url = self.storage.upload_file(page_file, s3_key)
            pages_metadata.append([i, page_s3_url])

        # 3. Detect Blocks & Crop Layout Regions Page-by-Page
        global_block_counter = 1

        blocks_metadata: List[List[Any]] = []
        block_contents_metadata: List[List[Any]] = []
        answer_to_question_lookup: List[Dict[str, Any]] = []

        # Mapping logic variables
        current_question_anchor: str | None = None
        active_answers_list: List[Dict[str, Any]] = []

        for page_no, page_file in enumerate(page_files, start=1):
            image = cv2.imread(str(page_file))
            if image is None:
                print(
                    f"[PIPELINE] [WARNING] Could not read page image: {page_file}"
                )
                continue

            # Run YOLO layout inference
            detections = self.detector.predict(page_file)

            # Save annotated page overlay image (Directly to outputs/detected_images/)
            block_store.save_detected_image(image, detections, page_no)

            # Sort detected blocks in reading order on this page
            sorted_detections = self.detector.sort_blocks(detections)

            for block_idx, det in enumerate(sorted_detections, start=1):
                x1, y1, x2, y2 = det["bbox_raw"]
                class_name = det["class_name"]

                # Crop filename & path
                crop_filename = f"P{page_no}_B{block_idx}.png"
                local_crop_path = block_store.cropped_blocks_dir / crop_filename

                # Save Crop PNG file locally
                success = block_store.save_crop(
                    image, [x1, y1, x2, y2], local_crop_path
                )
                if not success:
                    continue

                # Upload crop to S3
                s3_key = f"{base_prefix}/answer_blocks/{crop_filename}"
                block_s3_url = self.storage.upload_file(local_crop_path, s3_key)

                # Check if it is a question anchor (question Class ID: 5, sub_question Class ID: 6)
                is_anchor = det["class_id"] in {5, 6}

                # Run OCR text extraction
                ocr_text = self.ocr.extract_text(
                    local_crop_path,
                    is_question_anchor=is_anchor,
                    block_idx=global_block_counter,
                )

                # Extract anchor name
                anchor_name = (
                    self.normalizer.normalize(ocr_text) if is_anchor else None
                )

                # Bounding Box spec format: {x, y, w, h}
                bbox_spec = {
                    "x": max(0, x1),
                    "y": max(0, y1),
                    "w": min(image.shape[1], x2) - max(0, x1),
                    "h": min(image.shape[0], y2) - max(0, y1),
                }

                # Add to blocks metadata
                blocks_metadata.append(
                    [
                        global_block_counter,
                        block_s3_url,
                        class_name,
                        bbox_spec,
                        is_anchor,
                        anchor_name,
                    ]
                )

                # Add to contents metadata
                block_contents_metadata.append(
                    [global_block_counter, block_s3_url, ocr_text]
                )

                # Lookup Mapping
                if is_anchor:
                    # If we had a previous active question anchor, save its answers
                    if current_question_anchor is not None and active_answers_list:
                        answer_to_question_lookup.append(
                            {
                                "question_anchor": current_question_anchor,
                                "answer_blocks": active_answers_list,
                            }
                        )

                    current_question_anchor = anchor_name
                    active_answers_list = []
                else:
                    # Standard student response block
                    if class_name != "block_empty":
                        active_answers_list.append(
                            {
                                "block_number": global_block_counter,
                                "block_type": class_name,
                                "block_s3_url": block_s3_url,
                            }
                        )

                global_block_counter += 1

        # Add the final mapping group
        if current_question_anchor is not None and active_answers_list:
            answer_to_question_lookup.append(
                {
                    "question_anchor": current_question_anchor,
                    "answer_blocks": active_answers_list,
                }
            )

        # Compile and write response JSONs
        return block_store.compile_and_save(
            marksense_uuid=marksense_uuid,
            student_id=student_id,
            question_paper_uuid=question_paper_uuid,
            pages_metadata=pages_metadata,
            blocks_metadata=blocks_metadata,
            block_contents_metadata=block_contents_metadata,
            answer_to_question_lookup=answer_to_question_lookup,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-End Marksense PDF Answer Sheet Processing Pipeline"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input answer sheet PDF or page image"
    )
    parser.add_argument(
        "--school-name", default="scholars_home", help="School identifier folder name"
    )
    parser.add_argument(
        "--academic-year", default="2025-2026", help="Academic Year subfolder"
    )
    parser.add_argument(
        "--class", dest="class_name", default="9th", help="Class subfolder"
    )
    parser.add_argument("--section", default="9th-A", help="Section subfolder")
    parser.add_argument("--subject", default="SCIENCE_CHEMISTRY", help="Subject subfolder")
    parser.add_argument(
        "--assessment-id", default="Assessment1", help="Assessment ID subfolder"
    )
    parser.add_argument(
        "--student-id", default="11", help="Student UUID or Roll Number"
    )
    parser.add_argument(
        "--marksense-uuid", default="ms_456", help="Marksense Session ID"
    )
    parser.add_argument(
        "--question-paper-uuid", default="qp_789", help="Question Paper UUID"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where JSON results will be written",
    )

    args = parser.parse_args()

    pipeline = MarksensePipeline()
    pipeline.run(
        pdf_path=args.input,
        school_name=args.school_name,
        academic_year=args.academic_year,
        class_name=args.class_name,
        section=args.section,
        subject=args.subject,
        assessment_id=args.assessment_id,
        student_id=args.student_id,
        marksense_uuid=args.marksense_uuid,
        question_paper_uuid=args.question_paper_uuid,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
