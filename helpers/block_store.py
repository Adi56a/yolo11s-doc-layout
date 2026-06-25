from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np


class BlockStore:
    """Manages saving cropped block images, compiling structured JSON responses,

    and writing final outputs to the filesystem.
    """

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.given_images_dir = self.output_dir / "given_images"
        self.detected_images_dir = self.output_dir / "detected_images"
        self.cropped_blocks_dir = self.output_dir / "cropped_blocks"

        self.given_images_dir.mkdir(parents=True, exist_ok=True)
        self.detected_images_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_blocks_dir.mkdir(parents=True, exist_ok=True)

    def save_crop(
        self, image: np.ndarray, bbox: List[int], crop_path: Path
    ) -> bool:
        """Crops the bounding box area from the page image and saves it to crop_path."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Bound check coordinates
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w, x2)
        cy2 = min(h, y2)

        if cx2 <= cx1 or cy2 <= cy1:
            return False

        crop_img = image[cy1:cy2, cx1:cx2]
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(crop_path), crop_img)

    def save_detected_image(
        self, image: np.ndarray, detections: List[Dict[str, Any]], page_no: int
    ) -> Path:
        """Draws annotated bounding boxes and labels on the image and saves it to detected_images/."""
        annotated_img = image.copy()
        overlay = annotated_img.copy()

        colors = {
            0: (241, 102, 99),  # block_text (Indigo - BGR)
            1: (0, 234, 255),  # block_diagram (Yellow - BGR)
            2: (129, 185, 16),  # block_table (Green - BGR)
            3: (11, 158, 245),  # block_rough (Amber - BGR)
            4: (128, 114, 107),  # block_empty (Gray - BGR)
            5: (153, 72, 236),  # question (Pink - BGR)
            6: (246, 92, 139),  # sub_question (Purple - BGR)
            7: (212, 182, 6),  # block_graph (Cyan - BGR)
            8: (68, 68, 239),  # block_map (Red - BGR)
        }

        for idx, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["bbox_raw"]
            cls_id = det["class_id"]
            class_name = det["class_name"]
            conf = det["conf"]

            color = colors.get(cls_id, (0, 255, 0))

            # Draw overlay and border
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

            label = f"#{idx} {class_name} {conf:.0%}"
            (w_t, h_t), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(
                annotated_img, (x1, y1 - h_t - 5), (x1 + w_t, y1), color, -1
            )
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Alpha blend (35% overlay, 65% original)
        cv2.addWeighted(overlay, 0.35, annotated_img, 0.65, 0, annotated_img)

        dest_path = (
            self.detected_images_dir / f"detected_page_{page_no:03d}.png"
        )
        cv2.imwrite(str(dest_path), annotated_img)
        return dest_path

    def compile_and_save(
        self,
        marksense_uuid: str,
        student_id: str,
        question_paper_uuid: str,
        pages_metadata: List[List[Any]],
        blocks_metadata: List[List[Any]],
        block_contents_metadata: List[List[Any]],
        answer_to_question_lookup: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compiles output payloads and writes standard res_*.json files into output_dir."""
        pages_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "answer_sheet_pages": pages_metadata,
        }

        blocks_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "answer_sheet_blocks": blocks_metadata,
        }

        contents_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "answer_sheet_blocks_content": block_contents_metadata,
        }

        lookup_response = {
            "marksense_uuid": marksense_uuid,
            "student_uuid": student_id,
            "student_rollno": student_id,
            "question_paper_uuid": question_paper_uuid,
            "student_answer_mapping": answer_to_question_lookup,
        }

        # Write output JSON files
        with open(
            self.output_dir / "res_pages.json", "w", encoding="utf-8"
        ) as f:
            json.dump(pages_response, f, indent=2, ensure_ascii=False)

        with open(
            self.output_dir / "res_blocks.json", "w", encoding="utf-8"
        ) as f:
            json.dump(blocks_response, f, indent=2, ensure_ascii=False)

        with open(
            self.output_dir / "res_contents.json", "w", encoding="utf-8"
        ) as f:
            json.dump(contents_response, f, indent=2, ensure_ascii=False)

        with open(
            self.output_dir / "res_lookup.json", "w", encoding="utf-8"
        ) as f:
            json.dump(lookup_response, f, indent=2, ensure_ascii=False)

        print("\n[SUCCESS] Pipeline completed successfully!")
        print(f"JSON outputs saved to: {self.output_dir.resolve()}")
        print("  - res_pages.json")
        print("  - res_blocks.json")
        print("  - res_contents.json")
        print("  - res_lookup.json")

        return {
            "pages": pages_response,
            "blocks": blocks_response,
            "contents": contents_response,
            "lookup": lookup_response,
        }
