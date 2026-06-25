from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
from ultralytics import YOLO


class BlockDetector:
    """Loads custom fine-tuned YOLO layout models (.pt format) and detects

    semantic layout blocks on document page images, sorting them in reading order.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.weights_path = self._resolve_weights(weights_path)
        self.model = self._load_model()
        self.class_names = self.model.names

    def _resolve_weights(self, configured_path: str | Path | None) -> Path:
        """Finds PyTorch weights checkpoint (.pt) with system-level fallbacks."""
        configured_str = (
            str(configured_path)
            if configured_path
            else os.getenv("YOLO_WEIGHTS_PATH", "models/110-best.pt")
        )
        path = Path(configured_str)
        if path.exists():
            return path

        # Resolve relative fallback paths
        script_dir = Path(__file__).parent.parent.resolve()
        fallbacks = [
            script_dir / "models" / "110-best.pt",
            script_dir / "models" / "best.pt",
            Path("models") / "110-best.pt",
            Path("models") / "best.pt",
        ]
        for fallback_path in fallbacks:
            if fallback_path.exists():
                return fallback_path

        raise FileNotFoundError(
            f"YOLO PyTorch weights (.pt) not found. Checked: {configured_str}"
        )

    def _load_model(self) -> YOLO:
        """Loads and returns the YOLO PyTorch model instance."""
        print(f"[MODEL] Loading YOLO weights: {self.weights_path.resolve()}")
        return YOLO(str(self.weights_path))

    def predict(self, image_path: str | Path) -> List[Dict[str, Any]]:
        """Runs layout inference on the page image.

        Returns a list of dictionaries containing raw bounding boxes,
        confidence scores, class IDs, and semantic names.
        """
        results = self.model.predict(source=str(image_path), conf=self.conf_threshold)
        result = results[0]
        boxes = result.boxes

        detections: List[Dict[str, Any]] = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.class_names.get(cls_id, f"Class_{cls_id}")

            detections.append(
                {
                    "bbox_raw": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": class_name,
                    "conf": conf,
                }
            )
        return detections

    def sort_blocks(
        self, detections: List[Dict[str, Any]], y_threshold: int = 20
    ) -> List[Dict[str, Any]]:
        """Sorts layout block detections based on Y reading coordinates (top-to-bottom)

        and then X coordinates (left-to-right) for blocks sharing a similar Y row.
        """
        if not detections:
            return []

        # Sort primarily by y_min (top), then by x_min (left)
        sorted_by_y = sorted(
            detections, key=lambda d: (d["bbox_raw"][1], d["bbox_raw"][0])
        )

        grouped: List[List[Dict[str, Any]]] = []
        current_group = [sorted_by_y[0]]

        for det in sorted_by_y[1:]:
            # If the top Y distance is within threshold, group them in the same row
            if (
                abs(det["bbox_raw"][1] - current_group[-1]["bbox_raw"][1])
                <= y_threshold
            ):
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
