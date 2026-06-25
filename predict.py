from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone YOLO Layout Inference script"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to test image file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/110-best.pt",
        help="Path to weights file (default: models/110-best.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/visual_output.jpg",
        help="Path to save annotated output image",
    )
    args = parser.parse_args()

    # Verify input image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[x] Error: Image path does not exist: {img_path}")
        sys.exit(1)

    # Verify weights exist with flexible fallback order
    weights_path = Path(args.weights)
    if not weights_path.exists():
        fallback_names = ["110-best.pt", "best.pt"]
        found = False
        script_dir = Path(__file__).parent.resolve()
        for fname in fallback_names:
            paths_to_check = [
                script_dir / "models" / fname,
                Path("models") / fname,
                Path(fname),
            ]
            for p in paths_to_check:
                if p.exists():
                    weights_path = p
                    found = True
                    break
            if found:
                break
        if not found:
            project_weights = Path("../models/custom_best.pt")
            if project_weights.exists():
                weights_path = project_weights
            else:
                print(f"[x] Error: Weights file not found: {weights_path}")
                sys.exit(1)

    print(f"[*] Loading model via Ultralytics from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[*] Running inference on: {img_path}")
    results = model.predict(source=str(img_path), conf=args.conf)

    # Print results summary
    result = results[0]
    boxes = result.boxes
    print(f"[+] Found {len(boxes)} layout blocks.")

    # Class mappings
    class_names = model.names

    # Class colors for visual feedback (BGR format: Blue, Green, Red)
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

    # Load original image for custom styled rendering
    img = cv2.imread(str(img_path))
    if img is None:
        print("[x] Error: OpenCV failed to read the image.")
        sys.exit(1)

    overlay = img.copy()

    for idx, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names.get(cls_id, f"Class {cls_id}")

        print(
            f"    [{idx}] Class: {class_name} | Conf: {conf:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]"
        )

        color = colors.get(cls_id, (0, 255, 0))  # Default green

        # Draw transparent bounding box overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Solid border

        # Draw label tag
        label = f"#{idx} {class_name} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        # Draw label text
        cv2.putText(
            img,
            label,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Apply transparency to overlay (35% overlay, 65% original)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Save output image
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(
        f"[+] Inference complete! Annotated image saved to: {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()
