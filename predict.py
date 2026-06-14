import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    parser = argparse.ArgumentParser(description="Standalone YOLO Layout Inference script")
    parser.add_argument("--image", type=str, required=True, help="Path to test image file")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to weights file (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save annotated output image")
    args = parser.parse_args()

    # Verify input image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[✗] Error: Image path does not exist: {img_path}")
        sys.exit(1)

    # Verify weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Fall back to parent models folder if running from project root
        project_weights = Path("models/custom_best.pt")
        if project_weights.exists():
            weights_path = project_weights
        else:
            print(f"[✗] Error: Weights file not found: {weights_path}")
            sys.exit(1)

    print(f"[*] Loading model weights from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[*] Running inference on: {img_path}")
    results = model.predict(source=str(img_path), conf=args.conf)

    # Print results summary
    result = results[0]
    boxes = result.boxes
    print(f"[+] Found {len(boxes)} layout blocks.")

    # Class mappings
    class_names = model.names
    
    # Class colors for visual feedback: Diagram is yellow (0, 255, 255), Text is indigo (99, 102, 241)
    # OpenCV uses BGR: Diagram = (0, 255, 255), Text = (241, 102, 99)
    colors = {
        0: (0, 255, 255),  # Diagram (Yellow)
        1: (241, 102, 99)  # Text (Indigo/Pinkish-blue)
    }

    # Load original image for custom styled rendering
    img = cv2.imread(str(img_path))
    if img is None:
        print("[✗] Error: OpenCV failed to read the image.")
        sys.exit(1)

    overlay = img.copy()

    for idx, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names.get(cls_id, f"Class {cls_id}")

        print(f"    [{idx}] Class: {class_name} | Conf: {conf:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")

        color = colors.get(cls_id, (0, 255, 0)) # Default green

        # Draw transparent bounding box overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1) # Filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)     # Solid border

        # Draw label tag
        label = f"#{idx} {class_name} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Apply transparency to overlay (35% overlay, 65% original)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Save output image
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), img)
    print(f"[✓] Inference complete! Annotated image saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
