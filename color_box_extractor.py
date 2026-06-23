from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
    FITZ_IMPORT_ERROR = None
except Exception as exc:
    fitz = None
    FITZ_IMPORT_ERROR = exc


@dataclass
class Detection:
    block_id: str
    page_no: int
    class_name: str
    bbox: List[int]
    page_path: str
    source: str = "color_segmentation"
    semantic_name: Optional[str] = None
    crop_path: Optional[str] = None


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def convert_pdf_to_images(pdf_path: str, pages_dir: str, dpi: int = 200) -> List[str]:
    if fitz is None:
        raise RuntimeError(
            f"Could not import PyMuPDF (fitz) due to: {FITZ_IMPORT_ERROR}. "
            "Please install it using 'pip install pymupdf' to process PDF files."
        )
    ensure_dir(pages_dir)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page_paths: List[str] = []

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(pages_dir, f"page_{i:03d}.png")
        pix.save(out_path)
        page_paths.append(out_path)

    return page_paths


def load_input_pages(input_path: str, pages_dir: str, dpi: int = 200) -> List[str]:
    suffix = Path(input_path).suffix.lower()
    ensure_dir(pages_dir)

    if suffix == ".pdf":
        return convert_pdf_to_images(input_path, pages_dir, dpi=dpi)

    if is_image_file(input_path):
        image = Image.open(input_path).convert("RGB")
        out_path = os.path.join(pages_dir, "page_001.png")
        image.save(out_path)
        return [out_path]

    raise ValueError(f"Unsupported input type: {suffix}. Use PDF or image file.")


def sort_contours_reading_order(boxes: List[Tuple[int, int, int, int]], y_threshold: int = 15) -> List[Tuple[int, int, int, int]]:
    """
    Sorts a list of bounding boxes (x, y, w, h) according to reading order.
    It groups boxes that share a similar Y coordinate (difference <= y_threshold)
    and sorts them left-to-right (by X coordinate).
    The groups are sorted top-to-bottom (by Y coordinate).
    """
    if not boxes:
        return []
    
    # Sort primarily by y, then by x
    sorted_by_y = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    grouped: List[List[Tuple[int, int, int, int]]] = []
    current_group = [sorted_by_y[0]]
    
    for box in sorted_by_y[1:]:
        # If the Y distance to the last element of current group is within threshold, group them
        if abs(box[1] - current_group[-1][1]) <= y_threshold:
            current_group.append(box)
        else:
            grouped.append(current_group)
            current_group = [box]
    grouped.append(current_group)
    
    # Sort each group by X coordinate (left-to-right)
    final_sorted = []
    for g in grouped:
        final_sorted.extend(sorted(g, key=lambda b: b[0]))
        
    return final_sorted


def get_relational_mask(image: np.ndarray, c1: int, c2: int, c3: int) -> Optional[np.ndarray]:
    """
    Computes an adaptive relational mask based on channel dominance.
    This is extremely robust against JPEG compression noise, resizing,
    anti-aliased thin borders, and transparent colored fill highlights.
    """
    b_ch = image[:, :, 0].astype(int)
    g_ch = image[:, :, 1].astype(int)
    r_ch = image[:, :, 2].astype(int)
    
    # 1. Blue channel dominant (B >> G, B >> R, G ≈ R)
    if (c1 > c2 + 30 and c1 > c3 + 30 and abs(c2 - c3) < 30) or (c3 > c2 + 30 and c3 > c1 + 30 and abs(c2 - c1) < 30):
        mask = (b_ch - g_ch > 15) & (b_ch - r_ch > 15) & (np.abs(g_ch - r_ch) < 35)
        return (mask.astype(np.uint8) * 255)
        
    # 2. Green channel dominant (G >> B, G >> R, B ≈ R)
    if (c2 > c1 + 30 and c2 > c3 + 30 and abs(c1 - c3) < 30):
        mask = (g_ch - b_ch > 15) & (g_ch - r_ch > 15) & (np.abs(b_ch - r_ch) < 35)
        return (mask.astype(np.uint8) * 255)
        
    # 3. Red channel dominant (R >> B, R >> G, B ≈ G)
    if (c3 > c1 + 30 and c3 > c2 + 30 and abs(c1 - c2) < 30) or (c1 > c2 + 30 and c1 > c3 + 30 and abs(c2 - c3) < 30):
        mask = (r_ch - b_ch > 15) & (r_ch - g_ch > 15) & (np.abs(b_ch - g_ch) < 35)
        return (mask.astype(np.uint8) * 255)
        
    return None


def extract_blocks_by_color(
    page_paths: List[str],
    target_color_str: str,
    output_dir: str,
    exclude_border: int = 2,
    min_block_size: int = 20,
    tolerance: int = 10,
    no_question: bool = False,
) -> List[Detection]:
    """
    Detects rectangular boxes of a specific color, crops them,
    and saves them as P<PageNo>B<BlockNo>.png inside output_dir/block_crops.
    """
    # Parse color
    try:
        parts = [int(c.strip()) for c in target_color_str.split(",")]
        if len(parts) != 3:
            raise ValueError()
        c1, c2, c3 = parts
    except Exception:
        print(f"[ERROR] Invalid target-color format: '{target_color_str}'. Must be 'R,G,B' or 'B,G,R' like '99,102,241'.")
        sys.exit(1)

    block_crops_dir = os.path.join(output_dir, "block_crops")
    ensure_dir(block_crops_dir)

    detections: List[Detection] = []

    # Target colors (supporting both RGB and BGR representation of the input color)
    color_configs = [
        (c1, c2, c3), # BGR config
        (c3, c2, c1)  # RGB config (as BGR)
    ]

    print(f"[COLOR] Detecting boxes matching target color: {target_color_str} with tolerance +/- {tolerance}")
    
    for page_no, page_path in enumerate(page_paths, start=1):
        image = cv2.imread(page_path)
        if image is None:
            print(f"[WARN] Could not read page image: {page_path}")
            continue

        if no_question:
            # Detect pink and red pixels and replace them with white (255, 255, 255)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Hue range [150, 180] for pink and red, [0, 10] for red
            lower_pink_red1 = np.array([150, 30, 50])
            upper_pink_red1 = np.array([180, 255, 255])
            lower_pink_red2 = np.array([0, 30, 50])
            upper_pink_red2 = np.array([10, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_pink_red1, upper_pink_red1)
            mask2 = cv2.inRange(hsv, lower_pink_red2, upper_pink_red2)
            pink_red_mask = cv2.bitwise_or(mask1, mask2)
            
            image = image.copy()
            image[pink_red_mask > 0] = [255, 255, 255]

        h, w = image.shape[:2]
        
        # Combine masks for both color configurations
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use relational mask for robust detection under compression/anti-aliasing
        rel_mask = get_relational_mask(image, c1, c2, c3)
        if rel_mask is not None:
            combined_mask = rel_mask
        else:
            # Fallback to standard color range thresholding
            for b, g, r in color_configs:
                lower = np.array([max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance)], dtype=np.uint8)
                upper = np.array([min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance)], dtype=np.uint8)
                mask = cv2.inRange(image, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw >= min_block_size and bh >= min_block_size:
                raw_boxes.append((x, y, bw, bh))

        # Sort boxes according to reading order
        sorted_boxes = sort_contours_reading_order(raw_boxes)

        for block_idx, (x, y, bw, bh) in enumerate(sorted_boxes, start=1):
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            
            # Apply exclude_border to crop inside the colored border frame
            cx1 = max(0, x1 + exclude_border)
            cy1 = max(0, y1 + exclude_border)
            cx2 = min(w, x2 - exclude_border)
            cy2 = min(h, y2 - exclude_border)
            
            if cx2 > cx1 and cy2 > cy1:
                crop = image[cy1:cy2, cx1:cx2]
            else:
                crop = image[y1:y2, x1:x2]

            block_name = f"P{page_no}B{block_idx}"
            crop_filename = f"{block_name}.png"
            crop_path = os.path.join(block_crops_dir, crop_filename)
            
            cv2.imwrite(crop_path, crop)
            print(f"      Saved crop: {block_name} -> {crop_path}")

            detections.append(
                Detection(
                    block_id=f"p{page_no:03d}_b{block_idx:04d}",
                    page_no=page_no,
                    class_name="block_text",
                    bbox=[x1, y1, x2, y2],
                    page_path=page_path,
                    semantic_name=block_name,
                    crop_path=crop_path
                )
            )

    print(f"[COLOR] Completed color-based block extraction. Saved {len(detections)} blocks to {block_crops_dir}")
    return detections


def save_overlay_images(detections: List[Detection], output_dir: str) -> None:
    overlay_dir = os.path.join(output_dir, "detection_overlays")
    ensure_dir(overlay_dir)

    page_to_dets: Dict[int, List[Detection]] = {}
    for det in detections:
        page_to_dets.setdefault(det.page_no, []).append(det)

    for page_no, dets in sorted(page_to_dets.items()):
        image = cv2.imread(dets[0].page_path)
        if image is None:
            continue

        for det in dets:
            x1, y1, x2, y2 = det.bbox
            label = det.semantic_name or f"Block {det.block_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

        out_path = os.path.join(overlay_dir, f"page_{page_no:03d}_overlay.png")
        cv2.imwrite(out_path, image)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract document blocks from annotated YOLO output images by targeting box boundary colors.")
    parser.add_argument("--input", required=True, help="Input answer sheet PDF or image folder/file.")
    parser.add_argument("--output", default="outputs/run_color", help="Output directory.")
    parser.add_argument("--target-color", default="99,102,241", help="BGR/RGB boundary box color code (comma-separated, e.g. '99,102,241').")
    parser.add_argument("--exclude-border", type=int, default=2, help="Border exclusion width (in pixels) to shrink cropping boxes inside colored frames.")
    parser.add_argument("--min-block-size", type=int, default=20, help="Minimum width and height for a detected colored box block.")
    parser.add_argument("--color-tolerance", type=int, default=10, help="Tolerance range for matching target color values.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rendering PDF input pages.")
    parser.add_argument("--no-question", action="store_true", help="Only consider text block color; avoid or delete pixels in pink or red regions.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        return 1

    ensure_dir(args.output)
    pages_dir = os.path.join(args.output, "pages")

    print("[1/4] Loading pages...")
    page_paths = load_input_pages(args.input, pages_dir, dpi=args.dpi)
    print(f"      pages: {len(page_paths)}")

    print("[2/4] Running color-based block extraction...")
    detections = extract_blocks_by_color(
        page_paths,
        args.target_color,
        args.output,
        exclude_border=args.exclude_border,
        min_block_size=args.min_block_size,
        tolerance=args.color_tolerance,
        no_question=args.no_question
    )
    print(f"      detections found: {len(detections)}")

    print("[3/4] Saving overlays...")
    save_overlay_images(detections, args.output)

    print("[4/4] Exporting detections metadata...")
    write_json(os.path.join(args.output, "detections.json"), [asdict(d) for d in detections])

    print("\n[DONE]")
    print(f"Output folder: {args.output}")
    print(f"Block crops: {os.path.join(args.output, 'block_crops')}")
    print(f"Detections JSON: {os.path.join(args.output, 'detections.json')}")
    print(f"Detection overlays: {os.path.join(args.output, 'detection_overlays')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
