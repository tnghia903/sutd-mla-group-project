"""
data_prep.py — Whiteboard Dataset Preparation & Annotation Converter
=====================================================================
Module Owner: [Team Member A — Student ID: XXXXXXX]

Purpose:
    1. Generates a synthetic whiteboard dataset with COCO-format annotations
       for structural testing and pipeline validation.
    2. Converts COCO bounding-box annotations → YOLO format:
       <class_id> <x_center> <y_center> <width> <height>  (all normalised to [0,1])

    In production, replace the synthetic generator with a real whiteboard
    dataset (e.g., images captured from meeting rooms, lecture halls).

Whiteboard Category Mapping (COCO → YOLO class IDs):
    1 → 0  Handwriting
    2 → 1  Diagram
    3 → 2  Arrow
    4 → 3  Equation
    5 → 4  Sticky Note

Dependencies:
    pip install requests tqdm pillow numpy
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Number of synthetic whiteboard images to generate
SUBSET_SIZE: int = 30

# Output directories
RAW_DIR     = Path("data/raw")
IMAGES_DIR  = Path("data/images")
LABELS_DIR  = Path("data/labels")

# Whiteboard COCO category_id → YOLO class_id
COCO_TO_YOLO_CLASS: dict[int, int] = {
    1: 0,  # Handwriting
    2: 1,  # Diagram
    3: 2,  # Arrow
    4: 3,  # Equation
    5: 4,  # Sticky Note
}

# Human-readable class names (index = YOLO class_id)
CLASS_NAMES: list[str] = ["Handwriting", "Diagram", "Arrow", "Equation", "Sticky Note"]


# ---------------------------------------------------------------------------
# 2. SYNTHETIC WHITEBOARD GENERATOR
# ---------------------------------------------------------------------------

def _random_colour(base: tuple[int, int, int], variance: int = 30) -> tuple[int, int, int]:
    """Generate a colour near a base with random jitter."""
    return tuple(
        max(0, min(255, c + random.randint(-variance, variance)))
        for c in base
    )


def generate_synthetic_whiteboard(
    img_id: int,
    width: int = 800,
    height: int = 600,
) -> tuple[Image.Image, list[dict]]:
    """
    Generate a single synthetic whiteboard image with COCO-format annotations.

    Simulates a whiteboard photo with:
      - Off-white background (slight perspective/colour variation)
      - Handwriting regions (dark scribble lines)
      - Diagram regions (rectangles, circles)
      - Arrow regions (directional lines)
      - Equation regions (math-like text)
      - Sticky note regions (coloured rectangles)

    Returns:
        (PIL Image, list of COCO annotation dicts)
    """
    # Off-white background to simulate whiteboard surface
    bg_colour = _random_colour((245, 245, 240), variance=10)
    img = Image.new("RGB", (width, height), bg_colour)
    draw = ImageDraw.Draw(img)

    annotations: list[dict] = []
    ann_id_counter = img_id * 100  # unique annotation IDs

    # --- Handwriting regions (2-4 blocks) ---
    for _ in range(random.randint(2, 4)):
        x = random.randint(20, width - 250)
        y = random.randint(20, height - 80)
        w = random.randint(150, 300)
        h = random.randint(30, 70)
        # Draw scribble-like lines
        ink = _random_colour((30, 30, 80), variance=20)
        for line_y in range(y + 8, y + h - 5, 12):
            x_end = x + random.randint(w // 2, w)
            draw.line([(x + 5, line_y), (x_end, line_y + random.randint(-3, 3))],
                      fill=ink, width=2)
        annotations.append({
            "id": ann_id_counter, "image_id": img_id,
            "category_id": 1, "bbox": [x, y, w, h], "area": w * h,
        })
        ann_id_counter += 1

    # --- Diagram regions (1-2 shapes) ---
    for _ in range(random.randint(1, 2)):
        x = random.randint(20, width - 200)
        y = random.randint(20, height - 200)
        w = random.randint(100, 200)
        h = random.randint(80, 180)
        shape_colour = _random_colour((50, 50, 150), variance=30)
        if random.random() > 0.5:
            draw.rectangle([x, y, x + w, y + h], outline=shape_colour, width=3)
        else:
            draw.ellipse([x, y, x + w, y + h], outline=shape_colour, width=3)
        annotations.append({
            "id": ann_id_counter, "image_id": img_id,
            "category_id": 2, "bbox": [x, y, w, h], "area": w * h,
        })
        ann_id_counter += 1

    # --- Arrow regions (1-2) ---
    for _ in range(random.randint(1, 2)):
        x = random.randint(50, width - 150)
        y = random.randint(50, height - 50)
        w = random.randint(80, 150)
        h = random.randint(10, 40)
        arrow_colour = _random_colour((180, 50, 50), variance=20)
        draw.line([(x, y + h // 2), (x + w, y + h // 2)],
                  fill=arrow_colour, width=3)
        # Arrowhead
        draw.polygon([(x + w, y + h // 2), (x + w - 12, y), (x + w - 12, y + h)],
                     fill=arrow_colour)
        annotations.append({
            "id": ann_id_counter, "image_id": img_id,
            "category_id": 3, "bbox": [x, y, w, h], "area": w * h,
        })
        ann_id_counter += 1

    # --- Equation regions (1) ---
    x = random.randint(20, width - 300)
    y = random.randint(20, height - 60)
    w = random.randint(200, 350)
    h = random.randint(35, 55)
    eq_colour = _random_colour((20, 80, 20), variance=15)
    # Simulate equation with symbols
    eq_text = random.choice(["y = mx + b", "E = mc²", "∑ f(x)dx", "∇·F = 0", "a² + b² = c²"])
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((x + 5, y + 5), eq_text, fill=eq_colour, font=font)
    annotations.append({
        "id": ann_id_counter, "image_id": img_id,
        "category_id": 4, "bbox": [x, y, w, h], "area": w * h,
    })
    ann_id_counter += 1

    # --- Sticky note regions (0-2) ---
    for _ in range(random.randint(0, 2)):
        x = random.randint(20, width - 130)
        y = random.randint(20, height - 130)
        w = random.randint(80, 120)
        h = random.randint(80, 120)
        note_colour = random.choice([
            (255, 255, 150), (150, 255, 150), (255, 200, 150), (200, 200, 255)
        ])
        draw.rectangle([x, y, x + w, y + h], fill=note_colour, outline=(180, 180, 100), width=2)
        # Scribble text on the note
        for line_y in range(y + 10, y + h - 10, 14):
            draw.line([(x + 8, line_y), (x + w - 15, line_y)],
                      fill=(80, 80, 80), width=1)
        annotations.append({
            "id": ann_id_counter, "image_id": img_id,
            "category_id": 5, "bbox": [x, y, w, h], "area": w * h,
        })
        ann_id_counter += 1

    return img, annotations


# ---------------------------------------------------------------------------
# 3. COCO → YOLO CONVERSION
# ---------------------------------------------------------------------------

def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """
    Convert a single COCO bbox [x_min, y_min, w, h] (absolute pixels)
    to YOLO format [x_center, y_center, w, h] (normalised 0-1).

    Mathematical formulation:
        x_c = (x_min + w/2) / W
        y_c = (y_min + h/2) / H
        w_n = w / W
        h_n = h / H
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    w_norm   = w / img_w
    h_norm   = h / img_h
    return (x_center, y_center, w_norm, h_norm)


def convert_annotations(
    coco_json: dict[str, Any],
    image_ids: set[int],
) -> dict[int, list[str]]:
    """
    Build a mapping  image_id → list[YOLO-formatted annotation strings].

    Each annotation string: "<class_id> <x_c> <y_c> <w_n> <h_n>\n"
    """
    # Build fast lookup: image_id → (width, height)
    img_info: dict[int, tuple[int, int]] = {}
    for img in coco_json["images"]:
        if img["id"] in image_ids:
            img_info[img["id"]] = (img["width"], img["height"])

    labels: dict[int, list[str]] = {iid: [] for iid in image_ids}

    for ann in coco_json["annotations"]:
        iid = ann["image_id"]
        if iid not in image_ids:
            continue

        coco_cat = ann["category_id"]
        yolo_cls = COCO_TO_YOLO_CLASS.get(coco_cat)
        if yolo_cls is None:
            continue  # skip unknown categories

        w, h = img_info[iid]
        xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], w, h)
        labels[iid].append(f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    return labels


# ---------------------------------------------------------------------------
# 4. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def prepare_dataset(subset_size: int = SUBSET_SIZE) -> Path:
    """
    End-to-end dataset preparation:
        1. Generate synthetic whiteboard images with annotations.
        2. Save images and COCO-format JSON.
        3. Convert annotations from COCO → YOLO and write .txt label files.
        4. Generate a YOLO dataset YAML config.

    Returns:
        Path to the generated `dataset.yaml`.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Generating synthetic whiteboard dataset …")

    coco_json: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Handwriting"},
            {"id": 2, "name": "Diagram"},
            {"id": 3, "name": "Arrow"},
            {"id": 4, "name": "Equation"},
            {"id": 5, "name": "Sticky Note"},
        ],
    }

    img_width, img_height = 800, 600

    for i in range(subset_size):
        img_id = i + 1
        filename = f"whiteboard_{img_id:04d}.png"

        img, annotations = generate_synthetic_whiteboard(
            img_id, width=img_width, height=img_height
        )
        img.save(IMAGES_DIR / filename)

        coco_json["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": img_width,
            "height": img_height,
        })
        coco_json["annotations"].extend(annotations)

    print(f"[INFO] Generated {subset_size} synthetic whiteboard images.")

    # Save COCO JSON
    annot_path = RAW_DIR / "whiteboard_annotations.json"
    with open(annot_path, "w") as f:
        json.dump(coco_json, f, indent=2)
    print(f"[INFO] COCO annotations saved to {annot_path}")

    # --- Convert & write labels ---
    image_ids = {img["id"] for img in coco_json["images"]}
    labels = convert_annotations(coco_json, image_ids)
    for img_meta in coco_json["images"]:
        iid  = img_meta["id"]
        stem = Path(img_meta["file_name"]).stem
        label_path = LABELS_DIR / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.writelines(labels.get(iid, []))

    print(f"[INFO] Wrote {len(labels)} label files to {LABELS_DIR}/")

    # --- dataset.yaml ---
    yaml_path = Path("data/dataset.yaml")
    yaml_content = (
        f"path: {IMAGES_DIR.resolve().parent}\n"
        f"train: images\n"
        f"val: images\n\n"
        f"names:\n"
    )
    for idx, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path.write_text(yaml_content)
    print(f"[INFO] Dataset YAML written to {yaml_path}")

    return yaml_path


# ---------------------------------------------------------------------------
# 5. STANDALONE EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    yaml_out = prepare_dataset()
    print(f"\n✓ Data preparation complete.  Config → {yaml_out}")
"""
Module Owner: [Team Member A — Student ID: XXXXXXX]
"""
