"""
generate_composite.py -- Composite dataset: MathWriting equations on Roboflow whiteboards
=========================================================================================
Creates a 2-class YOLO dataset by pasting MathWriting equation glyphs into
Roboflow whiteboard scene images, plus synthetic hard negatives.

Output classes:
    0: equation   (tight bbox around pasted glyph)
    1: whiteboard (from Roboflow annotation)

Usage:
    # Step 1 -- download Roboflow data (one-time):
    python tests/fetch_real_dataset.py --source roboflow --api-key YOUR_KEY

    # Step 2 -- generate composite dataset:
    python tests/generate_composite.py
    python tests/generate_composite.py --roboflow-dir data/raw/roboflow_whiteboard \
        --mathwriting-dir data/raw/mathwriting \
        --synthetic-negatives 40 \
        --output-dir data/raw/composite
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EQUATION_CLASS_ID   = 0
WHITEBOARD_CLASS_ID = 1
CLASS_NAMES         = ["equation", "whiteboard"]

IMG_EXTS   = (".jpg", ".jpeg", ".png")
RANDOM_SEED = 42

# Equation paste parameters
MIN_EQ_PER_BOARD  = 1
MAX_EQ_PER_BOARD  = 4
EQ_WIDTH_MIN_FRAC = 0.15   # equation width as fraction of whiteboard width
EQ_WIDTH_MAX_FRAC = 0.40
EQ_HEIGHT_MAX_FRAC = 0.30  # max equation height as fraction of whiteboard height
INK_INTENSITY_MIN  = 10    # dark ink shade range
INK_INTENSITY_MAX  = 80


# ---------------------------------------------------------------------------
# 1. Roboflow label parsing
# ---------------------------------------------------------------------------

def _parse_roboflow_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    """
    Parse a Roboflow YOLO label file and return whiteboard bboxes
    as (x_center, y_center, width, height) in normalized [0,1] coords.

    Roboflow whiteboard-detect typically has a single class (index 0 = whiteboard).
    """
    bboxes = []
    if not label_path.exists():
        return bboxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # Accept any class -- Roboflow may have class 0 = whiteboard
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        bboxes.append((cx, cy, w, h))
    return bboxes


def _yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized bbox to absolute pixel (x1, y1, x2, y2)."""
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def _xyxy_to_yolo(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Convert absolute pixel bbox to YOLO normalized (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# 2. Equation pasting
# ---------------------------------------------------------------------------

def _load_equation_pool(mathwriting_dir: Path) -> list[Path]:
    """Collect all MathWriting equation image paths from train + valid."""
    pool = []
    for split in ("train", "valid"):
        img_dir = mathwriting_dir / split / "images"
        if img_dir.exists():
            pool.extend(
                p for p in sorted(img_dir.iterdir())
                if p.suffix.lower() in IMG_EXTS
            )
    if not pool:
        print("[ERROR] No MathWriting images found.")
        sys.exit(1)
    print(f"[PHASE 1] [STEP 2] Loaded {len(pool)} MathWriting equations.")
    return pool


def _paste_equation(
    scene: np.ndarray,
    eq_path: Path,
    wb_x1: int, wb_y1: int, wb_x2: int, wb_y2: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    """
    Paste a single equation glyph into the whiteboard region of a scene image.

    Returns (eq_x1, eq_y1, eq_x2, eq_y2) in absolute pixels, or None if
    the equation could not be placed (whiteboard too small).
    """
    eq_img = cv2.imread(str(eq_path))
    if eq_img is None:
        return None

    eq_h_orig, eq_w_orig = eq_img.shape[:2]
    wb_w = wb_x2 - wb_x1
    wb_h = wb_y2 - wb_y1

    if wb_w < 30 or wb_h < 30:
        return None

    # Scale equation: 15-40% of whiteboard width
    target_w = int(wb_w * rng.uniform(EQ_WIDTH_MIN_FRAC, EQ_WIDTH_MAX_FRAC))
    scale = target_w / max(eq_w_orig, 1)
    target_h = int(eq_h_orig * scale)

    # Clamp height
    max_h = int(wb_h * EQ_HEIGHT_MAX_FRAC)
    if target_h > max_h and max_h > 10:
        scale = max_h / max(eq_h_orig, 1)
        target_w = int(eq_w_orig * scale)
        target_h = max_h

    if target_w < 10 or target_h < 10:
        return None

    eq_resized = cv2.resize(eq_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Random position inside whiteboard, with 5% margin
    margin_x = max(1, int(wb_w * 0.05))
    margin_y = max(1, int(wb_h * 0.05))
    max_x = wb_x1 + wb_w - target_w - margin_x
    max_y = wb_y1 + wb_h - target_h - margin_y
    start_x = wb_x1 + margin_x
    start_y = wb_y1 + margin_y

    if max_x <= start_x or max_y <= start_y:
        return None

    paste_x = rng.randint(start_x, max_x)
    paste_y = rng.randint(start_y, max_y)

    # Threshold-based masking: MathWriting has white bg, black ink
    gray = cv2.cvtColor(eq_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Randomize ink color for realism
    ink_val = rng.randint(INK_INTENSITY_MIN, INK_INTENSITY_MAX)
    colored_eq = np.full_like(eq_resized, ink_val)

    # Optional slight Gaussian blur for realism
    if rng.random() > 0.5:
        colored_eq = cv2.GaussianBlur(colored_eq, (3, 3), 0.7)

    # Blend into scene
    mask_3ch = cv2.merge([mask, mask, mask])
    roi = scene[paste_y:paste_y + target_h, paste_x:paste_x + target_w]

    if roi.shape[:2] != (target_h, target_w):
        return None

    blended = np.where(mask_3ch > 0, colored_eq, roi)
    scene[paste_y:paste_y + target_h, paste_x:paste_x + target_w] = blended

    return (paste_x, paste_y, paste_x + target_w, paste_y + target_h)


# ---------------------------------------------------------------------------
# 3. Synthetic hard negatives
# ---------------------------------------------------------------------------

def _generate_synthetic_negatives(
    output_img_dir: Path,
    output_lbl_dir: Path,
    count: int,
    rng: random.Random,
) -> int:
    """
    Generate synthetic ceiling/wall images with no whiteboard or equations.
    Each gets an empty label file (YOLO treats as 'no objects').
    """
    generated = 0
    for i in range(count):
        w = rng.randint(480, 800)
        h = rng.randint(360, 600)

        # Base: off-white/gray ceiling or wall
        base_val = rng.randint(200, 245)
        img = np.full((h, w, 3), base_val, dtype=np.uint8)

        # Add subtle color tint
        tint = np.array([rng.randint(-10, 10) for _ in range(3)])
        img = np.clip(img.astype(np.int16) + tint, 0, 255).astype(np.uint8)

        # Add noise texture
        noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add 1-3 fluorescent light panels (bright white rectangles)
        n_lights = rng.randint(1, 3)
        for _ in range(n_lights):
            lw = rng.randint(w // 6, w // 3)
            lh = rng.randint(15, 40)
            lx = rng.randint(0, max(1, w - lw))
            ly = rng.randint(0, max(1, h // 3))  # lights in top third
            cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), -1)
            # Glow effect
            glow_roi = img[max(0, ly - 5):min(h, ly + lh + 5),
                           max(0, lx - 5):min(w, lx + lw + 5)]
            if glow_roi.size > 0:
                blurred = cv2.GaussianBlur(glow_roi, (11, 11), 3)
                img[max(0, ly - 5):min(h, ly + lh + 5),
                    max(0, lx - 5):min(w, lx + lw + 5)] = blurred

        # Optionally add structural lines (ceiling grid)
        if rng.random() > 0.4:
            n_lines = rng.randint(1, 4)
            for _ in range(n_lines):
                y_line = rng.randint(0, h)
                color = rng.randint(180, 220)
                cv2.line(img, (0, y_line), (w, y_line), (color, color, color), 1)

        fname = f"neg_synth_{i:04d}.png"
        cv2.imwrite(str(output_img_dir / fname), img)
        # Empty label = no objects
        (output_lbl_dir / f"neg_synth_{i:04d}.txt").write_text("")
        generated += 1

    return generated


def _copy_real_negatives(
    negatives_dir: Path,
    output_img_dir: Path,
    output_lbl_dir: Path,
) -> int:
    """Copy user-provided hard negative images with empty labels."""
    if not negatives_dir.exists():
        return 0
    copied = 0
    for img_path in sorted(negatives_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        dest_name = f"neg_real_{img_path.name}"
        shutil.copy2(img_path, output_img_dir / dest_name)
        (output_lbl_dir / (Path(dest_name).stem + ".txt")).write_text("")
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# 4. Main composite generation
# ---------------------------------------------------------------------------

def generate_composite(
    roboflow_dir: Path,
    mathwriting_dir: Path,
    output_dir: Path,
    negatives_dir: Path | None,
    synthetic_neg_count: int,
    val_split: float,
    seed: int,
) -> Path:
    """
    Generate the composite dataset.

    Returns path to the generated dataset_v2.yaml.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # --- Discover Roboflow splits ---
    rf_splits = {}
    for split in ("train", "valid", "test"):
        img_dir = roboflow_dir / split / "images"
        lbl_dir = roboflow_dir / split / "labels"
        if img_dir.exists():
            images = sorted(
                p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
            )
            rf_splits[split] = (img_dir, lbl_dir, images)

    if not rf_splits:
        print("[ERROR] No Roboflow image splits found. Run fetch_real_dataset.py first.")
        sys.exit(1)

    total_rf = sum(len(imgs) for _, _, imgs in rf_splits.values())
    print(f"[PHASE 1] [STEP 1] Found {total_rf} Roboflow images across "
          f"{list(rf_splits.keys())} splits.")

    # --- Load equation pool ---
    eq_pool = _load_equation_pool(mathwriting_dir)

    # --- Prepare output dirs ---
    for split in ("train", "valid"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- Process each Roboflow image ---
    # Map Roboflow splits: train->train, valid->valid, test->valid
    split_map = {"train": "train", "valid": "valid", "test": "valid"}

    stats = {"images": 0, "equations_pasted": 0, "whiteboards": 0, "skipped": 0}

    for rf_split, (img_dir, lbl_dir, images) in rf_splits.items():
        out_split = split_map.get(rf_split, "train")
        out_img_dir = output_dir / out_split / "images"
        out_lbl_dir = output_dir / out_split / "labels"

        print(f"[PHASE 1] [STEP 3] Compositing {rf_split} "
              f"({len(images)} images) -> {out_split} ...")

        for img_path in images:
            scene = cv2.imread(str(img_path))
            if scene is None:
                stats["skipped"] += 1
                continue

            img_h, img_w = scene.shape[:2]

            # Parse whiteboard bboxes
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            wb_bboxes_norm = _parse_roboflow_label(lbl_path)

            if not wb_bboxes_norm:
                stats["skipped"] += 1
                continue

            label_lines = []

            for cx, cy, w, h in wb_bboxes_norm:
                # Add whiteboard label (class 1)
                label_lines.append(
                    f"{WHITEBOARD_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                stats["whiteboards"] += 1

                # Paste equations inside this whiteboard
                wb_x1, wb_y1, wb_x2, wb_y2 = _yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                n_eq = rng.randint(MIN_EQ_PER_BOARD, MAX_EQ_PER_BOARD)

                for _ in range(n_eq):
                    eq_path = rng.choice(eq_pool)
                    result = _paste_equation(
                        scene, eq_path, wb_x1, wb_y1, wb_x2, wb_y2, rng
                    )
                    if result is not None:
                        ex1, ey1, ex2, ey2 = result
                        ecx, ecy, ew, eh = _xyxy_to_yolo(
                            ex1, ey1, ex2, ey2, img_w, img_h
                        )
                        label_lines.append(
                            f"{EQUATION_CLASS_ID} {ecx:.6f} {ecy:.6f} "
                            f"{ew:.6f} {eh:.6f}"
                        )
                        stats["equations_pasted"] += 1

            # Save composited image and label
            out_fname = f"comp_{img_path.stem}{img_path.suffix}"
            cv2.imwrite(str(out_img_dir / out_fname), scene)
            (out_lbl_dir / f"comp_{img_path.stem}.txt").write_text(
                "\n".join(label_lines) + "\n"
            )
            stats["images"] += 1

    print(f"[PHASE 1] [STEP 3] Composited {stats['images']} images, "
          f"{stats['equations_pasted']} equations pasted, "
          f"{stats['whiteboards']} whiteboards, "
          f"{stats['skipped']} skipped.")

    # --- Hard negatives ---
    print(f"[PHASE 1] [STEP 4] Generating hard negatives ...")

    # Distribute negatives: 85% train, 15% valid
    n_neg_train = max(1, int(synthetic_neg_count * 0.85))
    n_neg_valid = synthetic_neg_count - n_neg_train

    neg_train = _generate_synthetic_negatives(
        output_dir / "train" / "images",
        output_dir / "train" / "labels",
        n_neg_train, rng,
    )
    neg_valid = _generate_synthetic_negatives(
        output_dir / "valid" / "images",
        output_dir / "valid" / "labels",
        n_neg_valid, rng,
    )
    print(f"[PHASE 1] [STEP 4] Generated {neg_train + neg_valid} synthetic negatives "
          f"(train={neg_train}, valid={neg_valid}).")

    # Copy user-provided negatives (all to train)
    if negatives_dir and negatives_dir.exists():
        n_real = _copy_real_negatives(
            negatives_dir,
            output_dir / "train" / "images",
            output_dir / "train" / "labels",
        )
        print(f"[PHASE 1] [STEP 4] Copied {n_real} real hard negatives from {negatives_dir}.")

    # --- Write dataset_v2.yaml ---
    yaml_path = Path("data/dataset_v2.yaml")
    doc = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 2,
        "names": CLASS_NAMES,
    }
    yaml_path.write_text(yaml.dump(doc, default_flow_style=False, allow_unicode=True))
    print(f"[PHASE 1] [STEP 5] dataset_v2.yaml written to {yaml_path}")

    # --- Summary ---
    train_count = len(list((output_dir / "train" / "images").iterdir()))
    valid_count = len(list((output_dir / "valid" / "images").iterdir()))

    print()
    print("=" * 55)
    print("COMPOSITE DATASET SUMMARY")
    print("=" * 55)
    print(f"  Classes      : {CLASS_NAMES}")
    print(f"  Train images : {train_count}")
    print(f"  Valid images : {valid_count}")
    print(f"  Total        : {train_count + valid_count}")
    print(f"  Equations    : {stats['equations_pasted']}")
    print(f"  Whiteboards  : {stats['whiteboards']}")
    print(f"  Hard negs    : {neg_train + neg_valid} synthetic")
    print(f"  Location     : {output_dir.resolve()}")
    print("=" * 55)

    return yaml_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate composite 2-class YOLO dataset "
                    "(MathWriting equations pasted onto Roboflow whiteboards)"
    )
    parser.add_argument(
        "--roboflow-dir", type=Path,
        default=Path("data/raw/roboflow_whiteboard"),
        help="Roboflow whiteboard dataset root (default: data/raw/roboflow_whiteboard)",
    )
    parser.add_argument(
        "--mathwriting-dir", type=Path,
        default=Path("data/raw/mathwriting"),
        help="MathWriting dataset root (default: data/raw/mathwriting)",
    )
    parser.add_argument(
        "--negatives-dir", type=Path, default=None,
        help="Optional directory of real hard negative images (ceiling/wall photos)",
    )
    parser.add_argument(
        "--synthetic-negatives", type=int, default=40,
        help="Number of synthetic hard negatives to generate (default: 40)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/raw/composite"),
        help="Output directory (default: data/raw/composite)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Validation split fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    if not args.roboflow_dir.exists():
        print(f"[ERROR] Roboflow directory not found: {args.roboflow_dir}")
        print("        Run: python tests/fetch_real_dataset.py --source roboflow --api-key YOUR_KEY")
        sys.exit(1)

    if not args.mathwriting_dir.exists():
        print(f"[ERROR] MathWriting directory not found: {args.mathwriting_dir}")
        print("        Run: python tests/fetch_real_dataset.py --source mathwriting")
        sys.exit(1)

    # Clean previous output
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        print(f"[INFO] Removed previous composite dataset at {args.output_dir}")

    generate_composite(
        roboflow_dir=args.roboflow_dir,
        mathwriting_dir=args.mathwriting_dir,
        output_dir=args.output_dir,
        negatives_dir=args.negatives_dir,
        synthetic_neg_count=args.synthetic_negatives,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
