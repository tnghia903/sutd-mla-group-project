"""
merge_datasets.py — Merge handwriting + equation datasets into one YOLO dataset
================================================================================
Combines:
    data/raw/handwriting_detect/   class 0 (handwriting)  → project class 0 (Handwriting)
    data/raw/equation_detect/      classes 0-3 (math types) → project class 3 (Equation)

Output:
    data/raw/merged/
        train/images/  train/labels/
        valid/images/  valid/labels/

    data/dataset.yaml  ← updated to point at merged dataset

Usage:
    python tests/merge_datasets.py
    python tests/merge_datasets.py --val-split 0.15   # custom valid fraction
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HANDWRITING_ROOT = Path("data/raw/handwriting_detect")
EQUATION_ROOT    = Path("data/raw/equation_detect")
MERGED_ROOT      = Path("data/raw/merged")
YAML_PATH        = Path("data/dataset.yaml")

# Class remapping
# handwriting dataset: class 0 (handwriting)   → our class 0 (Handwriting)
HANDWRITING_REMAP: dict[int, int] = {0: 0}

# equation dataset: classes 0-3 (Differentiation, Integration, Limits, Trigonometry)
#                   all → our class 3 (Equation)
EQUATION_REMAP: dict[int, int] = {0: 3, 1: 3, 2: 3, 3: 3}

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    class_remap: dict[int, int],
    prefix: str = "",
) -> int:
    """
    Copy images and remap class IDs in label files from src → dst.
    prefix is prepended to filenames to avoid collisions between datasets.
    Returns number of image files copied.
    """
    if not src_img_dir.exists():
        return 0

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        dst_img = dst_img_dir / f"{prefix}{img_path.name}"
        shutil.copy2(img_path, dst_img)

        # Find matching label file
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        dst_lbl  = dst_lbl_dir / f"{prefix}{img_path.stem}.txt"

        if lbl_path.exists():
            _remap_label(lbl_path, dst_lbl, class_remap)
        else:
            # Write empty label (image with no annotations)
            dst_lbl.write_text("")

        copied += 1

    return copied


def _remap_label(src: Path, dst: Path, class_remap: dict[int, int]) -> None:
    """
    Read a YOLO label file and rewrite it with remapped class IDs.
    Lines with unmapped class IDs are dropped.
    """
    out_lines: list[str] = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        orig_cls = int(parts[0])
        new_cls  = class_remap.get(orig_cls)
        if new_cls is None:
            continue  # drop unmapped classes
        out_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
    dst.write_text("".join(out_lines))


def _make_valid_from_train(
    train_img_dir: Path,
    train_lbl_dir: Path,
    valid_img_dir: Path,
    valid_lbl_dir: Path,
    fraction: float,
) -> tuple[int, int]:
    """
    Move a random fraction of train files into valid.
    Returns (remaining_train_count, moved_to_valid_count).
    """
    valid_img_dir.mkdir(parents=True, exist_ok=True)
    valid_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(
        p for p in train_img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    random.seed(RANDOM_SEED)
    random.shuffle(all_imgs)

    n_valid = max(1, int(len(all_imgs) * fraction))
    to_move = all_imgs[:n_valid]

    for img in to_move:
        shutil.move(str(img), valid_img_dir / img.name)
        lbl = train_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), valid_lbl_dir / lbl.name)

    return len(all_imgs) - n_valid, n_valid


def _write_yaml(merged_root: Path) -> None:
    content = (
        f"path: {merged_root.resolve()}\n"
        "train: train/images\n"
        "val:   valid/images\n\n"
        "names:\n"
        "  0: Handwriting\n"
        "  1: Diagram\n"
        "  2: Arrow\n"
        "  3: Equation\n"
        "  4: Sticky Note\n"
    )
    YAML_PATH.write_text(content)
    print(f"[INFO] dataset.yaml updated → {YAML_PATH}")


def _count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge handwriting + equation datasets into one YOLO dataset"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.20,
        help="Fraction of equation train images to use as validation (default: 0.20)"
    )
    args = parser.parse_args()

    # Clean previous merge
    if MERGED_ROOT.exists():
        shutil.rmtree(MERGED_ROOT)
        print(f"[INFO] Removed old merged dataset at {MERGED_ROOT}")

    print("\n[STEP 1] Merging handwriting dataset (class 0 → Handwriting) …")
    for split in ("train", "valid", "test"):
        src_img = HANDWRITING_ROOT / split / "images"
        src_lbl = HANDWRITING_ROOT / split / "labels"
        # Map test → valid (we include it as extra validation data)
        dst_split = "valid" if split == "test" else split
        n = _copy_split(
            src_img, src_lbl,
            MERGED_ROOT / dst_split / "images",
            MERGED_ROOT / dst_split / "labels",
            class_remap=HANDWRITING_REMAP,
            prefix="hw_",
        )
        if n:
            print(f"  {split:5s} → {dst_split}: {n} images")

    print("\n[STEP 2] Merging equation dataset (classes 0-3 → Equation class 3) …")
    for split in ("train", "test"):
        src_img = EQUATION_ROOT / split / "images"
        src_lbl = EQUATION_ROOT / split / "labels"
        # equation has no valid split — we'll carve it from train below
        # map test → valid
        dst_split = "valid" if split == "test" else split
        n = _copy_split(
            src_img, src_lbl,
            MERGED_ROOT / dst_split / "images",
            MERGED_ROOT / dst_split / "labels",
            class_remap=EQUATION_REMAP,
            prefix="eq_",
        )
        if n:
            print(f"  {split:5s} → {dst_split}: {n} images")

    print(f"\n[STEP 3] Carving {args.val_split:.0%} of equation train → valid …")
    train_kept, moved = _make_valid_from_train(
        MERGED_ROOT / "train" / "images",
        MERGED_ROOT / "train" / "labels",
        MERGED_ROOT / "valid" / "images",
        MERGED_ROOT / "valid" / "labels",
        fraction=args.val_split,
    )
    print(f"  Equation train kept : {train_kept}")
    print(f"  Moved to valid      : {moved}")

    print("\n[STEP 4] Writing dataset.yaml …")
    _write_yaml(MERGED_ROOT)

    # Summary
    train_total = _count(MERGED_ROOT / "train" / "images")
    valid_total = _count(MERGED_ROOT / "valid" / "images")
    print("\n" + "=" * 50)
    print("MERGED DATASET SUMMARY")
    print("=" * 50)
    print(f"  train images : {train_total}")
    print(f"  valid images : {valid_total}")
    print(f"  total        : {train_total + valid_total}")
    print(f"  location     : {MERGED_ROOT.resolve()}")
    print("=" * 50)
    print("\n[NEXT] Start training:")
    print("  Open and run notebooks/train_layout.ipynb")


if __name__ == "__main__":
    main()
