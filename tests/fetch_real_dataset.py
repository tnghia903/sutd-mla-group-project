"""
fetch_real_dataset.py — Download open-source whiteboard/math datasets
======================================================================
Four sources available:

  --source roboflow     Real whiteboard photos + YOLO labels (requires free API key)
                        https://universe.roboflow.com/whiteboard-kw2vt/whiteboard-detect

  --source github       Handwritten diagram images from GitHub (no auth)
                        https://github.com/bernhardschaefer/handwritten-diagram-datasets

  --source math         Handwritten math symbol photos from GitHub (no auth)
                        https://github.com/wblachowski/bhmsds
                        ~27,000 images of 18 symbols (digits, operators, brackets)

  --source mathwriting  MathWriting 2024 excerpt — Google Research (CC BY-NC-SA 4.0)
                        https://arxiv.org/abs/2404.10690
                        Real handwritten math expressions as InkML ink traces.
                        This script rasterizes them to PNG using matplotlib
                        (no system dependencies required).

Usage:
    python tests/fetch_real_dataset.py --api-key YOUR_KEY        # Roboflow
    python tests/fetch_real_dataset.py --source github           # diagrams
    python tests/fetch_real_dataset.py --source math             # math symbols
    python tests/fetch_real_dataset.py --source mathwriting      # MathWriting excerpt

Get a FREE Roboflow API key at: https://app.roboflow.com/settings/api
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import yaml
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGES_DIR  = Path("data/images")
LABELS_DIR  = Path("data/labels")
RAW_DIR     = Path("data/raw")
YAML_PATH   = Path("data/dataset.yaml")

# Shared constants to avoid duplication
IMG_EXTS       = (".jpg", ".jpeg", ".png")
TRAIN_SUB      = "train/images"
VAL_SUB        = "valid/images"
PROJECT_NAMES  = ["Handwriting", "Diagram", "Arrow", "Equation", "Sticky Note"]

# Roboflow dataset identifiers
RF_WORKSPACE = "whiteboard-kw2vt"
RF_PROJECT   = "whiteboard-detect"
RF_VERSION   = 1

# GitHub source B: handwritten-diagram-datasets (Apache-2.0)
GITHUB_DIAGRAM_ZIP_URL = (
    "https://github.com/bernhardschaefer/handwritten-diagram-datasets"
    "/archive/refs/heads/master.zip"
)

# GitHub source C: BHMSDS — Basic Handwritten Math Symbols Dataset (MIT)
# 27k images of 18 classes: 0-9 digits, +, -, times, div, =, (, ), /
GITHUB_MATH_ZIP_URL = (
    "https://github.com/wblachowski/bhmsds"
    "/archive/refs/heads/master.zip"
)
# Maps BHMSDS directory names → our project class ID 3 (Equation)
# All symbols are mathematical, so they all map to Equation.
BHMSDS_CLASS_ID = 3   # Equation

# Validation split fraction for the math dataset
MATH_VAL_SPLIT = 0.15
RANDOM_SEED    = 42

# MathWriting 2024 excerpt (Google Research, CC BY-NC-SA 4.0)
MATHWRITING_EXCERPT_URL = (
    "https://storage.googleapis.com/mathwriting_data/"
    "mathwriting-2024-excerpt.tgz"
)
MATHWRITING_CLASS_ID  = 3    # Equation
MATHWRITING_VAL_SPLIT = 0.15

PNG_GLOB       = "*.png"
EXTRACTING_MSG = "[INFO] Extracting …"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (IMAGES_DIR, LABELS_DIR, RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _update_yaml() -> None:
    """Write dataset.yaml for a flat data/images/ directory (no train/val split)."""
    doc = {
        "path":  str(IMAGES_DIR.resolve().parent),
        "train": "images",
        "val":   "images",
        "nc":    5,
        "names": PROJECT_NAMES,
    }
    YAML_PATH.write_text(yaml.dump(doc, default_flow_style=False, allow_unicode=True))
    print(f"[INFO] dataset.yaml updated at {YAML_PATH}")


def _write_split_yaml(root: Path,
                      train_sub: str = TRAIN_SUB,
                      val_sub: str = VAL_SUB) -> None:
    """Write dataset.yaml pointing at a train/valid split under root."""
    doc = {
        "path":  str(root.resolve()),
        "train": train_sub,
        "val":   val_sub,
        "nc":    5,
        "names": PROJECT_NAMES,
    }
    YAML_PATH.write_text(yaml.dump(doc, default_flow_style=False, allow_unicode=True))
    print(f"[INFO] dataset.yaml written at {YAML_PATH}")


def _count(directory: Path, ext: str = PNG_GLOB) -> int:
    return (len(list(directory.glob(ext)))
            + len(list(directory.glob(ext.replace("png", "jpg")))))


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // max(total_size, 1))
    sys.stdout.write(f"\r       {pct}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
    sys.stdout.flush()


def _download_zip(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] Zip already exists at {dest}, skipping download.")
        return
    print(f"[INFO] Downloading {url} …")
    urllib.request.urlretrieve(url, dest, _progress_hook)
    print()
    print(f"[INFO] Saved to {dest}")


# ---------------------------------------------------------------------------
# Source A: Roboflow (requires free API key)
# ---------------------------------------------------------------------------

def download_roboflow(api_key: str) -> None:
    """
    Download the whiteboard-detect dataset from Roboflow in YOLOv8 format.
    ~1,124 real whiteboard photos with YOLO bounding-box annotations.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not found.  pip install roboflow")
        sys.exit(1)

    print("[INFO] Connecting to Roboflow …")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)

    download_dir = RAW_DIR / "roboflow_whiteboard"
    print(f"[INFO] Downloading dataset to {download_dir} …")
    version.download("yolov8", location=str(download_dir))
    print("[INFO] Download complete.")

    _wire_roboflow_yaml(download_dir)


def _wire_roboflow_yaml(rf_root: Path) -> None:
    """Point dataset.yaml at the Roboflow directory, using Roboflow's own class names."""
    rf_yaml_path = rf_root / "data.yaml"

    if not rf_yaml_path.exists():
        print("[WARN] Roboflow data.yaml not found — falling back to project schema.")
        _write_split_yaml(rf_root)
        return

    with rf_yaml_path.open() as fh:
        rf_meta = yaml.safe_load(fh)

    raw_names = rf_meta.get("names", [])
    names_list = (
        [raw_names[k] for k in sorted(raw_names)]
        if isinstance(raw_names, dict)
        else list(raw_names)
    )

    doc = {
        "path":  str(rf_root.resolve()),
        "train": TRAIN_SUB,
        "val":   VAL_SUB,
        "nc":    len(names_list),
        "names": names_list,
    }
    YAML_PATH.write_text(yaml.dump(doc, default_flow_style=False, allow_unicode=True))

    train_imgs = sum(1 for p in (rf_root / "train" / "images").glob("*")
                     if p.suffix.lower() in IMG_EXTS)
    valid_imgs = sum(1 for p in (rf_root / "valid" / "images").glob("*")
                     if p.suffix.lower() in IMG_EXTS)

    print(f"[INFO] dataset.yaml written at {YAML_PATH}")
    print(f"[INFO] Roboflow classes ({len(names_list)}): {names_list}")
    print(f"[INFO] Train images : {train_imgs}")
    print(f"[INFO] Valid images : {valid_imgs}")


# ---------------------------------------------------------------------------
# Source B: GitHub handwritten-diagram-datasets
# ---------------------------------------------------------------------------

def download_github() -> None:
    """
    Repo  : https://github.com/bernhardschaefer/handwritten-diagram-datasets
    License: Apache-2.0
    Content: ~500 scanned handwritten diagram images (flowcharts, UML)
    Annotations: COCO JSON — diagrams, arrows, text boxes, shapes
    """
    zip_path    = RAW_DIR / "handwritten_diagram_datasets.zip"
    extract_dir = RAW_DIR / "github_extract"

    _download_zip(GITHUB_DIAGRAM_ZIP_URL, zip_path)

    print(EXTRACTING_MSG)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    copied = 0
    for img_path in sorted(extract_dir.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        dest = IMAGES_DIR / f"diagram_{img_path.name}"
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1

    print(f"[INFO] Copied {copied} diagram images → {IMAGES_DIR}/")
    if copied == 0:
        print("[WARN] No images found — the repo structure may have changed.")
    else:
        print("[NOTE] No YOLO labels generated. Pipeline will run inference only.")

    _update_yaml()


# ---------------------------------------------------------------------------
# Source C: BHMSDS — handwritten math symbols (new)
# ---------------------------------------------------------------------------

def download_math() -> None:
    """
    Repo   : https://github.com/wblachowski/bhmsds
    License: MIT
    Content: ~27,000 real photos of 18 handwritten mathematical symbols
             Organised into per-class subdirectories:
               0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/
               add/ sub/ times/ div/ eq/ lpar/ rpar/ div2/
    Format : Pre-cropped symbol images (no bounding boxes needed —
             each image IS the symbol, so we synthesise a full-frame
             YOLO label: class_id 0.5 0.5 1.0 1.0).
    Mapped to project class 3 (Equation) with a 85/15 train/val split.
    """
    zip_path    = RAW_DIR / "bhmsds.zip"
    extract_dir = RAW_DIR / "bhmsds_extract"
    out_root    = RAW_DIR / "math_symbols"

    _download_zip(GITHUB_MATH_ZIP_URL, zip_path)

    print(EXTRACTING_MSG)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Collect all images grouped by symbol class directory
    all_images: list[Path] = []
    for img_path in sorted(extract_dir.rglob("*")):
        if img_path.suffix.lower() in IMG_EXTS:
            all_images.append(img_path)

    if not all_images:
        print("[WARN] No images found in archive — repo structure may have changed.")
        return

    print(f"[INFO] Found {len(all_images)} symbol images across "
          f"{len({p.parent.name for p in all_images})} classes.")

    # Shuffle and split into train / valid
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)
    n_val   = max(1, int(len(all_images) * MATH_VAL_SPLIT))
    val_set = {str(p) for p in all_images[:n_val]}

    for split in ("train", "valid"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "valid": 0}
    for img_path in all_images:
        split    = "valid" if str(img_path) in val_set else "train"
        # Prefix with symbol name to preserve context
        dest_name = f"math_{img_path.parent.name}_{img_path.name}"
        dest_img  = out_root / split / "images" / dest_name
        dest_lbl  = out_root / split / "labels" / (Path(dest_name).stem + ".txt")

        shutil.copy2(img_path, dest_img)

        # Full-frame YOLO label: <class_id> <cx> <cy> <w> <h> (all normalised)
        dest_lbl.write_text(f"{BHMSDS_CLASS_ID} 0.5 0.5 1.0 1.0\n")
        moved[split] += 1

    print(f"[INFO] Train images : {moved['train']}")
    print(f"[INFO] Valid images : {moved['valid']}")

    # Write dataset.yaml pointing at the split directories
    doc = {
        "path":  str(out_root.resolve()),
        "train": TRAIN_SUB,
        "val":   VAL_SUB,
        "nc":    5,
        "names": ["Handwriting", "Diagram", "Arrow", "Equation", "Sticky Note"],
    }
    YAML_PATH.write_text(yaml.dump(doc, default_flow_style=False, allow_unicode=True))
    print(f"[INFO] dataset.yaml written at {YAML_PATH}")
    print()
    print("[NOTE] All symbols are labelled as class 3 (Equation).")
    print("       Each image is treated as a single full-frame Equation region.")
    print("       Fine-tune with: open notebooks/train_layout.ipynb")


# ---------------------------------------------------------------------------
# Source D: MathWriting 2024 excerpt (InkML → PNG rasterization)
# ---------------------------------------------------------------------------

def _parse_inkml_strokes(
    inkml_path: Path,
) -> list[tuple[list[float], list[float]]]:
    """
    Parse an InkML file and return a list of (xs, ys) stroke coordinate lists.
    Returns an empty list if the file is malformed or contains no usable strokes.
    """
    ns = {"ink": "http://www.w3.org/2003/InkML"}
    try:
        tree = ET.parse(str(inkml_path))
    except ET.ParseError:
        return []

    root    = tree.getroot()
    traces  = root.findall(".//ink:trace", ns) or root.findall(".//trace")
    strokes = []

    for trace in traces:
        text = (trace.text or "").strip()
        xs, ys = [], []
        for point in text.split(","):
            parts = point.strip().split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
        if len(xs) >= 2:
            strokes.append((xs, ys))

    return strokes


def _rasterize_inkml(inkml_path: Path, out_png: Path, dpi: int = 96) -> bool:
    """
    Convert a single MathWriting InkML file to a PNG image using matplotlib.

    Returns True on success, False if the file has no usable strokes.
    """
    import matplotlib
    matplotlib.use("Agg")        # non-interactive backend — no display needed
    import matplotlib.pyplot as plt

    strokes = _parse_inkml_strokes(inkml_path)
    if not strokes:
        return False

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for xs, ys in strokes:
        # InkML y-axis grows downward; matplotlib y grows upward — invert
        ax.plot(xs, [-y for y in ys], color="black", linewidth=1.5)

    ax.set_aspect("equal")
    ax.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    return True


def download_mathwriting() -> None:
    """
    Download and rasterize the MathWriting 2024 excerpt dataset.

    Source  : https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz
    License : CC BY-NC-SA 4.0 (Google Research)
    Content : Real handwritten mathematical expressions as InkML ink traces.
              Each file = one complete math expression written by a human.

    Pipeline:
        1. Download the .tgz archive (~1.5 MB).
        2. Extract to data/raw/mathwriting_extract/.
        3. Rasterize every .inkml file → PNG using matplotlib.
        4. Assign full-frame YOLO label: class 3 (Equation) 0.5 0.5 1.0 1.0.
        5. Split 85/15 train/valid.
        6. Write data/dataset.yaml pointing at data/raw/mathwriting/.
    """
    tgz_path    = RAW_DIR / "mathwriting_excerpt.tgz"
    extract_dir = RAW_DIR / "mathwriting_extract"
    out_root    = RAW_DIR / "mathwriting"

    _download_zip(MATHWRITING_EXCERPT_URL, tgz_path)

    print(EXTRACTING_MSG)
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(extract_dir)

    # Collect all .inkml files recursively
    inkml_files = sorted(extract_dir.rglob("*.inkml"))
    if not inkml_files:
        print("[WARN] No .inkml files found. The archive structure may have changed.")
        print(f"       Inspect: {extract_dir}")
        return

    print(f"[INFO] Found {len(inkml_files)} InkML files. Rasterizing …")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    shuffled = list(inkml_files)
    random.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * MATHWRITING_VAL_SPLIT))
    val_set = {str(p) for p in shuffled[:n_val]}

    for split in ("train", "valid"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    ok = skipped = 0
    for inkml_path in shuffled:
        split     = "valid" if str(inkml_path) in val_set else "train"
        stem      = f"mw_{inkml_path.stem}"
        out_png   = out_root / split / "images" / f"{stem}.png"
        out_label = out_root / split / "labels" / f"{stem}.txt"

        if _rasterize_inkml(inkml_path, out_png):
            out_label.write_text(f"{MATHWRITING_CLASS_ID} 0.5 0.5 1.0 1.0\n")
            ok += 1
        else:
            skipped += 1

        if (ok + skipped) % 100 == 0:
            sys.stdout.write(f"\r       Rasterized {ok} / {ok + skipped} …")
            sys.stdout.flush()

    print(f"\n[INFO] Rasterized {ok} images  ({skipped} skipped — empty strokes)")

    n_train = sum(1 for _ in (out_root / "train" / "images").glob(PNG_GLOB))
    n_valid = sum(1 for _ in (out_root / "valid" / "images").glob(PNG_GLOB))
    print(f"[INFO] Train images : {n_train}")
    print(f"[INFO] Valid images : {n_valid}")

    _write_split_yaml(out_root)
    print("[NOTE] All samples labelled as class 3 (Equation).")
    print("       Fine-tune with: open notebooks/train_layout.ipynb")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a whiteboard / math dataset for training detect_layout.py"
    )
    parser.add_argument(
        "--source",
        choices=["roboflow", "github", "math", "mathwriting"],
        default="mathwriting",
        help=(
            "roboflow    : whiteboard photos + YOLO labels (requires --api-key). "
            "github      : handwritten diagram scans, no auth needed. "
            "math        : 27k handwritten math symbol photos, no auth needed. "
            "mathwriting : MathWriting 2024 excerpt, rasterized from InkML (recommended)."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (only for --source roboflow). "
             "Get a free key at https://app.roboflow.com/settings/api",
    )
    args = parser.parse_args()

    _ensure_dirs()

    before = _count(IMAGES_DIR)
    print(f"[INFO] Existing images in data/images/: {before}")

    if args.source == "roboflow":
        if not args.api_key:
            print(
                "[ERROR] --api-key is required for --source roboflow.\n"
                "        Get a FREE key at: https://app.roboflow.com/settings/api\n"
                "        Or try the no-auth options:\n"
                "          python tests/fetch_real_dataset.py --source math\n"
                "          python tests/fetch_real_dataset.py --source github"
            )
            sys.exit(1)
        download_roboflow(args.api_key)
    elif args.source == "github":
        download_github()
    elif args.source == "math":
        download_math()
    else:
        download_mathwriting()

    after = _count(IMAGES_DIR)
    print(f"\n[DONE] Total images in data/images/: {after} (+{after - before} new)")
    print("\nNext step — fine-tune the model:")
    print("  Open notebooks/train_layout.ipynb")
    print("\nOr test inference immediately:")
    print("  python tests/run_pipeline.py data/images/<filename>")


if __name__ == "__main__":
    main()
