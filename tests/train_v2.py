"""
train_v2.py -- Fine-tune YOLOv8-nano on the 2-class composite dataset
======================================================================
Trains from COCO-pretrained yolov8n.pt (NOT from the old 5-class checkpoint,
since the detection head dimensions are incompatible with the new 2-class task).

Saves the best checkpoint to model/best_v2.pt.

Usage:
    python tests/train_v2.py
    python tests/train_v2.py --epochs 40 --batch 8 --device mps
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_YAML    = "data/dataset_v2.yaml"
BASE_MODEL   = "notebooks/yolov8n.pt"
EXPERIMENT   = "whiteboard_composite_v2"
RUNS_DIR     = "runs/detect"
DEPLOY_PATH  = "model/best_v2.pt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-nano on 2-class composite dataset"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--device", default="mps", help="Device: mps, cuda, cpu (default: mps)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {DATA_YAML}")
        print("        Run: python tests/generate_composite.py")
        return

    base_model = Path(BASE_MODEL)
    if not base_model.exists():
        print(f"[ERROR] Base model not found: {BASE_MODEL}")
        print("        Expected COCO-pretrained yolov8n.pt in notebooks/")
        return

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[PHASE 2] [STEP 1] Loading base model: {BASE_MODEL}")
    model = YOLO(str(base_model))

    print(f"[PHASE 2] [STEP 1] Starting training: {args.epochs} epochs, "
          f"batch {args.batch}, device {args.device}")

    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,          # MPS-safe
        device=args.device,
        name=EXPERIMENT,
        project=RUNS_DIR,
        exist_ok=True,
        # Augmentations (matching v1 config)
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.40,
        fliplr=0.5,
        degrees=5.0,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        erasing=0.4,
        amp=True,
    )

    # ── Validate ──────────────────────────────────────────────────────────
    print("[PHASE 2] [STEP 2] Running validation ...")
    metrics = model.val(data=str(data_path.resolve()))

    # Report per-class mAP
    class_names = ["equation", "whiteboard"]
    print()
    print("=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    if hasattr(metrics, "box"):
        print(f"  mAP@50 (all)  : {metrics.box.map50:.4f}")
        print(f"  mAP@50:95     : {metrics.box.map:.4f}")
        if hasattr(metrics.box, "maps") and len(metrics.box.maps) >= 2:
            for i, name in enumerate(class_names):
                print(f"  mAP@50 ({name:10s}): {metrics.box.maps[i]:.4f}")
    print("=" * 50)

    # ── Deploy ────────────────────────────────────────────────────────────
    src = Path(RUNS_DIR) / EXPERIMENT / "weights" / "best.pt"
    dst = Path(DEPLOY_PATH)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.copy2(src, dst)
        print(f"[PHASE 2] [STEP 3] Deployed to {dst}")
    else:
        print(f"[ERROR] Best weights not found at {src}")
        # Try last.pt as fallback
        last = src.parent / "last.pt"
        if last.exists():
            shutil.copy2(last, dst)
            print(f"[PHASE 2] [STEP 3] Deployed last.pt to {dst} (best.pt unavailable)")

    print("[PHASE 2] Done.")


if __name__ == "__main__":
    main()
