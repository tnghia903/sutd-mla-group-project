"""
test_pipeline.py — Batch pipeline test on real whiteboard images
=================================================================
Runs detect_layout.py's full inference pipeline on all images inside
data/images/ and writes visualisations to output/detections/.

Usage:
    python tests/test_pipeline.py                  # test all images
    python tests/test_pipeline.py --limit 10       # test first 10 images
    python tests/test_pipeline.py --conf 0.10      # lower confidence threshold
    python tests/test_pipeline.py --model yolov8n.pt  # explicit model path

Output:
    output/detections/<image_name>_det.png  — annotated image per input
    output/results_summary.json             — machine-readable summary

The script prints a per-image table and a final summary table so you can
quickly spot which images had the most / fewest detections.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detect_layout import (
    image_file_to_array,
    preprocess_whiteboard,
    load_yolo_model,
    run_inference,
    draw_detections,
    CLASS_NAMES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGES_DIR   = Path("data/images")
OUTPUT_DIR   = Path("output/detections")
SUMMARY_PATH = Path("output/results_summary.json")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(limit: int | None = None) -> list[Path]:
    images = sorted(
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        images = images[:limit]
    return images


def _print_table_row(img_name: str, n_det: int, class_counts: dict[str, int], elapsed: float) -> None:
    counts_str = "  ".join(f"{n}×{cls[:4]}" for cls, n in class_counts.items() if n > 0)
    print(f"  {img_name:<40} {n_det:>4} det  [{counts_str or 'none'}]  {elapsed:.2f}s")


def _print_summary(results: list[dict]) -> None:
    total_imgs = len(results)
    total_dets = sum(r["n_detections"] for r in results)
    zero_det   = sum(1 for r in results if r["n_detections"] == 0)

    # Per-class totals
    class_totals: dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}
    for r in results:
        for cls, n in r["class_counts"].items():
            class_totals[cls] = class_totals.get(cls, 0) + n

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images tested : {total_imgs}")
    print(f"  Total detects : {total_dets}")
    print(f"  Zero-detect   : {zero_det}  (model may need fine-tuning on whiteboard classes)")
    print()
    print("  Detections by class:")
    for cls, n in class_totals.items():
        bar = "#" * min(n, 40)
        print(f"    {cls:<15} {n:>5}  {bar}")
    print("=" * 60)

    if zero_det == total_imgs:
        print(
            "\n[NOTE] All images returned 0 detections.\n"
            "  The stock yolov8n.pt is trained on COCO (80 everyday classes),\n"
            "  not whiteboard-specific classes. This is expected behaviour.\n"
            "  To get meaningful detections, fine-tune yolov8n on the downloaded\n"
            "  dataset using data/dataset.yaml — see the YOLOv8 training docs:\n"
            "  https://docs.ultralytics.com/modes/train/"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch test detect_layout.py on all images in data/images/"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of images to test (default: all)")
    parser.add_argument("--conf",  type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model path / checkpoint (default: yolov8n.pt)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving visualisation images (faster)")
    args = parser.parse_args()

    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        print(
            "[ERROR] data/images/ is empty.\n"
            "  Run the dataset downloader first:\n"
            "    python tests/fetch_real_dataset.py --source github\n"
            "  or\n"
            "    python tests/fetch_real_dataset.py --api-key YOUR_ROBOFLOW_KEY"
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    images = _collect_images(args.limit)
    print(f"[INFO] Found {len(images)} images in {IMAGES_DIR}/")
    print(f"[INFO] Model: {args.model}   Confidence threshold: {args.conf}")
    if args.limit:
        print(f"[INFO] Limiting to first {args.limit} images (--limit {args.limit})")
    print()
    print(f"  {'Image':<40} {'Dets':>4}       Class breakdown          Time")
    print("  " + "-" * 70)

    # Load model once, reuse for all images
    model = load_yolo_model(args.model)

    results = []

    for img_path in images:
        t0 = time.perf_counter()

        try:
            img = image_file_to_array(img_path)
            img = preprocess_whiteboard(img)
            regions = run_inference(model, img, conf=args.conf)
        except Exception as exc:
            print(f"  [SKIP] {img_path.name}: {exc}")
            continue

        elapsed = time.perf_counter() - t0

        # Class-level counts for this image
        class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}
        for r in regions:
            class_counts[r.class_name] = class_counts.get(r.class_name, 0) + 1

        _print_table_row(img_path.name, len(regions), class_counts, elapsed)

        if not args.no_vis:
            out_path = OUTPUT_DIR / f"{img_path.stem}_det.png"
            draw_detections(img, regions, output_path=out_path)

        results.append({
            "image": img_path.name,
            "n_detections": len(regions),
            "class_counts": class_counts,
            "inference_time_s": round(elapsed, 4),
            "detections": [
                {
                    "class_name": r.class_name,
                    "confidence": round(r.confidence, 4),
                    "bbox_xyxy": list(r.bbox_xyxy),
                }
                for r in regions
            ],
        })

    # Write JSON summary
    SUMMARY_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[INFO] Detailed results saved to {SUMMARY_PATH}")

    if not args.no_vis:
        print(f"[INFO] Annotated images saved to {OUTPUT_DIR}/")

    _print_summary(results)


if __name__ == "__main__":
    main()
