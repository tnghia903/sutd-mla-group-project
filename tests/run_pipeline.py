"""
run_pipeline.py — End-to-end whiteboard digitisation on a single image
=======================================================================
Chains all three stages:
    1. Layout detection  (YOLOv8)        → bounding boxes + crops
    2. Text recognition  (PP-OCRv3)      → raw transcribed text per region
    3. Markdown output                   → structured text printed to stdout
                                           and saved to output/<image_stem>.md

Usage:
    python tests/run_pipeline.py path/to/whiteboard.jpg
    python tests/run_pipeline.py path/to/whiteboard.jpg --model runs/detect/whiteboard_yolov8n/weights/best.pt
    python tests/run_pipeline.py path/to/whiteboard.jpg --conf 0.25 --no-vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detect_layout import (
    image_file_to_array,
    preprocess_whiteboard,
    load_yolo_model,
    run_inference,
    draw_detections,
    DEFAULT_MODEL_PATH,
)
from transcribe_ocr import OCREngine, transcribe_regions


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def _to_markdown(transcriptions) -> str:
    """
    Convert a list of TranscriptionResult into a structured Markdown string.
    Sections are grouped by class type in spatial (top-to-bottom) order.
    """
    lines = ["# Whiteboard Transcription\n"]

    for r in transcriptions:
        if r.region_class in ("Handwriting", "Equation"):
            if not r.text.strip():
                continue
            heading = "## Handwriting" if r.region_class == "Handwriting" else "## Equation"
            lines.append(heading)
            if r.region_class == "Equation":
                # Wrap equations in a LaTeX block
                lines.append(f"$$\n{r.text.strip()}\n$$")
            else:
                lines.append(r.text.strip())
            lines.append(f"> conf: {r.confidence:.2f}  |  lines: {r.line_count}\n")
        else:
            # Non-text regions: note their presence but no text to show
            x1, y1, x2, y2 = r.bbox_xyxy
            lines.append(f"## {r.region_class}")
            lines.append(f"_[{r.region_class} region at ({x1},{y1})–({x2},{y2})]_\n")

    if len(lines) == 1:
        lines.append("_No regions detected. Try a lower --conf threshold or fine-tune the model first._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end whiteboard digitisation: detection + OCR → Markdown"
    )
    parser.add_argument("image", help="Path to the whiteboard image")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"YOLOv8 checkpoint (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Detection confidence threshold (default: 0.10)"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip saving the annotated detection image"
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # ── Stage 1: Layout detection ──────────────────────────────────────────
    print(f"\n[Stage 1] Detecting regions in {img_path.name} …")
    img     = image_file_to_array(img_path)
    img     = preprocess_whiteboard(img)
    model   = load_yolo_model(args.model)
    regions = run_inference(model, img, conf=args.conf)

    if not regions:
        print("[WARN] No regions detected.")
        print("       If using the stock yolov8n.pt, fine-tune the model first.")
        print("       Try lowering --conf, e.g. --conf 0.10")

    # ── Annotated visualisation ────────────────────────────────────────────
    if not args.no_vis:
        vis_path = Path("output") / f"{img_path.stem}_det.png"
        draw_detections(img, regions, output_path=vis_path)

    # ── Stage 2: Text recognition ──────────────────────────────────────────
    print(f"\n[Stage 2] Transcribing text regions …")
    engine         = OCREngine()
    # target_classes=None: transcribe all detected regions regardless of class name,
    # so OCR works correctly before and after model retraining changes class names.
    transcriptions = transcribe_regions(regions, engine, target_classes=None)

    # ── Stage 3: Markdown output ───────────────────────────────────────────
    print(f"\n[Stage 3] Formatting output …")
    markdown = _to_markdown(transcriptions)

    out_dir  = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path  = out_dir / f"{img_path.stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    # Print to stdout
    print("\n" + "─" * 60)
    print(markdown)
    print("─" * 60)
    print(f"\n[Done] Markdown saved to {md_path}")
    if not args.no_vis:
        print(f"[Done] Annotated image saved to {vis_path}")


if __name__ == "__main__":
    main()
