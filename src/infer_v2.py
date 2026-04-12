"""
infer_v2.py -- V2 whiteboard digitisation pipeline (2-class detection + math OCR)
=================================================================================
Single-pass detection flow:
    1. Load image
    2. Run v2 model (model/best_v2.pt) -> equation + whiteboard bboxes
    3. Filter: discard detections in top 25% of image (ceiling suppression)
    4. Filter: keep equations whose center falls inside a whiteboard bbox
    5. Preprocess equation crops (CLAHE, binarize, upscale)
    6. Run LaTeX OCR via fallback chain (pix2tex -> TrOCR -> PaddleOCR)
    7. Output: annotated image + Markdown with LaTeX blocks

Usage:
    python src/infer_v2.py --image path/to/whiteboard.jpg
    python src/infer_v2.py --image path/to/whiteboard.jpg --model model/best_v2.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Make src/ importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_layout import (
    CroppedRegion,
    image_file_to_array,
    preprocess_whiteboard,
    load_yolo_model,
    run_inference,
    filter_by_center_y,
    merge_nearby_equations,
    equations_inside_whiteboards,
    draw_detections_v2,
    V2_MODEL_PATH,
    V2_CLASS_NAMES,
)
from transcribe_ocr import MathOCREngine


# ---------------------------------------------------------------------------
# Markdown formatter (v2)
# ---------------------------------------------------------------------------

def _to_markdown_v2(
    equations: list[CroppedRegion],
    ocr_results: list[tuple[str, float, str]],
) -> str:
    """
    Convert equation detections + OCR results into Markdown with LaTeX blocks.
    """
    lines = ["# Whiteboard Transcription (v2)\n"]

    if not equations:
        lines.append("_No equations detected._")
        return "\n".join(lines)

    for i, (eq, (text, conf, backend)) in enumerate(zip(equations, ocr_results)):
        lines.append(f"## Equation {i + 1}")
        if text.strip():
            if backend == "pix2tex":
                # pix2tex outputs LaTeX directly
                lines.append(f"$$\n{text.strip()}\n$$")
            else:
                # Other backends output plain text
                lines.append(f"```\n{text.strip()}\n```")
        else:
            lines.append("_[No text recognised]_")

        x1, y1, x2, y2 = eq.bbox_xyxy
        lines.append(
            f"> backend: {backend}  |  conf: {conf:.2f}  |  "
            f"bbox: ({x1},{y1})-({x2},{y2})\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline_v2(
    image_path: Path,
    model_path: str = V2_MODEL_PATH,
    conf: float = 0.25,
    save_vis: bool = True,
) -> str:
    """
    Run the full v2 pipeline on a single image.

    Returns:
        Markdown string with transcription results.
    """
    # ── Step 1: Load image ────────────────────────────────────────────────
    print(f"[PHASE 4] [STEP 1] Loading image: {image_path}")
    img = image_file_to_array(image_path)
    img = preprocess_whiteboard(img)
    img_h, img_w = img.shape[:2]
    print(f"[PHASE 4] [STEP 1] Image size: {img_w}x{img_h}")

    # ── Step 2: Detection ─────────────────────────────────────────────────
    print(f"[PHASE 4] [STEP 2] Running detection with {model_path} ...")
    model = load_yolo_model(model_path)

    # Override class names for v2 model
    all_regions = run_inference(model, img, conf=conf)

    # Remap class names using V2_CLASS_NAMES
    for r in all_regions:
        r.class_name = V2_CLASS_NAMES.get(r.class_id, r.class_name)

    # ── Step 3: Filter ceiling false positives ────────────────────────────
    before_count = len(all_regions)
    all_regions = filter_by_center_y(all_regions, img_h, min_ratio=0.25)
    ceiling_filtered = before_count - len(all_regions)

    # Separate by class
    whiteboards = [r for r in all_regions if r.class_id == 1]
    equations = [r for r in all_regions if r.class_id == 0]

    # ── Step 3b: Merge fragmented equation bboxes ─────────────────────────
    before_merge = len(equations)
    equations = merge_nearby_equations(equations, img)
    if before_merge != len(equations):
        print(f"[PHASE 4] [STEP 2] Merged {before_merge} fragments -> "
              f"{len(equations)} equation(s)")

    # ── Step 4: Keep equations inside whiteboards ─────────────────────────
    equations = equations_inside_whiteboards(equations, whiteboards)

    # Rebuild all_regions with merged equations for drawing
    all_regions = whiteboards + equations
    all_regions.sort(key=lambda r: (r.bbox_xyxy[1], r.bbox_xyxy[0]))

    print(f"[PHASE 4] [STEP 2] Detected {len(whiteboards)} whiteboard(s), "
          f"{len(equations)} equation(s) "
          f"(filtered {ceiling_filtered} ceiling FP)")

    # ── Step 5-6: OCR ─────────────────────────────────────────────────────
    ocr_results: list[tuple[str, float, str]] = []

    if equations:
        print(f"[PHASE 4] [STEP 3] Running math OCR on {len(equations)} equation(s) ...")
        try:
            engine = MathOCREngine()
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            print("[PHASE 4] [STEP 3] Falling back to empty OCR results.")
            engine = None

        for i, eq in enumerate(equations):
            if engine is not None:
                text, conf_ocr, backend = engine.recognise(eq.crop)
            else:
                text, conf_ocr, backend = "", 0.0, "none"
            ocr_results.append((text, conf_ocr, backend))
            display = text.replace("\n", " ")[:60]
            print(f"[PHASE 4] [STEP 3] Eq {i + 1}: {backend} "
                  f"(conf={conf_ocr:.2f}) -> {display}")

    # ── Step 7: Output ────────────────────────────────────────────────────
    # Build LaTeX overlay for visualisation
    latex_texts = {}
    eq_idx = 0
    for i, r in enumerate(all_regions):
        if r.class_id == 0 and eq_idx < len(ocr_results):
            text, _, _ = ocr_results[eq_idx]
            if text.strip():
                latex_texts[i] = text
            eq_idx += 1

    # Visualisation
    if save_vis:
        vis_path = Path("output") / f"{image_path.stem}_v2_det.png"
        draw_detections_v2(img, all_regions, latex_texts, output_path=vis_path)

    # Markdown
    markdown = _to_markdown_v2(equations, ocr_results)

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{image_path.stem}_v2.md"
    md_path.write_text(markdown, encoding="utf-8")

    print(f"\n[PHASE 4] [STEP 4] Markdown saved to {md_path}")
    if save_vis:
        print(f"[PHASE 4] [STEP 4] Annotated image saved to {vis_path}")

    return markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 whiteboard pipeline: 2-class detection + math OCR"
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to whiteboard image")
    parser.add_argument(
        "--model", default=V2_MODEL_PATH,
        help=f"YOLOv8 checkpoint (default: {V2_MODEL_PATH})",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip saving the annotated detection image",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    markdown = run_pipeline_v2(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        save_vis=not args.no_vis,
    )

    print("\n" + "-" * 60)
    print(markdown)
    print("-" * 60)


if __name__ == "__main__":
    main()
