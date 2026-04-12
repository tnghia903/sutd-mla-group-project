"""
validate_v2.py -- Visual validation of the v2 pipeline
=======================================================
Runs the full v2 pipeline on test images and checks:
    1. Zero detections in top 25% of image (ceiling FP suppression)
    2. Every equation bbox center falls inside a whiteboard bbox
    3. LaTeX output parseable by sympy.parsing.latex.parse_latex

Saves annotated images to output/validation/ and a JSON report.

Usage:
    python tests/validate_v2.py
    python tests/validate_v2.py --model model/best_v2.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detect_layout import (
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
# Test image discovery
# ---------------------------------------------------------------------------

def _discover_test_images(
    roboflow_valid_dir: Path | None = None,
) -> list[Path]:
    """
    Collect test images:
        1-3. The three images in data/images/
        4-5. First two from Roboflow validation set (if available)
    """
    images: list[Path] = []

    # Core test images
    data_images = Path("data/images")
    if data_images.exists():
        for p in sorted(data_images.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append(p)

    # Roboflow validation samples
    if roboflow_valid_dir and roboflow_valid_dir.exists():
        rf_imgs = sorted(
            p for p in roboflow_valid_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        images.extend(rf_imgs[:2])

    return images[:5]  # cap at 5


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_no_ceiling_detections(
    all_regions: list,
    img_h: int,
) -> tuple[bool, int]:
    """Check that no detections have center_y in top 25%."""
    ceiling_count = 0
    for r in all_regions:
        center_y = (r.bbox_xyxy[1] + r.bbox_xyxy[3]) / 2
        if center_y < img_h * 0.25:
            ceiling_count += 1
    return ceiling_count == 0, ceiling_count


def _check_equations_inside_whiteboards(
    equations: list,
    whiteboards: list,
) -> tuple[bool, int, int]:
    """Check that every equation center is inside at least one whiteboard bbox."""
    if not whiteboards:
        # No whiteboard detected — can't validate containment
        return True, len(equations), 0

    inside = 0
    for eq in equations:
        eq_cx = (eq.bbox_xyxy[0] + eq.bbox_xyxy[2]) / 2
        eq_cy = (eq.bbox_xyxy[1] + eq.bbox_xyxy[3]) / 2
        for wb in whiteboards:
            wx1, wy1, wx2, wy2 = wb.bbox_xyxy
            if wx1 <= eq_cx <= wx2 and wy1 <= eq_cy <= wy2:
                inside += 1
                break

    return inside == len(equations), len(equations), inside


def _check_latex_parseable(latex: str) -> bool:
    """Try to parse LaTeX with sympy."""
    if not latex.strip():
        return False
    try:
        from sympy.parsing.latex import parse_latex
        parse_latex(latex)
        return True
    except ImportError:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate v2 pipeline on test images"
    )
    parser.add_argument(
        "--model", default=V2_MODEL_PATH,
        help=f"Model checkpoint (default: {V2_MODEL_PATH})",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--roboflow-valid",
        type=Path,
        default=Path("data/raw/roboflow_whiteboard/valid/images"),
        help="Roboflow validation images directory",
    )
    args = parser.parse_args()

    # Discover test images
    test_images = _discover_test_images(args.roboflow_valid)
    if not test_images:
        print("[ERROR] No test images found.")
        sys.exit(1)

    print(f"[PHASE 5] Validating on {len(test_images)} images ...")
    print(f"[PHASE 5] Model: {args.model}")
    print()

    # Load model once
    model = load_yolo_model(args.model)

    # Init OCR engine
    try:
        ocr_engine = MathOCREngine()
    except RuntimeError as e:
        print(f"[WARN] MathOCR unavailable: {e}")
        ocr_engine = None

    # Output directory
    val_dir = Path("output/validation")
    val_dir.mkdir(parents=True, exist_ok=True)

    # Results
    report = []
    total_ceiling_pass = 0
    total_containment_pass = 0
    total_latex_pass = 0
    total_latex_tested = 0

    for img_path in test_images:
        print(f"[PHASE 5] [STEP {test_images.index(img_path) + 1}] "
              f"Testing {img_path.name} ...")

        img = image_file_to_array(img_path)
        img = preprocess_whiteboard(img)
        img_h, img_w = img.shape[:2]

        # Detect
        all_regions = run_inference(model, img, conf=args.conf)
        for r in all_regions:
            r.class_name = V2_CLASS_NAMES.get(r.class_id, r.class_name)

        # Check 1: No ceiling detections (before filtering)
        ceiling_ok, ceiling_count = _check_no_ceiling_detections(all_regions, img_h)
        if ceiling_ok:
            total_ceiling_pass += 1

        # Apply filters (mirrors the infer_v2 pipeline)
        filtered = filter_by_center_y(all_regions, img_h, min_ratio=0.25)
        whiteboards = [r for r in filtered if r.class_id == 1]
        equations = [r for r in filtered if r.class_id == 0]
        equations = merge_nearby_equations(equations, img)
        equations = equations_inside_whiteboards(equations, whiteboards)
        filtered = whiteboards + equations
        filtered.sort(key=lambda r: (r.bbox_xyxy[1], r.bbox_xyxy[0]))

        # Check 2: Equations inside whiteboards
        contain_ok, eq_total, eq_inside = _check_equations_inside_whiteboards(
            equations, whiteboards
        )
        if contain_ok:
            total_containment_pass += 1

        # OCR on equations
        latex_results = []
        latex_texts_overlay = {}
        eq_idx_in_filtered = [
            i for i, r in enumerate(filtered) if r.class_id == 0
        ]

        for i, eq in enumerate(equations):
            if ocr_engine is not None:
                text, conf_ocr, backend = ocr_engine.recognise(eq.crop)
            else:
                text, conf_ocr, backend = "", 0.0, "none"

            latex_results.append({
                "text": text, "confidence": conf_ocr, "backend": backend,
            })

            if i < len(eq_idx_in_filtered):
                latex_texts_overlay[eq_idx_in_filtered[i]] = text

            # Check 3: LaTeX parseable
            if backend == "pix2tex" and text.strip():
                total_latex_tested += 1
                if _check_latex_parseable(text):
                    total_latex_pass += 1
                    latex_results[-1]["sympy_parse"] = "pass"
                else:
                    latex_results[-1]["sympy_parse"] = "fail"

        # Draw annotated image
        vis_path = val_dir / f"{img_path.stem}_val.png"
        draw_detections_v2(img, filtered, latex_texts_overlay, output_path=vis_path)

        # Status
        c_sym = "pass" if ceiling_ok else f"FAIL ({ceiling_count} in ceiling)"
        e_sym = "pass" if contain_ok else f"FAIL ({eq_inside}/{eq_total})"
        print(f"  ceiling: {c_sym}  |  containment: {e_sym}  |  "
              f"wb={len(whiteboards)} eq={len(equations)}")

        report.append({
            "image": img_path.name,
            "image_size": [img_w, img_h],
            "whiteboards": len(whiteboards),
            "equations": len(equations),
            "ceiling_detections": ceiling_count,
            "ceiling_check": "pass" if ceiling_ok else "fail",
            "containment_check": "pass" if contain_ok else "fail",
            "ocr_results": latex_results,
        })

    # --- Summary ---
    print()
    print("=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    print(f"  Images tested     : {len(test_images)}")
    print(f"  Ceiling FP check  : {total_ceiling_pass}/{len(test_images)} passed")
    print(f"  Containment check : {total_containment_pass}/{len(test_images)} passed")
    if total_latex_tested > 0:
        print(f"  LaTeX parse check : {total_latex_pass}/{total_latex_tested} passed")
    else:
        print(f"  LaTeX parse check : n/a (no LaTeX output to validate)")
    print(f"  Results saved to  : {val_dir}/")
    print("=" * 55)

    # Save JSON report
    report_path = val_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[PHASE 5] Report saved to {report_path}")

    # Exit code
    all_passed = (
        total_ceiling_pass == len(test_images) and
        total_containment_pass == len(test_images)
    )
    if all_passed:
        print("[PHASE 5] All validation checks PASSED.")
    else:
        print("[PHASE 5] Some validation checks FAILED. Review output/validation/")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
