"""
test_math_ocr.py -- Compare math OCR engines on MathWriting validation crops
=============================================================================
Runs pix2tex, TrOCR, and PaddleOCR on 5 sample equation images and prints
a side-by-side comparison. Optionally validates LaTeX output via sympy.

Usage:
    python tests/test_math_ocr.py
    python tests/test_math_ocr.py --samples 10
    python tests/test_math_ocr.py --image path/to/single_crop.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transcribe_ocr import MathOCREngine, _preprocess_math_crop


def _try_sympy_parse(latex: str) -> bool:
    """Attempt to parse a LaTeX string with sympy. Returns True on success."""
    if not latex.strip():
        return False
    try:
        from sympy.parsing.latex import parse_latex
        parse_latex(latex)
        return True
    except ImportError:
        print("[WARN] sympy not installed -- skipping LaTeX parse validation.")
        return False
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test math OCR engines on equation crops"
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Test a single image instead of sampling from validation set",
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Number of validation images to test (default: 5)",
    )
    parser.add_argument(
        "--mathwriting-dir", type=Path,
        default=Path("data/raw/mathwriting"),
        help="MathWriting dataset root",
    )
    args = parser.parse_args()

    # Collect test images
    if args.image:
        if not args.image.exists():
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        test_images = [args.image]
    else:
        val_dir = args.mathwriting_dir / "valid" / "images"
        if not val_dir.exists():
            print(f"[ERROR] Validation directory not found: {val_dir}")
            sys.exit(1)
        all_imgs = sorted(
            p for p in val_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        test_images = all_imgs[:args.samples]

    if not test_images:
        print("[ERROR] No test images found.")
        sys.exit(1)

    print(f"[PHASE 3] [STEP 4] Testing {len(test_images)} images ...")
    print()

    # Init engine (loads all available backends)
    engine = MathOCREngine()
    print()

    # Header
    print(f"{'Image':>40s} | {'Backend':>10s} | {'Conf':>5s} | {'sympy':>5s} | Output")
    print("-" * 110)

    passed = 0
    total = 0

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{'[skip] ' + img_path.name:>40s} | could not read image")
            continue

        crop_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text, conf, backend = engine.recognise(crop_rgb)

        parseable = _try_sympy_parse(text) if backend == "pix2tex" else False
        total += 1
        if parseable:
            passed += 1

        sympy_str = "pass" if parseable else "fail" if text.strip() else "n/a"
        display_text = text.replace("\n", " ")[:50]

        print(f"{img_path.name:>40s} | {backend:>10s} | {conf:>5.2f} | "
              f"{sympy_str:>5s} | {display_text}")

    print("-" * 110)
    print(f"\n[PHASE 3] [STEP 4] sympy parse: {passed}/{total} passed")


if __name__ == "__main__":
    main()
