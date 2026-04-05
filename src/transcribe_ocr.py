"""
sequence_transcriber.py — PP-OCRv3 / CRNN Sequence Recognition for Whiteboard Crops
=====================================================================================
Module Owner: [Team Member C — Student ID: XXXXXXX]

Purpose (Advanced Module):
    1. Accepts cropped numpy arrays of *Handwriting* and *Equation* regions
       produced by `layout_detector.py`.
    2. Passes each crop through the PP-OCRv3 recognition pipeline
       (detection → recognition, leveraging a CRNN backbone with CTC loss).
    3. Returns structured transcription results per region.

    Whiteboard-specific challenges addressed:
      - Low-contrast marker ink on reflective surfaces
      - Uneven lighting and glare from camera capture
      - Variable handwriting styles and orientations

Theoretical Background — CTC Loss:
    The CRNN encoder emits a probability matrix Y ∈ ℝ^{T × |Σ'|}
    where T is the number of time-steps, Σ' = Σ ∪ {blank}.

    CTC marginalises over all valid alignments π ∈ B^{-1}(l):

        P(l | Y) = Σ_{π ∈ B^{-1}(l)}  Π_{τ=1}^{T}  y_{π_τ}^τ

    The loss is:
        L_CTC = − ln P(l | Y)

    This module delegates the CTC-decoded inference to PaddleOCR's
    pre-trained PP-OCRv3 model (or falls back to EasyOCR/Tesseract).

Dependencies:
    pip install paddlepaddle paddleocr numpy opencv-python-headless
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TranscriptionResult:
    """Structured output from the OCR stage."""
    region_class:  str             # e.g. "Handwriting", "Equation"
    bbox_xyxy:     tuple[int, int, int, int]
    text:          str             # Decoded character sequence
    confidence:    float           # Average recognition confidence
    line_count:    int             # Number of text lines detected


# ---------------------------------------------------------------------------
# 2. OCR ENGINE ABSTRACTION
# ---------------------------------------------------------------------------

class OCREngine:
    """
    Thin wrapper around PaddleOCR's PP-OCRv3 model.

    PP-OCRv3 architecture (per the PaddlePaddle paper):
        Text Detection  → DB   (Differentiable Binarisation)
        Text Recognition → SVTR-LCNet backbone + CTC head

    Falls back to easyocr if PaddleOCR is unavailable.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self._engine = None
        self._backend: str = "none"
        self._lang = lang
        self._device = "gpu" if use_gpu else "cpu"
        self._initialise()

    # --- Private initialiser with fallback chain ---
    def _initialise(self) -> None:
        # Attempt 1: PaddleOCR (preferred)
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                use_textline_orientation=True,
                lang=self._lang,
                device=self._device,
            )
            self._backend = "paddleocr"
            print("[INFO] OCR backend: PaddleOCR (PP-OCRv3)")
            return
        except ImportError:
            print("[WARN] PaddleOCR not installed. Trying EasyOCR …")

        # Attempt 2: EasyOCR (fallback)
        try:
            import easyocr
            self._engine = easyocr.Reader(
                [self._lang], gpu=(self._device == "gpu"), verbose=False
            )
            self._backend = "easyocr"
            print("[INFO] OCR backend: EasyOCR")
            return
        except ImportError:
            print("[WARN] EasyOCR not installed. Trying Tesseract …")

        # Attempt 3: Tesseract (last resort)
        try:
            import pytesseract  # noqa: F401
            self._backend = "tesseract"
            print("[INFO] OCR backend: Tesseract")
            return
        except ImportError:
            raise RuntimeError(
                "No OCR backend available. Install one of: "
                "paddleocr, easyocr, pytesseract"
            )

    # --- Public transcription interface ---
    def recognise(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Run text recognition on a single cropped whiteboard region.

        Parameters:
            crop: RGB numpy array (H×W×3), uint8.

        Returns:
            (decoded_text, avg_confidence)
        """
        if crop.size == 0:
            return ("", 0.0)

        if self._backend == "paddleocr":
            return self._recognise_paddle(crop)
        elif self._backend == "easyocr":
            return self._recognise_easyocr(crop)
        elif self._backend == "tesseract":
            return self._recognise_tesseract(crop)
        else:
            return ("", 0.0)

    # ---- PaddleOCR ----
    def _recognise_paddle(self, crop: np.ndarray) -> tuple[str, float]:
        # PaddleOCR expects BGR
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        results = self._engine.predict(bgr)

        if not results:
            return ("", 0.0)

        # PaddleOCR 3.4+ returns list[OCRResult] with rec_texts/rec_scores
        item = results[0]
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])

        if not texts:
            return ("", 0.0)

        full_text = "\n".join(texts)
        avg_conf  = sum(scores) / len(scores) if scores else 0.0
        return (full_text, avg_conf)

    # ---- EasyOCR ----
    def _recognise_easyocr(self, crop: np.ndarray) -> tuple[str, float]:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        results = self._engine.readtext(bgr)

        if not results:
            return ("", 0.0)

        lines = [r[1] for r in results]
        confs = [r[2] for r in results]

        return ("\n".join(lines), sum(confs) / len(confs))

    # ---- Tesseract ----
    def _recognise_tesseract(self, crop: np.ndarray) -> tuple[str, float]:
        import pytesseract
        text = pytesseract.image_to_string(crop)
        # Tesseract doesn't return per-line confidence easily
        return (text.strip(), 0.85)  # [Inference] static placeholder


# ---------------------------------------------------------------------------
# 3. BATCH TRANSCRIPTION
# ---------------------------------------------------------------------------

def transcribe_regions(
    regions: list,  # list[layout_detector.CroppedRegion]
    engine: Optional[OCREngine] = None,
    target_classes: Optional[set[str]] = None,
) -> list[TranscriptionResult]:
    """
    Transcribe all text-bearing whiteboard regions in batch.

    Parameters:
        regions:        Output from layout_detector.run_inference().
        engine:         Pre-initialised OCREngine (created if None).
        target_classes: Set of class names to transcribe.
                        Default: {"Handwriting", "Equation"}.

    Returns:
        List[TranscriptionResult] preserving the input ordering.
    """
    if target_classes is None:
        target_classes = {"Handwriting", "Equation"}

    if engine is None:
        engine = OCREngine()

    results: list[TranscriptionResult] = []

    for region in regions:
        if region.class_name not in target_classes:
            # Non-text regions (Diagram, Arrow, Sticky Note) are passed
            # through with empty text for downstream handling.
            results.append(TranscriptionResult(
                region_class = region.class_name,
                bbox_xyxy    = region.bbox_xyxy,
                text         = "",
                confidence   = region.confidence,
                line_count   = 0,
            ))
            continue

        # --- Whiteboard-specific preprocessing ---
        preprocessed = _preprocess_whiteboard_crop(region.crop)

        text, conf = engine.recognise(preprocessed)
        line_count = text.count("\n") + 1 if text else 0

        results.append(TranscriptionResult(
            region_class = region.class_name,
            bbox_xyxy    = region.bbox_xyxy,
            text         = text,
            confidence   = conf,
            line_count   = line_count,
        ))

        print(f"  [OCR] {region.class_name}  "
              f"lines={line_count}  conf={conf:.3f}  "
              f"chars={len(text)}")

    return results


# ---------------------------------------------------------------------------
# 4. WHITEBOARD-SPECIFIC IMAGE PREPROCESSING
# ---------------------------------------------------------------------------

def _preprocess_whiteboard_crop(
    crop: np.ndarray,
    target_height: int = 64,
) -> np.ndarray:
    """
    Whiteboard-optimised preprocessing to improve OCR accuracy:
        1. Convert to greyscale.
        2. CLAHE contrast enhancement (handles uneven whiteboard lighting).
        3. Morphological opening to reduce noise from board texture.
        4. Adaptive thresholding (Gaussian) to binarise marker strokes.
        5. Deskew correction for tilted handwriting.
        6. Resize to a fixed height while preserving aspect ratio.
        7. Convert back to 3-channel for model compatibility.

    Parameters:
        crop:          RGB numpy array from a whiteboard region.
        target_height: Fixed height for the normalised output.

    Returns:
        Preprocessed RGB numpy array.
    """
    if crop.ndim == 3:
        grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        grey = crop

    # --- Step 1: CLAHE contrast enhancement ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grey = clahe.apply(grey)

    # --- Step 2: Morphological opening (reduce board texture noise) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)

    # --- Step 3: Adaptive thresholding for binarisation ---
    binary = cv2.adaptiveThreshold(
        grey, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=4,
    )

    # --- Step 4: Deskew correction ---
    binary = _deskew(binary)

    # --- Step 5: Resize to target height, preserve aspect ratio ---
    h, w = binary.shape[:2]
    if h > 0:
        scale = target_height / h
        new_w = max(1, int(w * scale))
        binary = cv2.resize(binary, (new_w, target_height),
                            interpolation=cv2.INTER_CUBIC)

    # Back to 3-channel RGB
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def _deskew(binary_image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Correct slight rotation in handwritten text by estimating skew angle
    from the binarised image using horizontal projection analysis.

    Parameters:
        binary_image: Binarised (white bg, dark text) numpy array.
        max_angle:    Maximum correction angle in degrees.

    Returns:
        Deskewed binary image.
    """
    # Find non-zero pixels
    coords = np.column_stack(np.where(binary_image < 128))
    if len(coords) < 20:
        return binary_image  # too few points to estimate angle

    # Fit minimum-area bounding rectangle
    try:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
    except cv2.error:
        return binary_image

    # Normalise angle to [-max_angle, max_angle]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return binary_image  # don't correct extreme angles

    # Rotate
    h, w = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ---------------------------------------------------------------------------
# 5. STANDALONE EXECUTION (DEMO)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("sequence_transcriber.py — Standalone Demo (Whiteboard)")
    print("=" * 60)

    # Create a synthetic test crop (whiteboard-like: off-white bg, dark marker)
    test_crop = np.ones((64, 320, 3), dtype=np.uint8) * 240
    cv2.putText(test_crop, "Hello MLA 42.515", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 80), 2)

    engine = OCREngine()
    text, conf = engine.recognise(test_crop)
    print(f"\nRecognised: '{text}'  (conf={conf:.3f})")
