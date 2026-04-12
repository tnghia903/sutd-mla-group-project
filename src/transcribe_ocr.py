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

def _preprocess_whiteboard_crop(crop: np.ndarray) -> np.ndarray:
    """
    Whiteboard-optimised preprocessing to improve OCR accuracy.

    Applies CLAHE contrast enhancement on the luminance channel to handle
    uneven whiteboard lighting and glare from camera capture.  The crop is
    returned at its original resolution so that PaddleOCR's internal text
    detector operates at full quality; heavy binarisation and forced rescaling
    to 64 px were removed because they shrink large crops to unusable sizes.

    Parameters:
        crop: RGB numpy array from a whiteboard region.

    Returns:
        Contrast-enhanced RGB numpy array (same spatial dimensions as input).
    """
    if crop.size == 0:
        return crop

    # CLAHE on the L channel of LAB (preserves colour, boosts local contrast)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _preprocess_math_crop(crop: np.ndarray) -> np.ndarray:
    """
    Math-specific preprocessing for equation crops before OCR.

    Pipeline:
        1. Convert to grayscale
        2. CLAHE contrast enhancement (clipLimit=3.0)
        3. Otsu binarization only if the image is noisy (stddev > 40)
        4. Upscale to minimum 320px height (preserves glyph detail)
        5. Convert back to 3-channel RGB for OCR engines

    Parameters:
        crop: RGB numpy array from an equation region.

    Returns:
        Preprocessed RGB numpy array.
    """
    if crop.size == 0:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # CLAHE for low-contrast chalk/marker strokes
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu binarization only if noisy
    if np.std(enhanced) > 40:
        _, enhanced = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Upscale to minimum 320px height
    h, w = enhanced.shape[:2]
    if h < 320:
        scale = 320 / h
        enhanced = cv2.resize(
            enhanced, (int(w * scale), 320), interpolation=cv2.INTER_LINEAR
        )

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# 5. MATH-SPECIFIC OCR ENGINE (pix2tex -> TrOCR -> PaddleOCR)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MathTranscriptionResult:
    """Structured output from the math OCR stage."""
    region_class: str
    bbox_xyxy:    tuple[int, int, int, int]
    latex:        str             # LaTeX string
    plain_text:   str             # Plain text fallback
    confidence:   float
    backend:      str             # Which engine produced the result
    line_count:   int


class MathOCREngine:
    """
    Math-aware OCR engine with fallback chain:
        1. pix2tex (LatexOCR) — trained on im2latex, outputs LaTeX directly
        2. TrOCR (microsoft/trocr-base-handwritten) — handwriting transformer
        3. PaddleOCR — last resort for any remaining text

    Usage:
        engine = MathOCREngine()
        latex, confidence, backend = engine.recognise(crop)
    """

    def __init__(self, use_gpu: bool = False):
        self._engines: list[tuple[str, object]] = []
        self._use_gpu = use_gpu
        self._init_pix2tex()
        self._init_trocr()
        self._init_paddle_fallback()
        if not self._engines:
            raise RuntimeError(
                "No math OCR backend available. Install one of: "
                "pix2tex, transformers (for TrOCR), paddleocr"
            )
        names = [name for name, _ in self._engines]
        print(f"[INFO] MathOCR backends: {' -> '.join(names)}")

    # --- Backend initialisation ---

    def _init_pix2tex(self) -> None:
        try:
            from pix2tex.cli import LatexOCR
            self._latex_ocr = LatexOCR()
            self._engines.append(("pix2tex", self._recognise_pix2tex))
            print("[INFO] MathOCR: pix2tex (LatexOCR) loaded.")
        except ImportError:
            print("[WARN] pix2tex not installed. pip install 'pix2tex[gui]'")
        except Exception as e:
            print(f"[WARN] pix2tex init failed: {e}")

    def _init_trocr(self) -> None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            model_name = "microsoft/trocr-base-handwritten"
            self._trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self._engines.append(("trocr", self._recognise_trocr))
            print("[INFO] MathOCR: TrOCR (handwritten) loaded.")
        except ImportError:
            print("[WARN] transformers not installed for TrOCR. pip install transformers")
        except Exception as e:
            print(f"[WARN] TrOCR init failed: {e}")

    def _init_paddle_fallback(self) -> None:
        try:
            self._paddle_engine = OCREngine(use_gpu=self._use_gpu)
            self._engines.append(("paddleocr", self._recognise_paddle))
        except RuntimeError:
            pass

    # --- Public interface ---

    def recognise(self, crop: np.ndarray) -> tuple[str, float, str]:
        """
        Run math OCR on a single equation crop with fallback chain.

        Parameters:
            crop: RGB numpy array (H x W x 3), uint8.

        Returns:
            (text, confidence, backend_name)
        """
        if crop.size == 0:
            return ("", 0.0, "none")

        preprocessed = _preprocess_math_crop(crop)

        for name, fn in self._engines:
            try:
                text, conf = fn(preprocessed)
                if text.strip() and conf > 0.3:
                    return (text.strip(), conf, name)
            except Exception as e:
                print(f"[WARN] MathOCR {name} failed: {e}")
                continue

        return ("", 0.0, "none")

    # --- Individual backends ---

    def _recognise_pix2tex(self, crop: np.ndarray) -> tuple[str, float]:
        from PIL import Image as PILImage
        # pix2tex expects a PIL Image
        pil_img = PILImage.fromarray(crop)
        latex = self._latex_ocr(pil_img)
        # pix2tex does not return confidence; use 0.85 as default
        return (latex, 0.85)

    def _recognise_trocr(self, crop: np.ndarray) -> tuple[str, float]:
        import torch
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(crop)
        pixel_values = self._trocr_processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values
        with torch.no_grad():
            outputs = self._trocr_model.generate(
                pixel_values, max_new_tokens=128
            )
        text = self._trocr_processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        # TrOCR returns plain text, not LaTeX
        return (text, 0.75)

    def _recognise_paddle(self, crop: np.ndarray) -> tuple[str, float]:
        return self._paddle_engine.recognise(crop)


# ---------------------------------------------------------------------------
# 6. BATCH MATH TRANSCRIPTION
# ---------------------------------------------------------------------------

def transcribe_math_regions(
    regions: list,
    engine: Optional[MathOCREngine] = None,
) -> list[MathTranscriptionResult]:
    """
    Transcribe equation regions using the math-specific OCR engine.

    Parameters:
        regions: Output from layout_detector.run_inference() filtered
                 to equation-class regions only.
        engine:  Pre-initialised MathOCREngine (created if None).

    Returns:
        List[MathTranscriptionResult] preserving input ordering.
    """
    if engine is None:
        engine = MathOCREngine()

    results: list[MathTranscriptionResult] = []

    for region in regions:
        text, conf, backend = engine.recognise(region.crop)
        line_count = text.count("\n") + 1 if text else 0

        results.append(MathTranscriptionResult(
            region_class=region.class_name,
            bbox_xyxy=region.bbox_xyxy,
            latex=text if backend == "pix2tex" else "",
            plain_text=text if backend != "pix2tex" else "",
            confidence=conf,
            backend=backend,
            line_count=line_count,
        ))

        print(f"  [MathOCR] {backend}  conf={conf:.3f}  "
              f"chars={len(text)}  text={text[:60]}...")

    return results



# ---------------------------------------------------------------------------
# 7. STANDALONE EXECUTION (DEMO)
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
