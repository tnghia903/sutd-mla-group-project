"""
infer.py — CTCRecogniser: load trained CRNN checkpoint and run inference.

Returns both the decoded LaTeX string and the raw probability matrix
P(c_t | h_t) so the calling notebook can plot the CTC heatmap.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .charset import LatexTokenizer
from .decode import ctc_collapse_sequence, ctc_greedy_decode
from .model import CRNN


def preprocess_for_ctc(
    crop: np.ndarray,
    target_height: int = 32,
    binarize: bool = True,
) -> np.ndarray:
    """
    Preprocess an equation crop for CRNN input.

    Pipeline:
        1. White padding (20 px) — compensates for tight YOLO crops.
        2. CLAHE on LAB luminance — boosts low-contrast marker strokes.
        3. Grayscale conversion.
        4. Otsu binarization (optional) — maps whiteboard photo to clean
           black-on-white, closing the gap with synthetic training images.
        5. Aspect-ratio-preserving resize to target_height.

    Parameters:
        crop:          RGB or grayscale uint8 numpy array.
        target_height: Output height (must match CRNN training height).
        binarize:      Apply Otsu thresholding after CLAHE.  Set True for
                       real whiteboard crops; False for already-clean renders.

    Returns:
        Grayscale uint8 array of shape (target_height, W).
    """
    if crop.size == 0:
        return np.ones((target_height, 32), dtype=np.uint8) * 255

    # Ensure RGB
    if crop.ndim == 2:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        rgb = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    else:
        rgb = crop.copy()

    # Step 1: White padding
    pad = 20
    padded = cv2.copyMakeBorder(
        rgb, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )

    # Step 2: CLAHE on LAB luminance
    lab = cv2.cvtColor(padded, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 3: Grayscale
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    # Step 4: Otsu binarization — stroke pixels → 0, background → 255
    if binarize:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Resize to target_height preserving aspect ratio
    h, w = gray.shape
    scale = target_height / h
    new_w = max(16, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, target_height), interpolation=interp)

    return resized


def _image_to_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W) uint8 grayscale to (1, 1, H, W) float tensor in [0,1]."""
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)


class CTCRecogniser:
    """
    Thin wrapper around a trained CRNN checkpoint.

    Usage::

        rec = CTCRecogniser("path/to/ctc_best.pt")
        latex, probs, greedy_indices = rec.recognise(crop)
        # probs: (T, num_classes) float32 numpy — P(c_t | h_t)
        # greedy_indices: (T,) int32 numpy — argmax path π
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self._device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self._device = torch.device(device)

        ckpt = torch.load(str(checkpoint_path), map_location=self._device, weights_only=True)
        num_classes  = ckpt.get("num_classes", 174)
        hidden_size  = ckpt.get("hidden_size", 256)

        self._model = CRNN(num_classes=num_classes, hidden_size=hidden_size)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(self._device)
        self._model.eval()

        self._tokenizer = LatexTokenizer()
        self._idx_to_char: dict[int, str] = {
            i: self._tokenizer.decode([i]) for i in range(num_classes)
        }

        epoch = ckpt.get("epoch", "?")
        cer   = ckpt.get("val_cer", float("nan"))
        print(f"[CTCRecogniser] Loaded checkpoint (epoch={epoch}, val_CER={cer:.4f})")
        print(f"[CTCRecogniser] Device: {self._device}")

    @torch.no_grad()
    def recognise(
        self,
        crop: np.ndarray,
        target_height: int = 32,
        use_beam: bool = False,
        beam_width: int = 5,
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Run the full recognition pipeline on an equation crop.

        Parameters:
            crop:          RGB (or grayscale) uint8 equation image.
            target_height: Must match training height (default 32).
            use_beam:      If True, use beam search instead of greedy.
            beam_width:    Beam width (only used when use_beam=True).

        Returns:
            (latex_str, probs, greedy_indices)
            - latex_str:      Decoded LaTeX string.
            - probs:          (T, num_classes) float32 ndarray — P(c_t | h_t).
            - greedy_indices: (T,) int32 ndarray — raw argmax path π.
        """
        gray = preprocess_for_ctc(crop, target_height=target_height)
        tensor = _image_to_tensor(gray, self._device)

        # Forward pass
        probs = self._model.get_probs(tensor)   # (1, T, C)
        probs_np = probs.squeeze(0).cpu().numpy()  # (T, C) float32

        # Greedy path
        greedy_indices = np.argmax(probs_np, axis=1)  # (T,)

        if use_beam:
            from .decode import ctc_beam_search
            results = ctc_beam_search(
                probs_np,
                blank_idx=self._tokenizer.blank_idx,
                beam_width=beam_width,
                idx_to_char=self._idx_to_char,
            )
            latex_str = results[0][0] if results else ""
        else:
            collapsed = ctc_collapse_sequence(
                greedy_indices,
                blank_idx=self._tokenizer.blank_idx,
            )
            latex_str = self._tokenizer.decode(collapsed)

        return latex_str, probs_np, greedy_indices

    @property
    def tokenizer(self) -> LatexTokenizer:
        return self._tokenizer

    @property
    def idx_to_char(self) -> dict[int, str]:
        return self._idx_to_char


def results_to_markdown(
    equations: list,
    latex_results: list[tuple[str, str]],
) -> str:
    """
    Format pipeline results as Markdown with $$LaTeX$$ equation blocks.

    Parameters:
        equations:     List of CroppedRegion instances with bbox_xyxy.
        latex_results: List of (ctc_latex, fallback_latex) per equation.

    Returns:
        Markdown string.
    """
    lines = ["# Whiteboard Digitisation — Results\n"]
    for i, (region, (ctc_latex, fallback_latex)) in enumerate(
        zip(equations, latex_results), start=1
    ):
        x1, y1, x2, y2 = region.bbox_xyxy
        lines.append(f"## Equation {i}")
        lines.append(f"**Bounding box**: ({x1}, {y1}) → ({x2}, {y2})\n")

        if ctc_latex.strip():
            lines.append("**CTC output** (trained CRNN):")
            lines.append(f"$$\n{ctc_latex}\n$$\n")
        else:
            lines.append("*CTC: out-of-distribution — using pix2tex fallback*\n")

        if fallback_latex and fallback_latex.strip():
            lines.append("**pix2tex fallback**:")
            lines.append(f"$$\n{fallback_latex}\n$$\n")

    return "\n".join(lines)
