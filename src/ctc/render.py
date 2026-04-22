"""
render.py — Synthetic LaTeX image renderer and MathWriting InkML adapter.

Two data sources for CRNN+CTC training:
  1. Synthetic: render LaTeX strings as rasterised images via matplotlib
     mathtext.  Fast, unlimited quantity, controlled vocabulary.
  2. Real (InkML): parse MathWriting-2024 stroke files, rasterise via
     OpenCV polylines.  Provides handwriting style variation.
"""

from __future__ import annotations

import io
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# matplotlib is heavy — import lazily
_plt = None
_fig_cache: dict = {}


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# ---------------------------------------------------------------------------
# 1. Synthetic LaTeX renderer
# ---------------------------------------------------------------------------

def render_latex_to_image(
    latex: str,
    target_height: int = 32,
    dpi: int = 150,
    augment: bool = True,
) -> np.ndarray | None:
    """
    Render a LaTeX math string to a grayscale numpy array of height=32.

    Uses matplotlib's mathtext renderer — no full LaTeX installation needed.
    The formula is rendered as $<latex>$ on a white background.

    Parameters:
        latex:         LaTeX string (without surrounding $).
        target_height: Output height in pixels (width varies).
        dpi:           Rendering DPI (higher = more detail, slower).
        augment:       If True, apply random photometric augmentations.

    Returns:
        Grayscale uint8 numpy array of shape (target_height, W), or None if
        rendering fails (e.g. unsupported LaTeX command).
    """
    plt = _get_plt()

    try:
        fig = plt.figure(figsize=(12, 1), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # Wrap in math mode delimiters
        display_str = f"${latex}$"
        text_obj = ax.text(
            0.5, 0.5, display_str,
            ha="center", va="center",
            fontsize=18, color="black",
            transform=ax.transAxes,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
        plt.close(fig)
        buf.seek(0)

        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Crop tight bounding box of non-white pixels
        mask = gray < 250
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        pad = 4
        r_min = max(0, r_min - pad)
        r_max = min(gray.shape[0], r_max + pad)
        c_min = max(0, c_min - pad)
        c_max = min(gray.shape[1], c_max + pad)
        cropped = gray[r_min:r_max+1, c_min:c_max+1]

        if cropped.size == 0 or cropped.shape[0] < 4 or cropped.shape[1] < 4:
            return None

        # Resize to target_height preserving aspect ratio
        h, w = cropped.shape
        scale = target_height / h
        new_w = max(8, round(w * scale))
        resized = cv2.resize(cropped, (new_w, target_height),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

        if augment:
            resized = _augment_image(resized)

        return resized

    except Exception:
        return None


def _augment_image(img: np.ndarray) -> np.ndarray:
    """Apply mild photometric augmentations to a grayscale image."""
    # Random brightness shift
    if random.random() < 0.5:
        delta = random.randint(-20, 20)
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Random Gaussian blur (simulates camera defocus)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Random slight rotation ±3°
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)

    return img


# ---------------------------------------------------------------------------
# 2. Synthetic LaTeX expression corpus
# ---------------------------------------------------------------------------

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_VARS = list("xyzabcnm")
_GREEK = [r"\alpha", r"\beta", r"\gamma", r"\theta", r"\lambda",
          r"\mu", r"\pi", r"\sigma", r"\phi", r"\omega", r"\varepsilon"]
_TRIG = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln"]


def _rand_var() -> str:
    return random.choice(_VARS + _GREEK)


def _rand_num(lo: int = 1, hi: int = 9) -> str:
    return str(random.randint(lo, hi))


def _rand_expr(depth: int = 0) -> str:
    """Recursively generate a random short LaTeX expression."""
    if depth >= 2:
        # Leaf: variable or number
        return random.choice([_rand_var(), _rand_num()])

    kind = random.random()
    if kind < 0.20:
        # fraction
        num = _rand_expr(depth + 1)
        den = _rand_expr(depth + 1)
        return rf"\frac{{{num}}}{{{den}}}"
    elif kind < 0.35:
        # power
        base = _rand_var()
        exp = random.choice([_rand_num(), _rand_var(), "2", "n", "-1"])
        return f"{base}^{{{exp}}}"
    elif kind < 0.45:
        # subscript
        base = _rand_var()
        sub = random.choice([_rand_num(0, 4), _rand_var()])
        return f"{base}_{{{sub}}}"
    elif kind < 0.55:
        # sqrt
        inner = _rand_expr(depth + 1)
        return rf"\sqrt{{{inner}}}"
    elif kind < 0.62:
        # sum or product
        op = random.choice([r"\sum", r"\prod"])
        var = _rand_var()
        lo = random.choice(["0", "1"])
        hi = random.choice(["n", "N", r"\infty"])
        return rf"{op}_{{i={lo}}}^{{{hi}}} {var}"
    elif kind < 0.67:
        # trig function
        fn = random.choice(_TRIG)
        arg = _rand_var()
        return rf"{fn}({arg})"
    else:
        # simple arithmetic
        left = _rand_expr(depth + 1)
        op = random.choice(["+", "-", "=", r"\cdot"])
        right = _rand_expr(depth + 1)
        return f"{left} {op} {right}"


_TEMPLATES = [
    # Linear / polynomial
    lambda: f"{_rand_var()} = {_rand_num()}{_rand_var()} + {_rand_num()}",
    lambda: f"{_rand_var()} = {_rand_num()}{_rand_var()}^2 + {_rand_num()}{_rand_var()} + {_rand_num()}",
    # Fractions
    lambda: rf"\frac{{{_rand_expr()}}}{{{_rand_expr()}}}",
    lambda: rf"{_rand_var()} = \frac{{{_rand_expr()}}}{{{_rand_expr()}}}",
    # Equations with equals
    lambda: f"{_rand_expr()} = {_rand_expr()}",
    # Bayes-like
    lambda: (
        r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}"
    ),
    lambda: (
        r"P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}"
    ),
    # Powers and indices
    lambda: f"{_rand_var()}^{{{_rand_num()}}} + {_rand_var()}^{{{_rand_num()}}}",
    lambda: f"e^{{{_rand_var()}}} = \sum_{{n=0}}^{{\infty}} \\frac{{{_rand_var()}^n}}{{n!}}",
    # Simple numbers / algebra
    lambda: f"{_rand_num()} + {_rand_num()} = {_rand_num()}",
    lambda: f"{_rand_num()} \\times {_rand_num()} = {_rand_num(1, 81)}",
    # Trig identities
    lambda: r"\sin^2(\theta) + \cos^2(\theta) = 1",
    lambda: rf"\sin({_rand_var()}) = \cos(\frac{{\pi}}{{2}} - {_rand_var()})",
    # Integrals
    lambda: rf"\int_{{0}}^{{{_rand_num()}}} {_rand_var()} \, d{_rand_var()}",
    lambda: rf"\int {_rand_var()}^{{{_rand_num()}}} \, d{_rand_var()} = \frac{{{_rand_var()}^{{{_rand_num()+1}}}}}{{{_rand_num()+1}}}",
    # Limits
    lambda: r"\lim_{x \to 0} f(x)",
    # Random composed expression
    lambda: _rand_expr(0),
    lambda: _rand_expr(0),
    lambda: _rand_expr(0),
]


class LatexCorpus:
    """
    Generator of (image, latex) training pairs from synthetic templates.

    Usage::

        corpus = LatexCorpus(n_train=30000, n_val=2000, seed=42)
        for img, latex in corpus.train():
            ...   # img: (32, W) grayscale uint8, latex: str
    """

    def __init__(
        self,
        n_train: int = 30_000,
        n_val: int = 2_000,
        target_height: int = 32,
        seed: int = 42,
    ) -> None:
        self.n_train = n_train
        self.n_val = n_val
        self.target_height = target_height
        self.seed = seed

    def _sample_latex(self, rng: random.Random) -> str:
        tmpl = rng.choice(_TEMPLATES)
        return tmpl()

    def _stream(
        self, n: int, rng: random.Random, augment: bool
    ) -> Iterator[tuple[np.ndarray, str]]:
        count = 0
        attempts = 0
        while count < n:
            attempts += 1
            if attempts > n * 5:
                break  # give up to avoid infinite loop
            latex = self._sample_latex(rng)
            img = render_latex_to_image(
                latex, target_height=self.target_height, augment=augment
            )
            if img is not None:
                yield img, latex
                count += 1

    def train(self) -> Iterator[tuple[np.ndarray, str]]:
        rng = random.Random(self.seed)
        return self._stream(self.n_train, rng, augment=True)

    def val(self) -> Iterator[tuple[np.ndarray, str]]:
        rng = random.Random(self.seed + 1)
        return self._stream(self.n_val, rng, augment=False)


# ---------------------------------------------------------------------------
# 3. MathWriting InkML adapter
# ---------------------------------------------------------------------------

_INKML_NS = "http://www.w3.org/2003/InkML"


def inkml_to_image(
    path: str | Path,
    target_height: int = 32,
    stroke_width: int = 2,
    augment: bool = True,
) -> tuple[np.ndarray, str] | None:
    """
    Parse a MathWriting InkML file and rasterise the ink strokes to an image.

    Coordinate normalisation:
        All traces are collected, bounding-boxed, and scaled so the total
        ink height maps to target_height − 4 px (2 px margin each side).
        Width scales proportionally.

    Parameters:
        path:          Path to an .inkml file.
        target_height: Output image height.
        stroke_width:  Pen radius for polylines.
        augment:       If True, apply mild augmentations.

    Returns:
        (image, normalised_label) or None on parse failure.
    """
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()

        # Extract normalised LaTeX label
        label = ""
        for ann in root.findall(f"{{{_INKML_NS}}}annotation"):
            if ann.get("type") == "normalizedLabel":
                label = (ann.text or "").strip()
                break
        if not label:
            for ann in root.findall(f"{{{_INKML_NS}}}annotation"):
                if ann.get("type") == "label":
                    label = (ann.text or "").strip()
                    break

        if not label:
            return None

        # Parse all traces (X, Y coordinate pairs; ignore T)
        all_points: list[np.ndarray] = []
        for trace in root.findall(f"{{{_INKML_NS}}}trace"):
            raw = (trace.text or "").strip()
            if not raw:
                continue
            pts = []
            for point_str in raw.split(","):
                coords = point_str.strip().split()
                if len(coords) >= 2:
                    try:
                        pts.append([float(coords[0]), float(coords[1])])
                    except ValueError:
                        continue
            if pts:
                all_points.append(np.array(pts, dtype=np.float32))

        if not all_points:
            return None

        # Compute global bounding box
        all_xy = np.vstack(all_points)
        x_min, y_min = all_xy.min(axis=0)
        x_max, y_max = all_xy.max(axis=0)
        ink_h = y_max - y_min
        ink_w = x_max - x_min

        if ink_h < 1 or ink_w < 1:
            return None

        margin = 2
        canvas_h = target_height
        scale = (canvas_h - 2 * margin) / ink_h
        canvas_w = max(16, int(ink_w * scale) + 2 * margin)

        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

        for pts in all_points:
            # Normalise coordinates to canvas space
            px = ((pts[:, 0] - x_min) * scale + margin).astype(np.int32)
            py = ((pts[:, 1] - y_min) * scale + margin).astype(np.int32)
            px = np.clip(px, 0, canvas_w - 1)
            py = np.clip(py, 0, canvas_h - 1)
            poly = np.stack([px, py], axis=1).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], isClosed=False,
                          color=0, thickness=stroke_width)

        if augment:
            canvas = _augment_image(canvas)

        return canvas, label

    except Exception:
        return None


def load_inkml_dataset(
    inkml_dir: str | Path,
    target_height: int = 32,
    augment: bool = True,
) -> list[tuple[np.ndarray, str]]:
    """
    Load all InkML files from a directory as (image, label) pairs.

    Parameters:
        inkml_dir:     Directory containing .inkml files.
        target_height: Image height.
        augment:       Whether to apply augmentations.

    Returns:
        List of (grayscale_image, latex_label) tuples.
    """
    pairs: list[tuple[np.ndarray, str]] = []
    for p in sorted(Path(inkml_dir).glob("*.inkml")):
        result = inkml_to_image(p, target_height=target_height, augment=augment)
        if result is not None:
            pairs.append(result)
    return pairs
