"""
charset.py — Vocabulary and token-level tokeniser for the CRNN+CTC model.

Token-level (not character-level): each LaTeX control sequence like \\frac
is treated as a single token so the model learns LaTeX syntax, not character
soup.  The blank token ε is always at index 0 (CTC convention).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Index 0 is reserved for the CTC blank token (ε).
# All other tokens are ordered: digits → lowercase → uppercase → operators →
# LaTeX control sequences.
BLANK_TOKEN = "ε"

_DIGITS     = list("0123456789")
_LOWER      = list("abcdefghijklmnopqrstuvwxyz")
_UPPER      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_OPERATORS  = list("+-=()[]{}.,;:^_|/\\!? ")
_LATEX_CMDS = [
    r"\frac", r"\sqrt", r"\sum", r"\int", r"\prod",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon",
    r"\theta", r"\lambda", r"\mu", r"\nu", r"\pi", r"\rho", r"\sigma",
    r"\tau", r"\phi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Pi", r"\Sigma",
    r"\Phi", r"\Psi", r"\Omega",
    r"\log", r"\ln", r"\sin", r"\cos", r"\tan",
    r"\lim", r"\inf", r"\sup", r"\max", r"\min",
    r"\partial", r"\nabla", r"\infty", r"\pm", r"\mp",
    r"\times", r"\div", r"\cdot", r"\circ",
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv",
    r"\in", r"\notin", r"\subset", r"\supset",
    r"\cup", r"\cap", r"\forall", r"\exists",
    r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\Leftrightarrow",
    r"\ldots", r"\cdots", r"\vdots",
    r"\text",
]

CHARSET: list[str] = (
    [BLANK_TOKEN]
    + _DIGITS
    + _LOWER
    + _UPPER
    + _OPERATORS
    + _LATEX_CMDS
)

BLANK_IDX: int = 0
NUM_CLASSES: int = len(CHARSET)

# Build lookup tables
_TOKEN_TO_IDX: dict[str, int] = {tok: i for i, tok in enumerate(CHARSET)}
_IDX_TO_TOKEN: dict[int, str] = {i: tok for i, tok in enumerate(CHARSET)}


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Regex that greedily matches LaTeX commands before single characters
_CMD_PAT = re.compile(
    r"("
    + "|".join(re.escape(c) for c in sorted(_LATEX_CMDS, key=len, reverse=True))
    + r"|.)",
    re.DOTALL,
)


class LatexTokenizer:
    """
    Token-level encoder/decoder for LaTeX strings.

    Tokenisation strategy:
        1. Greedily match known LaTeX control sequences (longest first).
        2. Fall back to single-character tokens.
        3. Unknown tokens are silently dropped on encode; unknown indices
           are represented as '?' on decode.

    Usage::

        tok = LatexTokenizer()
        ids  = tok.encode(r"P(A|B) = \\frac{P(B|A) P(A)}{P(B)}")
        back = tok.decode(ids)
    """

    def encode(self, latex: str) -> list[int]:
        """Convert a LaTeX string to a list of token indices."""
        tokens = _CMD_PAT.findall(latex)
        return [_TOKEN_TO_IDX[t] for t in tokens if t in _TOKEN_TO_IDX]

    def decode(self, indices: list[int] | list) -> str:
        """Convert token indices back to a LaTeX string."""
        return "".join(_IDX_TO_TOKEN.get(int(i), "?") for i in indices)

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def blank_idx(self) -> int:
        return BLANK_IDX
