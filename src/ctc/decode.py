"""
decode.py — CTC greedy and beam-search decoding.

Implements the collapsing function B: (Σ')^T → Σ* described in
Graves et al. (2006) — "Connectionist Temporal Classification".
"""

from __future__ import annotations

import numpy as np


def ctc_collapse_sequence(
    indices: list[int] | np.ndarray,
    blank_idx: int = 0,
    idx_to_char: dict[int, str] | None = None,
) -> str | list[int]:
    """
    CTC greedy decoding: collapse a raw alignment π into a label l.

    The collapsing function B operates in two steps:

        Step 1 — Remove consecutive duplicates:
            π = [ε, ε, a, a, ε, b, b, ε]
                         ↓
            deduped = [ε, a, ε, b, ε]

        Step 2 — Remove all blank tokens (ε):
            deduped = [ε, a, ε, b, ε]
                         ↓
                  l = [a, b]  →  "ab"

    Multiple alignments map to the same label (many-to-one):
        B(ε-a-a-ε-b-b-ε) = "ab"
        B(a-ε-ε-b-ε-ε-ε) = "ab"
        B(a-a-a-b-b-b-b) = "ab"

    This many-to-one property is why CTC loss marginalises over all valid
    paths:  P(l | Y) = Σ_{π ∈ B⁻¹(l)} Π_{t=1}^{T} y_{π_t}^t

    Parameters:
        indices:     Array of predicted character indices (length T).
        blank_idx:   Index of the CTC blank token ε (default 0).
        idx_to_char: Optional dict mapping index → string token.
                     If provided, returns a decoded string; otherwise
                     returns the collapsed list of integer indices.

    Returns:
        Decoded string (if idx_to_char given) or list[int].
    """
    # Step 1: Remove consecutive duplicate indices
    deduped: list[int] = []
    prev = None
    for idx in indices:
        idx = int(idx)
        if idx != prev:
            deduped.append(idx)
        prev = idx

    # Step 2: Remove blank tokens
    collapsed = [i for i in deduped if i != blank_idx]

    if idx_to_char is not None:
        return "".join(idx_to_char.get(i, "?") for i in collapsed)
    return collapsed


def ctc_greedy_decode(
    probs: np.ndarray,
    blank_idx: int = 0,
    idx_to_char: dict[int, str] | None = None,
) -> tuple[str | list[int], np.ndarray]:
    """
    Full greedy CTC decoding from a probability matrix.

    Parameters:
        probs:       (T, num_classes) probability matrix P(c_t | h_t).
        blank_idx:   CTC blank index.
        idx_to_char: Optional index → token mapping.

    Returns:
        (decoded_label, greedy_indices)  where greedy_indices is the raw
        argmax path π before collapsing (length T).
    """
    greedy_indices = np.argmax(probs, axis=1)   # (T,)
    decoded = ctc_collapse_sequence(greedy_indices, blank_idx, idx_to_char)
    return decoded, greedy_indices


def ctc_beam_search(
    probs: np.ndarray,
    blank_idx: int = 0,
    beam_width: int = 5,
    idx_to_char: dict[int, str] | None = None,
) -> list[tuple[str | list[int], float]]:
    """
    Simple CTC beam search decoding (pedagogic, not production-optimised).

    Returns the top `beam_width` hypotheses sorted by descending log-prob.

    Parameters:
        probs:      (T, num_classes) probability matrix.
        blank_idx:  CTC blank index.
        beam_width: Number of beams to keep at each step.
        idx_to_char: Optional index → token mapping.

    Returns:
        List of (label, log_prob) tuples, best-first.
    """
    T, C = probs.shape
    log_probs = np.log(probs + 1e-10)

    # Each beam: (label_indices: tuple[int, ...], log_prob: float)
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]

    for t in range(T):
        candidates: dict[tuple[int, ...], float] = {}

        for label, lp in beams:
            for c in range(C):
                new_lp = lp + log_probs[t, c]

                if c == blank_idx:
                    key = label
                elif label and label[-1] == c:
                    # Duplicate — same label as before
                    key = label
                else:
                    key = label + (c,)

                if key in candidates:
                    # Log-sum-exp merge
                    old = candidates[key]
                    candidates[key] = np.logaddexp(old, new_lp)
                else:
                    candidates[key] = new_lp

        # Keep top beams
        sorted_beams = sorted(candidates.items(), key=lambda x: -x[1])
        beams = [(label, lp) for label, lp in sorted_beams[:beam_width]]

    results = []
    for label_indices, lp in beams:
        if idx_to_char is not None:
            decoded = "".join(idx_to_char.get(i, "?") for i in label_indices)
        else:
            decoded = list(label_indices)
        results.append((decoded, float(lp)))

    return results
