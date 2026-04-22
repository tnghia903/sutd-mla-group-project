"""src/ctc — CRNN+CTC sequence recognition package."""

from .charset import BLANK_IDX, BLANK_TOKEN, CHARSET, NUM_CLASSES, LatexTokenizer
from .decode import ctc_beam_search, ctc_collapse_sequence, ctc_greedy_decode
from .infer import CTCRecogniser, preprocess_for_ctc, results_to_markdown
from .model import CRNN, BidirectionalLSTM
from .render import LatexCorpus, inkml_to_image, load_inkml_dataset, render_latex_to_image
from .train import TrainConfig, train_ctc

__all__ = [
    "BLANK_IDX",
    "BLANK_TOKEN",
    "CHARSET",
    "NUM_CLASSES",
    "LatexTokenizer",
    "ctc_beam_search",
    "ctc_collapse_sequence",
    "ctc_greedy_decode",
    "CTCRecogniser",
    "preprocess_for_ctc",
    "results_to_markdown",
    "CRNN",
    "BidirectionalLSTM",
    "LatexCorpus",
    "inkml_to_image",
    "load_inkml_dataset",
    "render_latex_to_image",
    "TrainConfig",
    "train_ctc",
]
