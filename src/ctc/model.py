"""
model.py — CRNN architecture for sequence recognition (Shi et al., 2015).

Architecture:
    Input  (B, 1, 32, W)
        → CNN backbone (VGG-style, 7 conv layers) → (B, 512, 1, W')
        → map-to-sequence: squeeze H, permute      → (B, T, 512)
        → BiLSTM ×2                                 → (B, T, num_classes)
        → LogSoftmax                                → log-probs for CTCLoss

T ≈ W/4 − 1 after the CNN downsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    """
    Single BiLSTM layer.

    At each timestep t the hidden state captures both directions:
        h_t^→ = LSTM_fwd(x_t, h_{t-1}^→)     ← left context
        h_t^← = LSTM_bwd(x_t, h_{t+1}^←)     ← right context
        h_t   = [h_t^→ ∥ h_t^←] ∈ ℝ^{2·d}    ← full context

    This bidirectional context is critical for OCR because a character's
    identity often depends on its neighbours (e.g. 'rn' vs 'm').
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size,
            bidirectional=True, batch_first=True,
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)         # (B, T, 2·hidden)
        return self.linear(recurrent)       # (B, T, output_size)


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for sequence recognition.

    Input:  (B, 1, 32, W)  — batch of grayscale images, height fixed at 32
    Output: (B, T, |Σ'|)   — log-softmax probabilities over extended alphabet

    T is determined by image width after CNN downsampling: T ≈ W/4 − 1.
    The output is log-softmax so it can be fed directly to nn.CTCLoss.
    """

    def __init__(self, num_classes: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── Stage 1: CNN Feature Extractor (VGG-style) ──
        # Reduces spatial dims: (1, 32, W) → (512, 1, W')
        self.cnn = nn.Sequential(
            # Block 1: 1→64, pool 2×2          → (64, 16, W/2)
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2: 64→128, pool 2×2         → (128, 8, W/4)
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3: 128→256 + BN             → (256, 8, W/4)
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            # Block 4: 256→256, pool height only → (256, 4, W/4)
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Block 5: 256→512 + BN             → (512, 4, W/4)
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            # Block 6: 512→512, pool height only → (512, 2, W/4)
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Block 7: kernel 2×2, no padding   → (512, 1, W/4 − 1)
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True),
        )

        # ── Stage 2: BiLSTM Sequence Modeller (2 stacked layers) ──
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x: (B, 1, 32, W) float tensor

        Returns:
            (B, T, num_classes) log-softmax probabilities
        """
        conv = self.cnn(x)                          # (B, 512, 1, W')
        # Map-to-Sequence: each column becomes one timestep
        conv = conv.squeeze(2).permute(0, 2, 1)     # (B, T, 512)
        logits = self.rnn(conv)                      # (B, T, num_classes)
        return F.log_softmax(logits, dim=-1)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax probabilities (not log) — used for heatmap visualisation.

        Returns:
            (B, T, num_classes) probabilities in [0, 1]
        """
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)
        logits = self.rnn(conv)
        return F.softmax(logits, dim=-1)
