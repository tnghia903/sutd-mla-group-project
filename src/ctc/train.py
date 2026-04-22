"""
train.py — CRNN+CTC training loop.

Training pipeline:
    1. Build dataset from synthetic renders + optional InkML inks.
    2. Collate variable-width images into padded batches.
    3. Optimise with AdamW + ReduceLROnPlateau, using nn.CTCLoss.
    4. Evaluate character error rate (CER) on the val set each epoch.
    5. Save best checkpoint by val CER.
"""

from __future__ import annotations

import dataclasses
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .charset import LatexTokenizer
from .decode import ctc_greedy_decode
from .model import CRNN
from .render import LatexCorpus, load_inkml_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    # Data
    n_train: int = 30_000
    n_val: int = 2_000
    inkml_dirs: list[str] = dataclasses.field(default_factory=list)
    cache_dir: str = "/content/ctc_data"
    # Model
    hidden_size: int = 256
    # Training
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    # Checkpoint
    checkpoint_dir: str = "/content/drive/MyDrive/mla"
    checkpoint_name: str = "ctc_best.pt"
    # Device
    device: str = "auto"
    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CTCDataset(Dataset):
    """
    In-memory dataset of (grayscale image tensor, encoded label) pairs.

    Images are stored as (H, W) uint8 numpy arrays.
    Labels are stored as lists of integer token indices.
    """

    def __init__(
        self,
        pairs: list[tuple[np.ndarray, list[int]]],
    ) -> None:
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[int]]:
        return self._pairs[idx]


def _collate(
    batch: list[tuple[np.ndarray, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate variable-width images and variable-length labels into padded batches.

    CTC requires:
        images:         (B, 1, H, W_max)  float32, values in [0, 1]
        labels:         (sum of all label lengths,)  int32
        input_lengths:  (B,) int32 — T for each sample
        label_lengths:  (B,) int32 — label length for each sample
    """
    imgs, labels = zip(*batch)

    # Pad images to the maximum width in the batch
    h = imgs[0].shape[0]
    w_max = max(img.shape[1] for img in imgs)
    padded = np.ones((len(imgs), h, w_max), dtype=np.float32)
    for i, img in enumerate(imgs):
        w = img.shape[1]
        padded[i, :, :w] = img.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(padded).unsqueeze(1)  # (B, 1, H, W_max)

    # CTC needs input_length = T (timesteps after CNN)
    # T ≈ W/4 − 1 for the CRNN architecture
    input_lengths = torch.tensor(
        [max(1, w_max // 4 - 1) for _ in imgs], dtype=torch.long
    )

    label_tensor = torch.tensor(
        [idx for label in labels for idx in label], dtype=torch.long
    )
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    return img_tensor, label_tensor, input_lengths, label_lengths


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_pairs(
    cfg: TrainConfig,
    tokenizer: LatexTokenizer,
    split: str,
    verbose: bool = True,
) -> list[tuple[np.ndarray, list[int]]]:
    """Build (image, label_indices) pairs for a given split."""
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    corpus = LatexCorpus(
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        seed=cfg.seed,
    )

    n_target = cfg.n_train if split == "train" else cfg.n_val
    stream: Iterator[tuple[np.ndarray, str]]
    if split == "train":
        stream = corpus.train()
    else:
        stream = corpus.val()

    pairs: list[tuple[np.ndarray, list[int]]] = []
    bar = (
        _tqdm(total=n_target, desc=f"Rendering {split}", unit="img", dynamic_ncols=True)
        if verbose and _tqdm is not None
        else None
    )
    for img, latex in stream:
        ids = tokenizer.encode(latex)
        if ids:
            pairs.append((img, ids))
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()

    # Add InkML real data (train only)
    if split == "train" and cfg.inkml_dirs:
        ink_count_before = len(pairs)
        for inkml_dir in cfg.inkml_dirs:
            dir_name = Path(inkml_dir).name
            ink_pairs = load_inkml_dataset(inkml_dir, augment=True)
            if verbose:
                print(f"  InkML [{dir_name}]: {len(ink_pairs)} samples")
            for img, latex in ink_pairs:
                ids = tokenizer.encode(latex)
                if ids:
                    pairs.append((img, ids))
        print(f"[CTC] InkML samples added: {len(pairs) - ink_count_before}")

    rng = random.Random(cfg.seed + (0 if split == "train" else 1))
    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cer(pred: str, target: str) -> float:
    """Character error rate via Levenshtein distance."""
    if not target:
        return 0.0 if not pred else 1.0
    # Dynamic programming
    m, n = len(pred), len(target)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / n


@torch.no_grad()
def evaluate(
    model: CRNN,
    loader: DataLoader,
    device: torch.device,
    tokenizer: LatexTokenizer,
    n_samples: int = 200,
) -> tuple[float, list[tuple[str, str]]]:
    """
    Compute average CER on up to n_samples batches.

    Returns:
        (avg_cer, sample_pairs)  where sample_pairs is a list of
        (predicted_latex, ground_truth_latex) for the first batch.
    """
    model.eval()
    idx_to_char = {i: c for i, c in enumerate(tokenizer.decode([i]) for i in range(tokenizer.num_classes))}

    # Simpler: decode via tokenizer directly
    def decode_indices(indices: np.ndarray) -> str:
        from .decode import ctc_collapse_sequence
        collapsed = ctc_collapse_sequence(
            indices, blank_idx=tokenizer.blank_idx
        )
        return tokenizer.decode(collapsed)

    total_cer = 0.0
    total_samples = 0
    sample_pairs: list[tuple[str, str]] = []

    for batch_idx, (imgs, labels_flat, input_lengths, label_lengths) in enumerate(loader):
        if total_samples >= n_samples:
            break

        imgs = imgs.to(device)
        log_probs = model(imgs)  # (B, T, C)
        probs = log_probs.exp().cpu().numpy()  # (B, T, C)

        # Reconstruct individual labels from the flat concatenation
        offset = 0
        labels_list: list[list[int]] = []
        for ll in label_lengths.tolist():
            labels_list.append(labels_flat[offset: offset + ll].tolist())
            offset += ll

        for i in range(len(labels_list)):
            pred_decoded, _ = ctc_greedy_decode(probs[i], blank_idx=tokenizer.blank_idx)
            if isinstance(pred_decoded, list):
                pred_str = tokenizer.decode(pred_decoded)
            else:
                pred_str = pred_decoded

            gt_str = tokenizer.decode(labels_list[i])
            total_cer += _cer(pred_str, gt_str)
            total_samples += 1

            if batch_idx == 0 and i < 4:
                sample_pairs.append((pred_str, gt_str))

    avg_cer = total_cer / max(total_samples, 1)
    model.train()
    return avg_cer, sample_pairs


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_ctc(cfg: TrainConfig | None = None) -> Path:
    """
    Train a CRNN+CTC model and return the path to the best checkpoint.

    Parameters:
        cfg: TrainConfig (defaults used if None).

    Returns:
        Path to the saved best checkpoint (.pt file).
    """
    if cfg is None:
        cfg = TrainConfig()

    # Resolve device
    if cfg.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(cfg.device)
    print(f"[CTC] Device: {device}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    tokenizer = LatexTokenizer()
    print(f"[CTC] Vocabulary size |Σ'| = {tokenizer.num_classes} "
          f"(blank at index {tokenizer.blank_idx})")

    # Build datasets
    print("[CTC] Building training set …")
    train_pairs = _build_pairs(cfg, tokenizer, "train")
    print(f"[CTC] Train samples: {len(train_pairs)}")

    print("[CTC] Building validation set …")
    val_pairs = _build_pairs(cfg, tokenizer, "val")
    print(f"[CTC] Val samples:   {len(val_pairs)}")

    train_ds = CTCDataset(train_pairs)
    val_ds   = CTCDataset(val_pairs)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=2,
    )

    # Model
    model = CRNN(num_classes=tokenizer.num_classes, hidden_size=cfg.hidden_size)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CTC] CRNN parameters: {total_params:,}")

    # Loss, optimiser, scheduler
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Checkpoint setup
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / cfg.checkpoint_name
    best_cer = float("inf")

    print("\n" + "═" * 60)
    print(f"Starting CTC training — {cfg.epochs} epochs")
    print("═" * 60)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for imgs, labels_flat, input_lengths, label_lengths in train_loader:
            imgs        = imgs.to(device)
            labels_flat = labels_flat.to(device)

            optimiser.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    log_probs = model(imgs)          # (B, T, C)
                    # CTCLoss expects (T, B, C)
                    loss = ctc_loss(
                        log_probs.permute(1, 0, 2),
                        labels_flat,
                        input_lengths,
                        label_lengths,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimiser)
                scaler.update()
            else:
                log_probs = model(imgs)
                loss = ctc_loss(
                    log_probs.permute(1, 0, 2),
                    labels_flat,
                    input_lengths,
                    label_lengths,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_cer, sample_pairs = evaluate(
            model, val_loader, device, tokenizer, n_samples=200
        )
        scheduler.step(val_cer)

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={avg_loss:.4f}  val_CER={val_cer:.4f}  "
            f"lr={optimiser.param_groups[0]['lr']:.2e}"
        )

        # Print a few decoded samples
        for pred, gt in sample_pairs[:2]:
            print(f"    GT : {gt[:60]}")
            print(f"    Pred: {pred[:60]}")

        # Save best checkpoint
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_cer": val_cer,
                    "num_classes": tokenizer.num_classes,
                    "hidden_size": cfg.hidden_size,
                },
                best_ckpt,
            )
            print(f"    ✓ Best checkpoint saved (CER={best_cer:.4f}) → {best_ckpt}")

    print(f"\n[CTC] Training complete. Best val CER: {best_cer:.4f}")
    print(f"[CTC] Checkpoint: {best_ckpt}")
    return best_ckpt
