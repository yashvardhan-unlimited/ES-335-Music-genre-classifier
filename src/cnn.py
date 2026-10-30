"""Log-mel CNN pieces: window dataset, train-only normalisation, augmentation, model.

Each clip's full log-mel (128 mel bins x 1292 frames, features/mel/<clip_id>.npy) is
cut into 130-frame (3 s) windows. Training uses random crops with augmentation;
evaluation uses the 10 fixed windows of each clip, scored per clip by soft vote.
"""
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.utils import load_config

CFG = load_config()
CNN = CFG["cnn"]
WIN = CNN["crop_frames"]
N_WINDOWS = CFG["audio"]["n_windows"]
MEL_DIR = Path(CFG["paths"]["features"]) / "mel"


def load_mels(clip_ids: Sequence[str], mel_dir: Path = MEL_DIR) -> np.ndarray:
    """Full-clip log-mels stacked as float32, shape (n_clips, n_mels, n_frames)."""
    return np.stack([np.load(Path(mel_dir) / f"{c}.npy") for c in clip_ids]).astype(np.float32)


def window_starts(n_frames: int, win: int = WIN, n: int = N_WINDOWS) -> np.ndarray:
    """Start frames of the n fixed eval windows, evenly spaced from 0 to n_frames - win.

    For 1292 frames: 0, 129, 258, ..., 1162 (the last window ends exactly at frame 1292).
    """
    return np.linspace(0, n_frames - win, n).round().astype(int)


def compute_norm_stats(train_clip_ids: Sequence[str], mel_dir: Path = MEL_DIR,
                       window_idx: dict[str, list[int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-mel-bin mean and std over every frame of the given (training) clips only.

    window_idx (window-level split A): use only the frames of each clip's listed windows.
    """
    total = sq = 0.0
    n = 0
    for c in train_clip_ids:
        m = np.load(Path(mel_dir) / f"{c}.npy").astype(np.float64)
        if window_idx is not None:
            starts = window_starts(m.shape[1])
            m = np.concatenate([m[:, starts[w]:starts[w] + WIN] for w in window_idx[c]], axis=1)
        total = total + m.sum(axis=1)
        sq = sq + (m ** 2).sum(axis=1)
        n += m.shape[1]
    mean = total / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def random_gain(x: np.ndarray, max_db: float = CNN["augment"]["gain_db"]) -> np.ndarray:
    """Add one uniform random offset in [-max_db, +max_db] dB to a whole log-mel window."""
    return x + float(torch.empty(()).uniform_(-max_db, max_db))


def spec_augment(x: torch.Tensor, freq_masks: int = CNN["augment"]["freq_masks"],
                 freq_width: int = CNN["augment"]["freq_width"], time_masks: int = CNN["augment"]["time_masks"],
                 time_width: int = CNN["augment"]["time_width"]) -> torch.Tensor:
    """SpecAugment (Park et al. 2019) without time warping: zero random bands of mel bins and of frames.

    x is a normalised (..., n_mels, n_frames) window, so 0 is the training mean. Mask widths
    are uniform in [0, width]; returns a masked copy.
    """
    x = x.clone()
    n_mels, n_frames = x.shape[-2:]
    for _ in range(freq_masks):
        w = int(torch.randint(0, freq_width + 1, ()))
        f0 = int(torch.randint(0, n_mels - w + 1, ()))
        x[..., f0:f0 + w, :] = 0
    for _ in range(time_masks):
        w = int(torch.randint(0, time_width + 1, ()))
        t0 = int(torch.randint(0, n_frames - w + 1, ()))
        x[..., t0:t0 + w] = 0
    return x


class MelWindowDataset(Dataset):
    """3 s log-mel windows of whole clips, normalised per mel bin with given (train) stats.

    Clip-level splits (window_idx None): mode "train" gives crops_per_clip items per clip, each a
    random 130-frame crop anywhere in the clip; mode "eval" gives the 10 fixed windows of each clip.
    Window-level split A (window_idx {clip_id: [window indices]}): items are exactly the listed
    windows at their fixed positions in both modes, since the rest of the clip belongs to other parts.
    Train items get random gain (before normalisation) and SpecAugment (after); eval items none.
    self.clip_ids and self.item_labels give each item's clip and label, for voting.
    """

    def __init__(self, clip_ids: Sequence[str], labels: Sequence[int], mel_dir: Path, mode: str,
                 norm_stats: tuple[np.ndarray, np.ndarray], crops_per_clip: int = CNN["crops_per_clip"],
                 window_idx: dict[str, list[int]] | None = None):
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.mels = load_mels(clip_ids, mel_dir)
        self.mean, self.std = (np.asarray(s, dtype=np.float32)[:, None] for s in norm_stats)
        self.mode = mode
        self.starts = window_starts(self.mels.shape[2])
        if window_idx is None:
            per_clip = crops_per_clip if mode == "train" else N_WINDOWS
            self.item_clip = np.repeat(np.arange(len(clip_ids)), per_clip)
            self.item_window = np.tile(np.arange(per_clip), len(clip_ids))  # unused for random crops
            self.random_crop = mode == "train"
        else:
            pairs = [(i, w) for i, c in enumerate(clip_ids) for w in window_idx[c]]
            self.item_clip, self.item_window = (np.array(v) for v in zip(*pairs))
            self.random_crop = False
        self.clip_ids = np.asarray(clip_ids)[self.item_clip]
        self.item_labels = np.asarray(labels, dtype=np.int64)[self.item_clip]

    def __len__(self) -> int:
        return len(self.item_clip)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        clip, train = self.item_clip[i], self.mode == "train"
        if self.random_crop:
            start = int(torch.randint(0, self.mels.shape[2] - WIN + 1, ()))
        else:
            start = self.starts[self.item_window[i]]
        x = self.mels[clip, :, start:start + WIN]
        if train:
            x = random_gain(x)
        x = torch.from_numpy((x - self.mean) / self.std)[None]  # (1, n_mels, WIN)
        if train:
            x = spec_augment(x)
        return x, int(self.item_labels[i])


class GenreCNN(nn.Module):
    """Blocks of Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2, then global average pooling,
    dropout and one linear layer. Input (batch, 1, n_mels, frames); output class logits.
    """

    def __init__(self, channels: Sequence[int] = CNN["channels"], n_classes: int = len(CFG["genres"]),
                 dropout: float = CNN["dropout"]):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            # no conv bias: BatchNorm's shift (beta) does the same job
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def n_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
