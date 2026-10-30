"""Architecture numbers quoted in learn/10_cnn.md: parameters, shapes, receptive field, MACs,
normalisation stats and augmentation coverage. Uses split C train only (for the stats).

Run from repo root: python scripts/lesson10_examples.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cnn import CNN, GenreCNN, n_params  # noqa: E402

model = GenreCNN().eval()
print("total trainable parameters", n_params(model))
x, rf, jump, macs = torch.zeros(1, 1, 128, 130), 1, 1, 0
print(f"{'layer':<12}{'output (C x mel x frames)':<28}{'params':>8}{'MACs':>14}{'receptive field':>18}")
for layer in list(model.features) + list(model.head):
    y = layer(x)
    p = sum(q.numel() for q in layer.parameters())
    m = 0
    if isinstance(layer, nn.Conv2d):
        m = y.shape[1] * y.shape[2] * y.shape[3] * layer.in_channels * 9
        rf += 2 * jump
    elif isinstance(layer, nn.MaxPool2d):
        rf += jump
        jump *= 2
    macs += m
    shape = " x ".join(map(str, y.shape[1:]))
    print(f"{type(layer).__name__:<12}{shape:<28}{p:>8}{m:>14,}{rf:>12} ({rf * 512 / 22050:.2f} s)")
    x = y
print(f"forward multiply-adds per window: {macs:,}")
dense = 128 * 130 * 32 * 128 * 130
print(f"a dense layer from the 128x130 input to the first block's 32x128x130 output: {dense:,} weights")

norm = json.loads(Path("results/mel_norm_C_s0.json").read_text())
mean, std = np.array(norm["mean"]), np.array(norm["std"])
print(f"norm stats from {norm['n_clips']} {norm['fit_on']} clips: mean {mean.min():.1f} to {mean.max():.1f} dB "
      f"(bin {mean.argmax()} highest), std {std.min():.1f} to {std.max():.1f} dB")
a = CNN["augment"]
print(f"SpecAugment expected masked share (overlaps ignored): mel bins {a['freq_masks'] * a['freq_width'] / 2 / 128:.3f}, "
      f"frames {a['time_masks'] * a['time_width'] / 2 / CNN['crop_frames']:.3f}")
print(f"possible crop offsets per clip: {1292 - CNN['crop_frames'] + 1}")
