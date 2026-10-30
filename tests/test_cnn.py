"""CNN windowing, train-only normalisation, augmentation, model shape and a tiny training run."""
import numpy as np
import pytest
import torch

from src.cnn import (GenreCNN, MelWindowDataset, compute_norm_stats, n_params, random_gain, spec_augment,
                     window_starts)
from src.train_cnn import fit
from src.utils import set_seed


@pytest.fixture
def mel_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(12):
        np.save(tmp_path / f"c{i}.npy", (rng.normal(size=(128, 300)) * 10 + i).astype(np.float16))
    return tmp_path


def test_window_starts_cover_clip_without_overrun():
    s = window_starts(1292)
    assert s.tolist() == [0, 129, 258, 387, 516, 646, 775, 904, 1033, 1162] and s[-1] + 130 == 1292


def test_norm_stats_use_only_the_given_clips(mel_dir):
    mean, std = compute_norm_stats(["c0", "c1"], mel_dir)
    both = np.concatenate([np.load(mel_dir / f"c{i}.npy").astype(np.float64) for i in (0, 1)], axis=1)
    np.testing.assert_allclose(mean, both.mean(axis=1), rtol=1e-5)
    np.testing.assert_allclose(std, both.std(axis=1), rtol=1e-4)
    assert not np.allclose(mean, compute_norm_stats(["c0", "c11"], mel_dir)[0])


def test_eval_items_are_fixed_normalised_windows(mel_dir):
    stats = compute_norm_stats(["c0", "c1"], mel_dir)
    ds = MelWindowDataset(["c2", "c3"], [4, 7], mel_dir, "eval", stats)
    assert len(ds) == 20 and ds.clip_ids[10] == "c3"
    x, y = ds[13]  # clip c3, window 3
    s = window_starts(300)[3]
    raw = np.load(mel_dir / "c3.npy").astype(np.float32)[:, s:s + 130]
    expected = (raw - stats[0][:, None]) / stats[1][:, None]
    assert y == 7 and x.shape == (1, 128, 130)
    np.testing.assert_allclose(x[0].numpy(), expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(ds[13][0], x)  # no randomness in eval mode


def test_train_items_are_random_augmented_crops(mel_dir):
    stats = compute_norm_stats(["c0"], mel_dir)
    ds = MelWindowDataset(["c0", "c1"], [0, 1], mel_dir, "train", stats, crops_per_clip=5)
    assert len(ds) == 10 and ds[7][1] == 1
    torch.manual_seed(0)
    a = ds[0][0]
    assert a.shape == (1, 128, 130) and not torch.equal(a, ds[0][0])


def test_augmentations():
    torch.manual_seed(0)
    x = torch.ones(1, 128, 130)
    m = spec_augment(x, freq_masks=2, freq_width=15, time_masks=2, time_width=20)
    assert torch.equal(x, torch.ones(1, 128, 130))  # input untouched
    zero_rows = (m[0] == 0).all(dim=1).sum().item()
    zero_cols = (m[0] == 0).all(dim=0).sum().item()
    assert zero_rows <= 30 and zero_cols <= 40
    g = random_gain(np.zeros((128, 130), np.float32), max_db=6)
    assert np.ptp(g) == 0 and abs(g[0, 0]) <= 6


def test_model_shape_and_parameter_count():
    model = GenreCNN()
    assert model(torch.zeros(3, 1, 128, 130)).shape == (3, 10)
    convs = 9 * (1 * 32 + 32 * 64 + 64 * 128 + 128 * 128)
    bns = 2 * (32 + 64 + 128 + 128)
    assert n_params(model) == convs + bns + 128 * 10 + 10


def test_fit_early_stops_and_is_reproducible(mel_dir):
    ids, y = [f"c{i}" for i in range(8)], np.arange(8) % 4
    runs = []
    for _ in range(2):
        set_seed(0)
        runs.append(fit(ids[:6], y[:6], ids[6:], y[6:], epochs=3, device=torch.device("cpu"), patience=1,
                        mel_dir=mel_dir))
    a, b = runs
    assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]
    assert 1 <= a["best_epoch"] <= len(a["history"]) <= 3
    assert a["report"]["n_valid_clips"] == 2 and a["report"]["n_valid_windows"] == 20
    refit = fit(ids, y, None, None, epochs=1, device=torch.device("cpu"), mel_dir=mel_dir)
    assert refit["best_epoch"] == 1 and refit["report"] is None and "valid_loss" not in refit["history"][0]


def test_window_level_items_and_stats_use_only_listed_windows(mel_dir):
    # split A: clip c0 has windows 1 and 4 in this part, c1 has window 0
    win = {"c0": [1, 4], "c1": [0]}
    starts = window_starts(300)
    mean, std = compute_norm_stats(["c0", "c1"], mel_dir, win)
    frames = np.concatenate([np.load(mel_dir / "c0.npy").astype(np.float64)[:, starts[w]:starts[w] + 130] for w in (1, 4)]
                            + [np.load(mel_dir / "c1.npy").astype(np.float64)[:, :130]], axis=1)
    np.testing.assert_allclose(mean, frames.mean(axis=1), rtol=1e-5)
    for mode in ("train", "eval"):
        ds = MelWindowDataset(["c0", "c1"], [3, 5], mel_dir, mode, (mean, std), window_idx=win)
        assert len(ds) == 3 and ds.clip_ids.tolist() == ["c0", "c0", "c1"] and ds.item_labels.tolist() == [3, 3, 5]
    ev = MelWindowDataset(["c0", "c1"], [3, 5], mel_dir, "eval", (mean, std), window_idx=win)
    raw = np.load(mel_dir / "c0.npy").astype(np.float32)[:, starts[4]:starts[4] + 130]
    np.testing.assert_allclose(ev[1][0][0].numpy(), (raw - mean[:, None]) / std[:, None], rtol=1e-5, atol=1e-5)


def test_rows_to_clips_marks_window_level_parts():
    import pandas as pd
    from src.train_cnn import rows_to_clips
    full = pd.DataFrame({"clip_id": np.repeat(["a", "b"], 10), "window_idx": np.tile(range(10), 2), "genre": "rock"})
    genres, win = rows_to_clips(full)
    assert win is None and genres.index.tolist() == ["a", "b"]
    genres, win = rows_to_clips(full.iloc[[0, 5, 12]])
    assert win == {"a": [0, 5], "b": [2]}
