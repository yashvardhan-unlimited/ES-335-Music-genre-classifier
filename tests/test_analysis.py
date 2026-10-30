"""Pieces used by scripts/analysis_val.py: windowing, feature groups, calibration, artist sampling."""
import numpy as np
import pytest

from src.evaluate import ece, fit_temperature, reliability, softmax_t
from src.features import N_CLIP, feature_group, feature_names, windows
from src.splits import artist_groups, load_split, sample_artists


@pytest.mark.parametrize("n, length", [(30, 22050), (10, 66150), (6, 110250), (3, 220500)])
def test_windows_of_any_length_tile_the_clip(n, length):
    y = np.arange(N_CLIP, dtype=np.float32)
    w = windows(y, n)
    assert w.shape == (n, length) and w[1, 0] == length and w[-1, -1] == N_CLIP - 1
    with pytest.raises(AssertionError):
        windows(y, 8)  # 661,500 / 8 is not whole


def test_every_feature_is_in_exactly_one_group():
    groups = [feature_group(n) for n in feature_names()]
    assert {g: groups.count(g) for g in set(groups)} == {
        "mfcc": 40, "chroma": 2, "spectral_shape": 6, "zcr_rms": 4, "hpss": 4, "tempo": 1}
    assert feature_group("harm_rms_var") == "hpss" and feature_group("rms_mean") == "zcr_rms"


def test_reliability_and_ece_by_hand():
    probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.65, 0.35], [0.65, 0.35]])
    y = np.array([0, 1, 0, 0])  # bin (0.9, 1]: conf 0.95, acc 0.5; bin (0.6, 0.7]: conf 0.65, acc 1.0
    t = reliability(probs, y, 10)
    assert t["n"].tolist() == [0, 0, 0, 0, 0, 0, 2, 0, 0, 2]
    assert ece(probs, y, 10) == pytest.approx(0.5 * 0.45 + 0.5 * 0.35)


def test_temperature_recovers_known_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20000, 10)) * 2
    y = np.array([rng.choice(10, p=p) for p in softmax_t(z)])  # labels drawn from softmax(z)
    t = fit_temperature(3 * z, y)                              # logits 3x too confident
    assert t == pytest.approx(3, rel=0.05)
    p = softmax_t(3 * z, t)
    assert (p.argmax(1) == (3 * z).argmax(1)).all()             # same predictions
    assert ece(p, y) < ece(softmax_t(3 * z), y)


def test_sample_artists_keeps_whole_artists():
    ids = list(load_split("C", "train").clip_id.unique())
    groups = artist_groups(ids)
    sub = sample_artists(ids, 0.25, seed=0)
    kept = set(groups[sub])
    assert set(groups[groups.isin(kept)].index) == set(sub)        # every clip of a kept artist
    assert len(kept) == round(0.25 * groups.nunique())
    assert sample_artists(ids, 1.0, seed=5) == ids
    assert sample_artists(ids, 0.5, seed=1) == sample_artists(ids, 0.5, seed=1)
