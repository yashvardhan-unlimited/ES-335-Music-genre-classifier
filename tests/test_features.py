from pathlib import Path

import numpy as np
import pytest

from src.features import (N_CLIP, N_WIN, clip_features, feature_names, load_clip, windows)
from src.utils import load_config

AUDIO = Path(load_config()["paths"]["audio_dir"])
needs_data = pytest.mark.skipif(not AUDIO.exists(), reason="GTZAN not downloaded (make data)")


def test_windows_shape_and_order():
    y = np.arange(N_CLIP, dtype=np.float32)
    w = windows(y)
    assert w.shape == (10, N_WIN)
    assert w[1, 0] == N_WIN and w[-1, -1] == N_CLIP - 1  # consecutive, non-overlapping


def test_feature_names():
    names = feature_names()
    assert len(names) == 57 and len(set(names)) == 57
    assert names[0] == "mfcc1_mean" and names[39] == "mfcc20_var" and names[-1] == "tempo"


@needs_data
def test_short_clip_is_padded_to_30s():
    y, padded = load_clip(AUDIO / "hiphop" / "hiphop.00032.wav")  # 29.932 s
    assert padded and y.size == N_CLIP and y.dtype == np.float32
    y, padded = load_clip(AUDIO / "blues" / "blues.00000.wav")    # 30.013 s, trimmed
    assert not padded and y.size == N_CLIP


@needs_data
def test_clip_features_real_file():
    feats, mel, padded = clip_features(AUDIO / "blues" / "blues.00000.wav")
    assert feats.shape == (10, 57) and feats.dtype == np.float32
    assert np.isfinite(feats).all()
    assert (feats[:, -1] == feats[0, -1]).all()  # one tempo per clip
    assert mel.shape == (128, 1292) and mel.dtype == np.float16
    assert np.isfinite(mel).all()
