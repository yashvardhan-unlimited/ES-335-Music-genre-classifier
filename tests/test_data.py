import hashlib
from pathlib import Path

import pytest

from src.data import audit_clip, list_clips
from src.utils import load_config

CFG = load_config()
AUDIO = Path(CFG["paths"]["audio_dir"])
pytestmark = pytest.mark.skipif(not AUDIO.exists(), reason="GTZAN not downloaded (make data)")


def test_list_clips_ids_and_genres():
    clips = list_clips(AUDIO)
    assert len(clips) == 1000
    assert clips["clip_id"].iloc[0] == "blues.00000"
    assert set(clips["genre"]) == set(CFG["genres"])
    assert (clips["clip_id"].str.split(".").str[0] == clips["genre"]).all()


@pytest.mark.parametrize("clip", ["blues.00000", "classical.00000"])
def test_audit_good_clip(clip):
    path = AUDIO / clip.split(".")[0] / f"{clip}.wav"
    a = audit_clip(path, CFG["audio"]["sr"])
    assert a["load_ok"] and a["error"] == ""
    assert 29 < a["duration_s"] < 31
    assert a["n_samples"] == round(a["duration_s"] * CFG["audio"]["sr"])
    assert a["md5"] == hashlib.md5(path.read_bytes()).hexdigest()
    assert 0 < a["peak"] <= 1.0 + 1e-6
    assert 0 <= a["clip_share"] <= 1
    assert a["rms_db"] < 0


def test_audit_corrupt_clip_does_not_raise():
    a = audit_clip(AUDIO / "jazz" / "jazz.00054.wav", CFG["audio"]["sr"])
    assert not a["load_ok"]
    assert a["error"]
    assert len(a["md5"]) == 32
