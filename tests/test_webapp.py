import io
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src import webapp as w
from src.cnn import GenreCNN
from src.features import SR
from src.splits import load_split, split_file
from src.train_classical import valid_report

needs_model = pytest.mark.skipif(not list(Path("models").glob("cnn_C_s*.pt")), reason="CNN not trained")
needs_mels = pytest.mark.skipif(not Path("features/mel").exists(), reason="log-mels not extracted")
needs_audio = pytest.mark.skipif(not Path("data/raw").exists(), reason="GTZAN not downloaded")


def tiny_classifier(seed: int = 0) -> w.Classifier:
    """Untrained CNN with neutral normalisation, for shape checks."""
    torch.manual_seed(seed)
    return w.Classifier(GenreCNN().eval(), np.zeros((128, 1), np.float32), np.ones((128, 1), np.float32),
                        seed, 0.0, 0.0)


def audio_bytes(seconds: float, fmt: str = "WAV", amp: float = 0.3) -> io.BytesIO:
    t = np.arange(int(seconds * SR)) / SR
    buf = io.BytesIO()
    sf.write(buf, amp * np.sin(2 * np.pi * 440 * t), SR, format=fmt)
    buf.seek(0)
    return buf


def encoded_bytes(y: np.ndarray, fmt: str = "adts", rate: int = SR) -> io.BytesIO:
    """Encode (samples,) or (channels, samples) floats as AAC, in raw ADTS ("adts") or M4A ("ipod")."""
    y = np.atleast_2d(y).astype(np.float32)
    layout = {1: "mono", 2: "stereo", 6: "5.1"}[len(y)]
    buf = io.BytesIO()
    with av.open(buf, "w", format=fmt) as out:
        stream = out.add_stream("aac", rate=rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(y, format="fltp", layout=layout)
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    buf.seek(0)
    return buf


def sine(seconds: float, amp: float = 0.3, rate: int = SR) -> np.ndarray:
    return amp * np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate)


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)))


@pytest.mark.parametrize("fmt", ["WAV", "FLAC", "OGG", "MP3"])
def test_load_audio_reads_uploaded_bytes(fmt):
    y = w.load_audio(audio_bytes(4, fmt))
    assert abs(y.size / SR - 4) < 0.2 and y.ndim == 1


@pytest.mark.parametrize("fmt", ["adts", "ipod"])
def test_load_audio_reads_aac_at_the_project_rate(fmt):
    y = w.load_audio(encoded_bytes(sine(4), fmt))
    assert abs(y.size / SR - 4) < 0.2 and y.ndim == 1
    spectrum = np.abs(np.fft.rfft(y))
    assert abs(spectrum.argmax() * SR / y.size - 440) < 2  # wrong rate would move the pitch


def test_load_audio_cuts_long_audio(monkeypatch):
    monkeypatch.setitem(w.APP, "max_seconds", 5)
    assert abs(w.load_audio(audio_bytes(8)).size / SR - 5) < 0.01
    assert abs(w.load_audio(encoded_bytes(sine(8), "adts")).size / SR - 5) < 0.01


@pytest.mark.parametrize("fmt", ["adts", "ipod"])
@pytest.mark.parametrize("channels", [2, 6])
def test_aac_channels_are_averaged_like_librosa(fmt, channels):
    # the CNN uses a fixed dB reference, so a level error is a skew against training
    tone = sine(4, rate=44100)
    y = w.load_audio(encoded_bytes(np.tile(tone, (channels, 1)), fmt, 44100))
    live = 5 / 6 if channels == 6 else 1  # the encoder low-passes 5.1's LFE channel, removing this tone
    assert rms(y) == pytest.approx(rms(tone) * live, rel=0.03)


def test_aac_stream_that_changes_rate_or_channels_is_read():
    mono = encoded_bytes(sine(4, rate=44100), rate=44100).getvalue()
    other_rate = encoded_bytes(sine(4, rate=48000), rate=48000).getvalue()
    stereo = encoded_bytes(np.tile(sine(4, rate=44100), (2, 1)), rate=44100).getvalue()
    for data in (mono + other_rate, mono + stereo):
        y = w.load_audio(io.BytesIO(data))
        assert abs(y.size / SR - 8) < 0.3 and rms(y) == pytest.approx(0.3 / np.sqrt(2), rel=0.03)


def test_aac_cut_off_or_damaged_keeps_what_decodes():
    data = encoded_bytes(sine(6)).getvalue()
    assert abs(w.load_audio(io.BytesIO(data[:int(len(data) * 0.7)])).size / SR - 4.2) < 0.3
    damaged = bytearray(data)
    damaged[len(data) // 2:len(data) // 2 + 200] = bytes(200)
    assert w.load_audio(io.BytesIO(bytes(damaged))).size / SR > 4


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_load_audio_rejects_non_finite_samples(bad):
    y = sine(4).astype(np.float32)
    y[1000] = bad
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="FLOAT")
    buf.seek(0)
    with pytest.raises(ValueError, match="could not be read"):
        w.load_audio(buf)


@pytest.mark.parametrize("source,message", [(io.BytesIO(b"this is not audio" * 100), "could not be read"),
                                            (audio_bytes(1), "too short"),
                                            (audio_bytes(4, amp=0.0), "silent")])
def test_load_audio_rejects_bad_input_with_a_readable_message(source, message):
    with pytest.raises(ValueError, match=message):
        w.load_audio(source)


@pytest.mark.parametrize("frames", [130, 400, 1292, 9000])
def test_classify_mel_any_length_gives_10_probability_rows(frames):
    p = w.classify_mel(tiny_classifier(), np.random.default_rng(0).normal(size=(128, frames)).astype(np.float16))
    assert p.shape == (10, 10)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)


def test_summarise_is_a_soft_vote_not_a_majority():
    probs = np.full((3, 10), 0.0)
    probs[:2, 1] = 0.6                  # two windows lean a little to genre 1
    probs[:2, 2] = 0.4
    probs[2, 2] = 1.0                   # one window is sure of genre 2
    out = w.summarise(probs)
    assert out["scores"].index[0] == w.GENRES[2]
    assert out["windows_agreeing"] == 1 and out["n_windows"] == 3
    assert out["scores"].iloc[0] == pytest.approx(0.6)


@pytest.mark.parametrize("top,unsure", [(0.27, True), (0.69, True), (0.7, False), (0.95, False)])
def test_summarise_is_unsure_below_the_cutoff(monkeypatch, top, unsure):
    monkeypatch.setitem(w.APP, "min_confidence", 0.7)
    probs = np.full((1, 10), (1 - top) / 9, dtype=np.float64)
    probs[0, 0] = top
    assert w.summarise(probs)["unsure"] is unsure


@needs_model
@needs_mels
def test_unsure_cutoff_separates_weak_answers_on_valid():
    clf = w.load_classifier()
    clips = load_split("C", "valid").drop_duplicates("clip_id")
    results = [w.classify_sample(clf, c.clip_id) for c in clips.itertuples()]
    right = np.array([r["scores"].index[0] == g for r, g in zip(results, clips["genre"])])
    unsure = np.array([r["unsure"] for r in results])
    assert right[unsure].mean() < 0.5 < 0.8 < right[~unsure].mean()  # the wording in app.py relies on this


@pytest.mark.parametrize("model", ["dummy_stratified", "cnn"])
def test_load_results_scores_match_their_files(model):
    r = w.load_results()
    row = r[(r["split"] == "C") & (r["seed"] == 0) & (r["model"] == model)].iloc[0]
    saved = json.loads(Path(f"results/val/C_s0_{model}.json").read_text())
    assert row["macro_f1"] == saved["valid_macro_f1"] and row["accuracy"] == saved["valid_accuracy"]


def test_load_results_skips_analysis_files_and_reads_nothing_from_test():
    r = w.load_results()
    assert set(r["model"]) <= set(w.MODEL_NAMES) and set(r["split"]) <= set(w.SPLIT_SEEDS)
    assert not list(Path("results/val").glob("*test*"))


def fake_results() -> pd.DataFrame:
    row = {"per_genre_f1": dict(zip(w.GENRES, np.linspace(0, 0.9, 10))),
           "confusion": (np.eye(10, dtype=int) * 3 + 1).tolist(),
           "majority_macro_f1": 0.4, "accuracy": 0.5, "window_accuracy": 0.3}
    return pd.DataFrame([{**row, "split": "C", "seed": s, "model": "cnn", "macro_f1": f} for s, f in [(0, 0.5), (1, 0.7)]]
                        + [{**row, "split": "C", "seed": 0, "model": "svm", "macro_f1": 0.4}])


def test_tables_average_runs_and_rank_models():
    t = w.model_table(fake_results())
    assert list(t.index) == ["cnn", "svm"]
    assert t.loc["cnn", "macro_f1"] == pytest.approx(0.6) and t.loc["cnn", "runs"] == 2
    assert np.isnan(t.loc["svm", "sd"])                                     # one run has no sd


def test_ladder_marks_unfinished_cells_instead_of_dropping_them():
    t = w.ladder(fake_results()).set_index(["model", "split"])
    assert len(t) == len(w.MODEL_NAMES) * len(w.SPLIT_SEEDS)
    assert t.loc[("cnn", "C"), "runs"] == 2 and np.isnan(t.loc[("cnn", "A"), "mean"])


def test_genre_f1_and_confusion():
    assert w.genre_f1(fake_results(), "cnn").index[0] == w.GENRES[-1]
    c = w.confusion(fake_results(), "cnn")
    np.testing.assert_allclose(c.sum(axis=1), 1)
    assert c.iloc[0, 0] == pytest.approx(4 / 13)


def test_missing_runs_lists_what_the_protocol_still_needs():
    all_runs = w.missing_runs(w.load_results().iloc[:0])
    assert len(all_runs) == 86
    assert "C_s2_cnn" in all_runs and "Bp_s4_xgb" in all_runs and "A_s1_svm" not in all_runs


def test_samples_are_valid_clips_never_test():
    s = w.sample_clips()
    assert (s.groupby("genre").size() == w.APP["samples_per_genre"]).all() and len(s) == 20
    assert set(s["clip_id"]) <= set(load_split("C", "valid")["clip_id"])
    split = pd.read_csv(split_file("C"))
    assert not set(s["clip_id"]) & set(split.loc[split["part"] == "test", "clip_id"])
    pd.testing.assert_frame_equal(s, w.sample_clips())                      # same draw every run


@needs_model
def test_served_model_is_the_best_valid_seed():
    clf = w.load_classifier()
    runs = json.loads(Path("results/val/C_cnn_summary.json").read_text())["per_seed"]
    assert clf.valid_macro_f1 == max(r["valid_macro_f1"] for r in runs)
    assert not clf.model.training


@needs_model
@needs_mels
def test_app_pipeline_reproduces_the_stored_valid_score():
    """The app's path, end to end on all 197 valid clips, must equal the evaluated model."""
    clf = w.load_classifier()
    valid = load_split("C", "valid")
    clips = valid.drop_duplicates("clip_id")
    probs = np.concatenate([w.classify_mel(clf, np.load(w.MEL_DIR / f"{c}.npy")) for c in clips["clip_id"]])
    y = pd.Series(np.repeat(clips["genre"].to_numpy(), 10), index=np.repeat(clips["clip_id"].to_numpy(), 10))
    assert valid_report(probs, np.array(w.GENRES), y)["valid_macro_f1"] == pytest.approx(clf.valid_macro_f1)


@needs_model
@needs_audio
def test_real_audio_file_end_to_end():
    clip = w.sample_clips().iloc[0]
    out = w.classify_audio(w.load_classifier(), clip["path"])
    assert out["scores"].sum() == pytest.approx(1, abs=1e-5) and out["n_windows"] == 10


@needs_model
@needs_mels
def test_app_runs_sample_and_results_tabs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=120).run()
    assert not at.exception
    assert any(s.value.startswith(("Prediction: ", "Not sure: ")) for s in at.subheader)
    assert any("validation" in i.value for i in at.info)            # scores are labelled as validation
    at.radio[0].set_value("Upload your own").run()
    assert not at.exception

    at.file_uploader[0].set_value(("tone.wav", audio_bytes(4).getvalue(), "audio/wav")).run()
    assert not at.exception and any(s.value.startswith(("Prediction: ", "Not sure: ")) for s in at.subheader)
    at.file_uploader[0].set_value(("bad.wav", b"this is not audio" * 100, "audio/wav")).run()
    assert not at.exception and any("could not be read" in e.value for e in at.error)


@needs_model
@needs_mels
@pytest.mark.parametrize("keep", ["cnn", "nothing"])
def test_app_survives_unfinished_results(monkeypatch, keep):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    results = w.load_results()
    partial = results[results["model"] == keep].reset_index(drop=True)
    monkeypatch.setattr(w, "load_results", lambda: partial)
    st.cache_data.clear()
    try:
        at = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=120).run()
        assert not at.exception
    finally:
        st.cache_data.clear()


def test_load_results_with_no_files_keeps_its_columns(tmp_path):
    assert {"split", "seed", "model"} <= set(w.load_results(tmp_path).columns)
    assert len(w.missing_runs(w.load_results(tmp_path))) > 0
