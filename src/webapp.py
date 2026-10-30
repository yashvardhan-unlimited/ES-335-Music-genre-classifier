"""Logic behind the local web app (app.py): classify audio with a saved CNN, pick sample clips,
read validation results. No Streamlit here, so pytest can reach all of it.

Test parts are never touched: samples come from the valid part, scores from results/val only.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import av
import librosa
import numpy as np
import pandas as pd
import torch

from src.cnn import MEL_DIR, WIN, GenreCNN, window_starts
from src.features import N_WIN, SR, clip_mel
from src.splits import load_split
from src.utils import load_config

CFG = load_config()
APP = CFG["app"]
GENRES = CFG["genres"]
RESULTS = Path(CFG["paths"]["results"])
VAL_DIR = RESULTS / "val"
MODEL_NAMES = {"cnn": "CNN (log-mel)", "svm": "RBF SVM", "logreg": "Logistic regression", "xgb": "XGBoost",
               "knn": "kNN", "dummy_stratified": "Dummy (stratified)", "dummy_most_frequent": "Dummy (most frequent)"}
SPLIT_SEEDS = {"A": [0], "B": CFG["split_seeds"], "Bp": CFG["split_seeds"], "C": [0]}
RESULT_FILE = re.compile(rf"({'|'.join(SPLIT_SEEDS)})_s(\d+)_({'|'.join(MODEL_NAMES)})\.json")


@dataclass
class Classifier:
    """A trained GenreCNN with the normalisation stats of its own training part."""
    model: GenreCNN
    mean: np.ndarray            # (n_mels, 1)
    std: np.ndarray             # (n_mels, 1)
    seed: int
    valid_macro_f1: float
    clip_ece: float             # calibration error of its clip scores on valid


def load_classifier(split: str = APP["split"]) -> Classifier:
    """The CNN run with the highest valid clip macro-F1 on this split (ties: first seed)."""
    runs = json.loads((VAL_DIR / f"{split}_cnn_summary.json").read_text())["per_seed"]
    best = max(runs, key=lambda r: r["valid_macro_f1"])
    seed = best["seed"]
    ck = torch.load(Path(CFG["paths"]["models"]) / f"cnn_{split}_s{seed}.pt", map_location="cpu")
    assert ck["classes"] == GENRES, "checkpoint class order differs from config"
    model = GenreCNN()
    model.load_state_dict(ck["state_dict"])
    norm = json.loads((RESULTS / f"mel_norm_{split}_s{seed}.json").read_text())
    calibration = json.loads((VAL_DIR / "analysis_calibration.json").read_text())
    return Classifier(model.eval(), np.array(norm["mean"], dtype=np.float32)[:, None],
                      np.array(norm["std"], dtype=np.float32)[:, None], seed, best["valid_macro_f1"],
                      calibration["models"][f"cnn_s{seed}"]["before"]["clip_ece"])


def decode_with_av(source: str | Path | BinaryIO) -> np.ndarray:
    """AAC and M4A, which librosa's backends cannot read: mono at the project rate, cut likewise."""
    if hasattr(source, "seek"):
        source.seek(0)  # librosa has already read from it
    limit = SR * APP["max_seconds"]
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
    chunks, kept = [], 0
    with av.open(str(source) if isinstance(source, Path) else source) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().ravel())
                kept += chunks[-1].size
            if kept >= limit:  # a small upload can hold hours of low-bitrate audio
                break
    return np.concatenate(chunks)[:limit]


def decode(source: str | Path | BinaryIO) -> np.ndarray:
    """librosa first, so every format it reads stays on the evaluated path; then PyAV."""
    try:
        return librosa.load(source, sr=SR, mono=True, duration=APP["max_seconds"])[0]
    except Exception:
        return decode_with_av(source)


def load_audio(source: str | Path | BinaryIO) -> np.ndarray:
    """Decode a file or file-like object to mono at the project rate, cut at app.max_seconds.

    Uploads are untrusted input: raises ValueError with a readable message for anything we cannot
    classify (not audio, under one 3 s window, silent).
    """
    try:
        y = decode(source)
    except Exception as e:  # each decoder raises its own errors on a bad file
        raise ValueError("This file could not be read as audio.") from e
    if y.size < N_WIN:
        raise ValueError(f"This audio is too short: at least {N_WIN / SR:.0f} seconds are needed.")
    if not np.abs(y).max() > 0:  # False for all zeros and for NaN
        raise ValueError("This audio is silent, so there is nothing to classify.")
    return y


def classify_mel(clf: Classifier, mel: np.ndarray) -> np.ndarray:
    """Window probabilities (n_windows, n_genres) for a log-mel of any length >= one window.

    Same steps as evaluation: 10 evenly spaced windows, normalised with the train stats. A 30 s
    clip gives exactly the windows the stored scores were measured on.
    """
    x = np.stack([mel[:, s:s + WIN] for s in window_starts(mel.shape[1])]).astype(np.float32)
    x = torch.from_numpy((x - clf.mean) / clf.std)[:, None]  # (n_windows, 1, n_mels, WIN)
    with torch.no_grad():
        return torch.softmax(clf.model(x), 1).numpy()


def summarise(window_probs: np.ndarray) -> dict:
    """Soft vote: scores (genre -> mean probability, best first) and how many windows agree."""
    scores = pd.Series(window_probs.mean(axis=0), index=GENRES).sort_values(ascending=False)
    agree = int((window_probs.argmax(axis=1) == GENRES.index(scores.index[0])).sum())
    return {"scores": scores, "windows_agreeing": agree, "n_windows": len(window_probs)}


def classify_audio(clf: Classifier, source: str | Path | BinaryIO) -> dict:
    """Audio file -> soft-vote summary."""
    return summarise(classify_mel(clf, clip_mel(load_audio(source))))


def classify_sample(clf: Classifier, clip_id: str) -> dict:
    """A GTZAN clip from its cached log-mel: the exact input its stored valid score used."""
    return summarise(classify_mel(clf, np.load(MEL_DIR / f"{clip_id}.npy")))


def sample_clips(n: int = APP["samples_per_genre"]) -> pd.DataFrame:
    """n random valid clips per genre (clip_id, genre, path), drawn without looking at any score."""
    clips = load_split(APP["split"], "valid").drop_duplicates("clip_id")[["clip_id", "genre"]]
    paths = pd.read_csv(Path(CFG["paths"]["splits"]) / "metadata.csv")[["clip_id", "path"]]
    picked = clips.merge(paths, on="clip_id").groupby("genre").sample(n, random_state=CFG["seed"])
    return picked.sort_values(["genre", "clip_id"]).reset_index(drop=True)


def load_results(val_dir: Path = VAL_DIR) -> pd.DataFrame:
    """One row per per-run validation file; analysis and summary files do not match the pattern."""
    rows = []
    for path in sorted(val_dir.glob("*.json")):
        m = RESULT_FILE.fullmatch(path.name)
        if m:
            r = json.loads(path.read_text())
            rows.append({"split": m[1], "seed": int(m[2]), "model": m[3], "macro_f1": r["valid_macro_f1"],
                         "majority_macro_f1": r["valid_macro_f1_majority"], "accuracy": r["valid_accuracy"],
                         "window_accuracy": r["window_accuracy"], "per_genre_f1": r["per_genre_f1"],
                         "confusion": r["confusion"]["counts"]})
    return pd.DataFrame(rows)


def missing_runs(results: pd.DataFrame) -> list[str]:
    """Result files the full protocol expects but that do not exist yet."""
    have = set(zip(results["split"], results["seed"], results["model"]))
    want = [(s, r, m) for s, seeds in SPLIT_SEEDS.items() for m in MODEL_NAMES
            for r in (CFG["cnn"]["seeds"] if s == "C" and m == "cnn" else seeds)]
    return [f"{s}_s{r}_{m}" for s, r, m in want if (s, r, m) not in have]


def model_table(results: pd.DataFrame, split: str = APP["split"]) -> pd.DataFrame:
    """Scores per model on one split, best macro-F1 first; several runs are averaged."""
    g = results[results["split"] == split].groupby("model")
    t = g[["macro_f1", "accuracy", "majority_macro_f1", "window_accuracy"]].mean()
    return t.assign(sd=g["macro_f1"].std(), runs=g.size()).sort_values("macro_f1", ascending=False)


def ladder(results: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 mean, sd and run count for every model x split; NaN where nothing has run yet."""
    t = results.groupby(["model", "split"])["macro_f1"].agg(["mean", "std", "count"])
    full = pd.MultiIndex.from_product([MODEL_NAMES, SPLIT_SEEDS], names=["model", "split"])
    return t.rename(columns={"std": "sd", "count": "runs"}).reindex(full).reset_index()


def genre_f1(results: pd.DataFrame, model: str, split: str = APP["split"]) -> pd.Series:
    """Per-genre F1 of a model on one split, averaged over its runs, best first."""
    rows = results[(results["split"] == split) & (results["model"] == model)]
    return pd.DataFrame(list(rows["per_genre_f1"])).mean().sort_values(ascending=False)


def confusion(results: pd.DataFrame, model: str, split: str = APP["split"]) -> pd.DataFrame:
    """Row-normalised confusion (true genre rows, predicted columns), pooled over the model's runs."""
    rows = results[(results["split"] == split) & (results["model"] == model)]
    counts = np.sum(list(rows["confusion"]), axis=0)
    return pd.DataFrame(counts / counts.sum(axis=1, keepdims=True), index=GENRES, columns=GENRES)
