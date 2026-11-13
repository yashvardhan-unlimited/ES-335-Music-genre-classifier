"""Clip-level predictions of every selected model on one split part, rebuilt from saved choices.

Classical models are refit on train with the settings stored in results/val/{split}_s{seed}_{model}.json
(the same train-only refit as src/train_classical.py, so the fit is deterministic); XGBoost is refit
with its early-stopped tree count. CNNs are loaded from models/cnn_{split}_s{seed}.pt with their saved
train normalisation. On valid this reproduces every stored score (checked in scripts/evaluate_part.py);
on test it is the only path that scores test, and it needs the pre-test-freeze tag (load_split).
refit=True gives the final models of DECISIONS.md (2026-11-14): the same settings fit on train+valid
(XGBoost: the same tree count; CNN: models/cnn_{split}_s{seed}_refit.pt). They have seen valid, so
they are scored on test only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.cnn import MEL_DIR, GenreCNN, MelWindowDataset
from src.evaluate import aggregate_clips
from src.models import MODELS, xgb_model
from src.splits import credit_clusters, load_split
from src.train_classical import load_windows
from src.train_cnn import encode, predict, rows_to_clips
from src.utils import load_config

CFG = load_config()
GENRES = np.array(CFG["genres"])
RESULTS = Path(CFG["paths"]["results"])
VAL_DIR = RESULTS / "val"
MODEL_DIR = Path(CFG["paths"]["models"])


def stored_result(split: str, seed: int, model: str) -> dict:
    """The validation result file that holds a run's chosen settings and valid score."""
    return json.loads((VAL_DIR / f"{split}_s{seed}_{model}.json").read_text())


def _clip_table(window_probs: np.ndarray, classes: np.ndarray, clip_ids, y_window) -> pd.DataFrame:
    """Window probabilities -> one row per clip: clip_id, y_true, y_pred, p_<genre> (config order)."""
    ids, probs, pred = aggregate_clips(window_probs, clip_ids)
    probs = pd.DataFrame(probs, columns=classes).reindex(columns=GENRES, fill_value=0.0)
    truth = pd.Series(np.asarray(y_window), index=np.asarray(clip_ids)).groupby(level=0).first()
    out = pd.DataFrame({"clip_id": ids, "y_true": truth.loc[ids].to_numpy(), "y_pred": classes[pred]})
    return pd.concat([out, probs.add_prefix("p_")], axis=1)


def classical_clip_predictions(split: str, seed: int, model: str, part: str, refit: bool = False) -> pd.DataFrame:
    """Fit the stored best setting on train (refit: train+valid) and predict `part` (soft vote per clip)."""
    r = stored_result(split, seed, model)
    allow = part == "test"
    X_tr, y_tr = load_windows(split, "train", seed)
    if refit:
        X_va, y_va = load_windows(split, "valid", seed)
        X_tr, y_tr = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
    X_ev, y_ev = load_windows(split, part, seed, allow_test=allow)
    if model == "xgb":  # all threads, as in tune_xgb: the thread count changes XGBoost's float sums
        fixed = {k: v for k, v in r["fixed_params"].items() if k != "early_stopping_rounds"}
        est = xgb_model(**{**fixed, "n_estimators": r["n_trees"]}, **r["best_params"])
        est.fit(X_tr, pd.Categorical(y_tr, categories=GENRES).codes, verbose=False)
        classes = GENRES
    else:
        with threadpool_limits(1):  # as in train_classical.tune_and_score: lbfgs depends on thread count
            est = clone(MODELS[model][0]()).set_params(**{f"clf__{k}": v for k, v in r["best_params"].items()})
            est.fit(X_tr, y_tr)
        classes = np.asarray(est.classes_)
    return _clip_table(est.predict_proba(X_ev), classes, X_ev.index, y_ev)


def cnn_clip_predictions(split: str, seed: int, part: str, refit: bool = False) -> pd.DataFrame:
    """Load models/cnn_{split}_s{seed}[_refit].pt and its normalisation, predict `part`'s fixed windows."""
    suffix = "_refit" if refit else ""
    ck = torch.load(MODEL_DIR / f"cnn_{split}_s{seed}{suffix}.pt", map_location="cpu")
    assert ck["classes"] == list(GENRES), "checkpoint class order differs from config"
    model = GenreCNN()
    model.load_state_dict(ck["state_dict"])
    norm = json.loads((RESULTS / f"mel_norm_{split}_s{seed}{suffix}.json").read_text())
    clips, windows = rows_to_clips(load_split(split, part, seed, allow_test=part == "test"))
    ds = MelWindowDataset(list(clips.index), encode(clips), MEL_DIR, "eval",
                          (np.array(norm["mean"]), np.array(norm["std"])), window_idx=windows)
    probs = predict(model, ds, torch.device("cpu"))
    return _clip_table(probs, GENRES, ds.clip_ids, GENRES[ds.item_labels])


def clip_predictions(split: str, seed: int, model: str, part: str, refit: bool = False) -> pd.DataFrame:
    """Clip predictions of one run, plus each clip's artist credit cluster (for the bootstrap)."""
    if part not in ("valid", "test"):
        raise ValueError(f"part must be 'valid' or 'test', got {part!r}")
    if refit and part != "test":
        raise ValueError("refit models have seen valid; score them on test only")
    df = (cnn_clip_predictions(split, seed, part, refit) if model == "cnn"
          else classical_clip_predictions(split, seed, model, part, refit))
    meta = pd.read_csv(Path(CFG["paths"]["splits"]) / "metadata.csv").set_index("clip_id")
    df.insert(1, "cluster", credit_clusters(meta["artist"]).loc[df["clip_id"]].to_numpy())
    return df
