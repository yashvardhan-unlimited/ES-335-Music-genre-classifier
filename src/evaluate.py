"""Clip-level evaluation: window -> clip aggregation, scores, validation scorer,
and artist-cluster bootstrap confidence intervals.

Every reported score is per clip (the unit of prediction), never per window.
"""
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import PredefinedSplit

Metric = Callable[[np.ndarray, np.ndarray], float]


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Unweighted mean of per-class F1 over the classes present in y_true or y_pred."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def aggregate_clips(window_probs: np.ndarray, clip_ids: Sequence, method: str = "soft"
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine window predictions into one prediction per clip.

    window_probs: (n_windows, n_classes) probabilities; clip_ids: (n_windows,).
    soft: clip_probs = mean of window probabilities, prediction = argmax.
    majority: clip_probs = share of windows voting for each class; ties between the
    most-voted classes are broken by the higher mean probability.
    Returns (sorted unique clip_ids, clip_probs, clip_pred as class column indices).
    """
    window_probs = np.asarray(window_probs, dtype=float)
    ids, inv = np.unique(np.asarray(clip_ids), return_inverse=True)
    n_win = np.bincount(inv)[:, None]
    mean_probs = np.zeros((len(ids), window_probs.shape[1]))
    np.add.at(mean_probs, inv, window_probs)
    mean_probs /= n_win
    if method == "soft":
        return ids, mean_probs, mean_probs.argmax(axis=1)
    if method != "majority":
        raise ValueError(f"method must be 'soft' or 'majority', got {method!r}")
    votes = np.zeros_like(mean_probs)
    np.add.at(votes, (inv, window_probs.argmax(axis=1)), 1)
    tied = votes == votes.max(axis=1, keepdims=True)
    pred = np.where(tied, mean_probs, -np.inf).argmax(axis=1)
    return ids, votes / n_win, pred


def clip_scores(y_true_clip: Sequence, y_pred_clip: Sequence, labels: Sequence | None = None) -> dict:
    """macro_f1, accuracy, per_genre_f1 {label: f1}, confusion counts and row-normalised.

    labels fixes the class order of per_genre_f1 and the confusion matrix
    (rows = true, columns = predicted); default: sorted labels present.
    """
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true_clip), np.asarray(y_pred_clip)]))
    labels = list(labels)
    per = f1_score(y_true_clip, y_pred_clip, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true_clip, y_pred_clip, labels=labels)
    rows = cm.sum(axis=1, keepdims=True)
    return {
        "macro_f1": float(per.mean()),
        "accuracy": float(accuracy_score(y_true_clip, y_pred_clip)),
        "per_genre_f1": {lab: float(f) for lab, f in zip(labels, per)},
        "confusion": cm,
        "confusion_norm": np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0),
    }


def window_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Share of windows predicted correctly (diagnostic only, not a reported score)."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def make_predefined_split(parts: Sequence[str]) -> PredefinedSplit:
    """One train/valid fold for GridSearchCV: 'train' -> -1 (always train), 'valid' -> 0."""
    codes = {"train": -1, "valid": 0}
    bad = set(parts) - codes.keys()
    if bad:
        raise ValueError(f"parts must be 'train' or 'valid' only, got {sorted(bad)}")
    return PredefinedSplit([codes[p] for p in parts])


class ClipMacroF1Scorer:
    """GridSearchCV scorer: window probabilities -> soft vote per clip -> clip macro-F1.

    X must be a DataFrame (sklearn keeps its index when it slices rows). Clip ids are
    looked up through that index: clip_ids.loc[X.index] if a Series is given,
    otherwise X.index itself is taken to hold the clip ids.
    """

    def __init__(self, clip_ids: pd.Series | None = None, method: str = "soft"):
        self.clip_ids = clip_ids
        self.method = method

    def __call__(self, estimator, X: pd.DataFrame, y: Sequence) -> float:
        ids = X.index.to_numpy() if self.clip_ids is None else self.clip_ids.loc[X.index].to_numpy()
        classes = estimator.classes_
        clip, _, pred_idx = aggregate_clips(estimator.predict_proba(X), ids, self.method)
        truth = pd.Series(np.asarray(y), index=ids).groupby(level=0).agg(["first", "nunique"])
        assert (truth["nunique"] == 1).all(), "windows of one clip have different labels"
        y_clip = truth.loc[clip, "first"].to_numpy()
        return float(f1_score(y_clip, classes[pred_idx], labels=classes, average="macro", zero_division=0))


def _cluster_rows(df: pd.DataFrame, cluster_col: str) -> list[np.ndarray]:
    """Row positions per cluster; a row with a missing cluster value is its own cluster."""
    key = df[cluster_col].astype(object).to_numpy().copy()
    missing = pd.isna(key)
    key[missing] = [f"__solo_{i}" for i in np.flatnonzero(missing)]
    return [np.flatnonzero(key == k) for k in pd.unique(key)]


def _resamples(groups: list[np.ndarray], n: int, seed: int):
    """Yield n row-position arrays, each built from len(groups) clusters drawn with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def cluster_bootstrap_ci(df: pd.DataFrame, metric_fn: Metric, cluster_col: str = "artist",
                         n: int = 2000, ci: float = 95, seed: int = 0,
                         true_col: str = "y_true", pred_col: str = "y_pred") -> tuple[float, float, float]:
    """Percentile CI for metric_fn(y_true, y_pred) on one row per clip, resampling clusters.

    Returns (point estimate on the full data, low, high).
    """
    y, p = df[true_col].to_numpy(), df[pred_col].to_numpy()
    stats = [metric_fn(y[r], p[r]) for r in _resamples(_cluster_rows(df, cluster_col), n, seed)]
    tail = (100 - ci) / 2
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(metric_fn(y, p)), float(low), float(high)


def paired_bootstrap_diff(df: pd.DataFrame, pred_col_a: str, pred_col_b: str, metric_fn: Metric,
                          cluster_col: str = "artist", n: int = 2000, ci: float = 95, seed: int = 0,
                          true_col: str = "y_true") -> tuple[float, float, float, float]:
    """metric(A) - metric(B) with a CI, using the same resampled clusters for both models.

    Returns (diff on the full data, low, high, share of resamples with diff > 0).
    """
    y, a, b = df[true_col].to_numpy(), df[pred_col_a].to_numpy(), df[pred_col_b].to_numpy()
    diffs = np.array([metric_fn(y[r], a[r]) - metric_fn(y[r], b[r])
                      for r in _resamples(_cluster_rows(df, cluster_col), n, seed)])
    tail = (100 - ci) / 2
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(metric_fn(y, a) - metric_fn(y, b)), float(low), float(high), float(np.mean(diffs > 0))


def reliability(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: confidence = max probability, equal-width bins on [0, 1].

    One row per bin: lo, hi, n, mean confidence, accuracy (NaN for empty bins).
    """
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y_true_idx)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)  # (lo, hi], first bin [0, hi]
    rows = [{"lo": edges[k], "hi": edges[k + 1], "n": int((b == k).sum()),
             "confidence": float(conf[b == k].mean()) if (b == k).any() else np.nan,
             "accuracy": float(correct[b == k].mean()) if (b == k).any() else np.nan} for k in range(n_bins)]
    return pd.DataFrame(rows)


def ece(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: sum over bins of (n_bin / n) * |accuracy - confidence|."""
    t = reliability(probs, y_true_idx, n_bins)
    t = t[t["n"] > 0]
    return float((t["n"] / t["n"].sum() * (t["accuracy"] - t["confidence"]).abs()).sum())


def softmax_t(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax of logits / temperature, row-wise."""
    from scipy.special import softmax
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1)


def fit_temperature(logits: np.ndarray, y_true_idx: np.ndarray) -> float:
    """Temperature T > 0 minimising the mean cross-entropy of softmax(logits / T) (Guo et al. 2017)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax
    z, y = np.asarray(logits, dtype=float), np.asarray(y_true_idx)

    def nll(log_t: float) -> float:
        return float(-log_softmax(z / np.exp(log_t), axis=1)[np.arange(len(y)), y].mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
