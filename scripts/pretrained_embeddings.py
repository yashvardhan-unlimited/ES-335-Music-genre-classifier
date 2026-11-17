"""Exploratory, after the test run: pretrained PANNs CNN14 embeddings + logistic regression, split C valid.

Run from repo root: python scripts/pretrained_embeddings.py
Needs the checkpoint (config pretrained.checkpoint; download from pretrained.url) and the AudioSet
label list at ~/panns_data/class_labels_indices.csv, which panns_inference reads on import.

Loads only C train and C valid; the test part is never read, so this cannot change any test result.
Each 3 s window is resampled to 32 kHz and embedded by CNN14 (2048 numbers). Logistic regression is
tuned on valid with the same grid, scaler, scorer and soft vote as the handcrafted models
(src.train_classical.tune_and_score), so the only difference from the handcrafted logistic
regression is the input. Compared with it, the SVM and the CNN on the same resampled artists.
Writes results/val/exploratory/C_panns_logreg.json and its clip predictions; caches embeddings in
features/panns_cnn14_C.npz.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import aggregate_clips, bootstrap_resamples, macro_f1, percentile_ci  # noqa: E402
from src.features import load_clip, resample_windows, windows  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.train_classical import tune_and_score  # noqa: E402
from src.utils import get_logger, load_config, set_seed, write_json  # noqa: E402

CFG = load_config()
P = CFG["pretrained"]
VAL = Path(CFG["paths"]["results"]) / "val"
OUT = VAL / "exploratory"
CACHE = Path(CFG["paths"]["features"]) / "panns_cnn14_C.npz"
log = get_logger("pretrained")


def embed_clips(clips: pd.DataFrame) -> np.ndarray:
    """(n_clips, 10, 2048) CNN14 embeddings of each clip's ten 3 s windows; cached by clip list."""
    if CACHE.exists():
        z = np.load(CACHE)
        if list(z["clip_ids"]) == list(clips["clip_id"]):
            return z["emb"]
    from panns_inference import AudioTagging  # here: the import reads the AudioSet label file
    model = AudioTagging(checkpoint_path=P["checkpoint"], device="cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for i, path in enumerate(clips["path"]):
        y, _ = load_clip(path)
        out.append(model.inference(resample_windows(windows(y), P["sr"]))[1])
        if i % 100 == 0:
            log.info("embedded %d / %d clips", i, len(clips))
    emb = np.stack(out).astype(np.float32)
    np.savez(CACHE, emb=emb, clip_ids=clips["clip_id"].to_numpy(dtype=str))  # str, not object: loads without pickle
    return emb


def part_matrix(rows: pd.DataFrame, emb: np.ndarray, pos: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Window rows of one part -> X (2048 columns, index = clip_id) and y (genre), in row order."""
    X = pd.DataFrame(emb[[pos[c] for c in rows["clip_id"]], rows["window_idx"].to_numpy()],
                     columns=[f"e{i}" for i in range(emb.shape[2])], index=pd.Index(rows["clip_id"], name="clip_id"))
    return X, pd.Series(rows["genre"].to_numpy(), index=X.index, name="genre")


def compare(df: pd.DataFrame, cols: dict[str, list[str]]) -> dict:
    """Macro-F1 with CI per model, and new-model minus each other model, on the same artist resamples.

    cols: label -> prediction columns (several columns = runs, averaged as in the headline).
    Same rows, clusters and seed give the same resamples, so subtracting arrays is a paired bootstrap.
    """
    point = {k: float(np.mean([macro_f1(df["y_true"], df[c]) for c in v])) for k, v in cols.items()}
    boot = {k: np.mean([bootstrap_resamples(df, macro_f1, pred_col=c) for c in v], axis=0) for k, v in cols.items()}
    new = next(iter(cols))
    out = {}
    for k in cols:
        lo, hi = percentile_ci(boot[k])
        out[k] = {"macro_f1": point[k], "ci_low": lo, "ci_high": hi, "n_runs": len(cols[k])}
        if k != new:
            dlo, dhi = percentile_ci(boot[new] - boot[k])
            out[k]["new_minus_this"] = {"diff": point[new] - point[k], "ci_low": dlo, "ci_high": dhi,
                                       "share_new_better": float(np.mean(boot[new] > boot[k]))}
    return out


def main() -> None:
    set_seed(CFG["seed"])
    tr = load_split("C", "train").sort_values(["clip_id", "window_idx"])
    va = load_split("C", "valid").sort_values(["clip_id", "window_idx"])
    meta = pd.read_csv(Path(CFG["paths"]["splits"]) / "metadata.csv")
    clips = meta[meta["clip_id"].isin(set(tr["clip_id"]) | set(va["clip_id"]))].sort_values("clip_id")
    assert (clips["split_C"].isin(["train", "valid"])).all(), "a non-train/valid clip reached the embedder"
    emb = embed_clips(clips[["clip_id", "path"]].reset_index(drop=True))
    pos = {c: i for i, c in enumerate(clips["clip_id"])}
    X_tr, y_tr = part_matrix(tr, emb, pos)
    X_va, y_va = part_matrix(va, emb, pos)
    log.info("train %s, valid %s", X_tr.shape, X_va.shape)

    res = tune_and_score("logreg", X_tr, y_tr, X_va, y_va)
    with threadpool_limits(1):  # same refit as tune_and_score, kept to get the clip predictions
        est = clone(MODELS["logreg"][0]()).set_params(clf__C=res["best_params"]["C"]).fit(X_tr, y_tr)
    ids, _, soft = aggregate_clips(est.predict_proba(X_va), y_va.index)
    pred = pd.DataFrame({"clip_id": ids, "pretrained": est.classes_[soft]})

    df = pd.read_csv(VAL / "predictions" / "C_s0_logreg.csv")[["clip_id", "cluster", "y_true", "y_pred"]]
    df = df.rename(columns={"y_pred": "logreg"}).merge(pred, on="clip_id", validate="1:1")
    for name in ["svm", "cnn"]:
        for f in ([0] if name == "svm" else [0, 1, 2]):
            p = pd.read_csv(VAL / "predictions" / f"C_s{f}_{name}.csv")[["clip_id", "y_pred"]]
            df = df.merge(p.rename(columns={"y_pred": f"{name}_{f}"}), on="clip_id", validate="1:1")
    assert len(df) == 197 and np.isclose(macro_f1(df["y_true"], df["pretrained"]), res["valid_macro_f1"])

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "C_panns_logreg_predictions.csv", index=False)
    cols = {"pretrained": ["pretrained"], "logreg": ["logreg"], "svm": ["svm_0"], "cnn": ["cnn_0", "cnn_1", "cnn_2"]}
    c = compare(df, cols)
    write_json(OUT / "C_panns_logreg.json", {
        "status": "exploratory, run after the test part was scored; validation only, never scored on test",
        "model": "PANNs CNN14 (AudioSet) window embeddings + standardised logistic regression, soft vote",
        "split": "C", "part": "valid", "fit_on": "train", "checkpoint": Path(P["checkpoint"]).name,
        **{k: res[k] for k in ["best_params", "grid", "valid_macro_f1", "valid_accuracy", "per_genre_f1", "confusion"]},
        "compare": c,
    })
    for k, v in c.items():
        d = v.get("new_minus_this")
        log.info("%-10s %.3f [%.3f, %.3f]%s", k, v["macro_f1"], v["ci_low"], v["ci_high"],
                 f"  pretrained minus this {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "")


if __name__ == "__main__":
    main()
