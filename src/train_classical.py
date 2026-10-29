"""Tune classical models on window features; score them on the valid part, per clip.

Run from repo root:
    python -m src.train_classical --split C --seed 0 --models dummy,knn,logreg
Writes results/val/{split}_s{seed}_{model}.json, results/figures/val_knn_k.png
(when knn ran) and prints a summary table. Only train and valid are ever loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from src.evaluate import (ClipMacroF1Scorer, aggregate_clips, clip_scores, macro_f1,
                          make_predefined_split, window_accuracy)
from src.models import MODELS
from src.splits import load_split
from src.utils import get_logger, load_config, set_seed

CFG = load_config()
GENRES = CFG["genres"]
VAL_DIR = Path(CFG["paths"]["results"]) / "val"
FIG_DIR = Path(CFG["paths"]["results"]) / "figures"
log = get_logger("train_classical")


def load_windows(split: str, part: str, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Handcrafted features of one split part: X (index = clip_id, 57 named columns), y (genre)."""
    feats = Path(CFG["paths"]["features"])
    index = pd.read_csv(feats / "index.csv").reset_index(names="row")
    names = json.loads((feats / "feature_names.json").read_text())
    rows = load_split(split, part, seed).merge(index[["clip_id", "window_idx", "row"]],
                                               on=["clip_id", "window_idx"], how="left", validate="1:1")
    assert rows["row"].notna().all(), "split rows missing from features/index.csv"
    X = pd.DataFrame(np.load(feats / "handcrafted.npy")[rows["row"]], columns=names,
                     index=pd.Index(rows["clip_id"], name="clip_id"))
    return X, pd.Series(rows["genre"].to_numpy(), index=X.index, name="genre")


def param_grid(name: str) -> dict:
    """Config grid for a model, keyed for its Pipeline step ("clf__"); {} = nothing to tune."""
    key = MODELS[name][1]
    return {f"clf__{k}": v for k, v in CFG["grids"][key].items()} if key else {}


def tune_and_score(name: str, X_tr: pd.DataFrame, y_tr: pd.Series,
                   X_va: pd.DataFrame, y_va: pd.Series) -> dict:
    """Grid search on the single train->valid fold, then refit the best on train and score valid."""
    X = pd.concat([X_tr, X_va])
    y = pd.concat([y_tr, y_va])
    cv = make_predefined_split(["train"] * len(X_tr) + ["valid"] * len(X_va))
    gs = GridSearchCV(MODELS[name][0](), param_grid(name), scoring=ClipMacroF1Scorer(),
                      cv=cv, refit=False, n_jobs=-1)
    gs.fit(X, y)
    res = gs.cv_results_
    best = res["params"][int(np.argmin(res["rank_test_score"]))]  # first of any ties

    # refit on train only, so valid stays unseen by the model we score. One BLAS thread, as in
    # the n_jobs=-1 grid workers: lbfgs results depend on the thread count otherwise.
    with threadpool_limits(1):
        est = clone(gs.estimator).set_params(**best).fit(X_tr, y_tr)
    probs = est.predict_proba(X_va)
    classes = est.classes_
    ids, _, soft = aggregate_clips(probs, X_va.index)
    _, _, major = aggregate_clips(probs, X_va.index, "majority")
    y_clip = y_va.groupby(level=0).first().loc[ids].to_numpy()
    s = clip_scores(y_clip, classes[soft], labels=GENRES)
    assert np.isclose(s["macro_f1"], res["mean_test_score"][int(np.argmin(res["rank_test_score"]))])
    return {
        "best_params": {k.removeprefix("clf__"): v for k, v in best.items()},
        "valid_macro_f1": s["macro_f1"],
        "valid_accuracy": s["accuracy"],
        "valid_macro_f1_majority": macro_f1(y_clip, classes[major]),
        "window_accuracy": window_accuracy(y_va, classes[probs.argmax(axis=1)]),
        "per_genre_f1": s["per_genre_f1"],
        "confusion": {"labels": GENRES, "counts": s["confusion"].tolist()},
        "n_train_windows": len(X_tr), "n_valid_windows": len(X_va), "n_valid_clips": len(ids),
        "grid": [{"params": {k.removeprefix("clf__"): v for k, v in p.items()}, "valid_macro_f1": float(m)}
                 for p, m in zip(res["params"], res["mean_test_score"])],
    }


def plot_knn_k(result: dict, out: Path, floor: float | None = None) -> None:
    """Valid clip macro-F1 against k, one line per weighting; y from 0, optional dummy floor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = pd.DataFrame([{**g["params"], "f1": g["valid_macro_f1"]} for g in result["grid"]])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (w, sub), color in zip(grid.groupby("weights"), ["#2a78d6", "#eb6834"]):
        sub = sub.sort_values("n_neighbors")
        ax.plot(sub["n_neighbors"], sub["f1"], color=color, lw=2, marker="o", ms=7,
                markeredgecolor="#fcfcfb", markeredgewidth=2, label=w)
    ks = sorted(grid["n_neighbors"].unique())
    if floor is not None:
        ax.axhline(floor, color="#8a8984", lw=1.2, ls="--")
        ax.text(ks[-1], floor + 0.01, f"stratified dummy {floor:.3f}", color="#52514e", ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set(xlabel="k (neighbours, log scale)", ylabel="valid clip macro-F1",
           title=f"kNN on split C valid: best k={result['best_params']['n_neighbors']}, "
                 f"{result['best_params']['weights']} ({result['valid_macro_f1']:.3f})")
    ax.set_ylim(0, max(0.6, grid["f1"].max() + 0.05))
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title="weights", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="C", choices=["A", "B", "Bp", "C"])
    ap.add_argument("--seed", type=int, default=0, help="split seed (B, Bp); ignored for A and C files")
    ap.add_argument("--models", default="dummy,knn,logreg",
                    help="comma list; 'dummy' = dummy_most_frequent + dummy_stratified")
    args = ap.parse_args()
    names = [n for m in args.models.split(",")
             for n in (["dummy_most_frequent", "dummy_stratified"] if m == "dummy" else [m])]
    set_seed(CFG["seed"])
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = load_windows(args.split, "train", args.seed)
    X_va, y_va = load_windows(args.split, "valid", args.seed)
    log.info("split %s seed %d: %d train / %d valid windows", args.split, args.seed, len(X_tr), len(X_va))

    summary, results = [], {}
    for name in names:
        r = {"split": args.split, "seed": args.seed, "model": name,
             **tune_and_score(name, X_tr, y_tr, X_va, y_va)}
        path = VAL_DIR / f"{args.split}_s{args.seed}_{name}.json"
        path.write_text(json.dumps(r, indent=1))
        results[name] = r
        log.info("%s -> %s", name, path)
        summary.append({"model": name, "best_params": r["best_params"], "macro_f1": r["valid_macro_f1"],
                        "macro_f1_majority": r["valid_macro_f1_majority"], "accuracy": r["valid_accuracy"],
                        "window_acc": r["window_accuracy"]})
    if "knn" in results:
        floor = results.get("dummy_stratified", {}).get("valid_macro_f1")
        plot_knn_k(results["knn"], FIG_DIR / "val_knn_k.png", floor)
    print(pd.DataFrame(summary).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
