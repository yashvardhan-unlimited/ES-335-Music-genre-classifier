"""Tune classical models on window features; score them on the valid part, per clip.

Run from repo root:
    python -m src.train_classical --split C --seed 0 --models dummy,knn,logreg,svm,xgb
Writes results/val/{split}_s{seed}_{model}.json, the figures val_knn_k.png, val_svm_heatmap.png
and val_xgb_curve.png in results/figures/ (for the models that ran) and prints a summary table
of every model with a result file for this split and seed. Only train and valid are ever loaded.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.evaluate import (ClipMacroF1Scorer, aggregate_clips, clip_scores, macro_f1,
                          make_predefined_split, window_accuracy)
from src.models import MODELS, xgb_model
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


def valid_report(probs: np.ndarray, classes: np.ndarray, X_va: pd.DataFrame, y_va: pd.Series) -> dict:
    """Clip scores (soft vote; majority as a diagnostic) and window accuracy from valid window probs."""
    ids, _, soft = aggregate_clips(probs, X_va.index)
    _, _, major = aggregate_clips(probs, X_va.index, "majority")
    y_clip = y_va.groupby(level=0).first().loc[ids].to_numpy()
    s = clip_scores(y_clip, classes[soft], labels=GENRES)
    return {
        "valid_macro_f1": s["macro_f1"],
        "valid_accuracy": s["accuracy"],
        "valid_macro_f1_majority": macro_f1(y_clip, classes[major]),
        "window_accuracy": window_accuracy(y_va, classes[probs.argmax(axis=1)]),
        "per_genre_f1": s["per_genre_f1"],
        "confusion": {"labels": GENRES, "counts": s["confusion"].tolist()},
        "n_valid_windows": len(X_va), "n_valid_clips": len(ids),
    }


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
    report = valid_report(est.predict_proba(X_va), est.classes_, X_va, y_va)
    assert np.isclose(report["valid_macro_f1"], res["mean_test_score"][int(np.argmin(res["rank_test_score"]))])
    return {
        "best_params": {k.removeprefix("clf__"): v for k, v in best.items()},
        **report,
        "n_train_windows": len(X_tr),
        "grid": [{"params": {k.removeprefix("clf__"): v for k, v in p.items()}, "valid_macro_f1": float(m)}
                 for p, m in zip(res["params"], res["mean_test_score"])],
    }


def tune_xgb(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series,
             grid: dict | None = None, fixed: dict | None = None) -> dict:
    """Manual grid for XGBoost: every setting early-stops on valid window mlogloss, the best
    setting is the one with the highest valid clip macro-F1 (ties -> first in grid order).

    grid/fixed default to config grids.xgb and xgb. The best setting's train and valid
    mlogloss per boosting round are kept under "curve" for the plot.
    """
    grid = CFG["grids"]["xgb"] if grid is None else grid
    fixed = CFG["xgb"] if fixed is None else fixed
    classes = np.array(GENRES)
    y_tr_code = pd.Categorical(y_tr, categories=GENRES).codes  # XGBoost wants labels 0..K-1
    y_va_code = pd.Categorical(y_va, categories=GENRES).codes
    assert (y_tr_code >= 0).all() and (y_va_code >= 0).all(), "label outside config genres"

    rows, best = [], None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        est = xgb_model(**fixed, **params).fit(X_tr, y_tr_code, verbose=False,
                                               eval_set=[(X_tr, y_tr_code), (X_va, y_va_code)])
        # predict_proba uses only the trees up to best_iteration (early stopping)
        report = valid_report(est.predict_proba(X_va), classes, X_va, y_va)
        rows.append({"params": params, "best_iteration": int(est.best_iteration),
                     "valid_mlogloss": float(est.best_score), "valid_macro_f1": report["valid_macro_f1"]})
        log.info("xgb %s: %d trees, valid mlogloss %.4f, clip macro-F1 %.3f", params,
                 est.best_iteration + 1, est.best_score, report["valid_macro_f1"])
        if best is None or report["valid_macro_f1"] > best[1]["valid_macro_f1"]:
            hist = est.evals_result()
            best = (rows[-1], report, {"train_mlogloss": hist["validation_0"][fixed["eval_metric"]],
                                       "valid_mlogloss": hist["validation_1"][fixed["eval_metric"]]})
    row, report, curve = best
    if row["best_iteration"] + 1 >= fixed["n_estimators"]:
        log.warning("xgb best setting used all %d trees; raise n_estimators", fixed["n_estimators"])
    return {"best_params": row["params"], "best_iteration": row["best_iteration"],
            "n_trees": row["best_iteration"] + 1, "fixed_params": fixed, **report,
            "n_train_windows": len(X_tr), "grid": rows, "curve": curve}


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


def plot_svm_heatmap(result: dict, out: Path) -> None:
    """Valid clip macro-F1 over C (rows) x gamma (columns, numeric order; 'scale' at its value)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    g_scale = result["gamma_scale_value"]
    grid = pd.DataFrame([{**g["params"], "f1": g["valid_macro_f1"]} for g in result["grid"]])
    grid["g"] = grid["gamma"].map(lambda v: g_scale if v == "scale" else float(v))
    table = grid.pivot_table(index="C", columns="g", values="f1").sort_index(ascending=False)
    col_names = [f"scale\n(={g:.4f})" if np.isclose(g, g_scale) else f"{g:g}" for g in table.columns]

    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    # colour range = the grid's own range (rounded out to 0.05), so differences are visible
    lo, hi = np.floor(grid["f1"].min() * 20) / 20, np.ceil(grid["f1"].max() * 20) / 20
    im = ax.imshow(table.to_numpy(), cmap=blues, vmin=lo, vmax=hi, aspect="auto")
    best = result["best_params"]
    for i, c in enumerate(table.index):
        for j, g in enumerate(table.columns):
            v = table.iloc[i, j]
            is_best = c == best["C"] and np.isclose(g, g_scale if best["gamma"] == "scale" else float(best["gamma"]))
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=11,
                    fontweight="bold" if is_best else "normal", color="#fcfcfb" if v > (lo + hi) / 2 else "#1f1f1d")
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=3))
    ax.set_xticks(range(len(table.columns)), col_names)
    ax.set_yticks(range(len(table.index)), [f"{c:g}" for c in table.index])
    ax.set(xlabel="gamma (RBF width; larger = narrower kernel)", ylabel="C (margin penalty)",
           title=f"RBF SVM on split C valid: best C={best['C']:g}, gamma={best['gamma']} "
                 f"({result['valid_macro_f1']:.3f})")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, label=f"valid clip macro-F1 (colour scale {lo:.2f} to {hi:.2f})", shrink=0.85)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_xgb_curve(result: dict, out: Path) -> None:
    """Train and valid window mlogloss against boosting rounds for the chosen XGBoost setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve, it = result["curve"], result["best_iteration"]
    rounds = np.arange(1, len(curve["train_mlogloss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, label, color in [("train_mlogloss", "train", "#2a78d6"), ("valid_mlogloss", "valid", "#eb6834")]:
        ax.plot(rounds, curve[key], color=color, lw=2, label=label)
        ax.text(rounds[-1], curve[key][-1], f" {label}", color="#52514e", va="center")
    ax.axvline(it + 1, color="#8a8984", lw=1.2, ls="--")
    ax.text(it + 1, ax.get_ylim()[1] * 0.95, f" best: {it + 1} trees\n valid {curve['valid_mlogloss'][it]:.3f}",
            color="#52514e", va="top")
    p = result["best_params"]
    ax.set(xlabel="boosting rounds (trees per class)", ylabel="window mlogloss",
           title=f"XGBoost on split C: max_depth={p['max_depth']}, learning_rate={p['learning_rate']}\n"
                 f"early-stopped on valid mlogloss; valid clip macro-F1 {result['valid_macro_f1']:.3f}")
    ax.set_ylim(0, None)
    ax.set_xlim(0, rounds[-1] * 1.08)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def summary_table(split: str, seed: int) -> pd.DataFrame:
    """One row per model with a valid result file for this split and seed, in MODELS order."""
    rows = []
    for name in [*MODELS, "xgb"]:
        path = VAL_DIR / f"{split}_s{seed}_{name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            rows.append({"model": name, "best_params": r["best_params"], "macro_f1": r["valid_macro_f1"],
                         "macro_f1_majority": r["valid_macro_f1_majority"], "accuracy": r["valid_accuracy"],
                         "window_acc": r["window_accuracy"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="C", choices=["A", "B", "Bp", "C"])
    ap.add_argument("--seed", type=int, default=0, help="split seed (B, Bp); ignored for A and C files")
    ap.add_argument("--models", default="dummy,knn,logreg,svm,xgb",
                    help="comma list; 'dummy' = dummy_most_frequent + dummy_stratified")
    args = ap.parse_args()
    names = [n for m in args.models.split(",")
             for n in (["dummy_most_frequent", "dummy_stratified"] if m == "dummy" else [m])]
    set_seed(CFG["seed"])
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = load_windows(args.split, "train", args.seed)
    X_va, y_va = load_windows(args.split, "valid", args.seed)
    log.info("split %s seed %d: %d train / %d valid windows", args.split, args.seed, len(X_tr), len(X_va))

    results = {}
    for name in names:
        r = tune_xgb(X_tr, y_tr, X_va, y_va) if name == "xgb" else tune_and_score(name, X_tr, y_tr, X_va, y_va)
        r = {"split": args.split, "seed": args.seed, "model": name, **r}
        if name == "svm":  # sklearn's gamma="scale" = 1 / (n_features * variance of the scaled train X)
            Xs = StandardScaler().fit_transform(X_tr)
            r["gamma_scale_value"] = float(1 / (Xs.shape[1] * Xs.var()))
        path = VAL_DIR / f"{args.split}_s{args.seed}_{name}.json"
        path.write_text(json.dumps(r, indent=1))
        results[name] = r
        log.info("%s -> %s", name, path)
    if "knn" in results:
        floor = results.get("dummy_stratified", {}).get("valid_macro_f1")
        plot_knn_k(results["knn"], FIG_DIR / "val_knn_k.png", floor)
    if "svm" in results:
        plot_svm_heatmap(results["svm"], FIG_DIR / "val_svm_heatmap.png")
    if "xgb" in results:
        plot_xgb_curve(results["xgb"], FIG_DIR / "val_xgb_curve.png")
    print(summary_table(args.split, args.seed).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
