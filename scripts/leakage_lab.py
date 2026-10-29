"""Leakage lab: the correct SVM tuning on split C against three deliberate mistakes.

Each variant changes exactly one thing from the correct setup (scaler inside the
pipeline, train -> valid predefined split, clip macro-F1). For each we record the
chosen params, the score the procedure itself reports, and the true valid clip
macro-F1 of those params under the correct setup. Uses split C train and valid only.

Run from repo root: python scripts/leakage_lab.py
Writes results/val/leakage_lab.json and results/figures/leakage_lab.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import ClipMacroF1Scorer, make_predefined_split  # noqa: E402
from src.models import svm_pipeline  # noqa: E402
from src.train_classical import FIG_DIR, VAL_DIR, load_windows, param_grid  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

CFG = load_config()
log = get_logger("leakage_lab")


def search(estimator, X: pd.DataFrame, y: pd.Series, cv, scoring) -> GridSearchCV:
    """SVM grid search (config grid), no refit: we only need the scores per setting."""
    return GridSearchCV(estimator, param_grid("svm"), scoring=scoring, cv=cv, refit=False, n_jobs=-1).fit(X, y)


def key(params: dict) -> str:
    """Stable text key for one grid setting, e.g. 'C=100, gamma=0.001'."""
    return ", ".join(f"{k.removeprefix('clf__')}={v}" for k, v in sorted(params.items()))


def summarise(name: str, what: str, gs: GridSearchCV, metric: str, truth: dict[str, float]) -> dict:
    """Chosen params (ties -> first), its reported selection score, and its true valid clip macro-F1."""
    res = gs.cv_results_
    best = int(np.argmin(res["rank_test_score"]))
    chosen = res["params"][best]
    return {
        "name": name, "mistake": what, "selection_metric": metric,
        "chosen_params": {k.removeprefix("clf__"): v for k, v in chosen.items()},
        "reported_score": float(res["mean_test_score"][best]),
        "true_valid_clip_macro_f1": truth[key(chosen)],
        # does the selection score rank the 16 settings like the true score does? (1 = same order)
        "spearman_selection_vs_true": float(spearmanr(res["mean_test_score"],
                                                      [truth[key(p)] for p in res["params"]]).statistic),
        "grid": [{"params": key(p), "selection_score": float(s), "true_valid_clip_macro_f1": truth[key(p)]}
                 for p, s in zip(res["params"], res["mean_test_score"])],
    }


def sibling_share(clip_ids: np.ndarray, cv: KFold) -> float:
    """Share of held-out windows whose clip also has a window in that fold's training part."""
    hits = [np.isin(clip_ids[te], clip_ids[tr]) for tr, te in cv.split(clip_ids)]
    return float(np.concatenate(hits).mean())


def plot(variants: list[dict], out: Path) -> None:
    """Reported vs true score per variant, as paired bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["correct", "(1) scaler fit\non train+valid", "(2) KFold over\ntrain windows",
              "(3) select by\nwindow accuracy"]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for dx, field, label, color in [(-0.2, "reported_score", "reported by the procedure", "#2a78d6"),
                                    (0.2, "true_valid_clip_macro_f1", "true valid clip macro-F1", "#eb6834")]:
        vals = [v[field] for v in variants]
        ax.bar(x + dx, vals, width=0.38, color=color, label=label, edgecolor="#fcfcfb", linewidth=2)
        for xi, val in zip(x + dx, vals):
            ax.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#1f1f1d",
                    bbox={"facecolor": "#ffffff", "edgecolor": "none", "pad": 1})
    best = variants[0]["true_valid_clip_macro_f1"]
    ax.axhline(best, color="#8a8984", lw=1.2, ls="--", zorder=0, label=f"true score of the correct setup ({best:.3f})")
    ax.set_xticks(x, [f"{lab}\nC={v['chosen_params']['C']:g}, gamma={v['chosen_params']['gamma']}"
                      for lab, v in zip(labels, variants)], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("SVM on split C: what each procedure reports vs what it really gets on valid\n"
                 "(variant 3 reports valid window accuracy, the metric it selected by)")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    set_seed(CFG["seed"])
    X_tr, y_tr = load_windows("C", "train")
    X_va, y_va = load_windows("C", "valid")
    X, y = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
    predefined = make_predefined_split(["train"] * len(X_tr) + ["valid"] * len(X_va))
    clip_f1 = ClipMacroF1Scorer()

    # correct: scaler inside the pipeline (fit on train rows of each fit), train -> valid, clip macro-F1.
    # Its grid IS the true valid clip macro-F1 of every setting, so it is the lookup for all variants.
    log.info("correct setup")
    gs = search(svm_pipeline(), X, y, predefined, clip_f1)
    truth = {key(p): float(s) for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])}
    saved = {key({f"clf__{k}": v for k, v in g["params"].items()}): g["valid_macro_f1"]
             for g in json.loads((VAL_DIR / "C_s0_svm.json").read_text())["grid"]}
    assert all(np.isclose(truth[k], saved[k]) for k in truth), "correct grid differs from C_s0_svm.json"
    variants = [summarise("correct", "none", gs, "valid clip macro-F1", truth)]

    # (1) scaler fit on train+valid, applied before a pipeline without its own scaler
    log.info("mistake 1: global scaler")
    leaky = StandardScaler().fit(X)
    X_leaky = pd.DataFrame(leaky.transform(X), index=X.index, columns=X.columns)
    no_scaler = svm_pipeline().set_params(scale="passthrough")
    gs1 = search(no_scaler, X_leaky, y, predefined, clip_f1)
    variants.append(summarise("scaler_on_train_valid", "StandardScaler fit on train+valid before the model",
                              gs1, "valid clip macro-F1 (leaky scaling)", truth))

    # (2) 5-fold CV with shuffling over train windows: windows of one clip land on both sides
    log.info("mistake 2: KFold over windows")
    kfold = KFold(5, shuffle=True, random_state=CFG["seed"])
    gs2 = search(svm_pipeline(), X_tr, y_tr, kfold, clip_f1)
    variants.append(summarise("kfold_windows", "KFold(5, shuffle=True) over train windows, no grouping",
                              gs2, "5-fold CV clip macro-F1 over train windows", truth))

    # (3) correct split, but select by window accuracy
    log.info("mistake 3: window accuracy")
    gs3 = search(svm_pipeline(), X, y, predefined, "accuracy")
    variants.append(summarise("window_accuracy", "hyperparameters chosen by valid window accuracy",
                              gs3, "valid window accuracy", truth))

    # diagnostics for the lesson
    shift = (leaky.mean_ - X_tr.mean().to_numpy()) / X_tr.std(ddof=0).to_numpy()
    ratio = np.sqrt(leaky.var_) / X_tr.std(ddof=0).to_numpy()
    p1 = variants[1]["chosen_params"]
    # with a scaler still inside the pipeline, pre-scaling is an affine map the inner scaler undoes
    # (up to floating point), so the leak only bites when the model itself has no scaler
    inner = svm_pipeline(**p1).fit(X_leaky.iloc[:len(X_tr)], y_tr).predict_proba(X_leaky.iloc[len(X_tr):])
    clean = svm_pipeline(**p1).fit(X_tr, y_tr).predict_proba(X_va)
    diagnostics = {
        "mistake1_mean_shift_in_train_sd": {"max_abs": float(np.abs(shift).max()),
                                            "median_abs": float(np.median(np.abs(shift)))},
        "mistake1_sd_ratio_all_over_train": {"min": float(ratio.min()), "max": float(ratio.max())},
        "mistake1_inner_scaler_max_prob_diff": float(np.abs(inner - clean).max()),
        "mistake1_inner_scaler_changed_window_predictions": int((inner.argmax(1) != clean.argmax(1)).sum()),
        "mistake2_heldout_windows_with_sibling_in_train_fold": sibling_share(X_tr.index.to_numpy(), kfold),
        "n_train_windows": len(X_tr), "n_valid_windows": len(X_va),
    }
    out = {"split": "C", "model": "svm", "grid": {k: v for k, v in CFG["grids"]["svm"].items()},
           "variants": variants, "diagnostics": diagnostics}
    path = VAL_DIR / "leakage_lab.json"
    path.write_text(json.dumps(out, indent=1))
    plot(variants, FIG_DIR / "leakage_lab.png")
    log.info("-> %s, %s", path, FIG_DIR / "leakage_lab.png")
    print(pd.DataFrame([{"variant": v["name"], "chosen": v["chosen_params"], "reported": v["reported_score"],
                         "true_valid_clip_f1": v["true_valid_clip_macro_f1"]} for v in variants])
          .to_string(index=False, float_format="%.3f"))
    print(json.dumps(diagnostics, indent=1))


if __name__ == "__main__":
    main()
