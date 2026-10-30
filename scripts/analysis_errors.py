"""Error analysis on split C. Implemented so far: part 4, permutation importance on VALIDATION.

Parts 1-3, 5, 6 (confusion cases, error log, universal errors, Grad-CAM, robustness) use the
test part and are added after the pre-test-freeze tag exists.

4 permutation importance for the tuned SVM and XGBoost (refit on train, as in tuning), scored
  by valid clip macro-F1 (soft vote), 10 repeats: per feature (sklearn) and per feature group
  (all columns of a group shuffled with one shared row permutation). XGBoost's built-in gain
  importance is stored alongside for comparison.

Run from repo root: python scripts/analysis_errors.py --parts 4
Writes results/val/permutation_importance.json and results/figures/permutation_importance.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import ClipMacroF1Scorer  # noqa: E402
from src.features import feature_group  # noqa: E402
from src.models import svm_pipeline, xgb_model  # noqa: E402
from src.train_classical import load_windows  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

CFG = load_config()
GENRES = np.array(CFG["genres"])
VAL_DIR = Path(CFG["paths"]["results"]) / "val"
FIG_DIR = Path(CFG["paths"]["results"]) / "figures"
N_REPEATS = CFG["analysis"]["permutation_repeats"]
log = get_logger("analysis_errors")


class Decoded:
    """Wrap an XGBoost model fit on integer codes so it predicts genre names (for the clip scorer)."""

    def __init__(self, est):
        self.est, self.classes_ = est, GENRES

    def fit(self, *_):
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)


def group_importance(est, X: pd.DataFrame, y: pd.Series, scorer, seed: int) -> dict[str, dict]:
    """Drop in score when all columns of one feature group get the same random row permutation."""
    base = scorer(est, X, y)
    groups = pd.Series([feature_group(c) for c in X.columns], index=X.columns)
    rng = np.random.default_rng(seed)
    out = {}
    for g in CFG["analysis"]["feature_groups"]:
        cols = groups.index[groups == g]
        drops = []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - scorer(est, Xp, y))
        out[g] = {"n_features": len(cols), "mean": float(np.mean(drops)), "sd": float(np.std(drops, ddof=1))}
    return out


def part_permutation() -> dict:
    X_tr, y_tr = load_windows("C", "train")
    X_va, y_va = load_windows("C", "valid")
    scorer, seed = ClipMacroF1Scorer(), CFG["seed"]
    svm = json.loads((VAL_DIR / "C_s0_svm.json").read_text())
    xgb = json.loads((VAL_DIR / "C_s0_xgb.json").read_text())
    with threadpool_limits(1):
        svm_est = svm_pipeline(**svm["best_params"]).fit(X_tr, y_tr)
    fixed = {k: v for k, v in xgb["fixed_params"].items() if k not in ("n_estimators", "early_stopping_rounds")}
    xgb_raw = xgb_model(**fixed, **xgb["best_params"], n_estimators=xgb["n_trees"]).fit(
        X_tr, pd.Categorical(y_tr, categories=GENRES).codes)
    models = {"svm": (svm_est, svm), "xgb": (Decoded(xgb_raw), xgb)}

    out = {"split": "C", "part": "valid", "metric": "clip macro-F1 (soft vote)", "n_repeats": N_REPEATS,
           "models": {}}
    for name, (est, saved) in models.items():
        t0 = time.perf_counter()
        base = scorer(est, X_va, y_va)
        assert np.isclose(base, saved["valid_macro_f1"]), f"{name} refit differs from its tuning result"
        pi = permutation_importance(est, X_va, y_va, scoring=scorer, n_repeats=N_REPEATS, random_state=seed)
        per_feature = (pd.DataFrame({"feature": X_va.columns, "mean": pi.importances_mean, "sd": pi.importances_std})
                       .assign(group=lambda d: d["feature"].map(feature_group))
                       .sort_values("mean", ascending=False))
        out["models"][name] = {"baseline_valid_macro_f1": float(base),
                               "per_feature": per_feature.to_dict(orient="records"),
                               "per_group": group_importance(est, X_va, y_va, scorer, seed)}
        if name == "xgb":
            gain = pd.Series(xgb_raw.get_booster().get_score(importance_type="gain")).reindex(X_va.columns).fillna(0)
            out["models"][name]["builtin_gain"] = (gain / gain.sum()).sort_values(ascending=False).to_dict()
        log.info("%s: baseline %.3f, %.0f s", name, base, time.perf_counter() - t0)
    (VAL_DIR / "permutation_importance.json").write_text(json.dumps(out, indent=1))
    plot(out, FIG_DIR / "permutation_importance.png")
    return out


def plot(out: dict, path: Path) -> None:
    """Per model: feature-group importance (top) and the 12 most important single features (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [1, 1.6]})
    for col, (name, title, color) in enumerate([("svm", "RBF SVM", "#2a78d6"), ("xgb", "XGBoost", "#eb6834")]):
        m = out["models"][name]
        g = pd.DataFrame(m["per_group"]).T.sort_values("mean")
        ax = axes[0, col]
        ax.barh([f"{i} ({int(r['n_features'])})" for i, r in g.iterrows()], g["mean"], xerr=g["sd"], color=color,
                height=0.6, capsize=3)
        ax.set(title=f"{title}: feature groups (baseline {m['baseline_valid_macro_f1']:.3f})",
               xlabel="drop in valid clip macro-F1 when the group is shuffled")
        f = pd.DataFrame(m["per_feature"]).head(12).iloc[::-1]
        ax = axes[1, col]
        ax.barh(f["feature"], f["mean"], xerr=f["sd"], color=color, height=0.6, capsize=3)
        ax.set(title=f"{title}: 12 most important single features", xlabel="drop in valid clip macro-F1")
        for a in axes[:, col]:
            a.axvline(0, color="#52514e", lw=1)
            a.grid(axis="x", color="#e4e3df", lw=0.8)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    fig.suptitle(f"Permutation importance on split C valid ({out['n_repeats']} repeats, mean +/- sd)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


PARTS = {4: part_permutation}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="4", help="implemented: 4 (others need the test freeze)")
    args = ap.parse_args()
    set_seed(CFG["seed"])
    for p in map(int, args.parts.split(",")):
        PARTS[p]()


if __name__ == "__main__":
    main()
