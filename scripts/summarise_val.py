"""Collect every validation result into chosen params, a summary table and a figure.

Run from repo root: python scripts/summarise_val.py
Reads results/val/{split}_s{seed}_{model}.json for splits A (seed 0), B and Bp (seeds 0-4), C
(classical seed 0, CNN run seeds from config). Writes results/chosen_params.json
(split -> seed -> model -> params, CNN: best epoch), results/val/summary.csv and
results/figures/val_summary_splits.png. Missing result files are reported and fail the run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config  # noqa: E402

CFG = load_config()
RESULTS = Path(CFG["paths"]["results"])
VAL_DIR = RESULTS / "val"
CLASSICAL = ["dummy_most_frequent", "dummy_stratified", "knn", "logreg", "svm", "xgb"]
SPLIT_SEEDS = {"A": [0], "B": CFG["split_seeds"], "Bp": CFG["split_seeds"], "C": [0]}
LABELS = {"A": "A\nwindow-random", "B": "B\nclip-random", "Bp": "B'\nclip-random,\nfault-filtered",
          "C": "C\nartist-disjoint"}
log = get_logger("summarise_val")


def runs() -> list[tuple[str, int, str, int | None, Path]]:
    """(split, split seed, model, CNN run seed, result file) for every expected result."""
    out = []
    for split, seeds in SPLIT_SEEDS.items():
        for seed in seeds:
            out += [(split, seed, m, None, VAL_DIR / f"{split}_s{seed}_{m}.json") for m in CLASSICAL]
            run_seeds = CFG["cnn"]["seeds"] if split == "C" else [seed]
            out += [(split, seed, "cnn", r, VAL_DIR / f"{split}_s{r}_cnn.json") for r in run_seeds]
    return out


def model_params(model: str, r: dict) -> dict:
    """What was chosen on valid: grid params; XGBoost also its tree count; CNN its best epoch."""
    if model == "cnn":
        return {"best_epoch": r["best_epoch"]}
    if model == "xgb":
        return {**r["best_params"], "n_trees": r["n_trees"]}
    return r["best_params"]


def main() -> None:
    expected = runs()
    missing = [str(p) for *_, p in expected if not p.exists()]
    if missing:
        raise SystemExit("missing result files:\n" + "\n".join(missing))

    rows, chosen = [], {}
    for split, seed, model, run_seed, path in expected:
        r = json.loads(path.read_text())
        params = model_params(model, r)
        rows.append({"split": split, "split_seed": seed, "model": model, "cnn_run_seed": run_seed,
                     "valid_macro_f1": r["valid_macro_f1"], "valid_macro_f1_majority": r["valid_macro_f1_majority"],
                     "valid_accuracy": r["valid_accuracy"], "window_accuracy": r["window_accuracy"],
                     "n_valid_clips": r["n_valid_clips"], "params": json.dumps(params)})
        slot = chosen.setdefault(split, {}).setdefault(str(seed), {})
        if model == "cnn" and split == "C":
            slot.setdefault("cnn", {"best_epoch_by_run_seed": {}})["best_epoch_by_run_seed"][str(run_seed)] = r["best_epoch"]
        else:
            slot[model] = params
    epochs = chosen["C"]["0"]["cnn"]["best_epoch_by_run_seed"]
    chosen["C"]["0"]["cnn"]["median_best_epoch"] = float(np.median(list(epochs.values())))

    df = pd.DataFrame(rows)
    df.to_csv(VAL_DIR / "summary.csv", index=False)
    (RESULTS / "chosen_params.json").write_text(json.dumps(chosen, indent=1))
    log.info("-> %s, %s", VAL_DIR / "summary.csv", RESULTS / "chosen_params.json")

    agg = (df.groupby(["model", "split"], sort=False)["valid_macro_f1"]
           .agg(["mean", "std", "count"]).reset_index())
    table = agg.assign(cell=lambda d: d["mean"].map("{:.3f}".format)
                       + np.where(d["count"] > 1, " +/- " + d["std"].map("{:.3f}".format), ""))
    print(table.pivot(index="model", columns="split", values="cell")
          .reindex(index=CLASSICAL + ["cnn"], columns=list(SPLIT_SEEDS)).to_string())
    plot(agg, RESULTS / "figures" / "val_summary_splits.png")


def plot(agg: pd.DataFrame, out: Path) -> None:
    """Valid clip macro-F1 per model across splits A -> B -> B' -> C (mean +/- sd over seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(SPLIT_SEEDS)
    x = np.arange(len(splits))
    models = [("knn", "kNN", "#2a78d6"), ("logreg", "logistic regression", "#eb6834"),
              ("svm", "RBF SVM", "#1baf7a"), ("xgb", "XGBoost", "#eda100"), ("cnn", "CNN (log-mel)", "#e87ba4")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    offsets = np.linspace(-0.12, 0.12, len(models))
    for (m, label, color), dx in zip(models, offsets):
        a = agg[agg["model"] == m].set_index("split").reindex(splits)
        ax.errorbar(x + dx, a["mean"], yerr=a["std"].fillna(0), color=color, lw=2, marker="o", ms=7,
                    capsize=3, label=label)
        ax.text(x[-1] + dx + 0.08, a["mean"].iloc[-1], f"{a['mean'].iloc[-1]:.3f}", color="#52514e",
                fontsize=8, va="center")
    floor = agg[agg["model"] == "dummy_stratified"].set_index("split").reindex(splits)["mean"]
    ax.plot(x, floor, color="#8a8984", lw=1.2, ls="--", label="stratified dummy (floor)")
    ax.set_xticks(x, [LABELS[s] for s in splits])
    ax.set_xlim(-0.4, len(splits) - 0.4)
    ax.set(ylabel="valid clip macro-F1 (soft vote)", ylim=(0, 1),
           title="Validation macro-F1 per model as leakage is removed\n"
                 "(mean +/- sd over split seeds: B, B' 5 seeds; C CNN 3 run seeds; A one seed)")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
