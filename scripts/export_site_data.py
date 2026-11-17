"""Export the numbers the public site shows into docs/data/site.json, and copy its figures to docs/img.

Run from repo root: python scripts/export_site_data.py
Reads results/test (after the pre-test-freeze run) or, until then, results/val, and says which in
"part". Only numbers, clip ids and figures go to docs/: no audio, no model weights, no lessons.
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config  # noqa: E402

CFG = load_config()
RESULTS = Path(CFG["paths"]["results"])
SITE_DATA, SITE_IMG = Path(CFG["paths"]["site_data"]), Path(CFG["paths"]["site_img"])
GENRES = CFG["genres"]
LABELS = {"dummy_most_frequent": "Dummy (most frequent)", "dummy_stratified": "Dummy (stratified)",
          "knn": "k-nearest neighbours", "logreg": "Logistic regression", "svm": "RBF SVM",
          "xgb": "XGBoost", "cnn": "CNN on log-mels"}
SPLITS = {
    "A": ("A", "Window-random", "3 s windows shuffled at random: windows of one clip sit in train and test."),
    "B": ("B", "Clip-random", "Whole clips at random from all 999, stratified by genre (5 seeds)."),
    "Bp": ("B′", "Clip-random, fault-filtered", "As B, but only from the 929 clips left after removing "
           "duplicates, mislabels and distortions (5 seeds)."),
    "D": ("D", "Artist-grouped, random", "As B′, but every artist's clips stay in one part (5 seeds)."),
    "C": ("C", "Published artist-disjoint", "The published fault-filtered split (Kereliuk, Sturm & Larsen 2015). "
          "Main result."),
}
FIGURES = ["mel_per_genre.png", "val_svm_heatmap.png", "cnn_curves_C.png", "permutation_importance.png",
           "analysis_learning_curves.png", "analysis_calibration.png", "leakage_lab.png", "rms_by_genre.png"]
log = get_logger("export_site_data")


def source() -> tuple[str, Path]:
    """('test', results/test) once the test run exists, else ('valid', results/val)."""
    test = RESULTS / "test"
    return ("test", test) if (test / "headline.json").exists() else ("valid", RESULTS / "val")


def r3(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def ladder(scores: pd.DataFrame, head: dict) -> dict:
    """model -> split -> mean, CI, sd over runs, and every run's score (train-only ladder models)."""
    lad = scores[scores["set"].eq("ladder")] if "set" in scores else scores
    out: dict = {}
    for key, v in head["ladder"].items():
        split, model = key.split("/")
        runs = lad[(lad["split"] == split) & (lad["model"] == model)]
        out.setdefault(model, {})[split] = {
            "mean": r3(v["macro_f1"]), "ci_low": r3(v["ci_low"]), "ci_high": r3(v["ci_high"]),
            "sd": r3(v["sd_over_runs"]), "n_runs": v["n_runs"],
            "runs": [{"seed": int(s), "macro_f1": r3(f)} for s, f in zip(runs["run_seed"], runs["macro_f1"])]}
    return out


def c_extras(scores: pd.DataFrame, head: dict, final: bool) -> dict:
    """model -> accuracy (mean of runs), valid macro-F1 (mean of runs) and train-only macro-F1 on C."""
    rows = scores[(scores["split"] == "C") & scores["set"].eq("final" if final else "ladder")]
    val = pd.read_csv(RESULTS / "val" / "summary.csv").query("split == 'C'")
    return {m: {"accuracy": r3(rows.loc[rows["model"] == m, "accuracy"].mean()),
                "valid_macro_f1": r3(val.loc[val["model"] == m, "valid_macro_f1"].mean()),
                "train_only_macro_f1": r3(head["ladder"][f"C/{m}"]["macro_f1"])}
            for m in LABELS if f"C/{m}" in head["ladder"]}


def c_block(summary: dict, extras: dict) -> dict:
    """Models on C (mean of runs), paired CNN comparisons and C-strict, rounded."""
    models = [{"model": m, "label": LABELS[m], **{k: r3(v[k]) for k in ("macro_f1", "ci_low", "ci_high")},
               "n_runs": v["n_runs"], **extras.get(m, {})}
              for key, v in summary["ladder"].items() for split, m in [key.split("/")] if split == "C"]
    return {"models": sorted(models, key=lambda d: -d["macro_f1"]),
            "paired": {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in summary["paired_on_C"].items()},
            "strict": {m: {k: (r3(v) if isinstance(v, float) else v) for k, v in d.items()}
                       for m, d in summary["C_strict"].items()}}


def confusion_runs(pred_dir: Path, prefix: str, details: dict) -> dict:
    """For each run on C: confusion counts, per-genre F1, support, and the clips in every cell
    as [clip id, top score, artist] (gtzan-bind name, "" when unknown)."""
    artist = pd.read_csv(Path(CFG["paths"]["splits"]) / "metadata.csv").set_index("clip_id")["artist"].fillna("")
    out = {}
    for path in sorted(pred_dir.glob(f"{prefix}C_s*_*.csv")):
        model = path.stem.split("_", 3 if prefix else 2)[-1]
        seed = int(path.stem.removeprefix(prefix).split("_")[1][1:])
        if model.startswith("dummy"):
            continue
        df = pd.read_csv(path)
        probs = df[[f"p_{g}" for g in GENRES]]
        df["conf"] = probs.max(axis=1)
        cells: dict = {}
        for (t, p), g in df.groupby(["y_true", "y_pred"]):
            cells[f"{t}|{p}"] = [[c, round(float(x), 3), artist.get(c, "")] for c, x in
                                 g.sort_values("conf", ascending=False)[["clip_id", "conf"]].itertuples(index=False)]
        d = details[f"{model}_s{seed}"]
        out[f"{model}_s{seed}"] = {"model": model, "seed": seed, "label": LABELS[model] + (
            f" (seed {seed})" if model == "cnn" else ""), "confusion": d["confusion"],
            "per_genre_f1": {g: r3(f) for g, f in d["per_genre_f1"].items()}, "support": d["support"],
            "cells": cells}
    return out


def splits_info() -> dict:
    """Per split: labels, description, and the test clips sharing an artist cluster with train."""
    summ = json.loads((RESULTS / "split_summary.json").read_text(encoding="utf-8"))
    out = {}
    for key, (short, name, desc) in SPLITS.items():
        rows = [v for k, v in summ.items() if isinstance(v, dict) and (k == key or k.startswith(f"{key}_seed"))]
        out[key] = {"short": short, "name": name, "description": desc,
                    "test_share_artist_in_train": r3(sum(r["test_clips_credit_cluster_in_train"] for r in rows)
                                                     / sum(r["clips_per_part"]["test"] for r in rows)),
                    "test_clips_also_in_train": rows[0]["test_clips_also_in_train"]}
    return out


def main() -> None:
    part, src = source()
    head = json.loads((src / "headline.json").read_text())
    scores = pd.read_csv(src / "scores.csv")
    final = part == "test" and "final_on_C" in head
    c_summary = head["final_on_C"] if final else head
    c = c_block(c_summary, c_extras(scores, head, final))
    if final:  # train-only models on C, so the site can show what refitting on train+valid changed
        c["paired_train_only"] = {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in head["paired_on_C"].items()}
    cnn = next(m for m in c["models"] if m["model"] == "cnn")
    floor = next(m for m in c["models"] if m["model"] == "dummy_stratified")
    data = json.loads((RESULTS / "data_summary.json").read_text())
    site = {
        "part": part, "generated": date.today().isoformat(),
        "fit_on_C": "train+valid" if final else "train",
        "headline": {**cnn, "floor": floor["macro_f1"], "n_clips": int(scores.query("split == 'C'")["n_clips"].iloc[0])},
        "c": c, "ladder": ladder(scores, head),
        "drops": {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in head["drops"].items()},
        "splits": splits_info(),
        "confusion": confusion_runs(src / "predictions", "final_" if final else "", c_summary["C_details"]),
        "genres": GENRES, "labels": LABELS,
        "data": {"n_files": data["n_files"], "n_loadable": data["n_loadable"], "failures": data["failures"]},
    }
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "site.json").write_text(json.dumps(site, indent=1), encoding="utf-8")
    SITE_IMG.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copy2(RESULTS / "figures" / name, SITE_IMG / name)
    log.info("%s numbers -> %s, %d figures -> %s", part, SITE_DATA / "site.json", len(FIGURES), SITE_IMG)


if __name__ == "__main__":
    main()
