"""Score every selected model on one part with artist-cluster bootstrap CIs (the only script that scores test).

Run from repo root:
    python scripts/evaluate_part.py --part valid    # rebuilt predictions must equal every stored valid score
    python scripts/evaluate_part.py --part test     # needs the git tag pre-test-freeze; run it once
Two sets of models (DECISIONS.md, 2026-11-14, test protocol):
    ladder  the selected models, trained on train only, on every split (A, B, B', D, C): the leakage ladder
    final   test only: the same settings refit on train+valid, on C: the headline (CNN = mean of its runs)
Writes, under results/val or results/test:
    predictions/{split}_s{seed}_{model}.csv  one row per clip: cluster, y_true, y_pred, p_<genre>
                                             (final models: final_C_s{seed}_{model}.csv)
    scores.csv      one row per run: set, macro-F1 with its 95% CI, accuracy
    headline.json   seed means with CIs, leakage drops between splits, paired comparisons on C,
                    C-strict (clips whose artist cluster is not in C train), per-genre details on C
Bootstrap draws depend on the split and split seed only, so every model scored on the same clips gets
the same resamples (paired comparisons), and different clip sets get unrelated draws.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import bootstrap_resamples, clip_scores, macro_f1, percentile_ci  # noqa: E402
from src.predict import VAL_DIR, clip_predictions, stored_result  # noqa: E402
from src.splits import credit_clusters, is_test_unlocked  # noqa: E402
from src.utils import get_logger, load_config, write_json  # noqa: E402

CFG = load_config()
RESULTS = Path(CFG["paths"]["results"])
OUT_DIR = {"valid": VAL_DIR, "test": RESULTS / "test"}
SPLIT_SEEDS = {"A": [0], "B": CFG["split_seeds"], "Bp": CFG["split_seeds"], "D": CFG["split_seeds"], "C": [0]}
MODELS = ["dummy_most_frequent", "dummy_stratified", "knn", "logreg", "svm", "xgb", "cnn"]
DROPS = [("A", "B"), ("B", "Bp"), ("Bp", "D"), ("D", "C"), ("Bp", "C")]
log = get_logger("evaluate_part")

Run = tuple[str, int, str, int]  # (split, split seed, model, file seed)


def runs(splits: list[str] = list(SPLIT_SEEDS)) -> list[Run]:
    """Every expected run; the CNN on C has its config run seeds, all on split seed 0."""
    out = []
    for split in splits:
        for s in SPLIT_SEEDS[split]:
            for m in MODELS:
                file_seeds = CFG["cnn"]["seeds"] if (m == "cnn" and split == "C") else [s]
                out += [(split, s, m, f) for f in file_seeds]
    return out


def result_file(run: Run, refit: bool) -> Path:
    """The file that must exist before a run can be scored (refit CNN: its own refit result)."""
    split, _, model, f = run
    suffix = "_refit" if (refit and model == "cnn") else ""
    return VAL_DIR / f"{split}_s{f}_{model}{suffix}.json"


def boot_seed(split: str, split_seed: int) -> int:
    """One bootstrap seed per clip set: the same for all models on it, different between clip sets."""
    return CFG["seed"] + 1000 * list(SPLIT_SEEDS).index(split) + split_seed


def interval(point: float, stats: np.ndarray) -> dict:
    low, high = percentile_ci(stats)
    return {"macro_f1": point, "ci_low": low, "ci_high": high}


def score_runs(part: str, todo: list[Run], refit: bool, allow_missing: bool) -> tuple[list[dict], dict]:
    """Predict, check (valid) and bootstrap each run. Returns score rows and {run: (df, point, stats)}."""
    pred_dir = OUT_DIR[part] / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, data, missing = [], {}, []
    for run in todo:
        split, s, model, f = run
        if not result_file(run, refit).exists():
            missing.append(result_file(run, refit).stem)
            continue
        df = clip_predictions(split, f, model, part, refit=refit)
        df.to_csv(pred_dir / f"{'final_' if refit else ''}{split}_s{f}_{model}.csv", index=False)
        point = macro_f1(df["y_true"], df["y_pred"])
        if part == "valid":
            stored = stored_result(split, f, model)["valid_macro_f1"]
            if not np.isclose(point, stored, rtol=0, atol=1e-9):
                raise SystemExit(f"{split}_s{f}_{model}: rebuilt valid macro-F1 {point:.6f} != stored {stored:.6f}")
        stats = bootstrap_resamples(df, macro_f1, seed=boot_seed(split, s))
        data[run] = (df, point, stats)
        rows.append({"set": "final" if refit else "ladder", "fit_on": "train+valid" if refit else "train",
                     "split": split, "split_seed": s, "model": model, "run_seed": f, **interval(point, stats),
                     "accuracy": float((df["y_true"] == df["y_pred"]).mean()), "n_clips": len(df),
                     "n_clusters": int(df["cluster"].fillna(df["clip_id"]).nunique())})
        log.info("%s%s_s%d_%s: %.3f [%.3f, %.3f]", "final " if refit else "", split, f, model, point,
                 rows[-1]["ci_low"], rows[-1]["ci_high"])
    if missing:
        msg = "no result file for: " + ", ".join(missing)
        if not allow_missing:
            raise SystemExit(msg)
        log.warning(msg)
    return rows, data


def seed_means(data: dict) -> dict:
    """(split, model) -> (mean macro-F1 over runs, mean bootstrap array, sd over runs, n runs)."""
    groups: dict = {}
    for (split, _, model, _), (_, point, stats) in data.items():
        groups.setdefault((split, model), []).append((point, stats))
    return {k: (float(np.mean([p for p, _ in v])), np.mean([s for _, s in v], axis=0),
                float(np.std([p for p, _ in v], ddof=1)) if len(v) > 1 else None, len(v))
            for k, v in groups.items()}


def ladder_table(means: dict) -> dict:
    return {f"{split}/{model}": {**interval(p, st), "sd_over_runs": sd, "n_runs": n}
            for (split, model), (p, st, sd, n) in means.items()}


def drops(means: dict) -> dict:
    """Mean macro-F1 of split a minus split b per model; CI from independent resamples of the two."""
    out = {}
    for model in MODELS:
        for a, b in DROPS:
            if (a, model) in means and (b, model) in means:
                (pa, sa, *_), (pb, sb, *_) = means[(a, model)], means[(b, model)]
                lo, hi = percentile_ci(sa - sb)
                out[f"{model}: {a} - {b}"] = {"diff": pa - pb, "ci_low": lo, "ci_high": hi}
    return out


def paired_on_C(data: dict, means: dict) -> dict:
    """CNN (mean of its runs) minus every other model on C, paired: the same resampled clusters."""
    c_runs = [df for (split, *_), (df, _, _) in data.items() if split == "C"]
    assert len({tuple(df["clip_id"]) for df in c_runs}) <= 1, "runs on C must cover the same clips in order"
    if ("C", "cnn") not in means:
        return {}
    pc, sc, *_ = means[("C", "cnn")]
    out = {}
    for model in MODELS[:-1]:
        if ("C", model) in means:
            pm, sm, *_ = means[("C", model)]
            lo, hi = percentile_ci(sc - sm)
            out[f"cnn - {model}"] = {"diff": pc - pm, "ci_low": lo, "ci_high": hi,
                                     "share_resamples_cnn_better": float(np.mean(sc - sm > 0))}
    return out


def c_strict(data: dict) -> dict:
    """C scored only on clips whose artist cluster has no clip in C train (gtzan-bind names)."""
    meta = pd.read_csv(Path(CFG["paths"]["splits"]) / "metadata.csv").set_index("clip_id")
    clusters = credit_clusters(meta["artist"])
    train = set(clusters[meta["split_C"] == "train"].dropna())
    out: dict = {}
    for (split, s, model, _), (df, _, _) in data.items():
        if split == "C":
            keep = df[~df["cluster"].isin(train)].reset_index(drop=True)
            out.setdefault(model, []).append((macro_f1(keep["y_true"], keep["y_pred"]),
                                              bootstrap_resamples(keep, macro_f1, seed=boot_seed("C", s)), len(keep)))
    return {m: {**interval(float(np.mean([p for p, _, _ in v])), np.mean([b for _, b, _ in v], axis=0)),
                "n_clips": v[0][2], "n_runs": len(v)} for m, v in out.items()}


def c_details(data: dict) -> dict:
    """Per-genre F1, support and confusion counts of every run on C."""
    out = {}
    for (split, _, model, f), (df, _, _) in data.items():
        if split == "C":
            sc = clip_scores(df["y_true"], df["y_pred"], labels=CFG["genres"])
            support = df["y_true"].value_counts().reindex(CFG["genres"], fill_value=0)
            out[f"{model}_s{f}"] = {"per_genre_f1": sc["per_genre_f1"],
                                    "support": {g: int(n) for g, n in support.items()},
                                    "confusion": sc["confusion"].tolist()}
    return out


def summary(data: dict) -> dict:
    means = seed_means(data)
    return {"ladder": ladder_table(means), "drops": drops(means), "paired_on_C": paired_on_C(data, means),
            "C_strict": c_strict(data), "C_details": c_details(data)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", required=True, choices=["valid", "test"])
    ap.add_argument("--allow-missing", action="store_true", help="skip runs without a result file (drafts only)")
    args = ap.parse_args()
    if args.part == "test" and not is_test_unlocked():
        raise SystemExit("test set locked: create the pre-test-freeze tag first (DECISIONS.md)")
    rows, data = score_runs(args.part, runs(), refit=False, allow_missing=args.allow_missing)
    out = {"part": args.part, "metric": "clip macro-F1 (soft vote)",
           "ci": f"{CFG['bootstrap']['ci']}% percentile, {CFG['bootstrap']['n']} artist-cluster resamples "
                 "(credit clusters; a clip without an artist is its own cluster)",
           "ladder_models": "selected settings, trained on train only", **summary(data)}
    if args.part == "test":
        final_rows, final = score_runs("test", runs(["C"]), refit=True, allow_missing=args.allow_missing)
        rows += final_rows
        out["final_models"] = "the same settings refit on train+valid (headline: CNN mean of its runs)"
        out["final_on_C"] = summary(final)
    pd.DataFrame(rows).to_csv(OUT_DIR[args.part] / "scores.csv", index=False)
    write_json(OUT_DIR[args.part] / "headline.json", out)
    log.info("-> %s", OUT_DIR[args.part])


if __name__ == "__main__":
    main()
