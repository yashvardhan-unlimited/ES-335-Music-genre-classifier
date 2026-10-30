"""Validation analyses on split C (train and valid only).

1 window length: handcrafted features for 1/3/5/10 s windows (features/ablation/), SVM with its
  tuned params -> valid clip macro-F1 per length.
2 learning curves: 25/50/75/100% of training artists x 3 random repeats, SVM and CNN (fixed
  median best epoch) -> valid clip macro-F1 mean and sd.
3 feature groups: SVM with one feature group dropped at a time -> change in valid clip macro-F1.
4 calibration: reliability (10 bins) and ECE for SVM, XGBoost, CNN; CNN temperature scaling.
5 voting: soft vs majority clip macro-F1 for every model with a valid result file.

Run from repo root: python scripts/analysis_val.py [--parts 1,2,3,4,5]
Parts 2 and 4 need the CNN results (results/val/C_cnn_summary.json, models/cnn_C_s*.pt).
Writes results/val/analysis_{window_length,learning_curves,feature_groups,calibration,voting}.json
and the matching results/figures/analysis_*.png.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cnn import MEL_DIR, GenreCNN, MelWindowDataset  # noqa: E402
from src.evaluate import aggregate_clips, ece, fit_temperature, reliability, softmax_t  # noqa: E402
from src.features import N_CLIP, SR, clip_handcrafted, feature_group, feature_names, load_clip  # noqa: E402
from src.models import svm_pipeline, xgb_model  # noqa: E402
from src.splits import artist_groups, sample_artists  # noqa: E402
from src.train_classical import load_windows, valid_report  # noqa: E402
from src.train_cnn import clip_labels, encode, fit, pick_device  # noqa: E402
from src.utils import get_logger, load_config, set_seed  # noqa: E402

CFG = load_config()
A = CFG["analysis"]
GENRES = np.array(CFG["genres"])
VAL_DIR = Path(CFG["paths"]["results"]) / "val"
FIG_DIR = Path(CFG["paths"]["results"]) / "figures"
ABL_DIR = Path(CFG["paths"]["features"]) / "ablation"
MODEL_DIR = Path(CFG["paths"]["models"])
COLORS = {"svm": "#2a78d6", "xgb": "#eb6834", "cnn": "#1baf7a", "logreg": "#eda100"}
log = get_logger("analysis_val")


def load_json(name: str) -> dict:
    return json.loads((VAL_DIR / name).read_text())


def save(name: str, obj: dict) -> None:
    (VAL_DIR / f"analysis_{name}.json").write_text(json.dumps(obj, indent=1))
    log.info("-> %s", VAL_DIR / f"analysis_{name}.json")


def svm_params() -> dict:
    return load_json("C_s0_svm.json")["best_params"]


def svm_probs(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit the tuned SVM on train (one BLAS thread, as in tuning); valid window probs and classes."""
    with threadpool_limits(1):
        est = svm_pipeline(**svm_params()).fit(X_tr, y_tr)
        return est.predict_proba(X_va), est.classes_


def axes_style(ax) -> None:
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plt_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------- 1. window length
def _ablation_clip(path: str, counts: list[int]) -> dict[int, np.ndarray]:
    warnings.filterwarnings("ignore")
    return clip_handcrafted(load_clip(path)[0], counts)


def extract_ablation(clip_ids: list[str]) -> None:
    """features/ablation/win{L}s.npy (+ index csv) for every window length, train+valid clips only."""
    lengths = A["window_seconds"]
    if all((ABL_DIR / f"win{L}s.npy").exists() for L in lengths):
        return
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    counts = [N_CLIP // (L * SR) for L in lengths]
    meta = pd.read_csv(Path(CFG["paths"]["splits"]) / "metadata.csv").set_index("clip_id")
    t0 = time.perf_counter()
    feats = Parallel(n_jobs=-1)(delayed(_ablation_clip)(meta.loc[c, "path"], counts) for c in clip_ids)
    log.info("ablation features for %d clips in %.0f s", len(clip_ids), time.perf_counter() - t0)
    for L, n in zip(lengths, counts):
        np.save(ABL_DIR / f"win{L}s.npy", np.concatenate([f[n] for f in feats]).astype(np.float32))
        pd.DataFrame({"clip_id": np.repeat(clip_ids, n), "window_idx": np.tile(np.arange(n), len(clip_ids)),
                      "genre": np.repeat(meta.loc[clip_ids, "genre"].to_numpy(), n)}
                     ).to_csv(ABL_DIR / f"index_win{L}s.csv", index=False)


def ablation_windows(L: int, clip_ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Features of the given clips at window length L, indexed by clip_id, in clip order."""
    idx = pd.read_csv(ABL_DIR / f"index_win{L}s.csv")
    X = pd.DataFrame(np.load(ABL_DIR / f"win{L}s.npy"), columns=feature_names(), index=pd.Index(idx["clip_id"]))
    y = pd.Series(idx["genre"].to_numpy(), index=X.index)
    return X.loc[clip_ids], y.loc[clip_ids]


def part_window_length() -> dict:
    train, valid = clip_labels("C", "train"), clip_labels("C", "valid")
    extract_ablation(list(train.index) + list(valid.index))
    # 3 s ablation features must reproduce the main features (same code, recomputed)
    X3, _ = ablation_windows(3, list(valid.index))
    Xm, _ = load_windows("C", "valid")
    Xm = Xm.loc[list(valid.index)]
    rel = (np.abs(X3.to_numpy() - Xm.to_numpy()) / (np.abs(Xm.to_numpy()) + 1e-6)).max()
    rows = []
    for L in A["window_seconds"]:
        X_tr, y_tr = ablation_windows(L, list(train.index))
        X_va, y_va = ablation_windows(L, list(valid.index))
        t0 = time.perf_counter()
        probs, classes = svm_probs(X_tr, y_tr, X_va)
        r = valid_report(probs, classes, y_va)
        rows.append({"window_seconds": L, "windows_per_clip": len(X_va) // len(valid),
                     "n_train_windows": len(X_tr), "fit_seconds": time.perf_counter() - t0,
                     **{k: r[k] for k in ("valid_macro_f1", "valid_macro_f1_majority", "valid_accuracy",
                                          "window_accuracy", "per_genre_f1")}})
        log.info("window %d s: valid clip macro-F1 %.3f", L, r["valid_macro_f1"])
    out = {"model": "svm", "params": svm_params(), "note": "params tuned at 3 s, held fixed",
           "max_relative_diff_3s_vs_main_features": float(rel), "results": rows}
    save("window_length", out)

    plt = plt_module()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    d = pd.DataFrame(rows)
    ax.plot(d["window_seconds"], d["valid_macro_f1"], color=COLORS["svm"], lw=2, marker="o", ms=8,
            markeredgecolor="#ffffff", markeredgewidth=2, label="soft vote")
    ax.plot(d["window_seconds"], d["valid_macro_f1_majority"], color=COLORS["svm"], lw=1.5, ls="--", marker="o",
            ms=6, alpha=0.6, label="majority vote")
    for _, r in d.iterrows():
        ax.annotate(f"{r['valid_macro_f1']:.3f}\n({r['windows_per_clip']} win/clip)", (r["window_seconds"],
                    r["valid_macro_f1"]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xticks(d["window_seconds"])
    ax.set(xlabel="window length (s)", ylabel="valid clip macro-F1", ylim=(0, 0.7),
           title=f"SVM (C={svm_params()['C']}, gamma={svm_params()['gamma']}, tuned at 3 s) by window length")
    axes_style(ax)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "analysis_window_length.png", dpi=120)
    plt.close()
    return out


# ---------------------------------------------------------------- 2. learning curves
@torch.no_grad()
def cnn_valid_logits(state: dict, stats: tuple, valid: pd.Series, device: torch.device) -> np.ndarray:
    """Logits of a trained GenreCNN on the 10 fixed windows of every valid clip (item order)."""
    model = GenreCNN().to(device)
    model.load_state_dict(state)
    model.eval()
    ds = MelWindowDataset(list(valid.index), encode(valid), MEL_DIR, "eval", stats)
    return np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in DataLoader(ds, batch_size=128)])


def window_labels(valid: pd.Series, per_clip: int) -> pd.Series:
    """Per-window genre labels indexed by clip id, item order of the eval dataset."""
    return pd.Series(np.repeat(valid.to_numpy(), per_clip), index=np.repeat(valid.index, per_clip))


def part_learning_curves() -> dict:
    train, valid = clip_labels("C", "train"), clip_labels("C", "valid")
    X_tr, y_tr = load_windows("C", "train")
    X_va, y_va = load_windows("C", "valid")
    epochs = int(round(load_json("C_cnn_summary.json")["median_best_epoch"]))
    device = pick_device("auto")
    n_groups = artist_groups(list(train.index)).nunique()
    rows = []
    for frac in A["artist_fractions"]:
        for rep in range(A["repeats"]):
            seed = CFG["seed"] + rep
            sub = sample_artists(list(train.index), frac, seed)
            base = {"fraction": frac, "repeat": rep, "seed": seed, "n_clips": len(sub),
                    "n_artist_groups": int(artist_groups(sub).nunique()), "n_genres": int(train[sub].nunique())}
            probs, classes = svm_probs(X_tr.loc[sub], y_tr.loc[sub], X_va)
            rows.append({**base, "model": "svm", "valid_macro_f1": valid_report(probs, classes, y_va)["valid_macro_f1"]})
            t0 = time.perf_counter()
            set_seed(seed)
            out = fit(sub, encode(train[sub]), None, None, epochs, device)
            logits = cnn_valid_logits(out["state"], out["norm_stats"], valid, device)
            f1 = valid_report(softmax_t(logits), GENRES, window_labels(valid, 10))["valid_macro_f1"]
            rows.append({**base, "model": "cnn", "epochs": epochs, "valid_macro_f1": f1,
                         "seconds": time.perf_counter() - t0})
            log.info("fraction %.2f repeat %d (%d clips): svm %.3f, cnn %.3f", frac, rep, len(sub),
                     rows[-2]["valid_macro_f1"], f1)
    d = pd.DataFrame(rows)
    agg = (d.groupby(["model", "fraction"])
           .agg(mean=("valid_macro_f1", "mean"), sd=("valid_macro_f1", "std"), mean_clips=("n_clips", "mean"))
           .reset_index())
    out = {"cnn_epochs": epochs, "n_artist_groups_in_train": int(n_groups), "runs": rows,
           "summary": agg.to_dict(orient="records")}
    save("learning_curves", out)

    plt = plt_module()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, label in [("svm", "SVM (handcrafted features)"), ("cnn", f"CNN ({epochs} epochs)")]:
        a = agg[agg["model"] == m]
        pts = d[d["model"] == m]
        ax.scatter(pts["fraction"], pts["valid_macro_f1"], color=COLORS[m], alpha=0.35, s=18)
        ax.errorbar(a["fraction"], a["mean"], yerr=a["sd"], color=COLORS[m], lw=2, marker="o", ms=8,
                    capsize=4, label=f"{label}: mean +/- sd")
    clips = agg[agg["model"] == "svm"].set_index("fraction")["mean_clips"]
    ax.set_xticks(clips.index, [f"{f:.0%}\n~{c:.0f} clips" for f, c in clips.items()])
    ax.set(xlabel="share of training artists (mean training clips)", ylabel="valid clip macro-F1", ylim=(0, 0.8),
           title=f"Learning curves on split C ({A['repeats']} random artist subsets per point)")
    axes_style(ax)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "analysis_learning_curves.png", dpi=120)
    plt.close()
    return out


# ---------------------------------------------------------------- 3. feature groups
def part_feature_groups() -> dict:
    X_tr, y_tr = load_windows("C", "train")
    X_va, y_va = load_windows("C", "valid")
    groups = pd.Series([feature_group(c) for c in X_tr.columns], index=X_tr.columns)
    full = valid_report(*svm_probs(X_tr, y_tr, X_va), y_va)["valid_macro_f1"]
    rows = []
    for g in A["feature_groups"]:
        keep = groups.index[groups != g]
        f1 = valid_report(*svm_probs(X_tr[keep], y_tr, X_va[keep]), y_va)["valid_macro_f1"]
        rows.append({"dropped": g, "n_features_dropped": int((groups == g).sum()), "n_features_left": len(keep),
                     "valid_macro_f1": f1, "change": f1 - full})
        log.info("without %s: %.3f (%+.3f)", g, f1, f1 - full)
    out = {"model": "svm", "params": svm_params(), "full_valid_macro_f1": full, "results": rows}
    save("feature_groups", out)

    plt = plt_module()
    d = pd.DataFrame(rows).sort_values("change")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f"{g} ({n})" for g, n in zip(d["dropped"], d["n_features_dropped"])], d["change"],
            color=["#eb6834" if c < 0 else "#2a78d6" for c in d["change"]], height=0.6)
    for i, c in enumerate(d["change"]):
        ax.text(c, i, f" {c:+.3f} ", va="center", ha="left" if c >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#52514e", lw=1)
    ax.set(xlabel=f"change in valid clip macro-F1 when dropped (full model {full:.3f})",
           title="SVM feature-group ablation (group, features dropped)\nnegative = the group was helping")
    lim = max(0.05, d["change"].abs().max() * 1.4)
    ax.set_xlim(-lim, lim)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "analysis_feature_groups.png", dpi=120)
    plt.close()
    return out


# ---------------------------------------------------------------- 4. calibration
def calibration_entry(window_probs: np.ndarray, y_win: pd.Series) -> dict:
    """ECE and reliability at window and clip level (clip = soft-vote mean probabilities)."""
    bins = A["calibration_bins"]
    y_idx = encode(y_win)
    ids, clip_probs, _ = aggregate_clips(window_probs, y_win.index)
    y_clip = encode(y_win.groupby(level=0).first().loc[ids])
    return {"window_ece": ece(window_probs, y_idx, bins), "clip_ece": ece(clip_probs, y_clip, bins),
            "clip_macro_f1": valid_report(window_probs, GENRES, y_win)["valid_macro_f1"],
            "clip_reliability": reliability(clip_probs, y_clip, bins).to_dict(orient="records"),
            "window_reliability": reliability(window_probs, y_idx, bins).to_dict(orient="records")}


def cross_fit_temperature(logits: np.ndarray, y_win: pd.Series, seed: int) -> tuple[np.ndarray, list[float]]:
    """2-fold by valid artist groups: T fit on one half's windows is applied to the other half."""
    groups = artist_groups(list(dict.fromkeys(y_win.index)))
    uniq = np.sort(groups.unique())
    half = set(np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 2])
    in_a = groups.loc[y_win.index].isin(half).to_numpy()
    probs, temps, y_idx = np.zeros_like(logits), [], encode(y_win)
    for fit_mask in (in_a, ~in_a):
        t = fit_temperature(logits[fit_mask], y_idx[fit_mask])
        probs[~fit_mask] = softmax_t(logits[~fit_mask], t)
        temps.append(t)
    return probs, temps


def part_calibration() -> dict:
    X_tr, y_tr = load_windows("C", "train")
    X_va, y_va = load_windows("C", "valid")
    out = {"bins": A["calibration_bins"], "models": {}}

    probs, classes = svm_probs(X_tr, y_tr, X_va)
    assert list(classes) == list(GENRES)
    out["models"]["svm"] = calibration_entry(probs, y_va)

    xgb = load_json("C_s0_xgb.json")
    fixed = {k: v for k, v in xgb["fixed_params"].items() if k not in ("n_estimators", "early_stopping_rounds")}
    est = xgb_model(**fixed, **xgb["best_params"], n_estimators=xgb["n_trees"]).fit(X_tr, encode(y_tr))
    out["models"]["xgb"] = calibration_entry(est.predict_proba(X_va), y_va)
    assert np.isclose(out["models"]["xgb"]["clip_macro_f1"], xgb["valid_macro_f1"]), "xgb refit differs"

    valid = clip_labels("C", "valid")
    y_win = window_labels(valid, 10)
    device = pick_device("auto")
    for s in CFG["cnn"]["seeds"]:
        ck = torch.load(MODEL_DIR / f"cnn_C_s{s}.pt", map_location=device)
        norm = json.loads(Path(f"results/mel_norm_C_s{s}.json").read_text())
        logits = cnn_valid_logits(ck["state_dict"], (np.array(norm["mean"]), np.array(norm["std"])), valid, device)
        before = calibration_entry(softmax_t(logits), y_win)
        assert np.isclose(before["clip_macro_f1"], load_json(f"C_s{s}_cnn.json")["valid_macro_f1"]), "cnn differs"
        t = fit_temperature(logits, encode(y_win))
        cf_probs, cf_t = cross_fit_temperature(logits, y_win, CFG["seed"])
        out["models"][f"cnn_s{s}"] = {
            "before": before, "temperature_in_sample": t, "after_in_sample": calibration_entry(softmax_t(logits, t), y_win),
            "temperatures_cross_fit": cf_t, "after_cross_fit": calibration_entry(cf_probs, y_win),
            "window_predictions_changed_by_t": int((softmax_t(logits, t).argmax(1) != logits.argmax(1)).sum())}
        log.info("cnn s%d: T=%.2f, clip ECE %.3f -> %.3f (in-sample), %.3f (cross-fit)", s, t, before["clip_ece"],
                 out["models"][f"cnn_s{s}"]["after_in_sample"]["clip_ece"],
                 out["models"][f"cnn_s{s}"]["after_cross_fit"]["clip_ece"])
    save("calibration", out)

    plt = plt_module()
    s0 = out["models"]["cnn_s0"]
    panels = [("SVM (Platt)", [("svm", out["models"]["svm"], COLORS["svm"])]),
              ("XGBoost", [("xgb", out["models"]["xgb"], COLORS["xgb"])]),
              (f"CNN seed 0: before / after T={s0['temperature_in_sample']:.2f}",
               [("before", s0["before"], COLORS["cnn"]), ("after T (cross-fit)", s0["after_cross_fit"], "#52514e")])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (title, lines) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color="#8a8984", lw=1, ls="--")
        for label, e, color in lines:
            t = pd.DataFrame(e["clip_reliability"]).dropna()
            ax.plot(t["confidence"], t["accuracy"], color=color, lw=2, marker="o", label=f"{label}: ECE {e['clip_ece']:.3f}")
            for _, r in t.iterrows():
                ax.annotate(f"{int(r['n'])}", (r["confidence"], r["accuracy"]), textcoords="offset points",
                            xytext=(4, -10), fontsize=7, color=color)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence in bin (clip, soft vote)", title=title)
        axes_style(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("accuracy in bin")
    fig.suptitle(f"Reliability on split C valid (197 clips, {A['calibration_bins']} bins; numbers = clips per bin)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "analysis_calibration.png", dpi=120)
    plt.close()
    return out


# ---------------------------------------------------------------- 5. voting
def part_voting() -> dict:
    order = ["dummy_most_frequent", "dummy_stratified", "knn", "logreg", "svm", "xgb"] + \
            [f"cnn_s{s}" for s in CFG["cnn"]["seeds"]]
    rows = []
    for name in order:
        file = f"C_s{name.split('_s')[-1]}_cnn.json" if name.startswith("cnn") else f"C_s0_{name}.json"
        r = load_json(file)
        rows.append({"model": name, "soft": r["valid_macro_f1"], "majority": r["valid_macro_f1_majority"],
                     "soft_minus_majority": r["valid_macro_f1"] - r["valid_macro_f1_majority"]})
    out = {"results": rows}
    save("voting", out)

    plt = plt_module()
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d["majority"], d["soft"], color="#c9c8c3", lw=3)
    ax.scatter(d["majority"], y, color="#eb6834", s=60, zorder=3, label="majority vote")
    ax.scatter(d["soft"], y, color="#2a78d6", s=60, zorder=3, label="soft vote")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(max(r["soft"], r["majority"]) + 0.015, yi, f"{r['soft_minus_majority']:+.3f}", va="center", fontsize=9)
    ax.set_yticks(y, d["model"])
    ax.set(xlabel="valid clip macro-F1", xlim=(0, 0.75), title="Soft vs majority vote per model (split C valid)")
    axes_style(ax)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "analysis_voting.png", dpi=120)
    plt.close()
    return out


PARTS = {1: part_window_length, 2: part_learning_curves, 3: part_feature_groups, 4: part_calibration, 5: part_voting}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="1,2,3,4,5")
    args = ap.parse_args()
    set_seed(CFG["seed"])
    for p in map(int, args.parts.split(",")):
        t0 = time.perf_counter()
        PARTS[p]()
        log.info("part %d done in %.0f s", p, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
