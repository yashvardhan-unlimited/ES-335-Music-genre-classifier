"""Train the log-mel CNN on one split; early-stop on valid clip macro-F1 (soft vote).

Run from repo root:
    python -m src.train_cnn --split C --seed 0 --epochs 60 --device auto
    python -m src.train_cnn --split C --seed 0 --refit_epochs 20   # train+valid, fixed epochs, no valid
    python -m src.train_cnn --split C --summary                    # config seeds -> summary JSON + figure
Writes models/cnn_{split}_s{seed}.pt, results/mel_norm_{split}_s{seed}.json and
results/val/{split}_s{seed}_cnn.json (best-epoch valid scores + per-epoch curves).
Refit mode writes the same names with a _refit suffix. Only train and valid are ever loaded.
Split A assigns windows, not clips, to parts: there the CNN trains on the fixed train windows
(no random crops, which would read valid/test windows), takes normalisation stats from those
windows only, and scores each clip on its valid windows, like the classical models.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cnn import CNN, MEL_DIR, N_WINDOWS, GenreCNN, MelWindowDataset, compute_norm_stats, n_params
from src.splits import load_split
from src.train_classical import valid_report
from src.utils import get_logger, load_config, set_seed

CFG = load_config()
GENRES = np.array(CFG["genres"])
RESULTS = Path(CFG["paths"]["results"])
VAL_DIR = RESULTS / "val"
FIG_DIR = RESULTS / "figures"
MODEL_DIR = Path(CFG["paths"]["models"])
log = get_logger("train_cnn")


def clip_labels(split: str, part: str, seed: int = 0) -> pd.Series:
    """One genre per clip of a split part, indexed by clip_id, in split-file order."""
    return load_split(split, part, seed).groupby("clip_id", sort=False)["genre"].first()


def rows_to_clips(rows: pd.DataFrame) -> tuple[pd.Series, dict[str, list[int]] | None]:
    """Window rows (clip_id, window_idx, genre) -> (genre per clip, window indices per clip).
    The window lists are None when every clip has all its windows (clip-level splits B, B', C)."""
    g = rows.groupby("clip_id", sort=False)
    windows = g["window_idx"].apply(lambda w: sorted(set(w)))
    complete = bool((windows.str.len() == N_WINDOWS).all())
    return g["genre"].first(), None if complete else windows.to_dict()


def encode(genres: pd.Series) -> np.ndarray:
    """Genre names -> integer codes in config order."""
    codes = pd.Categorical(genres, categories=GENRES).codes.astype(np.int64)
    assert (codes >= 0).all(), "label outside config genres"
    return codes


def pick_device(name: str) -> torch.device:
    """'auto' -> cuda if available, else cpu."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@torch.no_grad()
def predict(model: nn.Module, ds: MelWindowDataset, device: torch.device) -> np.ndarray:
    """Softmax probabilities for every item of an eval dataset, in item order."""
    model.eval()
    loader = DataLoader(ds, batch_size=CNN["batch_size"] * 2, shuffle=False)
    return np.concatenate([torch.softmax(model(x.to(device)), 1).cpu().numpy() for x, _ in loader])


def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    """One pass over the training crops; returns the mean cross-entropy per crop."""
    model.train()
    loss_fn, total, n = nn.CrossEntropyLoss(), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def fit(train_ids: list[str], train_y: np.ndarray, valid_ids: list[str] | None, valid_y: np.ndarray | None,
        epochs: int, device: torch.device, patience: int = CNN["patience"], mel_dir: Path = MEL_DIR,
        train_windows: dict | None = None, valid_windows: dict | None = None) -> dict:
    """Train a GenreCNN. With a valid part: evaluate valid clip macro-F1 after every epoch, keep
    the best weights, stop after `patience` epochs without improvement. Without one (refit):
    run exactly `epochs` epochs and keep the last weights. Seed the RNGs before calling.
    train_windows/valid_windows: window lists per clip for split A (see MelWindowDataset).
    """
    stats = compute_norm_stats(train_ids, mel_dir, train_windows)
    train_ds = MelWindowDataset(train_ids, train_y, mel_dir, "train", stats, window_idx=train_windows)
    loader = DataLoader(train_ds, batch_size=CNN["batch_size"], shuffle=True)
    model = GenreCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CNN["lr"], weight_decay=CNN["weight_decay"])
    if valid_ids is not None:
        valid_ds = MelWindowDataset(valid_ids, valid_y, mel_dir, "eval", stats, window_idx=valid_windows)
        y_win = pd.Series(GENRES[valid_ds.item_labels], index=valid_ds.clip_ids)  # per window
        y_win_codes = torch.as_tensor(valid_ds.item_labels)

    history, best, bad = [], None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        row = {"epoch": epoch, "train_loss": train_epoch(model, loader, opt, device)}
        row["train_seconds"] = time.perf_counter() - t0
        if valid_ids is not None:
            probs = predict(model, valid_ds, device)
            row["valid_loss"] = float(nn.functional.nll_loss(torch.log(torch.as_tensor(probs).clamp_min(1e-12)),
                                                             y_win_codes))
            report = valid_report(probs, GENRES, y_win)
            row.update({k: report[k] for k in ("valid_macro_f1", "valid_accuracy", "window_accuracy")})
        row["seconds"] = time.perf_counter() - t0
        history.append(row)
        log.info("epoch %d: %s", epoch, ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "epoch"))

        if valid_ids is None:
            continue
        if best is None or row["valid_macro_f1"] > best["report"]["valid_macro_f1"]:
            best = {"epoch": epoch, "report": report,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("early stop: no better valid clip macro-F1 for %d epochs", patience)
                break

    if best is None:  # refit: last weights
        best = {"epoch": epochs, "report": None,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return {"state": best["state"], "best_epoch": best["epoch"], "report": best["report"],
            "history": history, "norm_stats": stats, "n_params": n_params(model)}


def run(split: str, seed: int, epochs: int, device: torch.device, refit_epochs: int | None) -> dict:
    """Load the split, train, save weights, norm stats and the result JSON; return the result."""
    set_seed(CFG["seed"] + seed)
    train, train_w = rows_to_clips(load_split(split, "train", seed))
    valid, valid_w = rows_to_clips(load_split(split, "valid", seed))
    suffix = "_refit" if refit_epochs else ""
    if refit_epochs:
        both, both_w = rows_to_clips(pd.concat([load_split(split, p, seed) for p in ("train", "valid")]))
        out = fit(list(both.index), encode(both), None, None, refit_epochs, device, train_windows=both_w)
    else:
        out = fit(list(train.index), encode(train), list(valid.index), encode(valid), epochs, device,
                  train_windows=train_w, valid_windows=valid_w)

    stub = f"{split}_s{seed}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights = MODEL_DIR / f"cnn_{stub}{suffix}.pt"
    torch.save({"state_dict": out["state"], "channels": CNN["channels"], "classes": [str(g) for g in GENRES],
                "best_epoch": out["best_epoch"]}, weights)
    norm_path = RESULTS / f"mel_norm_{stub}{suffix}.json"
    mean, std = out["norm_stats"]
    norm_path.write_text(json.dumps({"split": split, "seed": seed, "fit_on": "train+valid" if refit_epochs else "train",
                                     "n_clips": len(both) if refit_epochs else len(train),
                                     "window_level": (both_w if refit_epochs else train_w) is not None,
                                     "mean": mean.tolist(), "std": std.tolist()}, indent=1))
    secs = [h["seconds"] for h in out["history"]]
    result = {"split": split, "seed": seed, "model": "cnn" + suffix, "torch_seed": CFG["seed"] + seed,
              "device": str(device), "torch_threads": torch.get_num_threads(), "n_params": out["n_params"],
              "best_epoch": out["best_epoch"], "epochs_run": len(out["history"]),
              "median_seconds_per_epoch": float(np.median(secs)), **(out["report"] or {}),
              "n_train_clips": len(both) if refit_epochs else len(train),
              "weights": weights.as_posix(), "norm_stats": norm_path.as_posix(), "history": out["history"]}
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    (VAL_DIR / f"{stub}_cnn{suffix}.json").write_text(json.dumps(result, indent=1))
    return result


def plot_curves(results: list[dict], out: Path, reference: float | None = None) -> None:
    """Left: train and valid loss per epoch; right: valid clip macro-F1; one colour per seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for r, color in zip(results, colors):
        h = pd.DataFrame(r["history"])
        a1.plot(h["epoch"], h["train_loss"], color=color, lw=2, label=f"seed {r['seed']} train")
        a1.plot(h["epoch"], h["valid_loss"], color=color, lw=2, ls="--", label=f"seed {r['seed']} valid")
        a2.plot(h["epoch"], h["valid_macro_f1"], color=color, lw=2, label=f"seed {r['seed']}")
        b = h.loc[h["epoch"] == r["best_epoch"]].iloc[0]
        a2.plot(b["epoch"], b["valid_macro_f1"], "o", color=color, ms=9, markeredgecolor="#ffffff", markeredgewidth=2)
    if reference is not None:
        a2.axhline(reference, color="#8a8984", lw=1.2, ls=":", label=f"logistic regression {reference:.3f}")
    a1.set(xlabel="epoch", ylabel="cross-entropy per window",
           title="Loss (train: augmented random crops, dropout on;\nvalid: 10 fixed windows per clip)")
    a2.set(xlabel="epoch", ylabel="valid clip macro-F1 (soft vote)", ylim=(0, 1),
           title="Valid clip macro-F1; dot = best epoch (weights kept)")
    for ax in (a1, a2):
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"CNN on log-mels, split {results[0]['split']}, {len(results)} seeds")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def summarise(split: str) -> dict:
    """Combine the per-seed results for the config seeds into a summary JSON and the curves figure."""
    results = [json.loads((VAL_DIR / f"{split}_s{s}_cnn.json").read_text()) for s in CNN["seeds"]]
    f1 = np.array([r["valid_macro_f1"] for r in results])
    summary = {
        "split": split, "seeds": CNN["seeds"],
        "per_seed": [{k: r[k] for k in ("seed", "best_epoch", "epochs_run", "valid_macro_f1", "valid_accuracy",
                                        "window_accuracy", "median_seconds_per_epoch", "device", "torch_threads")}
                     for r in results],
        "median_best_epoch": float(np.median([r["best_epoch"] for r in results])),
        "valid_macro_f1_mean": float(f1.mean()), "valid_macro_f1_std": float(f1.std(ddof=1)),
        "valid_macro_f1_min": float(f1.min()), "valid_macro_f1_max": float(f1.max()),
        "n_params": results[0]["n_params"],
    }
    (VAL_DIR / f"{split}_cnn_summary.json").write_text(json.dumps(summary, indent=1))
    ref_path = VAL_DIR / f"{split}_s0_logreg.json"
    ref = json.loads(ref_path.read_text())["valid_macro_f1"] if ref_path.exists() else None
    plot_curves(results, FIG_DIR / f"cnn_curves_{split}.png", ref)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="C", choices=["A", "B", "Bp", "C"])
    ap.add_argument("--seed", type=int, default=0, help="run seed (torch seed = config seed + this); "
                                                        "also the split file seed for B and Bp")
    ap.add_argument("--epochs", type=int, default=CNN["epochs"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--refit_epochs", type=int, default=None, help="train on train+valid for N epochs, no valid")
    ap.add_argument("--summary", action="store_true", help="summarise the config seeds and plot, no training")
    args = ap.parse_args()
    if args.summary:
        print(json.dumps(summarise(args.split), indent=1))
        return
    device = pick_device(args.device)
    log.info("split %s seed %d on %s (%d torch threads)", args.split, args.seed, device, torch.get_num_threads())
    r = run(args.split, args.seed, args.epochs, device, args.refit_epochs)
    log.info("best epoch %d of %d, valid clip macro-F1 %s, median %.1f s/epoch", r["best_epoch"], r["epochs_run"],
             f"{r['valid_macro_f1']:.3f}" if "valid_macro_f1" in r else "n/a (refit)", r["median_seconds_per_epoch"])


if __name__ == "__main__":
    main()
