"""Exploratory figures and data_summary.json from splits/metadata.csv.

Run from repo root: python scripts/eda.py
"""
import json
import sys
import warnings
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, load_config  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger("eda")


def log_mel(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Log-mel spectrogram (dB) with the project's mel settings."""
    m = cfg["mel"]
    S = librosa.feature.melspectrogram(y=y, sr=cfg["audio"]["sr"], n_fft=m["n_fft"],
                                       hop_length=m["hop_length"], n_mels=m["n_mels"])
    return librosa.power_to_db(S, ref=m["ref"])


def centroid_tempo(path: str, sr: int) -> tuple[float, float]:
    """Mean spectral centroid (Hz) and estimated tempo (BPM) of one clip."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])
    return centroid, tempo


def genre_box(df: pd.DataFrame, col: str, ylabel: str, title: str, ax, order) -> None:
    sns.boxplot(data=df, x="genre", y=col, order=order, ax=ax, color="lightsteelblue", fliersize=2)
    ax.set(xlabel="", ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=45)


def main() -> None:
    cfg = load_config()
    sr, genres = cfg["audio"]["sr"], cfg["genres"]
    fig_dir = Path(cfg["paths"]["results"]) / "figures"
    meta = pd.read_csv(Path(cfg["paths"]["splits"]) / "metadata.csv")
    ok = meta[meta["load_ok"]].copy()

    # class counts: all files vs loadable
    counts = pd.DataFrame({"files": meta.groupby("genre").size(),
                           "loadable": ok.groupby("genre").size()}).reindex(genres).fillna(0)
    ax = counts.plot.bar(figsize=(9, 4), color=["lightgray", "steelblue"])
    ax.set(xlabel="", ylabel="clips", title="Clips per genre", ylim=(0, 115))
    ax.bar_label(ax.containers[1], fontsize=8)
    ax.legend(loc="upper center", ncol=2)
    plt.tight_layout(); plt.savefig(fig_dir / "class_counts.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ok["duration_s"], bins=40, color="steelblue")
    ax.axvline(cfg["audio"]["clip_seconds"], color="red", ls="--", label="30 s")
    ax.set(xlabel="duration (s)", ylabel="clips", title="Clip durations (loadable files)")
    ax.legend(); plt.tight_layout(); plt.savefig(fig_dir / "duration_hist.png", dpi=120); plt.close()

    ok["clip_pct"] = 100 * ok["clip_share"]
    fig, ax = plt.subplots(figsize=(9, 4))
    genre_box(ok, "clip_pct", "% samples with |x| >= 0.999", "Clipping by genre", ax, genres)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(fig_dir / "clipping_by_genre.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    genre_box(ok, "rms_db", "RMS level (dBFS)", "Loudness by genre", ax, genres)
    plt.tight_layout(); plt.savefig(fig_dir / "rms_by_genre.png", dpi=120); plt.close()

    # one log-mel per genre: first loadable clip
    fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharey=True)
    for ax, g in zip(axes.flat, genres):
        row = ok[ok["genre"] == g].iloc[0]
        y, _ = librosa.load(row["path"], sr=sr, mono=True)
        img = librosa.display.specshow(log_mel(y, cfg), sr=sr, hop_length=cfg["mel"]["hop_length"],
                                       x_axis="time", y_axis="mel", ax=ax, vmin=-80, vmax=20)
        ax.set_title(row["clip_id"]); ax.label_outer()
    fig.colorbar(img, ax=axes, format="%+.0f dB", shrink=0.8)
    plt.savefig(fig_dir / "mel_per_genre.png", dpi=110, bbox_inches="tight"); plt.close()

    # centroid and tempo for all loadable clips (one file per worker at a time)
    log.info("computing centroid and tempo for %d clips", len(ok))
    res = Parallel(n_jobs=-1)(delayed(centroid_tempo)(p, sr) for p in ok["path"])
    ok[["centroid_hz", "tempo_bpm"]] = res
    ok[["clip_id", "genre", "centroid_hz", "tempo_bpm"]].to_csv(
        Path(cfg["paths"]["results"]) / "eda_descriptors.csv", index=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 4.5))
    genre_box(ok, "centroid_hz", "mean spectral centroid (Hz)", "Spectral centroid by genre", a1, genres)
    genre_box(ok, "tempo_bpm", "estimated tempo (BPM)", "Tempo by genre", a2, genres)
    plt.tight_layout(); plt.savefig(fig_dir / "centroid_tempo_by_genre.png", dpi=120); plt.close()

    short = ok[ok["duration_s"] < cfg["audio"]["clip_seconds"]]
    clipped = ok[ok["clip_share"] > 0.01].sort_values("clip_share", ascending=False)
    dup_groups = [g["clip_id"].tolist() for _, g in meta.groupby("md5") if len(g) > 1]
    summary = {
        "n_files": int(len(meta)),
        "n_loadable": int(len(ok)),
        "counts_per_genre": {g: {"files": int(counts.loc[g, "files"]),
                                 "loadable": int(counts.loc[g, "loadable"])} for g in genres},
        "failures": meta.loc[~meta["load_ok"], "clip_id"].tolist(),
        "duration_s": {"min": float(ok["duration_s"].min()), "max": float(ok["duration_s"].max())},
        "shorter_than_30s": {r.clip_id: round(r.duration_s, 3) for r in short.itertuples()},
        "clip_share_over_1pct": {r.clip_id: round(r.clip_share, 4) for r in clipped.itertuples()},
        "identical_md5_groups": dup_groups,
    }
    out = Path(cfg["paths"]["results"]) / "data_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s and 6 figures", out)


if __name__ == "__main__":
    main()
