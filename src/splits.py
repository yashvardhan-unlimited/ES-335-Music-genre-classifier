"""Train/valid/test splits A, B, B', C and the locked loader.

Every split file has one row per window of the 999 usable clips:
clip_id, window_idx (0-9), genre, part in {train, valid, test, unused}.
All splits use the same sizes (config split_sizes, in clips; x10 for windows),
so they differ only in what can leak, not in how much training data they have.
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config

CFG = load_config()
SPLITS = Path(CFG["paths"]["splits"])
SIZES = CFG["split_sizes"]
N_WINDOWS = CFG["audio"]["n_windows"]
PARTS = ["train", "valid", "test"]


def add_split_C_to_metadata() -> pd.DataFrame:
    """Add artist and split_C (published fault-filtered split) to splits/metadata.csv."""
    meta = pd.read_csv(SPLITS / "metadata.csv").drop(columns=["artist", "split_C"], errors="ignore")
    bind = pd.read_csv(SPLITS / "gtzan_bind.csv").rename(
        columns={"track_id": "clip_id", "artist_name": "artist", "fault_filtered_split": "split_C"})
    meta = meta.merge(bind[["clip_id", "artist", "split_C"]], on="clip_id", how="left", validate="1:1")
    counts = meta["split_C"].value_counts()
    assert [counts.get(p, 0) for p in PARTS] == [443, 197, 290], counts
    meta.to_csv(SPLITS / "metadata.csv", index=False)
    return meta


def _usable() -> pd.DataFrame:
    meta = pd.read_csv(SPLITS / "metadata.csv")
    return meta[meta["load_ok"]].reset_index(drop=True)


def _expand(clips: pd.DataFrame) -> pd.DataFrame:
    """Clip-level (clip_id, genre, part) -> one row per window."""
    out = clips.loc[clips.index.repeat(N_WINDOWS), ["clip_id", "genre", "part"]].reset_index(drop=True)
    out.insert(1, "window_idx", np.tile(np.arange(N_WINDOWS), len(clips)))
    return out


def _stratified_parts(clips: pd.DataFrame, seed: int) -> pd.Series:
    """Assign exactly SIZES clips to train/valid/test, stratified by genre; rest 'unused'."""
    n_used = sum(SIZES.values())
    part = pd.Series("unused", index=clips.index)
    used = clips.index
    if len(clips) > n_used:
        used, _ = train_test_split(clips.index, train_size=n_used, stratify=clips["genre"], random_state=seed)
    tr, rest = train_test_split(used, train_size=SIZES["train"], stratify=clips.loc[used, "genre"],
                                random_state=seed)
    va, te = train_test_split(rest, train_size=SIZES["valid"], stratify=clips.loc[rest, "genre"],
                              random_state=seed)
    part[tr], part[va], part[te] = "train", "valid", "test"
    return part


def make_split_A(seed: int = 0) -> pd.DataFrame:
    """Window-level random split: windows of one clip can land in different parts."""
    win = _expand(_usable().assign(part="unused"))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(win))
    bounds = np.cumsum([SIZES[p] * N_WINDOWS for p in PARTS])
    for p, idx in zip(PARTS, np.split(order, bounds)[:3]):
        win.loc[idx, "part"] = p
    win.to_csv(SPLITS / "A.csv", index=False)
    return win


def make_split_B(seed: int) -> pd.DataFrame:
    """Clip-level random split from all 999 usable clips, stratified by genre."""
    clips = _usable()
    clips["part"] = _stratified_parts(clips, seed)
    out = _expand(clips)
    out.to_csv(SPLITS / f"B_seed{seed}.csv", index=False)
    return out


def make_split_Bprime(seed: int) -> pd.DataFrame:
    """Clip-level random split restricted to the fault-filtered (split C) clips."""
    clips = _usable()
    in_C = clips["split_C"].notna()
    clips["part"] = "unused"
    clips.loc[in_C, "part"] = _stratified_parts(clips[in_C], seed)
    out = _expand(clips)
    out.to_csv(SPLITS / f"Bp_seed{seed}.csv", index=False)
    return out


def make_split_C() -> pd.DataFrame:
    """The published fault-filtered split (Kereliuk, Sturm & Larsen 2015)."""
    clips = _usable()
    clips["part"] = clips["split_C"].fillna("unused")
    out = _expand(clips)
    out.to_csv(SPLITS / "C.csv", index=False)
    return out


def is_test_unlocked() -> bool:
    """True only if the git tag pre-test-freeze exists."""
    res = subprocess.run(["git", "tag", "-l", "pre-test-freeze"], capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "pre-test-freeze"


def split_file(name: str, seed: int = 0) -> Path:
    return SPLITS / {"A": "A.csv", "B": f"B_seed{seed}.csv", "Bp": f"Bp_seed{seed}.csv", "C": "C.csv"}[name]


def load_split(name: str, part: str, seed: int = 0, allow_test: bool = False) -> pd.DataFrame:
    """Rows (windows) of one part of split name in {A, B, Bp, C}. The test part is locked."""
    if part == "test" and not (allow_test and is_test_unlocked()):
        raise PermissionError("test set locked until pre-test-freeze tag")
    df = pd.read_csv(split_file(name, seed))
    return df[df["part"] == part].reset_index(drop=True)
