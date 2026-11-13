"""Train/valid/test splits A, B, B', D, C and the locked loader.

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


def _grouped_parts(clips: pd.DataFrame, groups: pd.Series, seed: int, slack: float = 2,
                   max_tries: int = 50) -> pd.Series:
    """Assign whole groups to train/valid/test with exactly SIZES clips per part, balancing genres.

    Groups go largest first (random order among equal sizes). Each goes to a random part, weighted
    by free room, among the parts it fits in without taking any genre more than `slack` clips past
    its quota (each part's share of every genre); if no part allows that, to the part where it adds
    least to the squared distance from the quotas. A try that leaves a group with nowhere to fit
    is redrawn with the next random state.
    Then single-clip groups are swapped between parts to even out the genres (_balance_by_swaps).
    """
    sizes = {p: SIZES[p] for p in PARTS}
    assert sum(sizes.values()) == len(clips), "D uses the whole pool"
    share = {p: sizes[p] / len(clips) for p in PARTS}
    genre_total = clips["genre"].value_counts()
    members = {g: idx for g, idx in clips.groupby(groups.loc[clips.index].to_numpy()).groups.items()}
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        keys = list(members)
        rng.shuffle(keys)
        keys.sort(key=lambda k: -len(members[k]))  # stable: equal sizes keep the shuffled order
        part = pd.Series("", index=clips.index)
        free = dict(sizes)
        filled = {p: pd.Series(0.0, index=genre_total.index) for p in PARTS}
        ok = True
        for k in keys:
            idx = members[k]
            g_counts = clips.loc[idx, "genre"].value_counts()
            fits = [p for p in PARTS if free[p] >= len(idx)]
            if not fits:
                ok = False
                break
            # random part, weighted by free room (as plain random assignment would be), among those
            # where the group keeps every genre within `slack` clips of its quota; if there is none,
            # the part where it adds least to the squared distance from the quotas
            quota = {p: share[p] * genre_total[g_counts.index] for p in fits}
            room = [p for p in fits if (filled[p][g_counts.index] + g_counts <= quota[p] + slack).all()]
            if room:
                w = np.array([free[p] for p in room], dtype=float)
                best = room[rng.choice(len(room), p=w / w.sum())]
            else:
                cost = {p: float(((filled[p][g_counts.index] + g_counts - quota[p]) ** 2
                                  - (filled[p][g_counts.index] - quota[p]) ** 2).sum()) for p in fits}
                best = min(cost, key=cost.get)
            part[idx] = best
            free[best] -= len(idx)
            filled[best] = filled[best].add(g_counts, fill_value=0)
        if ok:
            singles = [members[k][0] for k in keys if len(members[k]) == 1]
            return _balance_by_swaps(part, clips["genre"], singles, share, genre_total)
    raise RuntimeError(f"no exact grouped split found in {max_tries} tries")


def _balance_by_swaps(part: pd.Series, genre: pd.Series, singles: list, share: dict,
                      genre_total: pd.Series) -> pd.Series:
    """Swap single-clip groups of different genres between two parts while that lowers the squared
    distance of the genre counts from their quotas. Sizes and group-disjointness are unchanged."""
    part = part.copy()
    quota = pd.DataFrame({p: share[p] * genre_total for p in PARTS})
    counts = pd.crosstab(genre, part).reindex(index=quota.index, columns=PARTS, fill_value=0).astype(float)
    improved = True
    while improved:
        improved = False
        for i, a in enumerate(singles):
            for b in singles[i + 1:]:
                pa, pb, ga, gb = part[a], part[b], genre[a], genre[b]
                if pa == pb or ga == gb:
                    continue
                cells = [(ga, pa), (ga, pb), (gb, pa), (gb, pb)]
                before = sum((counts.at[g, p] - quota.at[g, p]) ** 2 for g, p in cells)
                moved = {(ga, pa): -1, (ga, pb): 1, (gb, pb): -1, (gb, pa): 1}
                after = sum((counts.at[g, p] + moved[(g, p)] - quota.at[g, p]) ** 2 for g, p in cells)
                if after < before - 1e-9:
                    for (g, p), d in moved.items():
                        counts.at[g, p] += d
                    part[a], part[b] = pb, pa
                    improved = True
    return part


def make_split_D(seed: int) -> pd.DataFrame:
    """Random artist-disjoint split of the fault-filtered clips (the same pool as B' and C).

    Artist groups are credit clusters (credit_clusters); a clip without an artist is its own group.
    B' -> D removes only artist leakage (same pool, same sizes); D -> C compares a random
    artist-disjoint draw with the one published partition.
    """
    clips = _usable()
    in_C = clips["split_C"].notna()
    groups = credit_clusters(clips["artist"])
    groups = groups.where(groups.notna(), "__solo_" + clips["clip_id"])
    clips["part"] = "unused"
    clips.loc[in_C, "part"] = _grouped_parts(clips[in_C], groups, seed)
    out = _expand(clips)
    out.to_csv(SPLITS / f"D_seed{seed}.csv", index=False)
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
    try:
        res = subprocess.run(["git", "tag", "-l", "pre-test-freeze"], capture_output=True, text=True)
    except OSError:  # no git on this machine: stay locked
        return False
    return res.returncode == 0 and res.stdout.strip() == "pre-test-freeze"


def split_file(name: str, seed: int = 0) -> Path:
    return SPLITS / {"A": "A.csv", "B": f"B_seed{seed}.csv", "Bp": f"Bp_seed{seed}.csv",
                     "D": f"D_seed{seed}.csv", "C": "C.csv"}[name]


def load_split(name: str, part: str, seed: int = 0, allow_test: bool = False) -> pd.DataFrame:
    """Rows (windows) of one part of split name in {A, B, Bp, D, C}. The test part is locked."""
    if part == "test" and not (allow_test and is_test_unlocked()):
        raise PermissionError("test set locked until pre-test-freeze tag")
    if part not in (*PARTS, "unused"):
        raise ValueError(f"unknown part {part!r}")
    df = pd.read_csv(split_file(name, seed))
    return df[df["part"] == part].reset_index(drop=True)


def artist_groups(clip_ids: list[str]) -> pd.Series:
    """Artist of each clip (index clip_id); a clip without an artist is its own group."""
    meta = pd.read_csv(SPLITS / "metadata.csv").set_index("clip_id")
    artist = meta.loc[list(clip_ids), "artist"]
    return artist.where(artist.notna(), "__solo_" + artist.index.to_series())


def sample_artists(clip_ids: list[str], fraction: float, seed: int) -> list[str]:
    """Clips of a random round(fraction * n_groups) artist groups (see artist_groups), in input order."""
    groups = artist_groups(clip_ids)
    unique = np.sort(groups.unique())
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(unique, size=max(1, round(fraction * len(unique))), replace=False))
    return [c for c in clip_ids if groups[c] in keep]


def credit_clusters(artists: pd.Series) -> pd.Series:
    """Cluster of each artist credit: credits that share a performer, even through a third, are one.

    Credits split on ', ' and ' & ', so 'Bob Marley & The Wailers' joins 'Bob Marley'. Band names
    such as 'Earth, Wind & Fire' split too, but into parts no other credit uses. Missing stays
    missing (the bootstrap makes such a clip its own cluster). The artist bootstrap uses this;
    artist_groups keeps exact names for the stored validation analyses (DECISIONS.md).
    """
    parent: dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            name = parent[name]
        return name

    credits = artists.dropna().str.replace(" & ", ", ", regex=False).str.split(", ")
    for first, *rest in credits:
        for other in rest:
            parent[find(other)] = find(first)
    return credits.str[0].map(find).reindex(artists.index)
