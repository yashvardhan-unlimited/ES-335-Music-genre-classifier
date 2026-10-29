"""Build splits A, B (seeds), B' (seeds), C and results/split_summary.json.

Run from repo root: python scripts/make_splits.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.splits import (PARTS, add_split_C_to_metadata, make_split_A, make_split_B,  # noqa: E402
                        make_split_Bprime, make_split_C)
from src.utils import get_logger, load_config  # noqa: E402

log = get_logger("make_splits")


def summarise(win: pd.DataFrame, meta: pd.DataFrame, genres: list[str]) -> dict:
    """Genre counts per part and three leakage measures for the test part.

    A clip counts as 'in' a part if any of its windows is there (matters for A only).
    Shared artist: test clip whose (non-missing) artist has a clip in train.
    """
    m = meta.set_index("clip_id")
    in_part = {p: set(win.loc[win["part"] == p, "clip_id"]) for p in PARTS}
    test, train = in_part["test"], in_part["train"]
    train_artists = set(m.loc[list(train), "artist"].dropna())
    train_md5 = set(m.loc[list(train), "md5"])
    test_art = m.loc[list(test), "artist"]
    return {
        "windows_per_part": {p: int((win["part"] == p).sum()) for p in PARTS},
        "clips_per_part": {p: len(in_part[p]) for p in PARTS},
        "genre_counts": {p: {g: int(m.loc[list(in_part[p]), "genre"].eq(g).sum()) for g in genres}
                         for p in PARTS},
        "test_clips_also_in_train": len(test & train),
        "test_clips_with_identical_file_in_train": sum(
            m.at[c, "md5"] in train_md5 and c not in train for c in test),
        "test_clips_missing_artist": int(test_art.isna().sum()),
        "test_clips_artist_in_train": int(test_art.isin(train_artists).sum()),
        "share_test_clips_artist_in_train": round(float(test_art.isin(train_artists).mean()), 4),
    }


def main() -> None:
    cfg = load_config()
    meta = add_split_C_to_metadata()
    log.info("jazz.00054 split_C=%s load_ok=%s", *meta.set_index("clip_id").loc["jazz.00054", ["split_C", "load_ok"]])

    C = meta[meta["split_C"].notna()]
    crossing = C.dropna(subset=["artist"]).groupby("artist")["split_C"].nunique()
    crossing = sorted(crossing[crossing > 1].index)
    log.info("%d artists appear in more than one split_C part", len(crossing))

    splits = {"A": make_split_A(seed=cfg["split_seeds"][0])}
    for s in cfg["split_seeds"]:
        splits[f"B_seed{s}"] = make_split_B(s)
        splits[f"Bp_seed{s}"] = make_split_Bprime(s)
    splits["C"] = make_split_C()

    summary = {name: summarise(win, meta, cfg["genres"]) for name, win in splits.items()}
    summary["C_artists_in_more_than_one_part"] = crossing
    out = Path(cfg["paths"]["results"]) / "split_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for name, s in summary.items():
        if isinstance(s, dict):
            log.info("%-9s clips %s  test clips also in train %3d  identical file in train %d  "
                     "artist in train %3d (%.1f%%)", name, list(s["clips_per_part"].values()),
                     s["test_clips_also_in_train"], s["test_clips_with_identical_file_in_train"],
                     s["test_clips_artist_in_train"], 100 * s["share_test_clips_artist_in_train"])


if __name__ == "__main__":
    main()
