"""Fetch GTZAN artist metadata and the fault-filtered split from gtzan-bind.

Run from repo root: python -m src.fetch_meta  -> splits/gtzan_bind.csv
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.utils import load_config

COLUMNS = ["track_id", "path", "title", "artist_name", "album", "fault_filtered_split"]


def map_to_clip_id(path: str) -> str:
    """gtzan-bind path 'reggae/reggae.00052.wav' -> our clip_id 'reggae.00052'."""
    return Path(path).stem


def fetch_bind() -> pd.DataFrame:
    """Download gtzan-bind, keep COLUMNS, add clip_id; empty strings become NaN."""
    ds = load_dataset("seungheondoh/gtzan-bind", split="gtzan_bind_v1")
    df = ds.select_columns(COLUMNS).to_pandas()
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df.insert(0, "clip_id", df["path"].map(map_to_clip_id))
    return df


def main() -> None:
    cfg = load_config()
    df = fetch_bind()
    ours = set(pd.read_csv(Path(cfg["paths"]["splits"]) / "metadata.csv")["clip_id"])
    # one-to-one: every row maps to a distinct clip_id, the set equals our 1000 files,
    # and the path stem agrees with gtzan-bind's own track_id
    assert df["clip_id"].is_unique and len(df) == 1000, "mapping not one-to-one"
    assert set(df["clip_id"]) == ours, "clip_ids differ from our audio files"
    assert (df["clip_id"] == df["track_id"]).all(), "path stem != track_id"
    out = Path(cfg["paths"]["splits"]) / "gtzan_bind.csv"
    df.drop(columns="clip_id").to_csv(out, index=False)

    print("rows:", len(df))
    print(df[["clip_id", "track_id", "path", "artist_name", "fault_filtered_split"]].head(5).to_string())
    print(df["fault_filtered_split"].value_counts(dropna=False).to_string())
    print("missing artist_name:", df["artist_name"].isna().sum())


if __name__ == "__main__":
    main()
