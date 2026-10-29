import pandas as pd
import pytest

from src.splits import PARTS, SIZES, load_split, split_file
from src.utils import load_config

CFG = load_config()
META = pd.read_csv("splits/metadata.csv").set_index("clip_id")
SEEDS = CFG["split_seeds"]
CLIP_SPLITS = [("B", s) for s in SEEDS] + [("Bp", s) for s in SEEDS] + [("C", 0)]
# Published C is artist-disjoint by Sturm's labels but not by gtzan-bind's names;
# these 13 cross parts (DECISIONS.md). Any change to C would change this set.
C_CROSSING_ARTISTS = {
    "Anthrax", "Bob Marley", "Boney M.", "Burning Spear", "Danzig", "Dark Tranquillity",
    "Earth, Wind & Fire", "Freddie McGregor", "KC and the Sunshine Band", "Metallica",
    "Peter Tosh", "Slayer",
    "Juilliard String Quartet, Robert Mann, Joel Smirnoff, Samuel Rhodes & Joel Krosnick",
}


def read(name, seed=0):
    return pd.read_csv(split_file(name, seed))


def test_split_A_window_counts():
    df = read("A")
    assert len(df) == 9990
    for p in PARTS:
        assert (df["part"] == p).sum() == SIZES[p] * 10
    assert (df["part"] == "unused").sum() == 9990 - 10 * sum(SIZES.values())


@pytest.mark.parametrize("name,seed", CLIP_SPLITS)
def test_clip_splits_exact_counts_and_whole_clips(name, seed):
    df = read(name, seed)
    assert len(df) == 9990
    per_clip = df.groupby("clip_id")["part"].agg(["nunique", "first", "size"])
    assert (per_clip["nunique"] == 1).all()          # every window of a clip shares one part
    assert (per_clip["size"] == 10).all()
    for p in PARTS:
        assert (per_clip["first"] == p).sum() == SIZES[p]


@pytest.mark.parametrize("name,seed", [("Bp", s) for s in SEEDS] + [("C", 0)])
def test_Bprime_and_C_use_only_fault_filtered_clips(name, seed):
    df = read(name, seed)
    used = df.loc[df["part"] != "unused", "clip_id"].unique()
    assert META.loc[used, "split_C"].notna().all()


def test_C_matches_published_split():
    df = read("C").drop_duplicates("clip_id").set_index("clip_id")
    used = df[df["part"] != "unused"]
    assert (used["part"] == META.loc[used.index, "split_C"]).all()
    assert "jazz.00054" not in df.index           # unreadable, so no windows anywhere


def test_C_artist_overlap_is_exactly_the_documented_set():
    c = META[META["split_C"].notna()].dropna(subset=["artist"])
    n_parts = c.groupby("artist")["split_C"].nunique()
    assert set(n_parts[n_parts > 1].index) == C_CROSSING_ARTISTS


def test_test_part_is_locked():
    with pytest.raises(PermissionError, match="pre-test-freeze"):
        load_split("C", "test")
    with pytest.raises(PermissionError):
        load_split("C", "test", allow_test=True)   # no tag yet
    assert len(load_split("C", "train")) == SIZES["train"] * 10
