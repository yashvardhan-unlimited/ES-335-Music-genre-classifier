import pandas as pd
import pytest

from src import splits
from src.splits import PARTS, SIZES, credit_clusters, load_split, split_file
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


def test_test_part_is_locked_without_the_tag(monkeypatch):
    monkeypatch.setattr(splits, "is_test_unlocked", lambda: False)
    with pytest.raises(PermissionError, match="pre-test-freeze"):
        load_split("C", "test")
    with pytest.raises(PermissionError):
        load_split("C", "test", allow_test=True)
    assert len(load_split("C", "train")) == SIZES["train"] * 10


def test_test_part_needs_the_tag_and_the_explicit_flag(monkeypatch, tmp_path):
    # a stand-in split file, so no real test row is read even when the lock is open
    pd.DataFrame({"clip_id": ["x"], "window_idx": [0], "genre": ["rock"], "part": ["test"]}).to_csv(
        tmp_path / "C.csv", index=False)
    monkeypatch.setattr(splits, "SPLITS", tmp_path)
    monkeypatch.setattr(splits, "is_test_unlocked", lambda: True)
    with pytest.raises(PermissionError):
        load_split("C", "test")                    # the tag alone does not open it
    assert len(load_split("C", "test", allow_test=True)) == 1


def test_is_test_unlocked_stays_locked_without_git(monkeypatch):
    def no_git(*_, **__):
        raise FileNotFoundError("git")
    monkeypatch.setattr(splits.subprocess, "run", no_git)
    assert splits.is_test_unlocked() is False


def test_unknown_part_is_an_error_not_an_empty_table():
    with pytest.raises(ValueError, match="vaild"):
        load_split("C", "vaild")


def test_credit_clusters_link_any_shared_performer():
    a = pd.Series(["Bob Marley & The Wailers", "Bob Marley", "Big Youth & Dennis Brown", "Dennis Brown, Bongo Herman",
                   "Bongo Herman", "Earth, Wind & Fire", None])
    c = credit_clusters(a)
    assert c[0] == c[1] and c[2] == c[3] == c[4]         # the link can run through a third credit
    assert c[0] != c[2] and c[5] not in (c[0], c[2])
    assert pd.isna(c[6])                                 # no artist stays missing: the bootstrap makes it solo


def test_credit_clusters_are_what_the_bootstrap_needs_on_C():
    # exact names say 22 test clips share an artist with train; shared credits find 3 more
    c = META[META["load_ok"]].assign(cluster=credit_clusters(META["artist"]))
    train = set(c.loc[c["split_C"] == "train", "cluster"].dropna())
    test = c[c["split_C"] == "test"]
    assert credit_clusters(META["artist"]).nunique() == 294 and META["artist"].nunique() == 321
    assert test["cluster"].isin(train).sum() == 25 and test["artist"].isin(set(c.loc[c["split_C"] == "train", "artist"].dropna())).sum() == 22
    assert test["cluster"].nunique() == 113
