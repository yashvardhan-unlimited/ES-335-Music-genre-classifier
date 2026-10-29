"""Tiny hand-checkable cases for src/evaluate.py."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.evaluate import (ClipMacroF1Scorer, _cluster_rows, aggregate_clips, clip_scores,
                          cluster_bootstrap_ci, macro_f1, make_predefined_split,
                          paired_bootstrap_diff, window_accuracy)


def test_soft_vote_averages_probabilities():
    # clip "a" = rows 0-2: mean p0 = (0.9 + 0.2 + 0.3) / 3 = 0.467 < 0.533 -> class 1
    # clip "b" = row 3 (listed first, ids come back sorted): (0.4, 0.6) -> class 1
    probs = np.array([[0.4, 0.6], [0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    ids, clip_probs, pred = aggregate_clips(probs, ["b", "a", "a", "a"])
    assert list(ids) == ["a", "b"]
    np.testing.assert_allclose(clip_probs, [[1.4 / 3, 1.6 / 3], [0.4, 0.6]])
    assert list(pred) == [1, 1]


def test_soft_and_majority_can_disagree():
    # two confident votes for 0 lose to three weak votes for 1 under majority,
    # but win under soft voting: mean p0 = (0.99*2 + 0.4*3)/5 = 0.636
    probs = np.array([[0.99, 0.01]] * 2 + [[0.4, 0.6]] * 3)
    ids = ["c"] * 5
    assert aggregate_clips(probs, ids, "soft")[2][0] == 0
    _, shares, pred = aggregate_clips(probs, ids, "majority")
    assert pred[0] == 1
    np.testing.assert_allclose(shares, [[0.4, 0.6]])


def test_majority_tie_broken_by_mean_probability():
    # 4 windows, 2 votes each for class 0 and 1; mean p1 = 0.45 > mean p0 = 0.40
    probs = np.array([[0.6, 0.3, 0.1],
                      [0.5, 0.4, 0.1],
                      [0.2, 0.7, 0.1],
                      [0.3, 0.4, 0.3]])
    _, shares, pred = aggregate_clips(probs, ["t"] * 4, "majority")
    np.testing.assert_allclose(shares, [[0.5, 0.5, 0.0]])
    assert pred[0] == 1


def test_bad_method_raises():
    with pytest.raises(ValueError):
        aggregate_clips(np.ones((2, 2)) / 2, ["a", "a"], "median")


def test_three_class_macro_f1_by_hand():
    # class 0: P=2/2, R=2/3, F1=0.8 | class 1: P=1/2, R=1/2, F1=0.5 | class 2: P=1/2, R=1/1, F1=2/3
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    s = clip_scores(y_true, y_pred, labels=[0, 1, 2])
    assert s["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)   # 0.6556
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["per_genre_f1"] == pytest.approx({0: 0.8, 1: 0.5, 2: 2 / 3})
    np.testing.assert_array_equal(s["confusion"], [[2, 1, 0], [0, 1, 1], [0, 0, 1]])
    np.testing.assert_allclose(s["confusion_norm"], [[2 / 3, 1 / 3, 0], [0, 0.5, 0.5], [0, 0, 1]])
    assert macro_f1(y_true, y_pred) == pytest.approx(s["macro_f1"])


def test_absent_class_counts_as_zero_and_empty_row_is_zero():
    s = clip_scores(["rock", "rock"], ["rock", "rock"], labels=["jazz", "rock"])
    assert s["per_genre_f1"] == {"jazz": 0.0, "rock": 1.0}
    assert s["macro_f1"] == 0.5
    np.testing.assert_array_equal(s["confusion_norm"], [[0, 0], [0, 1]])


def test_window_accuracy():
    assert window_accuracy([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5


def test_predefined_split():
    ps = make_predefined_split(["train", "valid", "train", "valid"])
    (tr, va), = list(ps.split())
    assert list(tr) == [0, 2] and list(va) == [1, 3]
    with pytest.raises(ValueError, match="test"):
        make_predefined_split(["train", "test"])


class FixedProba:
    """Stub estimator returning stored probabilities for the rows of X."""
    classes_ = np.array(["jazz", "rock"])

    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, X):
        return self.probs.loc[X.index].to_numpy()


def test_clip_scorer_aggregates_through_the_index():
    # rows r0-r3; clips: a = r0,r1 (jazz), b = r2,r3 (rock)
    X = pd.DataFrame({"f": [0, 0, 0, 0]}, index=["r0", "r1", "r2", "r3"])
    probs = pd.DataFrame([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]], index=X.index)
    clip_ids = pd.Series(["a", "a", "b", "b"], index=X.index)
    y = ["jazz", "jazz", "rock", "rock"]
    scorer = ClipMacroF1Scorer(clip_ids)
    # a: mean (0.55, 0.45) -> jazz ok; b: mean (0.45, 0.55) -> rock ok
    assert scorer(FixedProba(probs), X, y) == 1.0
    # shuffled rows must give the same result (alignment via index, not position)
    order = ["r3", "r0", "r2", "r1"]
    assert scorer(FixedProba(probs), X.loc[order], [y[int(r[1])] for r in order]) == 1.0
    # window accuracy is only 2/4 here, clip score is still 1.0
    assert window_accuracy(y, FixedProba.classes_[probs.to_numpy().argmax(1)]) == 0.5


def test_clip_scorer_works_inside_gridsearch():
    rng = np.random.default_rng(0)
    n_clips, w = 20, 5
    y_clip = np.repeat(["a", "b"], n_clips // 2)
    X = pd.DataFrame(rng.normal(size=(n_clips * w, 2)) + np.repeat((y_clip == "b") * 2.0, w)[:, None],
                     index=np.repeat([f"c{i}" for i in range(n_clips)], w))
    y = np.repeat(y_clip, w)
    parts = np.repeat(["train", "valid"] * (n_clips // 2), w)
    gs = GridSearchCV(LogisticRegression(), {"C": [0.01, 1.0]}, scoring=ClipMacroF1Scorer(),
                      cv=make_predefined_split(parts), refit=False)
    gs.fit(X, y)
    assert gs.cv_results_["mean_test_score"].max() == 1.0


def test_missing_artist_is_its_own_cluster():
    df = pd.DataFrame({"artist": ["x", "x", None, None, "y"]})
    groups = _cluster_rows(df, "artist")
    assert sorted(map(list, groups)) == [[0, 1], [2], [3], [4]]


def test_cluster_bootstrap_perfect_predictions():
    df = pd.DataFrame({"y_true": [0, 1, 2, 0], "y_pred": [0, 1, 2, 0], "artist": ["x", "x", "y", None]})
    assert cluster_bootstrap_ci(df, macro_f1, n=200) == (1.0, 1.0, 1.0)


def test_cluster_bootstrap_two_clusters_spans_zero_to_one():
    # artist x: both clips right; artist y: both wrong. Resamples are xx, xy, yx, yy,
    # so accuracy is 1, 0.5 or 0 -> 95% interval [0, 1]; point estimate 0.5.
    acc = lambda t, p: float(np.mean(t == p))  # noqa: E731
    df = pd.DataFrame({"y_true": [1, 1, 1, 1], "y_pred": [1, 1, 0, 0], "artist": ["x", "x", "y", "y"]})
    assert cluster_bootstrap_ci(df, acc, n=500, seed=1) == (0.5, 0.0, 1.0)
    # same seed -> same interval
    assert cluster_bootstrap_ci(df, acc, n=500, seed=1) == cluster_bootstrap_ci(df, acc, n=500, seed=1)


def test_paired_bootstrap():
    df = pd.DataFrame({"y_true": [0, 1, 0, 1, 1], "a": [0, 1, 0, 1, 1], "b": [1, 0, 1, 0, 0],
                       "artist": ["x", "y", "y", "z", None]})
    acc = lambda t, p: float(np.mean(t == p))  # noqa: E731
    assert paired_bootstrap_diff(df, "a", "b", acc, n=300) == (1.0, 1.0, 1.0, 1.0)
    assert paired_bootstrap_diff(df, "a", "a", acc, n=300) == (0.0, 0.0, 0.0, 0.0)
