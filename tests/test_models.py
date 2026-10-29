"""Model factories and the train/valid feature loader."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models import MODELS, knn_pipeline, logreg_pipeline, svm_pipeline
from src.train_classical import load_windows, param_grid


@pytest.mark.parametrize("factory", [knn_pipeline, logreg_pipeline, svm_pipeline])
def test_scaler_is_inside_pipeline_and_fit_on_train_only(factory):
    pipe = factory()
    assert isinstance(pipe.steps[0][1], StandardScaler) and pipe.steps[-1][0] == "clf"
    X = np.r_[np.zeros((10, 2)), np.ones((10, 2))] + np.arange(20)[:, None] * 0.01
    pipe.fit(X, [0] * 10 + [1] * 10)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, X.mean(axis=0))


def test_every_model_gives_probabilities_and_grid_keys_match():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)))
    y = np.repeat(["a", "b"], 20)
    for name, (factory, _) in MODELS.items():
        est = factory().set_params(**{k: v[0] for k, v in param_grid(name).items()})
        p = est.fit(X, y).predict_proba(X)
        assert p.shape == (40, 2) and np.allclose(p.sum(axis=1), 1)


@pytest.mark.skipif(not Path("features/handcrafted.npy").exists(), reason="features not extracted")
def test_load_windows_rows_match_features_index():
    X, y = load_windows("C", "valid")
    assert X.shape == (1970, 57) and X.index.nunique() == 197
    index = pd.read_csv("features/index.csv")
    row = index.index[(index["clip_id"] == X.index[0]) & (index["window_idx"] == 0)][0]
    np.testing.assert_array_equal(X.iloc[0].to_numpy(), np.load("features/handcrafted.npy")[row])
    assert (y.groupby(level=0).nunique() == 1).all()


def test_load_windows_refuses_test():
    with pytest.raises(PermissionError):
        load_windows("C", "test")
