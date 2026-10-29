import numpy as np
import torch

from src.utils import load_config, set_seed


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["audio"]["sr"] == 22050
    assert sum(cfg["split_sizes"].values()) == 930
    assert len(cfg["genres"]) == 10


def test_set_seed_repeats_draws():
    set_seed(42)
    a_np, a_t = np.random.rand(5), torch.rand(5)
    set_seed(42)
    b_np, b_t = np.random.rand(5), torch.rand(5)
    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_t, b_t)
