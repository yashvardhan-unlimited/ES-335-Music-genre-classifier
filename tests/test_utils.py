import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import load_config, set_seed, write_json


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["audio"]["sr"] == 22050
    assert sum(cfg["split_sizes"].values()) == 929
    assert len(cfg["genres"]) == 10


def test_set_seed_repeats_draws():
    set_seed(42)
    a_np, a_t = np.random.rand(5), torch.rand(5)
    set_seed(42)
    b_np, b_t = np.random.rand(5), torch.rand(5)
    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_t, b_t)


def test_write_json_never_leaves_half_a_file(tmp_path, monkeypatch):
    path = tmp_path / "r.json"
    write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1} and [p.name for p in tmp_path.iterdir()] == ["r.json"]

    def killed(*_):
        raise OSError("killed before the swap")
    monkeypatch.setattr(Path, "replace", killed)
    with pytest.raises(OSError):
        write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}     # the old result survives whole
