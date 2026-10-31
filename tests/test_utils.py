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


def test_lesson_reference_is_repaired_and_a_bare_name_is_left_alone():
    from scripts.check_lesson_refs import fix_line
    defs = {"load_split": [("src/splits.py", 117, 124)], "main": [("a.py", 1, 2), ("b.py", 5, 9)]}
    assert fix_line("- **L100-110 `load_split`**: lock", defs, None)[0] == "- **L117-124 `load_split`**: lock"
    assert fix_line("Lines 117-124, `load_split`", defs, None) == ("Lines 117-124, `load_split`", [])
    assert fix_line("L1-2 `main`", defs, "b.py")[0] == "L5-9 `main`"          # the current file settles it
    assert fix_line("L1-2 `main`", defs, None)[0] == "L1-2 `main`"            # ambiguous: untouched
    assert fix_line("L3 `unknown_name`", defs, None)[1] == []
