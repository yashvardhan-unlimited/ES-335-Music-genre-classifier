"""scripts/evaluate_part.py and src/predict.py: the test lock, refit guard, run list, bootstrap seeds."""
import importlib.util
import sys
from pathlib import Path

import pytest

from src import predict
from src.utils import load_config

CFG = load_config()
_spec = importlib.util.spec_from_file_location("evaluate_part", Path("scripts/evaluate_part.py"))
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def test_test_part_refuses_without_the_tag(monkeypatch):
    monkeypatch.setattr(ep, "is_test_unlocked", lambda: False)
    monkeypatch.setattr(sys, "argv", ["evaluate_part.py", "--part", "test"])
    with pytest.raises(SystemExit, match="pre-test-freeze"):
        ep.main()


def test_refit_models_are_never_scored_on_valid():
    with pytest.raises(ValueError, match="test only"):
        predict.clip_predictions("C", 0, "svm", "valid", refit=True)


def test_run_list_covers_every_split_model_and_cnn_seed():
    runs = ep.runs()
    n_seeds = len(CFG["split_seeds"])
    clip_sets = 1 + 3 * n_seeds + 1                          # A, B/B'/D seeds, C
    extra_cnn = len(CFG["cnn"]["seeds"]) - 1                 # C has several CNN runs on one clip set
    assert len(runs) == clip_sets * len(ep.MODELS) + extra_cnn
    assert {r[3] for r in runs if r[0] == "C" and r[2] == "cnn"} == set(CFG["cnn"]["seeds"])
    assert {r[1] for r in runs if r[0] == "C"} == {0}       # one clip set: all C runs are paired


def test_bootstrap_seed_is_per_clip_set():
    seeds = {(sp, s): ep.boot_seed(sp, s) for sp, ss in ep.SPLIT_SEEDS.items() for s in ss}
    assert len(set(seeds.values())) == len(seeds)
