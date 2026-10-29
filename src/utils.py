"""Shared helpers: config loading, seeding, logging."""
import logging
import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Read the project config (all constants live here)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline uses: random, numpy, torch (CPU and CUDA).

    PYTHONHASHSEED only affects interpreters started after this call
    (e.g. DataLoader workers); the current process's hash seed is fixed at startup.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints 'time level name: message' to stderr, once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
