# testing/test_run_seed.py
"""Fit contract: ``optimization.seed`` makes a fit deterministic (all of ME's randomness is numpy's global RNG)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import parse_tm_production_config
from utils.seeding import seed_everything


def test_seed_everything_makes_global_numpy_draws_repeatable() -> None:
    seed_everything(5)
    a = np.random.rand(4)
    seed_everything(5)
    b = np.random.rand(4)
    np.testing.assert_array_equal(a, b)


def test_no_seed_is_a_no_op() -> None:
    seed_everything(None)  # must not raise and must not reseed deterministically


def test_the_tm_payload_carries_the_seed_into_the_config() -> None:
    from testing.test_config import _tm_payload

    payload = _tm_payload()
    payload["optimization"]["seed"] = 11
    assert parse_tm_production_config(payload).optimization.seed == 11
    assert parse_tm_production_config(_tm_payload()).optimization.seed is None
