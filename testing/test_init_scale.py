# testing/test_init_scale.py
"""The generator's "bad seed" cause planted as a scale of the starting weights (data backlog C7 note, 4c46cab): ``optimization.init_scale``, default 1.0."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import parse_tm_production_config
from config.constants import EngineBackend, ModelType
from src.controller import ModelController
from src.launcher.payload import build_pipeline_payload
from testing.test_launcher import _spec
from utils.seeding import seed_everything


def _weights(**kw):
    seed_everything(7)
    ctl = ModelController(data_provider=None, learning_rate=0.05)
    ctl.initialize_network_from_dimensions(input_dim=4, output_dim=1, model_type=ModelType.BINARY_CLASSIFICATION, hidden_layers=[6, 3], optimizer_name="adam",
                                           backend=EngineBackend.NUMPY, use_batch_norm=True, **kw)
    return ctl.model


def test_the_default_scale_gives_the_weights_of_every_run_so_far_and_another_scale_multiplies_them() -> None:
    base, one, half = _weights(), _weights(init_scale=1.0), _weights(init_scale=0.5)
    for a, b, c in zip(base.weights, one.weights, half.weights):
        assert np.array_equal(a, b) and np.array_equal(a * 0.5, c)
    assert all(np.array_equal(a, b) for a, b in zip(base.gammas, half.gammas)) and all(not x.any() for x in half.biases)      # only the weights are scaled


def test_the_scale_is_for_the_plain_network_only() -> None:
    ctl = ModelController(data_provider=None, learning_rate=0.05)
    with pytest.raises(ValueError, match="plain network"):
        ctl.initialize_network_from_dimensions(input_dim=4, output_dim=2, model_type=ModelType.CNN, hidden_layers=[], optimizer_name="adam", backend=EngineBackend.NUMPY,
                                               cnn_config={"input_shape": [1, 4, 4], "spatial_pipeline": [], "dense_head": []}, init_scale=2.0)


def test_the_option_reaches_the_payload_and_the_config_only_when_set(tmp_path) -> None:
    plain = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    assert "init_scale" not in plain["optimization"] and parse_tm_production_config(plain, profile="pipeline").optimization.init_scale == 1.0
    on = build_pipeline_payload(_spec(tmp_path, init_scale=0.25), work_dir=tmp_path / "w")
    assert parse_tm_production_config(on, profile="pipeline").optimization.init_scale == 0.25
