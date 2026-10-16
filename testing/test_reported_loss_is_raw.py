"""The loss a step reports is the raw cost, whatever the L2 weight (ml-engine#52): the penalty is scaled by the batch size in a train loss and by the validation-set size in a
validation loss, so a penalised pair is not comparable; training itself (the gradient) is unchanged."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import training_session
from src.model_factory import ModelFactory


def _model(lam_l2: float):
    np.random.seed(0)
    return ModelFactory.create_model(model_type="binary_classification", layer_sizes=(4, 8, 1), lr=0.01, optimizer="adam", lam_l1=0.0, lam_l2=lam_l2,
                                     p_dropout=0.0, use_batch_norm=False, bn_momentum=0.9)


def test_step_loss_is_raw_cost_with_weight_decay():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    y = (rng.random((32, 1)) > 0.5).astype(np.float32)
    plain, decayed = _model(0.0), _model(5.0)
    decayed.weights = [w.copy() for w in plain.weights]
    decayed.biases = [b.copy() for b in plain.biases]
    loss_plain = plain._compute_grads(X, y)[0]
    loss_decayed, gw_decayed = decayed._compute_grads(X, y)[:2]
    assert loss_decayed == loss_plain                       # the reported number does not carry the penalty
    assert decayed.compute_total_loss(decayed.forward(X) if hasattr(decayed, "forward") else decayed.predict(X), y) > loss_decayed   # the penalised value still exists for whoever asks


def test_engine_declares_raw_reporting():
    assert training_session.REPORTED_LOSS_INCLUDES_L2 is False



def _cnn(lam_l2: float):
    from config.constants import EngineBackend

    np.random.seed(3)
    cfg = {"input_shape": [1, 6, 6], "spatial_pipeline": [{"type": "conv", "in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "pad": 0}, {"type": "relu"}, {"type": "flatten"}], "dense_head": [4]}
    return ModelFactory.create_model(model_type="cnn", layer_sizes=[36, 3], optimizer="sgd", lr=0.01, lam_l1=0.0, lam_l2=lam_l2, p_dropout=0.0, use_batch_norm=False, max_norm=1000.0,
                                     cnn_config=cfg, backend=EngineBackend.NUMPY)


def test_cnn_step_loss_is_raw_cost_with_weight_decay():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 1, 6, 6)).astype(np.float32)
    y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=4)]
    plain, decayed = _cnn(0.0), _cnn(5.0)
    decayed.weights, decayed.biases = [w.copy() for w in plain.weights], [b.copy() for b in plain.biases]
    out_p, cache_p = plain.forward_train(X)
    out_d, cache_d = decayed.forward_train(X)
    loss_p = plain._compute_grads_from_cache(cache_p, y)[0]
    loss_d = decayed._compute_grads_from_cache(cache_d, y)[0]
    assert loss_d == loss_p == decayed.calculate_raw_cost(out_d, y)
    assert decayed.compute_total_loss(out_d, y) > loss_d          # the penalised number still exists for whoever asks


def test_validation_loss_reported_by_a_fit_epoch_is_the_raw_cost():
    """``advance_fit_epoch`` hands the engine the raw validation cost, not the penalised one (the source line, read from the file: a fit needs a data provider, and the real-run check is in the PR)."""
    import inspect
    import re

    from src import training_session

    src = inspect.getsource(training_session.TrainingSession.advance_fit_epoch)
    assert re.search(r"current_val_loss\s*=\s*current_val_raw_cost", src)
    assert "compute_total_loss" not in src
