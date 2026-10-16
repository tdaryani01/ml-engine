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
