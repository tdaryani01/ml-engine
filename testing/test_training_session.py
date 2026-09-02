# testing/test_training_session.py
"""Phase C gates: TrainingSession, TrainStepResult, grad/apply split."""
import copy
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import EngineBackend
from src.model_factory import ModelFactory
from src.training_session import TrainStepResult, TrainingSession


def _tiny_cnn():
    cnn_config = {
        "input_shape": [1, 28, 28],
        "spatial_pipeline": [
            {"type": "conv", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1},
            {"type": "relu"},
            {"type": "pool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
        ],
        "dense_head": [8],
    }
    return ModelFactory.create_model(
        model_type="cnn",
        layer_sizes=[4],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        cnn_config=cnn_config,
    )


def _clone_cnn_weights(template, target) -> None:
    for w_t, w_s in zip(target.weights, template.weights):
        w_t[...] = w_s
    for b_t, b_s in zip(target.biases, template.biases):
        b_t[...] = b_s
    if hasattr(target, "gammas") and hasattr(template, "gammas"):
        for g_t, g_s in zip(target.gammas, template.gammas):
            g_t[...] = g_s
    if hasattr(target, "betas") and hasattr(template, "betas"):
        for b_t, b_s in zip(target.betas, template.betas):
            b_t[...] = b_s


def _batch():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 1, 28, 32), dtype=np.float32)
    y = np.eye(4, dtype=np.float32)
    return X, y


def test_c1_train_step_result_serializable():
    gw = [np.ones((2, 3), dtype=np.float32)]
    gb = [np.zeros((1, 3), dtype=np.float32)]
    result = TrainStepResult(step_id=1, loss=0.5, grad_weights=gw, grad_biases=gb, m_samples=4)
    round_trip = pickle.loads(pickle.dumps(result))
    assert round_trip.step_id == 1
    assert np.allclose(round_trip.grad_weights[0], gw[0])
    print("[PASSED] C1: TrainStepResult serializable")


def test_c2_train_step_no_weight_change():
    model = _tiny_cnn()
    session = TrainingSession(model=model, data_provider=None, initial_lr=0.01)
    X, y = _batch()
    w_before = copy.deepcopy(model.weights)
    result = session.train_step(X, y, lr=0.01)
    assert isinstance(result, TrainStepResult)
    for a, b in zip(model.weights, w_before):
        assert np.allclose(a, b)
    print("[PASSED] C2: train_step does not mutate weights")


def test_c3_apply_step_matches_backward():
    rng = np.random.default_rng(1)

    for _ in range(3):
        model_a = _tiny_cnn()
        model_b = _tiny_cnn()
        for wa, wb, ba, bb in zip(model_a.weights, model_b.weights, model_a.biases, model_b.biases):
            wa[...] = wb
            ba[...] = bb

        X_step = rng.standard_normal((4, 1, 28, 32), dtype=np.float32)
        y_step = np.eye(4, dtype=np.float32)

        loss_legacy = model_a.backward(X_step, y_step, active_lr=0.01)

        sess = TrainingSession(model=model_b, data_provider=None, initial_lr=0.01)
        result = sess.train_step(X_step, y_step, lr=0.01)
        sess.apply_step(result, lr=0.01)

        for wa, wb in zip(model_a.weights, model_b.weights):
            assert np.allclose(wa, wb, rtol=1e-5, atol=1e-5)
        assert abs(loss_legacy - result.loss) < 1e-5
    print("[PASSED] C3: train_step + apply_step matches backward")


def test_c_two_training_sessions():
    """Phase C exit gate: two sessions do not cross-contaminate."""
    model_a = _tiny_cnn()
    model_b = _tiny_cnn()
    session_a = TrainingSession(model=model_a, data_provider=None, initial_lr=0.01)
    session_b = TrainingSession(model=model_b, data_provider=None, initial_lr=0.01)
    X, y = _batch()

    w_b_before = [np.copy(w) for w in model_b.weights]
    opt_t_b_before = getattr(model_b.optimizer, "t", 0)

    result = session_a.train_step(X, y, lr=0.01)
    session_a.apply_step(result, lr=0.01)

    for w, ref in zip(model_b.weights, w_b_before):
        assert np.allclose(w, ref)
    assert getattr(model_b.optimizer, "t", 0) == opt_t_b_before
    assert session_a.step_id == 1
    assert session_b.step_id == 0
    print("[PASSED] C: two TrainingSession instances are independent")


def test_e_bridge_sequential_worker_grad_parity():
    """E-bridge: two sequential workers (same weights) emit identical TrainStepResult."""
    template = _tiny_cnn()
    X, y = _batch()

    worker_a = _tiny_cnn()
    worker_b = _tiny_cnn()
    _clone_cnn_weights(template, worker_a)
    _clone_cnn_weights(template, worker_b)

    result_a = TrainingSession(model=worker_a, data_provider=None, initial_lr=0.01).train_step(X, y, lr=0.01)
    result_b = TrainingSession(model=worker_b, data_provider=None, initial_lr=0.01).train_step(X, y, lr=0.01)

    assert abs(result_a.loss - result_b.loss) < 1e-5
    for gw_a, gw_b in zip(result_a.grad_weights, result_b.grad_weights):
        assert np.allclose(gw_a, gw_b, rtol=1e-5, atol=1e-5)
    for gb_a, gb_b in zip(result_a.grad_biases, result_b.grad_biases):
        assert np.allclose(gb_a, gb_b, rtol=1e-5, atol=1e-5)
    print("[PASSED] E-bridge: sequential worker grad parity (native, separate models)")


PHASE_C_TESTS = [
    test_c1_train_step_result_serializable,
    test_c2_train_step_no_weight_change,
    test_c3_apply_step_matches_backward,
    test_c_two_training_sessions,
    test_e_bridge_sequential_worker_grad_parity,
]


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PHASE C TRAINING SESSION TESTS ")
    print("=" * 60)
    failed = []
    for fn in PHASE_C_TESTS:
        try:
            fn()
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"[FAILED] {fn.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"[FAILURE] {len(failed)} test(s) failed.")
        sys.exit(1)
    print(f"[SUCCESS] All {len(PHASE_C_TESTS)} Phase C tests passed.")
    print("=" * 60)
