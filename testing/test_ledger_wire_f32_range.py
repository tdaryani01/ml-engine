"""A diverged run's loss can be finite and beyond the 32-bit range: the wire must record it (as infinity), not raise OverflowError and kill the ledger writer."""
import math

import numpy as np

from src.ledger_wire import _f32, pack_step_result, unpack_step_result
from src.training_session import TrainStepResult


def _result(loss: float) -> TrainStepResult:
    return TrainStepResult(step_id=7, loss=loss, grad_weights=[np.zeros((2, 2), dtype=np.float32)], grad_biases=[np.zeros(2, dtype=np.float32)], m_samples=32)


def test_a_loss_beyond_float32_is_saturated_to_infinity_not_an_error():
    assert math.isinf(unpack_step_result(pack_step_result(_result(1e300))).loss) and unpack_step_result(pack_step_result(_result(1e300))).loss > 0
    assert unpack_step_result(pack_step_result(_result(-1e300))).loss == -math.inf
    assert math.isinf(unpack_step_result(pack_step_result(_result(3.5e38))).loss)           # just over the float32 maximum


def test_every_other_value_is_unchanged_bit_for_bit():
    for v in (0.0, 0.3, -2.5, 1e-30, float(np.finfo(np.float32).max), float("inf"), float("-inf")):
        got = unpack_step_result(pack_step_result(_result(v))).loss
        assert got == float(np.float32(v)) or (math.isinf(v) and got == v)
    assert math.isnan(unpack_step_result(pack_step_result(_result(float("nan")))).loss)
    assert _f32(0.1) == 0.1 and _f32(1.0e38) == 1.0e38
