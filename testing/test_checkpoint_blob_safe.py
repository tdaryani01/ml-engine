"""A checkpoint blob is loaded without ever running code from it (security review 2026-10-14)."""
import os
import pickle

import numpy as np
import pytest

from src.manager_heartbeat import decode_checkpoint_blob, encode_checkpoint_blob


def test_a_normal_blob_round_trips_with_its_arrays():
    body = {"version": 7, "weights": [np.arange(6, dtype=np.float32).reshape(2, 3)], "cfg": {"lr": 0.01, "name": "x", "flags": [True, None]}}
    out = decode_checkpoint_blob(encode_checkpoint_blob(body))
    assert out["version"] == 7 and out["cfg"] == body["cfg"]
    assert np.array_equal(out["weights"][0], body["weights"][0]) and out["weights"][0].dtype == np.float32


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned > /tmp/should-never-exist-checkpoint-test",))


def test_a_blob_that_would_run_a_command_is_refused_and_nothing_runs(tmp_path):
    marker = "/tmp/should-never-exist-checkpoint-test"
    if os.path.exists(marker):
        os.remove(marker)
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        decode_checkpoint_blob(pickle.dumps({"x": _Evil()}))
    assert not os.path.exists(marker)


def test_a_legacy_blob_of_numpy_objects_is_refused_not_loaded():
    with pytest.raises(pickle.UnpicklingError):
        decode_checkpoint_blob(pickle.dumps({"w": np.zeros(3)}))


def test_an_object_dtype_in_the_wire_format_is_refused():
    blob = pickle.dumps({"w": {"__ndarray__": True, "dtype": "O", "shape": [1], "data": b"12345678"}})
    with pytest.raises(TypeError, match="object"):
        decode_checkpoint_blob(blob)
