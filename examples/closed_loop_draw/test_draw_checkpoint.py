# Draw checkpoint wire encode (no native .so required).
from __future__ import annotations

import numpy as np

from examples.closed_loop_draw.draw_checkpoint import knobs_from_cfg
from src.manager_heartbeat import decode_checkpoint_blob, encode_checkpoint_blob


def test_encode_decode_draw_style_blob():
    body = {
        "kind": "draw_student_v1",
        "version": 1,
        "weights": {"cnn_w": [np.arange(12, dtype=np.float32).reshape(3, 4)]},
        "config": {"optimization": {"learning_rate": 0.001}},
        "knobs": knobs_from_cfg(
            {
                "optimization": {"learning_rate": 0.001, "seed": 0},
                "closed_loop": {"max_steps": 10, "train_patience": 40},
            },
            lr=0.001,
        ),
    }
    blob = encode_checkpoint_blob(body)
    out = decode_checkpoint_blob(blob)
    assert out["kind"] == "draw_student_v1"
    assert out["weights"]["cnn_w"][0].shape == (3, 4)
    assert out["knobs"]["train_patience"] == 40
