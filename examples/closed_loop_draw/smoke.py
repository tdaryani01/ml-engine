# examples/closed_loop_draw/smoke.py
"""
Closed-loop draw smoke: one trajectory, check finite non-zero grads.

Usage (repo root):
  .venv/bin/python -m examples.closed_loop_draw.smoke
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from examples.closed_loop_draw.assemble import assemble, load_config, make_target
from src.closed_loop import TokenInterleaver
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime


def main() -> int:
    bootstrap_im2col_gemm_runtime()
    cfg_path = Path(__file__).resolve().parent / "config_draw_smoke.yaml"
    cfg = load_config(cfg_path)
    app = assemble(cfg, seed=0)
    try:
        B = app.batch_size
        target = make_target(cfg, batch_size=B)
        command_ids = np.zeros(B, dtype=np.int64)

        w_adapt0 = app.adapter.W.copy()
        w_cnn0 = [w.copy() for w in app.cnn.weights]
        w_mhsa0 = [w.copy() for w in app.mhsa.weights]
        g0 = app.conditioning.embeddings.copy()

        result = app.trainer.rollout_train(
            command_ids=command_ids,
            target=target,
            max_steps=app.max_steps,
            lr=app.lr,
            apply_updates=True,
        )

        assert np.isfinite(result.total_loss), "loss not finite"
        assert result.total_loss > 0.0, "loss should be positive on empty→circle"
        assert len(result.actions) == app.max_steps
        assert result.seq_lens == [
            TokenInterleaver.seq_len(t) for t in range(1, app.max_steps + 1)
        ]

        dX = app.mhsa.get_last_dX()
        assert dX is not None and np.isfinite(dX).all(), "dX missing/non-finite"
        assert float(np.abs(dX).max()) > 0.0, "dX should be non-zero"
        assert float(np.abs(app.adapter.W - w_adapt0).max()) > 0.0, "adapter did not update"
        assert float(np.abs(app.conditioning.embeddings - g0).max()) > 0.0, "goal emb did not update"
        assert any(
            float(np.abs(a - b).max()) > 0.0 for a, b in zip(app.cnn.weights, w_cnn0)
        ), "CNN weights did not update"
        assert any(
            float(np.abs(a - b).max()) > 0.0 for a, b in zip(app.mhsa.weights, w_mhsa0)
        ), "MHSA weights did not update"

        print(
            f"closed_loop_draw smoke OK  loss={result.total_loss:.6f}  "
            f"steps={app.max_steps}  |dX|max={float(np.abs(dX).max()):.4e}"
        )
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
