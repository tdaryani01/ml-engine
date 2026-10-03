
# examples/closed_loop_draw/test_model_sanity_check.py
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from examples.closed_loop_draw.goal import DrawGoal
from examples.closed_loop_draw.assemble import (
    assemble,
    load_config,
    make_target,
)
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    force=True,
)


def _snapshot_weights(weights):
    return [np.array(w, copy=True) for w in weights]


def _weight_delta(wa, wb):
    total = 0.0
    for a, b in zip(wa, wb):
        total += float(np.sum((a - b) ** 2))
    return np.sqrt(total).item()


def test_closed_loop_single_batch_overfit():
    bootstrap_im2col_gemm_runtime()
    cfg_path = Path(__file__).resolve().parent / 'config_draw_train.yaml'
    cfg = load_config(str(cfg_path))
    seed = int(cfg.get('optimization', {}).get('seed', 0))
    app = assemble(cfg, seed=seed)
    try:
        B = app.batch_size
        lr = app.lr
        command_id = 0
        command_ids = np.full(B, command_id, dtype=np.int64)
        target = make_target(cfg, batch_size=B)
        print('=' * 72)
        print('  Closed-Loop Draw - Single-Batch Overfit Sanity Check')
        print(f'  batch_size={B}  max_steps={app.max_steps}  lr={lr}')
        print(f'  command_id={command_id}  target_shape={target.shape}')
        print('=' * 72)
        np.random.seed(seed)
        original_update = app.mhsa.optimizer.update
        captured_grad_norms = []

        def _capturing_update(
            weights, biases, grad_weights, grad_biases,
            m_samples, lam_l2, active_lr,
            gammas=None, betas=None,
            grad_gammas=None, grad_betas=None,
        ):
            gn = 0.0
            for gw in grad_weights:
                if gw is not None:
                    gn += float(np.sum(gw.astype(np.float64) ** 2))
            captured_grad_norms.append(np.sqrt(gn).item())
            return original_update(
                weights, biases, grad_weights, grad_biases,
                m_samples, lam_l2, active_lr,
                gammas=gammas, betas=betas,
                grad_gammas=grad_gammas, grad_betas=grad_betas,
            )

        app.mhsa.optimizer.update = _capturing_update
        app.mhsa.ensure_adam_moments()
        opt = app.mhsa.optimizer
        canvas_pre = app.trainer.rollout_eval(
            goal=DrawGoal.render(command_ids)
        ).extras["frames"][-1]
        W0 = _snapshot_weights(app.mhsa.weights)
        n_iters = 100
        log_every = 10
        losses = []
        weight_deltas = []
        grad_norms = []
        adam_m_means = []
        adam_v_means = []
        print(f'\n  Running {n_iters} training iterations...\n')
        for i in range(1, n_iters + 1):
            W_before = _snapshot_weights(app.mhsa.weights)
            result = app.trainer.rollout_train(
                goal=DrawGoal(command_ids=command_ids, target=target),
                lr=lr,
                apply_updates=True,
            )
            loss = float(result.total_loss)
            losses.append(loss)
            W_after = _snapshot_weights(app.mhsa.weights)
            delta_norm = _weight_delta(W_after, W_before)
            weight_deltas.append(delta_norm)
            gn = captured_grad_norms[-1] if captured_grad_norms else 0.0
            grad_norms.append(gn)
            ms_means = [float(np.mean(m)) for m in opt.ms_w]
            vs_means = [float(np.mean(v)) for v in opt.vs_w]
            m_mean = float(np.mean(ms_means))
            v_mean = float(np.mean(vs_means))
            adam_m_means.append(m_mean)
            adam_v_means.append(v_mean)
            if i % log_every == 0 or i == 1:
                sl = result.extras.get('step_losses', [])
                print(f'  step {i:3d}/{n_iters}  '
                      f'L_t={loss:.8f}  '
                      f'||dL/dw||_2={gn:.6f}  '
                      f'||dW||_2={delta_norm:.6e}  '
                      f'mean(m)={m_mean:.6f}  '
                      f'mean(v)={v_mean:.6e}  '
                      f'step_losses={sl}')
        canvas_post = app.trainer.rollout_eval(
            goal=DrawGoal.render(command_ids)
        ).extras["frames"][-1]
        print(f'\n  {chr(9472) * 68}')
        print(f'  Initial loss:            {losses[0]:.8f}')
        print(f'  Final loss (step 100):   {losses[-1]:.8f}')
        r = losses[-1] / losses[0]
        print(f'  Loss ratio (final/init): {r:.8f}')
        tot_w = _weight_delta(W_after, W0)
        print(f'  ||W_100 - W_0||_2:        {tot_w:.8f}')
        print(f'  Final ||dL/dw||_2:        {grad_norms[-1]:.6f}')
        print(f'  Final mean(m):           {adam_m_means[-1]:.6f}')
        print(f'  Final mean(v):           {adam_v_means[-1]:.6e}')
        print(f'\n  -- Canvas Prediction Statistics --')
        for lbl, cv in [('step 0 (pre-train)', canvas_pre),
                        ('step 100 (post-train)', canvas_post)]:
            print(f'  {lbl:30s}  min={cv.min():.6f}  max={cv.max():.6f}  '
                  f'mean={cv.mean():.6f}  std={cv.std():.6f}')
        assert r < 0.2, f'Loss ratio {r:.6f} not < 0.2  init={losses[0]:.6f} final={losses[-1]:.6f}'
        print('\n  [OK] Loss dropped by >=80%')
        assert tot_w > 0.0, 'Weights did not change!'
        print('  [OK] Weights changed in memory')
        assert any(np.any(m != 0.0) for m in opt.ms_w), 'ms_w all zeros'
        assert any(np.any(v != 0.0) for v in opt.vs_w), 'vs_w all zeros'
        print('  [OK] Adam moments non-zero')
        assert not np.allclose(canvas_post, canvas_pre, atol=1e-6), 'Canvas unchanged'
        print('  [OK] Canvas changed step 0 -> 100')
        assert canvas_post.max() > canvas_post.min(), 'Canvas uniform'
        print('  [OK] Canvas non-uniform')
        print(f'\n  {chr(9552) * 68}')
        print('  Model is actively learning.  Overfit test PASSED.')
        print(f'  {chr(9552) * 68}')
    finally:
        app.close()


if __name__ == '__main__':
    test_closed_loop_single_batch_overfit()
    print('All checks passed.')
