# src/closed_loop/trainer.py
"""Python-orchestrated closed-loop BPTT over pluggable encoder / env / loss."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.closed_loop.adapter import LinearAdapter
from src.closed_loop.conditioning import ConditioningBank
from src.closed_loop.interleaver import TokenInterleaver
from src.closed_loop.protocols import Environment, TrajectoryLoss, UpstreamEncoder
from src.mhsa_network import MHSANetwork


@dataclass
class RolloutResult:
    total_loss: float
    actions: list[np.ndarray]
    seq_lens: list[int]
    extras: dict[str, Any] = field(default_factory=dict)


class ClosedLoopTrainer:
    """
    Encode → interleave → MHSA action → env step → accumulate loss; reverse BPTT.

    Expects ``env`` to expose a step stack after the forward rollout:
      - ``backward_step(t, d_after) -> (dA, d_before)`` for soft render grads
      - ``obs_after(t)`` for loss ∂/∂canvas at step t

    MHSA Adam is deferred: reverse steps use ``apply_adam=False``, grads are
    summed, then applied once via ``MHSANetwork._apply_grads``.
    """

    def __init__(
        self,
        *,
        mhsa: MHSANetwork,
        encoder: UpstreamEncoder,
        adapter: LinearAdapter,
        action_embed: LinearAdapter,
        conditioning: ConditioningBank,
        env: Environment,
        loss_fn: TrajectoryLoss,
        interleaver: TokenInterleaver | None = None,
    ) -> None:
        if adapter.d_out != mhsa.d_model:
            raise ValueError("adapter.d_out must equal mhsa.d_model")
        if action_embed.d_in != mhsa.action_dim or action_embed.d_out != mhsa.d_model:
            raise ValueError("action_embed must map action_dim → d_model")
        if conditioning.d_model != mhsa.d_model:
            raise ValueError("conditioning.d_model must equal mhsa.d_model")
        self.mhsa = mhsa
        self.encoder = encoder
        self.adapter = adapter
        self.action_embed = action_embed
        self.conditioning = conditioning
        self.env = env
        self.loss_fn = loss_fn
        self.interleaver = interleaver or TokenInterleaver(mhsa.d_model)

    def _zero_all(self) -> None:
        self.encoder.zero_grad()
        self.adapter.zero_grad()
        self.action_embed.zero_grad()
        self.conditioning.zero_grad()

    def _new_mhsa_grad_acc(self) -> dict[str, Any]:
        m = self.mhsa
        n_ln = int(m.num_layers) * 2
        return {
            "dw": [np.zeros_like(w, dtype=np.float64) for w in m.weights],
            "db": [np.zeros_like(b, dtype=np.float64) for b in m.biases],
            "dln_g": [np.zeros_like(m.ln1_gamma[0], dtype=np.float64) for _ in range(n_ln)],
            "dln_b": [np.zeros_like(m.ln1_beta[0], dtype=np.float64) for _ in range(n_ln)],
            "m_samples": 0,
        }

    def _accumulate_mhsa(
        self, acc: dict[str, Any], dw: list, db: list, m_samples: int
    ) -> None:
        for i, g in enumerate(dw):
            acc["dw"][i] += np.asarray(g, dtype=np.float64)
        for i, g in enumerate(db):
            acc["db"][i] += np.asarray(g, dtype=np.float64).reshape(acc["db"][i].shape)
        acc["m_samples"] = max(int(acc["m_samples"]), int(m_samples))
        rt = self.mhsa._contract_runtime
        if rt is None or not rt._mhsa_dln_f32:
            return
        gi = 0
        for layer_dln in rt._mhsa_dln_f32:
            for key_g, key_b in (("ln1_g", "ln1_b"), ("ln2_g", "ln2_b")):
                acc["dln_g"][gi] += layer_dln[key_g].astype(np.float64).reshape(
                    acc["dln_g"][gi].shape
                )
                acc["dln_b"][gi] += layer_dln[key_b].astype(np.float64).reshape(
                    acc["dln_b"][gi].shape
                )
                gi += 1

    def rollout_train(
        self,
        *,
        command_ids: np.ndarray,
        target: np.ndarray,
        max_steps: int,
        lr: float,
        apply_updates: bool = True,
    ) -> RolloutResult:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        need_T = TokenInterleaver.seq_len(max_steps)
        if need_T > int(self.mhsa.max_seq_len):
            raise ValueError(
                f"seq_len({max_steps})={need_T} exceeds mhsa.max_seq_len="
                f"{self.mhsa.max_seq_len}"
            )

        command_ids = np.asarray(command_ids, dtype=np.int64).reshape(-1)
        B = int(command_ids.shape[0])
        self._zero_all()

        goal = self.conditioning.embed(command_ids)
        obs = self.env.reset(B)

        states_S: list[np.ndarray] = []
        action_embs: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        Xs: list[np.ndarray] = []
        V_list: list[np.ndarray] = []
        A_list: list[np.ndarray] = []
        step_losses: list[float] = []

        total_loss = 0.0
        for t in range(1, max_steps + 1):
            V = np.ascontiguousarray(self.encoder.encode(obs), dtype=np.float32)
            V_list.append(V)
            S = self.adapter.forward(V)
            states_S.append(np.array(S, copy=True))

            X = self.interleaver.build(goal, states_S, action_embs)
            Xs.append(X)
            A = np.ascontiguousarray(self.mhsa.predict(X), dtype=np.float32)
            actions.append(A)
            A_list.append(A)

            obs = self.env.step(A)
            if hasattr(self.env, "obs_after"):
                loss_obs = self.env.obs_after(t - 1)  # type: ignore[attr-defined]
            else:
                loss_obs = obs
            sl = float(self.loss_fn.step_loss(loss_obs, target, t - 1))
            step_losses.append(sl)
            total_loss += sl

            if t < max_steps:
                action_embs.append(self.action_embed.forward(A))

        mhsa_acc = self._new_mhsa_grad_acc()
        dG_acc = np.zeros_like(goal, dtype=np.float32)
        dA_emb_pending: list[np.ndarray | None] = [None] * max(0, max_steps - 1)

        # Canvas carry: ∂L/∂C_t from later render steps (into this step's "after").
        if hasattr(self.env, "obs_after"):
            d_carry = np.zeros_like(self.env.obs_after(max_steps - 1), dtype=np.float32)  # type: ignore[attr-defined]
        else:
            d_carry = None

        for t in range(max_steps, 0, -1):
            ti = t - 1
            if hasattr(self.env, "obs_after"):
                after = self.env.obs_after(ti)  # type: ignore[attr-defined]
                d_loss = self.loss_fn.step_obs_grad(after, target, ti)
                d_after = d_loss if d_carry is None else (d_loss + d_carry)
                if not hasattr(self.env, "backward_step"):
                    raise RuntimeError(
                        "Environment must implement backward_step(t, d_after) "
                        "for closed-loop BPTT"
                    )
                dA, d_before = self.env.backward_step(ti, d_after)  # type: ignore[attr-defined]
                d_carry = d_before
            else:
                dA = self.env.action_grad(
                    self.loss_fn.step_obs_grad(obs, target, ti)
                )

            if t < max_steps and dA_emb_pending[ti] is not None:
                self.action_embed.forward(A_list[ti])
                dA = dA + self.action_embed.backward(dA_emb_pending[ti])

            dw, db, dX = self.mhsa.backward_from_dA(
                Xs[ti], dA, apply_adam=False, lr=0.0
            )
            self._accumulate_mhsa(mhsa_acc, dw, db, m_samples=B)

            dG, d_states, d_act_embs = self.interleaver.split_dX(dX, t)
            dG_acc += dG

            for k, dSk in enumerate(d_states, start=1):
                self.adapter.forward(V_list[k - 1])
                dV = self.adapter.backward(dSk)
                if hasattr(self.encoder, "backward_at"):
                    self.encoder.backward_at(k - 1, dV)  # type: ignore[attr-defined]
                elif k == t:
                    # Single-cache encoders: re-encode obs_before then backward.
                    if hasattr(self.env, "obs_before"):
                        _ = self.encoder.encode(self.env.obs_before(k - 1))  # type: ignore[attr-defined]
                    self.encoder.backward(dV)

            for k, dAk in enumerate(d_act_embs, start=1):
                idx = k - 1
                if dA_emb_pending[idx] is None:
                    dA_emb_pending[idx] = np.array(dAk, copy=True)
                else:
                    dA_emb_pending[idx] += dAk

        self.conditioning.backward(dG_acc, command_ids)

        if apply_updates:
            self.conditioning.apply_pending(lr)
            self.adapter.apply_pending(lr)
            self.action_embed.apply_pending(lr)
            self.encoder.apply_pending(lr)
            self.mhsa._apply_grads(
                mhsa_acc["dw"],
                mhsa_acc["db"],
                mhsa_acc["m_samples"] or B,
                lr,
                grad_gammas=mhsa_acc["dln_g"],
                grad_betas=mhsa_acc["dln_b"],
            )

        return RolloutResult(
            total_loss=float(total_loss),
            actions=actions,
            seq_lens=[int(x.shape[1]) for x in Xs],
            extras={"step_losses": step_losses},
        )
