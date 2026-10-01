# examples/closed_loop_draw/actor.py
"""DrawActor — the drawing-domain implementation of the generic Actor protocol.

Phase 1: this adapter absorbs every hardcoded sub-module that used to live in
``ClosedLoopTrainer`` (encoder, V→D adapter, A→D action embed, conditioning
bank, MHSA, token interleaver). The trainer is NOT yet wired to it; the
mathematical behaviour is a verbatim move of the existing rollout logic.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.closed_loop.adapter import LinearAdapter
from src.closed_loop.conditioning import ConditioningBank
from src.closed_loop.interleaver import TokenInterleaver


class DrawActor:
    """Actor protocol implementation for the CNN→MHSA→SoftCanvas draw stack."""

    def __init__(
        self,
        *,
        mhsa: Any,
        encoder: Any,
        adapter: LinearAdapter,
        action_embed: LinearAdapter,
        conditioning: ConditioningBank,
        checkpoint_fn: Callable[[int, float], bytes] | None = None,
    ) -> None:
        if adapter.d_out != mhsa.d_model:
            raise ValueError("adapter.d_out must equal mhsa.d_model")
        if action_embed.d_in != mhsa.action_dim or action_embed.d_out != mhsa.d_model:
            raise ValueError("action_embed must map action_dim -> d_model")
        if conditioning.d_model != mhsa.d_model:
            raise ValueError("conditioning.d_model must equal mhsa.d_model")
        self.mhsa = mhsa
        self.encoder = encoder
        self.adapter = adapter
        self.action_embed = action_embed
        self.conditioning = conditioning
        self.interleaver = TokenInterleaver(mhsa.d_model)
        self.action_dim = int(mhsa.action_dim)
        self.max_seq_len = int(mhsa.max_seq_len)
        # Bound by reset(); the interleaver grammar is seq_len(t) = 2 * t.
        self._max_steps = 0
        self._B = 0
        # Optional blob writer (wired by assemble.py; keeps the protocol free of
        # app-layer config).
        self.checkpoint_fn: Callable[[int, float], bytes] | None = checkpoint_fn
        self._clear()

    # -- cache lifecycle ---------------------------------------------------

    def _clear(self) -> None:
        self._B = 0
        self._command_ids: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._states_S: list[np.ndarray] = []
        self._action_embs: list[np.ndarray] = []
        self._Xs: list[np.ndarray] = []
        self._V_list: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._dA_emb_pending: list[np.ndarray | None] = []
        self._dG_acc: np.ndarray | None = None
        self._mhsa_acc: dict[str, Any] | None = None

    # -- Actor protocol ----------------------------------------------------

    def zero_grad(self) -> None:
        """Formerly ``ClosedLoopTrainer._zero_all()``."""
        self.encoder.zero_grad()
        self.adapter.zero_grad()
        self.action_embed.zero_grad()
        self.conditioning.zero_grad()

    @staticmethod
    def _command_ids_from(goal: Any) -> np.ndarray:
        """Accept a composite DrawGoal or a raw command-id array."""
        ids = getattr(goal, "command_ids", None)
        if ids is None:
            ids = goal
        return np.asarray(ids, dtype=np.int64).reshape(-1)

    def reset(self, batch_size: int, goal: Any, max_steps: int) -> None:
        """Bind the goal payload + explicit trajectory length."""
        command_ids = self._command_ids_from(goal)
        self._clear()
        self._B = int(batch_size)
        self._max_steps = max(1, int(max_steps))
        self._command_ids = command_ids
        self._goal = self.conditioning.embed(command_ids)
        self._dA_emb_pending = [None] * max(0, self._max_steps - 1)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Encoder → adapter → interleave → MHSA predict (one stroke)."""
        V = np.ascontiguousarray(self.encoder.encode(obs), dtype=np.float32)
        self._V_list.append(V)
        S = self.adapter.forward(V)
        self._states_S.append(np.array(S, copy=True))

        X = self.interleaver.build(self._goal, self._states_S, self._action_embs)
        self._Xs.append(X)
        A = np.ascontiguousarray(self.mhsa.predict(X), dtype=np.float32)
        self._actions.append(A)

        if len(self._states_S) < self._max_steps:
            self._action_embs.append(self.action_embed.forward(A))
        return A

    def accumulate_grads(self, t: int, d_action: np.ndarray) -> None:
        """Reverse step: ∂L/∂action_t → MHSA / adapters / encoder accumulators."""
        ti = t - 1
        if self._mhsa_acc is None:
            self._mhsa_acc = self._new_mhsa_grad_acc()
            self._dG_acc = np.zeros_like(self._goal, dtype=np.float32)
        dA = np.asarray(d_action, dtype=np.float32)

        if t < self._max_steps and self._dA_emb_pending[ti] is not None:
            self.action_embed.forward(self._actions[ti])
            dA = dA + self.action_embed.backward(self._dA_emb_pending[ti])

        dw, db, dX = self.mhsa.backward_from_dA(
            self._Xs[ti], dA, apply_adam=False, lr=0.0
        )
        self._accumulate_mhsa(self._mhsa_acc, dw, db, m_samples=self._B)

        dG, d_states, d_act_embs = self.interleaver.split_dX(dX, t)
        self._dG_acc += dG

        for k, dSk in enumerate(d_states, start=1):
            self.adapter.forward(self._V_list[k - 1])
            dV = self.adapter.backward(dSk)
            if hasattr(self.encoder, "backward_at"):
                self.encoder.backward_at(k - 1, dV)
            elif k == t:
                raise RuntimeError(
                    "DrawActor requires a multi-cache encoder exposing "
                    "backward_at(step_idx, dV)"
                )

        for k, dAk in enumerate(d_act_embs, start=1):
            idx = k - 1
            if self._dA_emb_pending[idx] is None:
                self._dA_emb_pending[idx] = np.array(dAk, copy=True)
            else:
                self._dA_emb_pending[idx] += dAk

    def backward_done(self) -> None:
        """Flush the goal-embedding gradient into the conditioning bank."""
        if self._mhsa_acc is None:
            self._mhsa_acc = self._new_mhsa_grad_acc()
        if self._dG_acc is not None:
            self.conditioning.backward(self._dG_acc, self._command_ids)

    def apply_updates(self, lr: float) -> None:
        """One optimizer step over every actor parameter."""
        self.conditioning.apply_pending(lr)
        self.adapter.apply_pending(lr)
        self.action_embed.apply_pending(lr)
        self.encoder.apply_pending(lr)
        acc = self._mhsa_acc or self._new_mhsa_grad_acc()
        self.mhsa._apply_grads(
            acc["dw"],
            acc["db"],
            acc["m_samples"] or self._B,
            lr,
            grad_gammas=acc["dln_g"],
            grad_betas=acc["dln_b"],
        )

    def seq_len(self, t: int) -> int:
        """Policy token-grammar length when predicting the action at step t."""
        return int(self.interleaver.seq_len(t))

    def checkpoint(self, version: int, val_loss: float) -> bytes:
        """Serialize actor weights + config (delegates to the app-layer writer)."""
        if self.checkpoint_fn is None:
            raise NotImplementedError(
                "DrawActor.checkpoint requires checkpoint_fn (wired by assemble.py)"
            )
        return self.checkpoint_fn(int(version), float(val_loss))

    # -- internal MHSA grad plumbing (moved verbatim from the trainer) ------

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


__all__ = ["DrawActor"]
