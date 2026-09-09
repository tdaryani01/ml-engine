# src/mhsa_network.py
"""Thin causal MHSA model shell — weights + contract; compute is native (stubs for now)."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config.constants import EngineBackend
from src.contract import mhsa_contract_factory
from src.trainable_model import TrainableModel
from utils.engine_ops import create_engine_context


class MHSANetwork(TrainableModel):
    """
    Causal multi-head self-attention + continuous action head.

    Python owns parameter storage and compiles the MHSA contract list.
    Forward/backward live in native ``mhsa_kernels`` (stubs until implemented).
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        action_dim: int,
        optimizer_instance: Any,
        ffn_mult: int = 4,
        backend: EngineBackend = EngineBackend.NATIVE,
        engine_ctx=None,
        lam_l1: float = 0.01,
        lam_l2: float = 0.01,
        max_norm: float = 5.0,
        contract_list_enabled: bool = True,
        native_async_submit: bool = False,
        **kwargs: Any,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        kwargs.pop("p_dropout", None)
        kwargs.pop("use_batch_norm", None)
        kwargs.pop("bn_momentum", None)
        kwargs.pop("bn_moments", None)
        super().__init__(
            optimizer_instance,
            lam_l1=lam_l1,
            lam_l2=lam_l2,
            p_dropout=0.0,
            max_norm=max_norm,
            contract_list_enabled=False,
            native_async_submit=native_async_submit,
            contract_factory=kwargs.pop("contract_factory", None) or mhsa_contract_factory,
            **kwargs,
        )
        self.engine_ctx = engine_ctx or create_engine_context(backend)
        self.backend = self.engine_ctx.backend
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads
        self.max_seq_len = int(max_seq_len)
        self.action_dim = int(action_dim)
        self.ffn_mult = int(ffn_mult)
        self.ffn_hidden = self.d_model * self.ffn_mult

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_parameters()

        # Param bank indices for contract / ledger (stable order).
        # 0 W_qkv, 1 W_o, 2 W_ff1, 3 W_ff2, 4 W_act
        self._param_names = ["W_qkv", "W_o", "W_ff1", "W_ff2", "W_act"]

        if contract_list_enabled:
            if self.backend != EngineBackend.NATIVE:
                raise ValueError("MHSA contract path requires NATIVE backend")
            self.enable_contract_list(native_async_submit=native_async_submit)

        logging.info(
            "[MHSA] d_model=%d heads=%d d_head=%d T_max=%d action_dim=%d ffn=%d "
            "(native block+action fwd/bwd)",
            self.d_model,
            self.num_heads,
            self.d_head,
            self.max_seq_len,
            self.action_dim,
            self.ffn_hidden,
        )

    def _xavier(self, rows: int, cols: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (rows + cols))
        return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float64)

    def _init_parameters(self) -> None:
        D = self.d_model
        Hff = self.ffn_hidden
        A = self.action_dim
        # Row-major [in, out] to match dense GEMM habits elsewhere.
        self.weights = [
            self._xavier(D, 3 * D),  # W_qkv
            self._xavier(D, D),  # W_o
            self._xavier(D, Hff),  # W_ff1
            self._xavier(Hff, D),  # W_ff2
            self._xavier(D, A),  # W_act
        ]
        self.biases = [
            np.zeros((1, 3 * D), dtype=np.float64),
            np.zeros((1, D), dtype=np.float64),
            np.zeros((1, Hff), dtype=np.float64),
            np.zeros((1, D), dtype=np.float64),
            np.zeros((1, A), dtype=np.float64),
        ]
        # LayerNorm scale/bias for attn block and FFN block (stored after dense biases).
        self.ln1_gamma = np.ones((1, D), dtype=np.float64)
        self.ln1_beta = np.zeros((1, D), dtype=np.float64)
        self.ln2_gamma = np.ones((1, D), dtype=np.float64)
        self.ln2_beta = np.zeros((1, D), dtype=np.float64)

    def predict(self, processed_data: np.ndarray) -> np.ndarray:
        """X (B,T,D) → continuous actions (B, action_dim) via native MHSA forward."""
        if self._contract_runtime is None:
            self.enable_contract_list()
        return self._contract_runtime.run_mhsa_forward(processed_data)

    def calculate_raw_cost(self, output: np.ndarray, y: np.ndarray) -> float:
        """MSE on continuous actions."""
        return float(np.mean((output - y) ** 2))

    def compute_total_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        return self.calculate_raw_cost(output, y)

    def _apply_grads(
        self,
        grad_weights,
        grad_biases,
        m_samples,
        lr,
        grad_gammas=None,
        grad_betas=None,
    ) -> None:
        self.optimizer.update(
            self.weights,
            self.biases,
            grad_weights,
            grad_biases,
            m_samples,
            self.lam_l2,
            lr,
            gammas=[self.ln1_gamma, self.ln2_gamma],
            betas=[self.ln1_beta, self.ln2_beta],
            grad_gammas=grad_gammas,
            grad_betas=grad_betas,
        )
