# src/mhsa_network.py
"""Thin causal MHSA model shell — stacked Pre-LN blocks + action head (native)."""
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
    Stacked causal Pre-LN MHSA+FFN blocks + continuous action head.

    Python owns parameter storage and compiles the MHSA contract list.
    Forward/backward live in native ``mhsa_kernels``.
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
        num_layers: int = 1,
        use_pos_encoding: bool = True,
        use_input_proj: bool = False,
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
        num_layers = int(num_layers)
        if num_layers < 1 or num_layers > 8:
            raise ValueError(f"num_layers must be in 1..8, got {num_layers}")
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
        self.num_layers = num_layers
        self.use_pos_encoding = bool(use_pos_encoding)
        self.use_input_proj = bool(use_input_proj)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.ln1_gamma: list[np.ndarray] = []
        self.ln1_beta: list[np.ndarray] = []
        self.ln2_gamma: list[np.ndarray] = []
        self.ln2_beta: list[np.ndarray] = []
        self.pos_embed: np.ndarray | None = None
        self._ms_pos: np.ndarray | None = None
        self._vs_pos: np.ndarray | None = None
        self.W_in: np.ndarray | None = None
        self.b_in: np.ndarray | None = None
        self._ms_W_in: np.ndarray | None = None
        self._vs_W_in: np.ndarray | None = None
        self._ms_b_in: np.ndarray | None = None
        self._vs_b_in: np.ndarray | None = None
        self._init_parameters()

        # Per layer: W_qkv, W_o, W_ff1, W_ff2; then W_act.
        self._param_names: list[str] = []
        for li in range(self.num_layers):
            self._param_names.extend(
                [f"L{li}.W_qkv", f"L{li}.W_o", f"L{li}.W_ff1", f"L{li}.W_ff2"]
            )
        self._param_names.append("W_act")

        if contract_list_enabled:
            if self.backend != EngineBackend.NATIVE:
                raise ValueError("MHSA contract path requires NATIVE backend")
            self.enable_contract_list(native_async_submit=native_async_submit)

        logging.info(
            "[MHSA] layers=%d d_model=%d heads=%d d_head=%d T_max=%d action_dim=%d "
            "ffn=%d pos=%s in_proj=%s",
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.d_head,
            self.max_seq_len,
            self.action_dim,
            self.ffn_hidden,
            self.use_pos_encoding,
            self.use_input_proj,
        )

    def _xavier(self, rows: int, cols: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (rows + cols))
        return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)

    def _init_parameters(self) -> None:
        D = self.d_model
        Hff = self.ffn_hidden
        A = self.action_dim
        self.weights = []
        self.biases = []
        self.ln1_gamma = []
        self.ln1_beta = []
        self.ln2_gamma = []
        self.ln2_beta = []
        for _ in range(self.num_layers):
            self.weights.extend(
                [
                    self._xavier(D, 3 * D),
                    self._xavier(D, D),
                    self._xavier(D, Hff),
                    self._xavier(Hff, D),
                ]
            )
            self.biases.extend(
                [
                    np.zeros((1, 3 * D), dtype=np.float32),
                    np.zeros((1, D), dtype=np.float32),
                    np.zeros((1, Hff), dtype=np.float32),
                    np.zeros((1, D), dtype=np.float32),
                ]
            )
            self.ln1_gamma.append(np.ones((1, D), dtype=np.float32))
            self.ln1_beta.append(np.zeros((1, D), dtype=np.float32))
            self.ln2_gamma.append(np.ones((1, D), dtype=np.float32))
            self.ln2_beta.append(np.zeros((1, D), dtype=np.float32))
        self.weights.append(self._xavier(D, A))
        self.biases.append(np.zeros((1, A), dtype=np.float32))
        if self.use_pos_encoding:
            # Small init so early steps stay near token features.
            self.pos_embed = (
                np.random.randn(self.max_seq_len, D).astype(np.float32) * 0.02
            )
            self._ms_pos = np.zeros_like(self.pos_embed)
            self._vs_pos = np.zeros_like(self.pos_embed)
        else:
            self.pos_embed = None
            self._ms_pos = None
            self._vs_pos = None
        if self.use_input_proj:
            self.W_in = self._xavier(D, D)
            self.b_in = np.zeros((1, D), dtype=np.float32)
            self._ms_W_in = np.zeros_like(self.W_in)
            self._vs_W_in = np.zeros_like(self.W_in)
            self._ms_b_in = np.zeros_like(self.b_in)
            self._vs_b_in = np.zeros_like(self.b_in)
        else:
            self.W_in = None
            self.b_in = None
            self._ms_W_in = None
            self._vs_W_in = None
            self._ms_b_in = None
            self._vs_b_in = None

    def ensure_adam_moments(self) -> None:
        """Allocate Adam m/v on live f32 banks (weights, biases, LN, pos, W_in)."""
        opt = self.optimizer
        if not getattr(opt, "_setup_done", False):
            gammas, betas = self._ln_param_lists()
            opt.setup(self.weights, self.biases, gammas, betas)
        if self.use_pos_encoding and self.pos_embed is not None:
            if self._ms_pos is None or self._ms_pos.shape != self.pos_embed.shape:
                self._ms_pos = np.zeros_like(self.pos_embed)
                self._vs_pos = np.zeros_like(self.pos_embed)
        if self.use_input_proj and self.W_in is not None:
            if self._ms_W_in is None or self._ms_W_in.shape != self.W_in.shape:
                self._ms_W_in = np.zeros_like(self.W_in)
                self._vs_W_in = np.zeros_like(self.W_in)
            if self.b_in is not None and (
                self._ms_b_in is None or self._ms_b_in.shape != self.b_in.shape
            ):
                self._ms_b_in = np.zeros_like(self.b_in)
                self._vs_b_in = np.zeros_like(self.b_in)

    def predict(self, processed_data: np.ndarray) -> np.ndarray:
        """X (B,T,D) → continuous actions (B, action_dim) via native MHSA forward."""
        if self._contract_runtime is None:
            self.enable_contract_list()
        return self._contract_runtime.run_mhsa_forward(processed_data)

    def calculate_raw_cost(self, output: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((output - y) ** 2))

    def compute_total_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        return self.calculate_raw_cost(output, y)

    def _ln_param_lists(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        gammas: list[np.ndarray] = []
        betas: list[np.ndarray] = []
        for li in range(self.num_layers):
            gammas.append(self.ln1_gamma[li])
            gammas.append(self.ln2_gamma[li])
            betas.append(self.ln1_beta[li])
            betas.append(self.ln2_beta[li])
        return gammas, betas

    def _apply_grads(
        self,
        grad_weights,
        grad_biases,
        m_samples,
        lr,
        grad_gammas=None,
        grad_betas=None,
    ) -> None:
        gammas, betas = self._ln_param_lists()
        self.optimizer.update(
            self.weights,
            self.biases,
            grad_weights,
            grad_biases,
            m_samples,
            self.lam_l2,
            lr,
            gammas=gammas,
            betas=betas,
            grad_gammas=grad_gammas,
            grad_betas=grad_betas,
        )
