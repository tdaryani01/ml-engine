# src/trainable_model.py
"""Shared trainable surface for MLP and CNN (session/engine-agnostic)."""
from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Callable, Optional

import numpy as np

from src.contract import ContractList

# Build a ContractList for a live model instance (CNN, future MHSA, …).
ContractFactory = Callable[[Any], ContractList]


class TrainableModel(ABC):
    """
    Common base for networks driven by TrainingSession / TrainingEngine.

    Contracts are attached via an optional factory or an explicit ContractList —
    models do not hard-code a single compile path.
    """

    def __init__(
        self,
        optimizer_instance: Any,
        *,
        lam_l1: float = 0.01,
        lam_l2: float = 0.01,
        p_dropout: float = 0.0,
        max_norm: float = 5.0,
        contract_list_enabled: bool = False,
        native_async_submit: bool = False,
        contract_factory: ContractFactory | None = None,
        **kwargs: Any,
    ) -> None:
        self.optimizer = optimizer_instance
        self.lam_l1 = lam_l1
        self.lam_l2 = lam_l2
        self.p_dropout = p_dropout
        self.max_norm = max_norm
        self.diagnostic_counter = 0
        self.contract_list_enabled = False
        self._contract_runtime = None
        self._contract: ContractList | None = None
        self._contract_factory = contract_factory
        # Swallow unknown kwargs so CNN/MLP constructors can forward freely.
        if kwargs:
            logging.debug(
                "[%s] Ignoring unused kwargs: %s",
                type(self).__name__,
                sorted(kwargs.keys()),
            )
        if contract_list_enabled:
            self.enable_contract_list(native_async_submit=native_async_submit)

    def set_contract_factory(self, factory: ContractFactory | None) -> None:
        """Replace the compile path used by enable_contract_list."""
        self._contract_factory = factory

    def enable_contract_list(
        self,
        *,
        native_async_submit: bool = False,
        contract: ContractList | None = None,
        contract_factory: ContractFactory | None = None,
    ) -> None:
        """Attach a contract runtime. Pass contract= or contract_factory= as needed."""
        if self._contract_runtime is not None:
            return
        if contract_factory is not None:
            self._contract_factory = contract_factory
        if contract is None:
            if self._contract_factory is None:
                raise RuntimeError(
                    f"{type(self).__name__}: enable_contract_list requires a "
                    "ContractList or contract_factory"
                )
            contract = self._contract_factory(self)
        from src.contract_runtime import ContractRuntime

        self.contract_list_enabled = True
        self._contract = contract
        self._contract_runtime = ContractRuntime(
            self, contract, native_async_submit=native_async_submit
        )
        logging.info(
            "[%s] Contract list enabled: %d ops (native_async_submit=%s)",
            type(self).__name__,
            contract.op_count,
            native_async_submit,
        )

    def add_training_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> str:
        """Manager handshake: OK if accepted, BUSY if single native slot occupied."""
        if self._contract_runtime is None:
            raise RuntimeError("Contract path not initialized")
        if self._contract_runtime.try_submit_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token
        ):
            return "OK"
        return "BUSY"

    def prepare_training_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> bool:
        if self._contract_runtime is None:
            raise RuntimeError("Contract path not initialized")
        return self._contract_runtime.prepare_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token
        )

    def contract_busy(self) -> bool:
        if self._contract_runtime is None:
            return False
        return self._contract_runtime.is_busy()

    def submit_contract_train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
    ) -> str:
        return self.add_training_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token
        )

    def try_reap_contract_train_step(
        self,
    ) -> tuple[float, list, list, int] | None:
        if self._contract_runtime is None:
            return None
        return self._contract_runtime.try_reap_step()

    def run_contract_train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        *,
        apply_adam: bool = False,
        step_token: int | None = None,
        tick_fn: Callable[[], None] | None = None,
    ) -> tuple[float, list, list, int]:
        if self._contract_runtime is None:
            raise RuntimeError("Contract path not initialized")
        return self._contract_runtime.run_step(
            X, y, lr, apply_adam=apply_adam, step_token=step_token, tick_fn=tick_fn
        )
