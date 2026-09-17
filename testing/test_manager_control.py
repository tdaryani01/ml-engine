# testing/test_manager_control.py
"""TM control-plane correctness on the engine: boot idle, Start, restore weights."""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import IngestionMode, ModelType
from src.ledger import LedgerConfig, TrainingLedger, capture_model_checkpoint
from src.ledger_store import FileLedgerStore
from src.manager_heartbeat import ManagerCommand
from src.models import MultiClassNetwork
from src.optimizers import AdamOptimizer
from src.training_engine import TrainingEngine
from src.training_session import SessionStatus


@dataclass
class _StubHB:
    """In-process stand-in for ManagerHeartbeat (no HTTP)."""

    idle_sleep_s: float = 0.02
    _active_checkpoint: dict[str, Any] | None = None
    _commands: list[ManagerCommand] = field(default_factory=list)
    acks: list[tuple[str, bool, str | None]] = field(default_factory=list)
    blobs: dict[str, bytes] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._cfg = type("Cfg", (), {"idle_sleep_s": self.idle_sleep_s})()

    @property
    def active_checkpoint(self) -> dict[str, Any] | None:
        return None if self._active_checkpoint is None else dict(self._active_checkpoint)

    def set_active_checkpoint(self, active: dict[str, Any] | None) -> None:
        self._active_checkpoint = None if active is None else dict(active)

    def set_metrics(self, metrics: dict[str, Any] | None) -> None:
        if metrics:
            self.metrics.update(dict(metrics))

    def maybe_ping(self, *, force: bool = False) -> bool:
        return False

    def poll_commands(self) -> list[ManagerCommand]:
        out = list(self._commands)
        self._commands.clear()
        return out

    def mark_command_seen(self, command_id: str) -> None:
        return None

    def queue_ack(
        self, command_id: str, *, ok: bool, detail: str | None = None
    ) -> None:
        self.acks.append((command_id, ok, detail))

    def fetch_blob(self, blob_key: str) -> bytes | None:
        return self.blobs.get(blob_key)

    def push_command(self, action: str, payload: dict[str, Any] | None = None) -> str:
        cid = f"cmd-{action}-{len(self.acks)}-{len(self._commands)}"
        self._commands.append(
            ManagerCommand(
                id=cid,
                action=action,
                payload=dict(payload or {}),
                created_at=time.time(),
            )
        )
        return cid


class _Prov:
    def __init__(self, X, y, Xv, yv, batches: int = 2):
        self.X, self.y = X, y
        self.Xv, self.yv = Xv, yv
        self.batches = batches
        self._i = 0
        self.batch_size = X.shape[0]

    def reset_epoch(self):
        self._i = 0

    def has_more_batches(self):
        return self._i < self.batches

    def next_batch(self):
        self._i += 1
        return self.X, self.y

    def normalize(self, X):
        return X

    def get_validation_set(self):
        return self.Xv, self.yv


def _bundle(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((8, 4)).astype(np.float64)
    y = np.eye(3)[rng.integers(0, 3, size=8)]
    Xv = rng.standard_normal((4, 4)).astype(np.float64)
    yv = np.eye(3)[rng.integers(0, 3, size=4)]
    model = MultiClassNetwork(
        layer_sizes=[4, 8, 3],
        optimizer_instance=AdamOptimizer(lr=0.05),
        use_batch_norm=False,
        lam_l1=0.0,
        lam_l2=0.0,
    )
    return model, _Prov(X, y, Xv, yv)


def _run_brief(engine: TrainingEngine, seconds: float = 0.15) -> None:
    def _go() -> None:
        engine.run()

    t = threading.Thread(target=_go, name="engine-run", daemon=True)
    t.start()
    time.sleep(seconds)
    engine.request_stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "engine.run did not exit after request_stop"


def test_boot_stays_idle_despite_sticky_desired_running():
    """Boot without Start must not train (no auto-authorize)."""
    hb = _StubHB()
    model, prov = _bundle()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=100, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="boot-idle",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=2,
        )
        _run_brief(engine, 0.2)
        assert engine._run_authorized is False
        assert ledger.version == 0
        assert len(engine.sessions) == 1
        assert engine.sessions[0].status == SessionStatus.ACTIVE
        assert not engine.sessions[0]._fit_ready
        engine.close()
    print("[PASSED] boot stays idle without start authorize")


def test_start_command_authorizes_training():
    hb = _StubHB()
    model, prov = _bundle(seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=100, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="after-start",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=1,
        )

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.05)
        assert ledger.version == 0
        hb.push_command("start")
        # Wake from idle sleep sooner.
        deadline = time.time() + 3.0
        while time.time() < deadline and ledger.version == 0:
            time.sleep(0.05)
        engine.request_stop()
        t.join(timeout=3.0)
        assert engine._run_authorized is True
        assert ledger.version > 0
        assert any(a[0].startswith("cmd-start") and a[1] for a in hb.acks)
        engine.close()
    print("[PASSED] start command authorizes training")


def test_restore_applies_weights_and_version_while_idle():
    """Restore must hydrate weights + ledger version without authorizing training."""
    hb = _StubHB()
    model, prov = _bundle(seed=13)
    # Capture "beginning" (v0) state, then corrupt weights.
    beginning = capture_model_checkpoint(model, version=0, val_loss=None)
    w0 = [np.copy(w) for w in beginning["weights"]]
    for w in model.weights:
        w[...] = 9.0
    assert not np.allclose(model.weights[0], w0[0])

    blob_key = "ckpts/test/v0.bin"
    hb.blobs[blob_key] = pickle.dumps(beginning)

    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        # Simulate a progressed local ledger that restore should rewind.
        ledger.version = 12
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=100, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="restore-live",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=1,
        )
        # Drop session (park after fit) — restore must still work via held model.
        engine.drop_session("restore-live")
        assert engine.sessions == []
        assert engine._held_model_for_restore is model

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.05)
        hb.push_command(
            "restore",
            {"version": 0, "blob_key": blob_key},
        )
        deadline = time.time() + 3.0
        while time.time() < deadline and engine._restored_checkpoint_version != 0:
            time.sleep(0.05)
        engine.request_stop()
        t.join(timeout=3.0)

        assert engine._restored_checkpoint_version == 0
        assert ledger.version == 0
        assert np.allclose(model.weights[0], w0[0])
        assert engine._run_authorized is False
        ok_acks = [a for a in hb.acks if a[1] and "restored" in (a[2] or "")]
        assert ok_acks, f"expected restore ACK, got {hb.acks}"
        engine.close()
    print("[PASSED] restore applies beginning weights + version while idle")


def test_restore_after_training_recovers_checkpoint_weights():
    """Train → capture mid checkpoint → drift weights → restore → match capture."""
    hb = _StubHB()
    model, prov = _bundle(seed=17)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=1, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="train-restore",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=1,
        )

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.05)
        hb.push_command("start")
        deadline = time.time() + 3.0
        while time.time() < deadline and ledger.version < 1:
            time.sleep(0.05)
        engine.request_stop()
        t.join(timeout=3.0)

        assert ledger.version >= 1
        mid = capture_model_checkpoint(model, version=int(ledger.version))
        mid_w = [np.copy(w) for w in mid["weights"]]
        mid_ver = int(mid["version"])

        for w in model.weights:
            w[...] += 1.5
        assert not np.allclose(model.weights[0], mid_w[0])

        blob_key = f"ckpts/test/v{mid_ver}.bin"
        hb.blobs[blob_key] = pickle.dumps(mid)
        # Simulate park after fit: drop session, keep held model.
        if engine.sessions:
            engine.drop_session(engine.sessions[0].session_id)
        engine._run_authorized = False

        t2 = threading.Thread(target=_go, daemon=True)
        t2.start()
        time.sleep(0.05)
        hb.push_command("restore", {"version": mid_ver, "blob_key": blob_key})
        deadline = time.time() + 3.0
        while (
            time.time() < deadline
            and engine._restored_checkpoint_version != mid_ver
        ):
            time.sleep(0.05)
        engine.request_stop()
        t2.join(timeout=3.0)

        assert engine._restored_checkpoint_version == mid_ver
        assert ledger.version == mid_ver
        assert np.allclose(model.weights[0], mid_w[0])
        assert engine._run_authorized is False
        engine.close()
    print("[PASSED] restore after training recovers checkpoint weights")


def test_cancel_revokes_run_authorization():
    hb = _StubHB()
    model, prov = _bundle(seed=19)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=100, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="cancel-flow",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=5,
        )

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.05)
        hb.push_command("start")
        deadline = time.time() + 2.0
        while time.time() < deadline and not engine._run_authorized:
            time.sleep(0.02)
        hb.push_command("cancel")
        deadline = time.time() + 2.0
        while time.time() < deadline and engine._run_authorized:
            time.sleep(0.02)
        engine.request_stop()
        t.join(timeout=3.0)
        assert engine._run_authorized is False
        engine.close()
    print("[PASSED] cancel revokes run authorization")


def test_stale_shutdown_command_stops_engine():
    """If TM redelivers shutdown on boot, engine exits — register must prevent this."""
    hb = _StubHB()
    model, prov = _bundle(seed=23)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=100, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="shut-boot",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=5,
        )

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.05)
        hb.push_command("shutdown")
        t.join(timeout=3.0)
        assert not t.is_alive(), "engine should exit after shutdown"
        assert engine._shutdown_accepted is True
        engine.close()
    print("[PASSED] shutdown command stops engine (stale redelivery hazard)")


def test_boot_idle_then_start_trains_without_auto_stop():
    """Happy path: paused → start → training; must NOT flip to stopped on its own."""
    hb = _StubHB()
    model, prov = _bundle(seed=29)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mlp",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=100, checkpoint_on_local_best=False
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.start_session(
            model=model,
            data_provider=prov,
            initial_lr=0.05,
            session_id="no-autostop",
            steps=2,
            source_mode=IngestionMode.CSV,
            model_type=ModelType.MULTI_CLASS,
            early_stopping_enabled=False,
            max_epochs=1,
        )

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.08)
        assert t.is_alive()
        assert engine._run_authorized is False
        hb.push_command("start")
        deadline = time.time() + 3.0
        while time.time() < deadline and ledger.version == 0:
            time.sleep(0.05)
        assert ledger.version > 0
        assert t.is_alive(), "engine died during training without shutdown"
        assert engine._run_authorized is True
        # Still alive after training work
        time.sleep(0.15)
        assert t.is_alive()
        assert not engine._stop.is_set()
        engine.request_stop()
        t.join(timeout=3.0)
        engine.close()
    print("[PASSED] boot idle → start trains without auto-stop")


if __name__ == "__main__":
    test_boot_stays_idle_despite_sticky_desired_running()
    test_start_command_authorizes_training()
    test_restore_applies_weights_and_version_while_idle()
    test_restore_after_training_recovers_checkpoint_weights()
    test_cancel_revokes_run_authorization()
    test_stale_shutdown_command_stops_engine()
    test_boot_idle_then_start_trains_without_auto_stop()
    print("all manager control tests passed")
