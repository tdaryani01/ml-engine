# Closed-loop draw is a TrainingEngine config — one claim/HB/command loop only.
from __future__ import annotations

import inspect
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.ledger import LedgerConfig, TrainingLedger
from src.ledger_store import FileLedgerStore
from src.manager_heartbeat import ManagerCommand
from src.training_engine import TrainingEngine


@dataclass
class _StubHB:
    idle_sleep_s: float = 0.02
    _active_checkpoint: dict[str, Any] | None = None
    _commands: list[ManagerCommand] = field(default_factory=list)
    acks: list[tuple[str, bool, str | None]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    job_bound: bool = False
    job_id: str | None = None
    claim_calls: int = 0
    claim_jobs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active_checkpoint(self) -> dict[str, Any] | None:
        return self._active_checkpoint

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics = dict(metrics)

    def maybe_ping(self, force: bool = False) -> None:
        del force

    def poll_commands(self) -> list[ManagerCommand]:
        out = list(self._commands)
        self._commands.clear()
        return out

    def push_command(self, action: str, payload: dict[str, Any] | None = None) -> None:
        self._commands.append(
            ManagerCommand(
                id=f"cmd-{action}-{len(self.acks)}",
                action=action,
                payload=payload or {},
                created_at=time.time(),
            )
        )

    def mark_command_seen(self, cmd_id: str) -> None:
        del cmd_id

    def queue_ack(self, cmd_id: str, *, ok: bool, detail: str | None = None) -> None:
        self.acks.append((cmd_id, ok, detail))

    def try_claim_work(self, *, lease_s: float | None = None) -> dict[str, Any] | None:
        del lease_s
        self.claim_calls += 1
        if not self.claim_jobs:
            return None
        job = dict(self.claim_jobs.pop(0))
        self.job_bound = True
        self.job_id = str(job["job_id"])
        return job


def test_regression_closed_loop_has_no_second_claim_loop():
    """Agent module must not own claim/HB drain — TrainingEngine is authority."""
    from examples.closed_loop_draw import agent as draw_agent
    from examples.closed_loop_draw import run_lease

    src = inspect.getsource(draw_agent.DrawStudentAgent)
    assert "try_claim_work" not in src
    assert "_maybe_claim_work" not in src
    assert "_drain_commands" not in src
    assert "def run(" not in src
    main_src = inspect.getsource(draw_agent.main)
    assert "build_closed_loop_engine" in main_src
    assert "engine.run()" in main_src
    wire_src = inspect.getsource(run_lease.build_closed_loop_engine)
    assert "TrainingEngine" in wire_src
    assert "set_external_step" in wire_src
    assert "set_control_hooks" in wire_src
    assert "on_claim_config" in wire_src
    assert "on_release_config" in wire_src


def test_regression_claim_applies_job_config():
    hb = _StubHB()
    hb.claim_jobs.append(
        {
            "job_id": "job-cfg",
            "model_id": "draw-1",
            "kind": "train",
            "claim_token": "t",
            "config": {"closed_loop": {"sigma": 0.12}, "optimization": {"learning_rate": 0.003}},
        }
    )
    seen: list[dict[str, Any]] = []

    def on_claim(job: dict[str, Any]) -> bool:
        seen.append(dict(job))
        return True

    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="closed_loop_draw",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=10**9,
                checkpoint_on_local_best=False,
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.set_control_hooks(on_claim_config=on_claim)
        assert engine._maybe_claim_tm_work() is True
        assert len(seen) == 1
        assert seen[0]["config"]["closed_loop"]["sigma"] == 0.12
        assert engine._run_authorized is True
        engine.close()


def test_regression_release_resets_config():
    hb = _StubHB()
    released = {"n": 0}

    def on_release() -> None:
        released["n"] += 1

    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="closed_loop_draw",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=10**9,
                checkpoint_on_local_best=False,
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.set_control_hooks(on_release_config=on_release)
        engine._work_lease_active = True
        engine._run_authorized = True
        hb.job_bound = False
        engine._sync_work_lease()
        assert released["n"] == 1
        assert engine._run_authorized is False
        engine.close()


def test_regression_engine_claim_drives_external_step():
    """Idle claim authorizes run; external_step is the train body (closed-loop shape)."""
    hb = _StubHB()
    hb.claim_jobs.append(
        {"job_id": "job-1", "model_id": "tm-brain", "kind": "train", "claim_token": "t"}
    )
    ticks = {"n": 0}

    def step() -> bool:
        ticks["n"] += 1
        if ticks["n"] >= 2:
            return False
        return True

    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="closed_loop_draw",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=10**9,
                checkpoint_on_local_best=False,
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.set_external_step(step)

        def _go() -> None:
            engine.run()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        deadline = time.time() + 3.0
        while time.time() < deadline and ticks["n"] < 2:
            time.sleep(0.02)
        engine.request_stop()
        t.join(timeout=3.0)
        assert hb.claim_calls >= 1
        assert engine._run_authorized is True
        assert ticks["n"] >= 2
        engine.close()


def test_regression_pause_gate_blocks_claim():
    hb = _StubHB()
    hb.claim_jobs.append(
        {"job_id": "job-blocked", "model_id": "x", "kind": "train", "claim_token": "t"}
    )
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="closed_loop_draw",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=10**9,
                checkpoint_on_local_best=False,
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.set_control_hooks(pause_gate=lambda: True)
        assert engine._maybe_claim_tm_work() is False
        assert hb.claim_calls == 0
        engine.close()


def test_regression_bl026_engine_does_not_overwrite_metrics_while_leased():
    """BL-026: while job_bound, engine idle must not clobber worker ES output."""
    hb = _StubHB()
    hb.job_bound = True
    hb.metrics = {"state": "es-onset", "traj": 352}
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="closed_loop_draw",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=10**9,
                checkpoint_on_local_best=False,
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine._work_lease_active = True
        engine._run_authorized = True
        engine._publish_manager_metrics("idle")
        engine._idle_heartbeat()
        assert hb.metrics.get("state") == "es-onset"
        assert hb.metrics.get("traj") == 352
        engine.close()


def test_regression_bl027_no_desired_gate_trains_when_authorized():
    """BL-027: authorized + job_bound, no desired field — external_step runs."""
    hb = _StubHB()
    hb.job_bound = True
    ticks = {"n": 0}

    def step() -> bool:
        ticks["n"] += 1
        return ticks["n"] < 2

    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="closed_loop_draw",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(
                checkpoint_every_steps=10**9,
                checkpoint_on_local_best=False,
            ),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )
        engine.set_external_step(step)
        engine._work_lease_active = True
        engine._run_authorized = True
        engine.request_resume()

        def _go() -> None:
            engine.run(job_scoped=True)

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        deadline = time.time() + 3.0
        while time.time() < deadline and ticks["n"] < 2:
            time.sleep(0.02)
        engine.request_stop()
        t.join(timeout=3.0)
        assert ticks["n"] >= 2
        assert engine._run_authorized is True
        assert not hasattr(hb, "desired_state") or getattr(hb, "desired_state", None) is None
        engine.close()
