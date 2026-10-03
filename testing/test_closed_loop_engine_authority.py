# Closed-loop draw is a TrainingEngine config — direct run, no claim/HB loop.
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


def _engine(hb: Any) -> TrainingEngine:
    tmp = tempfile.mkdtemp()
    ledger = TrainingLedger(
        store=FileLedgerStore(tmp),
        branch_id="main",
        architecture_id="closed_loop_draw",
    )
    return TrainingEngine(
        ledger=ledger,
        config=LedgerConfig(
            checkpoint_every_steps=10**9,
            checkpoint_on_local_best=False,
        ),
        manager_heartbeat=hb,  # type: ignore[arg-type]
    )


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
    assert "on_start_resume" in wire_src
    assert "on_release_config" in wire_src
    # No pool claim hooks anywhere in the wiring.
    assert "on_claim_config" not in wire_src
    assert "try_claim_work" not in wire_src


def test_regression_boot_authorizes_and_drives_external_step():
    """Standalone: authorized on boot; external_step is the train body."""
    hb = _StubHB()
    ticks = {"n": 0}

    def step() -> bool:
        ticks["n"] += 1
        return ticks["n"] < 2

    engine = _engine(hb)
    engine.set_external_step(step)

    t = threading.Thread(target=engine.run, daemon=True)
    t.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and ticks["n"] < 2:
        time.sleep(0.02)
    engine.request_stop()
    t.join(timeout=3.0)
    assert engine._run_authorized is True
    assert ticks["n"] >= 2
    engine.close()


def test_regression_pause_gate_blocks_training():
    hb = _StubHB()
    ticks = {"n": 0}
    engine = _engine(hb)
    engine.set_external_step(lambda: ticks.__setitem__("n", ticks["n"] + 1) or True)
    engine.set_control_hooks(pause_gate=lambda: True)

    t = threading.Thread(target=engine.run, daemon=True)
    t.start()
    time.sleep(0.15)
    engine.request_stop()
    t.join(timeout=3.0)
    assert ticks["n"] == 0, "pause_gate must block training"
    engine.close()


def test_regression_cancel_triggers_release_config():
    hb = _StubHB()
    released = {"n": 0}
    engine = _engine(hb)
    engine.set_control_hooks(on_release_config=lambda: released.__setitem__("n", released["n"] + 1))

    t = threading.Thread(target=engine.run, daemon=True)
    t.start()
    time.sleep(0.05)
    hb.push_command("cancel")
    deadline = time.time() + 2.0
    while time.time() < deadline and engine._run_authorized:
        time.sleep(0.02)
    engine.request_stop()
    t.join(timeout=3.0)
    assert engine._run_authorized is False
    assert released["n"] == 1
    engine.close()
