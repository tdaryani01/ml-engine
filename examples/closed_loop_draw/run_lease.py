# examples/closed_loop_draw/run_lease.py
"""Closed-loop TrainingEngine wiring: assemble from boot YAML, train directly."""
from __future__ import annotations

from pathlib import Path

from examples.closed_loop_draw.agent import DrawStudentAgent
from src.ledger import LedgerConfig, TrainingLedger
from src.ledger_store import FileLedgerStore
from src.training_engine import TrainingEngine


def build_closed_loop_engine(
    agent: DrawStudentAgent,
    *,
    model_instance_id: str,
    ledger_dir: Path | str,
) -> TrainingEngine:
    """Wire TrainingEngine + external_step + control hooks for a DrawStudentAgent."""
    ledger_path = Path(ledger_dir)
    ledger_path.mkdir(parents=True, exist_ok=True)
    ledger = TrainingLedger(
        store=FileLedgerStore(str(ledger_path)),
        branch_id=str(agent._branch_id),
        architecture_id="closed_loop_draw",
        model_instance_id=str(model_instance_id),
    )
    engine = TrainingEngine(
        ledger=ledger,
        config=LedgerConfig(
            checkpoint_every_steps=10**9,
            checkpoint_on_local_best=False,
        ),
        manager_heartbeat=agent.hb,
        # TM start policy from the boot training_manager block (default: train now).
        authorize_on_boot=bool(
            (agent._boot_cfg.get("training_manager") or {}).get("authorize_on_boot", True)
        ),
    )
    engine.set_external_step(agent.train_tick)
    # Manual ES ends the run via request_stop; Autopilot ES parks via request_pause.
    agent.bind_engine_stop(engine.request_stop)
    agent.bind_engine_pause(engine.request_pause)
    engine.set_control_hooks(
        on_start_resume=agent.on_engine_start_resume,
        on_pause=agent.on_engine_pause,
        on_cancel=agent.on_engine_cancel,
        on_restore=agent.on_engine_restore,
        pause_gate=agent.pause_gate,
        on_release_config=agent.on_release_config,
    )
    return engine


