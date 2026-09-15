# examples/closed_loop_draw/run_lease.py
"""Closed-loop TrainingEngine lease: assemble from job.config, train, release."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping

from examples.closed_loop_draw.agent import DrawStudentAgent
from src.ledger import LedgerConfig, TrainingLedger
from src.ledger_store import FileLedgerStore
from src.training_engine import TrainingEngine

_REPO = Path(__file__).resolve().parents[2]


def tm_dict_from_heartbeat(hb: Any) -> dict[str, Any]:
    """Pool worker training_manager identity (keep across job.config overlays)."""
    cfg = getattr(hb, "_cfg", None) or getattr(hb, "cfg", None)
    if cfg is None:
        return {"enabled": True}
    return {
        "enabled": bool(getattr(cfg, "enabled", True)),
        "uri": str(getattr(cfg, "uri", "") or ""),
        "instance_id": str(getattr(cfg, "instance_id", "") or ""),
        "kind": str(getattr(cfg, "kind", "engine") or "engine"),
        "label": getattr(cfg, "label", None),
        "advertise_url": str(getattr(cfg, "advertise_url", "http://127.0.0.1:0") or ""),
        "capabilities": list(getattr(cfg, "capabilities", ()) or ()),
        "interval_s": float(getattr(cfg, "interval_s", 10.0)),
        "timeout_s": float(getattr(cfg, "timeout_s", 0.5)),
        "idle_sleep_s": float(getattr(cfg, "idle_sleep_s", 10.0)),
        "park_when_idle": bool(getattr(cfg, "park_when_idle", True)),
    }


def build_closed_loop_engine(
    agent: DrawStudentAgent,
    *,
    model_instance_id: str,
    ledger_dir: Path | str,
    claim_inside_engine: bool = True,
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
    )
    engine.set_external_step(agent.train_tick)
    # Manual ES ends the lease via request_stop → pool_worker ack → idle.
    bind = getattr(agent, "bind_engine_stop", None)
    if callable(bind):
        bind(engine.request_stop)
    hooks: dict[str, Any] = {
        "on_start_resume": agent.on_engine_start_resume,
        "on_pause": agent.on_engine_pause,
        "on_cancel": agent.on_engine_cancel,
        "on_restore": agent.on_engine_restore,
        "pause_gate": agent.pause_gate,
        "on_release_config": agent.on_release_config,
    }
    if claim_inside_engine:
        hooks["on_claim_config"] = agent.on_claim_config
    engine.set_control_hooks(**hooks)
    return engine


def run_closed_loop_claimed_job(
    hb: Any,
    job: Mapping[str, Any],
    *,
    boot_tm: Mapping[str, Any] | None = None,
) -> None:
    """Claim already held: assemble from job.config, adopt lease, job-scoped run."""
    raw = job.get("config") if isinstance(job.get("config"), Mapping) else {}
    cfg = copy.deepcopy(dict(raw))
    tm = dict(boot_tm) if boot_tm is not None else tm_dict_from_heartbeat(hb)
    cfg["training_manager"] = copy.deepcopy(tm)

    mid = str(job.get("model_id") or "").strip()
    if not mid:
        raise ValueError("closed_loop claim missing model_id")

    out_dir = Path(
        (cfg.get("meta") or {}).get("output_dir")
        or "diagnostics_output/closed_loop_draw_agent"
    )
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = DrawStudentAgent(cfg, heartbeat=hb)
    try:
        ok = bool(agent.on_claim_config(dict(job)))
    except Exception:
        agent.close()
        raise
    if not ok:
        agent.close()
        raise RuntimeError("claim_config_rejected")

    engine = build_closed_loop_engine(
        agent,
        model_instance_id=mid,
        ledger_dir=out_dir / "engine_ledger",
        claim_inside_engine=False,
    )
    logging.info(
        "[ClosedLoopLease] job=%s model=%s — training",
        job.get("job_id"),
        mid,
    )
    try:
        engine.adopt_claimed_job(dict(job))
        engine.run(job_scoped=True)
    finally:
        engine.close()
        agent.close()
    # Normal finish (e.g. manual ES already acked inside the agent): if the
    # lease is still bound, release it here so TM does not stay claimed.
    if bool(getattr(hb, "job_bound", False)):
        ack = getattr(hb, "ack_work", None)
        if callable(ack):
            out = ack(
                result={
                    "ok": True,
                    "reason": "lease_complete",
                    "model_id": mid,
                    "job_id": job.get("job_id"),
                }
            )
            if out is not None:
                logging.info(
                    "[ClosedLoopLease] job=%s acked after run — claim released",
                    job.get("job_id"),
                )
            else:
                logging.warning(
                    "[ClosedLoopLease] job=%s still bound — ack did not release claim",
                    job.get("job_id"),
                )
