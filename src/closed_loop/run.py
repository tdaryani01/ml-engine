"""Run one closed-loop fit from a Training Manager payload, the way ``run_pipeline`` runs a supervised one."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


def rotate_plates(run: Any, fit: dict[str, Any]) -> int:
    """Autopilot: every earlier stretch that ended in an early stop moved the training plate on, so stretch N starts on the
    plate N-1 steps after the first. Stateless (the stretch index says how many), so each fit is a fresh run."""
    if not fit.get("rotate_on_es"):
        return 0
    n = max(0, int(fit.get("stretch_index") or 1) - 1)
    for _ in range(n):
        run.data.next_plate()
    return n


def run_closed_loop_payload(payload: dict[str, Any], *, boot_yaml: str) -> int:
    from config.config_loader import load_boot_host_context, parse_tm_production_config
    from src.closed_loop.assembler import assemble_closed_loop
    from src.closed_loop.fit import fit_closed_loop
    from src.closed_loop.state import restore_state
    from src.ledger import TrainingLedger, document_from_bytes
    from src.ledger_store import create_ledger_store
    from utils.seeding import seed_everything

    host_identity, host_output_dir = load_boot_host_context(boot_yaml)
    cfg = parse_tm_production_config(payload, host_identity=host_identity, output_dir=host_output_dir, profile="closed_loop")
    fit = dict(cfg.get("fit") or {})
    led = dict(cfg.get("ledger") or {})
    out_dir = Path(str((cfg.get("meta") or {}).get("output_dir") or host_output_dir or "diagnostics_output/closed_loop"))
    ledger_dir = out_dir / str(led.get("path") or "ledger")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    # Autopilot: stretch N shuffles with seed + N - 1, so a retry from the same weights is a new run, not a copy.
    seed = int((cfg.get("optimization") or {}).get("seed") or fit.get("seed") or 0) + max(0, int(fit.get("stretch_index") or 1) - 1)
    seed_everything(seed)
    logging.warning("[closed-loop] assembling %s", (cfg.get("assembly") or {}).get("modules"))
    run = assemble_closed_loop(cfg, seed=seed)

    rotate_plates(run, fit)

    restore = led.get("restore_checkpoint_path")
    if restore:
        body = document_from_bytes(Path(str(restore)).read_bytes()).body
        if not isinstance(body.get("state"), dict) or not body["state"]:
            raise ValueError(f"checkpoint {restore} carries no model state to restore")
        restore_state(run.actor, body["state"])

    store = create_ledger_store(str(led.get("store_backend") or "file_streaming"), ledger_dir)
    ledger = TrainingLedger(store=store, branch_id=str(led.get("branch_id") or "main"), model_instance_id=str(payload.get("model_id") or "closed-loop"),
                            architecture_id="closed_loop")
    ledger.run_config = led.get("run_config") or None
    budget = int(fit.get("run_budget") or 0)
    remaining = budget - int(fit.get("resume_from") or 0) if budget else int((cfg.get("closed_loop") or {}).get("max_steps") or 1)
    lr = float(fit.get("lr") or (cfg.get("optimization") or {}).get("learning_rate") or 0.0)
    if lr <= 0.0:
        raise ValueError("a closed-loop fit needs a learning rate above 0 (fit.lr or optimization.learning_rate)")
    try:
        if remaining <= 0:  # the run budget is already used up: nothing to train, end like a finished run
            ledger.push_run_end({"reason": "success", "epochs_run": 0, "best_version": None, "best_val_loss": None, "final_version": None})
            logging.warning("[closed-loop] budget %s already used (resume_from %s): nothing to do", budget, fit.get("resume_from"))
            return 0
        end = fit_closed_loop(
            run, ledger,
            lr=lr,
            steps=remaining,
            patience=int(fit.get("patience") or 0),
            es_warmup=int(fit.get("es_warmup") if fit.get("es_warmup") is not None else 10),
            checkpoint_every=int(fit.get("checkpoint_every") or led.get("checkpoint_every_steps") or 25),
            es_min_delta=float(fit.get("es_min_delta") if fit.get("es_min_delta") is not None else 1e-3),
            model_instance_id=ledger.model_instance_id,
        )
        logging.warning("[closed-loop] done: %s", end)
    finally:
        store.close()
        run.close()
    return 0
