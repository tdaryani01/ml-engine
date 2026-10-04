"""Run one closed-loop fit from a Training Manager payload, the way ``run_pipeline`` runs a supervised one."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


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
    seed = int((cfg.get("optimization") or {}).get("seed") or fit.get("seed") or 0)
    seed_everything(seed)
    logging.warning("[closed-loop] assembling %s", (cfg.get("assembly") or {}).get("modules"))
    run = assemble_closed_loop(cfg, seed=seed)

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
    steps = max(1, budget - int(fit.get("resume_from") or 0)) if budget else int((cfg.get("closed_loop") or {}).get("max_steps") or 1)
    try:
        end = fit_closed_loop(
            run, ledger,
            lr=float(fit.get("lr") or (cfg.get("optimization") or {}).get("learning_rate") or 0.0),
            steps=steps,
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
