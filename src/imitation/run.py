"""Run one imitation fit from a Training Manager payload (the twin of ``closed_loop.run``): same ledger, same checkpoint rule."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_TRAIN_KEYS = ("history_k", "batch_size", "metric_train_max", "alpha", "temperature", "source_ratios", "architecture",
               "awr_beta", "awr_weight_max")


def run_imitation_payload(payload: dict[str, Any], *, boot_yaml: str) -> int:
    from config.config_loader import load_boot_host_context
    from src.imitation.corpus import load_corpus
    from src.imitation.fit import blob_from_state, fit_imitation
    from src.ledger import TrainingLedger, document_from_bytes
    from src.ledger_store import create_ledger_store

    _host_identity, host_output_dir = load_boot_host_context(boot_yaml)
    cfg = dict(payload)
    fit = dict(cfg.get("fit") or {})
    led = dict(cfg.get("ledger") or {})
    im = dict(cfg.get("imitation") or {})
    out_dir = Path(str((cfg.get("meta") or {}).get("output_dir") or host_output_dir or "diagnostics_output/imitation"))
    ledger_dir = out_dir / str(led.get("path") or "ledger")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if not im.get("corpus_path"):
        raise ValueError("an imitation fit needs imitation.corpus_path (the frozen corpus the engine was handed)")
    episodes = load_corpus(im["corpus_path"])

    init_blob = None
    restore = led.get("restore_checkpoint_path")
    if restore:
        body = document_from_bytes(Path(str(restore)).read_bytes()).body
        if not isinstance(body.get("state"), dict) or "blob" not in body["state"]:
            raise ValueError(f"checkpoint {restore} carries no brain weights to restore")
        init_blob = blob_from_state(body["state"])

    # Autopilot: stretch N uses seed + N - 1, so a retry from the same weights splits and shuffles differently.
    seed = int(fit.get("seed") or im.get("seed") or 0) + max(0, int(fit.get("stretch_index") or 1) - 1)
    lr = float(fit.get("lr") or (cfg.get("optimization") or {}).get("learning_rate") or 0.0)
    if lr <= 0.0:
        raise ValueError("an imitation fit needs a learning rate above 0 (fit.lr or optimization.learning_rate)")
    per_round = max(1, int(im.get("steps_per_round") or 5))
    budget = int(fit.get("run_budget") or 0)
    steps = (budget - int(fit.get("resume_from") or 0)) if budget else int(im.get("steps") or 100)

    store = create_ledger_store(str(led.get("store_backend") or "file_streaming"), ledger_dir)
    ledger = TrainingLedger(store=store, branch_id=str(led.get("branch_id") or "main"), model_instance_id=str(payload.get("model_id") or "imitation"),
                            architecture_id="imitation")
    ledger.run_config = led.get("run_config") or None
    try:
        if steps <= 0:
            ledger.push_run_end({"reason": "success", "epochs_run": 0, "best_version": None, "best_val_loss": None, "final_version": None})
            return 0
        end = fit_imitation(
            episodes, ledger,
            lr=lr, steps=steps, seed=seed, holdout_frac=float(im.get("holdout_frac") or 0.2), steps_per_round=per_round,
            patience=int(fit.get("patience") or 0),
            es_warmup=int(fit.get("es_warmup") if fit.get("es_warmup") is not None else 3),
            checkpoint_every=int(fit.get("checkpoint_every") or 5),
            train_kwargs={k: im[k] for k in _TRAIN_KEYS if im.get(k) is not None},
            init_blob=init_blob, model_instance_id=ledger.model_instance_id,
        )
        logging.warning("[imitation] done: %s", {k: v for k, v in end.items() if k != "events"})
    finally:
        store.close()
    return 0
