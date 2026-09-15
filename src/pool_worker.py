# src/pool_worker.py
"""Engine pool worker: idle register → claim → apply config → train → release."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping

from config.config_loader import load_job_pipeline_config
from config.schema import PipelineConfig


_CLOSED_LOOP_KEYS = ("closed_loop", "cnn_encoder", "tm_brain")


def job_is_closed_loop(config: Mapping[str, Any] | None) -> bool:
    """True when job.config is closed-loop shaped (assemble DrawApp, not PipelineConfig)."""
    if not isinstance(config, Mapping):
        return False
    if any(k in config for k in _CLOSED_LOOP_KEYS):
        return True
    meta = config.get("meta")
    if isinstance(meta, Mapping):
        name = str(meta.get("pipeline_name") or "").lower()
        if "closed_loop" in name or "draw" in name:
            return True
    return False


def materialize_job_config(
    boot_yaml: str,
    job: Mapping[str, Any],
) -> PipelineConfig:
    """Apply claimed job.config onto boot YAML (supervised PipelineConfig path)."""
    cfg_map = job.get("config") if isinstance(job.get("config"), Mapping) else {}
    return load_job_pipeline_config(boot_yaml, cfg_map)


def restore_job_weights(hb: Any, model: Any, job: Mapping[str, Any]) -> None:
    """If job pins resume_checkpoint.blob_key, fetch blob and restore into model."""
    data = job.get("data") if isinstance(job.get("data"), Mapping) else {}
    cfg = job.get("config") if isinstance(job.get("config"), Mapping) else {}
    resume = data.get("resume_checkpoint")
    if not isinstance(resume, Mapping):
        resume = cfg.get("resume_checkpoint")
    if not isinstance(resume, Mapping):
        return
    blob_key = resume.get("blob_key")
    if not blob_key:
        return
    fetch = getattr(hb, "fetch_blob", None)
    if not callable(fetch):
        raise RuntimeError("heartbeat cannot fetch_blob for resume_checkpoint")
    raw = fetch(str(blob_key))
    if raw is None:
        raise RuntimeError(f"resume blob not found: {blob_key}")
    from src.ledger import restore_model_checkpoint
    from src.manager_heartbeat import decode_checkpoint_blob

    body = decode_checkpoint_blob(raw)
    restore_model_checkpoint(model, body)
    logging.info(
        "[PoolWorker] restored resume_checkpoint blob=%s version=%s",
        blob_key,
        resume.get("version"),
    )


def wait_for_claim(
    hb: Any,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Register/HB idle and claim one job. Returns None if stop requested."""
    sleep_s = float(getattr(hb, "idle_sleep_s", None) or 10.0)
    while True:
        if should_stop is not None and should_stop():
            return None
        hb.set_metrics({"run_state": "idle", "state": "idle"})
        ping = getattr(hb, "maybe_ping", None)
        if callable(ping):
            try:
                ping(force=True)
            except Exception:  # noqa: BLE001
                logging.exception("[PoolWorker] idle ping failed")
        claim = getattr(hb, "try_claim_work", None)
        if not callable(claim):
            raise RuntimeError("heartbeat missing try_claim_work")
        try:
            job = claim()
        except Exception:  # noqa: BLE001
            logging.exception("[PoolWorker] try_claim_work failed")
            job = None
        if isinstance(job, dict) and job.get("job_id"):
            return dict(job)
        time.sleep(max(0.1, sleep_s))


def run_pool_worker_loop(
    *,
    boot_yaml: str,
    boot_cfg: PipelineConfig,
    hb: Any,
    run_supervised_job: Callable[[PipelineConfig, Any, dict[str, Any]], None],
    should_stop: Callable[[], bool] | None = None,
    boot_tm: Mapping[str, Any] | None = None,
) -> None:
    """Idle → claim → closed-loop or supervised assemble → ack/fail → idle."""
    del boot_cfg  # identity lives on hb; job materializes train cfg
    logging.info(
        "[PoolWorker] idle — pool=%s waiting for claim",
        getattr(hb, "pool_session_id", None),
    )
    while True:
        if should_stop is not None and should_stop():
            break
        job = wait_for_claim(hb, should_stop=should_stop)
        if job is None:
            break
        jid = job.get("job_id")
        mid = job.get("model_id")
        cfg_map = job.get("config") if isinstance(job.get("config"), Mapping) else {}
        logging.info(
            "[PoolWorker] claimed job=%s model=%s closed_loop=%s — applying config",
            jid,
            mid,
            job_is_closed_loop(cfg_map),
        )
        try:
            if job_is_closed_loop(cfg_map):
                from examples.closed_loop_draw.run_lease import run_closed_loop_claimed_job

                run_closed_loop_claimed_job(hb, job, boot_tm=boot_tm)
            else:
                cfg = materialize_job_config(boot_yaml, job)
                run_supervised_job(cfg, hb, job)
            if bool(getattr(hb, "job_bound", False)):
                ack = getattr(hb, "ack_work", None)
                if callable(ack):
                    out = ack(result={"ok": True, "model_id": mid, "job_id": jid})
                    if out is not None:
                        logging.info(
                            "[PoolWorker] job=%s complete — claim released, back to idle",
                            jid,
                        )
                    else:
                        logging.warning(
                            "[PoolWorker] job=%s finished locally but ack did not "
                            "release claim on TM",
                            jid,
                        )
                else:
                    logging.info("[PoolWorker] job=%s complete — back to idle", jid)
            else:
                logging.info(
                    "[PoolWorker] job=%s complete — already unbound, back to idle",
                    jid,
                )
        except Exception as exc:  # noqa: BLE001
            logging.exception("[PoolWorker] job=%s failed: %s", jid, exc)
            if bool(getattr(hb, "job_bound", False)):
                fail = getattr(hb, "fail_work", None)
                if callable(fail):
                    try:
                        fail(error=str(exc)[:800], requeue=True)
                    except Exception:  # noqa: BLE001
                        logging.exception("[PoolWorker] fail_work failed")
