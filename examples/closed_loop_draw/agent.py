# examples/closed_loop_draw/agent.py
"""
Closed-loop draw config for the training engine (like CNN/MLP — not a second loop).

TrainingEngine owns claim / HB / command drain. This module supplies:
  - train_tick (external_step)
  - control hooks (start/resume config, pause holds, draw-blob restore)
  - ES / tm-brain act + remix (draw-specific physiology)

Episodes + outcomes stay on the live write path for brain training.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from examples.closed_loop_draw.assemble import assemble, load_config, make_target
from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS
from examples.closed_loop_draw.draw_checkpoint import (
    apply_checkpoint_blob,
    build_checkpoint_blob,
    job_overlay_only,
    load_checkpoint_blob,
    public_run_config,
)
from src.manager_heartbeat import _diag, maybe_from_settings
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TM-managed draw student agent")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config_draw_agent.yaml"),
        help="YAML config path",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Append status lines here (default: <output_dir>/draw-student.log)",
    )
    return p.parse_args()


_LOG_FP = None


def _open_log_file(path: Path) -> None:
    global _LOG_FP
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FP = open(path, "a", encoding="utf-8")  # noqa: SIM115 — process lifetime
    print(f"[draw-student] logging to {path}", flush=True)


def _emit(state: str, *, loss: float | None = None, traj: int = 0) -> None:
    bits = [f"[draw-student] status={state}"]
    if traj > 0:
        bits.append(f"traj={traj}")
    if loss is not None:
        bits.append(f"loss={loss:.4f}")
    line = " ".join(bits)
    print(line, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(line + "\n")
        _LOG_FP.flush()


class DrawStudentAgent:
    """Headless closed-loop draw *student* controlled by TM dial-out commands."""

    def __init__(self, cfg: dict, *, heartbeat: Any | None = None) -> None:
        bootstrap_im2col_gemm_runtime()
        seed = int(cfg.get("optimization", {}).get("seed", 0))
        self.cfg = cfg
        self._boot_cfg = copy.deepcopy(cfg)
        self._configured = False
        self._config_version: int | None = None
        self.app = assemble(cfg, seed=seed)
        cl = cfg["closed_loop"]
        self.B = self.app.batch_size
        self.lr = self.app.lr
        self.command_id = int(cl.get("command_id", 0))
        self.command_ids = np.full(self.B, self.command_id, dtype=np.int64)
        self.target = make_target(cfg, batch_size=self.B)
        # Sealed-ish probe: same geometry family, different params / seed offset.
        probe_cfg = copy.deepcopy(cfg)
        probe_cfg.setdefault("closed_loop", {})
        tgt = dict(probe_cfg["closed_loop"].get("target") or {})
        tgt["kind"] = tgt.get("kind") or "stock"
        # Force a distinct probe by bumping seed used only for target generation.
        self._probe_target = make_target(probe_cfg, batch_size=self.B)
        # Nudge probe away from train target when both are stock circles.
        if hasattr(self._probe_target, "shape"):
            noise = np.random.RandomState(seed + 17).randn(*self._probe_target.shape)
            self._probe_target = np.clip(
                self._probe_target + 0.05 * noise.astype(np.float32), 0.0, 1.0
            )

        led = cfg.get("ledger") or {}
        self._ledger_on = bool(led.get("enabled", True))
        self._checkpoint_every = max(1, int(led.get("checkpoint_every", 25)))
        self._branch_id = str(led.get("branch_id", "main"))

        self._train_patience = max(1, int(cl.get("train_patience", 32)))
        self._es_min_traj = max(1, int(cl.get("es_min_traj", 20)))
        # On early-stop: sample a different stock target mix (default on).
        self._remix_data_on_es = bool(cl.get("remix_data_on_es", True))
        self._remix_count = 0
        # Armed by restore_best / successful restore; consumed once (restore or after_restore).
        self._remix_after_restore = False
        brain = cfg.get("tm_brain") or {}
        self._es_shadow = bool(brain.get("shadow_on_es", True))
        # Apply rule decisions only under Autopilot (see _maybe_es_shadow).
        self._es_apply = bool(brain.get("apply_on_es", True))
        # Job overlay: Start Autopilot checkbox. Default off = manual run-once.
        self._autopilot = bool(cfg.get("autopilot", False)) or (
            str(cfg.get("source") or "").strip().lower() == "autopilot"
        )

        tm = cfg.get("training_manager") or {}
        caps = list(tm.get("capabilities") or [])
        if "restore" not in caps:
            caps.append("restore")
            tm = dict(tm)
            tm["capabilities"] = caps
            cfg["training_manager"] = tm
        if heartbeat is not None:
            self.hb = heartbeat
        else:
            self.hb = maybe_from_settings(tm, ledger_enabled=self._ledger_on)
            if self.hb is None:
                raise RuntimeError(
                    "training_manager.enabled + uri required "
                    "(see config_draw_agent.yaml)"
                )

        self._run_authorized = False
        self._paused = False
        self._user_pause_hold = False
        self._stop = False
        self._shutdown_accepted = False
        self._traj = 0
        self._last_loss: float | None = None
        self._last_ink_miss: float | None = None
        self._last_probe: float | None = None
        self._best_ink_miss: float | None = None
        self._best_probe: float | None = None
        self._stale = 0
        self._es_tripped = False
        self._es_park_hold = False
        self._es_run_done = False
        self._handled_onset_version: int | None = None
        self._pending_outcome_episode: str | None = None
        self._outcome_horizon = max(1, int(brain.get("outcome_horizon", 10)))
        self._outcome_due_traj: int | None = None
        self._checkpoint_version: int | None = None
        self._process_started_at = time.time()
        self._sigma = float(cl.get("sigma", 0.06))
        self._max_steps = int(cl.get("max_steps", 10))
        self._continuity_weight = float(cl.get("continuity_weight", 0.0))
        self._last_status: str | None = None
        # Bound by run_lease → TrainingEngine.request_stop (manual ES ends job).
        self._engine_stop: Callable[[], None] | None = None
        # Bound by run_lease → TrainingEngine.request_pause (Autopilot ES park).
        self._engine_pause: Callable[[], None] | None = None

    def bind_engine_stop(self, stop: Callable[[], None] | None) -> None:
        """Wire TrainingEngine.request_stop so manual ES can finish the lease."""
        self._engine_stop = stop

    def bind_engine_pause(self, pause: Callable[[], None] | None) -> None:
        """Wire TrainingEngine.request_pause for Autopilot ES local park."""
        self._engine_pause = pause

    def close(self) -> None:
        self.app.close()

    def _tm_agent_id(self) -> str:
        """TM agent id for control-plane routes and ckpt keys.

        Pool workers keep ``hb.cfg.instance_id`` as the anonymous session id.
        Job-bound routes must use ``bound_model_id`` (e.g. tm-brain) or ES act
        404s against the session.
        """
        mid = getattr(self.hb, "bound_model_id", None)
        if callable(mid):
            mid = mid()
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
        return str(getattr(self.hb.cfg, "instance_id", "") or "").strip()

    @staticmethod
    def _deep_merge(base: dict, overlay: dict) -> dict:
        out = copy.deepcopy(base) if isinstance(base, dict) else {}
        for key, value in (overlay or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = DrawStudentAgent._deep_merge(out[key], value)
            else:
                out[key] = copy.deepcopy(value)
        return out

    def _reload_live_knobs(self) -> None:
        """Refresh live fields from self.cfg after assemble / merge."""
        cl = self.cfg.get("closed_loop") or {}
        opt = self.cfg.get("optimization") or {}
        brain = self.cfg.get("tm_brain") or {}
        led = self.cfg.get("ledger") or {}
        self.B = self.app.batch_size
        self.lr = float(opt.get("learning_rate", self.app.lr))
        self.command_id = int(cl.get("command_id", 0))
        self.command_ids = np.full(self.B, self.command_id, dtype=np.int64)
        self.target = make_target(self.cfg, batch_size=self.B)
        probe_cfg = copy.deepcopy(self.cfg)
        probe_cfg.setdefault("closed_loop", {})
        self._probe_target = make_target(probe_cfg, batch_size=self.B)
        self._ledger_on = bool(led.get("enabled", True))
        self._checkpoint_every = max(1, int(led.get("checkpoint_every", 25)))
        self._branch_id = str(led.get("branch_id", "main"))
        self._train_patience = max(1, int(cl.get("train_patience", 32)))
        self._es_min_traj = max(1, int(cl.get("es_min_traj", 20)))
        self._remix_data_on_es = bool(cl.get("remix_data_on_es", True))
        self._es_shadow = bool(brain.get("shadow_on_es", True))
        self._es_apply = bool(brain.get("apply_on_es", True))
        self._outcome_horizon = max(1, int(brain.get("outcome_horizon", 10)))
        self._sigma = float(cl.get("sigma", 0.06))
        self._max_steps = int(cl.get("max_steps", 10))
        self._continuity_weight = float(cl.get("continuity_weight", 0.0))
        self._stale = 0
        self._es_tripped = False
        self._best_ink_miss = None
        self._best_probe = None

    def apply_run_config(self, config: dict | None, *, rebuild: bool = False) -> dict:
        """Apply YAML-shaped or flat knobs (claim + brain resume share this path)."""
        if not isinstance(config, dict) or not config:
            return {}
        overlay_keys = {"source", "autopilot", "gym"}
        body = {k: v for k, v in config.items() if k not in overlay_keys}
        has_structure = any(
            k in body for k in ("closed_loop", "mhsa", "cnn_encoder", "optimization", "ledger", "tm_brain")
        )
        if rebuild or has_structure:
            tm = copy.deepcopy(self._boot_cfg.get("training_manager") or {})
            merged = self._deep_merge(self.cfg, body)
            merged["training_manager"] = tm
            seed = int(merged.get("optimization", {}).get("seed", 0))
            try:
                self.app.close()
            except Exception:  # noqa: BLE001
                pass
            self.cfg = merged
            self.app = assemble(merged, seed=seed)
            self._reload_live_knobs()
            self._configured = True
            _emit("config:applied:rebuild", traj=self._traj)
            return self._live_config()
        # Flat knob overlay (brain retune dialect).
        confirmed = self._apply_config_payload(body)
        self._configured = True
        return confirmed

    def on_claim_config(self, job: dict) -> bool:
        """Configure from job; if resume_checkpoint pinned, restore that ckpt first."""
        try:
            cfg = dict(job.get("config") or {})
            data = dict(job.get("data") or {})
            _diag(
                "agent_claim_config_enter",
                job_id=job.get("job_id"),
                model_id=job.get("model_id"),
                pool=self.hb.cfg.instance_id,
                config_keys=sorted(cfg.keys()),
                has_resume=bool(
                    isinstance(data.get("resume_checkpoint"), dict)
                    or isinstance(cfg.get("resume_checkpoint"), dict)
                ),
                boot_assembled=True,
                configured=self._configured,
            )
            resume = data.get("resume_checkpoint")
            if not isinstance(resume, dict):
                resume = cfg.get("resume_checkpoint")
            if isinstance(resume, dict) and resume.get("blob_key"):
                ok = self._restore_checkpoint(
                    version=resume.get("version"),
                    blob_key=str(resume["blob_key"]),
                    command_id=None,
                )
                if not ok:
                    _diag("agent_claim_config_resume_fail", job_id=job.get("job_id"))
                    return False
                # Overlay feed/autopilot only — hot knobs come from the checkpoint.
                for k, v in job_overlay_only(cfg).items():
                    self.cfg[k] = copy.deepcopy(v)
                self._autopilot = bool(cfg.get("autopilot", False)) or (
                    str(cfg.get("source") or "").strip().lower() == "autopilot"
                )
                self._es_park_hold = False
                self._es_run_done = False
                self._user_pause_hold = False
                clear_si = getattr(self.hb, "clear_site_interrupt", None)
                if callable(clear_si):
                    clear_si()
                _emit(
                    f"claim-config:resume-ckpt="
                    f"v{resume.get('version')} job={job.get('job_id')}",
                    traj=self._traj,
                )
                _diag(
                    "agent_claim_config_resume_ok",
                    job_id=job.get("job_id"),
                    version=resume.get("version"),
                )
                return True

            raw_ver = cfg.get("config_version")
            try:
                self._config_version = (
                    int(raw_ver) if raw_ver is not None else None
                )
            except (TypeError, ValueError):
                self._config_version = None
            self.apply_run_config(cfg, rebuild=True)
            self._autopilot = bool(cfg.get("autopilot", False)) or (
                str(cfg.get("source") or "").strip().lower() == "autopilot"
            )
            self._es_park_hold = False
            self._es_run_done = False
            self._user_pause_hold = False
            clear_si = getattr(self.hb, "clear_site_interrupt", None)
            if callable(clear_si):
                clear_si()
            _emit(
                f"claim-config:job={job.get('job_id')} "
                f"model={job.get('model_id')} "
                f"config_v={self._config_version}",
                traj=self._traj,
            )
            _diag(
                "agent_claim_config_rebuild_ok",
                job_id=job.get("job_id"),
                model_id=job.get("model_id"),
                config_v=self._config_version,
                sigma=self._sigma,
                lr=float(self.lr),
                max_steps=self._max_steps,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            _emit(f"claim-config:fail:{exc}", traj=self._traj)
            _diag("agent_claim_config_fail", error=str(exc), job_id=job.get("job_id"))
            return False

    def on_release_config(self) -> None:
        """Reset to bootstrap YAML when lease returns to the pool."""
        seed = int(self._boot_cfg.get("optimization", {}).get("seed", 0))
        try:
            self.app.close()
        except Exception:  # noqa: BLE001
            pass
        self.cfg = copy.deepcopy(self._boot_cfg)
        self.app = assemble(self.cfg, seed=seed)
        self._reload_live_knobs()
        self._configured = False
        self._config_version = None
        self._es_park_hold = False
        self._es_run_done = False
        self._user_pause_hold = False
        self._remix_after_restore = False
        _emit("config:reset:pool", traj=self._traj)

    def _set_status(self, state: str, *, loss: float | None = None) -> None:
        """Publish metrics always; print only when the run status changes."""
        changed = state != self._last_status
        self._publish(state)
        if changed:
            self._last_status = state
            _emit(state, loss=loss if loss is not None else self._last_loss, traj=self._traj)

    def _publish(self, state: str) -> None:
        metrics: dict = {
            "state": state,
            "traj": self._traj,
            "command_id": self.command_id,
            "sigma": self._sigma,
            "max_steps": self._max_steps,
            "continuity_weight": self._continuity_weight,
            "lr": float(self.lr),
            "train_patience": int(self._train_patience),
            "es_stale": int(self._stale),
            "ledger": "on" if self._ledger_on else "off",
        }
        if self._last_loss is not None:
            metrics["loss"] = float(self._last_loss)
        if self._last_ink_miss is not None:
            metrics["ink_miss"] = float(self._last_ink_miss)
            metrics["val_loss"] = float(self._last_ink_miss)
        if self._last_probe is not None:
            metrics["probe_loss"] = float(self._last_probe)
        if self._checkpoint_version is not None:
            metrics["checkpoint_version"] = int(self._checkpoint_version)
            metrics["version"] = int(self._checkpoint_version)
        self.hb.set_metrics(metrics)
        self.hb.maybe_ping(force=True)

    def _append_step(self) -> None:
        if not self._ledger_on:
            return
        version = int(self._traj)
        train = float(self._last_loss) if self._last_loss is not None else None
        val = float(self._last_ink_miss) if self._last_ink_miss is not None else None
        probe = float(self._last_probe) if self._last_probe is not None else None
        gap = None if train is None or val is None else float(val - train)
        body = {
            "version": version,
            "step_id": version,
            "train_loss": train,
            "val_loss": val,
            "probe_loss": probe,
            "metrics": {
                "version": version,
                "train_loss": train,
                "val_loss": val,
                "probe_loss": probe,
                "train_val_gap": gap,
                "ink_miss": self._last_ink_miss,
                "command_id": self.command_id,
                "lr": float(self.lr),
                "sigma": self._sigma,
                "max_steps": self._max_steps,
                "continuity_weight": self._continuity_weight,
            },
        }
        self.hb.append_ledger_doc(
            doc_type="step.complete",
            body=body,
            branch_id=self._branch_id,
        )
        _diag(
            "agent_step_append",
            traj=version,
            pool=self.hb.cfg.instance_id,
            bound=getattr(self.hb, "_bound_model_id", None),
            job=getattr(self.hb, "_job_id", None),
            train=train,
            val=val,
        )

    def _maybe_checkpoint(self) -> None:
        if not self._ledger_on:
            return
        if self._traj <= 0 or (self._traj % self._checkpoint_every) != 0:
            return
        version = int(self._traj)
        ink = self._last_ink_miss
        is_best = False
        if ink is not None and (
            self._best_ink_miss is None or ink < self._best_ink_miss - 1e-6
        ):
            # best tracked in _train_one; still mark local best when improved
            is_best = True
        blob_key = f"ckpts/{self._tm_agent_id()}/v{version}.pkl"
        try:
            payload = build_checkpoint_blob(
                self.app,
                version=version,
                cfg=self.cfg,
                lr=float(self.lr),
                val_loss=float(ink) if ink is not None else None,
            )
            uploaded = self.hb.put_blob(blob_key, payload, timeout_s=30.0)
        except Exception as exc:  # noqa: BLE001
            _emit(f"checkpoint-blob-failed:{exc}", traj=self._traj)
            uploaded = False
            blob_key = None  # type: ignore[assignment]
        body = {
            "version": version,
            "val_loss": float(ink) if ink is not None else None,
            "ink_miss": ink,
            "probe_loss": self._last_probe,
            "is_local_best": is_best,
            "weights_in_process": not uploaded,
            "knobs": {
                "learning_rate": float(self.lr),
                "train_patience": int(self._train_patience),
                "sigma": self._sigma,
                "max_steps": self._max_steps,
            },
            # BL-008a: full live run config; identity = checkpoint version.
            "config": public_run_config(self.cfg, lr=float(self.lr)),
            "config_snapshot": True,
        }
        ok = self.hb.append_ledger_doc(
            doc_type="checkpoint",
            body=body,
            branch_id=self._branch_id,
            blob_key=blob_key if uploaded else None,
        )
        if ok:
            self._checkpoint_version = version

    def _live_config(self) -> dict:
        return {
            "learning_rate": float(self.lr),
            "train_patience": int(self._train_patience),
            "sigma": float(self._sigma),
            "max_steps": int(self._max_steps),
            "continuity_weight": float(self._continuity_weight),
        }

    def _apply_config_payload(self, config: dict | None) -> dict:
        """Apply TM start/resume ``config``; return confirmed live values."""
        if not isinstance(config, dict) or not config:
            return {}
        confirmed: dict = {}
        if config.get("learning_rate") is not None:
            self.lr = float(config["learning_rate"])
            self.cfg.setdefault("optimization", {})["learning_rate"] = self.lr
            confirmed["learning_rate"] = float(self.lr)
        if config.get("train_patience") is not None:
            self._train_patience = max(1, int(config["train_patience"]))
            self.cfg.setdefault("closed_loop", {})["train_patience"] = self._train_patience
            confirmed["train_patience"] = int(self._train_patience)
        if config.get("sigma") is not None:
            self._sigma = float(config["sigma"])
            self.cfg.setdefault("closed_loop", {})["sigma"] = self._sigma
            confirmed["sigma"] = float(self._sigma)
        if config.get("max_steps") is not None:
            self._max_steps = int(config["max_steps"])
            self.cfg.setdefault("closed_loop", {})["max_steps"] = self._max_steps
            confirmed["max_steps"] = int(self._max_steps)
        if config.get("continuity_weight") is not None:
            self._continuity_weight = float(config["continuity_weight"])
            self.cfg.setdefault("closed_loop", {})[
                "continuity_weight"
            ] = self._continuity_weight
            confirmed["continuity_weight"] = float(self._continuity_weight)
        return confirmed

    def _print_config(self, tag: str, knobs: dict) -> None:
        if not knobs:
            return
        bits = " ".join(f"{k}={v}" for k, v in sorted(knobs.items()))
        _emit(f"config:{tag} {bits}", traj=self._traj)

    def _restore_checkpoint(
        self,
        *,
        version: Any,
        blob_key: str,
        command_id: str | None,
    ) -> bool:
        """Load checkpoint blob: config then weights (stateless continuity)."""
        data = self.hb.fetch_blob(str(blob_key))
        if data is None:
            if command_id is not None:
                self.hb.queue_ack(
                    command_id, ok=False, detail=f"blob not found: {blob_key}"
                )
            return False
        try:
            body = load_checkpoint_blob(data)
            cfg = body.get("config") if isinstance(body.get("config"), dict) else {}
            knobs = body.get("knobs") if isinstance(body.get("knobs"), dict) else {}
            # Config first (may rebuild app), then weights onto that app.
            if cfg:
                self.apply_run_config(cfg, rebuild=True)
            elif knobs:
                self._apply_config_payload(knobs)
            apply_checkpoint_blob(self.app, body)
        except Exception as exc:  # noqa: BLE001
            if command_id is not None:
                self.hb.queue_ack(
                    command_id, ok=False, detail=f"restore failed: {exc}"
                )
            _emit(f"restore-fail:{exc}", traj=self._traj)
            return False
        if version is not None:
            try:
                ver_i = int(version)
            except (TypeError, ValueError):
                ver_i = None
            if ver_i is not None:
                self._checkpoint_version = ver_i
                self._config_version = ver_i
                # Continue traj numbering from the restored point.
                self._traj = ver_i
        self._stale = 0
        self._es_tripped = False
        self.hb.set_active_checkpoint(
            {
                "version": int(version) if version is not None else None,
                "blob_key": str(blob_key),
            }
        )
        detail = f"restored:v{version}"
        if command_id is not None:
            self.hb.queue_ack(command_id, ok=True, detail=detail)
        _emit(detail, loss=self._last_loss, traj=self._traj)
        return True

    def _restore_from_command(self, cmd) -> None:
        """Restore weights + checkpoint config (explicit / cancel / brain)."""
        payload = dict(cmd.payload or {})
        blob_key = payload.get("blob_key")
        version = payload.get("version")
        if not blob_key:
            self.hb.queue_ack(cmd.id, ok=False, detail="missing blob_key")
            return
        ok = self._restore_checkpoint(
            version=version, blob_key=str(blob_key), command_id=cmd.id
        )
        if not ok:
            return
        # Onset/ES → restore first; then remix terrain for the next stretch.
        if self._remix_data_on_es:
            self._remix_after_restore = True
            self._consume_terrain_remix(reason="post-restore")

    def on_engine_start_resume(self, cmd) -> bool:
        """Config hook for TrainingEngine — apply payload; False blocks authorize."""
        payload = dict(cmd.payload or {})
        src = str(payload.get("source") or "").strip().lower()
        phase = str(payload.get("phase") or "").strip().lower()
        if self._user_pause_hold and src == "tm_brain":
            _emit("blocked:user_paused", loss=self._last_loss, traj=self._traj)
            return False
        if src != "tm_brain":
            self._user_pause_hold = False
            clear_si = getattr(self.hb, "clear_site_interrupt", None)
            if callable(clear_si):
                clear_si()
        cfg = (
            payload.get("config") if isinstance(payload.get("config"), dict) else None
        )
        confirmed = self.apply_run_config(cfg, rebuild=False)
        self._es_park_hold = False
        self._es_run_done = False
        if src == "tm_brain" and phase == "after_restore":
            # BL-005i: TM stock plate wins over local remix when stamped.
            if not self._apply_stock_from_feed(cfg):
                self._consume_terrain_remix(reason="after_restore")
        live = self._live_config()
        if confirmed:
            self._print_config("confirmed", confirmed)
        else:
            self._print_config("live", live)
        _emit(
            f"{cmd.action.strip().lower()}:config_ok",
            loss=self._last_loss,
            traj=self._traj,
        )
        return True

    def on_site_interrupt(self) -> None:
        """BL-023: TM unreachable long enough — park like a user Pause."""
        self._user_pause_hold = True
        self._paused = True
        _emit("paused:site_interrupt", loss=self._last_loss, traj=self._traj)

    def on_engine_pause(self, cmd) -> None:
        payload = dict(cmd.payload or {})
        src = str(payload.get("source") or "").strip().lower()
        if src != "tm_brain":
            self._user_pause_hold = True
        phase = payload.get("phase")
        _emit(
            f"paused:{phase}" if phase else "paused",
            loss=self._last_loss,
            traj=self._traj,
        )

    def on_engine_cancel(self, cmd) -> None:
        del cmd
        self._user_pause_hold = False
        self._es_park_hold = False
        self._es_run_done = False
        _emit("cancelled", loss=self._last_loss, traj=self._traj)

    def on_engine_restore(self, cmd) -> bool:
        """Draw-blob restore — fully handled (skip ledger restore)."""
        self.hb.mark_command_seen(cmd.id)
        self._restore_from_command(cmd)
        return True

    def pause_gate(self) -> bool:
        """Durable pause only (human / site-interrupt) — blocks claim/unpause.

        Autopilot ES soft-hold is train_tick + local engine pause (not here).
        After UI Resume unbound the lease, clear ``_user_pause_hold`` so the
        worker can reclaim the requeued job.
        """
        if bool(getattr(self.hb, "site_interrupt_hold", False)):
            return True
        if not self._user_pause_hold:
            return False
        if not bool(getattr(self.hb, "job_bound", False)):
            self._user_pause_hold = False
            return False
        return True

    def train_tick(self) -> bool:
        """One closed-loop traj for TrainingEngine.external_step."""
        if self._es_run_done or self._es_park_hold or self._user_pause_hold:
            return False
        loss = self._train_one()
        if self._es_run_done or self._es_park_hold:
            return False
        prev = self._last_status
        self._set_status("training", loss=loss)
        if prev == "training" and self._traj % 10 == 0:
            _emit("training", loss=loss, traj=self._traj)
        return True

    def _handle_start_resume(self, cmd) -> None:
        """Test/compat: config hook + local authorize (engine path uses hooks)."""
        if not self.on_engine_start_resume(cmd):
            self.hb.mark_command_seen(cmd.id)
            self.hb.queue_ack(cmd.id, ok=False, detail="blocked:user_paused")
            return
        self._run_authorized = True
        self._paused = False
        self.hb.mark_command_seen(cmd.id)
        live = self._live_config()
        ack = {
            "ok": True,
            "action": cmd.action,
            "phase": (cmd.payload or {}).get("phase")
            if isinstance(cmd.payload, dict)
            else None,
            "config_confirmed": live,
        }
        self.hb.queue_ack(cmd.id, ok=True, detail=json.dumps(ack, sort_keys=True))

    def _onset_within_patience(self) -> tuple[bool, int | None]:
        """True when a *new* unhealthy onset is still inside train_patience.

        Uses the latest unhandled onset from session-health ``onsets[]`` —
        not a sticky first-on-tape mark.
        """
        path = f"/api/instances/{self._tm_agent_id()}/session-health"
        health = self.hb.get_json(path, timeout_s=3.0)
        if not health:
            return False, None
        raw_onsets = health.get("onsets")
        onsets: list[dict] = (
            [o for o in raw_onsets if isinstance(o, dict)]
            if isinstance(raw_onsets, list)
            else []
        )
        if not onsets and health.get("onset_version") is not None:
            onsets = [{"version": health.get("onset_version")}]
        handled = self._handled_onset_version
        patience = int(self._train_patience)
        for o in onsets:
            try:
                onset_i = int(o["version"])
            except (TypeError, ValueError, KeyError):
                continue
            if handled is not None and onset_i <= handled:
                continue
            age = max(0, int(self._traj) - onset_i)
            if age <= patience:
                return True, onset_i
        return False, None

    def _apply_stock_command(self, command_id: int) -> None:
        """Swap train (+ probe) stock target to ``command_id``; reset patience."""
        pick = int(command_id)
        cl = self.cfg.setdefault("closed_loop", {})
        cl["command_id"] = pick
        tgt = dict(cl.get("target") or {})
        tgt["kind"] = "stock"
        cl["target"] = tgt
        self.command_id = pick
        self.command_ids = np.full(self.B, self.command_id, dtype=np.int64)
        self.target = make_target(self.cfg, batch_size=self.B)

        probe_cfg = copy.deepcopy(self.cfg)
        probe_tgt = dict(probe_cfg["closed_loop"].get("target") or {})
        probe_tgt["kind"] = probe_tgt.get("kind") or "stock"
        probe_cfg["closed_loop"]["target"] = probe_tgt
        probe_pool = [int(c) for c in STOCK_COMMAND_IDS if int(c) != pick]
        if probe_pool:
            probe_cfg["closed_loop"]["command_id"] = int(
                probe_pool[(int(self._remix_count) + 1) % len(probe_pool)]
            )
        self._probe_target = make_target(probe_cfg, batch_size=self.B)

        self._remix_count += 1
        self._stale = 0
        self._best_ink_miss = None
        self._best_probe = None
        self._last_ink_miss = None
        self._last_probe = None

    def _remix_training_data(self) -> None:
        """Generate a different stock target mix (new terrain) and reset patience."""
        cl = self.cfg.setdefault("closed_loop", {})
        cur = int(cl.get("command_id", self.command_id))
        pool = [int(c) for c in STOCK_COMMAND_IDS if int(c) != cur]
        if not pool:
            pool = [int(c) for c in STOCK_COMMAND_IDS]
        # Deterministic walk through stock mix, seeded by remix count + traj.
        pick = pool[(int(self._remix_count) + int(self._traj)) % len(pool)]
        self._apply_stock_command(pick)
        _emit(
            f"es-remix:command_id={self.command_id} n={self._remix_count}",
            traj=self._traj,
        )

    def _apply_stock_from_feed(self, config: dict | None) -> bool:
        """Prefer TM stock plate ``command_id`` over local remix. True if applied."""
        if not isinstance(config, dict):
            return False
        feed = config.get("feed") if isinstance(config.get("feed"), dict) else {}
        raw = feed.get("command_id")
        if raw is None:
            return False
        try:
            cid = int(raw)
        except (TypeError, ValueError):
            return False
        if cid not in set(STOCK_COMMAND_IDS):
            return False
        self._apply_stock_command(cid)
        self._remix_after_restore = False
        self._es_park_hold = False
        _emit(
            f"es-feed-stock:command_id={self.command_id} n={self._remix_count}",
            traj=self._traj,
        )
        return True

    def _consume_terrain_remix(self, *, reason: str) -> None:
        """Remix once per restore cycle; clear park so training continues."""
        if not self._remix_data_on_es or not self._remix_after_restore:
            return
        self._remix_after_restore = False
        try:
            self._remix_training_data()
        except Exception as exc:  # noqa: BLE001
            _emit(f"es-remix:fail:{exc}", traj=self._traj)
            return
        # Restore cycle done — do not leave the run parked under Autopilot.
        self._es_park_hold = False
        _emit(f"es-remix:via={reason}", traj=self._traj)

    def _maybe_es_shadow(self) -> None:
        if not self._es_shadow:
            return
        if self._traj < self._es_min_traj:
            return
        patience_hit = self._stale >= self._train_patience
        onset_hit, onset_ver = self._onset_within_patience()
        new_onset = (
            onset_hit
            and onset_ver is not None
            and (
                self._handled_onset_version is None
                or int(onset_ver) > int(self._handled_onset_version)
            )
        )
        # While tripped, only a brand-new onset re-arms (ink improve clears trip).
        if self._es_tripped and not new_onset:
            return
        if not patience_hit and not onset_hit:
            return
        self._es_tripped = True
        if onset_ver is not None:
            self._handled_onset_version = int(onset_ver)
        trip = "es-onset" if onset_hit and not patience_hit else "es-trip"
        self._set_status(trip, loss=self._last_loss)

        # Manual Start (no Autopilot): ES ends the run. Brain is informational
        # only — never restore/retune/park claimed (Resume → "cannot resume
        # job in state claimed" was the bug).
        if not self._autopilot:
            self._manual_es_inform_and_finish()
            return

        # BL-024 / BL-027: Autopilot continuous — publish trip + soft hold +
        # local engine pause. TM ES driver sees metrics.state (es-onset/es-trip),
        # runs restore+plate, and resumes via control commands. No /work/pause
        # and no /tm-brain/act (train blew 15s). Soft hold must not release the
        # lease; train_tick no-ops while _es_park_hold.
        self._es_park_hold = True
        pause = getattr(self, "_engine_pause", None)
        if callable(pause) and bool(getattr(self.hb, "job_bound", False)):
            try:
                pause()
            except Exception:  # noqa: BLE001
                _emit("es-stop:engine_pause_fail", traj=self._traj)
        if self._es_apply and self._remix_data_on_es:
            # Arm remix; cleared/consumed on after_restore if feed stock applied.
            self._remix_after_restore = True
        _emit("es-stop:await_tm", traj=self._traj)

    def _manual_es_inform_and_finish(self) -> None:
        """Manual Start + ES: end the job. No tm-brain/act (that logs durable
        episodes the UI reads as live steering). Worker emits would-line only
        if we ever add a read-only shadow later — for now stay quiet.
        """
        self._remix_after_restore = False
        self._es_park_hold = False
        self._es_run_done = True
        self._paused = False
        self._set_status("es-stop:manual", loss=self._last_loss)
        stop = self._engine_stop
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001
                _emit("es-stop:engine_stop_fail", traj=self._traj)
        else:
            _emit("es-stop:no_engine_stop", traj=self._traj)
        # Training finished → release the claim (ack).
        ack = getattr(self.hb, "ack_work", None)
        if callable(ack) and bool(getattr(self.hb, "job_bound", False)):
            try:
                out = ack(
                    result={
                        "ok": True,
                        "reason": "es_manual_stop",
                        "model_id": self._tm_agent_id(),
                    }
                )
                if out is not None:
                    jid = out.get("job_id") or getattr(self.hb, "job_id", None)
                    _emit(
                        f"claim-released:job={jid} reason=es_manual_stop",
                        traj=self._traj,
                    )
                else:
                    _emit("es-stop:ack_rejected", traj=self._traj)
            except Exception as exc:  # noqa: BLE001
                _emit(f"es-stop:ack_fail:{exc}", traj=self._traj)
        elif not bool(getattr(self.hb, "job_bound", False)):
            _emit("es-stop:already_unbound", traj=self._traj)
        else:
            _emit("es-stop:no_ack_work", traj=self._traj)

    def _maybe_label_outcome(self) -> None:
        if not self._pending_outcome_episode or self._outcome_due_traj is None:
            return
        if self._traj < self._outcome_due_traj:
            return
        # Positive outcome if probe improved vs best-at-trip baseline.
        baseline = self._best_probe
        now = self._last_probe
        if baseline is None or now is None:
            outcome = 0.0
        else:
            outcome = float(baseline - now)  # improvement => positive
        self.hb.post_json(
            f"/api/tm-brain/episodes/{self._pending_outcome_episode}/outcome",
            {"outcome": outcome},
            timeout_s=5.0,
        )
        self._pending_outcome_episode = None
        self._outcome_due_traj = None

    def _train_one(self) -> float:
        # Pause/cancel arrive via TrainingEngine.drain_manager_commands before
        # train_tick; holds are checked there. This method is the traj body only.
        result = self.app.trainer.rollout_train(
            command_ids=self.command_ids,
            target=self.target,
            max_steps=self.app.max_steps,
            lr=self.lr,
            apply_updates=True,
        )
        self._traj += 1
        self._last_loss = float(result.total_loss)
        canvas = self.app.env.canvas
        if canvas is not None:
            try:
                self._last_ink_miss = float(
                    self.app.loss_fn.ink_miss(canvas, self.target)
                )
            except Exception:  # noqa: BLE001
                self._last_ink_miss = None
            try:
                self._last_probe = float(
                    self.app.loss_fn.ink_miss(canvas, self._probe_target)
                )
            except Exception:  # noqa: BLE001
                self._last_probe = None
        # Patience / ES on train metric (ink_miss); probe is for outcomes.
        ink = self._last_ink_miss
        if ink is not None:
            if self._best_ink_miss is None or ink < self._best_ink_miss - 1e-6:
                self._best_ink_miss = float(ink)
                self._stale = 0
                self._es_tripped = False
            else:
                self._stale += 1
        if self._last_probe is not None:
            if self._best_probe is None or self._last_probe < self._best_probe:
                self._best_probe = float(self._last_probe)
        self._append_step()
        self._maybe_checkpoint()
        self._maybe_es_shadow()
        self._maybe_label_outcome()
        return self._last_loss


DrawingExpertAgent = DrawStudentAgent


def main() -> int:
    from examples.closed_loop_draw.run_lease import build_closed_loop_engine

    args = _parse_args()
    cfg = load_config(args.config)
    out_dir = Path(
        (cfg.get("meta") or {}).get("output_dir")
        or "diagnostics_output/closed_loop_draw_agent"
    )
    if not out_dir.is_absolute():
        out_dir = Path(_REPO) / out_dir
    log_path = (
        Path(args.log_file).expanduser()
        if args.log_file
        else out_dir / "draw-student.log"
    )
    _open_log_file(log_path)
    agent = DrawStudentAgent(cfg)
    engine = build_closed_loop_engine(
        agent,
        model_instance_id=str(agent.hb.cfg.instance_id),
        ledger_dir=out_dir / "engine_ledger",
        claim_inside_engine=True,
    )
    try:
        engine.run()
    finally:
        engine.close()
        agent.close()
        if _LOG_FP is not None:
            _LOG_FP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
