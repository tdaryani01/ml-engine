# examples/closed_loop_draw/agent.py
"""
Draw student: closed-loop draw worker under Training Manager dial-out.

Registers, trains while desired=running, appends step/checkpoint docs with
weight blobs + config snapshots, honors pause/resume/shutdown/restore.
On ES / first physiology onset: POST tm-brain/act — rules decide (model shadows),
actuators run (pause→restore→resume+config, retune, or park_stop). Episodes
+ outcomes are logged for brain training. Start/resume ``config`` is applied
and confirmed in the command ack.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from examples.closed_loop_draw.assemble import assemble, load_config, make_target
from examples.closed_loop_draw.draw_checkpoint import (
    apply_checkpoint_blob,
    build_checkpoint_blob,
    load_checkpoint_blob,
)
from src.manager_heartbeat import maybe_from_settings
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

    def __init__(self, cfg: dict) -> None:
        bootstrap_im2col_gemm_runtime()
        seed = int(cfg.get("optimization", {}).get("seed", 0))
        self.cfg = cfg
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
        brain = cfg.get("tm_brain") or {}
        self._es_shadow = bool(brain.get("shadow_on_es", True))
        # Apply rule decisions (model still shadows inside decide). Default on.
        self._es_apply = bool(brain.get("apply_on_es", True))

        tm = cfg.get("training_manager") or {}
        caps = list(tm.get("capabilities") or [])
        if "restore" not in caps:
            caps.append("restore")
            tm = dict(tm)
            tm["capabilities"] = caps
            cfg["training_manager"] = tm
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

    def close(self) -> None:
        self.app.close()

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
        blob_key = f"ckpts/{self.hb.cfg.instance_id}/v{version}.pkl"
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

    def _restore_from_command(self, cmd) -> None:
        """Restore weights only. Config changes arrive on the following start/resume."""
        payload = dict(cmd.payload or {})
        blob_key = payload.get("blob_key")
        version = payload.get("version")
        if not blob_key:
            self.hb.queue_ack(cmd.id, ok=False, detail="missing blob_key")
            return
        data = self.hb.fetch_blob(str(blob_key))
        if data is None:
            self.hb.queue_ack(cmd.id, ok=False, detail=f"blob not found: {blob_key}")
            return
        try:
            body = load_checkpoint_blob(data)
            apply_checkpoint_blob(self.app, body)  # weights; knobs ignored here
        except Exception as exc:  # noqa: BLE001
            self.hb.queue_ack(cmd.id, ok=False, detail=f"restore failed: {exc}")
            return
        if version is not None:
            self._checkpoint_version = int(version)
        self._stale = 0
        self._es_tripped = False
        self.hb.set_active_checkpoint(
            {
                "version": int(version) if version is not None else None,
                "blob_key": str(blob_key),
            }
        )
        detail = f"restored_weights:v{version}"
        self.hb.queue_ack(cmd.id, ok=True, detail=detail)
        _emit(detail, loss=self._last_loss, traj=self._traj)

    def _handle_start_resume(self, cmd) -> None:
        payload = dict(cmd.payload or {})
        src = str(payload.get("source") or "").strip().lower()
        # Human Pause must stick: ignore brain resume/start until user clears hold.
        if self._user_pause_hold and src == "tm_brain":
            self._paused = True
            self.hb.set_desired_state("paused")
            self.hb.mark_command_seen(cmd.id)
            self.hb.queue_ack(
                cmd.id,
                ok=False,
                detail="blocked:user_paused",
            )
            _emit(
                "blocked:user_paused",
                loss=self._last_loss,
                traj=self._traj,
            )
            return
        if src != "tm_brain":
            self._user_pause_hold = False
        confirmed = self._apply_config_payload(
            payload.get("config") if isinstance(payload.get("config"), dict) else None
        )
        self._run_authorized = True
        self._paused = False
        self._es_park_hold = False
        self.hb.set_desired_state("running")
        self.hb.mark_command_seen(cmd.id)
        live = self._live_config()
        if confirmed:
            self._print_config("confirmed", confirmed)
        else:
            self._print_config("live", live)
        ack = {
            "ok": True,
            "action": cmd.action,
            "phase": payload.get("phase"),
            "episode_id": payload.get("episode_id"),
            "config_requested": payload.get("config") or {},
            "config_confirmed": confirmed or live,
        }
        self.hb.queue_ack(cmd.id, ok=True, detail=json.dumps(ack, sort_keys=True))
        _emit(
            f"{cmd.action.strip().lower()}:config_ok",
            loss=self._last_loss,
            traj=self._traj,
        )

    def _drain_commands(self) -> None:
        hb = self.hb
        for cmd in hb.poll_commands():
            action = cmd.action.strip().lower()
            try:
                if action in ("start", "resume"):
                    self._handle_start_resume(cmd)
                elif action == "pause":
                    payload = dict(cmd.payload or {})
                    src = str(payload.get("source") or "").strip().lower()
                    # Non-brain pause (UI / shadow park) arms the hold.
                    if src != "tm_brain":
                        self._user_pause_hold = True
                    self._paused = True
                    hb.set_desired_state("paused")
                    hb.mark_command_seen(cmd.id)
                    phase = payload.get("phase")
                    hb.queue_ack(
                        cmd.id,
                        ok=True,
                        detail=f"paused:{phase}" if phase else "paused",
                    )
                    _emit(
                        f"paused:{phase}" if phase else "paused",
                        loss=self._last_loss,
                        traj=self._traj,
                    )
                elif action == "cancel":
                    self._user_pause_hold = False
                    self._run_authorized = False
                    self._paused = False
                    self._es_park_hold = False
                    hb.set_desired_state("idle")
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True, detail="cancelled")
                    _emit("cancelled", loss=self._last_loss, traj=self._traj)
                elif action == "shutdown":
                    created = float(cmd.created_at or 0.0)
                    if created > 0.0 and created < (self._process_started_at - 2.0):
                        hb.mark_command_seen(cmd.id)
                        hb.queue_ack(
                            cmd.id, ok=True, detail="ignored_stale_pre_boot_shutdown"
                        )
                        continue
                    self._shutdown_accepted = True
                    self._stop = True
                    hb.set_desired_state("stopped")
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "restore":
                    hb.mark_command_seen(cmd.id)
                    self._restore_from_command(cmd)
                else:
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=False, detail=f"unknown action {action}")
            except Exception as exc:  # noqa: BLE001
                hb.mark_command_seen(cmd.id)
                hb.queue_ack(cmd.id, ok=False, detail=str(exc))

    def _apply_desired(self) -> None:
        desired = (self.hb.desired_state or "").strip().lower()
        if desired == "stopped":
            if self._shutdown_accepted:
                self._stop = True
            return
        if desired == "paused":
            self._paused = True
        elif desired == "running":
            # ES park / human Pause must stick until an explicit start/resume command.
            # Heartbeat otherwise re-applies TM sticky desired=running from
            # the earlier Autopilot Start and unpauses without a decision.
            if self._es_park_hold or self._user_pause_hold:
                self._paused = True
                self.hb.set_desired_state("paused")
                return
            if self._run_authorized:
                self._paused = False
        elif desired in ("idle", ""):
            if not self._es_park_hold and not self._user_pause_hold:
                self._paused = False

    def _allows_train(self) -> bool:
        if not self._run_authorized or self._paused or self._stop or self._es_park_hold:
            return False
        desired = (self.hb.desired_state or "").strip().lower()
        return desired == "running"

    def _onset_within_patience(self) -> tuple[bool, int | None]:
        """True when a *new* unhealthy onset is still inside train_patience.

        Uses the latest unhandled onset from session-health ``onsets[]`` —
        not a sticky first-on-tape mark.
        """
        path = f"/api/instances/{self.hb.cfg.instance_id}/session-health"
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

        # Rules decide + apply; model shadows inside decide. Episodes logged on TM.
        path = f"/api/instances/{self.hb.cfg.instance_id}/tm-brain/act"
        body = {
            "patience": float(self._train_patience),
            "lr": float(self.lr),
            "window": 32,
            "apply": True,
            "dry_run": False if self._es_apply else True,
        }
        out = self.hb.post_json(path, body, timeout_s=15.0)
        decision = out.get("decision") if isinstance(out, dict) else None
        actuation = out.get("actuation") if isinstance(out, dict) else None
        if isinstance(decision, dict):
            ep = decision.get("episode_id")
            if ep:
                self._pending_outcome_episode = str(ep)
                self._outcome_due_traj = self._traj + self._outcome_horizon
            action = decision.get("action")
            reason = decision.get("reason")
            model_a = decision.get("model_action")
            _emit(
                f"brain:{action} auth={decision.get('authority')} "
                f"model={model_a} ep={ep}",
                traj=self._traj,
            )
            if reason:
                _emit(f"brain-reason:{reason}", traj=self._traj)
            self._set_status(f"es-act:{action}", loss=self._last_loss)
        else:
            action = None
            self._set_status("es-act:fail", loss=self._last_loss)

        if isinstance(actuation, dict):
            seq = actuation.get("sequence") or actuation.get("would_steps")
            cfg = actuation.get("config")
            _emit(
                f"actuation:applied={actuation.get('applied')} "
                f"seq={seq} config={cfg}",
                traj=self._traj,
            )
            # Only hard-hold on park_stop; restore/retune resume via commands.
            if action == "park_stop" and actuation.get("applied"):
                self._es_park_hold = True
                self._paused = True
                self.hb.set_desired_state("paused")
        elif not self._es_apply:
            # Dry-run / shadow-only: park so a human can inspect.
            self._paused = True
            self._es_park_hold = True
            self.hb.set_desired_state("paused")
            self.hb.post_json(
                f"/api/instances/{self.hb.cfg.instance_id}/control/pause",
                {"payload": {"source": "es_shadow_only"}},
                timeout_s=5.0,
            )

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
        # Honor Pause that arrived while the previous traj was in flight.
        self._drain_commands()
        self._apply_desired()
        if not self._allows_train():
            return float(self._last_loss or 0.0)
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

    def run(self) -> None:
        idle_s = float(self.hb.cfg.idle_sleep_s)
        self._set_status("idle")
        while not self._stop:
            self._drain_commands()
            self._apply_desired()
            if self._stop:
                break
            if not self._allows_train():
                state = "paused" if self._paused else "idle"
                self._set_status(state)
                time.sleep(max(0.1, idle_s))
                continue
            loss = self._train_one()
            # ES parks inside _train_one; do not stamp "training" over that.
            if not self._allows_train():
                state = "paused" if self._paused else "idle"
                self._set_status(state, loss=loss)
                continue
            prev = self._last_status
            self._set_status("training", loss=loss)
            # Progress pings while status stays training.
            if prev == "training" and self._traj % 10 == 0:
                _emit("training", loss=loss, traj=self._traj)
        self._set_status("stopped", loss=self._last_loss)


DrawingExpertAgent = DrawStudentAgent


def main() -> int:
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
    try:
        agent.run()
    finally:
        agent.close()
        if _LOG_FP is not None:
            _LOG_FP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
