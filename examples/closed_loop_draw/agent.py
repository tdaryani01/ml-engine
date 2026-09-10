# examples/closed_loop_draw/agent.py
"""
DrawingExpert: closed-loop draw trainer parked under Training Manager.

Registers as a fleet instance, idles until Start, trains trajectories while
desired=running, honors pause / resume / shutdown. Restore is acknowledged
as unsupported for now (weights live in-process only).

Usage (repo root; TM API must be up):
  .venv/bin/python -m examples.closed_loop_draw.agent
  .venv/bin/python -m examples.closed_loop_draw.agent \\
      --config examples/closed_loop_draw/config_draw_agent.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from examples.closed_loop_draw.assemble import assemble, load_config, make_target
from src.manager_heartbeat import maybe_from_settings
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TM-managed DrawingExpert agent")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config_draw_agent.yaml"),
        help="YAML config path",
    )
    return p.parse_args()


def _emit(state: str, *, loss: float | None = None, traj: int = 0) -> None:
    bits = [f"[DrawingExpert] status={state}"]
    if traj > 0:
        bits.append(f"traj={traj}")
    if loss is not None:
        bits.append(f"loss={loss:.4f}")
    print(" ".join(bits), flush=True)


class DrawingExpertAgent:
    """Headless closed-loop draw worker controlled by TM dial-out commands."""

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

        tm = cfg.get("training_manager") or {}
        self.hb = maybe_from_settings(tm, ledger_enabled=False)
        if self.hb is None:
            raise RuntimeError(
                "training_manager.enabled + uri required "
                "(see config_draw_agent.yaml)"
            )

        self._run_authorized = False
        self._paused = False
        self._stop = False
        self._shutdown_accepted = False
        self._traj = 0
        self._last_loss: float | None = None
        self._process_started_at = time.time()

    def close(self) -> None:
        self.app.close()

    def _publish(self, state: str) -> None:
        metrics: dict = {
            "state": state,
            "traj": self._traj,
            "command_id": self.command_id,
        }
        if self._last_loss is not None:
            metrics["loss"] = float(self._last_loss)
        self.hb.set_metrics(metrics)
        self.hb.maybe_ping(force=True)

    def _drain_commands(self) -> None:
        hb = self.hb
        for cmd in hb.poll_commands():
            action = cmd.action.strip().lower()
            try:
                if action in ("start", "resume"):
                    self._run_authorized = True
                    self._paused = False
                    hb.set_desired_state("running")
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "pause":
                    self._paused = True
                    hb.set_desired_state("paused")
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
                elif action == "cancel":
                    self._run_authorized = False
                    self._paused = False
                    hb.set_desired_state("idle")
                    hb.mark_command_seen(cmd.id)
                    hb.queue_ack(cmd.id, ok=True)
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
                    hb.queue_ack(
                        cmd.id,
                        ok=False,
                        detail="DrawingExpert restore not implemented yet",
                    )
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
            # Sticky running from a prior process must not auto-train.
            if self._run_authorized:
                self._paused = False
        elif desired in ("idle", ""):
            self._paused = False

    def _allows_train(self) -> bool:
        if not self._run_authorized or self._paused or self._stop:
            return False
        desired = (self.hb.desired_state or "").strip().lower()
        return desired == "running"

    def _train_one(self) -> float:
        result = self.app.trainer.rollout_train(
            command_ids=self.command_ids,
            target=self.target,
            max_steps=self.app.max_steps,
            lr=self.lr,
            apply_updates=True,
        )
        self._traj += 1
        self._last_loss = float(result.total_loss)
        return self._last_loss

    def run(self) -> None:
        idle_s = float(self.hb.cfg.idle_sleep_s)
        _emit("idle")
        self._publish("idle")
        while not self._stop:
            self._drain_commands()
            self._apply_desired()
            if self._stop:
                break
            if not self._allows_train():
                state = "paused" if self._paused else "idle"
                self._publish(state)
                time.sleep(max(0.1, idle_s))
                continue
            loss = self._train_one()
            self._publish("training")
            if self._traj == 1 or self._traj % 10 == 0:
                _emit("training", loss=loss, traj=self._traj)
        _emit("stopped", loss=self._last_loss, traj=self._traj)
        self._publish("stopped")


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    agent = DrawingExpertAgent(cfg)
    try:
        agent.run()
    finally:
        agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
