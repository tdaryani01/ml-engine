# Drawing-expert gym worker (BL-002 Slice B).
#
# Polls TM drawing-expert sticky gym settings + state; runs curriculum episodes
# while armed+training; stops between episodes on Pause; trains; idles when
# val_target + min_n_train are met.
#
# Does NOT re-register as drawing-expert (in-process TM already owns control
# apply/ack). Only heartbeats progress metrics (gym_round / gym_last_val).
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from examples.closed_loop_draw.assemble import load_config
from examples.closed_loop_draw.expert_gym import run_one

DRAWING_EXPERT_ID = "drawing-expert"


def _get_json(url: str, timeout_s: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, body: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def read_gym_from_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    m = metrics if isinstance(metrics, dict) else {}
    return {
        "armed": _bool(m.get("tm_gym_armed"), False),
        "enabled": _bool(m.get("tm_gym_enabled"), False),
        "user_paused": _bool(m.get("tm_user_paused"), False),
        "state": str(m.get("state") or "idle").strip().lower(),
        "batch": max(1, int(m.get("tm_gym_batch") or 16)),
        "pre_traj": max(1, int(m.get("tm_gym_pre_traj") or 10)),
        "post_traj": max(1, int(m.get("tm_gym_post_traj") or 10)),
        "val_target": float(m.get("tm_gym_val_target") or 0.70),
        "min_n_train": max(1, int(m.get("tm_gym_min_n_train") or 40)),
        "complexity": str(m.get("tm_gym_complexity") or "mixed"),
        "gym_round": int(m.get("gym_round") or 0),
        "gym_last_val": m.get("gym_last_val"),
        "val_accuracy": m.get("val_accuracy"),
        "n_train": m.get("n_train") or m.get("n_episodes"),
    }


def decide_gym_action(metrics: dict[str, Any] | None) -> str:
    """Pure policy: run | wait | stop_done.

    stop_done = armed but stop criteria already met (caller should disarm).
    """
    g = read_gym_from_metrics(metrics)
    if not g["armed"]:
        return "wait"
    if g["user_paused"] or g["state"] == "paused":
        return "wait"
    if g["state"] in ("down",):
        return "wait"
    # Stop criteria from last pulsed val / n_train when present.
    val = g["val_accuracy"] if g["val_accuracy"] is not None else g["gym_last_val"]
    n_train = g["n_train"]
    try:
        if (
            val is not None
            and n_train is not None
            and float(val) >= float(g["val_target"])
            and int(n_train) >= int(g["min_n_train"])
        ):
            return "stop_done"
    except (TypeError, ValueError):
        pass
    if g["state"] in ("training", "running"):
        return "run"
    # Armed but idle (e.g. after restore): wait for Start.
    return "wait"


def fetch_expert_metrics(tm_uri: str, instance_id: str = DRAWING_EXPERT_ID) -> dict[str, Any]:
    base = tm_uri.rstrip("/")
    rows = _get_json(f"{base}/api/instances")
    if isinstance(rows, list):
        hit = next((r for r in rows if r.get("id") == instance_id), None)
        if hit is None:
            raise RuntimeError(f"instance not found: {instance_id}")
        return dict(hit.get("metrics") or {})
    raise RuntimeError("unexpected /api/instances shape")


def pulse_gym_progress(
    tm_uri: str,
    *,
    instance_id: str = DRAWING_EXPERT_ID,
    gym_round: int,
    gym_last_val: float | None = None,
    state: str = "training",
    extra: dict[str, Any] | None = None,
) -> None:
    metrics: dict[str, Any] = {
        "state": state,
        "gym_round": int(gym_round),
    }
    if gym_last_val is not None:
        metrics["gym_last_val"] = float(gym_last_val)
    if extra:
        metrics.update(extra)
    _post_json(
        f"{tm_uri.rstrip('/')}/api/instances/{instance_id}/heartbeat",
        {"metrics": metrics},
    )


def disarm_gym(tm_uri: str, instance_id: str = DRAWING_EXPERT_ID) -> None:
    """Pause then cancel → idle + disarm sticky gym (Slice A control plane)."""
    base = tm_uri.rstrip("/")
    try:
        _post_json(f"{base}/api/instances/{instance_id}/control/pause", {})
    except urllib.error.HTTPError:
        pass
    _post_json(f"{base}/api/instances/{instance_id}/control/cancel", {})


def train_expert(tm_uri: str, *, min_outcome: float = 0.0) -> dict[str, Any]:
    return _post_json(
        f"{tm_uri.rstrip('/')}/api/experts/drawing/train",
        {"min_outcome": min_outcome},
        timeout_s=120.0,
    )


def run_worker_round(
    *,
    cfg: dict[str, Any],
    tm_uri: str,
    gym: dict[str, Any],
    seed0: int,
    episode_fn: Callable[..., dict[str, Any]] = run_one,
    instance_tag: str = "expert-gym-worker",
) -> list[dict[str, Any]]:
    """Run one gym batch (batch episodes). Pause is checked by caller between rounds."""
    rows: list[dict[str, Any]] = []
    complexity = gym["complexity"]
    if complexity == "mixed":
        complexity = None
    for i in range(int(gym["batch"])):
        row = episode_fn(
            cfg=cfg,
            tm_uri=tm_uri,
            instance_id=instance_tag,
            pre_traj=int(gym["pre_traj"]),
            post_traj=int(gym["post_traj"]),
            seed=int(seed0) + i,
            command_id=None,
            complexity=complexity,
        )
        rows.append(row)
    return rows


def worker_loop(
    *,
    tm_uri: str,
    cfg: dict[str, Any],
    poll_s: float = 2.0,
    seed0: int = 100,
    max_rounds: int = 10_000,
    episode_fn: Callable[..., dict[str, Any]] = run_one,
    fetch_fn: Callable[[str], dict[str, Any]] = fetch_expert_metrics,
    train_fn: Callable[[str], dict[str, Any]] = train_expert,
    pulse_fn: Callable[..., None] = pulse_gym_progress,
    disarm_fn: Callable[[str], None] = disarm_gym,
    sleep_fn: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Drive gym until disarmed, stop_done, max_rounds, or should_stop."""
    rounds = 0
    last_train: dict[str, Any] | None = None
    while rounds < max_rounds:
        if should_stop is not None and should_stop():
            return {"ok": True, "rounds": rounds, "reason": "should_stop", "last_train": last_train}
        metrics = fetch_fn(tm_uri)
        action = decide_gym_action(metrics)
        gym = read_gym_from_metrics(metrics)
        if action == "wait":
            sleep_fn(poll_s)
            continue
        if action == "stop_done":
            disarm_fn(tm_uri)
            return {
                "ok": True,
                "rounds": rounds,
                "reason": "val_target",
                "last_train": last_train,
                "gym": gym,
            }
        # run
        rounds += 1
        seed = seed0 + rounds * int(gym["batch"])
        print(
            f"[gym-worker] round={rounds} batch={gym['batch']} "
            f"val_target={gym['val_target']} min_n={gym['min_n_train']}",
            flush=True,
        )
        run_worker_round(
            cfg=cfg,
            tm_uri=tm_uri,
            gym=gym,
            seed0=seed,
            episode_fn=episode_fn,
        )
        # Re-check pause before train (Pause between episodes/rounds).
        metrics2 = fetch_fn(tm_uri)
        if decide_gym_action(metrics2) == "wait":
            pulse_fn(
                tm_uri,
                gym_round=rounds,
                gym_last_val=gym.get("gym_last_val"),
                state="paused",
            )
            sleep_fn(poll_s)
            continue
        try:
            last_train = train_fn(tm_uri)
        except Exception as exc:  # noqa: BLE001
            print(f"[gym-worker] train failed: {exc}", flush=True)
            sleep_fn(poll_s)
            continue
        val = last_train.get("val_accuracy")
        n_train = last_train.get("n_train")
        pulse_fn(
            tm_uri,
            gym_round=rounds,
            gym_last_val=None if val is None else float(val),
            state="training",
            extra={
                k: last_train[k]
                for k in ("val_accuracy", "n_train", "train_accuracy", "n_val")
                if k in last_train and last_train[k] is not None
            },
        )
        print(
            f"[gym-worker] trained v={last_train.get('version')} "
            f"n_train={n_train} val={val}",
            flush=True,
        )
        # Immediate stop check without waiting for next poll semantics.
        metrics3 = {
            **metrics2,
            "val_accuracy": val,
            "n_train": n_train,
            "gym_last_val": val,
            "tm_gym_armed": True,
            "state": "training",
        }
        # Keep sticky targets from gym.
        metrics3["tm_gym_val_target"] = gym["val_target"]
        metrics3["tm_gym_min_n_train"] = gym["min_n_train"]
        metrics3["tm_gym_armed"] = True
        if decide_gym_action(metrics3) == "stop_done":
            disarm_fn(tm_uri)
            return {
                "ok": True,
                "rounds": rounds,
                "reason": "val_target",
                "last_train": last_train,
            }
    return {"ok": True, "rounds": rounds, "reason": "max_rounds", "last_train": last_train}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TM drawing-expert gym worker (Slice B)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/closed_loop_draw/config_draw_interactive.yaml"),
    )
    p.add_argument("--tm-uri", default="http://127.0.0.1:8000")
    p.add_argument("--poll-s", type=float, default=2.0)
    p.add_argument("--seed0", type=int, default=100)
    p.add_argument("--max-rounds", type=int, default=10_000)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    cl = cfg.setdefault("closed_loop", {})
    cl["batch_size"] = min(int(cl.get("batch_size", 4)), 4)

    print(f"[gym-worker] tm={args.tm_uri} poll={args.poll_s}s", flush=True)
    out = worker_loop(
        tm_uri=args.tm_uri,
        cfg=cfg,
        poll_s=args.poll_s,
        seed0=args.seed0,
        max_rounds=args.max_rounds,
    )
    print(f"[gym-worker] exit {json.dumps({k: out.get(k) for k in ('ok','rounds','reason')})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
