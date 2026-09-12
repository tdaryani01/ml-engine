# Headless gym: random shape → train → expert steer → apply → train → outcome.
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from examples.closed_loop_draw.apply_deltas import apply_drawing_deltas
from examples.closed_loop_draw.assemble import assemble, load_config, make_target
from examples.closed_loop_draw.commands import STOCK_COMMAND_IDS


def _post_json(url: str, body: dict[str, Any], timeout_s: float = 10.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ink_miss(app, target: np.ndarray) -> float:
    canvas = app.env.canvas
    if canvas is None:
        obs = app.env.reset(int(target.shape[0]))
    else:
        obs = canvas
    return float(app.loss_fn.ink_miss(obs, target))


def _train_block(
    app,
    *,
    target: np.ndarray,
    command_ids: np.ndarray,
    n_traj: int,
    lr: float,
) -> tuple[float, float]:
    """Train n_traj steps; return (last_loss, ink_miss)."""
    last = 0.0
    for _ in range(max(1, int(n_traj))):
        result = app.trainer.rollout_train(
            command_ids=command_ids,
            target=target,
            max_steps=app.max_steps,
            lr=float(lr),
            apply_updates=True,
        )
        last = float(result.total_loss)
    return last, _ink_miss(app, target)


def run_one(
    *,
    cfg: dict[str, Any],
    tm_uri: str,
    instance_id: str,
    pre_traj: int,
    post_traj: int,
    seed: int,
    command_id: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cid = (
        int(command_id)
        if command_id is not None
        else int(rng.choice(list(STOCK_COMMAND_IDS)))
    )
    cfg = json.loads(json.dumps(cfg))  # deep copy via json
    cfg.setdefault("closed_loop", {})["command_id"] = cid
    cfg["closed_loop"]["target"] = {"kind": "stock"}

    app = assemble(cfg, seed=seed)
    try:
        B = app.batch_size
        target = make_target(cfg, batch_size=B)
        ids = np.full(B, cid, dtype=np.int64)
        lr = float(cfg.get("optimization", {}).get("learning_rate", app.lr))

        loss_pre, ink_pre = _train_block(
            app, target=target, command_ids=ids, n_traj=pre_traj, lr=lr
        )
        # Heuristic plateau if train is low but ink still high.
        plateau = bool(loss_pre < 0.02 and ink_pre > 0.15)

        suggest_body = {
            "instance_id": instance_id,
            "ink_miss": ink_pre,
            "loss": loss_pre,
            "trajs": pre_traj,
            "sigma": float(app.env.sigma),
            "max_steps": int(app.max_steps),
            "lr": float(lr),
            "continuity_weight": float(app.env.continuity_weight),
            "plateau": plateau,
            "closed": ink_pre < 0.2,
            "meta": {"command_id": cid, "seed": seed, "phase": "pre"},
        }
        suggest = _post_json(
            f"{tm_uri.rstrip('/')}/api/experts/drawing/suggest",
            suggest_body,
        )
        episode_id = str(suggest.get("episode_id") or "")
        deltas = list(suggest.get("deltas") or [])
        applied, lr = apply_drawing_deltas(app, deltas, train_lr=lr)

        loss_post, ink_post = _train_block(
            app, target=target, command_ids=ids, n_traj=post_traj, lr=float(lr or app.lr)
        )
        outcome = float(ink_pre - ink_post)
        if episode_id:
            _post_json(
                f"{tm_uri.rstrip('/')}/api/experts/drawing/episodes/{episode_id}/outcome",
                {"outcome": outcome},
            )
        return {
            "command_id": cid,
            "episode_id": episode_id,
            "authority": suggest.get("authority"),
            "applied": applied,
            "ink_pre": ink_pre,
            "ink_post": ink_post,
            "loss_pre": loss_pre,
            "loss_post": loss_post,
            "outcome": outcome,
            "summary": suggest.get("summary"),
        }
    finally:
        app.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Drawing expert interactive-style gym")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/closed_loop_draw/config_draw_interactive.yaml"),
    )
    p.add_argument("--tm-uri", default="http://127.0.0.1:8000")
    p.add_argument("--instance-id", default="expert-gym")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--pre-traj", type=int, default=20)
    p.add_argument("--post-traj", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--command-id", type=int, default=None)
    p.add_argument("--train-after", action="store_true", help="POST /train when done")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    # Smaller batches for gym speed if config is heavy.
    cl = cfg.setdefault("closed_loop", {})
    cl["batch_size"] = min(int(cl.get("batch_size", 4)), 4)

    print(
        f"[expert-gym] tm={args.tm_uri} episodes={args.episodes} "
        f"pre={args.pre_traj} post={args.post_traj}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for i in range(max(1, args.episodes)):
        try:
            row = run_one(
                cfg=cfg,
                tm_uri=args.tm_uri,
                instance_id=args.instance_id,
                pre_traj=args.pre_traj,
                post_traj=args.post_traj,
                seed=args.seed + i,
                command_id=args.command_id,
            )
        except urllib.error.URLError as exc:
            print(f"[expert-gym] TM unreachable: {exc}", file=sys.stderr)
            return 2
        rows.append(row)
        print(
            f"  ep{i+1}: cmd={row['command_id']} ink {row['ink_pre']:.3f}→{row['ink_post']:.3f} "
            f"Δ={row['outcome']:+.4f} auth={row.get('authority')} "
            f"applied={row.get('applied')}",
            flush=True,
        )

    if args.train_after:
        try:
            out = _post_json(f"{args.tm_uri.rstrip('/')}/api/experts/drawing/train", {})
            print(f"[expert-gym] trained: {out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[expert-gym] train failed: {exc}", file=sys.stderr)
            return 3

    n_pos = sum(1 for r in rows if r["outcome"] > 0)
    print(
        f"[expert-gym] done  n={len(rows)} positive_outcomes={n_pos}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
