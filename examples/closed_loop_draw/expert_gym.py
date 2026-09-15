# Headless gym: curriculum of shapes + configs → expert steer → outcome → optional train.
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from examples.closed_loop_draw.assemble import load_config
from examples.closed_loop_draw.commands import COMMANDS, STOCK_COMMAND_IDS

# Shape difficulty (stock only). Harder shapes need more strokes / closure.
COMPLEXITY_POOLS: dict[str, tuple[int, ...]] = {
    "easy": (0, 1),  # circle, line
    "medium": (2, 4),  # square, ring
    "hard": (3,),  # cross
}


def _post_json(url: str, body: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
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


def _pick_complexity(rng: np.random.Generator, name: str | None) -> str:
    if name and name in COMPLEXITY_POOLS:
        return name
    # Mild curriculum bias: more easy/medium than hard.
    return str(rng.choice(["easy", "easy", "medium", "medium", "hard"]))


def _jitter_config(
    cfg: dict[str, Any],
    rng: np.random.Generator,
    *,
    complexity: str,
) -> dict[str, Any]:
    """Vary knobs the drawing expert owns — diverse states for L0→L1 episodes."""
    out = json.loads(json.dumps(cfg))
    cl = out.setdefault("closed_loop", {})
    opt = out.setdefault("optimization", {})

    # Complexity-tied effort: harder shapes get more steps / slightly higher continuity.
    base_steps = {"easy": 8, "medium": 12, "hard": 16}[complexity]
    cl["max_steps"] = int(base_steps + int(rng.integers(-2, 3)))
    cl["max_steps"] = max(6, min(20, int(cl["max_steps"])))

    cl["sigma"] = float(rng.choice([0.04, 0.05, 0.06, 0.08, 0.10]))
    cl["continuity_weight"] = float(
        rng.choice([0.0, 0.01, 0.05, 0.1, 0.2])
        if complexity != "easy"
        else rng.choice([0.0, 0.01, 0.05])
    )
    # Sometimes start "broken" (catch-fail style): high continuity + blur off-ish.
    if float(rng.random()) < 0.35:
        cl["continuity_weight"] = float(rng.choice([0.1, 0.2, 0.3]))
        cl["loss_edt_weight"] = float(rng.choice([0.0, 0.5, 1.0]))
        cl["loss_edt_sym_weight"] = float(cl["loss_edt_weight"])
    else:
        cl["loss_edt_weight"] = float(rng.choice([0.5, 1.0, 1.0]))
        cl["loss_edt_sym_weight"] = float(cl["loss_edt_weight"])
        cl["loss_edt_soft_tau"] = float(rng.choice([1.0, 2.0, 3.0]))

    opt["learning_rate"] = float(rng.choice([3e-4, 5e-4, 7e-4, 1e-3]))
    cl["batch_size"] = min(int(cl.get("batch_size", 4)), 4)
    return out


def run_one(
    *,
    cfg: dict[str, Any],
    tm_uri: str,
    instance_id: str,
    pre_traj: int,
    post_traj: int,
    seed: int,
    command_id: int | None,
    complexity: str | None,
) -> dict[str, Any]:
    from examples.closed_loop_draw.gym_domain import GymTask, get_gym_domain

    domain = get_gym_domain("drawing_expert")
    task = domain.sample(seed=seed, complexity=complexity)
    if command_id is not None:
        cid = int(command_id)
        task = GymTask(
            domain=task.domain,
            seed=task.seed,
            complexity=task.complexity,
            command_id=cid,
            meta={"command": COMMANDS.get(cid, str(cid))},
        )
    return domain.run(
        task,
        cfg=cfg,
        tm_uri=tm_uri,
        instance_id=instance_id,
        pre_traj=pre_traj,
        post_traj=post_traj,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Drawing expert curriculum gym (shapes × configs → TM episodes)"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/closed_loop_draw/config_draw_interactive.yaml"),
    )
    p.add_argument("--tm-uri", default="http://127.0.0.1:8000")
    p.add_argument("--instance-id", default="expert-gym")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--pre-traj", type=int, default=15)
    p.add_argument("--post-traj", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--command-id", type=int, default=None)
    p.add_argument(
        "--complexity",
        choices=("easy", "medium", "hard"),
        default=None,
        help="Force one tier; default mixes easy/medium/hard",
    )
    p.add_argument("--train-after", action="store_true", help="POST /train when done")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    cl = cfg.setdefault("closed_loop", {})
    cl["batch_size"] = min(int(cl.get("batch_size", 4)), 4)

    print(
        f"[expert-gym] tm={args.tm_uri} episodes={args.episodes} "
        f"pre={args.pre_traj} post={args.post_traj} complexity={args.complexity or 'mixed'}",
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
                complexity=args.complexity,
            )
        except urllib.error.URLError as exc:
            print(f"[expert-gym] TM unreachable: {exc}", file=sys.stderr)
            return 2
        rows.append(row)
        print(
            f"  ep{i+1}: {row['complexity']}/{row['command']} "
            f"ink {row['ink_pre']:.3f}→{row['ink_post']:.3f} "
            f"Δ={row['outcome']:+.4f} auth={row.get('authority')} "
            f"applied={row.get('applied')}",
            flush=True,
        )

    if args.train_after:
        try:
            out = _post_json(
                f"{args.tm_uri.rstrip('/')}/api/experts/drawing/train",
                {"min_outcome": 0.0},
            )
            print(f"[expert-gym] trained: {out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[expert-gym] train failed: {exc}", file=sys.stderr)
            return 3

    n_pos = sum(1 for r in rows if r["outcome"] > 0)
    by_tier: dict[str, int] = {}
    for r in rows:
        by_tier[str(r["complexity"])] = by_tier.get(str(r["complexity"]), 0) + 1
    print(
        f"[expert-gym] done  n={len(rows)} positive_outcomes={n_pos} tiers={by_tier}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
