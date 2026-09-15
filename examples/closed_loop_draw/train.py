# examples/closed_loop_draw/train.py
"""
Closed-loop draw training app.

Usage (repo root):
  .venv/bin/python -m examples.closed_loop_draw.train
  .venv/bin/python -m examples.closed_loop_draw.train \\
      --config examples/closed_loop_draw/config_draw_train.yaml \\
      --trajectories 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from examples.closed_loop_draw.assemble import (
    assemble,
    load_config,
    make_target,
    rollout_forward,
)
from examples.closed_loop_draw.viz import ascii_preview, save_canvas_npy, save_canvas_png
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train closed-loop draw (CNN→MHSA→canvas)")
    p.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "config_draw_train.yaml"
        ),
        help="YAML config path",
    )
    p.add_argument("--trajectories", type=int, default=None, help="Override trajectory count")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--max-steps", type=int, default=None, help="Override stroke steps")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    p.add_argument("--log-every", type=int, default=None, help="Override loss log interval")
    p.add_argument("--dump-every", type=int, default=None, help="Override canvas dump interval")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override dump dir (loss.jsonl, canvases)",
    )
    p.add_argument("--no-dump", action="store_true", help="Skip canvas PNG/NPY dumps")
    p.add_argument("--ascii", action="store_true", help="Print ASCII canvas preview on dump")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    bootstrap_im2col_gemm_runtime()
    cfg = load_config(args.config)

    opt = cfg.setdefault("optimization", {})
    cl = cfg.setdefault("closed_loop", {})
    if args.trajectories is not None:
        opt["trajectories"] = int(args.trajectories)
    if args.lr is not None:
        opt["learning_rate"] = float(args.lr)
    if args.batch_size is not None:
        cl["batch_size"] = int(args.batch_size)
    if args.max_steps is not None:
        cl["max_steps"] = int(args.max_steps)
    if args.seed is not None:
        opt["seed"] = int(args.seed)
    if args.log_every is not None:
        opt["log_every"] = int(args.log_every)
    if args.dump_every is not None:
        opt["dump_every"] = int(args.dump_every)

    seed = int(opt.get("seed", 0))
    n_traj = int(opt.get("trajectories", 100))
    log_every = int(opt.get("log_every", 10))
    dump_every = int(opt.get("dump_every", 50))
    out_dir = Path(
        args.output_dir
        or cfg.get("meta", {}).get("output_dir", "diagnostics_output/closed_loop_draw")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_path = out_dir / "loss.jsonl"

    app = assemble(cfg, seed=seed)
    B = app.batch_size
    lr = app.lr
    command_id = int(cl.get("command_id", 0))
    command_ids = np.full(B, command_id, dtype=np.int64)
    target = make_target(cfg, batch_size=B)

    # Initial canvas dump (empty policy rollout).
    if not args.no_dump:
        canvas0 = rollout_forward(app, command_ids=command_ids)
        save_canvas_png(out_dir / "canvas_step0000.png", canvas0)
        save_canvas_npy(out_dir / "canvas_step0000.npy", canvas0)
        save_canvas_png(out_dir / "target.png", target)
        if args.ascii:
            print("=== target ===")
            print(ascii_preview(target))
            print("=== rollout @0 ===")
            print(ascii_preview(canvas0))

    print(
        f"[draw-train] trajectories={n_traj} batch={B} max_steps={app.max_steps} "
        f"lr={lr} out={out_dir}"
    )
    losses: list[float] = []
    t0 = time.perf_counter()
    try:
        with open(loss_path, "w", encoding="utf-8") as loss_f:
            for i in range(1, n_traj + 1):
                result = app.trainer.rollout_train(
                    command_ids=command_ids,
                    target=target,
                    max_steps=app.max_steps,
                    lr=lr,
                    apply_updates=True,
                )
                losses.append(float(result.total_loss))
                rec = {
                    "traj": i,
                    "loss": float(result.total_loss),
                    "step_losses": [float(x) for x in result.extras.get("step_losses", [])],
                }
                loss_f.write(json.dumps(rec) + "\n")
                loss_f.flush()

                if i % log_every == 0 or i == 1 or i == n_traj:
                    window = losses[-log_every:]
                    print(
                        f"  traj {i:4d}/{n_traj}  loss={result.total_loss:.6f}  "
                        f"avg{log_every}={float(np.mean(window)):.6f}"
                    )

                if not args.no_dump and (i % dump_every == 0 or i == n_traj):
                    canvas = rollout_forward(app, command_ids=command_ids)
                    tag = f"{i:04d}"
                    save_canvas_png(out_dir / f"canvas_step{tag}.png", canvas)
                    save_canvas_npy(out_dir / f"canvas_step{tag}.npy", canvas)
                    if args.ascii:
                        print(f"=== rollout @{i} ===")
                        print(ascii_preview(canvas))
    finally:
        app.close()

    elapsed = time.perf_counter() - t0
    summary = {
        "trajectories": n_traj,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_best": float(min(losses)) if losses else None,
        "elapsed_s": elapsed,
        "config": str(args.config),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[draw-train] done  first={summary['loss_first']:.6f}  "
        f"last={summary['loss_last']:.6f}  best={summary['loss_best']:.6f}  "
        f"({elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
