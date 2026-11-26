# examples/closed_loop_draw/interactive.py
"""
Interactive closed-loop draw window.

Stock commands: circle / line / square / cross / ring
Or draw on the sketch pad → Use as target → Train / Draw.

Usage (repo root, needs a display):
  .venv/bin/python -m examples.closed_loop_draw.interactive
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from examples.closed_loop_draw.goal import DrawGoal
from examples.closed_loop_draw.assemble import assemble, load_config
from examples.closed_loop_draw.commands import (
    COMMANDS,
    STOCK_COMMAND_IDS,
    ensure_stock_images,
    load_command_target,
    resolve_command,
)
from examples.closed_loop_draw.sketch_pad import SketchPad
from examples.closed_loop_draw.viz import canvas_to_u8
from utils.conv_dispatch import bootstrap_im2col_gemm_runtime



def _rollout_frames(app, ids):
    """Inference frames via the generic trainer (rollout_eval)."""
    return app.trainer.rollout_eval(goal=DrawGoal.render(ids)).extras["frames"]


class _BrainStopped(Exception):
    pass


class DrawInteractiveApp:
    def __init__(self, cfg_path: Path, brain: dict | None = None) -> None:
        bootstrap_im2col_gemm_runtime()
        self.cfg_path = Path(cfg_path)
        self.brain_args = brain
        self.brain_demo = None
        self.cfg = load_config(cfg_path)
        cl = self.cfg["closed_loop"]
        C, H, W = (int(x) for x in cl["canvas"])
        ensure_stock_images(height=H, width=W)
        cl["num_commands"] = max(int(cl.get("num_commands", 4)), len(COMMANDS))

        self.app = assemble(
            self.cfg, seed=int(self.cfg.get("optimization", {}).get("seed", 0))
        )
        self.H, self.W, self.C = H, W, C
        self.icfg = self.cfg.get("interactive", {})
        self.stroke_delay = float(self.icfg.get("stroke_delay_s", 0.25))
        self.train_n = int(self.icfg.get("train_trajectories_per_click", 40))
        self.train_until_close = bool(self.icfg.get("train_until_close", True))
        self.train_rel_tol = float(self.icfg.get("train_rel_tol", 0.08))
        self.train_abs_tol = float(self.icfg.get("train_abs_tol", 0.01))
        self.train_min_n = int(self.icfg.get("train_min_trajectories", 80))
        self.train_max_n = int(self.icfg.get("train_max_trajectories", 800))
        self.train_patience = int(self.icfg.get("train_patience", 120))
        self.train_improve_eps = float(self.icfg.get("train_improve_eps", 5e-4))
        self.train_lr = float(self.icfg.get("train_lr", self.app.lr))
        self.train_lr_decay = float(self.icfg.get("train_lr_decay", 0.8))
        self.train_lr_boosts = int(self.icfg.get("train_lr_boosts", 2))
        self.warmup_n = int(self.icfg.get("warmup_trajectories", 60))
        self.auto_warmup = bool(self.icfg.get("auto_warmup", True))
        self.preview_every = int(self.icfg.get("preview_every", 1))
        self.preview_pause_s = float(self.icfg.get("preview_pause_s", 0.04))

        self.command_id = int(cl.get("command_id", 0))
        self.command_name = COMMANDS[self.command_id]
        self.custom_target: np.ndarray | None = None  # (1,C,H,W) from sketch pad
        self.busy = False
        self._closing = False
        self._stop_requested = False
        self._did_warmup = False

        self.root = tk.Tk()
        self.root.title("Closed-loop draw")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._apply_command(self.command_name)
        if self.auto_warmup and not self.brain_args:      # the brain's Try builds a fresh model per run: the window's own warmup would train a model nothing uses
            self.root.after(200, lambda: self._run_action(self.warmup, label="auto-warmup"))

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        self.cmd_var = tk.StringVar(value=self.command_name)
        self.entry = ttk.Entry(top, textvariable=self.cmd_var, width=18)
        if not self.brain_args:
            ttk.Label(top, text="Command:").pack(side=tk.LEFT)
            self.entry.pack(side=tk.LEFT, padx=6)
            self.entry.bind("<Return>", lambda _e: self._on_set_command())
            ttk.Button(top, text="Set", command=self._on_set_command).pack(side=tk.LEFT, padx=2)
        else:      # --brain c4: draw anything on the pad; Set takes it as the target, Try lets the brain train to match it, Sleep lets the brain keep what it can trust
            ttk.Button(top, text="Set", command=self._use_sketch_as_target).pack(side=tk.LEFT, padx=2)
        self.btn_draw = ttk.Button(top, text="Draw", command=lambda: self._run_action(self.animate_draw))
        self.btn_train = ttk.Button(
            top, text="Auto-train", command=lambda: self._run_action(self.train_then_draw)
        )
        self.btn_warm = ttk.Button(top, text="Warmup", command=lambda: self._run_action(self.warmup))
        self.btn_stop = ttk.Button(top, text="Stop", command=self._request_stop)
        self.btn_brain = self.btn_sweep = self.btn_cont = self.btn_auto = None
        if self.brain_args:      # --brain c4: Train = one round steered by the brain that trains to match the drawing, continuing the one model from where the last round ended, whatever the drawing was (saved on disk), with a plain RESULT line saying whether it improved; Start over = forget the model; Sleep = the brain goes over the runs so far and keeps what it can trust
            self.btn_brain = ttk.Button(top, text="Train", command=lambda: self._run_action(lambda: self.brain_try(resume=True), label="train"))
            self.btn_sweep = ttk.Button(top, text="Sleep", command=lambda: self._run_action(self.brain_sleep, label="sleep"))
            self.btn_cont = ttk.Button(top, text="Start over", command=lambda: self._run_action(self.brain_start_over, label="start over"))
            self.auto_var = tk.BooleanVar(value=False)
            self.btn_auto = None      # Auto is a switch (a check button), not a button: it is never disabled while a round runs, so it can be turned off
            self.auto_switch = ttk.Checkbutton(top, text="Auto (random drawings)", variable=self.auto_var, command=self._on_auto_toggle)
            self.btn_brain.pack(side=tk.LEFT, padx=(14, 2))
            self.btn_sweep.pack(side=tk.LEFT, padx=2)
            self.btn_cont.pack(side=tk.LEFT, padx=2)
            self.auto_switch.pack(side=tk.LEFT, padx=(14, 2))
        else:
            self.btn_draw.pack(side=tk.LEFT, padx=4)
            self.btn_train.pack(side=tk.LEFT, padx=2)
            self.btn_warm.pack(side=tk.LEFT, padx=2)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        self.btn_stop.configure(state=tk.DISABLED)

        hint = ttk.Label(
            self.root,
            text=(
                "Closed-loop train (shape-agnostic loss)  |  "
                "stock: Set → Auto-train  |  sketch: draw → Use as target → Auto-train  |  "
                f"loss={self.app.loss_fn.kind}"
                + (
                    f"+blur{list(self.app.loss_fn.blur_sigmas)}"
                    + (
                        f" anneal→{[round(w, 2) for w in self.app.loss_fn.blur_weights_end]}"
                        if self.app.loss_fn.anneal_enabled
                        else ""
                    )
                    if any(s > 0 for s in self.app.loss_fn.blur_sigmas)
                    else ""
                )
                + (
                    f"+edt×{self.app.loss_fn.edt_weight:g}"
                    + (
                        f"+sym×{self.app.loss_fn.edt_sym_weight:g}"
                        if self.app.loss_fn.edt_sym_weight > 0
                        else ""
                    )
                    if self.app.loss_fn.edt_enabled
                    else ""
                )
                + (
                    f"+cont×{self.app.env.continuity_weight:g}"
                    if getattr(self.app.env, "continuity_weight", 0) > 0
                    else ""
                )
            ),
            padding=(8, 0),
        )
        hint.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="starting…")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 4)).pack(fill=tk.X)

        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.log_box = None
        if self.brain_args:
            self.log_box = tk.Text(self.root, height=9, wrap="word", state="disabled")
            self.log_box.pack(fill=tk.X, padx=8, pady=(0, 4))

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        self.sketch = SketchPad(
            body,
            out_h=self.H,
            out_w=self.W,
            channels=self.C,
            view_size=224,
            brush_radius=7,
        )
        self.sketch.pack(side=tk.LEFT, padx=(0, 12))
        self.sketch.set_use_command(self._use_sketch_as_target)
        if self.brain_args:      # Set does this; one button
            self.sketch.use_btn.pack_forget()

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7.2, 3.6), dpi=100)
        self.ax_target = self.fig.add_subplot(1, 2, 1)
        self.ax_canvas = self.fig.add_subplot(1, 2, 2)
        for ax, title in (
            (self.ax_target, "Target"),
            (self.ax_canvas, "Canvas (model)"),
        ):
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        self.im_target = self.ax_target.imshow(
            np.zeros((self.H, self.W)),
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        self.im_canvas = self.ax_canvas.imshow(
            np.zeros((self.H, self.W)),
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        self.fig.tight_layout()

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _pump(self) -> None:
        if not self._closing:
            self.root.update_idletasks()
            self.root.update()

    def _set_progress(self, frac: float) -> None:
        self.progress["value"] = float(np.clip(frac, 0.0, 1.0) * 100.0)

    def _show_canvas(self, nchw: np.ndarray) -> None:
        self.im_canvas.set_data(canvas_to_u8(nchw, sample=0))
        self.mpl_canvas.draw_idle()

    def _show_target(self, nchw: np.ndarray) -> None:
        self.im_target.set_data(canvas_to_u8(nchw, sample=0))
        self.mpl_canvas.draw_idle()

    def _preview_rollout(self) -> None:
        ids = np.full(1, self.command_id, dtype=np.int64)
        frame = None
        for frame in _rollout_frames(self.app, ids):
            pass
        if frame is not None:
            self._show_canvas(frame)

    def _target_batch(self, batch_size: int | None = None) -> np.ndarray:
        B = int(batch_size if batch_size is not None else self.app.batch_size)
        if self.command_name == "sketch":
            if self.custom_target is None:
                raise RuntimeError("No sketch target yet — draw and click Use as target")
            return np.repeat(self.custom_target, B, axis=0)
        return load_command_target(
            self.command_id,
            batch_size=B,
            height=self.H,
            width=self.W,
            channels=self.C,
        )

    def _use_sketch_as_target(self) -> None:
        if self.busy:
            self.status_var.set("busy — wait for current action")
            return
        if self.sketch.is_empty():
            messagebox.showinfo("Empty sketch", "Draw something on the pad first.")
            return
        self.custom_target = self.sketch.to_nchw(batch_size=1)
        self.command_id, self.command_name = resolve_command("sketch")
        self.cmd_var.set("sketch")
        self._show_target(self.custom_target)
        blank = np.zeros((1, self.C, self.H, self.W), dtype=np.float32)
        self._show_canvas(blank)
        ink = float(self.custom_target.mean())
        self.status_var.set(
            f"drawing set as target (ink={ink:.3f}) — press Train" if self.brain_args else f"sketch set as target (ink={ink:.3f}) — Auto-train (or Warmup)"
        )

    def _apply_command(self, text: str) -> None:
        cid, name = resolve_command(text)
        self.command_id = cid
        self.command_name = name
        self.cmd_var.set(name)
        if name == "sketch":
            if self.custom_target is None:
                blank = np.zeros((1, self.C, self.H, self.W), dtype=np.float32)
                self._show_target(blank)
                self.status_var.set("command=sketch — draw on the pad, then Use as target")
            else:
                self._show_target(self.custom_target)
                self.status_var.set("command=sketch (using your pad target)")
        else:
            target = self._target_batch(batch_size=1)
            self._show_target(target)
            self.status_var.set(f"command={name} (id={cid})")
        blank = np.zeros((1, self.C, self.H, self.W), dtype=np.float32)
        self._show_canvas(blank)

    def _on_set_command(self) -> None:
        try:
            self._apply_command(self.cmd_var.get())
        except ValueError as e:
            messagebox.showerror("Unknown command", str(e))

    def _request_stop(self) -> None:
        self._stop_requested = True
        self.status_var.set("stop requested — finishing current trajectory…")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for b in (self.btn_draw, self.btn_train, self.btn_warm, self.btn_brain, self.btn_sweep, self.btn_cont):
            if b is not None:
                b.configure(state=state)
        self.entry.configure(state=state)
        # Stop stays enabled while busy so the user can abort auto-train.
        self.btn_stop.configure(state=tk.NORMAL if busy else tk.DISABLED)
        try:
            self.sketch.use_btn.configure(state=state)
        except Exception:
            pass

    def _run_action(self, fn, *, label: str | None = None) -> None:
        if self.busy:
            self.status_var.set("busy — wait for current action to finish")
            return
        self._stop_requested = False
        self._set_busy(True)
        self._pump()
        try:
            fn()
        except Exception as e:
            self.status_var.set(f"error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self._stop_requested = False
            self._set_busy(False)
            self._set_progress(0.0)
            self._pump()

    def animate_draw(self) -> None:
        if self.command_name == "sketch" and self.custom_target is None:
            raise RuntimeError("Draw a sketch and click Use as target first")
        ids = np.full(1, self.command_id, dtype=np.int64)
        n = self.app.max_steps
        self.status_var.set(f"drawing '{self.command_name}' …")
        for i, frame in enumerate(_rollout_frames(self.app, ids)):
            if self._closing:
                return
            self._show_canvas(frame)
            self._set_progress(i / max(n, 1))
            self.status_var.set(f"drawing '{self.command_name}'  stroke {i}/{n}")
            self._pump()
            if i > 0:
                time.sleep(self.stroke_delay)
        tip = ""
        if not self._did_warmup:
            tip = "  (model still random — click Warmup or Train)"
        self.status_var.set(f"done drawing '{self.command_name}'{tip}")
        self._set_progress(1.0)

    def _empty_baseline_loss(self, target: np.ndarray) -> float:
        empty = np.zeros_like(target)
        t_last = self.app.max_steps - 1
        return float(self.app.loss_fn.step_loss(empty, target, t_last))

    def _snapshot_trainable(self) -> dict:
        """Deep-copy weights so we can roll back after a collapse."""
        mhsa = self.app.mhsa
        cnn = self.app.cnn
        snap = {
            "mhsa_w": [np.array(w, copy=True) for w in mhsa.weights],
            "mhsa_b": [np.array(b, copy=True) for b in mhsa.biases],
            "ln1_g": [np.array(g, copy=True) for g in mhsa.ln1_gamma],
            "ln1_b": [np.array(b, copy=True) for b in mhsa.ln1_beta],
            "ln2_g": [np.array(g, copy=True) for g in mhsa.ln2_gamma],
            "ln2_b": [np.array(b, copy=True) for b in mhsa.ln2_beta],
            "cnn_w": [np.array(w, copy=True) for w in cnn.weights],
            "cnn_b": [np.array(b, copy=True) for b in cnn.biases],
            "adapter_W": np.array(self.app.adapter.W, copy=True),
            "adapter_b": np.array(self.app.adapter.b, copy=True),
            "act_W": np.array(self.app.action_embed.W, copy=True),
            "act_b": np.array(self.app.action_embed.b, copy=True),
            "cond": np.array(self.app.conditioning.embeddings, copy=True),
        }
        if mhsa.pos_embed is not None:
            snap["pos"] = np.array(mhsa.pos_embed, copy=True)
        if mhsa.W_in is not None:
            snap["W_in"] = np.array(mhsa.W_in, copy=True)
            snap["b_in"] = np.array(mhsa.b_in, copy=True)
        return snap

    @staticmethod
    def _zero_like_list(bufs) -> None:
        if not bufs:
            return
        for x in bufs:
            if x is not None:
                np.asarray(x).fill(0.0)

    def _restore_trainable(self, snap: dict) -> None:
        mhsa = self.app.mhsa
        cnn = self.app.cnn
        for dst, src in zip(mhsa.weights, snap["mhsa_w"]):
            dst[...] = src
        for dst, src in zip(mhsa.biases, snap["mhsa_b"]):
            dst[...] = src
        for dst, src in zip(mhsa.ln1_gamma, snap["ln1_g"]):
            dst[...] = src
        for dst, src in zip(mhsa.ln1_beta, snap["ln1_b"]):
            dst[...] = src
        for dst, src in zip(mhsa.ln2_gamma, snap["ln2_g"]):
            dst[...] = src
        for dst, src in zip(mhsa.ln2_beta, snap["ln2_b"]):
            dst[...] = src
        for dst, src in zip(cnn.weights, snap["cnn_w"]):
            dst[...] = src
        for dst, src in zip(cnn.biases, snap["cnn_b"]):
            dst[...] = src
        self.app.adapter.W[...] = snap["adapter_W"]
        self.app.adapter.b[...] = snap["adapter_b"]
        self.app.action_embed.W[...] = snap["act_W"]
        self.app.action_embed.b[...] = snap["act_b"]
        self.app.conditioning.embeddings[...] = snap["cond"]
        if "pos" in snap and mhsa.pos_embed is not None:
            mhsa.pos_embed[...] = snap["pos"]
        if "W_in" in snap and mhsa.W_in is not None:
            mhsa.W_in[...] = snap["W_in"]
            mhsa.b_in[...] = snap["b_in"]

        mhsa.ensure_adam_moments()
        opt = mhsa.optimizer
        self._zero_like_list(getattr(opt, "ms_w", None))
        self._zero_like_list(getattr(opt, "vs_w", None))
        self._zero_like_list(getattr(opt, "ms_b", None))
        self._zero_like_list(getattr(opt, "vs_b", None))
        self._zero_like_list(getattr(opt, "ms_g", None))
        self._zero_like_list(getattr(opt, "vs_g", None))
        self._zero_like_list(getattr(opt, "ms_beta", None))
        self._zero_like_list(getattr(opt, "vs_beta", None))
        if getattr(opt, "t", None) is not None:
            opt.t = 0
        for name in ("_ms_pos", "_vs_pos", "_ms_W_in", "_vs_W_in", "_ms_b_in", "_vs_b_in"):
            buf = getattr(mhsa, name, None)
            if buf is not None:
                buf.fill(0.0)
        for ad in (self.app.adapter, self.app.action_embed):
            ad._mW.fill(0.0)
            ad._vW.fill(0.0)
            ad._mb.fill(0.0)
            ad._vb.fill(0.0)
            ad._t = 0
            ad.zero_grad()
        cond = self.app.conditioning
        cond._m.fill(0.0)
        cond._v.fill(0.0)
        cond._t = 0
        cond.zero_grad()
        self.app.encoder.zero_grad()

    def _close_enough(self, loss: float, empty_loss: float) -> bool:
        thr = max(self.train_abs_tol, self.train_rel_tol * max(empty_loss, 1e-8))
        return float(loss) <= thr

    def train_then_draw(self) -> None:
        if self.command_name == "sketch" and self.custom_target is None:
            raise RuntimeError("Draw a sketch and click Use as target first")
        B = self.app.batch_size
        ids = np.full(B, self.command_id, dtype=np.int64)
        target = self._target_batch(batch_size=B)
        loss_fn = self.app.loss_fn
        loss_fn.set_anneal(0.0)
        empty_loss = self._empty_baseline_loss(target)
        goal = max(self.train_abs_tol, self.train_rel_tol * max(empty_loss, 1e-8))
        lr = float(self.train_lr)
        anneal_on = bool(loss_fn.anneal_enabled)

        if self.train_until_close:
            max_n = self.train_max_n
            min_n = self.train_min_n
            self.status_var.set(
                f"auto-train '{self.command_name}' until loss≤{goal:.4f} "
                f"(blank={empty_loss:.4f} → ratio≤{self.train_rel_tol:.2f}, lr={lr:g}"
                f"{', blur-anneal' if anneal_on else ''})…"
            )
        else:
            max_n = self.train_n
            min_n = 1
            self.status_var.set(f"training '{self.command_name}' ×{max_n} …")
        self._pump()

        last = empty_loss
        best = float("inf")
        best_sharp = float("inf")
        best_snap: dict | None = None
        restored = False
        stale = 0
        boosts_left = self.train_lr_boosts
        stopped_reason = "max"
        ink_miss = 1.0
        i = 0
        t_last = self.app.max_steps - 1
        while i < max_n:
            if self._closing or self._stop_requested:
                stopped_reason = "stopped"
                break
            i += 1
            # Traj-fraction anneal: coarse catch early → sharp refine late.
            if anneal_on:
                u = 0.0 if max_n <= 1 else (i - 1) / (max_n - 1)
                loss_fn.set_anneal(u)
                empty_loss = self._empty_baseline_loss(target)
                goal = max(
                    self.train_abs_tol, self.train_rel_tol * max(empty_loss, 1e-8)
                )

            result = self.app.trainer.rollout_train(
                goal=DrawGoal(command_ids=ids, target=target),
                lr=lr,
                apply_updates=True,
            )
            last = float(result.total_loss)
            # Checkpoint on sharp loss so coarse-fat solutions don't win restore.
            canvas = self.app.env.canvas
            assert canvas is not None
            sharp = float(loss_fn.sharp_step_loss(canvas, target, t_last))
            if sharp < best_sharp - self.train_improve_eps:
                best_sharp = sharp
                best = last
                stale = 0
                best_snap = self._snapshot_trainable()
            else:
                stale += 1

            # Collapse guard: under anneal, compare sharp (scale-stable); else train loss.
            collapsed = False
            if best_snap is not None:
                if anneal_on:
                    collapsed = sharp > max(best_sharp * 1.75, best_sharp + 0.15)
                else:
                    collapsed = last > max(best * 1.75, best + 0.15)
            if collapsed:
                self._restore_trainable(best_snap)
                restored = True
                last = best
                lr = max(lr * self.train_lr_decay, 1e-5)
                stale = 0
                self.status_var.set(
                    f"collapse — restored best sharp={best_sharp:.4f}, lr→{lr:g}"
                    if anneal_on
                    else f"collapse — restored best={best:.4f}, lr→{lr:g}"
                )
                self._pump()

            ratio = last / max(empty_loss, 1e-8)
            if self.train_until_close:
                frac = 0.0
                if empty_loss > goal:
                    frac = (empty_loss - min(last, empty_loss)) / (empty_loss - goal + 1e-8)
                self._set_progress(float(np.clip(frac, 0.0, 1.0)))
            else:
                self._set_progress(i / max_n)

            do_preview = (
                i == 1
                or i % max(1, self.preview_every) == 0
                or i == max_n
            )
            if do_preview:
                frame = None
                for frame in _rollout_frames(
                    self.app, np.full(1, self.command_id, dtype=np.int64)
                ):
                    pass
                if frame is not None:
                    self._show_canvas(frame)
                    ink_miss = float(loss_fn.ink_miss(frame, target[:1]))
                anneal_tag = (
                    f" anneal={loss_fn.anneal_progress:.2f}" if anneal_on else ""
                )
                self.status_var.set(
                    f"auto-train '{self.command_name}'  {i}/{max_n}  "
                    f"loss={last:.4f} best={best:.4f} sharp={best_sharp:.4f}  "
                    f"blank={empty_loss:.4f} ratio={ratio:.2f} ink_miss={ink_miss:.2f}  "
                    f"lr={lr:g}{anneal_tag}"
                )
                self._pump()
                if self.preview_pause_s > 0:
                    time.sleep(self.preview_pause_s)

            if self.train_until_close and i >= min_n and self._close_enough(last, empty_loss):
                stopped_reason = "close"
                break

            if self.train_until_close and i >= min_n and stale >= self.train_patience:
                if boosts_left > 0 and not self._close_enough(last, empty_loss):
                    boosts_left -= 1
                    if best_snap is not None:
                        self._restore_trainable(best_snap)
                        restored = True
                        last = best
                    lr = max(lr * self.train_lr_decay, 1e-5)
                    stale = 0
                    self.status_var.set(
                        f"plateau — restored best, lr→{lr:g}  ({boosts_left} boosts left)…"
                    )
                    self._pump()
                else:
                    stopped_reason = "patience"
                    break

        if best_snap is not None and (last > best + 1e-6 or restored):
            self._restore_trainable(best_snap)
            last = best
            restored = True

        # Leave anneal at end so draw/report match refine stage.
        if anneal_on:
            loss_fn.set_anneal(1.0)
            empty_loss = self._empty_baseline_loss(target)

        self.animate_draw()
        tag = "  [restored best]" if restored else ""
        ratio = last / max(empty_loss, 1e-8)
        frame = None
        for frame in _rollout_frames(
            self.app, np.full(1, self.command_id, dtype=np.int64)
        ):
            pass
        if frame is not None:
            self._show_canvas(frame)
            ink_miss = float(loss_fn.ink_miss(frame, target[:1]))
        self.status_var.set(
            f"auto-train '{self.command_name}' done ({stopped_reason})  "
            f"traj={i}  loss={last:.4f} best={best:.4f}  "
            f"blank={empty_loss:.4f} ratio={ratio:.2f} ink_miss={ink_miss:.2f}"
            f"{tag}  — Draw anytime"
        )

    # -- the current brain (maze design points 4c, 7, 8, 9, 9a, 11): rounds steered by the C4 walker, the sleep between sweeps ---------------------------------------------
    def _brain_log(self, line: str) -> None:
        print(line, flush=True)
        if self.log_box is not None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.status_var.set(line[:200])
        self._pump()

    def _brain_make_session(self, shape: int, seed: int, val_shape: int):
        from examples.closed_loop_draw.session import DrawSession

        cfg = copy.deepcopy(self.cfg)
        cfg["closed_loop"]["target_conditioning"] = True      # the brain window's model sees the drawing it is asked for (a learned map of the target image into its goal), so one model can learn many drawings; the plain window is unchanged
        s = DrawSession(cfg, seed=seed, train_command=shape, val_command=val_shape, train_target=self.custom_target if shape == resolve_command("sketch")[0] else None)
        s.record_ring = False      # the window skips the ring-next rollout per boundary (a recorded column only, nothing reads it): faster rounds; the stage runs keep it
        self._brain_calls = 0
        self._show_target(s.target())

        def tick(kind: str) -> None:
            self._brain_calls += 1
            if self._stop_requested:
                raise _BrainStopped()
            if self._brain_calls % 5 == 0:      # a round takes minutes: say what it is doing and move the bar (the bar is an estimate: about 500 steps of training and rollouts per run)
                self.status_var.set(f"working: {self._brain_calls} steps so far in this run ({'training' if kind == 'train' else 'checking the result'})")
                self._set_progress(min(self._brain_calls / 500.0, 1.0))
            if kind == "train" and self._brain_calls % 10 == 0:
                self._show_canvas(s.canvas())
            self._pump()

        s.tick = tick
        return s

    def _brain_demo(self):
        if self.brain_demo is None:
            root = self.brain_args["tm_root"]
            for sub in ("maze", "walker_tree", "src"):
                p = os.path.join(root, sub)
                if p not in sys.path:
                    sys.path.insert(0, p)
            import draw_window

            unit = None
            if self.brain_args.get("null_result"):
                unit = json.loads(Path(self.brain_args["null_result"]).read_text())["noise_unit"]["unit"]
            fresh = not any(Path(self.brain_args["state_dir"]).glob("*")) if Path(self.brain_args["state_dir"]).exists() else True
            self.brain_demo = draw_window.BrainDemo(os.path.join(root, "maze"), self.cfg, self.brain_args["state_dir"], unit, STOCK_COMMAND_IDS, self._brain_make_session, self._brain_log, bars_path=self.brain_args.get("bars_file"), unit_by_shape=self._brain_units(), baseline_skills=True)
            self._brain_log("This window starts from a fresh state: nothing is remembered yet. Every number here is how well the model draws the drawing it is trained on (lower is better)." if fresh else "This window continues its saved state: the model and what the brain learned are kept. Every number is how well the model draws the drawing it is trained on (lower is better).")
        return self.brain_demo

    def _brain_units(self) -> dict | None:
        """Per-shape noise units from the null result's cells (position in the stock shapes), as the registered stage 2 uses them. The window is judged on the fit to the trained drawing (``DrawSession.fit``), so ``--null-result`` must be the fit-unit file (``null_read_fit.json``); the ring-next units of ``null_read.json`` would mis-scale every score."""
        if not self.brain_args.get("null_result"):
            return None
        cells = json.loads(Path(self.brain_args["null_result"]).read_text())["noise_unit"]["cells"]
        return {cid: float(cells[str(i)]["median_abs_d"]) for i, cid in enumerate(STOCK_COMMAND_IDS)}

    def brain_round(self) -> None:
        try:
            d = self._brain_demo()
            sweep, shape = d.next()
            self._brain_log(f"--- round {d.arm.round_no + 1}: sweep {sweep}, shape '{COMMANDS[shape]}' ---")
            d.round()
        except _BrainStopped:
            self._brain_log("stopped: this round is not counted")

    def brain_try(self, resume: bool = False) -> None:
        if self.custom_target is None:
            raise RuntimeError("Draw something on the pad and press Set first")
        name = self._drawing_name()      # the same drawing keeps one record, another drawing starts its own
        self._save_drawing(name)
        try:
            d = self._brain_demo()
            self._brain_log(f"--- train on your drawing ({name}): {'continues from where it got to, if it has been here before' if resume else 'fresh'} ---")
            d.try_drawing(resolve_command("sketch")[0], name, resume=resume)
        except _BrainStopped:
            self._brain_log("stopped: this try is not counted")

    def _random_target(self) -> np.ndarray:
        """A random drawing: one to three strokes (straight or bent) of 2-3 px, as (1, C, H, W) in [0, 1]."""
        rng = np.random.default_rng()
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        img = np.zeros((self.H, self.W), dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            pts = rng.uniform(3, [self.H - 3, self.W - 3], size=(int(rng.integers(2, 4)), 2))
            for a, b in zip(pts[:-1], pts[1:]):
                d = b - a
                t = np.clip(((yy - a[0]) * d[0] + (xx - a[1]) * d[1]) / max(float(d @ d), 1e-6), 0.0, 1.0)
                dist = np.hypot(yy - (a[0] + t * d[0]), xx - (a[1] + t * d[1]))
                img = np.maximum(img, np.clip(1.6 - dist, 0.0, 1.0).astype(np.float32))
        return np.repeat(img[None, None, :, :], self.C, axis=1)

    AUTO_SLEEP_EVERY = 5      # Auto sleeps after every 5 rounds (the demo's sweep of five)

    def _on_auto_toggle(self) -> None:
        """Auto on: rounds on new random drawings, one after the other, the one model carrying on, a sleep after every 5 rounds, until it is switched off (the running round finishes first) or Stop is pressed (the running round is dropped)."""
        if self.auto_var.get():
            if self.busy:
                self.auto_var.set(False)
                self.status_var.set("busy — wait for the current action to finish, then switch Auto on")
                return
            self._run_action(self.brain_auto, label="auto")
        else:
            self.status_var.set("auto: stopping after the current round…")

    def brain_auto(self) -> None:
        errs: list[float] = []
        demo = self._brain_demo()
        i = 0
        try:
            while self.auto_var.get() and not self._stop_requested:
                i += 1
                self.custom_target = self._random_target()
                self.command_id, self.command_name = resolve_command("sketch")
                self._show_target(self.custom_target)
                self._brain_log(f"--- auto round {i}: a random drawing ---")
                try:
                    name = self._drawing_name()
                    self._save_drawing(name)
                    r = demo.try_drawing(resolve_command("sketch")[0], name, resume=True)
                except _BrainStopped:
                    self._brain_log("stopped: this round is not counted")
                    break
                if r.get("ratio") is not None and r["ratio"] > 0:
                    errs.append(float(r["ratio"]))      # error of the carried model / error of a fresh model on the same drawing with the same effort: the difficulty of the drawing cancels
                if i % self.AUTO_SLEEP_EVERY == 0 and self.auto_var.get() and not self._stop_requested:      # the sleep between rounds: the brain goes over the rows so far and keeps what it can trust
                    self._brain_log(f"--- auto: sleep after round {i} ---")
                    demo.sleep_now()
        finally:
            self.auto_var.set(False)
        if len(errs) >= 4:
            import math

            h = len(errs) // 2
            a, b = math.exp(sum(math.log(x) for x in errs[:h]) / h), math.exp(sum(math.log(x) for x in errs[-h:]) / h)
            self._brain_log(f"AUTO SUMMARY {len(errs)} random drawings, error as a multiple of a fresh model's on the same drawing (below 1 = better than a beginner): first {h} rounds {a:.2f}x, last {h} rounds {b:.2f}x -> " + (f"improving ({(1 - b / a) * 100:.0f}% lower)" if b < a else f"not improving ({(b / a - 1) * 100:.0f}% higher)"))
        elif errs:
            self._brain_log(f"AUTO SUMMARY {len(errs)} round(s), error vs a fresh model: " + ", ".join(f"{e:.2f}x" for e in errs))

    def _drawing_name(self) -> str:
        if self.custom_target is None:
            raise RuntimeError("Draw something on the pad and press Set first")
        import hashlib

        # the name is per drawing (a hash of its pixels), not per window or per model: the same drawing keeps one record across tries and windows
        return "mine_" + hashlib.sha256(np.ascontiguousarray(self.custom_target).tobytes()).hexdigest()[:8]

    def _save_drawing(self, name: str) -> None:
        """Keep the drawing beside the trace (``drawings/<name>.npy`` in the state directory), so any round can be replayed. Auto saves one file per random drawing."""
        d = Path(self.brain_args["state_dir"]) / "drawings"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / f"{name}.npy").exists():
            np.save(d / f"{name}.npy", np.asarray(self.custom_target))

    def brain_start_over(self) -> None:
        self._brain_demo().start_over()
        self._brain_log("--- start over: forgot the model (its weights) and the round history; the next Train starts from a fresh model. What the brain learned about decisions stays ---")

    def brain_sleep(self) -> None:
        self._brain_log("--- sleep ---")
        self._brain_demo().sleep_now()

    def brain_sweep(self) -> None:
        k = len(STOCK_COMMAND_IDS)
        for _ in range(k - (self._brain_demo().arm.round_no % k)):
            self.brain_round()
            if self._stop_requested:
                break

    def warmup(self) -> None:
        B = self.app.batch_size
        n = self.warmup_n
        stock = list(STOCK_COMMAND_IDS)
        sketch_id = resolve_command("sketch")[0]
        include_sketch = self.custom_target is not None
        warmup_ids = list(stock)
        if include_sketch:
            # Interleave sketch so freehand gets the same warmup treatment.
            warmup_ids.append(sketch_id)
        label = "stock+sketch" if include_sketch else "stock shapes"
        self.status_var.set(f"warmup {n} trajectories on {label}…")
        self._pump()
        for i in range(1, n + 1):
            if self._closing or self._stop_requested:
                break
            cid = warmup_ids[(i - 1) % len(warmup_ids)]
            ids = np.full(B, cid, dtype=np.int64)
            if cid == sketch_id:
                assert self.custom_target is not None
                target = np.repeat(self.custom_target, B, axis=0)
            else:
                target = load_command_target(
                    cid, batch_size=B, height=self.H, width=self.W, channels=self.C
                )
            result = self.app.trainer.rollout_train(
                goal=DrawGoal(command_ids=ids, target=target),
                lr=self.app.lr,
                apply_updates=True,
            )
            self._set_progress(i / n)
            if i == 1 or i % max(1, self.preview_every) == 0 or i == n:
                self.command_id = cid
                self.command_name = COMMANDS[cid]
                self.cmd_var.set(self.command_name)
                self._show_target(target[:1])
                self.status_var.set(
                    f"warmup {i}/{n}  cmd={COMMANDS[cid]}  loss={result.total_loss:.4f}"
                )
                self._preview_rollout()
                self._pump()
        self._did_warmup = True
        if include_sketch:
            self._apply_command("sketch")
            self.animate_draw()
            self.status_var.set(
                "warmup done (included your sketch) — Draw / Train anytime"
            )
        else:
            self._apply_command("circle")
            self.animate_draw()
            self.status_var.set(
                "warmup done — draw on the pad → Use as target, then Warmup again to include it"
            )

    def _on_close(self) -> None:
        self._closing = True
        try:
            self.app.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive closed-loop draw window")
    ap.add_argument("config", nargs="?", default=str(Path(__file__).resolve().parent / "config_draw_interactive.yaml"))
    ap.add_argument("--brain", choices=("c4",), default=None, help="steer runs with the current brain (the C4 walker of the Training Manager's maze) and sleep between sweeps")
    ap.add_argument("--tm-root", default=os.environ.get("TM_HARNESS_ROOT"), help="training-manager/brain_rl_data_engine (or env TM_HARNESS_ROOT)")
    ap.add_argument("--brain-dir", default=os.path.expanduser("~/.draw_brain"), help="where the arm keeps its ledger, memory rows and log (the sleep survives closing the window)")
    ap.add_argument("--bars-file", default=None, help="the draw bars (draw_bars.json); without it the bank bars are used and the log says so")
    ap.add_argument("--null-result", default=None, help="the unsteered-null result JSON (its noise unit); without it the runs are shown but nothing is judged")
    a = ap.parse_args()
    brain = None
    if a.brain:
        if not a.tm_root:
            raise SystemExit("--brain c4 needs --tm-root (training-manager/brain_rl_data_engine) or TM_HARNESS_ROOT")
        brain = {"tm_root": a.tm_root, "state_dir": a.brain_dir, "null_result": a.null_result, "bars_file": a.bars_file}
    cfg = Path(a.config)
    print(f"[interactive] config={cfg}")
    print("[interactive] commands:", ", ".join(COMMANDS[i] for i in sorted(COMMANDS)))
    print("[interactive] draw on the left pad → Use as target → Train / Draw")
    DrawInteractiveApp(cfg, brain=brain).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
