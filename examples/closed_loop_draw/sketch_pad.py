# examples/closed_loop_draw/sketch_pad.py
"""Tk freehand sketch pad → model-resolution float canvas (app layer)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np


class SketchPad(ttk.Frame):
    """
    Mouse-drawable black board. Exports a soft grayscale image as NCHW float32.

    Drawing is captured on an internal Hi-res buffer, then downsampled to
    ``out_h × out_w`` for the closed-loop target.
    """

    def __init__(
        self,
        master,
        *,
        out_h: int = 28,
        out_w: int = 28,
        channels: int = 1,
        view_size: int = 224,
        brush_radius: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.channels = int(channels)
        self.view_size = int(view_size)
        self.brush_radius = int(brush_radius)
        # Internal float buffer in [0,1], view resolution.
        self._buf = np.zeros((self.view_size, self.view_size), dtype=np.float32)
        self._drawing = False
        self._last: tuple[int, int] | None = None

        ttk.Label(self, text="Your sketch (draw with mouse)").pack(anchor="w")
        self.canvas = tk.Canvas(
            self,
            width=self.view_size,
            height=self.view_size,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#666666",
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(0, 4))
        self.use_btn = ttk.Button(row, text="Use as target")
        self.use_btn.pack(side=tk.LEFT)

    def set_use_command(self, fn) -> None:
        self.use_btn.configure(command=fn)

    def clear(self) -> None:
        self._buf.fill(0.0)
        self._last = None
        self.canvas.delete("stroke")

    def _paint(self, x: int, y: int) -> None:
        r = self.brush_radius
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(self.view_size, x + r + 1), min(self.view_size, y + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        patch = self._buf[y0:y1, x0:x1]
        patch[mask] = np.maximum(patch[mask], 1.0)
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="#ffffff", outline="", tags="stroke"
        )

    def _on_down(self, event) -> None:
        self._drawing = True
        self._last = (int(event.x), int(event.y))
        self._paint(*self._last)

    def _on_move(self, event) -> None:
        if not self._drawing:
            return
        x, y = int(event.x), int(event.y)
        if self._last is not None:
            # Stamp along the segment for continuous strokes.
            x0, y0 = self._last
            steps = max(1, int(np.hypot(x - x0, y - y0) // max(1, self.brush_radius // 2)))
            for t in np.linspace(0.0, 1.0, steps + 1):
                self._paint(int(x0 + t * (x - x0)), int(y0 + t * (y - y0)))
        self._last = (x, y)

    def _on_up(self, _event) -> None:
        self._drawing = False
        self._last = None

    def to_nchw(self, batch_size: int = 1, *, thicken: int = 0) -> np.ndarray:
        """Downsample sketch → (B, C, H, W) float32 in [0,1]."""
        # Block-average downsample.
        vs = self.view_size
        sh = vs // self.out_h
        sw = vs // self.out_w
        cropped = self._buf[: self.out_h * sh, : self.out_w * sw]
        small = cropped.reshape(self.out_h, sh, self.out_w, sw).mean(axis=(1, 3))
        small = np.clip(small, 0.0, 1.0).astype(np.float32)
        if thicken > 0:
            small = dilate_ink(small, radius=int(thicken))
        img = small[None, None, :, :]
        if self.channels > 1:
            img = np.repeat(img, self.channels, axis=1)
        return np.repeat(img, int(batch_size), axis=0)

    def is_empty(self) -> bool:
        return float(self._buf.max()) < 1e-6


def dilate_ink(img: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Max-dilate a 2D [0,1] ink map so thin outlines remain learnable at 28²."""
    r = int(radius)
    if r < 1:
        return img
    out = img.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r:
                continue
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            out = np.maximum(out, shifted)
    return out.astype(np.float32)
