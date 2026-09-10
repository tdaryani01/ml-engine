# examples/closed_loop_draw/env.py
"""Canvas environment + reconstruction loss (app layer)."""
from __future__ import annotations

import numpy as np

from examples.closed_loop_draw.soft_renderer import soft_stroke, soft_stroke_action_grad


class SoftCanvasEnv:
    """
    Obs = canvas as NCHW float32 in [0,1].

    step(A): C <- clip(C + soft_stroke(action_scale * A), 0, 1)
    ``action_scale`` < 1 keeps tanh actions off the ±1 cliff so ∂tanh stays alive.
    Stores a stack for reverse ``backward_step``.
    """

    def __init__(
        self,
        *,
        height: int = 28,
        width: int = 28,
        channels: int = 1,
        sigma: float = 0.08,
        action_scale: float = 0.85,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.sigma = float(sigma)
        self.action_scale = float(action_scale)
        self.canvas: np.ndarray | None = None
        self.stack: list[dict] = []

    def reset(self, batch_size: int) -> np.ndarray:
        B = int(batch_size)
        self.canvas = np.zeros(
            (B, self.channels, self.height, self.width), dtype=np.float32
        )
        self.stack = []
        return self._obs()

    def _obs(self) -> np.ndarray:
        assert self.canvas is not None
        return np.array(self.canvas, copy=True)

    def step(self, action: np.ndarray) -> np.ndarray:
        assert self.canvas is not None
        before = np.array(self.canvas, copy=True)
        action = np.ascontiguousarray(action, dtype=np.float32)
        scaled = action * np.float32(self.action_scale)
        stroke, aux = soft_stroke(
            scaled, height=self.height, width=self.width, sigma=self.sigma
        )
        stroke_nchw = stroke[:, None, :, :]
        if self.channels > 1:
            stroke_nchw = np.repeat(stroke_nchw, self.channels, axis=1)
        after = np.clip(before + stroke_nchw.astype(np.float32), 0.0, 1.0)
        self.stack.append(
            {
                "before": before,
                "after": after,
                "action": np.array(action, copy=True),
                "scaled_action": np.array(scaled, copy=True),
                "aux": aux,
                "stroke_nchw": stroke_nchw.astype(np.float32),
            }
        )
        self.canvas = after
        return self._obs()

    def obs_before(self, t: int) -> np.ndarray:
        return np.array(self.stack[t]["before"], copy=True)

    def obs_after(self, t: int) -> np.ndarray:
        return np.array(self.stack[t]["after"], copy=True)

    def backward_step(
        self, t: int, d_after: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        d_after = ∂L/∂C_{t+1} (NCHW).

        Returns (dA, d_before) where dA is ∂L/∂raw_action (pre-scale).
        """
        step = self.stack[t]
        d_after = np.ascontiguousarray(d_after, dtype=np.float32)
        before = step["before"]
        stroke_nchw = step["stroke_nchw"]
        pre = before + stroke_nchw
        gate = ((pre >= 0.0) & (pre <= 1.0)).astype(np.float32)
        d_pre = d_after * gate
        d_before = d_pre
        d_stroke = np.sum(d_pre, axis=1)
        dA_scaled = soft_stroke_action_grad(d_stroke, step["aux"])
        dA = (dA_scaled * np.float32(self.action_scale)).astype(np.float32)
        return dA, d_before

    def action_grad(self, d_obs: np.ndarray) -> np.ndarray:
        if not self.stack:
            raise RuntimeError("action_grad without steps")
        dA, _ = self.backward_step(len(self.stack) - 1, d_obs)
        return dA

    def pop_obs_grad(self) -> np.ndarray | None:
        return None


class CanvasReconstructionLoss:
    """
    Terminal canvas↔target loss (shape-agnostic; no L/R/quad splits).

    ``kind``:
      - ``mse``: mean((C-T)^2) over all pixels
      - ``balanced``: 0.5*(mean(miss^2)+mean(extra^2)) over all pixels
      - ``fg_weighted``: mean(w (C-T)^2) with w=α on target foreground, else 1

    Optional ``blur_sigmas``: multi-scale Gaussian pre-filter on both canvases
    before the pixel loss (broad catchment for missing strokes, then sharp).
    Symmetric Gaussian → backward is the same blur applied to ∂L/∂blurred.

    Optional anneal: ``blur_weights`` → ``blur_weights_end`` via ``set_anneal(u)``
    with ``u`` in [0, 1] (0 = coarse / start, 1 = sharp / end).

    Optional EDT / Chamfer:
      - ``edt_weight * mean(C * D_hard(T))`` — snap predicted ink onto target skeleton
      - ``edt_sym_weight * mean(T * D_soft(C))`` — soft distance of target ink to
        predicted mass (differentiable; pulls coverage into gaps)
    ``edt_soft_tau`` is the softmin temperature in pixels.
    """

    def __init__(
        self,
        *,
        terminal_only: bool = True,
        max_steps: int | None = None,
        kind: str = "balanced",
        fg_alpha: float = 15.0,
        fg_thresh: float = 0.05,
        blur_sigmas: list[float] | tuple[float, ...] | None = None,
        blur_weights: list[float] | tuple[float, ...] | None = None,
        blur_weights_end: list[float] | tuple[float, ...] | None = None,
        edt_weight: float = 0.0,
        edt_sym_weight: float = 0.0,
        edt_soft_tau: float = 2.0,
    ) -> None:
        self.terminal_only = bool(terminal_only)
        self.max_steps = None if max_steps is None else int(max_steps)
        kind = str(kind).strip().lower()
        if kind not in ("mse", "balanced", "fg_weighted"):
            raise ValueError(
                f"unknown loss kind {kind!r}; expected mse|balanced|fg_weighted"
            )
        self.kind = kind
        self.fg_alpha = float(fg_alpha)
        self.fg_thresh = float(fg_thresh)
        if blur_sigmas is None:
            self.blur_sigmas = (0.0,)
        else:
            self.blur_sigmas = tuple(float(s) for s in blur_sigmas)
        if blur_weights is None:
            self.blur_weights_start = tuple(1.0 for _ in self.blur_sigmas)
        else:
            self.blur_weights_start = tuple(float(w) for w in blur_weights)
        if len(self.blur_weights_start) != len(self.blur_sigmas):
            raise ValueError("blur_weights length must match blur_sigmas")
        if blur_weights_end is None:
            self.blur_weights_end = self.blur_weights_start
        else:
            self.blur_weights_end = tuple(float(w) for w in blur_weights_end)
            if len(self.blur_weights_end) != len(self.blur_sigmas):
                raise ValueError("blur_weights_end length must match blur_sigmas")
        self.blur_weights = self.blur_weights_start
        self.anneal_progress = 0.0
        self._apply_anneal_weights(0.0)
        self.edt_weight = float(edt_weight)
        self.edt_sym_weight = float(edt_sym_weight)
        self.edt_soft_tau = float(edt_soft_tau)
        if self.edt_weight < 0.0 or self.edt_sym_weight < 0.0:
            raise ValueError("edt_weight / edt_sym_weight must be >= 0")
        if self.edt_soft_tau <= 0.0:
            raise ValueError("edt_soft_tau must be > 0")
        self._edt_cache_key: int | None = None
        self._edt_cache: np.ndarray | None = None
        self._dist_cache_hw: tuple[int, int] | None = None
        self._dist_cache: np.ndarray | None = None  # (HW, HW), pixels / diag

    @property
    def anneal_enabled(self) -> bool:
        return self.blur_weights_end != self.blur_weights_start

    @property
    def edt_enabled(self) -> bool:
        return self.edt_weight > 0.0 or self.edt_sym_weight > 0.0

    @staticmethod
    def _normalize_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
        wsum = float(sum(weights))
        if wsum <= 0:
            raise ValueError("blur weights must sum to > 0")
        return tuple(w / wsum for w in weights)

    def _apply_anneal_weights(self, progress: float) -> None:
        u = float(np.clip(progress, 0.0, 1.0))
        self.anneal_progress = u
        mixed = tuple(
            (1.0 - u) * a + u * b
            for a, b in zip(self.blur_weights_start, self.blur_weights_end)
        )
        self.blur_weights = mixed
        self._blur_w_norm = self._normalize_weights(mixed)

    def set_anneal(self, progress: float) -> None:
        """Lerp blur weights from start → end. ``progress`` in [0, 1]."""
        self._apply_anneal_weights(progress)

    def _is_scored(self, t: int) -> bool:
        if not self.terminal_only:
            return True
        if self.max_steps is None:
            return True
        return int(t) == int(self.max_steps) - 1

    def _fg_weights(self, target: np.ndarray) -> np.ndarray:
        w = np.ones_like(target, dtype=np.float64)
        w[target > self.fg_thresh] = self.fg_alpha
        return w

    @staticmethod
    def _gaussian_blur_nchw(x: np.ndarray, sigma: float) -> np.ndarray:
        """Separable Gaussian over H,W; sigma in pixels. sigma<=0 → identity."""
        if sigma <= 1e-8:
            return x
        from scipy.ndimage import gaussian_filter

        out = np.empty_like(x, dtype=np.float64)
        x64 = x.astype(np.float64)
        for b in range(x64.shape[0]):
            for c in range(x64.shape[1]):
                out[b, c] = gaussian_filter(
                    x64[b, c], sigma=float(sigma), mode="nearest"
                )
        return out

    def ink_miss(self, obs: np.ndarray, target: np.ndarray) -> float:
        """Fraction of target-ink energy still missing (1 = blank on ink, 0 = covered)."""
        c = obs.astype(np.float64)
        tgt = target.astype(np.float64)
        fg = tgt > self.fg_thresh
        if not np.any(fg):
            return 0.0
        miss = np.clip(tgt - c, 0.0, None)
        return float(miss[fg].sum() / (tgt[fg].sum() + 1e-8))

    def _target_edt(self, tgt: np.ndarray) -> np.ndarray:
        """
        Per-sample distance-to-ink map, NCHW, normalized by image diagonal.
        Cached while the target tensor is unchanged (same ndarray id + checksum).
        """
        from scipy.ndimage import distance_transform_edt

        key = (
            id(tgt),
            int(tgt.shape[0]),
            int(tgt.shape[-2]),
            int(tgt.shape[-1]),
            float(tgt.sum()),
        )
        if self._edt_cache is not None and self._edt_cache_key == key:
            return self._edt_cache

        H, W = int(tgt.shape[-2]), int(tgt.shape[-1])
        diag = float(np.hypot(H, W)) + 1e-8
        out = np.empty(tgt.shape, dtype=np.float64)
        for b in range(tgt.shape[0]):
            for c in range(tgt.shape[1]):
                fg = tgt[b, c] > self.fg_thresh
                if not np.any(fg):
                    out[b, c] = 0.0
                else:
                    # dist to nearest zero of (~fg) == dist to nearest ink pixel
                    out[b, c] = distance_transform_edt(~fg) / diag
        self._edt_cache_key = key
        self._edt_cache = out
        return out

    def _pairwise_dist(self, H: int, W: int) -> np.ndarray:
        """Cached (HW, HW) Euclidean distances normalized by image diagonal."""
        hw = (int(H), int(W))
        if self._dist_cache is not None and self._dist_cache_hw == hw:
            return self._dist_cache
        ys, xs = np.mgrid[0:H, 0:W]
        coords = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=1).astype(np.float64)
        # (HW,1,2) - (1,HW,2) → (HW,HW)
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        diag = float(np.hypot(H, W)) + 1e-8
        self._dist_cache = d / diag
        self._dist_cache_hw = hw
        return self._dist_cache

    def _soft_edt_plane(
        self, c_hw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Soft distance-to-ink for one HxW plane.

        Z_p = ∑_q C_q exp(-d_pq/τ) + eps
        D_p = max(0, -τ log(Z_p / (1+eps)))   # 0 when well-covered, >0 when blank
        Returns (D_hw, Z_flat, kern, gate_flat) with gate = 1_{shifted>0}.
        """
        H, W = c_hw.shape
        d = self._pairwise_dist(H, W)
        tau_n = float(self.edt_soft_tau) / (float(np.hypot(H, W)) + 1e-8)
        kern = np.exp(-d / tau_n)  # (HW, HW)
        c_flat = np.clip(c_hw.reshape(-1), 0.0, None)
        eps = 1e-2
        Z = kern @ c_flat + eps
        shifted = -tau_n * (np.log(Z) - np.log(1.0 + eps))
        gate = (shifted > 0.0).astype(np.float64)
        D = np.maximum(0.0, shifted).reshape(H, W)
        return D, Z, kern, gate

    def _soft_edt_grad_plane(
        self,
        dL_dD: np.ndarray,
        Z: np.ndarray,
        kern: np.ndarray,
        gate: np.ndarray,
    ) -> np.ndarray:
        """∂L/∂C from L depending on soft D; dL_dD is HxW."""
        H, W = dL_dD.shape
        tau_n = float(self.edt_soft_tau) / (float(np.hypot(H, W)) + 1e-8)
        # ∂shifted/∂C_q = -τ K_pq / Z_p; gate kills saturated region
        d_shift = (dL_dD.reshape(-1) * gate) * (-tau_n) / Z
        g_flat = kern.T @ d_shift
        return g_flat.reshape(H, W)

    def _edt_loss(self, c: np.ndarray, tgt: np.ndarray) -> float:
        total = 0.0
        if self.edt_weight > 0.0:
            D_t = self._target_edt(tgt)
            total += float(self.edt_weight * np.mean(c * D_t))
        if self.edt_sym_weight > 0.0:
            acc = 0.0
            n_planes = 0
            for b in range(c.shape[0]):
                for ch in range(c.shape[1]):
                    D_c, _, _, _ = self._soft_edt_plane(c[b, ch])
                    acc += float(np.mean(tgt[b, ch] * D_c))
                    n_planes += 1
            total += self.edt_sym_weight * (acc / max(n_planes, 1))
        return float(total)

    def _edt_grad(self, c: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        g = np.zeros_like(c)
        n = float(c.size)
        if self.edt_weight > 0.0:
            D_t = self._target_edt(tgt)
            g += (self.edt_weight * D_t) / n
        if self.edt_sym_weight > 0.0:
            n_planes = max(c.shape[0] * c.shape[1], 1)
            plane_scale = self.edt_sym_weight / n_planes
            for b in range(c.shape[0]):
                for ch in range(c.shape[1]):
                    _D_c, Z, kern, gate = self._soft_edt_plane(c[b, ch])
                    H, W = c.shape[-2], c.shape[-1]
                    dL_dD = plane_scale * (tgt[b, ch] / float(H * W))
                    g[b, ch] += self._soft_edt_grad_plane(dL_dD, Z, kern, gate)
        return g

    def _pixel_loss(self, c: np.ndarray, tgt: np.ndarray) -> float:
        diff = c - tgt
        if self.kind == "mse":
            return float(np.mean(diff * diff))
        if self.kind == "fg_weighted":
            w = self._fg_weights(tgt)
            return float(np.mean(w * diff * diff))
        miss = np.clip(-diff, 0.0, None)
        extra = np.clip(diff, 0.0, None)
        return float(0.5 * (np.mean(miss * miss) + np.mean(extra * extra)))

    def _pixel_grad(self, c: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        diff = c - tgt
        n = float(c.size)
        if self.kind == "mse":
            return (2.0 * diff) / n
        if self.kind == "fg_weighted":
            w = self._fg_weights(tgt)
            return (2.0 * w * diff) / n
        miss = np.clip(-diff, 0.0, None)
        extra = np.clip(diff, 0.0, None)
        return (extra - miss) / n

    def sharp_step_loss(self, obs: np.ndarray, target: np.ndarray, t: int) -> float:
        """Sharp pixel (+ EDT) — scale-stable checkpoint metric across blur anneal."""
        if not self._is_scored(t):
            return 0.0
        c = obs.astype(np.float64)
        tgt = target.astype(np.float64)
        return float(self._pixel_loss(c, tgt) + self._edt_loss(c, tgt))

    def step_loss(self, obs: np.ndarray, target: np.ndarray, t: int) -> float:
        if not self._is_scored(t):
            return 0.0
        c = obs.astype(np.float64)
        tgt = target.astype(np.float64)
        total = 0.0
        for sigma, wn in zip(self.blur_sigmas, self._blur_w_norm):
            cb = self._gaussian_blur_nchw(c, sigma)
            tb = self._gaussian_blur_nchw(tgt, sigma)
            total += wn * self._pixel_loss(cb, tb)
        total += self._edt_loss(c, tgt)
        return float(total)

    def step_obs_grad(self, obs: np.ndarray, target: np.ndarray, t: int) -> np.ndarray:
        if not self._is_scored(t):
            return np.zeros_like(obs, dtype=np.float32)
        c = obs.astype(np.float64)
        tgt = target.astype(np.float64)
        g = np.zeros_like(c)
        for sigma, wn in zip(self.blur_sigmas, self._blur_w_norm):
            cb = self._gaussian_blur_nchw(c, sigma)
            tb = self._gaussian_blur_nchw(tgt, sigma)
            gb = self._pixel_grad(cb, tb)
            # Gaussian is self-adjoint → blur(gb) back to canvas coords.
            g += wn * self._gaussian_blur_nchw(gb, sigma)
        g += self._edt_grad(c, tgt)
        return g.astype(np.float32)
