# examples/closed_loop_draw/soft_renderer.py
"""Differentiable soft line stroke for closed-loop canvas demos (app layer)."""
from __future__ import annotations

import numpy as np


def _segment_distance_grid(
    h: int,
    w: int,
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Pixel↔segment distances for a batch of lines.

    Coordinates are in [-1, 1] (action tanh space), mapped to pixel centers.
    Returns dist (B, H, W) and aux for backward.
    """
    B = int(x0.shape[0])
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )
    # (H, W)
    px = xx[None, :, :]
    py = yy[None, :, :]
    x0 = x0.reshape(B, 1, 1).astype(np.float64)
    y0 = y0.reshape(B, 1, 1).astype(np.float64)
    x1 = x1.reshape(B, 1, 1).astype(np.float64)
    y1 = y1.reshape(B, 1, 1).astype(np.float64)
    vx = x1 - x0
    vy = y1 - y0
    seg2 = vx * vx + vy * vy + 1e-12
    wx = px - x0
    wy = py - y0
    t = (wx * vx + wy * vy) / seg2
    t_clamped = np.clip(t, 0.0, 1.0)
    dx = px - (x0 + t_clamped * vx)
    dy = py - (y0 + t_clamped * vy)
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    aux = {
        "px": px,
        "py": py,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "vx": vx,
        "vy": vy,
        "seg2": seg2,
        "t": t,
        "t_clamped": t_clamped,
        "dx": dx,
        "dy": dy,
        "dist": dist,
    }
    return dist.astype(np.float32), aux


def soft_stroke(
    action: np.ndarray,
    *,
    height: int,
    width: int,
    sigma: float = 0.08,
) -> tuple[np.ndarray, dict]:
    """
    Map A=[x0,y0,x1,y1] ∈ (-1,1)^4 → soft stroke (B, H, W) in [0, 1].

    stroke = exp(-dist^2 / (2 σ^2))
    """
    a = np.ascontiguousarray(action, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"action must be (B,4), got {a.shape}")
    dist, aux = _segment_distance_grid(
        height, width, a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    )
    inv = 1.0 / (2.0 * float(sigma) ** 2)
    stroke = np.exp(-dist.astype(np.float64) ** 2 * inv).astype(np.float32)
    aux["sigma"] = float(sigma)
    aux["inv"] = inv
    aux["stroke"] = stroke
    aux["action"] = a
    return stroke, aux


def soft_stroke_action_grad(
    d_stroke: np.ndarray, aux: dict
) -> np.ndarray:
    """∂L/∂A from ∂L/∂stroke using the forward aux dict."""
    d_stroke = d_stroke.astype(np.float64)
    stroke = aux["stroke"].astype(np.float64)
    dist = aux["dist"].astype(np.float64)
    inv = float(aux["inv"])
    # d stroke / d dist = stroke * (-2 dist * inv) = -stroke * dist / sigma^2
    # inv = 1/(2 sigma^2) → 2*inv = 1/sigma^2
    d_dist = d_stroke * stroke * (-2.0 * dist * inv)

    dx = aux["dx"]
    dy = aux["dy"]
    # dist = sqrt(dx^2+dy^2); d dist / d dx = dx/dist
    d_dx = d_dist * dx / dist
    d_dy = d_dist * dy / dist

    x0, y0, x1, y1 = aux["x0"], aux["y0"], aux["x1"], aux["y1"]
    vx, vy, seg2 = aux["vx"], aux["vy"], aux["seg2"]
    t = aux["t"]
    t_clamped = aux["t_clamped"]
    px, py = aux["px"], aux["py"]

    # dx = px - (x0 + t_c * vx); dy similar.
    # Only interior t in (0,1) propagates through t; endpoints freeze t.
    interior = (t > 0.0) & (t < 1.0)
    # ∂dx/∂x0 = -1 - t_c*(-1) - vx * ∂t_c/∂x0 ... use analytic for endpoints first.
    # Closest-point: p* = (1-t)a + t b. d(p*,a) etc.
    # For clamped ends: if t<=0, p*=a; if t>=1, p*=b; else interpolate.

    # Per-pixel accumulators (reduce to (B,) at the end).
    d_x0 = np.zeros_like(dist)
    d_y0 = np.zeros_like(dist)
    d_x1 = np.zeros_like(dist)
    d_y1 = np.zeros_like(dist)

    # Endpoint region t<=0: p*=(x0,y0); dx=px-x0 → ∂dx/∂x0=-1
    mask0 = t <= 0.0
    d_x0 += np.where(mask0, -d_dx, 0.0)
    d_y0 += np.where(mask0, -d_dy, 0.0)
    # t>=1: p*=(x1,y1)
    mask1 = t >= 1.0
    d_x1 += np.where(mask1, -d_dx, 0.0)
    d_y1 += np.where(mask1, -d_dy, 0.0)

    # Interior: p* = (x0,y0) + t (vx,vy), t = dot(w,v)/|v|^2
    # Use: dx = px - x0 - t*vx  → contributions via t and via x0,x1 in vx.
    if np.any(interior):
        wx = px - x0
        wy = py - y0
        # ∂t/∂x0 = ∂/∂x0 [(wx*vx + wy*vy)/seg2]
        # wx depends on x0: ∂wx/∂x0=-1; vx depends on x0: ∂vx/∂x0=-1
        # Quotient rule — numerically integrate by summing pixel contributions.
        # d(p*)/d params via:
        # p*_x = x0 + t*vx
        # dp*_x/dx0 = 1 + dt/dx0 * vx + t * dvx/dx0
        dt_dx0 = ((-1) * vx + wx * (-1)) / seg2 - t * (2 * vx * (-1)) / seg2
        dt_dy0 = ((-1) * vy + wy * (-1)) / seg2 - t * (2 * vy * (-1)) / seg2
        dt_dx1 = (wx * 1) / seg2 - t * (2 * vx * 1) / seg2
        dt_dy1 = (wy * 1) / seg2 - t * (2 * vy * 1) / seg2
        # dx = px - p*_x → ∂dx/∂θ = -∂p*_x/∂θ
        dp_x_dx0 = 1.0 + dt_dx0 * vx + t_clamped * (-1.0)
        dp_y_dx0 = 0.0 + dt_dx0 * vy
        dp_x_dy0 = 0.0 + dt_dy0 * vx
        dp_y_dy0 = 1.0 + dt_dy0 * vy + t_clamped * (-1.0)
        dp_x_dx1 = 0.0 + dt_dx1 * vx + t_clamped * (1.0)
        dp_y_dx1 = 0.0 + dt_dx1 * vy
        dp_x_dy1 = 0.0 + dt_dy1 * vx
        dp_y_dy1 = 0.0 + dt_dy1 * vy + t_clamped * (1.0)

        d_x0 += np.where(interior, -d_dx * dp_x_dx0 - d_dy * dp_y_dx0, 0.0)
        d_y0 += np.where(interior, -d_dx * dp_x_dy0 - d_dy * dp_y_dy0, 0.0)
        d_x1 += np.where(interior, -d_dx * dp_x_dx1 - d_dy * dp_y_dx1, 0.0)
        d_y1 += np.where(interior, -d_dx * dp_x_dy1 - d_dy * dp_y_dy1, 0.0)

    dA = np.stack(
        [
            d_x0.reshape(d_x0.shape[0], -1).sum(axis=1),
            d_y0.reshape(d_y0.shape[0], -1).sum(axis=1),
            d_x1.reshape(d_x1.shape[0], -1).sum(axis=1),
            d_y1.reshape(d_y1.shape[0], -1).sum(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return dA
