# Gym domain adapters — first: drawing_expert (BL-002 Slice D).
#
# Worker is domain-agnostic: sample → run → score → train.
# Future domains plug the same surface without rewriting the control plane.
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from examples.closed_loop_draw.apply_deltas import apply_drawing_deltas
from examples.closed_loop_draw.assemble import assemble, make_target
from examples.closed_loop_draw.commands import COMMANDS, STOCK_COMMAND_IDS
from examples.closed_loop_draw.expert_gym import (
    COMPLEXITY_POOLS,
    _jitter_config,
    _pick_complexity,
    _train_block,
)


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


@dataclass(frozen=True)
class GymTask:
    """One sampled curriculum item (domain-specific payload in ``meta``)."""

    domain: str
    seed: int
    complexity: str
    command_id: int
    meta: dict[str, Any]


class GymDomainAdapter(Protocol):
    name: str

    def sample(self, *, seed: int, complexity: str | None = None) -> GymTask: ...

    def run(
        self,
        task: GymTask,
        *,
        cfg: dict[str, Any],
        tm_uri: str,
        instance_id: str,
        pre_traj: int,
        post_traj: int,
    ) -> dict[str, Any]: ...

    def score(self, result: dict[str, Any]) -> float: ...

    def train(self, tm_uri: str, *, min_outcome: float = 0.0) -> dict[str, Any]: ...


class DrawingExpertDomain:
    """Draw knobs expert: stock shapes × config jitter → suggest/outcome/train."""

    name = "drawing_expert"

    def sample(self, *, seed: int, complexity: str | None = None) -> GymTask:
        rng = np.random.default_rng(seed)
        tier = _pick_complexity(rng, complexity)
        pool = COMPLEXITY_POOLS[tier]
        cid = int(rng.choice(pool))
        if cid not in STOCK_COMMAND_IDS:
            cid = int(rng.choice(list(STOCK_COMMAND_IDS)))
        return GymTask(
            domain=self.name,
            seed=int(seed),
            complexity=tier,
            command_id=cid,
            meta={"command": COMMANDS.get(cid, str(cid))},
        )

    def run(
        self,
        task: GymTask,
        *,
        cfg: dict[str, Any],
        tm_uri: str,
        instance_id: str,
        pre_traj: int,
        post_traj: int,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(task.seed)
        cfg = _jitter_config(cfg, rng, complexity=task.complexity)
        cfg = json.loads(json.dumps(cfg))
        cfg.setdefault("closed_loop", {})["command_id"] = task.command_id
        cfg["closed_loop"]["target"] = {"kind": "stock"}

        app = assemble(cfg, seed=task.seed)
        try:
            B = app.batch_size
            target = make_target(cfg, batch_size=B)
            ids = np.full(B, task.command_id, dtype=np.int64)
            lr = float(cfg.get("optimization", {}).get("learning_rate", app.lr))

            loss_pre, ink_pre = _train_block(
                app, target=target, command_ids=ids, n_traj=pre_traj, lr=lr
            )
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
                "meta": {
                    "command_id": task.command_id,
                    "command": task.meta.get("command"),
                    "complexity": task.complexity,
                    "seed": task.seed,
                    "phase": "pre",
                    "domain": self.name,
                    "cfg_sigma": cfg["closed_loop"].get("sigma"),
                    "cfg_continuity": cfg["closed_loop"].get("continuity_weight"),
                    "cfg_max_steps": cfg["closed_loop"].get("max_steps"),
                    "cfg_lr": lr,
                },
            }
            suggest = _post_json(
                f"{tm_uri.rstrip('/')}/api/experts/drawing/suggest",
                suggest_body,
            )
            episode_id = str(suggest.get("episode_id") or "")
            deltas = list(suggest.get("deltas") or [])
            applied, lr = apply_drawing_deltas(app, deltas, train_lr=lr)

            loss_post, ink_post = _train_block(
                app,
                target=target,
                command_ids=ids,
                n_traj=post_traj,
                lr=float(lr or app.lr),
            )
            outcome = self.score(
                {"ink_pre": ink_pre, "ink_post": ink_post}
            )
            if episode_id:
                _post_json(
                    f"{tm_uri.rstrip('/')}/api/experts/drawing/episodes/{episode_id}/outcome",
                    {"outcome": outcome},
                )
            return {
                "domain": self.name,
                "command_id": task.command_id,
                "command": task.meta.get("command"),
                "complexity": task.complexity,
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

    def score(self, result: dict[str, Any]) -> float:
        """Positive when ink_miss drops after the expert steer."""
        return float(result.get("ink_pre", 0.0)) - float(result.get("ink_post", 0.0))

    def train(self, tm_uri: str, *, min_outcome: float = 0.0) -> dict[str, Any]:
        return _post_json(
            f"{tm_uri.rstrip('/')}/api/experts/drawing/train",
            {"min_outcome": min_outcome},
            timeout_s=120.0,
        )


_REGISTRY: dict[str, Callable[[], GymDomainAdapter]] = {
    "drawing_expert": DrawingExpertDomain,
}


def get_gym_domain(name: str | None) -> GymDomainAdapter:
    key = str(name or "drawing_expert").strip() or "drawing_expert"
    factory = _REGISTRY.get(key)
    if factory is None:
        raise KeyError(f"unknown gym domain: {key!r} (known: {sorted(_REGISTRY)})")
    return factory()


def list_gym_domains() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "DrawingExpertDomain",
    "GymDomainAdapter",
    "GymTask",
    "get_gym_domain",
    "list_gym_domains",
]
