"""Build a closed-loop run from a run config: pick each seat's option by name, wire them into the generic trainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.closed_loop.registry import SEATS, resolve
from src.closed_loop.trainer import ClosedLoopTrainer


@dataclass
class ClosedLoopRun:
    trainer: ClosedLoopTrainer
    encoder: Any
    actor: Any
    env: Any
    loss: Any
    data: Any
    cfg: dict[str, Any]

    def close(self) -> None:
        """Release native resources the policy holds (the contract runtime a transformer policy opens)."""
        mhsa = getattr(self.actor, "mhsa", None)
        rt = getattr(mhsa, "_contract_runtime", None)
        if rt is not None:
            rt.close()
            mhsa._contract_runtime = None


def modules_from_config(cfg: dict[str, Any]) -> dict[str, str]:
    modules = dict(((cfg.get("assembly") or {}).get("modules")) or {})
    missing = [s for s in SEATS if not str(modules.get(s) or "").strip()]
    if missing:
        raise ValueError("assembly.modules must name an option for every seat; missing: " + ", ".join(missing))
    return {s: str(modules[s]).strip() for s in SEATS}


def assemble_closed_loop(cfg: dict[str, Any], *, seed: int = 0) -> ClosedLoopRun:
    mods = modules_from_config(cfg)
    encoder = resolve("encoder", mods["encoder"])(cfg, seed)
    actor = resolve("policy", mods["policy"])(cfg, encoder, seed)
    env = resolve("env", mods["env"])(cfg)
    loss = resolve("loss", mods["loss"])(cfg)
    trainer = ClosedLoopTrainer(actor=actor, env=env, loss_fn=loss, max_steps=int(cfg["closed_loop"]["max_steps"]))
    data_option = str(((cfg.get("assembly") or {}).get("modules") or {}).get("data") or "stock_commands")
    data = resolve("data", data_option)(cfg)
    return ClosedLoopRun(trainer=trainer, encoder=encoder, actor=actor, env=env, loss=loss, data=data, cfg=cfg)
