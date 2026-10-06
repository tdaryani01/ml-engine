"""Build a closed-loop run from a run config: pick each seat's option by name, wire them into the generic trainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.closed_loop.registry import SEATS, loss_options_for, loss_types_of, resolve
from src.closed_loop.trainer import BackpropFeedback, ClosedLoopTrainer, PolicyGradientFeedback, TeacherForcingFeedback


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
    """The option each seat uses. The schema's ``loss_type`` is data and the loss option is code, so they must agree: a loss the
    assembly names must implement the schema's ``loss_type`` (refused otherwise), and a loss the assembly leaves out is the one
    option that implements it (refused when none or several do)."""
    modules = dict(((cfg.get("assembly") or {}).get("modules")) or {})
    loss_type = str(((cfg.get("schema_template") or {}).get("loss_type")) or "").strip().lower()
    if loss_type and not str(modules.get("loss") or "").strip():
        found = loss_options_for(loss_type)
        if len(found) == 1:
            modules["loss"] = found[0]
        elif len(found) > 1:
            raise ValueError(f"schema loss_type {loss_type!r} is implemented by several loss options ({', '.join(found)}); name one in assembly.modules.loss")
        else:
            raise ValueError(f"no loss option implements the schema loss_type {loss_type!r}")
    missing = [s for s in SEATS if not str(modules.get(s) or "").strip()]
    if missing:
        raise ValueError("assembly.modules must name an option for every seat; missing: " + ", ".join(missing))
    named = str(modules["loss"]).strip()
    declared = loss_types_of(named)
    if loss_type and declared and loss_type not in declared:
        raise ValueError(
            f"loss option {named!r} implements {list(declared)} but the schema says loss_type {loss_type!r}"
            + (f" (options for it: {', '.join(loss_options_for(loss_type))})" if loss_options_for(loss_type) else "")
        )
    return {s: str(modules[s]).strip() for s in SEATS}


_FEEDBACK_PARAMS = {"backprop": (), "policy_gradient": ("noise_std", "gamma", "normalize_advantage"), "teacher_forcing": ()}


def feedback_from_config(cfg: dict[str, Any], *, seed: int = 0) -> Any:
    """The trainer's feedback strategy from ``closed_loop.feedback``: a name (``"backprop"`` or ``"policy_gradient"``) or
    ``{"kind": name, ...parameters}``. Absent: ``None``, which is backprop (the trainer's default)."""
    raw = (cfg.get("closed_loop") or {}).get("feedback")
    if raw in (None, "", {}):
        return None
    spec = {"kind": raw} if isinstance(raw, str) else dict(raw)
    kind = str(spec.pop("kind", "backprop")).strip().lower()
    if kind not in _FEEDBACK_PARAMS:
        raise ValueError(f"unknown feedback {kind!r}; available: {', '.join(sorted(_FEEDBACK_PARAMS))}")
    unknown = sorted(set(spec) - set(_FEEDBACK_PARAMS[kind]))
    if unknown:
        raise ValueError(f"feedback {kind!r} takes {list(_FEEDBACK_PARAMS[kind]) or 'no parameters'}, not {unknown}")
    if kind == "backprop":
        return BackpropFeedback()
    if kind == "teacher_forcing":
        return TeacherForcingFeedback()
    return PolicyGradientFeedback(seed=int(seed), **spec)


def assemble_closed_loop(cfg: dict[str, Any], *, seed: int = 0) -> ClosedLoopRun:
    mods = modules_from_config(cfg)
    loss = resolve("loss", mods["loss"])(cfg)
    feedback = feedback_from_config(cfg, seed=seed)
    if isinstance(feedback, PolicyGradientFeedback) and not callable(getattr(loss, "step_reward", None)):  # before anything native is opened
        raise ValueError(f"feedback 'policy_gradient' needs a loss option with step_reward; {mods['loss']!r} has none")
    encoder = resolve("encoder", mods["encoder"])(cfg, seed)
    actor = resolve("policy", mods["policy"])(cfg, encoder, seed)
    env = resolve("env", mods["env"])(cfg)
    trainer = ClosedLoopTrainer(actor=actor, env=env, loss_fn=loss, max_steps=int(cfg["closed_loop"]["max_steps"]), feedback=feedback)
    data_option = str(((cfg.get("assembly") or {}).get("modules") or {}).get("data") or "stock_commands")
    data = resolve("data", data_option)(cfg)
    return ClosedLoopRun(trainer=trainer, encoder=encoder, actor=actor, env=env, loss=loss, data=data, cfg=cfg)
