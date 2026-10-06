"""Generators for the closed-loop classes: demonstrations (imitate), preferences (prefer), reach targets (reach) and road networks (navigate).

Each builds the file the family's data option reads (an ``.npz``) from the model's shapes (``obs_dim``, ``action_dim``, ``max_steps``). The teacher
(the expert, the hidden preference, the road layout) is drawn from the spec's seed alone, so every stretch and the held-out split share one concept.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.generators.base import ArrayData, Field, Generator, register

_DEFAULT_STEPS = 10


def _steps(shapes: dict[str, Any], p: dict[str, Any]) -> int:
    return int(shapes.get("max_steps") or p.get("steps") or _DEFAULT_STEPS)


class _ArrayGenerator(Generator):
    """A generator whose size is set by its own fields (``episodes``, ``targets``, ...), not the tabular ``rows``."""

    def fields(self) -> list[Field]:
        return self._fields()

    def _dims(self, shapes: dict[str, Any]) -> tuple[int, int]:
        return int(shapes.get("obs_dim") or 4), int(shapes.get("action_dim") or 2)


def _mse_baseline(actions: np.ndarray, noise: float) -> dict[str, Any]:
    return {"metric": "action_mse", "do_nothing": float(np.mean(actions ** 2)), "best_possible": float(noise ** 2),
            "note": "The error of a model that always outputs zero. A learner must get far below it; the floor is the noise in the demonstrations."}


class _Demos(_ArrayGenerator):
    classes = ("imitate",)

    def _fields(self):
        return [Field("episodes", "integer", "Episodes", 300, 20, 100_000, hint="How many demonstrated episodes."),
                Field("noise", "number", "Action noise", 0.05, 0.0, 1.0, hint="Noise added to the expert's actions: the error no model can get below.")]

    def _expert(self, shapes: dict[str, Any], p: dict[str, Any], concept: np.random.Generator):
        raise NotImplementedError

    def _make(self, shapes, p, rng, split, concept):
        obs_dim, act_dim = self._dims(shapes)
        steps = _steps(shapes, p)
        obs, act = self._episodes(shapes, p, rng, concept, obs_dim, act_dim, steps)
        act = act + p["noise"] * rng.standard_normal(act.shape)
        return ArrayData({"observations": obs.astype(np.float32), "actions": act.astype(np.float32)})

    def baseline(self, shapes, params, seed):
        d = self.build(shapes, {**(params or {}), "episodes": 400}, seed, "heldout")
        return _mse_baseline(d.arrays["actions"], self.resolve(params)["noise"])


class DemoLinearExpert(_Demos):
    name, label = "demo_linear_expert", "Linear expert"
    description = "An expert that answers each observation with a bounded linear rule (tanh of a random linear map). Observations are independent draws."

    def _episodes(self, shapes, p, rng, concept, obs_dim, act_dim, steps):
        w = concept.standard_normal((obs_dim, act_dim)) / np.sqrt(obs_dim)
        obs = rng.standard_normal((p["episodes"], steps, obs_dim))
        return obs, np.tanh(obs @ w * 2.0)


class DemoMlpExpert(_Demos):
    name, label = "demo_mlp_expert", "Neural expert"
    description = "An expert that is a random two-layer network: a smooth, non-linear rule the learner has to approximate."

    def _fields(self):
        return super()._fields() + [Field("hidden", "integer", "Expert size", 12, 2, 128, hint="Hidden units of the expert: more is a more tangled rule.")]

    def _episodes(self, shapes, p, rng, concept, obs_dim, act_dim, steps):
        w1 = concept.standard_normal((obs_dim, p["hidden"])) / np.sqrt(obs_dim)
        w2 = concept.standard_normal((p["hidden"], act_dim)) / np.sqrt(p["hidden"])
        obs = rng.standard_normal((p["episodes"], steps, obs_dim))
        return obs, np.tanh(np.tanh(obs @ w1 * 2.0) @ w2 * 2.0)


class DemoController(_Demos):
    name, label = "demo_controller", "Feedback controller"
    description = ("A linear system that drifts on its own; the expert is a controller that steers it back toward rest. "
                   "Each observation is the system's state, which the actions change: a real control demonstration.")

    def _fields(self):
        return super()._fields() + [Field("drift", "number", "Drift", 0.9, 0.5, 1.2, hint="How fast the state grows on its own (above 1 it is unstable without control).")]

    def _episodes(self, shapes, p, rng, concept, obs_dim, act_dim, steps):
        a = p["drift"] * np.eye(obs_dim) + 0.1 * concept.standard_normal((obs_dim, obs_dim)) / np.sqrt(obs_dim)
        b = concept.standard_normal((obs_dim, act_dim)) / np.sqrt(act_dim)
        gain = np.linalg.pinv(b) @ a  # the action that cancels the part of the next state the actions can reach
        x = rng.standard_normal((p["episodes"], obs_dim))
        obs, act = [], []
        for _ in range(steps):
            u = np.tanh(-(x @ gain.T))
            obs.append(x)
            act.append(u)
            x = x @ a.T + u @ b.T + 0.05 * rng.standard_normal(x.shape)
        return np.stack(obs, axis=1), np.stack(act, axis=1)


class Preferences(_ArrayGenerator):
    name, label = "prefs_teacher", "Hidden teacher"
    description = ("A hidden teacher prefers one action in every situation; the rejected action is the same answer turned by a fixed angle. "
                   "Chosen and rejected are equally far from zero, so a policy that outputs nothing scores at chance. Some pairs are mislabelled, as human choices are.")
    classes = ("prefer",)

    def _fields(self):
        return [Field("episodes", "integer", "Episodes", 300, 20, 100_000, hint="How many situations with a chosen and a rejected action."),
                Field("difference", "number", "Difference", 1.0, 0.1, 3.0, hint="How far the rejected action is turned from the chosen one (radians): small is subtle and hard."),
                Field("label_noise", "number", "Mislabelled", 0.1, 0.0, 0.4, hint="Share of pairs whose chosen and rejected are swapped: the ceiling on accuracy is one minus this.")]

    def _make(self, shapes, p, rng, split, concept):
        obs_dim, act_dim = self._dims(shapes)
        steps = _steps(shapes, p)
        w = concept.standard_normal((obs_dim, act_dim)) / np.sqrt(obs_dim)
        # The rejected action is the chosen one turned by a fixed angle in a fixed plane of the action space (the same turn everywhere), so it is exactly
        # as far from zero as the chosen one: outputting nothing cannot tell them apart. With a single action value the turn is a sign flip.
        if act_dim == 1:
            turn = -np.eye(1)
        else:
            q, _ = np.linalg.qr(concept.standard_normal((act_dim, 2)))
            a_, b_ = q[:, 0], q[:, 1]
            turn = np.eye(act_dim) + (np.cos(p["difference"]) - 1.0) * (np.outer(a_, a_) + np.outer(b_, b_)) + np.sin(p["difference"]) * (np.outer(b_, a_) - np.outer(a_, b_))
        obs = rng.standard_normal((p["episodes"], steps, obs_dim))
        chosen = np.tanh(obs @ w * 2.0)
        rejected = chosen @ turn.T
        swap = rng.random((p["episodes"], steps, 1)) < p["label_noise"]
        chosen, rejected = np.where(swap, rejected, chosen), np.where(swap, chosen, rejected)
        return ArrayData({"observations": obs.astype(np.float32), "chosen_actions": chosen.astype(np.float32), "rejected_actions": rejected.astype(np.float32)})

    def baseline(self, shapes, params, seed):
        d = self.build(shapes, {**(params or {}), "episodes": 400}, seed, "heldout").arrays
        c, r = d["chosen_actions"].reshape(-1, d["chosen_actions"].shape[-1]), d["rejected_actions"].reshape(-1, d["chosen_actions"].shape[-1])
        mean = np.concatenate([c, r]).mean(axis=0, keepdims=True)

        def acc(a):
            return float(np.mean(np.sum((a - c) ** 2, axis=1) < np.sum((a - r) ** 2, axis=1)))

        best = max(0.5, acc(np.zeros((1, c.shape[1]))), acc(mean))  # outputting nothing, or the same average answer everywhere
        return {"metric": "accuracy", "do_nothing": best, "best_possible": 1.0 - self.resolve(params)["label_noise"],
                "note": "How often a model that gives the same answer everywhere picks the chosen action. A learner must beat it clearly."}


def _arm():
    from src.closed_loop.reach_options import arm_from_config, forward_kinematics

    arm = arm_from_config({})
    return arm["dh"], arm["home"], forward_kinematics


class _Reach(_ArrayGenerator):
    classes = ("reach",)

    def _fields(self):
        return [Field("targets", "integer", "Targets", 400, 20, 100_000, hint="How many points to reach."),
                Field("spread", "number", "Spread", 0.6, 0.1, 1.5, hint="How far the arm's joints move from home to make a target (radians): bigger is a wider, harder workspace.")]

    def _check_shapes(self, shapes, p):
        ad = shapes.get("action_dim")
        return [f"the arm has 6 joints but the model's action size is {ad}"] if ad not in (None, 6) else []

    def _points(self, rng, p, n):
        dh, home, fk = _arm()
        q = home[None, :] + rng.uniform(-p["spread"], p["spread"], size=(n, len(dh)))
        return fk(dh, q)[0]

    def _make(self, shapes, p, rng, split, concept):
        return ArrayData({"targets": self._points(rng, p, p["targets"]).astype(np.float64)})

    def baseline(self, shapes, params, seed):
        d = self.build(shapes, {**(params or {}), "targets": 400}, seed, "heldout").arrays["targets"]
        dh, home, fk = _arm()
        start = fk(dh, home[None, :])[0][0]
        steps = int(shapes.get("max_steps") or 12)
        d2 = float(np.mean(np.sum((d - start) ** 2, axis=1)))
        return {"metric": "reach_loss", "do_nothing": d2 * (1.0 + 0.05 * (steps - 1)), "best_possible": 0.0,
                "mean_distance_m": float(np.mean(np.linalg.norm(d - start, axis=1))),
                "note": "The loss of an arm that never moves: the squared distance from its home pose to the target, counted at every step. A learner must get far below it."}


class ReachWorkspace(_Reach):
    name, label = "reach_workspace", "Workspace points"
    description = "Targets the arm can reach: the end-effector position of random joint angles around its home pose."


class ReachAboveTable(_Reach):
    name, label = "reach_above_table", "Points above the table"
    description = "Reachable targets that are also clear of the table plane (like picking from a shelf), so the arm must keep clear of the table on the way."

    def _fields(self):
        return super()._fields() + [Field("clearance", "number", "Clearance (m)", 0.1, 0.0, 0.5, hint="How far above the table plane a target must be.")]

    def _make(self, shapes, p, rng, split, concept):
        out: list[np.ndarray] = []
        while sum(len(o) for o in out) < p["targets"]:
            pts = self._points(rng, p, p["targets"])
            out.append(pts[pts[:, 2] >= p["clearance"]])
        return ArrayData({"targets": np.concatenate(out)[: p["targets"]].astype(np.float64)})


class _Roads(_ArrayGenerator):
    classes = ("navigate",)

    def _check_shapes(self, shapes, p):
        ad = shapes.get("action_dim")
        od = shapes.get("obs_dim")
        problems = []
        if ad is not None and od is not None and int(od) != 9 + 6 * int(ad):
            problems.append(f"{ad} road slots give an observation size of {9 + 6 * int(ad)}, but the model's is {od}")
        return problems

    def baseline(self, shapes, params, seed):
        from src.closed_loop.navigate_options import RoadEnv, RoadRoutes, RouteTime

        data = self.build(shapes, params, seed, "heldout")
        k = int(shapes.get("action_dim") or 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.npz"
            np.savez(path, **data.arrays)
            cl = {"graph_path": str(path), "max_roads": k, "max_steps": int(shapes.get("max_steps") or 60), "batch_size": 128, "val_fraction": 0.2}
            env, loss, routes = RoadEnv(cl), RouteTime(cl), RoadRoutes({"closed_loop": cl, "optimization": {"seed": seed}})
            goal = routes.val_goal(0)
            out: dict[str, float] = {}
            for rule in ("random", "toward_goal"):
                rng = np.random.default_rng(seed)
                obs = env.reset(len(goal.origin), goal)
                for t in range(cl["max_steps"]):
                    mask = env.action_mask()
                    if rule == "random":
                        score = rng.random(mask.shape) * mask
                    else:
                        slots = obs[:, 9:].reshape(len(obs), k, 6)
                        score = np.where(mask, slots[:, :, 2] + 1e-3 * rng.random(mask.shape), -1e9)
                    obs = env.step(score)
                out[rule] = float(loss.step_loss(obs, goal, cl["max_steps"] - 1))
                out[rule + "_arrival"] = float(np.mean(obs[:, 7] > 0.5))
        return {"metric": "excess_time", "do_nothing": out["random"], "best_possible": 0.0, "rule": out["toward_goal"],
                "arrival_do_nothing": out["random_arrival"], "arrival_rule": out["toward_goal_arrival"],
                "note": "Time over the quickest route (failing to arrive counts as a penalty). 'do_nothing' is a walker that picks any road at random; "
                        "'rule' heads toward the goal. A learner must beat the random walk clearly and is only interesting if it gets near or past the rule."}


class RoadGrid(_Roads):
    name, label = "road_grid", "City blocks"
    description = "A grid of city blocks with two-way roads (a share one-way) and a speed that varies from road to road."

    def _fields(self):
        return [Field("width", "integer", "Blocks across", 10, 3, 60), Field("height", "integer", "Blocks down", 10, 3, 60),
                Field("spacing", "number", "Block length (m)", 100.0, 20.0, 1000.0),
                Field("one_way", "number", "One-way share", 0.1, 0.0, 0.5, hint="Share of roads that go one way only.")]

    def _make(self, shapes, p, rng, split, concept):
        w, h = p["width"], p["height"]
        xy = np.array([[x * p["spacing"], y * p["spacing"]] for y in range(h) for x in range(w)], dtype=np.float64)
        src, dst = [], []
        for y in range(h):
            for x in range(w):
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx < w and y + dy < h:
                        a, b = y * w + x, (y + dy) * w + x + dx
                        src.append(a)
                        dst.append(b)
                        if concept.random() >= p["one_way"]:
                            src.append(b)
                            dst.append(a)
        src, dst = np.array(src), np.array(dst)
        speed = concept.uniform(8.0, 14.0, size=len(src))
        return ArrayData({"node_xy": xy, "edge_from": src, "edge_to": dst, "edge_time": np.linalg.norm(xy[dst] - xy[src], axis=1) / speed})


class RoadRadial(_Roads):
    name, label = "road_radial", "Ring roads and spokes"
    description = "A city with ring roads and fast spokes toward the centre: the quickest route often leaves the straight line to use a spoke."

    def _fields(self):
        return [Field("rings", "integer", "Rings", 5, 2, 20), Field("spokes", "integer", "Spokes", 12, 4, 48),
                Field("spoke_speedup", "number", "Spoke speed-up", 2.0, 1.0, 5.0, hint="How much faster the spokes are than the ring roads.")]

    def _make(self, shapes, p, rng, split, concept):
        r_n, s_n = p["rings"], p["spokes"]
        ang = np.linspace(0, 2 * np.pi, s_n, endpoint=False)
        xy = np.array([[0.0, 0.0]] + [[200.0 * (r + 1) * np.cos(a), 200.0 * (r + 1) * np.sin(a)] for r in range(r_n) for a in ang])
        node = lambda r, s: 1 + r * s_n + (s % s_n)  # noqa: E731
        src, dst, fast = [], [], []
        for s in range(s_n):
            for r in range(r_n):
                a, b = (0 if r == 0 else node(r - 1, s)), node(r, s)
                src += [a, b]
                dst += [b, a]
                fast += [True, True]
            for r in range(r_n):
                src += [node(r, s), node(r, s + 1)]
                dst += [node(r, s + 1), node(r, s)]
                fast += [False, False]
        src, dst, fast = np.array(src), np.array(dst), np.array(fast)
        speed = concept.uniform(8.0, 12.0, size=len(src)) * np.where(fast, p["spoke_speedup"], 1.0)
        return ArrayData({"node_xy": xy, "edge_from": src, "edge_to": dst, "edge_time": np.linalg.norm(xy[dst] - xy[src], axis=1) / speed})


for _g in (DemoLinearExpert(), DemoMlpExpert(), DemoController(), Preferences(), ReachWorkspace(), ReachAboveTable(), RoadGrid(), RoadRadial()):
    register(_g)
