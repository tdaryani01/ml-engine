"""Navigate: choose a route through a real road network, from where you are to where you are going.

The world is a directed road graph: ``node_xy`` (N, 2) metres in a local projection, and directed edges ``edge_from``, ``edge_to``,
``edge_time`` (seconds, from length and speed). The policy stands at an intersection and picks which outgoing road to take (up to
K, in order of compass direction), step by step, until it reaches the destination or runs out of steps. A choice among roads is
discrete, so this trains by policy gradient (``closed_loop.feedback: "categorical_policy_gradient"``), not by backprop.

Observation (9 + 6K values): ``[x, y, dx, dy, distance to goal, elapsed time, last edge time, arrived, invalid move]`` then per
road ``[exists, time, progress toward the goal, direction x, direction y, visited before]``. The reward is the negative travel time
relative to the shortest route, plus arrival. Held-out DESTINATIONS measure how much slower than optimal the routes are.

Seats: env ``road_graph``, loss ``route_time`` (schema ``loss_type`` ``route_time``), data ``road_routes``; encoder ``identity``
and policy ``mlp`` are shared (the policy's action_dim is K: one logit per road slot).

Config (``closed_loop``): ``graph_path`` (the ``.npz``), ``max_steps``, ``batch_size``, ``max_roads`` (K, default 6), ``obs_dim``
(9 + 6K) and ``action_dim`` (K), ``route_destinations`` (default 48), ``route_min_edges`` / ``route_max_edges`` (route length range in
median-edge-times, default 3 and 25), ``arrival_bonus`` (default 1), ``fail_penalty`` (default 2), ``val_fraction``.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.closed_loop import needs
from src.closed_loop.registry import register

_G = 9  # global observation values; then 6 per road slot


@dataclass(frozen=True)
class RoadGraph:
    xy: np.ndarray  # (N, 2) metres
    to: np.ndarray  # (N, K) next node, -1 where there is no road
    time: np.ndarray  # (N, K) seconds
    k: int
    scale: float  # metres: the map's size
    len_scale: float  # metres: a typical edge
    t_scale: float  # seconds: the time unit of the observation
    edge_time: float  # seconds: the median edge

    @property
    def n(self) -> int:
        return len(self.xy)


@lru_cache(maxsize=4)
def _load(path: str, k: int, mtime: float) -> RoadGraph:
    with np.load(path, allow_pickle=False) as z:
        missing = [key for key in ("node_xy", "edge_from", "edge_to", "edge_time") if key not in z]
        if missing:
            raise ValueError(f"{path}: needs node_xy (N, 2), edge_from, edge_to, edge_time (E,); missing {missing}")
        xy, src, dst, tim = (np.asarray(z[key]) for key in ("node_xy", "edge_from", "edge_to", "edge_time"))
    xy = xy.astype(np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError(f"{path}: node_xy must be (nodes, 2) with at least 2 nodes")
    src, dst, tim = src.astype(np.int64), dst.astype(np.int64), tim.astype(np.float64)
    if not (len(src) == len(dst) == len(tim)) or len(src) == 0:
        raise ValueError(f"{path}: edge_from, edge_to and edge_time must have the same, non-zero length")
    if src.min() < 0 or dst.min() < 0 or src.max() >= len(xy) or dst.max() >= len(xy) or np.any(tim <= 0):
        raise ValueError(f"{path}: edges must refer to existing nodes and have positive times")
    n = len(xy)
    to = np.full((n, k), -1, dtype=np.int64)
    time = np.zeros((n, k))
    out: list[list[int]] = [[] for _ in range(n)]
    for e, s in enumerate(src):
        out[int(s)].append(e)
    for node, edges in enumerate(out):
        if not edges:
            continue
        edges = sorted(edges, key=lambda e: tim[e])[:k]  # more roads than slots: keep the quickest
        ang = [np.arctan2(*(xy[dst[e]] - xy[node])[::-1]) for e in edges]
        for slot, e in enumerate(e for _, e in sorted(zip(ang, edges))):  # slots in compass order, so a slot means a direction
            to[node, slot], time[node, slot] = dst[e], tim[e]
    lengths = np.linalg.norm(xy[dst] - xy[src], axis=1)
    med_t = float(np.median(tim))
    scale = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))) or 1.0
    return RoadGraph(xy, to, time, k, scale, float(np.median(lengths)) or 1.0, 10.0 * med_t, med_t)


def load_graph(cl: dict[str, Any]) -> RoadGraph:
    path = str(cl.get("graph_path") or "").strip()
    if not path:
        raise ValueError("closed_loop.graph_path is required (the road network .npz: node_xy, edge_from, edge_to, edge_time)")
    k = int(cl.get("max_roads", 6))
    want_obs, want_act = _G + 6 * k, k
    if "obs_dim" in cl and int(cl["obs_dim"]) != want_obs:
        raise ValueError(f"closed_loop.obs_dim is {cl['obs_dim']} but {k} road slots give {want_obs} observation values ({_G} + 6 x {k})")
    if "action_dim" in cl and int(cl["action_dim"]) != want_act:
        raise ValueError(f"closed_loop.action_dim is {cl['action_dim']} but there are {k} road slots (one logit each)")
    return _load(str(Path(path).resolve()), k, Path(path).stat().st_mtime)


def times_to(g: RoadGraph, dest: int) -> np.ndarray:
    """Shortest travel time from every node to ``dest`` along the graph's roads (inf where there is no route)."""
    rev: list[list[tuple[int, float]]] = [[] for _ in range(g.n)]
    for u in range(g.n):
        for s in range(g.k):
            v = g.to[u, s]
            if v >= 0:
                rev[int(v)].append((u, float(g.time[u, s])))
    dist = np.full(g.n, np.inf)
    dist[dest] = 0.0
    heap = [(0.0, dest)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        for u, w in rev[v]:
            if d + w < dist[u]:
                dist[u] = d + w
                heapq.heappush(heap, (d + w, u))
    return dist


@dataclass
class RouteGoal:
    origin: np.ndarray  # (B,) node
    dest: np.ndarray  # (B,) node
    optimal: np.ndarray  # (B,) seconds of the quickest route

    @property
    def batch_size(self) -> int:
        return int(len(self.origin))


class RoadEnv:
    """Standing at an intersection; one road chosen per step. Only reset/step (no backward pass)."""

    def __init__(self, cl: dict[str, Any]) -> None:
        self.g = load_graph(cl)

    def obs_spec(self) -> tuple[int, ...]:
        return (_G + 6 * self.g.k,)

    def reset(self, batch_size: int, goal: RouteGoal) -> np.ndarray:
        g = self.g
        self.B, self.dest = batch_size, np.asarray(goal.dest, dtype=np.int64)
        self.node = np.asarray(goal.origin, dtype=np.int64).copy()
        self.visited = np.zeros((batch_size, g.n), dtype=bool)
        self.visited[np.arange(batch_size), self.node] = True
        self.elapsed = np.zeros(batch_size)
        self.arrived = self.node == self.dest
        self.last_time = np.zeros(batch_size)
        self.invalid = np.zeros(batch_size)
        return self._obs()

    def action_mask(self) -> np.ndarray:
        m = self.g.to[self.node] >= 0
        m[self.arrived] = True  # an arrived episode stands still: any choice is fine and ignored
        m[~m.any(axis=1)] = True  # a dead end: nothing is allowed, so let anything through (it will be an invalid move)
        return m

    def step(self, action: np.ndarray) -> np.ndarray:
        g = self.g
        slot = np.asarray(action).argmax(axis=1)
        nxt = g.to[self.node, slot]
        legal = (nxt >= 0) & ~self.arrived
        moved_time = np.where(legal, g.time[self.node, slot], 0.0)
        wasted = ~legal & ~self.arrived  # picked a road that does not exist: a step lost, with a time cost
        self.invalid = wasted.astype(np.float64)
        self.last_time = moved_time + np.where(wasted, g.edge_time, 0.0)
        self.elapsed = self.elapsed + self.last_time
        self.node = np.where(legal, nxt, self.node)
        self.visited[np.arange(self.B), self.node] = True
        self.arrived = self.arrived | (self.node == self.dest)
        return self._obs()

    def _obs(self) -> np.ndarray:
        g, B, k = self.g, self.B, self.g.k
        cur, goal = g.xy[self.node], g.xy[self.dest]
        dist = np.linalg.norm(goal - cur, axis=1)
        out = np.zeros((B, _G + 6 * k), dtype=np.float32)
        out[:, 0:2] = cur / g.scale
        out[:, 2:4] = (goal - cur) / g.scale
        out[:, 4] = dist / g.scale
        out[:, 5] = self.elapsed / g.t_scale
        out[:, 6] = self.last_time / g.t_scale
        out[:, 7] = self.arrived
        out[:, 8] = self.invalid
        nxt = g.to[self.node]  # (B, k)
        ok = nxt >= 0
        safe = np.where(ok, nxt, 0)
        step_vec = g.xy[safe] - cur[:, None, :]
        step_len = np.linalg.norm(step_vec, axis=2)
        d_next = np.linalg.norm(goal[:, None, :] - g.xy[safe], axis=2)
        slots = out[:, _G:].reshape(B, k, 6)
        slots[:, :, 0] = ok
        slots[:, :, 1] = np.where(ok, g.time[self.node] / g.t_scale, 0.0)
        slots[:, :, 2] = np.where(ok, (dist[:, None] - d_next) / g.len_scale, 0.0)
        unit = step_vec / np.maximum(step_len, 1e-9)[:, :, None]
        slots[:, :, 3:5] = np.where(ok[:, :, None], unit, 0.0)
        slots[:, :, 5] = ok & self.visited[np.arange(B)[:, None], safe]
        return out


class RouteTime:
    """The reward (and the held-out measure) of a route: time taken relative to the quickest route."""

    def __init__(self, cl: dict[str, Any]) -> None:
        self.g = load_graph(cl)
        self.T = int(cl["max_steps"])
        self.bonus = float(cl.get("arrival_bonus", 1.0))
        self.fail = float(cl.get("fail_penalty", 2.0))

    def step_reward(self, obs: np.ndarray, goal: RouteGoal, t: int) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float64)
        spent = o[:, 6] * self.g.t_scale
        arrived_now = (o[:, 7] > 0.5) & (spent > 0)
        r = -spent / goal.optimal + self.bonus * arrived_now
        if t == self.T - 1:
            r = r - self.fail * (o[:, 7] < 0.5)
        return r

    def step_loss(self, obs: np.ndarray, goal: RouteGoal, t: int) -> float:
        """At the last step: how much slower than the quickest route (0 = as quick), failing to arrive counted as a penalty."""
        if t != self.T - 1:
            return 0.0
        o = np.asarray(obs, dtype=np.float64)
        return float(np.mean(o[:, 5] * self.g.t_scale / goal.optimal + self.fail * (o[:, 7] < 0.5) - 1.0))


def _episode_metrics(self, obs: np.ndarray, goal: RouteGoal) -> dict[str, float]:
    """After a whole route: how many arrived, and how long the ones that did took compared with the quickest route."""
    o = np.asarray(obs, dtype=np.float64)
    arrived = o[:, 7] > 0.5
    out = {"arrival_rate": float(arrived.mean())}
    if arrived.any():
        ratio = o[arrived, 5] * self.g.t_scale / goal.optimal[arrived]
        out["median_time_ratio"] = float(np.median(ratio))
        out["mean_time_ratio"] = float(ratio.mean())
    return out


RouteTime.episode_metrics = _episode_metrics  # type: ignore[attr-defined]


class RoadRoutes:
    """Origin-destination pairs. Some destinations are held out whole, so validation is on places never trained toward."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        cl = cfg["closed_loop"]
        self.g = load_graph(cl)
        self.batch = int(cl["batch_size"])
        self.seed = int(cl.get("seed") or (cfg.get("optimization") or {}).get("seed") or 0)
        rng = np.random.default_rng([self.seed, 3])
        lo, hi = float(cl.get("route_min_edges", 3)) * self.g.edge_time, float(cl.get("route_max_edges", 25)) * self.g.edge_time
        dests = rng.permutation(self.g.n)[: int(cl.get("route_destinations", 48))]
        self.cases: list[tuple[int, np.ndarray, np.ndarray]] = []  # (destination, eligible origins, their quickest times)
        for d in dests:
            t = times_to(self.g, int(d))
            ok = np.where(np.isfinite(t) & (t >= lo) & (t <= hi))[0]
            if len(ok) >= 4:
                self.cases.append((int(d), ok, t[ok]))
        if len(self.cases) < 2:
            raise ValueError("the road network gives fewer than 2 destinations with routes of the requested length: "
                             "lower route_min_edges, raise route_max_edges, or use a better connected network")
        n_val = max(1, int(round(len(self.cases) * float(cl.get("val_fraction", 0.2)))))
        self.val_cases, self.train_cases = self.cases[:n_val], self.cases[n_val:]
        self._plate = 0
        # Each fit of a chain (Autopilot) is a stretch: it must draw NEW training batches, not replay the first fit's.
        self._stretch = max(0, int(((cfg.get("fit") or {}).get("stretch_index")) or 1) - 1)

    def _draw(self, rng: np.random.Generator, cases: list, count: int) -> RouteGoal:
        pick = rng.integers(0, len(cases), size=count)
        origin, dest, opt = np.zeros(count, np.int64), np.zeros(count, np.int64), np.zeros(count)
        for i, c in enumerate(pick):
            d, nodes, times = cases[int(c)]
            j = int(rng.integers(0, len(nodes)))
            origin[i], dest[i], opt[i] = nodes[j], d, times[j]
        return RouteGoal(origin, dest, opt)

    def train_goal(self, step: int) -> RouteGoal:
        return self._draw(np.random.default_rng([self.seed, self._stretch, self._plate, int(step)]), self.train_cases, self.batch)

    def val_goal(self, step: int) -> RouteGoal:
        return self._draw(np.random.default_rng([self.seed, 7]), self.val_cases, self.batch)

    def next_plate(self) -> None:
        self._plate += 1


@register("env", "road_graph")
def road_graph(cfg: dict[str, Any]):
    return RoadEnv(cfg["closed_loop"])


@register("loss", "route_time", loss_types=("route_time",))
def route_time(cfg: dict[str, Any]):
    return RouteTime(cfg["closed_loop"])


@register("data", "road_routes")
def road_routes(cfg: dict[str, Any]):
    return RoadRoutes(cfg)


needs.declare_needs("data", "road_routes", [
    {"name": "graph_path", "kind": "file", "label": "Road network (.npz)", "required": True, "config_key": "closed_loop.graph_path",
     "hint": "node_xy (nodes, 2) in metres, and directed edge_from, edge_to, edge_time (seconds). Built from OpenStreetMap by "
             "tools/osm_to_road_graph.py. It is read on this computer and never sent to TM."},
])
