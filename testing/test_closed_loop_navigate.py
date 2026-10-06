"""Navigate on a road network: choose a road at each intersection, trained by categorical policy gradient."""
from __future__ import annotations

import numpy as np
import pytest

from src.closed_loop.assembler import assemble_closed_loop, modules_from_config
from src.closed_loop.fit import fit_closed_loop
from src.closed_loop.navigate_options import RoadEnv, RouteGoal, RouteTime, load_graph, times_to
from src.closed_loop.trainer import CategoricalPolicyGradientFeedback
from testing.test_closed_loop_demonstrations import _Ledger

MODS = {"encoder": "identity", "policy": "mlp", "env": "road_graph", "loss": "route_time", "data": "road_routes"}


def _grid(path, w=10, h=10, spacing=100.0, seed=0, one_way_fraction=0.0):
    """A city-block grid: nodes every ``spacing`` metres, two-way roads (a few one-way), speeds that vary per road."""
    rng = np.random.default_rng(seed)
    xy = np.array([[x * spacing, y * spacing] for y in range(h) for x in range(w)], dtype=np.float64)
    idx = lambda x, y: y * w + x  # noqa: E731
    src, dst = [], []
    for y in range(h):
        for x in range(w):
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < w and y + dy < h:
                    a, b = idx(x, y), idx(x + dx, y + dy)
                    src += [a]
                    dst += [b]
                    if rng.random() >= one_way_fraction:
                        src += [b]
                        dst += [a]
    src, dst = np.array(src), np.array(dst)
    speed = rng.uniform(8.0, 14.0, size=len(src))
    np.savez(path, node_xy=xy, edge_from=src, edge_to=dst, edge_time=np.linalg.norm(xy[dst] - xy[src], axis=1) / speed)
    return path


def _cfg(path, **extra):
    k = 6
    cl = {"graph_path": str(path), "max_steps": 40, "batch_size": 32, "max_roads": k, "obs_dim": 9 + 6 * k, "action_dim": k, "hidden": 64,
          "seed": 0, "route_destinations": 24, "route_min_edges": 3, "route_max_edges": 12, "feedback": {"kind": "categorical_policy_gradient"}, **extra}
    return {"assembly": {"modules": dict(MODS)}, "closed_loop": cl, "optimization": {"learning_rate": 0.003}}


def test_the_graph_is_read_with_roads_in_compass_order_and_the_quickest_kept(tmp_path) -> None:
    path = _grid(tmp_path / "g.npz", w=3, h=3)
    g = load_graph({"graph_path": str(path), "max_roads": 4})
    centre = 4  # the middle node of 3 x 3 has four roads
    assert (g.to[centre] >= 0).sum() == 4
    angles = [np.arctan2(*(g.xy[n] - g.xy[centre])[::-1]) for n in g.to[centre]]
    assert angles == sorted(angles)  # slot order is direction order
    capped = load_graph({"graph_path": str(path), "max_roads": 2})
    kept = capped.time[centre][capped.to[centre] >= 0]
    assert len(kept) == 2 and kept.max() <= np.sort(g.time[centre])[1] + 1e-9  # only the two quickest remain


def test_a_bad_graph_or_dimensions_are_refused(tmp_path) -> None:
    np.savez(tmp_path / "bad.npz", node_xy=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="missing \\['edge_from', 'edge_to', 'edge_time'\\]"):
        load_graph({"graph_path": str(tmp_path / "bad.npz")})
    path = _grid(tmp_path / "g.npz")
    with pytest.raises(ValueError, match="give 45 observation values"):
        load_graph({"graph_path": str(path), "max_roads": 6, "obs_dim": 40})
    with pytest.raises(ValueError, match="graph_path is required"):
        load_graph({})


def test_the_quickest_route_time_is_the_shortest_path(tmp_path) -> None:
    np.savez(tmp_path / "t.npz", node_xy=np.array([[0, 0], [1, 0], [2, 0], [1, 1.0]]), edge_from=np.array([0, 1, 0, 3]), edge_to=np.array([1, 2, 3, 2]),
             edge_time=np.array([10.0, 10.0, 5.0, 4.0]))  # 0 -> 1 -> 2 costs 20; 0 -> 3 -> 2 costs 9
    g = load_graph({"graph_path": str(tmp_path / "t.npz"), "max_roads": 2})
    t = times_to(g, 2)
    assert t[0] == pytest.approx(9.0) and t[1] == pytest.approx(10.0) and np.isinf(times_to(g, 0)[2])


def test_the_environment_moves_wastes_invalid_choices_and_stands_still_on_arrival(tmp_path) -> None:
    path = _grid(tmp_path / "g.npz", w=3, h=1)  # a line of three nodes: 0 - 1 - 2
    env = RoadEnv({"graph_path": str(path), "max_roads": 4})
    goal = RouteGoal(np.array([0]), np.array([2]), np.array([20.0]))
    env.reset(1, goal)
    east = np.zeros((1, 4))
    east[0, int(np.argmax(env.g.to[0] >= 0))] = 1.0  # node 0's only road (to node 1)
    obs = env.step(east)
    assert env.node[0] == 1 and not env.arrived[0] and obs[0, 6] > 0
    bad = np.zeros((1, 4))
    bad[0, 3] = 1.0  # no fourth road at node 1
    obs = env.step(bad)
    assert env.node[0] == 1 and obs[0, 8] == 1.0 and env.last_time[0] == pytest.approx(env.g.edge_time)  # a wasted step, with a time cost
    roads = np.where(env.g.to[1] == 2)[0]
    go = np.zeros((1, 4))
    go[0, roads[0]] = 1.0
    env.step(go)
    assert env.arrived[0] and env.node[0] == 2
    env.step(go)
    assert env.node[0] == 2 and env.last_time[0] == 0.0  # arrived: it stays


def test_the_reward_pays_arrival_once_and_fails_at_the_end(tmp_path) -> None:
    path = _grid(tmp_path / "g.npz", w=3, h=1)
    cl = {"graph_path": str(path), "max_steps": 3, "max_roads": 4}
    loss, g = RouteTime(cl), load_graph(cl)
    goal = RouteGoal(np.array([0, 0]), np.array([2, 2]), np.array([20.0, 20.0]))
    obs = np.zeros((2, 9 + 24), dtype=np.float32)
    obs[:, 6] = 10.0 / g.t_scale
    obs[0, 7] = 1.0  # sample 0 arrives with this step; sample 1 is still travelling
    r = loss.step_reward(obs, goal, 0)
    assert r[0] == pytest.approx(-10 / 20 + 1.0) and r[1] == pytest.approx(-10 / 20)
    assert loss.step_reward(obs, goal, 2)[1] == pytest.approx(-10 / 20 - 2.0)  # the last step without arriving is a failure
    done = np.zeros_like(obs)
    done[:, 7] = 1.0
    assert np.all(loss.step_reward(done, goal, 1) == 0.0)  # after arrival nothing more is earned or lost


def test_masked_options_are_never_sampled() -> None:
    p = CategoricalPolicyGradientFeedback._probs(np.zeros((2, 4)), np.array([[True, False, False, True], [False, True, False, False]]))
    np.testing.assert_allclose(p, [[0.5, 0, 0, 0.5], [0, 1, 0, 0]], atol=1e-12)


def test_the_policy_learns_to_route_to_destinations_it_was_not_trained_toward(tmp_path) -> None:
    run = assemble_closed_loop(_cfg(_grid(tmp_path / "g.npz")), seed=0)
    led = _Ledger()
    fit_closed_loop(run, led, lr=0.003, steps=500, patience=0, checkpoint_every=250)
    excess = [d[3] for d in led.docs if d[0] == "step"]  # held-out destinations; 0 would be the quickest route every time
    first, last = float(np.mean(excess[:10])), float(np.mean(excess[-10:]))
    assert first > 1.0  # an untrained policy fails to arrive or wanders
    assert last < 0.3 * first and last < 0.5  # trained: routes within about 50% of the quickest, on destinations it never trained toward


def test_the_schema_loss_type_picks_the_route_loss_and_the_network_is_a_need() -> None:
    from src.closed_loop.needs import assembly_needs

    mods = {k: v for k, v in MODS.items() if k != "loss"}
    assert modules_from_config({"assembly": {"modules": mods}, "schema_template": {"loss_type": "route_time"}})["loss"] == "route_time"
    out = assembly_needs(MODS)["needs"]
    assert [n["name"] for n in out] == ["graph_path"] and out[0]["required"] is True


def test_the_fit_reports_how_many_routes_arrive_and_how_slow_they_are(tmp_path) -> None:
    run = assemble_closed_loop(_cfg(_grid(tmp_path / "g.npz")), seed=0)
    led = _Ledger()
    fit_closed_loop(run, led, lr=0.003, steps=200, patience=0, checkpoint_every=200)
    last = [e for e in led.extra if e][-1]
    assert 0.0 <= last["val_arrival_rate"] <= 1.0 and last["val_arrival_rate"] > 0.5  # most held-out routes arrive after training
    assert last["val_median_time_ratio"] >= 1.0  # no route is quicker than the quickest one
