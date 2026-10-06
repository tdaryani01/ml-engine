"""Generators for the closed-loop classes and for images / sequences: deterministic, new data per stretch, a fixed concept, never solvable by doing nothing,
and a file the family's own data option accepts."""
from __future__ import annotations

import numpy as np
import pytest

from src import generators as g

CL_SHAPES = {
    "imitate": {"obs_dim": 4, "action_dim": 2, "max_steps": 10}, "prefer": {"obs_dim": 4, "action_dim": 2, "max_steps": 10},
    "reach": {"obs_dim": 15, "action_dim": 6, "max_steps": 12}, "navigate": {"obs_dim": 45, "action_dim": 6, "max_steps": 60},
    "cnn": {"input_shape": [1, 16, 16], "num_classes": 4}, "mhsa": {"d_model": 16, "action_dim": 4, "max_seq_len": 8},
}
ALL = [(mc, gen.name) for mc in CL_SHAPES for gen in g.generators(mc)]


def test_every_new_class_has_generators() -> None:
    assert {mc for mc, _ in ALL} == set(CL_SHAPES)
    assert len(ALL) == 12


@pytest.mark.parametrize("mc,name", ALL)
def test_a_spec_builds_the_same_data_and_a_new_stretch_builds_new_train_data(mc, name) -> None:
    sh = CL_SHAPES[mc]
    spec = g.canonical_spec({"name": name, "seed": 5}, sh, mc)
    a, b = g.build(spec, sh, "train", 1), g.build(spec, sh, "train", 1)
    c, held1, held2 = g.build(spec, sh, "train", 2), g.build(spec, sh, "heldout", 1), g.build(spec, sh, "heldout", 7)
    for k in a.arrays:
        assert np.array_equal(a.arrays[k], b.arrays[k])
        assert np.array_equal(held1.arrays[k], held2.arrays[k])  # the held-out data never changes with the stretch
    if mc not in ("navigate",):  # a road network is the concept itself: one map for every stretch
        assert any(not np.array_equal(a.arrays[k], c.arrays[k]) for k in a.arrays)


@pytest.mark.parametrize("mc,name", ALL)
def test_each_reports_what_doing_nothing_scores(mc, name) -> None:
    sh = CL_SHAPES[mc]
    base = g.baseline(g.canonical_spec({"name": name, "seed": 5}, sh, mc), sh)
    assert base["metric"] and base["do_nothing"] == base["do_nothing"] and base["note"]
    if base["metric"] == "accuracy":
        assert base["do_nothing"] < 0.6  # not a task a constant answer already solves (the first preference file was 76% solved by zero)


def test_preferences_cannot_be_told_apart_by_size() -> None:
    sh = CL_SHAPES["prefer"]
    d = g.build(g.canonical_spec({"name": "prefs_teacher", "params": {"label_noise": 0.0}, "seed": 1}, sh, "prefer"), sh, "train").arrays
    assert np.allclose(np.linalg.norm(d["chosen_actions"], axis=-1), np.linalg.norm(d["rejected_actions"], axis=-1), atol=1e-4)


def test_the_files_load_in_the_families_own_data_options(tmp_path) -> None:
    from src.closed_loop.assembler import assemble_closed_loop

    mods = {
        "imitate": ({"encoder": "identity", "policy": "mlp", "env": "demo_replay", "loss": "action_mse", "data": "demonstrations"}, "demonstrations_path", {"feedback": "teacher_forcing"}),
        "prefer": ({"encoder": "identity", "policy": "mlp", "env": "demo_replay", "loss": "preference_margin", "data": "preferences"}, "preferences_path", {"feedback": "teacher_forcing"}),
        "reach": ({"encoder": "identity", "policy": "mlp", "env": "arm_reach", "loss": "reach_error", "data": "reach_targets"}, "targets_path", {"arm": "ur5"}),
        "navigate": ({"encoder": "identity", "policy": "mlp", "env": "road_graph", "loss": "route_time", "data": "road_routes"}, "graph_path",
                     {"feedback": {"kind": "categorical_policy_gradient", "entropy_weight": 0.01}, "max_roads": 6}),
    }
    for mc, (modules, key, extra) in mods.items():
        for gen in g.generators(mc):
            sh = CL_SHAPES[mc]
            path = g.write(g.build(g.canonical_spec({"name": gen.name, "seed": 2}, sh, mc), sh), tmp_path / f"{gen.name}.npz")
            cl = {**sh, "batch_size": 8, "hidden": 8, "val_fraction": 0.2, "seed": 0, key: str(path), **extra}
            assert assemble_closed_loop({"assembly": {"modules": modules}, "closed_loop": cl, "optimization": {"learning_rate": 0.01}}, seed=0) is not None


def test_image_and_sequence_files_have_the_shapes_their_loaders_want() -> None:
    sh = CL_SHAPES["cnn"]
    for gen in g.generators("cnn"):
        d = g.build(g.canonical_spec({"name": gen.name, "seed": 1}, sh, "cnn"), sh).arrays
        assert d["X"].shape[1:] == (1, 16, 16) and d["y"].dtype == np.int32 and set(np.unique(d["y"])) <= set(range(4))
    sh = CL_SHAPES["mhsa"]
    for gen in g.generators("mhsa"):
        d = g.build(g.canonical_spec({"name": gen.name, "seed": 1}, sh, "mhsa"), sh).arrays
        assert d["X"].shape[1:] == (8, 16) and d["y"].shape[1] == 4


def test_a_generator_refuses_shapes_it_cannot_serve() -> None:
    assert g.check({"name": "reach_workspace"}, {"action_dim": 3}, "reach")
    assert g.check({"name": "road_grid"}, {"obs_dim": 40, "action_dim": 6}, "navigate")
    assert g.check({"name": "seq_recall"}, {"d_model": 6, "action_dim": 4, "max_seq_len": 8}, "mhsa")
    assert g.check({"name": "image_shapes"}, {"input_shape": [1, 16, 16], "num_classes": 9}, "cnn")
