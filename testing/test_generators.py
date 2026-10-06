"""Synthetic data generators: each teacher is learnable, new for every stretch, never solvable by doing nothing, and writes a plate a model can read."""
from __future__ import annotations

import numpy as np
import pytest

from src import generators as g
from src.launcher.check import scan_plate

BIN = {"num_classes": 2}
MULTI = {"num_classes": 4}


def _knn_accuracy(train, held, k=7):
    d = ((held.X[:, None, :] - train.X[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    votes = train.y[idx]
    pred = np.array([np.bincount(v.astype(int)).argmax() for v in votes])
    return float((pred == held.y).mean())


CLASSIFIERS = [("tabular_blobs", "binary_classification", BIN, {"n_features": 4}), ("tabular_moons", "binary_classification", BIN, {"n_features": 3, "distractors": 0.3}),
               ("tabular_xor", "binary_classification", BIN, {"n_features": 2}), ("tabular_linear", "binary_classification", BIN, {"n_features": 4}),
               ("tabular_clusters", "multi_class", MULTI, {"n_features": 4}), ("tabular_random_network", "multi_class", MULTI, {"n_features": 4})]


def test_each_class_lists_the_generators_that_can_feed_it() -> None:
    names = lambda c: {d["name"] for d in g.describe(c)}  # noqa: E731
    assert names("binary_classification") >= {"tabular_blobs", "tabular_moons", "tabular_xor", "tabular_linear"}
    assert names("multi_class") >= {"tabular_clusters", "tabular_random_network"}
    assert names("regression") == {"tabular_sine_mix", "tabular_polynomial", "tabular_linear_regression"}
    assert names("cnn") == {"image_shapes", "image_gratings"} and names("mhsa") == {"seq_recall", "seq_marked_total"}
    assert names("some_unknown_class") == set()  # a class with no generator: the form offers none
    for d in g.describe():
        for f in d["fields"]:
            assert f["minimum"] is None or f["maximum"] is None or f["minimum"] <= f["default"] <= f["maximum"], (d["name"], f["name"])


def test_a_spec_is_checked_against_the_model_and_the_canonical_form_fills_the_defaults() -> None:
    assert g.check({"name": "tabular_blobs"}, BIN, "binary_classification") == []
    assert any("does not feed" in p for p in g.check({"name": "tabular_blobs"}, BIN, "regression"))
    assert any("unknown setting" in p for p in g.check({"name": "tabular_blobs", "params": {"nope": 1}}, BIN))
    assert any("at most" in p for p in g.check({"name": "tabular_blobs", "params": {"noise": 99}}, BIN))
    assert any("whole number" in p for p in g.check({"name": "tabular_blobs", "params": {"rows": 10.5}}, BIN))
    assert any("2 classes" in p for p in g.check({"name": "tabular_blobs"}, {"num_classes": 5}))
    assert any("not 5" in p for p in g.check({"name": "tabular_clusters"}, {"num_classes": 5}, "binary_classification"))
    assert any("no generator named" in p for p in g.check({"name": "nope"}))
    assert any("generator spec is" in p for p in g.check("blobs"))
    spec = g.canonical_spec({"name": "tabular_blobs", "seed": 5, "params": {"noise": 0.5}}, BIN, "binary_classification")
    assert spec == {"name": "tabular_blobs", "params": {"rows": 1000, "n_features": 8, "separation": 2.0, "noise": 0.5}, "seed": 5}
    with pytest.raises(ValueError, match="does not feed"):
        g.canonical_spec({"name": "tabular_blobs"}, BIN, "regression")


@pytest.mark.parametrize("name,cls,shapes,params", CLASSIFIERS)
def test_a_classification_teacher_is_learnable_and_not_solved_by_doing_nothing(name, cls, shapes, params) -> None:
    spec = {"name": name, "params": {**params, "rows": 600}, "seed": 1}
    train, held = g.build(spec, shapes, "train"), g.build(spec, shapes, "heldout")
    base = g.baseline(spec, shapes)
    chance = 1.0 / (shapes["num_classes"])
    assert base["metric"] == "accuracy" and base["do_nothing"] <= chance + 0.12  # always answering the commonest class is near chance
    assert train.classes == shapes["num_classes"] and set(np.unique(train.y)) <= set(range(shapes["num_classes"]))
    assert _knn_accuracy(train, held) >= base["do_nothing"] + 0.25  # a learner beats doing nothing by a wide margin


def test_xor_cannot_be_solved_by_a_straight_line() -> None:
    spec = {"name": "tabular_xor", "params": {"n_features": 2, "rows": 2000}, "seed": 2}
    t, h = g.build(spec, BIN, "train"), g.build(spec, BIN, "heldout")
    A = np.c_[t.X, np.ones(len(t.X))]
    w = np.linalg.lstsq(A, t.y * 2.0 - 1.0, rcond=None)[0]
    assert abs(float((((np.c_[h.X, np.ones(len(h.X))] @ w) > 0).astype(int) == h.y).mean()) - 0.5) < 0.08  # a linear model is at chance


def test_train_data_is_new_every_stretch_held_out_is_fixed_and_the_concept_never_changes() -> None:
    spec = {"name": "tabular_linear_regression", "params": {"n_features": 5, "noise": 0.2}, "seed": 9}
    a, b = g.build(spec, {}, "train", stretch=1), g.build(spec, {}, "train", stretch=2)
    assert not np.array_equal(a.X, b.X) and np.array_equal(a.X, g.build(spec, {}, "train", stretch=1).X)  # new draws, but reproducible
    assert np.array_equal(g.build(spec, {}, "heldout", stretch=1).X, g.build(spec, {}, "heldout", stretch=7).X)  # the held-out set is the same
    assert not np.array_equal(g.build({**spec, "seed": 10}, {}, "train").X, a.X)  # a different seed is different data
    w = np.linalg.lstsq(np.c_[a.X, np.ones(len(a.X))], a.y, rcond=None)[0]  # the teacher learned from one stretch predicts the next: one concept
    pred = np.c_[b.X, np.ones(len(b.X))] @ w
    assert 1 - float(((pred - b.y) ** 2).mean() / b.y.var()) > 0.9


@pytest.mark.parametrize("name,params", [("tabular_linear_regression", {"n_features": 4}), ("tabular_sine_mix", {"n_features": 4}), ("tabular_polynomial", {"n_features": 6})])
def test_a_regression_teacher_has_signal_well_above_its_noise(name, params) -> None:
    spec = {"name": name, "params": {**params, "noise": 0.1}, "seed": 3}
    base = g.baseline(spec, {})
    assert base["metric"] == "mse" and base["best_possible"] == pytest.approx(0.01)
    assert base["do_nothing"] > 20 * base["best_possible"]  # predicting the average is far worse than the best possible


def test_the_linear_regression_noise_floor_is_what_a_perfect_fit_scores() -> None:
    spec = {"name": "tabular_linear_regression", "params": {"n_features": 4, "noise": 0.3, "rows": 5000}, "seed": 4}
    t, h = g.build(spec, {}, "train"), g.build(spec, {}, "heldout")
    w = np.linalg.lstsq(np.c_[t.X, np.ones(len(t.X))], t.y, rcond=None)[0]
    mse = float(((np.c_[h.X, np.ones(len(h.X))] @ w - h.y) ** 2).mean())
    assert mse == pytest.approx(g.baseline(spec, {})["best_possible"], rel=0.2)


def test_the_plate_is_what_a_supervised_model_reads(tmp_path) -> None:
    spec = {"name": "tabular_clusters", "params": {"n_features": 3, "rows": 200}, "seed": 1}
    path = g.write_csv(g.build(spec, MULTI), tmp_path / "plates" / "gen.csv")
    header, rows, labels, problems = scan_plate(path)
    assert header == ["f0", "f1", "f2", "target"] and rows == 200 and problems == []
    assert set(labels) <= {"0", "1", "2", "3"}  # integer class labels in the last column
    reg = g.write_csv(g.build({"name": "tabular_sine_mix", "params": {"rows": 80}}, {}), tmp_path / "r.csv")
    assert scan_plate(reg)[1] == 80 and scan_plate(reg)[3] == []
