"""Hand-written teachers for the tabular classes (binary classification, multi-class, regression).

Each one defines a concept (the teacher) from the spec's seed alone, so train and held-out data, and every stretch, share one concept; only the
examples are fresh. Features are ``f0..f{n-1}``; the target is the last column.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.generators.base import Data, Field, Generator, register

_NF = Field("n_features", "integer", "Features", 8, 2, 512, hint="How many input columns the model sees.")
_DISTRACT = Field("distractors", "number", "Distractor scale", 1.0, 0.0, 10.0, hint="Spread of the columns that carry no signal.")


def _nf(p: dict[str, Any]) -> int:
    return int(p["n_features"])


def _classes(shapes: dict[str, Any], binary: bool) -> int:
    return 2 if binary else int(shapes.get("num_classes") or 4)


def _majority(y: np.ndarray) -> float:
    return float(np.bincount(y.astype(int)).max() / len(y))


def _bayes(prob_one: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """The best any model can do when the true chance of class 1 is known: its log-loss (nats) and its accuracy."""
    p = np.clip(prob_one, 1e-12, 1 - 1e-12)
    return {"best_possible_loss": float(-np.mean(np.where(y == 1, np.log(p), np.log(1 - p)))),
            "best_possible": float(np.mean((p > 0.5) == (y == 1)))}


def _phi(x: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _concept_direction(seed: int, n: int) -> np.ndarray:
    u = np.random.default_rng([int(seed), 2]).standard_normal(n)  # the first draw of the concept stream, as ``_make`` takes it
    return u / np.linalg.norm(u)


class _Classifier(Generator):
    binary_only = False

    def _check_shapes(self, shapes: dict[str, Any], p: dict[str, Any]) -> list[str]:
        k = shapes.get("num_classes")
        if self.binary_only and k not in (None, 2):
            return [f"this generator makes 2 classes; the model has {k}"]
        if not self.binary_only and k is not None and int(k) < 2:
            return ["num_classes must be at least 2"]
        return []

    def baseline(self, shapes: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
        d = self.build(shapes, {**(params or {}), "rows": 4000}, seed, "heldout")
        return {"metric": "accuracy", "do_nothing": _majority(d.y), "best_possible": None,
                "note": "A model that always answers the most common class scores this; a learner must beat it clearly."}


class Blobs(_Classifier):
    name, label = "tabular_blobs", "Two blobs"
    description = "Two Gaussian clouds on opposite sides of a random direction. The easiest classification: a straight line separates them."
    classes = ("binary_classification",)
    binary_only = True

    def _fields(self):
        return [_NF, Field("separation", "number", "Separation", 2.0, 0.0, 20.0, hint="Distance between the two centres."),
                Field("noise", "number", "Noise", 1.0, 0.05, 10.0, hint="Spread of each cloud (higher = more overlap).")]

    def _make(self, shapes, p, rng, split, concept):
        u = concept.standard_normal(_nf(p))
        u /= np.linalg.norm(u)
        y = rng.integers(0, 2, size=p["rows"])
        centre = (y[:, None] * 2 - 1) * (p["separation"] / 2.0) * u[None, :]
        return Data(centre + p["noise"] * rng.standard_normal((p["rows"], _nf(p))), y, "classification", 2)


def _blobs_baseline(self, shapes, params, seed):
    out = _Classifier.baseline(self, shapes, params, seed)
    p = self.resolve(params)
    d = self.build(shapes, {**(params or {}), "rows": 4000}, seed, "heldout")
    u = _concept_direction(seed, _nf(p))
    logit = p["separation"] * (d.X @ u) / p["noise"] ** 2
    out.update(_bayes(1.0 / (1.0 + np.exp(-logit)), d.y))
    return out


Blobs.baseline = _blobs_baseline  # type: ignore[method-assign]


class Moons(_Classifier):
    name, label = "tabular_moons", "Two moons"
    description = "Two interleaving half-circles in the first two columns, the rest noise. No straight line separates them."
    classes = ("binary_classification",)
    binary_only = True

    def _fields(self):
        return [_NF, Field("noise", "number", "Noise", 0.15, 0.0, 2.0, hint="Jitter around each moon."), _DISTRACT]

    def _make(self, shapes, p, rng, split, concept):
        n, f = p["rows"], _nf(p)
        y = rng.integers(0, 2, size=n)
        t = rng.uniform(0, np.pi, size=n)
        x0 = np.where(y == 0, np.cos(t), 1.0 - np.cos(t))
        x1 = np.where(y == 0, np.sin(t), 0.5 - np.sin(t))
        X = p["distractors"] * rng.standard_normal((n, f))
        X[:, 0], X[:, 1] = x0 + p["noise"] * rng.standard_normal(n), x1 + p["noise"] * rng.standard_normal(n)
        return Data(X, y, "classification", 2)


class Xor(_Classifier):
    name, label = "tabular_xor", "XOR"
    description = "The class is the sign of the product of the first two columns: a linear model scores about 50%. Other columns are noise."
    classes = ("binary_classification",)
    binary_only = True

    def _fields(self):
        return [_NF, Field("label_noise", "number", "Label noise", 0.05, 0.0, 0.5, hint="Chance a label is flipped.")]

    def _make(self, shapes, p, rng, split, concept):
        X = rng.standard_normal((p["rows"], _nf(p)))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        flip = rng.random(p["rows"]) < p["label_noise"]
        return Data(X, np.where(flip, 1 - y, y), "classification", 2)


def _xor_baseline(self, shapes, params, seed):
    out = _Classifier.baseline(self, shapes, params, seed)
    q = self.resolve(params)["label_noise"]
    out.update({"best_possible": 1.0 - q, "best_possible_loss": 0.0 if q <= 0 else float(-(q * np.log(q) + (1 - q) * np.log(1 - q)))})
    return out


Xor.baseline = _xor_baseline  # type: ignore[method-assign]


class LinearBoundary(_Classifier):
    name, label = "tabular_linear", "Linear boundary"
    description = "The class is which side of a random hyperplane a point falls on, with noise near the boundary."
    classes = ("binary_classification",)
    binary_only = True

    def _fields(self):
        return [_NF, Field("noise", "number", "Boundary noise", 0.5, 0.0, 5.0, hint="Noise added before the side is decided.")]

    def _make(self, shapes, p, rng, split, concept):
        w = concept.standard_normal(_nf(p))
        w /= np.linalg.norm(w)
        X = rng.standard_normal((p["rows"], _nf(p)))
        return Data(X, ((X @ w + p["noise"] * rng.standard_normal(p["rows"])) > 0).astype(int), "classification", 2)


def _linear_baseline(self, shapes, params, seed):
    out = _Classifier.baseline(self, shapes, params, seed)
    p = self.resolve(params)
    d = self.build(shapes, {**(params or {}), "rows": 4000}, seed, "heldout")
    w = _concept_direction(seed, _nf(p))
    z = d.X @ w
    out.update(_bayes(_phi(z / p["noise"]) if p["noise"] > 0 else (z > 0).astype(float), d.y))
    return out


LinearBoundary.baseline = _linear_baseline  # type: ignore[method-assign]


class Clusters(_Classifier):
    name, label = "tabular_clusters", "Gaussian clusters"
    description = "One Gaussian cloud per class, centred at a random point. The number of classes comes from the model."
    classes = ("multi_class", "binary_classification")

    def _fields(self):
        return [_NF, Field("separation", "number", "Separation", 3.0, 0.0, 30.0, hint="Distance of each centre from the origin."),
                Field("noise", "number", "Noise", 1.0, 0.05, 10.0, hint="Spread of each cloud.")]

    def _make(self, shapes, p, rng, split, concept):
        k = _classes(shapes, False)
        c = concept.standard_normal((k, _nf(p)))
        c = p["separation"] * c / np.linalg.norm(c, axis=1, keepdims=True)
        y = rng.integers(0, k, size=p["rows"])
        return Data(c[y] + p["noise"] * rng.standard_normal((p["rows"], _nf(p))), y, "classification", k)


class RandomNetwork(_Classifier):
    name, label = "tabular_random_network", "Random network teacher"
    description = "Labels come from a small random neural network: a generic, nonlinear concept for any number of features and classes (a plumbing test, not a real-world problem)."
    classes = ("multi_class", "binary_classification")

    def _fields(self):
        return [_NF, Field("hidden", "integer", "Teacher size", 16, 2, 256, hint="Hidden units of the teacher network."),
                Field("label_noise", "number", "Label noise", 0.02, 0.0, 0.5, hint="Chance a label is replaced at random.")]

    def _make(self, shapes, p, rng, split, concept):
        k, f, h = _classes(shapes, False), _nf(p), p["hidden"]
        w1, w2 = concept.standard_normal((f, h)) / np.sqrt(f), concept.standard_normal((h, k)) / np.sqrt(h)
        ref = np.tanh(concept.standard_normal((4000, f)) @ w1) @ w2
        bias = np.zeros(k)
        for _ in range(200):  # nudge each class score until the classes come out about equally often on a reference sample
            freq = np.bincount(np.argmax(ref + bias, axis=1), minlength=k) / len(ref)
            bias += 0.5 * (1.0 / k - freq)
        X = rng.standard_normal((p["rows"], f))
        y = np.argmax(np.tanh(X @ w1) @ w2 + bias, axis=1)
        flip = rng.random(p["rows"]) < p["label_noise"]
        return Data(X, np.where(flip, rng.integers(0, k, size=p["rows"]), y), "classification", k)


class _Regressor(Generator):
    noise_field = Field("noise", "number", "Noise", 0.1, 0.0, 10.0, hint="Standard deviation of the noise added to the target.")

    def baseline(self, shapes: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
        d = self.build(shapes, {**(params or {}), "rows": 4000}, seed, "heldout")
        noise = self.resolve(params)["noise"]
        return {"metric": "mse", "do_nothing": float(np.var(d.y)), "best_possible": float(noise**2),
                "note": "Always predicting the average scores 'do_nothing' (the target's variance); the noise sets the best any model can do."}


class SineMix(_Regressor):
    name, label = "tabular_sine_mix", "Sum of sines"
    description = "The target is a sum of sines of random projections of the features, plus noise."
    classes = ("regression",)

    def _fields(self):
        return [_NF, Field("components", "integer", "Components", 3, 1, 20, hint="How many sine terms."), self.noise_field]

    def _make(self, shapes, p, rng, split, concept):
        f, m = _nf(p), p["components"]
        w, a, ph = concept.standard_normal((m, f)) / np.sqrt(f), concept.uniform(0.5, 1.5, m), concept.uniform(0, 2 * np.pi, m)
        X = rng.standard_normal((p["rows"], f))
        y = (np.sin(X @ w.T * 2.0 + ph) * a).sum(axis=1) + p["noise"] * rng.standard_normal(p["rows"])
        return Data(X, y, "regression")


class Polynomial(_Regressor):
    name, label = "tabular_polynomial", "Polynomial"
    description = "The target is a random polynomial of a few of the features (the rest are noise), plus noise."
    classes = ("regression",)

    def _fields(self):
        return [_NF, Field("informative", "integer", "Informative features", 3, 1, 10, hint="Features that matter; the rest carry no signal."),
                Field("degree", "integer", "Degree", 2, 1, 3, hint="Highest power of any term."), self.noise_field]

    def _check_shapes(self, shapes, p):
        return [f"informative features ({p['informative']}) cannot exceed features ({p['n_features']})"] if p["informative"] > p["n_features"] else []

    def _make(self, shapes, p, rng, split, concept):
        f, k, deg = _nf(p), p["informative"], p["degree"]
        c1 = concept.standard_normal(k)
        c2 = concept.standard_normal((k, k)) * 0.5
        c3 = concept.standard_normal(k) * 0.3
        X = rng.standard_normal((p["rows"], f))
        Z = X[:, :k]
        y = Z @ c1
        if deg >= 2:
            y = y + np.einsum("ni,ij,nj->n", Z, c2, Z)
        if deg >= 3:
            y = y + (Z**3) @ c3
        return Data(X, y + p["noise"] * rng.standard_normal(p["rows"]), "regression")


class LinearRegression(_Regressor):
    name, label = "tabular_linear_regression", "Linear"
    description = "The target is a random linear function of the features plus noise: the simplest regression."
    classes = ("regression",)

    def _fields(self):
        return [_NF, self.noise_field]

    def _make(self, shapes, p, rng, split, concept):
        w, b = concept.standard_normal(_nf(p)), concept.standard_normal()
        X = rng.standard_normal((p["rows"], _nf(p)))
        return Data(X, X @ w + b + p["noise"] * rng.standard_normal(p["rows"]), "regression")


for _g in (Blobs(), Moons(), Xor(), LinearBoundary(), Clusters(), RandomNetwork(), SineMix(), Polynomial(), LinearRegression()):
    register(_g)
