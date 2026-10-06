"""Generators for the models that read images (cnn) and sequences (mhsa).

cnn: ``X`` (N, channels, height, width) float and ``y`` (N,) integer class, in an ``.npz`` the image loader reads. Shapes: ``input_shape``
([channels, height, width]) and ``num_classes``.

mhsa: ``X`` (N, T, d_model) float and ``y`` (N, action_dim) float, in an ``.npz`` the sequence loader reads. Shapes: ``d_model``,
``action_dim`` and ``max_seq_len``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.generators.base import ArrayData, Field
from src.generators.closed_loop import _ArrayGenerator


def _chw(shapes: dict[str, Any]) -> tuple[int, int, int]:
    c, h, w = shapes.get("input_shape") or [1, 16, 16]
    return int(c), int(h), int(w)


class _Images(_ArrayGenerator):
    classes = ("cnn",)

    def _fields(self):
        return [Field("images", "integer", "Images", 1200, 50, 100_000, hint="How many images."),
                Field("noise", "number", "Noise", 0.2, 0.0, 2.0, hint="Pixel noise added to every image: more is harder.")]

    def _check_shapes(self, shapes, p):
        problems: list[str] = []
        shp = shapes.get("input_shape")
        if shp is not None and (not isinstance(shp, (list, tuple)) or len(shp) != 3 or min(shp) < 1):
            problems.append("input_shape must be [channels, height, width]")
        elif shp is not None and min(shp[1], shp[2]) < 8:
            problems.append("images need at least 8 x 8 pixels")
        n = int(shapes.get("num_classes") or 0)
        if n and n > self.max_classes:
            problems.append(f"this generator draws at most {self.max_classes} classes, the model has {n}")
        if n == 1:
            problems.append("an image classifier needs at least 2 classes")
        return problems

    max_classes = 6

    def _classes(self, shapes) -> int:
        return int(shapes.get("num_classes") or 4)

    def _make(self, shapes, p, rng, split, concept):
        c, h, w = _chw(shapes)
        k = self._classes(shapes)
        y = rng.integers(0, k, size=p["images"])
        imgs = np.stack([self._draw(int(label), h, w, rng) for label in y])  # (N, h, w) in 0..1
        x = np.repeat(imgs[:, None, :, :], c, axis=1) * (1.0 + 0.1 * concept.standard_normal((1, c, 1, 1)))
        x = x + p["noise"] * rng.standard_normal(x.shape)
        return ArrayData({"X": x.astype(np.float32), "y": y.astype(np.int32)})

    def baseline(self, shapes, params, seed):
        k = self._classes(shapes)
        return {"metric": "accuracy", "do_nothing": 1.0 / k, "best_possible": None,
                "note": f"Chance with {k} equally common classes. A learner must beat it clearly."}


class ImageShapes(_Images):
    name, label = "image_shapes", "Shapes"
    description = "Small drawings (horizontal bar, vertical bar, square, cross, diagonal, ring) at a random position and size. The class is which shape."

    def _draw(self, label, h, w, rng):
        img = np.zeros((h, w))
        s = int(rng.integers(max(3, min(h, w) // 4), max(4, min(h, w) // 2) + 1))
        y0, x0 = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        ys, xs = np.mgrid[0:s, 0:s]
        mid = s // 2
        if label == 0:
            patch = (ys == mid)
        elif label == 1:
            patch = (xs == mid)
        elif label == 2:
            patch = (ys == 0) | (ys == s - 1) | (xs == 0) | (xs == s - 1)
        elif label == 3:
            patch = (ys == mid) | (xs == mid)
        elif label == 4:
            patch = (ys == xs)
        else:
            r = np.hypot(ys - (s - 1) / 2, xs - (s - 1) / 2)
            patch = np.abs(r - (s - 1) / 2) < 0.8
        img[y0:y0 + s, x0:x0 + s] = patch.astype(float)
        return img


class ImageGratings(_Images):
    name, label = "image_gratings", "Stripes"
    description = "Stripe patterns whose direction is the class (evenly spaced angles), with a random spacing and offset. Position tells nothing: only the texture does."
    max_classes = 12

    def _fields(self):
        return [Field("images", "integer", "Images", 1200, 50, 100_000, hint="How many images."),
                Field("noise", "number", "Noise", 0.9, 0.0, 2.0, hint="Pixel noise added to every image: more is harder.")]

    def _draw(self, label, h, w, rng):
        k = getattr(self, "_k", 4)
        theta = np.pi * label / k + 0.12 * rng.standard_normal()
        freq = rng.uniform(0.18, 0.3)
        ys, xs = np.mgrid[0:h, 0:w]
        return 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xs * np.cos(theta) + ys * np.sin(theta)) + rng.uniform(0, 2 * np.pi))

    def _make(self, shapes, p, rng, split, concept):
        self._k = self._classes(shapes)
        return super()._make(shapes, p, rng, split, concept)


class _Sequences(_ArrayGenerator):
    classes = ("mhsa",)

    def _fields(self):
        return [Field("sequences", "integer", "Sequences", 1200, 50, 100_000, hint="How many sequences."),
                Field("noise", "number", "Noise", 0.1, 0.0, 1.0, hint="Noise added to every token: more is harder.")]

    def _dims(self, shapes):  # type: ignore[override]
        return int(shapes.get("d_model") or 16), int(shapes.get("action_dim") or 4)

    def _check_shapes(self, shapes, p):
        d, a = self._dims(shapes)
        t = int(shapes.get("max_seq_len") or 8)
        problems = []
        if a > d:
            problems.append(f"the answer has {a} values but each token only {d}")
        if t < 4:
            problems.append("sequences need at least 4 steps")
        return problems


class SeqRecall(_Sequences):
    name, label = "seq_recall", "Cue recall"
    description = ("Each sequence is a few tokens carrying values, then a cue that names one of them by position; the answer is the named token's value. "
                   "The model must attend to the right place: guessing the average of the tokens scores poorly.")

    def _make(self, shapes, p, rng, split, concept):
        d, a = self._dims(shapes)
        t = int(shapes.get("max_seq_len") or 8)
        n = p["sequences"]
        x = rng.standard_normal((n, t, d)) * 0.5
        values = rng.standard_normal((n, t - 1, a))
        x[:, : t - 1, :a] = values  # the first a channels of each token carry its value
        which = rng.integers(0, t - 1, size=n)
        x[:, t - 1, :] = 0.0  # the cue token: a one-hot of the position, in the channels after the value
        x[np.arange(n), t - 1, a + which] = 3.0
        y = values[np.arange(n), which]
        x = x + p["noise"] * rng.standard_normal(x.shape)
        return ArrayData({"X": x.astype(np.float32), "y": y.astype(np.float32)})

    def _check_shapes(self, shapes, p):
        d, a = self._dims(shapes)
        t = int(shapes.get("max_seq_len") or 8)
        extra = [f"the cue names one of {t - 1} positions, which needs {t - 1} spare channels beyond the {a} answer values: d_model is {d}"] if d - a < t - 1 else []
        return super()._check_shapes(shapes, p) + extra

    def baseline(self, shapes, params, seed):
        d = self.build(shapes, {**(params or {}), "sequences": 800}, seed, "heldout").arrays
        y = d["y"]
        t = d["X"].shape[1]
        mean_tok = d["X"][:, : t - 1, : y.shape[1]].mean(axis=1)  # a model that ignores the cue answers with the average token
        return {"metric": "mse", "do_nothing": float(min(np.mean(y ** 2), np.mean((y - mean_tok) ** 2))), "best_possible": float((self.resolve(params)["noise"]) ** 2),
                "note": "The error of answering zero or the average of the tokens. A learner must attend to the cued token to get far below it."}


class SeqRunningTotal(_Sequences):
    name, label = "seq_marked_total", "Marked total"
    description = "Each token carries a value and a flag; the answer is the sum of the values of the flagged tokens. Order does not matter, but every token must be read."

    def _make(self, shapes, p, rng, split, concept):
        d, a = self._dims(shapes)
        t = int(shapes.get("max_seq_len") or 8)
        n = p["sequences"]
        x = rng.standard_normal((n, t, d)) * 0.3
        vals = rng.standard_normal((n, t, a))
        flag = (rng.random((n, t)) < 0.4)
        x[:, :, :a] = vals
        x[:, :, a] = np.where(flag, 2.0, -2.0) if d > a else 0.0
        y = (vals * flag[:, :, None]).sum(axis=1) / np.sqrt(t)
        x = x + p["noise"] * rng.standard_normal(x.shape)
        return ArrayData({"X": x.astype(np.float32), "y": y.astype(np.float32)})

    def _check_shapes(self, shapes, p):
        d, a = self._dims(shapes)
        return super()._check_shapes(shapes, p) + ([f"a flag channel needs d_model above the answer size ({d} <= {a})"] if d <= a else [])

    def baseline(self, shapes, params, seed):
        y = self.build(shapes, {**(params or {}), "sequences": 800}, seed, "heldout").arrays["y"]
        return {"metric": "mse", "do_nothing": float(np.mean(y ** 2)), "best_possible": float(self.resolve(params)["noise"] ** 2),
                "note": "The error of answering zero. A learner must read the flags to get far below it."}


from src.generators.base import register  # noqa: E402

for _g in (ImageShapes(), ImageGratings(), SeqRecall(), SeqRunningTotal()):
    register(_g)
