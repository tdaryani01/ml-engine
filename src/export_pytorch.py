# src/export_pytorch.py
"""Export a trained dense model (a ModelSerializer .npz: binary_classification, regression, multi_class) as a standalone PyTorch module.

    python -m src.export_pytorch model.npz out_dir

writes into out_dir:
    model.py         the network as plain torch code (no dependency on this project); ``load()`` returns it in eval mode
    weights.pt       the trained parameters and the input/output scaling, as a torch state_dict (float32)
    preprocess.json  what the module expects and returns, and anything the export could not carry over

The module takes RAW features (N, n_features), standardises them with the training mean/std, and returns the prediction in the units the
task is about: a probability for binary_classification, class probabilities for multi_class, values in the original units for regression.
The archive is read with numpy alone (no engine imports), and torch is needed only to write weights.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

FORMAT_VERSION = 1
LEAKY_SLOPE = 0.01        # BaseNeuralNetwork.leaky_relu
BN_EPS = 1e-5             # BaseNeuralNetwork.eps
HEADS = {"binary_classification": "sigmoid", "multi_class": "softmax", "regression": "identity"}
OUTPUT_TEXT = {
    "sigmoid": "probability of the positive class, shape (N, 1)",
    "softmax": "class probabilities, shape (N, n_classes); class k is the label k the engine trained with",
    "identity": "predicted values in the original units of the target, shape (N, n_outputs)",
}


class ExportError(ValueError):
    """The archive cannot be exported (not a dense model, or something the module needs is missing)."""


def _model_type(archive) -> str:
    raw = str(archive["model_type"]).strip()
    if "ModelType." in raw:
        raw = raw.split("ModelType.")[-1]
    return raw.lower()


def _flat(archive, key: str, n: int):
    """archive[key] as a float64 vector of length n, or None when it is not stored."""
    if key not in archive:
        return None
    v = np.asarray(archive[key], dtype=np.float64).reshape(-1)
    if v.size != n:
        raise ExportError(f"{key} has {v.size} values, the model needs {n}")
    return v


def read_archive(path: str) -> dict:
    """The parts of a saved dense model that the PyTorch module needs, validated."""
    with np.load(path, allow_pickle=False) as archive:
        mtype = _model_type(archive)
        if mtype not in HEADS:
            raise ExportError(f"'{mtype}' cannot be exported yet: only {', '.join(sorted(HEADS))} (the dense families)")
        sizes = [int(v) for v in np.asarray(archive["layer_sizes"]).reshape(-1)]
        if len(sizes) < 2:
            raise ExportError(f"layer_sizes {sizes} is not a network")
        n_layers = len(sizes) - 1
        use_bn = bool(archive["use_batch_norm"]) if "use_batch_norm" in archive else False
        warnings: list[str] = []
        weights, biases = [], []
        for i in range(n_layers):
            for key in (f"w_{i}", f"b_{i}"):
                if key not in archive:
                    raise ExportError(f"{key} is missing from the archive")
            w = np.asarray(archive[f"w_{i}"], dtype=np.float64)
            if w.shape != (sizes[i], sizes[i + 1]):
                raise ExportError(f"w_{i} has shape {w.shape}, layer_sizes say {(sizes[i], sizes[i + 1])}")
            weights.append(w)
            biases.append(_flat(archive, f"b_{i}", sizes[i + 1]))
        bn = []
        if use_bn:
            for i in range(n_layers - 1):
                layer = {}
                for name, key in (("gamma", "gamma"), ("beta", "beta"), ("mean", "rmean"), ("var", "rvar")):
                    layer[name] = _flat(archive, f"{key}_{i}", sizes[i + 1])
                    if layer[name] is None:
                        raise ExportError(f"{key}_{i} is missing: the archive says batch norm is on")
                bn.append(layer)
        x_mean, x_std = _flat(archive, "mean", sizes[0]), _flat(archive, "std", sizes[0])
        if x_mean is None or x_std is None:
            x_mean, x_std = None, None
            warnings.append("the archive holds no input mean/std (it was saved without a data provider): the module expects features that are already standardised")
        y_mean, y_std = None, None
        if mtype == "regression":
            y_mean, y_std = _flat(archive, "y_mean", sizes[-1]), _flat(archive, "y_std", sizes[-1])
            if y_mean is None or y_std is None:
                y_mean, y_std = None, None
                warnings.append("the archive holds no target mean/std (saved before they were stored): the module returns STANDARDISED predictions; multiply by the training std and add the training mean of the target")
        return {
            "model_type": mtype, "head": HEADS[mtype], "sizes": sizes, "use_bn": use_bn,
            "bn_momentum": float(archive["bn_momentum"]) if "bn_momentum" in archive else 0.9,
            "weights": weights, "biases": biases, "bn": bn, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "warnings": warnings,
        }


def build_state(spec: dict) -> dict:
    """The module's state_dict as float64 numpy arrays (export() writes them as float32 tensors)."""
    s: dict = {}
    for i, (w, b) in enumerate(zip(spec["weights"], spec["biases"])):
        s[f"linears.{i}.weight"] = np.ascontiguousarray(w.T)      # engine: (in, out); torch Linear: (out, in)
        s[f"linears.{i}.bias"] = b
    for i, layer in enumerate(spec["bn"]):
        s[f"norms.{i}.weight"] = layer["gamma"]
        s[f"norms.{i}.bias"] = layer["beta"]
        s[f"norms.{i}.running_mean"] = layer["mean"]
        s[f"norms.{i}.running_var"] = layer["var"]
        s[f"norms.{i}.num_batches_tracked"] = np.zeros((), dtype=np.int64)
    n_in = spec["sizes"][0]
    s["x_mean"] = spec["x_mean"] if spec["x_mean"] is not None else np.zeros(n_in)
    s["x_std"] = spec["x_std"] if spec["x_std"] is not None else np.ones(n_in)
    if spec["y_mean"] is not None:
        s["y_mean"], s["y_std"] = spec["y_mean"], spec["y_std"]
    return s


_SOURCE = '''\
"""A trained Anascope @MODEL_TYPE@ model as a plain PyTorch module (no Anascope code needed).

    from model import load
    net = load()                      # reads weights.pt next to this file; eval mode
    y = net(torch.tensor(x_raw))      # x_raw: (N, @N_IN@) float32 raw features -> @OUTPUT@

Inputs are standardised inside the module with the training mean/std (buffers x_mean, x_std).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = @SIZES@
USE_BATCH_NORM = @USE_BN@


class AnascopeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(i, o) for i, o in zip(SIZES[:-1], SIZES[1:])])
        if USE_BATCH_NORM:
            self.norms = nn.ModuleList([nn.BatchNorm1d(o, eps=@EPS@, momentum=@MOMENTUM@) for o in SIZES[1:-1]])
        self.register_buffer("x_mean", torch.zeros(SIZES[0]))
        self.register_buffer("x_std", torch.ones(SIZES[0]))
@Y_BUFFERS@
    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        last = len(self.linears) - 1
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i == last:
                break
            if USE_BATCH_NORM:
                x = self.norms[i](x)
            x = F.leaky_relu(x, @SLOPE@)
@HEAD@


def load(weights_path=None, device="cpu"):
    weights_path = weights_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pt")
    model = AnascopeModel()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device).eval()
'''

_HEAD_CODE = {
    "sigmoid": "        return torch.sigmoid(x)",
    "softmax": "        return torch.softmax(x, dim=1)",
}


def generate_source(spec: dict) -> str:
    head = spec["head"]
    y_buffers, head_code = "", _HEAD_CODE.get(head, "        return x")
    if head == "identity" and spec["y_mean"] is not None:
        y_buffers = '        self.register_buffer("y_mean", torch.zeros(SIZES[-1]))\n        self.register_buffer("y_std", torch.ones(SIZES[-1]))\n'
        head_code = "        return x * self.y_std + self.y_mean"
    out = {
        "@MODEL_TYPE@": spec["model_type"], "@N_IN@": str(spec["sizes"][0]), "@OUTPUT@": OUTPUT_TEXT[head],
        "@SIZES@": repr(spec["sizes"]), "@USE_BN@": repr(bool(spec["use_bn"])), "@EPS@": repr(BN_EPS),
        "@MOMENTUM@": repr(round(1.0 - spec["bn_momentum"], 12)),     # engine: running = m * running + (1 - m) * batch; torch: running = (1 - momentum) * running + momentum * batch
        "@SLOPE@": repr(LEAKY_SLOPE), "@Y_BUFFERS@": y_buffers, "@HEAD@": head_code,
    }
    text = _SOURCE
    for token, value in out.items():
        text = text.replace(token, value)
    return text


def export(npz_path: str, out_dir: str) -> dict:
    """Write model.py, weights.pt and preprocess.json for the saved dense model; returns what preprocess.json says."""
    try:
        import torch
    except ImportError as exc:
        raise ExportError("PyTorch is needed to write weights.pt: pip install torch") from exc
    spec = read_archive(npz_path)
    os.makedirs(out_dir, exist_ok=True)
    state = {k: torch.from_numpy(v.astype(np.int64 if v.dtype == np.int64 else np.float32)) for k, v in build_state(spec).items()}
    torch.save(state, os.path.join(out_dir, "weights.pt"))
    with open(os.path.join(out_dir, "model.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_source(spec))
    info = {
        "format_version": FORMAT_VERSION, "model_type": spec["model_type"], "layer_sizes": spec["sizes"], "use_batch_norm": spec["use_bn"],
        "input": f"raw features, shape (N, {spec['sizes'][0]}), float32", "standardises_inputs": spec["x_mean"] is not None,
        "output": OUTPUT_TEXT[spec["head"]] + ("" if spec["head"] != "identity" or spec["y_mean"] is not None else " (STANDARDISED: the target scaling was not stored)"),
        "dtype": "float32", "source": os.path.basename(npz_path), "warnings": spec["warnings"],
    }
    with open(os.path.join(out_dir, "preprocess.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a trained dense Anascope model (.npz) as a PyTorch module.")
    ap.add_argument("npz")
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    try:
        info = export(a.npz, a.out_dir)
    except (ExportError, OSError, KeyError) as exc:
        print(f"export-pytorch: {exc}", file=sys.stderr)
        return 2
    for w in info["warnings"]:
        print(f"export-pytorch: WARNING: {w}", file=sys.stderr)
    print(f"export-pytorch: {info['model_type']} {info['layer_sizes']} -> {a.out_dir} (model.py, weights.pt, preprocess.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
