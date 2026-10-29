# src/export_pytorch.py
"""Export a trained model (a ModelSerializer .npz) as a standalone PyTorch module: the dense families (binary_classification, regression, multi_class), cnn and mhsa.

    python -m src.export_pytorch model.npz out_dir

writes into out_dir:
    model.py         the network as plain torch code (no dependency on this project); ``load()`` returns it in eval mode
    weights.pt       the trained parameters (and, for the dense families, the input/output scaling), as a torch state_dict (float32)
    preprocess.json  what the module expects and returns, and anything the export could not carry over

Dense: the module takes RAW features (N, n_features), standardises them with the training mean/std, and returns the prediction in the units the
task is about: a probability for binary_classification, class probabilities for multi_class, values in the original units for regression.
CNN: images (N, C, H, W) as the engine's loader gives them, without the width padding; class probabilities (or a probability / values, by task).
MHSA: sequences (N, T, input_dim) with T <= max_seq_len; the action of the last position (tanh-bounded, or raw logits for discrete).
CNN and MHSA inputs are used as given: no mean/std is stored for them.
The archive is read with numpy alone (no engine imports). torch writes weights.pt; where torch is not installed (the Desktop) the same state_dict is written as weights.npz, and the generated load() reads either.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
import zlib

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
    """The archive cannot be exported (not a supported model, or something the module needs is missing or not reproducible)."""


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
    """The parts of a saved model (dense, cnn or mhsa) that the PyTorch module needs, validated. A file that is not a saved model (text, empty, truncated) is an ExportError."""
    try:
        return _read_archive(path)
    except (ExportError, FileNotFoundError):
        raise
    except (zipfile.BadZipFile, EOFError, zlib.error, KeyError, ValueError) as exc:
        raise ExportError(f"{path} is not a readable saved model (.npz): {type(exc).__name__}: {exc}") from exc


def _read_archive(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        mtype = _model_type(archive)
        if mtype == "cnn":
            return _read_cnn(archive)
        if mtype == "mhsa":
            return _read_mhsa(archive)
        if mtype not in HEADS:
            raise ExportError(f"'{mtype}' cannot be exported: only {', '.join(sorted(HEADS))} (the dense families), cnn and mhsa")
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
            "family": "dense", "model_type": mtype, "head": HEADS[mtype], "sizes": sizes, "use_bn": use_bn,
            "bn_momentum": float(archive["bn_momentum"]) if "bn_momentum" in archive else 0.9,
            "weights": weights, "biases": biases, "bn": bn, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "warnings": warnings,
        }


def build_state(spec: dict) -> dict:
    """The module's state_dict as float64 numpy arrays (export() writes them as float32 tensors)."""
    if spec["family"] == "cnn":
        return _state_cnn(spec)
    if spec["family"] == "mhsa":
        return _state_mhsa(spec)
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
    with torch.no_grad():
        y = net(torch.tensor(x_raw))  # x_raw: (N, @N_IN@) raw features, columns in the training file's order -> @OUTPUT@

Inputs (float32 or float64, or a single 1-D row) are standardised inside the module with the training mean/std (buffers x_mean, x_std).
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
        if x.ndim == 1:                       # one row, as model.predict accepts
            x = x.reshape(1, -1)
        x = x.to(self.x_mean.dtype)           # numpy / pandas hand over float64
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
    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = weights_path or next(p for p in (os.path.join(here, "weights.pt"), os.path.join(here, "weights.npz")) if os.path.exists(p))
    model = AnascopeModel()
    if str(weights_path).endswith(".npz"):      # written instead of weights.pt where torch was not installed to export
        import numpy as np
        with np.load(weights_path) as z:
            state = {k: torch.from_numpy(z[k]) for k in z.files}
    else:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
'''

_HEAD_CODE = {
    "sigmoid": "        return torch.sigmoid(x)",
    "softmax": "        return torch.softmax(x, dim=1)",
}


def generate_source(spec: dict) -> str:
    if spec["family"] == "cnn":
        return _source_cnn(spec)
    if spec["family"] == "mhsa":
        return _source_mhsa(spec)
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


def _fill(template: str, tokens: dict) -> str:
    for token, value in tokens.items():
        template = template.replace(token, value)
    return template


def _arr(archive, key: str, shape: tuple) -> np.ndarray:
    if key not in archive:
        raise ExportError(f"{key} is missing from the archive")
    a = np.asarray(archive[key], dtype=np.float64)
    if a.shape != tuple(shape):
        raise ExportError(f"{key} has shape {a.shape}, the model needs {tuple(shape)}")
    return a


# ---------------------------------------------------------------- CNN

SAVED_FORMAT = 2        # what ModelSerializer._save_cnn_or_mhsa writes; a newer layout is refused, not guessed
# CNNNetwork.apply_output_activation compares task_type verbatim: "multiclass" is softmax, "binary" is sigmoid and anything else is the identity. Only the one
# legitimate identity ("regression") is accepted here, so a value the engine would silently treat as raw outputs (a different case, a typo) is an export-time error.
CNN_HEADS = {"multiclass": "softmax", "binary": "sigmoid", "regression": "identity"}
CNN_OUTPUT_TEXT = {
    "softmax": "class probabilities, shape (N, n_classes)",
    "sigmoid": "probability of the positive class, shape (N, 1)",
    "identity": "raw output values, shape (N, n_outputs)",
}


def _check_saved_format(archive) -> None:
    if "format" in archive and int(archive["format"]) > SAVED_FORMAT:
        raise ExportError(f"the archive is format {int(archive['format'])}, newer than this exporter reads ({SAVED_FORMAT}): update the engine")


def _provider_warning(archive) -> list[str]:
    if "provider_standardised" in archive and bool(archive["provider_standardised"]):
        return ["the training data was scaled with a mean/std that this archive does not carry (the data provider had them when the model was saved): the module expects inputs scaled the same way"]
    return []


def _read_cnn(archive) -> dict:
    _check_saved_format(archive)
    try:
        layers = json.loads(str(archive["layers_json"]))
    except (KeyError, ValueError) as exc:
        raise ExportError(f"layers_json is missing or unreadable: {exc}") from exc
    if "task_type" not in archive:
        raise ExportError("task_type is missing from the archive: the output activation cannot be guessed")
    task = str(archive["task_type"])
    if task not in CNN_HEADS:
        raise ExportError(f"task_type '{task}' is not one of {', '.join(sorted(CNN_HEADS))}: the engine would not apply softmax or sigmoid to it, so the output cannot be reproduced")
    convs = [layer for layer in layers if layer.get("type") in ("conv", "conv_block")]
    if not convs:
        raise ExportError("a cnn needs at least one conv layer")
    if not any(layer.get("type") == "flatten" for layer in layers):
        raise ExportError("a cnn needs a flatten layer between the conv layers and the dense head")
    conv_w, conv_b = [], []
    channels = convs[0]["in"]
    for i, layer in enumerate(convs):
        if layer["in"] != channels:
            raise ExportError(f"conv layer {i} takes {layer['in']} channels but the layer before it gives {channels}")
        conv_w.append(_arr(archive, f"w_{i}", (layer["out"], layer["in"], layer["kernel"][0], layer["kernel"][1])))
        conv_b.append(_flat(archive, f"b_{i}", layer["out"]))
        if conv_b[-1] is None:
            raise ExportError(f"b_{i} is missing from the archive")
        channels = layer["out"]
    dense_w, dense_b = [], []
    i = len(convs)
    while f"w_{i}" in archive:
        w = np.asarray(archive[f"w_{i}"], dtype=np.float64)
        if w.ndim != 2:
            raise ExportError(f"w_{i} should be a dense matrix, it has shape {w.shape}")
        if dense_w and w.shape[0] != dense_w[-1].shape[1]:
            raise ExportError(f"dense layer w_{i} takes {w.shape[0]} inputs but the layer before it gives {dense_w[-1].shape[1]}")
        b = _flat(archive, f"b_{i}", w.shape[1])
        if b is None:
            raise ExportError(f"b_{i} is missing from the archive")
        dense_w.append(w)
        dense_b.append(b)
        i += 1
    if not dense_w:
        raise ExportError("a cnn needs a dense head")
    return {
        "family": "cnn", "model_type": "cnn", "task_type": task, "head": CNN_HEADS[task], "layers": layers,
        "conv_w": conv_w, "conv_b": conv_b, "dense_w": dense_w, "dense_b": dense_b,
        "warnings": ["inputs are used as given: no mean/std is stored for a cnn, so repeat any preprocessing (scaling, normalising, the /255 an image loader may do) you did before training"] + _provider_warning(archive),
    }


def _state_cnn(spec: dict) -> dict:
    s: dict = {}
    for i, (w, b) in enumerate(zip(spec["conv_w"], spec["conv_b"])):
        s[f"convs.{i}.weight"] = w                              # engine and torch Conv2d agree: (out, in, kh, kw)
        s[f"convs.{i}.bias"] = b
    for i, (w, b) in enumerate(zip(spec["dense_w"], spec["dense_b"])):
        s[f"denses.{i}.weight"] = np.ascontiguousarray(w.T)     # engine: (in, out); torch Linear: (out, in)
        s[f"denses.{i}.bias"] = b
    return s


_CNN_SOURCE = '''\
"""A trained Anascope cnn model as a plain PyTorch module (no Anascope code needed).

    from model import load
    net = load()                      # reads weights.pt next to this file; eval mode
    with torch.no_grad():
        y = net(torch.tensor(images)) # images: (N, C, H, W) as the engine's loader gives them, WITHOUT the width padding -> @OUTPUT@

No mean/std is stored for a cnn: pass the images the way the model was trained on them.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CONVS = @CONVS@          # (in, out, kernel_h, kernel_w, stride, pad), in forward order
DENSES = @DENSES@        # (in, out)
STEPS = @STEPS@          # the layer list: ("conv_block", conv, pool_size, pool_stride) is conv -> relu -> max-pool


class AnascopeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(i, o, (kh, kw), stride=s, padding=p) for i, o, kh, kw, s, p in CONVS])
        self.denses = nn.ModuleList([nn.Linear(i, o) for i, o in DENSES])

    def forward(self, x):
        if x.ndim == 3:                       # one image (C, H, W)
            x = x.unsqueeze(0)
        x = x.to(self.convs[0].weight.dtype)  # numpy / pandas hand over float64
        image_shape = tuple(x.shape[1:])
        for step in STEPS:
            kind = step[0]
            if kind == "conv_block":
                x = F.max_pool2d(F.relu(self.convs[step[1]](x)), step[2], step[3])
            elif kind == "conv":
                x = self.convs[step[1]](x)
            elif kind == "pool":
                x = F.max_pool2d(x, step[1], step[2])
            elif kind == "relu":
                x = F.relu(x)
            elif kind == "flatten":
                x = torch.flatten(x, 1)
                if x.shape[1] != DENSES[0][0]:
                    raise ValueError(f"images of shape {image_shape} give {x.shape[1]} features after the conv layers, but the dense head was trained on {DENSES[0][0]}: "
                                     "use images of the size the model was trained on (without the width padding)")
        last = len(self.denses) - 1
        for i, dense in enumerate(self.denses):
            x = dense(x)
            if i != last:
                x = F.relu(x)
@HEAD@


def load(weights_path=None, device="cpu"):
    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = weights_path or next(p for p in (os.path.join(here, "weights.pt"), os.path.join(here, "weights.npz")) if os.path.exists(p))
    model = AnascopeModel()
    if str(weights_path).endswith(".npz"):      # written instead of weights.pt where torch was not installed to export
        import numpy as np
        with np.load(weights_path) as z:
            state = {k: torch.from_numpy(z[k]) for k in z.files}
    else:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
'''


def _source_cnn(spec: dict) -> str:
    steps, ci = [], 0
    for layer in spec["layers"]:
        kind = layer["type"]
        if kind == "conv_block":
            steps.append(("conv_block", ci, layer["pool_size"], layer["pool_stride"]))
            ci += 1
        elif kind == "conv":
            steps.append(("conv", ci))
            ci += 1
        elif kind == "pool":
            steps.append(("pool", layer["pool_size"], layer["stride"]))
        elif kind in ("relu", "flatten"):
            steps.append((kind,))
        else:
            raise ExportError(f"unknown cnn layer type '{kind}'")
    convs = [(c["in"], c["out"], c["kernel"][0], c["kernel"][1], c["stride"], c["pad"]) for c in spec["layers"] if c["type"] in ("conv", "conv_block")]
    denses = [(int(w.shape[0]), int(w.shape[1])) for w in spec["dense_w"]]
    head_code = _HEAD_CODE.get(spec["head"], "        return x")
    return _fill(_CNN_SOURCE, {"@OUTPUT@": CNN_OUTPUT_TEXT[spec["head"]], "@CONVS@": repr(convs), "@DENSES@": repr(denses), "@STEPS@": repr(steps), "@HEAD@": head_code})


# ---------------------------------------------------------------- MHSA

LN_EPS = 1e-5     # mhsa_kernels.cpp kEps


def _read_mhsa(archive) -> dict:
    _check_saved_format(archive)
    need = ("d_model", "num_heads", "max_seq_len", "action_dim", "ffn_mult", "num_layers", "input_dim")
    missing = [k for k in need if k not in archive]
    if missing:
        raise ExportError(f"{', '.join(missing)} missing from the archive")
    D, H, T, A, mult, L, d_in = (int(archive[k]) for k in need)
    if D % H:
        raise ExportError(f"d_model {D} is not divisible by num_heads {H}")
    mode = str(archive["action_mode"]).strip().lower() if "action_mode" in archive else "continuous"
    if mode not in ("continuous", "discrete"):
        raise ExportError(f"action_mode '{mode}' is not continuous or discrete")
    ff = D * mult
    use_pos = bool(archive["use_pos_encoding"]) if "use_pos_encoding" in archive else False
    layers = []
    for li in range(L):
        k = 4 * li
        layers.append({
            "ln1_g": _flat(archive, f"ln1_gamma_{li}", D), "ln1_b": _flat(archive, f"ln1_beta_{li}", D),
            "ln2_g": _flat(archive, f"ln2_gamma_{li}", D), "ln2_b": _flat(archive, f"ln2_beta_{li}", D),
            "qkv_w": _arr(archive, f"w_{k}", (D, 3 * D)), "qkv_b": _arr(archive, f"b_{k}", (1, 3 * D)).reshape(-1),
            "o_w": _arr(archive, f"w_{k + 1}", (D, D)), "o_b": _arr(archive, f"b_{k + 1}", (1, D)).reshape(-1),
            "ff1_w": _arr(archive, f"w_{k + 2}", (D, ff)), "ff1_b": _arr(archive, f"b_{k + 2}", (1, ff)).reshape(-1),
            "ff2_w": _arr(archive, f"w_{k + 3}", (ff, D)), "ff2_b": _arr(archive, f"b_{k + 3}", (1, D)).reshape(-1),
        })
        for name in ("ln1_g", "ln1_b", "ln2_g", "ln2_b"):
            if layers[-1][name] is None:
                raise ExportError(f"layer {li}: a layer-norm array is missing from the archive")
    act_w, act_b = _arr(archive, f"w_{4 * L}", (D, A)), _arr(archive, f"b_{4 * L}", (1, A)).reshape(-1)
    pos = _arr(archive, "pos_embed", (T, D)) if use_pos else None
    if use_pos and pos is None:
        raise ExportError("pos_embed is missing from the archive")
    w_in = _arr(archive, "W_in", (d_in, D)) if "W_in" in archive else None
    b_in = _arr(archive, "b_in", (1, D)).reshape(-1) if w_in is not None else None
    if w_in is None and d_in != D:
        raise ExportError(f"input_dim {d_in} differs from d_model {D} but the archive has no input projection (W_in)")
    return {
        "family": "mhsa", "model_type": "mhsa", "d_model": D, "num_heads": H, "max_seq_len": T, "action_dim": A, "ffn_hidden": ff, "num_layers": L,
        "input_dim": d_in, "action_mode": mode, "layers": layers, "act_w": act_w, "act_b": act_b, "pos": pos, "w_in": w_in, "b_in": b_in,
        "head": "tanh" if mode == "continuous" else "logits",
        "warnings": ["inputs are used as given: no mean/std is stored for an mhsa model, so repeat any preprocessing you did before training"] + _provider_warning(archive),
    }


def _state_mhsa(spec: dict) -> dict:
    s: dict = {}
    if spec["w_in"] is not None:
        s["in_proj.weight"], s["in_proj.bias"] = np.ascontiguousarray(spec["w_in"].T), spec["b_in"]
    if spec["pos"] is not None:
        s["pos_embed"] = spec["pos"]
    for li, layer in enumerate(spec["layers"]):
        p = f"blocks.{li}."
        s[p + "ln1.weight"], s[p + "ln1.bias"] = layer["ln1_g"], layer["ln1_b"]
        s[p + "qkv.weight"], s[p + "qkv.bias"] = np.ascontiguousarray(layer["qkv_w"].T), layer["qkv_b"]
        s[p + "attn_out.weight"], s[p + "attn_out.bias"] = np.ascontiguousarray(layer["o_w"].T), layer["o_b"]
        s[p + "ln2.weight"], s[p + "ln2.bias"] = layer["ln2_g"], layer["ln2_b"]
        s[p + "ff1.weight"], s[p + "ff1.bias"] = np.ascontiguousarray(layer["ff1_w"].T), layer["ff1_b"]
        s[p + "ff2.weight"], s[p + "ff2.bias"] = np.ascontiguousarray(layer["ff2_w"].T), layer["ff2_b"]
    s["head.weight"], s["head.bias"] = np.ascontiguousarray(spec["act_w"].T), spec["act_b"]
    return s


_MHSA_SOURCE = '''\
"""A trained Anascope mhsa model as a plain PyTorch module (no Anascope code needed).

    from model import load
    net = load()                      # reads weights.pt next to this file; eval mode
    with torch.no_grad():
        a = net(torch.tensor(seq))    # seq: (N, T, @D_IN@) with T <= @MAX_T@ -> (N, @A@): @OUTPUT@

The action is read from the LAST position of the sequence. No mean/std is stored: pass the sequences the way the model was trained on them.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = @D@
NUM_HEADS = @H@
FFN_HIDDEN = @FF@
NUM_LAYERS = @L@
INPUT_DIM = @D_IN@
ACTION_DIM = @A@
MAX_SEQ_LEN = @MAX_T@
USE_INPUT_PROJ = @USE_PROJ@
USE_POS_ENCODING = @USE_POS@
LN_EPS = @LN_EPS@


class Block(nn.Module):
    """Pre-LN: x = x + attention(ln1(x)); x = x + ffn(ln2(x)); causal attention, GELU (tanh form)."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.attn_out = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.ff1 = nn.Linear(D_MODEL, FFN_HIDDEN)
        self.ff2 = nn.Linear(FFN_HIDDEN, D_MODEL)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (z.reshape(b, t, NUM_HEADS, d // NUM_HEADS).transpose(1, 2) for z in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_out(o.transpose(1, 2).reshape(b, t, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x)), approximate="tanh"))


class AnascopeModel(nn.Module):
    def __init__(self):
        super().__init__()
        if USE_INPUT_PROJ:
            self.in_proj = nn.Linear(INPUT_DIM, D_MODEL)
        if USE_POS_ENCODING:
            self.register_buffer("pos_embed", torch.zeros(MAX_SEQ_LEN, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.head = nn.Linear(D_MODEL, ACTION_DIM)

    def forward(self, x):
        if x.ndim == 2:                       # one sequence (T, input_dim)
            x = x.unsqueeze(0)
        x = x.to(self.head.weight.dtype)      # numpy / pandas hand over float64
        if USE_INPUT_PROJ:
            x = self.in_proj(x)
        if USE_POS_ENCODING:
            if x.shape[1] > MAX_SEQ_LEN:
                raise ValueError(f"sequence length {x.shape[1]} exceeds max_seq_len {MAX_SEQ_LEN} this model was trained with")
            x = x + self.pos_embed[: x.shape[1]]
        for block in self.blocks:
            x = block(x)
        a = self.head(x[:, -1])
@HEAD@


def load(weights_path=None, device="cpu"):
    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = weights_path or next(p for p in (os.path.join(here, "weights.pt"), os.path.join(here, "weights.npz")) if os.path.exists(p))
    model = AnascopeModel()
    if str(weights_path).endswith(".npz"):      # written instead of weights.pt where torch was not installed to export
        import numpy as np
        with np.load(weights_path) as z:
            state = {k: torch.from_numpy(z[k]) for k in z.files}
    else:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
'''


def _source_mhsa(spec: dict) -> str:
    head_code = "        return torch.tanh(a)" if spec["head"] == "tanh" else "        return a"
    output = "actions in (-1, 1) (tanh)" if spec["head"] == "tanh" else "raw logits (apply softmax for probabilities)"
    return _fill(_MHSA_SOURCE, {
        "@D@": str(spec["d_model"]), "@H@": str(spec["num_heads"]), "@FF@": str(spec["ffn_hidden"]), "@L@": str(spec["num_layers"]),
        "@D_IN@": str(spec["input_dim"]), "@A@": str(spec["action_dim"]), "@MAX_T@": str(spec["max_seq_len"]),
        "@USE_PROJ@": repr(spec["w_in"] is not None), "@USE_POS@": repr(spec["pos"] is not None), "@LN_EPS@": repr(LN_EPS),
        "@OUTPUT@": output, "@HEAD@": head_code,
    })


def _info(spec: dict, source: str) -> dict:
    base = {"format_version": FORMAT_VERSION, "model_type": spec["model_type"], "dtype": "float32", "source": source, "warnings": spec["warnings"]}
    if spec["family"] == "cnn":
        layers = spec["layers"]      # the full description of each layer (kernel, stride, pad, pool), in forward order
        return {**base, "task_type": spec["task_type"], "layers": layers, "dense_sizes": [int(spec["dense_w"][0].shape[0])] + [int(w.shape[1]) for w in spec["dense_w"]],
                "input": "images (N, C, H, W), float32, as the engine's loader gives them but without the width padding", "standardises_inputs": False,
                "output": CNN_OUTPUT_TEXT[spec["head"]], "summary": f"cnn, {len(spec['conv_w'])} conv layers, dense {[int(w.shape[1]) for w in spec['dense_w']]}"}
    if spec["family"] == "mhsa":
        return {**base, "d_model": spec["d_model"], "num_heads": spec["num_heads"], "num_layers": spec["num_layers"], "max_seq_len": spec["max_seq_len"],
                "input": f"sequences (N, T, {spec['input_dim']}), float32, T <= {spec['max_seq_len']}", "standardises_inputs": False,
                "output": ("actions in (-1, 1) (tanh)" if spec["head"] == "tanh" else "raw logits") + f", shape (N, {spec['action_dim']}), from the last position",
                "summary": f"mhsa, {spec['num_layers']} layers, d_model {spec['d_model']}, {spec['num_heads']} heads"}
    return {**base, "layer_sizes": spec["sizes"], "use_batch_norm": spec["use_bn"], "input": f"raw features, shape (N, {spec['sizes'][0]}), float32", "standardises_inputs": spec["x_mean"] is not None,
            "output": OUTPUT_TEXT[spec["head"]] + ("" if spec["head"] != "identity" or spec["y_mean"] is not None else " (STANDARDISED: the target scaling was not stored)"),
            "summary": f"{spec['model_type']} {spec['sizes']}"}


def export(npz_path: str, out_dir: str) -> dict:
    """Write model.py, the weights and preprocess.json for the saved model; returns what preprocess.json says.
    The weights are weights.pt when torch is installed here and weights.npz (the same float32 state_dict, as arrays) when it is not: the generated load() reads either."""
    spec = read_archive(npz_path)
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.astype(np.int64 if v.dtype == np.int64 else np.float32) for k, v in build_state(spec).items()}
    try:
        import torch
    except ImportError:
        weights_file = "weights.npz"
        np.savez(os.path.join(out_dir, weights_file), **state)
    else:
        weights_file = "weights.pt"
        torch.save({k: torch.from_numpy(v) for k, v in state.items()}, os.path.join(out_dir, weights_file))
    with open(os.path.join(out_dir, "model.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_source(spec))
    info = _info(spec, os.path.basename(npz_path))
    info["weights_file"] = weights_file
    with open(os.path.join(out_dir, "preprocess.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a trained Anascope model (.npz: dense, cnn or mhsa) as a PyTorch module.")
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
    print(f"export-pytorch: {info['summary']} -> {a.out_dir} (model.py, {info['weights_file']}, preprocess.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
