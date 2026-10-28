"""src/export_pytorch.py: a trained dense model, saved by ModelSerializer, gives the same predictions as the exported PyTorch module.

The reference is the engine's own forward pass (model.predict on standardised features, de-standardised for regression). The math is compared in float64
(tight); the files export() writes are float32, compared with a float32 tolerance.
"""
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from src import export_pytorch as ex  # noqa: E402
from src.model_factory import ModelFactory  # noqa: E402
from src.serializer import ModelSerializer  # noqa: E402

N_IN = 5


def _config(model_type, use_bn):
    return {"architecture": {"model_type": model_type, "use_batch_norm": use_bn, "bn_momentum": 0.9, "p_dropout": 0.0}, "optimization": {"optimizer": "adam", "learning_rate": 0.01}}


def _trained_like(model_type, use_bn, sizes, seed=0):
    """A model with every parameter and running statistic away from its initial value, so no term of the forward pass is trivially zero or one."""
    rng = np.random.default_rng(seed)
    model = ModelFactory.create_model(model_type=model_type, layer_sizes=sizes, lr=0.01, optimizer="adam", lam_l1=0.0, lam_l2=0.0, p_dropout=0.0, use_batch_norm=use_bn, bn_momentum=0.9)
    for i in range(len(model.weights)):
        model.weights[i] = rng.normal(0, 0.7, model.weights[i].shape)
        model.biases[i] = rng.normal(0, 0.3, model.biases[i].shape)
    if use_bn:
        for i in range(len(model.gammas)):
            model.gammas[i] = rng.uniform(0.5, 1.5, model.gammas[i].shape)
            model.betas[i] = rng.normal(0, 0.3, model.betas[i].shape)
            model.running_means[i] = rng.normal(0, 0.5, model.running_means[i].shape)
            model.running_vars[i] = rng.uniform(0.5, 2.0, model.running_vars[i].shape)
    return model


def _save(tmp_path, model, model_type, use_bn, provider=True):
    path = str(tmp_path / "model.npz")
    mean, std = np.linspace(-1, 2, N_IN), np.linspace(0.5, 3, N_IN)
    prov = types.SimpleNamespace(mean=mean, std=std) if provider else None
    ModelSerializer.save_model(model, _config(model_type, use_bn), data_provider=prov, file_path=path)
    return path, mean, std


def _reference(model, x_raw, mean, std):
    out = model.predict((x_raw - mean) / std)
    if getattr(model, "y_mean", None) is not None:
        out = out * model.y_std + model.y_mean
    return out


def _module(spec_or_dir):
    """The generated module from model.py in a directory."""
    s = importlib.util.spec_from_file_location("exported_model", os.path.join(spec_or_dir, "model.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CASES = [("binary_classification", (N_IN, 8, 6, 1)), ("multi_class", (N_IN, 8, 6, 4)), ("regression", (N_IN, 8, 6, 2)), ("binary_classification", (N_IN, 1))]


@pytest.mark.parametrize("use_bn", [True, False])
@pytest.mark.parametrize("model_type,sizes", CASES)
def test_the_exported_module_predicts_what_the_engine_predicts(tmp_path, model_type, sizes, use_bn):
    if len(sizes) == 2:
        use_bn = False        # no hidden layer, so nothing to normalise
    model = _trained_like(model_type, use_bn, sizes)
    if model_type == "regression":
        model.y_mean, model.y_std = np.array([3.0, -2.0]), np.array([4.0, 0.5])
    path, mean, std = _save(tmp_path, model, model_type, use_bn)
    x = np.random.default_rng(7).normal(0.5, 2.0, (40, N_IN))
    want = _reference(model, x, mean, std)

    # the math, in float64
    spec = ex.read_archive(path)
    mod = _module_from_source(tmp_path, ex.generate_source(spec))
    net = mod.AnascopeModel().double().eval()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in ex.build_state(spec).items()})
    got = net(torch.from_numpy(x)).detach().numpy()
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)

    # the files, in float32
    out = tmp_path / "out"
    info = ex.export(path, str(out))
    assert sorted(os.listdir(out)) == ["model.py", "preprocess.json", "weights.pt"] and info["warnings"] == []
    net32 = _module(str(out)).load(str(out / "weights.pt"))
    got32 = net32(torch.from_numpy(x.astype(np.float32))).detach().numpy()
    np.testing.assert_allclose(got32, want, rtol=2e-4, atol=2e-5)


def _module_from_source(tmp_path, source):
    d = tmp_path / "gen"
    d.mkdir(exist_ok=True)
    (d / "model.py").write_text(source, encoding="utf-8")
    return _module(str(d))


def test_binary_probabilities_and_class_probabilities_are_probabilities(tmp_path):
    for model_type, sizes in (("binary_classification", (N_IN, 6, 1)), ("multi_class", (N_IN, 6, 3))):
        model = _trained_like(model_type, True, sizes)
        path, _, _ = _save(tmp_path, model, model_type, True)
        ex.export(path, str(tmp_path / model_type))
        net = _module(str(tmp_path / model_type)).load()
        y = net(torch.randn(10, N_IN)).detach().numpy()
        assert ((y >= 0) & (y <= 1)).all()
        if model_type == "multi_class":
            np.testing.assert_allclose(y.sum(axis=1), 1.0, rtol=1e-5)


def test_the_serializer_keeps_the_regression_target_scaling_and_the_model_comes_back_with_it(tmp_path):
    model = _trained_like("regression", False, (N_IN, 6, 1))
    model.y_mean, model.y_std = np.array([10.0]), np.array([2.5])
    path, _, _ = _save(tmp_path, model, "regression", False)
    back = ModelSerializer.load_model(path)
    np.testing.assert_array_equal(back.y_mean, [10.0])
    np.testing.assert_array_equal(back.y_std, [2.5])


def test_an_old_regression_archive_without_the_target_scaling_exports_with_a_warning_not_silently(tmp_path):
    model = _trained_like("regression", False, (N_IN, 6, 1))
    path, _, _ = _save(tmp_path, model, "regression", False)        # y_mean is None on this model: nothing stored, like an archive saved before
    info = ex.export(path, str(tmp_path / "old"))
    assert any("target mean/std" in w for w in info["warnings"]) and "STANDARDISED" in info["output"]
    assert "y_mean" not in torch.load(str(tmp_path / "old/weights.pt"), weights_only=True)


def test_an_archive_without_input_scaling_warns_that_features_must_be_standardised(tmp_path):
    model = _trained_like("binary_classification", False, (N_IN, 6, 1))
    path, _, _ = _save(tmp_path, model, "binary_classification", False, provider=False)
    info = ex.export(path, str(tmp_path / "noprov"))
    assert info["standardises_inputs"] is False and any("input mean/std" in w for w in info["warnings"])


@pytest.mark.parametrize("model_type", ["closed_loop", "navigate"])
def test_a_family_that_cannot_be_exported_is_refused_by_name(tmp_path, model_type):
    path = str(tmp_path / "x.npz")
    np.savez(path, model_type=model_type, layer_sizes=np.array([3, 2]))
    with pytest.raises(ex.ExportError, match="dense families"):
        ex.read_archive(path)


def test_an_archive_with_missing_batch_norm_arrays_is_refused_not_guessed(tmp_path):
    model = _trained_like("binary_classification", True, (N_IN, 6, 1))
    path, _, _ = _save(tmp_path, model, "binary_classification", True)
    with np.load(path) as a:
        keep = {k: a[k] for k in a.files if not k.startswith("rvar_")}
    np.savez(path, **keep)
    with pytest.raises(ex.ExportError, match="rvar_0"):
        ex.read_archive(path)


def test_the_exported_module_takes_what_numpy_and_pandas_hand_over_and_a_single_row(tmp_path):
    model = _trained_like("multi_class", True, (N_IN, 6, 3))
    path, mean, std = _save(tmp_path, model, "multi_class", True)
    ex.export(path, str(tmp_path / "o"))
    net = _module(str(tmp_path / "o")).load()
    x = np.random.default_rng(3).normal(size=(6, N_IN))          # float64, as numpy makes it
    want = _reference(model, x, mean, std)
    with torch.no_grad():
        np.testing.assert_allclose(net(torch.from_numpy(x)).numpy(), want, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(net(torch.from_numpy(x[0])).numpy(), want[:1], rtol=2e-4, atol=2e-5)      # 1-D row, as model.predict accepts


def test_the_written_files_say_what_they_hold(tmp_path):
    model = _trained_like("regression", True, (N_IN, 6, 2))
    model.y_mean, model.y_std = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    path, _, _ = _save(tmp_path, model, "regression", True)
    info = ex.export(path, str(tmp_path / "o"))
    state = torch.load(str(tmp_path / "o/weights.pt"), weights_only=True)
    assert info["dtype"] == "float32"
    assert {v.dtype for k, v in state.items() if "num_batches_tracked" not in k} == {torch.float32}
    assert json.loads((tmp_path / "o/preprocess.json").read_text()) == info
    assert {"y_mean", "y_std", "x_mean", "x_std"} <= set(state)


@pytest.mark.parametrize("make", ["text", "empty", "truncated"])
def test_a_file_that_is_not_a_saved_model_exits_2_with_a_message_not_a_traceback(tmp_path, capsys, make):
    good, _, _ = _save(tmp_path, _trained_like("binary_classification", True, (N_IN, 6, 1)), "binary_classification", True)
    bad = tmp_path / "bad.npz"
    if make == "text":
        bad.write_text("not a model\n")
    elif make == "empty":
        bad.write_bytes(b"")
    else:
        bad.write_bytes(open(good, "rb").read()[:200])
    assert ex.main([str(bad), str(tmp_path / "o")]) == 2
    assert "export-pytorch:" in capsys.readouterr().err


def test_a_model_trained_a_few_steps_with_the_engine_exports_and_agrees(tmp_path):
    """Train, save and export in one go, with the target scaling set by the engine's own preprocess_targets."""
    rng = np.random.default_rng(11)
    x = rng.normal(2.0, 3.0, (300, N_IN))
    y = (x @ rng.normal(size=(N_IN, 1)) * 5.0 + 40.0).reshape(-1, 1)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-24
    xs = (x - mean) / std
    model = ModelFactory.create_model(model_type="regression", layer_sizes=(N_IN, 8, 1), lr=0.01, optimizer="adam", lam_l1=0.0, lam_l2=0.0, p_dropout=0.0, use_batch_norm=True, bn_momentum=0.9)
    y_train, _, _ = model.preprocess_targets(y, y[:20])
    for _ in range(40):
        model.backward(xs, y_train, 0.01)
    path = str(tmp_path / "trained.npz")
    ModelSerializer.save_model(model, _config("regression", True), data_provider=types.SimpleNamespace(mean=mean, std=std), file_path=path)
    ex.export(path, str(tmp_path / "o"))
    net = _module(str(tmp_path / "o")).load()
    want = model.predict(xs) * model.y_std + model.y_mean
    with torch.no_grad():
        got = net(torch.from_numpy(x.astype(np.float32))).numpy()
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-3)
    assert abs(float(got.mean()) - float(y.mean())) < 0.5 * float(y.std())      # the output is in the target's units, not standardised


# ---------------------------------------------------------------- cnn

from config.constants import EngineBackend  # noqa: E402

CNN_PIPELINES = {
    # ConvBlock (conv + relu + pool fused), a standalone Conv2D, a relu layer, flatten
    "block_conv_relu": [{"type": "conv", "in_channels": 2, "out_channels": 4, "kernel_size": 3, "stride": 1, "pad": 1}, {"type": "relu"}, {"type": "pool", "pool_size": 2, "stride": 2},
                        {"type": "conv", "in_channels": 4, "out_channels": 5, "kernel_size": 3, "stride": 1, "pad": 0}, {"type": "relu"}, {"type": "flatten"}],
    # a conv followed by a pool without a relu in between: Conv2D and MaxPool2D stay separate layers
    "conv_pool": [{"type": "conv", "in_channels": 2, "out_channels": 3, "kernel_size": 3, "stride": 2, "pad": 1}, {"type": "pool", "pool_size": 2, "stride": 2}, {"type": "flatten"}],
}
# The pure-NumPy max-pool fallback (conv_dispatch.maxpool_forward, last lines) reshapes by `stride` and ignores pool_size, so it is only right when they are equal:
# an overlapping pool is compared on the native kernels only.
CNN_OVERLAPPING_POOL = [{"type": "conv", "in_channels": 2, "out_channels": 3, "kernel_size": 3, "stride": 1, "pad": 0}, {"type": "pool", "pool_size": 3, "stride": 1}, {"type": "flatten"}]
CNN_INPUT = [2, 12, 10]


def _flat_dim(pipeline):
    from src.model_factory import ModelFactory as MF
    return MF._compute_flattened_dim(CNN_INPUT, pipeline)


def _cnn(pipeline_name, task_type, out_dim, backend=EngineBackend.NUMPY, seed=0, pipeline=None):
    pipeline = pipeline or CNN_PIPELINES[pipeline_name]
    cfg = {"input_shape": CNN_INPUT, "spatial_pipeline": pipeline, "dense_head": [7]}
    model = ModelFactory.create_model(model_type="cnn", layer_sizes=(1, out_dim), backend=backend, cnn_config=cfg, lr=0.01, optimizer="adam", lam_l1=0.0, lam_l2=0.0, task_type=task_type)
    rng = np.random.default_rng(seed)
    for i in range(len(model.weights)):
        model.weights[i][...] = rng.normal(0, 0.4, model.weights[i].shape)
        model.biases[i][...] = rng.normal(0, 0.2, model.biases[i].shape)
    model._sync_restored_weights()
    return model


def _images(n=9, seed=1):
    x = np.random.default_rng(seed).uniform(0, 1, (n, *CNN_INPUT)).astype(np.float32)
    padded = np.zeros((n, CNN_INPUT[0], CNN_INPUT[1], 16), dtype=np.float32)        # the loader pads the width to a multiple of 8
    padded[..., : CNN_INPUT[2]] = x
    return x, padded


def _save_non_dense(tmp_path, model):
    path = str(tmp_path / "m.npz")
    ModelSerializer.save_model(model, {}, data_provider=None, file_path=path)
    return path


@pytest.mark.parametrize("pipeline", sorted(CNN_PIPELINES))
@pytest.mark.parametrize("task_type,out_dim", [("multiclass", 4), ("binary", 1), ("regression", 2)])
def test_the_exported_cnn_predicts_what_the_engine_predicts(tmp_path, pipeline, task_type, out_dim):
    model = _cnn(pipeline, task_type, out_dim)
    x, padded = _images()
    want = np.asarray(model.predict(padded), dtype=np.float64)

    path = _save_non_dense(tmp_path, model)
    spec = ex.read_archive(path)
    mod = _module_from_source(tmp_path, ex.generate_source(spec))
    net = mod.AnascopeModel().double().eval()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in ex.build_state(spec).items()})
    with torch.no_grad():
        got = net(torch.from_numpy(x.astype(np.float64))).numpy()
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)          # the engine computes in float32

    out = tmp_path / "out"
    info = ex.export(path, str(out))
    assert info["model_type"] == "cnn" and info["standardises_inputs"] is False and any("no mean/std" in w for w in info["warnings"])
    net32 = _module(str(out)).load()
    with torch.no_grad():
        got32 = net32(torch.from_numpy(x)).numpy()
        one = net32(torch.from_numpy(x[0])).numpy()                      # a single (C, H, W) image
    np.testing.assert_allclose(got32, want, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(one, want[:1], rtol=1e-3, atol=1e-4)


@pytest.fixture(scope="module")
def native_kernels():
    try:
        m = _cnn("", "multiclass", 3, backend=EngineBackend.NATIVE, pipeline=CNN_PIPELINES["block_conv_relu"])
        m.predict(_images(2)[1])
        _mhsa_for_probe().predict(np.zeros((1, 3, 16), dtype=np.float32))
    except Exception as exc:  # noqa: BLE001  the native library is not built or cannot load here
        pytest.skip(f"the native kernels are not available here ({type(exc).__name__}): build them with build_native.sh")


def test_the_exported_cnn_matches_the_native_kernels_including_an_overlapping_pool(tmp_path, native_kernels):
    for name, pipeline in (("block", CNN_PIPELINES["block_conv_relu"]), ("overlap", CNN_OVERLAPPING_POOL)):
        model = _cnn("", "multiclass", 4, backend=EngineBackend.NATIVE, pipeline=pipeline)
        x, padded = _images()
        want = np.asarray(model.predict(padded), dtype=np.float64)
        sub = tmp_path / name
        sub.mkdir()
        path = _save_non_dense(sub, model)
        ex.export(path, str(sub / "out"))
        with torch.no_grad():
            got = _module(str(sub / "out")).load()(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4, err_msg=name)


def test_a_cnn_saved_for_export_cannot_be_loaded_back_into_the_engine_yet_and_says_so(tmp_path):
    path = _save_non_dense(tmp_path, _cnn("conv_pool", "multiclass", 3))
    with pytest.raises(ValueError, match="saved for export"):
        ModelSerializer.load_model(path)


def test_a_cnn_archive_with_a_gap_is_refused_not_guessed(tmp_path):
    path = _save_non_dense(tmp_path, _cnn("block_conv_relu", "multiclass", 3))
    with np.load(path) as a:
        keep = {k: a[k] for k in a.files}
    keep["layers_json"] = np.array(json.dumps([layer for layer in json.loads(str(keep["layers_json"])) if layer["type"] != "flatten"]))
    np.savez(path, **keep)
    with pytest.raises(ex.ExportError, match="flatten"):
        ex.read_archive(path)


# ---------------------------------------------------------------- mhsa

def _mhsa_cfg(d_model=16, heads=2, layers=2, max_t=8, action=3, mode="continuous", pos=True, input_dim=None, proj=False):
    return {"d_model": d_model, "num_heads": heads, "max_seq_len": max_t, "action_dim": action, "ffn_mult": 4, "num_layers": layers, "use_pos_encoding": pos,
            "use_input_proj": proj, "input_dim": input_dim, "action_mode": mode}


_LIVE_MHSA = []


@pytest.fixture(autouse=True)
def _release_native_tenants():
    """The native library has a handful of tenant slots; every MHSA model with a contract runtime holds one until it is closed."""
    yield
    for m in _LIVE_MHSA:
        runtime = getattr(m, "_contract_runtime", None)
        if runtime is not None:
            runtime.close()
    _LIVE_MHSA.clear()


def _mhsa(cfg, seed=0):
    model = ModelFactory.create_model(model_type="mhsa", layer_sizes=(1,), backend=EngineBackend.NATIVE, mhsa_config=cfg, lr=0.01, optimizer="adam", lam_l1=0.0, lam_l2=0.0)
    _LIVE_MHSA.append(model)
    rng = np.random.default_rng(seed)
    for arrays, centre in ((model.weights, 0.0), (model.biases, 0.0), (model.ln1_gamma, 1.0), (model.ln1_beta, 0.0), (model.ln2_gamma, 1.0), (model.ln2_beta, 0.0)):
        for a in arrays:
            a[...] = rng.normal(centre, 0.5, a.shape)
    for a in (model.pos_embed, model.W_in, model.b_in):
        if a is not None:
            a[...] = rng.normal(0, 0.3, a.shape)
    return model


def _mhsa_for_probe():
    return _mhsa(_mhsa_cfg())


@pytest.fixture(scope="module")
def native_mhsa(native_kernels):
    """MHSA runs on the native kernels only."""


MHSA_CASES = {
    "continuous_pos": _mhsa_cfg(),
    "discrete_pos_proj": _mhsa_cfg(mode="discrete", action=5, input_dim=7, proj=True, layers=1, heads=4),
    "continuous_nopos_deep": _mhsa_cfg(pos=False, layers=3, d_model=8, heads=2, action=2),
    "discrete_square_proj": _mhsa_cfg(mode="discrete", proj=True, input_dim=16),
}


@pytest.mark.parametrize("name", sorted(MHSA_CASES))
@pytest.mark.parametrize("seq_len", [1, 5, 8])
def test_the_exported_mhsa_predicts_what_the_native_engine_predicts(tmp_path, native_mhsa, name, seq_len):
    cfg = MHSA_CASES[name]
    model = _mhsa(cfg)
    d_in = model.input_dim
    x = np.random.default_rng(5).normal(0, 1.0, (6, seq_len, d_in)).astype(np.float32)
    want = np.asarray(model.predict(x), dtype=np.float64)

    path = _save_non_dense(tmp_path, model)
    spec = ex.read_archive(path)
    mod = _module_from_source(tmp_path, ex.generate_source(spec))
    net = mod.AnascopeModel().double().eval()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in ex.build_state(spec).items()})
    with torch.no_grad():
        got = net(torch.from_numpy(x.astype(np.float64))).numpy()
    assert got.shape == want.shape == (6, cfg["action_dim"])
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)

    out = tmp_path / "out"
    info = ex.export(path, str(out))
    assert info["model_type"] == "mhsa" and any("no mean/std" in w for w in info["warnings"])
    with torch.no_grad():
        got32 = _module(str(out)).load()(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got32, want, rtol=2e-3, atol=2e-4)


def test_an_mhsa_module_refuses_a_sequence_longer_than_it_was_trained_for(tmp_path, native_mhsa):
    path = _save_non_dense(tmp_path, _mhsa(_mhsa_cfg(max_t=4)))
    ex.export(path, str(tmp_path / "o"))
    net = _module(str(tmp_path / "o")).load()
    with pytest.raises(ValueError, match="max_seq_len"):
        net(torch.zeros(1, 5, 16))


def test_an_mhsa_archive_without_its_input_projection_is_refused_when_the_widths_differ(tmp_path, native_mhsa):
    path = _save_non_dense(tmp_path, _mhsa(_mhsa_cfg(input_dim=7, proj=True)))
    with np.load(path) as a:
        keep = {k: a[k] for k in a.files if k not in ("W_in", "b_in")}
    np.savez(path, **keep)
    with pytest.raises(ex.ExportError, match="input projection"):
        ex.read_archive(path)


def test_the_command_line_exports_and_a_bad_file_exits_2_with_a_message(tmp_path, capsys):
    model = _trained_like("multi_class", True, (N_IN, 6, 3))
    path, _, _ = _save(tmp_path, model, "multi_class", True)
    assert ex.main([path, str(tmp_path / "cli")]) == 0
    info = json.loads((tmp_path / "cli/preprocess.json").read_text())
    assert info["model_type"] == "multi_class" and info["layer_sizes"] == [N_IN, 6, 3]
    assert ex.main([str(tmp_path / "missing.npz"), str(tmp_path / "bad")]) == 2
    assert "export-pytorch:" in capsys.readouterr().err
