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


@pytest.mark.parametrize("model_type", ["cnn", "mhsa"])
def test_the_families_that_are_not_dense_are_refused_by_name(tmp_path, model_type):
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


def test_the_command_line_exports_and_a_bad_file_exits_2_with_a_message(tmp_path, capsys):
    model = _trained_like("multi_class", True, (N_IN, 6, 3))
    path, _, _ = _save(tmp_path, model, "multi_class", True)
    assert ex.main([path, str(tmp_path / "cli")]) == 0
    info = json.loads((tmp_path / "cli/preprocess.json").read_text())
    assert info["model_type"] == "multi_class" and info["layer_sizes"] == [N_IN, 6, 3]
    assert ex.main([str(tmp_path / "missing.npz"), str(tmp_path / "bad")]) == 2
    assert "export-pytorch:" in capsys.readouterr().err
