"""ML_DUMP_VAL_PREDS (src/val_dump.py): off by default, one npz of the epoch's validation predictions when set, and a fit's losses are bit-equal with it on or off."""
import numpy as np

from src import val_dump


def test_nothing_is_written_when_the_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(val_dump.ENV, raising=False)
    val_dump.maybe_dump(np.zeros((2, 2)), np.zeros(2), 1.0)
    assert list(tmp_path.iterdir()) == []


def test_the_last_epochs_predictions_targets_and_cost_are_written_and_replace_the_previous(tmp_path, monkeypatch):
    f = tmp_path / "v.npz"
    monkeypatch.setenv(val_dump.ENV, str(f))
    val_dump.maybe_dump(np.ones((3, 2)), np.array([0, 1, 1]), 0.5)
    val_dump.maybe_dump(np.full((3, 2), 0.25), np.array([0, 1, 1]), 0.25)
    z = np.load(f)
    assert np.allclose(z["preds"], 0.25) and list(z["y"]) == [0, 1, 1] and float(z["cost"]) == 0.25 and sorted(p.name for p in tmp_path.iterdir()) == ["v.npz"]
