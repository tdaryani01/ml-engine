# testing/test_launcher.py
"""The launcher: build a payload ML engine's parser accepts, run it as a black box, read the ledger and the checkpoints back."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.launcher import (
    CheckpointConfigError,
    FamilyFitSpec,
    FitSpecError,
    SupervisedFitSpec,
    apply_early_stop,
    build_pipeline_payload,
    check_config,
    config_from_checkpoint_bytes,
    read_physical_version,
    run_me_fit,
)


def csv_bytes(n=300, seed=1, labels=2) -> bytes:
    rnd = random.Random(seed)
    rows = ["Feature_1,Feature_2,Target"]
    for _ in range(n):
        y = rnd.randint(0, labels - 1)
        rows.append(f"{rnd.gauss(y * 3, 1):.4f},{rnd.gauss(-y * 3, 1):.4f},{y}.0")
    return "\n".join(rows).encode()


def pipeline_cfg(plate: str, epochs=3) -> dict:
    return {
        "ingestion": {"source_mode": "csv", "data_file_path": plate, "feature_names": "auto", "splits": {"train": 0.7, "val": 0.15}, "drain_on_empty": False},
        "architecture": {"model_type": "binary_classification", "backend": "numpy", "num_classes": 1, "hidden_layers": [8], "p_dropout": 0.0,
                         "use_batch_norm": False, "bn_momentum": 0.9},
        "optimization": {"optimizer": "adam", "epochs_full_dataset": epochs, "batch_size": 16, "learning_rate": 0.05, "seed": 4,
                         "early_stopping_enabled": False, "patience": 10, "min_delta": 1e-4},
        "ledger": {"enabled": True, "branch_id": "main", "checkpoint_every": 25},
        "assembly": {"family_id": "supervised", "template_id": "binary_classification"},
        "diet": {"bindings": []},
    }


def _spec(tmp_path, **kw) -> SupervisedFitSpec:
    plate = tmp_path / "easy.csv"
    plate.write_bytes(csv_bytes())
    return SupervisedFitSpec(model_type="binary_classification", data_path=str(plate), pipeline=pipeline_cfg(str(plate)), model_id="t", **kw)


def test_the_supervised_payload_is_one_ml_engines_own_parser_accepts(tmp_path) -> None:
    from config.config_loader import parse_tm_production_config

    payload = build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")
    cfg = parse_tm_production_config(payload, profile="pipeline")
    assert cfg.ledger.run_config["assembly"]["family_id"] == "supervised"  # the checkpoint will store the config TM sent


def test_a_stretch_number_varies_the_shuffle_seed(tmp_path) -> None:
    seeds = [build_pipeline_payload(_spec(tmp_path, stretch_index=s), work_dir=tmp_path / "w")["optimization"]["seed"] for s in (1, 2, 5)]
    assert seeds == [4, 5, 8]


def test_a_family_payload_keeps_tms_config_for_the_checkpoint_and_drops_its_own_sections(tmp_path) -> None:
    cfg = {"assembly": {"family_id": "closed_loop", "modules": {"encoder": "cnn_upstream"}}, "closed_loop": {"max_steps": 2}, "diet": {"x": 1}, "optimization": {"learning_rate": 0.1}}
    p = build_pipeline_payload(FamilyFitSpec(model_id="m", config=cfg, fit={"run_budget": 9, "stretch_index": 3, "junk": 1}, num_threads=2), work_dir=tmp_path)
    assert "diet" not in p and p["fit"] == {"run_budget": 9, "stretch_index": 3}
    assert p["ledger"]["run_config"] == cfg and p["optimization"]["num_threads"] == 2


def test_the_data_check_describes_the_plate_and_judges_the_config(tmp_path, monkeypatch) -> None:
    plates = tmp_path / "plates"
    plates.mkdir()
    (plates / "easy.csv").write_bytes(csv_bytes())
    monkeypatch.setenv("TM_DESKTOP_PLATES", str(plates))
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"kind": "me_pipeline", "config": pipeline_cfg("easy.csv")}))
    r = check_config(path)
    assert r["ok"] and r["manifest"]["rows"] == 300 and len(r["manifest"]["sha256"]) == 64
    cfg = pipeline_cfg("easy.csv")
    cfg["architecture"]["model_type"] = "regression"
    path.write_text(yaml.safe_dump({"kind": "me_pipeline", "config": cfg}))
    assert any("looks like a classification target" in w for w in check_config(path)["warnings"])
    path.write_text(yaml.safe_dump({"kind": "me_pipeline", "config": pipeline_cfg("nope.csv")}))
    assert not check_config(path)["ok"]


def test_early_stop_rules() -> None:
    cfg = {"optimization": {"patience": 10}}
    assert apply_early_stop(cfg, None) is cfg
    assert apply_early_stop(cfg, {"enabled": True})["optimization"] == {"patience": 10, "early_stopping_enabled": True}
    assert apply_early_stop(cfg, {"enabled": True, "patience": 3})["optimization"]["patience"] == 3
    assert apply_early_stop(cfg, {"enabled": False})["optimization"]["early_stopping_enabled"] is False
    for bad in ({"patience": 5}, {"enabled": True, "patience": 0}, {"enabled": True, "patience": "x"}, "yes"):
        with pytest.raises(FitSpecError):
            apply_early_stop(cfg, bad)


def test_a_black_box_run_yields_snapshots_and_the_checkpoint_gives_back_its_config(tmp_path) -> None:
    snaps = list(run_me_fit(_spec(tmp_path, epochs=2, checkpoint_every=10), work_dir=tmp_path / "w", should_stop=lambda: False))
    assert snaps and snaps[-1]["phase"] in ("train", "es_trip") and snaps[-1]["version"] > 0
    final = snaps[-1]["checkpoint_bytes"]
    assert final
    got = config_from_checkpoint_bytes(final, "chkpt_x")
    assert got["config"]["assembly"]["family_id"] == "supervised" and got["version"] == snaps[-1]["version"]
    ckpt = tmp_path / "c.bin"
    ckpt.write_bytes(final)
    assert read_physical_version(ckpt) == snaps[-1]["version"]


def test_checkpoint_readers_refuse_what_they_cannot_read(tmp_path) -> None:
    assert read_physical_version(None) == 0 and read_physical_version(tmp_path / "missing") == 0
    (tmp_path / "j.json").write_text(json.dumps({"version": 12}))
    assert read_physical_version(tmp_path / "j.json") == 12
    with pytest.raises(CheckpointConfigError) as e:
        config_from_checkpoint_bytes(b"not a checkpoint")
    assert e.value.status == 404


def test_versions_only_go_up_when_a_fit_restores_an_earlier_checkpoint(tmp_path) -> None:
    import json

    from src.launcher import base_version

    ckpt = tmp_path / "c.json"
    ckpt.write_bytes(json.dumps({"version": 125}).encode())
    assert base_version(ckpt) == 125  # a fresh run: the checkpoint's own version
    assert base_version(ckpt, run_head=204) == 204  # the run already reached 204: restore the weights, keep counting up
    assert base_version(ckpt, run_head=100) == 125  # an older head never pulls it back
    assert base_version(None, requested=7) == 7 and base_version(None, run_head=9) == 9 and base_version(None) == 0


def test_an_mhsa_fit_turns_the_contract_list_on_and_other_models_leave_it_off(tmp_path) -> None:
    from src.launcher import SupervisedFitSpec, build_pipeline_payload

    mh = SupervisedFitSpec(model_type="mhsa", data_path=str(tmp_path / "s.npz"), num_classes=4, mhsa={"d_model": 16, "num_heads": 2, "max_seq_len": 8, "action_dim": 4})
    assert build_pipeline_payload(mh, work_dir=tmp_path)["ledger"]["contract_list_enabled"] is True
    mlp = SupervisedFitSpec(model_type="regression", data_path=str(tmp_path / "p.csv"), feature_names=("a",))
    assert build_pipeline_payload(mlp, work_dir=tmp_path)["ledger"]["contract_list_enabled"] is False


def test_the_data_check_reads_an_npz_for_images_and_sequences(tmp_path) -> None:
    import numpy as np

    from src.launcher.check import _check_arrays
    from src.launcher import SupervisedFitSpec

    np.savez(tmp_path / "i.npz", X=np.zeros((10, 1, 8, 8), np.float32), y=np.zeros(10, np.int32))
    cnn = SupervisedFitSpec(model_type="cnn", data_path=str(tmp_path / "i.npz"), num_classes=2)
    ok = _check_arrays(cnn, tmp_path / "i.npz")
    assert ok["ok"] and ok["manifest"]["rows"] == 10 and ok["manifest"]["x_shape"] == [1, 8, 8]
    assert not _check_arrays(SupervisedFitSpec(model_type="mhsa", data_path="x", num_classes=2), tmp_path / "i.npz")["ok"]  # images are not sequences
    np.savez(tmp_path / "bad.npz", X=np.zeros((3, 1, 8, 8)))
    assert "needs the arrays" in _check_arrays(cnn, tmp_path / "bad.npz")["errors"][0]


def test_a_tm_cnn_or_mhsa_config_without_hidden_layers_is_accepted_by_the_parser(tmp_path) -> None:
    from config.config_loader import parse_tm_production_config

    from src.launcher import SupervisedFitSpec, build_pipeline_payload

    for mt, extra in (("cnn", {"cnn": {"input_shape": [1, 8, 8], "dense_head": [8], "spatial_pipeline": [{"type": "flatten"}]}}),
                      ("mhsa", {"mhsa": {"d_model": 16, "num_heads": 2, "num_layers": 1, "ffn_mult": 2, "max_seq_len": 8, "action_dim": 4}})):
        cfg = {"ingestion": {"source_mode": "csv", "data_file_path": "x.npz", "feature_names": "auto", "splits": {"train": 0.7, "val": 0.15}, "drain_on_empty": False},
               "architecture": {"model_type": mt, "num_classes": 4, **extra}, "optimization": {"epochs_full_dataset": 1, "batch_size": 8, "learning_rate": 0.01}}
        spec = SupervisedFitSpec(model_type=mt, data_path=str(tmp_path / "d.npz"), num_classes=4, pipeline=cfg)
        parse_tm_production_config(build_pipeline_payload(spec, work_dir=tmp_path), profile="pipeline")


def test_a_process_slow_to_exit_does_not_fail_a_fit_that_finished(monkeypatch) -> None:
    import subprocess

    from src.launcher import runner

    class Slow:
        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("run_pipeline.py", timeout)

    monkeypatch.setattr(runner, "_STOP_GRACE_S", 0.01)
    monkeypatch.setattr(runner, "_EXIT_GRACE_S", 0.01)
    runner._terminate(Slow())  # no exception: the fit's own outcome stands


def test_a_checkpoint_is_published_only_with_a_validation_measurement_behind_it(tmp_path) -> None:
    """A supervised fit validates when an epoch ends. A cadence checkpoint written before that (here every 3 steps, an epoch is 14) has no validation value: the snapshot used to carry the training loss
    in its ``loss`` field and the checkpoint with it, which then looked like the best validation loss of the run. No such checkpoint is published."""
    snaps = list(run_me_fit(_spec(tmp_path, epochs=3, checkpoint_every=3), work_dir=tmp_path / "w", should_stop=lambda: False))
    cps = [s for s in snaps if s.get("checkpoint_bytes")]
    assert cps, "later checkpoints are still published"
    assert all(s["val_measured"] for s in cps)
    assert all(abs(s["loss"] - s["train_loss"]) > 1e-12 for s in cps), "no published checkpoint stands on a training loss in place of a validation loss"
    first_epoch_end = next(s["step"] for s in snaps if s["val_measured"])
    assert min(s["version"] for s in cps) > first_epoch_end - 1
