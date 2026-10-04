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
