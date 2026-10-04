# testing/test_closed_loop_run.py
"""A closed-loop model runs through run_pipeline like a supervised one: same ledger documents, same checkpoint rules."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "closed_loop_draw" / "config_draw_smoke.yaml"


def _payload(out: Path, *, steps: int, patience: int = 0, restore: Path | None = None, es_warmup: int = 2) -> dict:
    cfg = yaml.safe_load(SMOKE.read_text())
    cfg["assembly"] = {"family_id": "closed_loop", "template_id": "match_reconstruct",
                       "modules": {"encoder": "cnn_upstream", "policy": "mhsa", "env": "soft_canvas", "loss": "canvas_reconstruction"}}
    cfg["optimization"] = {"learning_rate": 0.002, "seed": 3, "num_threads": 2}
    cfg["meta"] = {"output_dir": str(out)}
    cfg["model_id"] = "cl-test"
    cfg["fit"] = {"run_budget": steps, "resume_from": 0, "patience": patience, "es_warmup": es_warmup, "checkpoint_every": 3, "lr": 0.002}
    cfg["ledger"] = {"path": "ledger", "store_backend": "file_streaming", "run_config": {"assembly": cfg["assembly"], "optimization": dict(cfg["optimization"]), "closed_loop": cfg["closed_loop"]},
                     **({"restore_checkpoint_path": str(restore)} if restore else {})}
    return cfg


def _run(tmp: Path, payload: dict) -> list:
    sys.path.insert(0, str(ROOT))
    from src.ledger import FileLedgerStore

    (tmp / "payload.json").write_text(json.dumps(payload))
    (tmp / "boot.yaml").write_text(f'meta:\n  pipeline_name: "t"\n  stage: "dev"\n  suppress_logging: true\n  logging_level: "warning"\n  output_dir: "{tmp}/out"\ntraining_manager:\n  enabled: false\n  park_when_idle: false\n')
    env = dict(os.environ, ML_ENGINE_TM_PAYLOAD=str(tmp / "payload.json"), ML_ENGINE_BOOT_YAML=str(tmp / "boot.yaml"),
               ML_ENGINE_CONFIG_SOURCE="training_manager", PYTHONUNBUFFERED="1")
    r = subprocess.run([sys.executable, "run_pipeline.py"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return list(FileLedgerStore(str(Path(payload["meta"]["output_dir"]) / "ledger")).scan(1))


def test_a_closed_loop_payload_runs_through_run_pipeline_and_writes_a_supervised_shaped_ledger(tmp_path) -> None:
    docs = _run(tmp_path, _payload(tmp_path / "a", steps=6))
    kinds = [d.doc_type for d in docs]
    assert kinds.count("step.metrics") == 6 and kinds[-1] == "run.end"
    end = docs[-1].body
    assert end["reason"] == "success" and end["epochs_run"] == 6 and end["final_version"] == 6
    cps = [d for d in docs if d.doc_type == "checkpoint"]
    assert cps and all("config" in d.body and "knobs" in d.body and d.body.get("state") for d in cps)  # state + config + knobs, every one
    assert any(d.version == 6 for d in cps)  # the final model is a checkpoint


def test_a_closed_loop_run_can_continue_from_a_checkpoint_and_early_stop_ends_it(tmp_path) -> None:
    from src.ledger import document_to_bytes

    first = _run(tmp_path, _payload(tmp_path / "a", steps=5))
    last = max([d for d in first if d.doc_type == "checkpoint"], key=lambda d: d.version)
    (tmp_path / "restore.ckpt").write_bytes(document_to_bytes(last))
    second = _run(tmp_path, _payload(tmp_path / "b", steps=40, patience=2, es_warmup=1, restore=tmp_path / "restore.ckpt"))
    end = second[-1].body
    assert second[-1].doc_type == "run.end" and end["reason"] in ("es_trip", "success")
    first_val = [d.body["val_loss"] for d in second if d.doc_type == "step.metrics"][0]
    cold_val = [d.body["val_loss"] for d in first if d.doc_type == "step.metrics"][0]
    assert first_val != cold_val  # it started from the trained state, not from scratch
