# testing/test_imitation_run.py
"""The imitation family (tm-brain) runs through run_pipeline: whole-tape split, hard leak check, one ledger shape."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _row(tape: str, i: int) -> dict:
    bit = 1.0 if (i % 2 == 0) else 0.0
    state = {"train": 0.5, "val": 0.5, "gap": 0.0, "d_train": 0.0, "d_val": bit, "d_gap": 0.0, "rho": 0.0, "patience": 20.0,
             "lr": 1e-3, "center_val": 0.5, "center_distance": 0.0, "onset_age": -1.0, "tape_id": tape, "step": i}
    return {"id": f"{tape}-{i}", "instance_id": tape, "created_at": float(i), "state": state,
            "action": "restore_best" if bit > 0.5 else "noop", "reason": "t", "authority": "synthetic", "tape_id": tape, "step": i}


def _corpus(path: Path, *, tapes: int = 12, rows: int = 12, extra: list[dict] | None = None) -> Path:
    lines = [_row(f"tape{t}", i) for t in range(tapes) for i in range(rows)] + (extra or [])
    path.write_text("\n".join(json.dumps(r) for r in lines))
    return path


def _payload(out: Path, corpus: Path, *, steps: int = 10, patience: int = 0, restore: Path | None = None) -> dict:
    return {
        "model_id": "brain-test", "assembly": {"family_id": "imitation"}, "optimization": {"learning_rate": 0.002, "num_threads": 2},
        "meta": {"output_dir": str(out)},
        "fit": {"run_budget": steps, "resume_from": 0, "patience": patience, "es_warmup": 1, "checkpoint_every": 2, "lr": 0.002, "seed": 3},
        "imitation": {"corpus_path": str(corpus), "steps_per_round": 2, "batch_size": 32, "history_k": 2,
                      "architecture": {"d_model": 16, "num_heads": 2, "num_layers": 1, "ffn_mult": 2}},
        "ledger": {"path": "ledger", "store_backend": "file_streaming", "run_config": {"assembly": {"family_id": "imitation"}},
                   **({"restore_checkpoint_path": str(restore)} if restore else {})},
    }


def _run(tmp: Path, payload: dict, *, ok: bool = True):
    sys.path.insert(0, str(ROOT))
    from src.ledger import FileLedgerStore

    (tmp / "payload.json").write_text(json.dumps(payload))
    (tmp / "boot.yaml").write_text(f'meta:\n  pipeline_name: "t"\n  stage: "dev"\n  suppress_logging: true\n  logging_level: "warning"\n  output_dir: "{tmp}/out"\ntraining_manager:\n  enabled: false\n  park_when_idle: false\n')
    env = dict(os.environ, ML_ENGINE_TM_PAYLOAD=str(tmp / "payload.json"), ML_ENGINE_BOOT_YAML=str(tmp / "boot.yaml"),
               ML_ENGINE_CONFIG_SOURCE="training_manager", PYTHONUNBUFFERED="1")
    r = subprocess.run([sys.executable, "run_pipeline.py"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    if not ok:
        return r
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return list(FileLedgerStore(str(Path(payload["meta"]["output_dir"]) / "ledger")).scan(1))


def test_an_imitation_fit_writes_the_shared_ledger_shape_and_its_split_report(tmp_path) -> None:
    docs = _run(tmp_path, _payload(tmp_path / "a", _corpus(tmp_path / "c.jsonl")))
    kinds = [d.doc_type for d in docs]
    assert kinds.count("step.metrics") == 5 and kinds[-1] == "run.end"
    end = docs[-1].body
    assert end["reason"] == "success" and end["epochs_run"] == 5
    assert end["split"]["tape_overlap"] == 0 and end["split"]["n_holdout_tapes"] >= 1 and end["split"]["n_holdout_tapes"] + end["split"]["n_train_tapes"] == 12
    events = {e["event"] for e in end["events"]}
    assert {"split", "windows", "round"} <= events
    cps = [d for d in docs if d.doc_type == "checkpoint"]
    assert cps and all("config" in d.body and "knobs" in d.body and "blob" in d.body["state"] for d in cps)


def test_a_fit_can_continue_from_a_checkpoint(tmp_path) -> None:
    from src.ledger import document_to_bytes

    first = _run(tmp_path, _payload(tmp_path / "a", _corpus(tmp_path / "c.jsonl"), steps=6))
    last = max([d for d in first if d.doc_type == "checkpoint"], key=lambda d: d.version)
    (tmp_path / "r.ckpt").write_bytes(document_to_bytes(last))
    second = _run(tmp_path, _payload(tmp_path / "b", tmp_path / "c.jsonl", steps=6, restore=tmp_path / "r.ckpt"))
    v2 = [d.body["val_loss"] for d in second if d.doc_type == "step.metrics"][0]
    v1 = [d.body["val_loss"] for d in first if d.doc_type == "step.metrics"][0]
    assert v1 != v2


def test_a_corpus_with_one_tape_fails_the_fit(tmp_path) -> None:
    r = _run(tmp_path, _payload(tmp_path / "a", _corpus(tmp_path / "c.jsonl", tapes=1)), ok=False)
    assert r.returncode != 0 and "nothing to hold out" in (r.stdout + r.stderr)


class _Ledger:
    version = 0

    def __init__(self) -> None:
        self.docs: list[tuple[str, object]] = []
        self.store = type("S", (), {"flush": lambda self: None})()

    def push_step_metrics(self, *a, **k): self.docs.append(("step.metrics", a))
    def push_checkpoint_state(self, *a, **k): self.docs.append(("checkpoint", a))
    def push_run_end(self, body): self.docs.append(("run.end", body))


def test_identical_windows_on_two_tapes_are_detected_and_reported_but_tapes_never_overlap() -> None:
    from tm_brain_contracts import DecisionEpisode
    from tm_brain_contracts.windows import split_by_tape

    from src.imitation.fit import fit_imitation

    rows = [_row(f"tape{t}", i) for t in range(10) for i in range(10)]
    rows += [{**_row("tape0", i), "id": f"clone-{i}", "instance_id": "clone"} for i in range(10)]  # same content, another tape
    eps = [DecisionEpisode.from_public(r) for r in rows]
    seed = next(s for s in range(200)
                if ("tape0" in {e.instance_id for e in split_by_tape(eps, seed=s)[1]}) != ("clone" in {e.instance_id for e in split_by_tape(eps, seed=s)[1]}))
    led = _Ledger()
    end = fit_imitation(eps, led, lr=0.002, steps=2, seed=seed, steps_per_round=2,
                        train_kwargs={"history_k": 2, "architecture": {"d_model": 16, "num_heads": 2, "num_layers": 1, "ffn_mult": 2}})
    assert end["split"]["tape_overlap"] == 0
    hits = [e for e in end["events"] if e["event"] == "leakage.DETECTED"]
    assert hits and hits[0]["kind"] == "identical_windows" and hits[0]["overlap"] > 0


def test_the_launcher_hands_the_corpus_to_the_payload_and_the_check_judges_it(tmp_path) -> None:
    import yaml

    from src.launcher.check import check_config
    from src.launcher.payload import build_pipeline_payload
    from src.launcher.staged import spec_from_staged

    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"kind": "me_run", "model_id": "b", "config": {"assembly": {"family_id": "imitation"}}, "fit": {"lr": 0.01}}))
    good = _corpus(tmp_path / "good.jsonl")
    spec = spec_from_staged(cfg, good)
    assert build_pipeline_payload(spec, work_dir=tmp_path)["imitation"]["corpus_path"] == str(good)
    # check_config takes the staged path and the corpus the same way spec_from_staged does
    import src.launcher.check as chk
    ok = chk._check_corpus(spec)
    assert ok["ok"] and ok["manifest"]["tapes"] == 12 and ok["manifest"]["rows"] == 144
    one = chk._check_corpus(spec_from_staged(cfg, _corpus(tmp_path / "one.jsonl", tapes=1)))
    assert not one["ok"]
    assert chk._check_corpus(spec_from_staged(cfg, None))["warnings"]


def test_the_last_checkpoint_of_a_fit_holds_the_best_weights_and_reads_back_as_a_model(tmp_path) -> None:
    from src.launcher.checkpoint import imitation_model_from_checkpoint_bytes
    from src.ledger import document_to_bytes

    docs = _run(tmp_path, _payload(tmp_path / "a", _corpus(tmp_path / "c.jsonl"), steps=10))
    end = docs[-1].body
    final = [d for d in docs if d.doc_type == "checkpoint" and d.version == end["final_version"]][-1]
    best = [d for d in docs if d.doc_type == "checkpoint" and d.version == end["best_version"]][-1]
    got = imitation_model_from_checkpoint_bytes(document_to_bytes(final))
    assert got["blob"] == imitation_model_from_checkpoint_bytes(document_to_bytes(best))["blob"]
    assert got["val_loss"] == end["best_val_loss"]


def test_without_a_budget_the_fit_runs_until_early_stop_and_with_neither_it_is_refused(tmp_path) -> None:
    pl = _payload(tmp_path / "a", _corpus(tmp_path / "c.jsonl"), patience=3)
    pl["fit"].pop("run_budget")
    end = _run(tmp_path, pl)[-1].body
    assert end["reason"] == "es_trip" and end["epochs_run"] > 3
    pl2 = _payload(tmp_path / "b", tmp_path / "c.jsonl")
    pl2["fit"].pop("run_budget")
    r = _run(tmp_path, pl2, ok=False)
    assert r.returncode != 0 and "never end" in (r.stdout + r.stderr)


def test_the_split_does_not_move_between_stretches(tmp_path) -> None:
    held = []
    for stretch in (1, 2, 5):
        pl = _payload(tmp_path / f"s{stretch}", _corpus(tmp_path / "c.jsonl"), steps=2)
        pl["fit"]["stretch_index"] = stretch
        held.append(_run(tmp_path, pl)[-1].body["split"]["holdout_tapes"])
    assert held[0] == held[1] == held[2]
