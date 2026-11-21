# testing/test_thin_steps.py
"""Data design 9 (harness needs): ``ledger.thin_steps`` drops the per-step gradient arrays from the journal and changes nothing a harness run reads."""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.launcher import run_me_fit
from src.launcher.payload import build_pipeline_payload
from src.ledger import STEP_COMPLETE, document_from_bytes
from src.ledger_wire import iter_framed_records
from testing.test_launcher import _spec


def _feed(h, v) -> None:
    """Every number of the optimizer state (step count and moment buffers), arrays by their bytes."""
    if isinstance(v, dict):
        for k in sorted(v, key=str):
            h.update(str(k).encode())
            _feed(h, v[k])
    elif isinstance(v, (list, tuple)):
        for x in v:
            _feed(h, x)
    elif isinstance(v, np.ndarray):
        h.update(np.ascontiguousarray(v).tobytes())
    else:
        h.update(repr(v).encode())


def _weights_digest(ckpt: bytes) -> str:
    body = document_from_bytes(ckpt).body
    h = hashlib.sha256()
    for k in ("weights", "biases", "gammas", "betas"):
        for a in body.get(k) or []:
            h.update(np.ascontiguousarray(a).tobytes())
    _feed(h, body.get("optimizer") or {})
    return h.hexdigest()


def _run(tmp_path, name: str, **kw):
    work = tmp_path / name
    snaps = list(run_me_fit(_spec(tmp_path, epochs=3, checkpoint_every=10, **kw), work_dir=work, should_stop=lambda: False))
    journal = (work / "out" / "ledger" / "journal.bin").read_bytes()
    return snaps, journal


def _complete_docs(journal: bytes):
    return [d for d in (document_from_bytes(r) for r in iter_framed_records(journal)) if d.doc_type == STEP_COMPLETE]


def test_the_switch_is_off_in_the_payload_unless_set(tmp_path) -> None:
    assert "thin_steps" not in build_pipeline_payload(_spec(tmp_path), work_dir=tmp_path / "w")["ledger"]
    assert build_pipeline_payload(_spec(tmp_path, thin_steps=True), work_dir=tmp_path / "w")["ledger"]["thin_steps"] is True


def test_a_harness_run_reads_the_same_with_thin_steps_on_and_the_journal_is_smaller(tmp_path) -> None:
    off, j_off = _run(tmp_path, "off")
    on, j_on = _run(tmp_path, "on", thin_steps=True)
    assert len(off) == len(on) and any(s.get("checkpoint_bytes") for s in on)
    for a, b in zip(off, on):
        assert {k: v for k, v in a.items() if k != "checkpoint_bytes"} == {k: v for k, v in b.items() if k != "checkpoint_bytes"}
        assert bool(a.get("checkpoint_bytes")) == bool(b.get("checkpoint_bytes"))
        if a.get("checkpoint_bytes"):
            assert _weights_digest(a["checkpoint_bytes"]) == _weights_digest(b["checkpoint_bytes"])
    c_off, c_on = _complete_docs(j_off), _complete_docs(j_on)
    assert len(c_off) == len(c_on) > 0
    for a, b in zip(c_off, c_on):
        assert a.version == b.version and a.body["metrics"] == b.body["metrics"] and {k: v for k, v in a.body["command"].items() if k != "batch_ref"} == {k: v for k, v in b.body["command"].items() if k != "batch_ref"}
        assert a.body["_result"].loss == b.body["_result"].loss and a.body["_result"].m_samples == b.body["_result"].m_samples
        assert a.body["_result"].grad_weights and not b.body["_result"].grad_weights
    assert len(j_on) < len(j_off)


def test_a_thin_journal_cannot_be_replayed_and_a_full_one_still_can(tmp_path) -> None:
    import pytest

    from src.ledger import FileLedgerStore, ThinStepError, TrainingLedger

    class _Session:
        def __init__(self) -> None:
            self.applied = 0

        def apply_step(self, result, lr) -> None:
            self.applied += 1

    for name, thin in (("full", None), ("thin", True)):
        _run(tmp_path, name, **({"thin_steps": True} if thin else {}))
    sessions = {}
    for name in ("full", "thin"):
        led = TrainingLedger(store=FileLedgerStore(str(tmp_path / name / "out" / "ledger")), branch_id="main", model_instance_id="t", architecture_id="x", version=0)
        sessions[name] = (led, _Session())
    led, s = sessions["full"]
    led.replay_apply_from_version(s, 0, 10**9, 0.01)
    assert s.applied > 0
    led, s = sessions["thin"]
    with pytest.raises(ThinStepError):
        led.replay_apply_from_version(s, 0, 10**9, 0.01)
