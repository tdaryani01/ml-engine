# testing/test_direct_execution_contract.py
"""Direct-execution contract (review findings F2–F7).

F2  TM-authoritative configs are parsed strictly (agent Start/Resume + loader)
F3  TM thread budget comes from the payload — no silent default
F4  ``authorize_on_boot`` controls train-now vs wait-for-Start
F5  ledger is keyed by TM ``model_id`` (instance_id only as fallback)
F6  checkpoint W_in restore: warn on drop, clear ValueError on shape mismatch
F7  native MhsaBinding sizeof is checked against the ctypes mirror
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import TMConfigError, parse_tm_production_config
from config.config_source import (
    TM_PAYLOAD_ENV,
    ConfigSource,
    detect_config_source,
    load_tm_payload,
    tm_payload_model_id,
)


def _closed_loop_payload(**extra: Any) -> dict:
    payload = {
        "config_version": 7,
        "closed_loop": {"max_steps": 4, "batch_size": 2, "canvas": [16, 16]},
        "optimization": {"learning_rate": 0.01},
    }
    payload.update(extra)
    return payload


# --- F2: payload source + strict parse -------------------------------------


def test_tm_payload_file_and_model_id(tmp_path, monkeypatch) -> None:
    job = {"model_id": "tm-model-42", "config": _closed_loop_payload()}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(job))
    monkeypatch.setenv(TM_PAYLOAD_ENV, str(path))
    monkeypatch.delenv("ML_ENGINE_CONFIG_SOURCE", raising=False)
    assert detect_config_source() == ConfigSource.TRAINING_MANAGER
    loaded = load_tm_payload()
    assert tm_payload_model_id(loaded) == "tm-model-42"
    assert tm_payload_model_id({"config": {"model_id": "inner"}}) == "inner"
    assert tm_payload_model_id({"config": {}}) is None


def test_tm_payload_missing_env_raises(monkeypatch) -> None:
    monkeypatch.delenv(TM_PAYLOAD_ENV, raising=False)
    try:
        load_tm_payload()
    except ValueError as exc:
        assert TM_PAYLOAD_ENV in str(exc)
    else:
        raise AssertionError("TM run with no payload must not fall back silently")


def test_tm_start_policy_overrides_host_identity() -> None:
    host = {"enabled": True, "uri": "http://tm", "instance_id": "host-1"}
    cfg = parse_tm_production_config(
        _closed_loop_payload(training_manager={"authorize_on_boot": False, "uri": "evil"}),
        profile="closed_loop",
        host_identity=host,
    )
    tm = cfg["training_manager"]
    assert tm["authorize_on_boot"] is False
    # Only the start policy crosses over; identity stays host-local.
    assert tm["uri"] == "http://tm" and tm["instance_id"] == "host-1"


def test_tm_pipeline_requires_num_threads() -> None:
    from testing.test_config import _tm_payload

    payload = _tm_payload()
    del payload["optimization"]["num_threads"]
    try:
        parse_tm_production_config(payload)
    except TMConfigError as exc:
        assert "num_threads" in str(exc)
    else:
        raise AssertionError("TM pipeline payload without num_threads must be rejected")
    ok = parse_tm_production_config({**_tm_payload(), "model_id": "m-1", "job_id": "j-1"})
    assert ok.optimization.num_threads == 4  # control keys stripped, parse succeeds


def _fake_agent(monkeypatch):
    """A DrawStudentAgent with its heavy parts stubbed; real hook logic runs."""
    from examples.closed_loop_draw import agent as agent_mod

    built: list[dict] = []

    class _App:
        def close(self) -> None:
            pass

    monkeypatch.setattr(agent_mod, "assemble", lambda cfg, seed=0: built.append(cfg) or _App())

    class _Agent(agent_mod.DrawStudentAgent):
        def __init__(self) -> None:  # noqa: D401 — bypass the real boot
            self._boot_cfg = {
                "training_manager": {"enabled": True, "instance_id": "host-1"},
                "meta": {"output_dir": "out"},
            }
            self.cfg = {"closed_loop": {"max_steps": 1}, "optimization": {}}
            self.app = _App()
            self._user_pause_hold = False
            self._es_park_hold = False
            self._es_run_done = False
            self._autopilot = False
            self._config_version = None
            self._configured = False
            self._traj = 0
            self._last_loss = None
            self.merged: list[Any] = []

        def _reload_live_knobs(self) -> None:
            pass

        def _live_config(self) -> dict:
            return {"closed_loop": dict(self.cfg.get("closed_loop") or {})}

        def _print_config(self, *_a, **_k) -> None:
            pass

        def apply_run_config(self, config, *, rebuild=False):
            self.merged.append(config)
            return {}

    return _Agent(), built


def test_agent_strict_tm_config_applied(monkeypatch) -> None:
    agent, built = _fake_agent(monkeypatch)
    cmd = SimpleNamespace(action="start", payload={"config": _closed_loop_payload(autopilot=True)})
    assert agent.on_engine_start_resume(cmd) is True
    assert built and built[-1]["closed_loop"]["max_steps"] == 4
    assert agent._config_version == 7 and agent._autopilot is True
    assert agent.merged == []  # strict path, never the deep-merge
    assert agent.cfg["training_manager"]["instance_id"] == "host-1"


def test_agent_strict_tm_config_rejected_leaves_app(monkeypatch) -> None:
    agent, built = _fake_agent(monkeypatch)
    app_before, cfg_before = agent.app, dict(agent.cfg)
    bad = {"config_version": 7, "optimization": {}}  # no closed_loop
    cmd = SimpleNamespace(action="start", payload={"config": bad})
    try:
        agent.on_engine_start_resume(cmd)
    except TMConfigError as exc:
        assert "closed_loop" in str(exc)
    else:
        raise AssertionError("incomplete TM config must be rejected")
    assert agent.app is app_before and agent.cfg == cfg_before and not built


def test_agent_knob_overlay_still_merges(monkeypatch) -> None:
    agent, built = _fake_agent(monkeypatch)
    cmd = SimpleNamespace(action="resume", payload={"config": {"learning_rate": 0.002}})
    assert agent.on_engine_start_resume(cmd) is True
    assert agent.merged == [{"learning_rate": 0.002}] and not built


# --- F3: runtime thread budget ---------------------------------------------


def test_tm_runtime_budget_from_payload_no_default() -> None:
    from utils.runtime import load_runtime_settings, reset_tm_runtime_budget

    tm = ConfigSource.TRAINING_MANAGER
    reset_tm_runtime_budget()
    try:
        load_runtime_settings(config_source=tm)
    except ValueError as exc:
        assert "num_threads" in str(exc)
    else:
        raise AssertionError("TM run with no budget must not default to 4")
    s = load_runtime_settings(config_source=tm, num_threads=6, native_async_submit=True)
    assert s.num_threads == 6 and s.native_async_submit is True
    later = load_runtime_settings(config_source=tm)  # e.g. conv_dispatch re-read
    assert later.num_threads == 6 and later.native_async_submit is True
    assert "OMP_MAX_ACTIVE_LEVELS" in later.process_env()
    reset_tm_runtime_budget()


# --- F4: authorize_on_boot -------------------------------------------------


def _run_engine(authorize_on_boot: bool) -> tuple[int, Any]:
    from testing.test_closed_loop_engine_authority import _StubHB
    from src.ledger import LedgerConfig, TrainingLedger
    from src.ledger_store import FileLedgerStore
    from src.training_engine import TrainingEngine

    hb = _StubHB()
    engine = TrainingEngine(
        ledger=TrainingLedger(
            store=FileLedgerStore(tempfile.mkdtemp()),
            branch_id="main",
            architecture_id="t",
        ),
        config=LedgerConfig(checkpoint_every_steps=10**9, checkpoint_on_local_best=False),
        manager_heartbeat=hb,  # type: ignore[arg-type]
        authorize_on_boot=authorize_on_boot,
    )
    ticks = {"n": 0}

    def step() -> bool:
        ticks["n"] += 1
        return True

    engine.set_external_step(step)
    t = threading.Thread(target=engine.run, daemon=True)
    t.start()
    time.sleep(0.4)
    n_before_start = ticks["n"]
    hb.push_command("start")
    time.sleep(0.4)
    engine.request_stop()
    t.join(timeout=5)
    return n_before_start, ticks["n"]


def test_authorize_on_boot_false_waits_for_start() -> None:
    before, after = _run_engine(False)
    assert before == 0, "must not train before TM Start"
    assert after > 0, "Start must authorize training"


def test_authorize_on_boot_true_trains_immediately() -> None:
    before, _ = _run_engine(True)
    assert before > 0


# --- F5: ledger keyed by TM model_id ---------------------------------------


def _fit_capture(monkeypatch, **fit_kwargs) -> dict:
    import src.training_engine as te
    from config.schema import LedgerSettings
    from src.controller import ModelController

    seen: dict = {}

    class _Stop(Exception):
        pass

    def fake_create(ledger_dir, **kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(te, "create_training_engine", fake_create)
    ctl = ModelController.__new__(ModelController)
    ctl.model = object()
    ctl.data_provider = object()
    try:
        ctl.fit(
            steps=1,
            source_mode=None,
            model_type=SimpleNamespace(name="MHSA"),
            ledger_settings=LedgerSettings(enabled=True),
            output_dir=tempfile.mkdtemp(),
            **fit_kwargs,
        )
    except _Stop:
        pass
    return seen


def test_ledger_keyed_by_tm_model_id(monkeypatch) -> None:
    tm = {"enabled": False, "instance_id": "host-1", "authorize_on_boot": False}
    seen = _fit_capture(monkeypatch, training_manager=tm, model_id="tm-model-42")
    assert seen["model_instance_id"] == "tm-model-42"
    assert seen["authorize_on_boot"] is False
    fallback = _fit_capture(monkeypatch, training_manager=tm)
    assert fallback["model_instance_id"] == "host-1"


# --- F6: checkpoint W_in safety --------------------------------------------


def _mhsa(input_dim: int | None):
    from config.constants import EngineBackend
    from src.model_factory import ModelFactory
    from utils.conv_dispatch import bootstrap_im2col_gemm_runtime

    bootstrap_im2col_gemm_runtime()
    np.random.seed(0)
    return ModelFactory.create_model(
        "mhsa",
        layer_sizes=[3],
        backend=EngineBackend.NATIVE,
        optimizer="adam",
        mhsa_config={
            "d_model": 8, "num_heads": 2, "max_seq_len": 4, "action_dim": 3,
            "ffn_mult": 2, "num_layers": 1, "use_pos_encoding": False,
            "input_dim": input_dim,
        },
    )


def test_restore_w_in_shape_mismatch_raises_clear_error() -> None:
    from src.ledger import capture_model_checkpoint, restore_model_checkpoint

    body = capture_model_checkpoint(_mhsa(13), version=1)
    try:
        restore_model_checkpoint(_mhsa(5), body)
    except ValueError as exc:
        msg = str(exc)
        assert "W_in shape mismatch" in msg and "(13, 8)" in msg and "(5, 8)" in msg
    else:
        raise AssertionError("W_in shape mismatch must raise ValueError")


def test_restore_w_in_dropped_logs_warning() -> None:
    from src.ledger import capture_model_checkpoint, restore_model_checkpoint

    body = capture_model_checkpoint(_mhsa(13), version=1)
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _H(level=logging.WARNING)
    logging.getLogger("src.ledger").addHandler(h)
    try:
        restore_model_checkpoint(_mhsa(None), body)  # square model, no W_in
    finally:
        logging.getLogger("src.ledger").removeHandler(h)
    assert any("DROPPED" in r.getMessage() for r in records)


def test_restore_w_in_roundtrip() -> None:
    from src.ledger import capture_model_checkpoint, restore_model_checkpoint

    src_model = _mhsa(13)
    body = capture_model_checkpoint(src_model, version=1)
    dst = _mhsa(13)
    dst.W_in[...] = 0.0
    restore_model_checkpoint(dst, body)
    assert np.array_equal(dst.W_in, src_model.W_in)


# --- F7: native ABI self-report --------------------------------------------


def test_native_mhsa_binding_sizeof_matches_ctypes() -> None:
    import ctypes

    from src.contract_runtime import MhsaBinding, _load_conv_dll, _verify_native_mhsa_abi

    lib = _load_conv_dll()
    assert lib is not None, "native library not built"
    fn = lib.mhsa_binding_sizeof
    fn.restype = ctypes.c_int64
    assert int(fn()) == ctypes.sizeof(MhsaBinding) == 6672
    _verify_native_mhsa_abi(lib)


def test_stale_native_library_rejected() -> None:
    from src.contract_runtime import _verify_native_mhsa_abi

    class _NoExport:
        pass

    class _OldStruct:
        @staticmethod
        def mhsa_binding_sizeof() -> int:
            return 6664

    for lib, needle in ((_NoExport(), "predates"), (_OldStruct(), "sizeof=6664")):
        try:
            _verify_native_mhsa_abi(lib)
        except RuntimeError as exc:
            assert needle in str(exc)
        else:
            raise AssertionError("stale native library must be rejected")
