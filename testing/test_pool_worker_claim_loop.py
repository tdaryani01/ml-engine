# Pool worker: idle → claim → apply job config → job-scoped train → release.
from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.config_loader import deep_merge, load_job_pipeline_config, parse_production_config
from src.ledger import LedgerConfig, TrainingLedger
from src.ledger_store import FileLedgerStore
from src.pool_worker import job_is_closed_loop, materialize_job_config, run_pool_worker_loop
from src.training_engine import TrainingEngine


@dataclass
class _StubHB:
    idle_sleep_s: float = 0.02
    _desired_state: str | None = None
    _active_checkpoint: dict[str, Any] | None = None
    job_bound: bool = False
    job_id: str | None = None
    bound_model_id: str | None = None
    claim_jobs: list[dict[str, Any]] = field(default_factory=list)
    acks: list[dict[str, Any]] = field(default_factory=list)
    fails: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    pool_session_id: str = "87654321"
    _cfg: Any = None

    def __post_init__(self) -> None:
        if self._cfg is None:
            self._cfg = type(
                "C",
                (),
                {
                    "enabled": True,
                    "uri": "http://tm.test",
                    "instance_id": self.pool_session_id,
                    "kind": "engine",
                    "label": "test",
                    "advertise_url": "http://127.0.0.1:0",
                    "capabilities": ("train_step", "ledger"),
                    "interval_s": 10.0,
                    "timeout_s": 0.5,
                    "idle_sleep_s": 0.02,
                    "park_when_idle": True,
                },
            )()

    @property
    def desired_state(self) -> str | None:
        return self._desired_state

    def set_desired_state(self, state: str | None) -> None:
        self._desired_state = state

    @property
    def active_checkpoint(self) -> dict[str, Any] | None:
        return self._active_checkpoint

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics = dict(metrics)

    def maybe_ping(self, force: bool = False) -> None:
        del force

    def poll_commands(self) -> list:
        return []

    def try_claim_work(self, *, lease_s: float | None = None) -> dict[str, Any] | None:
        del lease_s
        if self.job_bound or not self.claim_jobs:
            return None
        job = dict(self.claim_jobs.pop(0))
        self.job_bound = True
        self.job_id = str(job["job_id"])
        self.bound_model_id = str(job.get("model_id") or "")
        return job

    def ack_work(self, *, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self.acks.append(dict(result or {}))
        self.job_bound = False
        self.job_id = None
        self.bound_model_id = None
        return {"ok": True}

    def fail_work(self, *, error: str, requeue: bool = True) -> dict[str, Any] | None:
        self.fails.append({"error": error, "requeue": requeue})
        self.job_bound = False
        self.job_id = None
        self.bound_model_id = None
        return {"ok": True}


def test_regression_job_is_closed_loop_detects_draw_shape():
    assert job_is_closed_loop({"closed_loop": {"sigma": 0.1}}) is True
    assert job_is_closed_loop({"architecture": {"model_type": "mhsa"}}) is False
    assert job_is_closed_loop({"meta": {"pipeline_name": "closed_loop_draw_agent"}}) is True


def test_regression_materialize_supervised_still_works():
    cfg = materialize_job_config(
        "config/config.yaml",
        {"job_id": "j1", "model_id": "engine-x", "config": {"optimization": {"learning_rate": 0.0003}}},
    )
    assert abs(cfg.optimization.learning_rate - 0.0003) < 1e-12


def test_regression_deep_merge_and_job_config_keeps_boot_tm():
    boot = {
        "meta": {"pipeline_name": "x", "stage": "d", "suppress_logging": False,
                 "logging_level": "warning", "output_dir": "out"},
        "ingestion": {
            "source_mode": "csv",
            "data_file_path": "data/samples/mhsa/cue_recall_quick.npz",
            "feature_names": "auto",
            "splits": {"train": 0.7, "val": 0.15},
            "drain_on_empty": False,
            "amqp_url": "",
            "queue_name": "",
            "val_queue_name": "",
        },
        "architecture": {
            "model_type": "mhsa",
            "backend": "native",
            "num_classes": 4,
            "hidden_layers": [],
            "p_dropout": 0.0,
            "use_batch_norm": False,
            "bn_momentum": 0.9,
            "mhsa": {"d_model": 64, "num_heads": 4, "max_seq_len": 32, "action_dim": 4,
                     "ffn_mult": 4, "num_layers": 2, "use_pos_encoding": True, "use_input_proj": False},
        },
        "optimization": {
            "num_threads": 4, "optimizer": "adam", "epochs_full_dataset": 1,
            "steps_streaming": 1, "batch_size": 8, "learning_rate": 0.001,
            "lr_scheduler": "none", "scheduler_decay_rate": 0.98,
            "scheduler_epochs_per_drop": 10, "scheduler_drop_ratio": 0.5,
            "early_stopping_enabled": False, "patience": 10, "min_delta": 1e-4,
            "gradient_clipping_max_norm": 5.0,
        },
        "regularization": {"lam_l1": 0.0, "lam_l2": 0.0, "sparsity_tolerance": 1e-5},
        "transformations": {"fourier_expansion": {"enabled": False, "num_frequencies": 4}},
        "persistence": {"load_saved_model": False, "model_asset_path": "x.npz"},
        "diagnostics": {
            "enabled": False, "metric_to_plot": "loss", "save_raw_logs": False,
            "figure_width": 8, "figure_height": 6, "plot_style": "default", "output_format": "png",
        },
        "ledger": {"enabled": False},
        "training_manager": {"enabled": True, "uri": "http://boot", "kind": "engine"},
    }
    merged = deep_merge(boot, {"optimization": {"learning_rate": 0.002}, "training_manager": {"uri": "http://job"}})
    assert merged["optimization"]["learning_rate"] == 0.002
    assert merged["training_manager"]["uri"] == "http://job"
    cfg = parse_production_config(boot)
    assert cfg.training_manager.uri == "http://boot"
    job_cfg = load_job_pipeline_config(
        "config/config.yaml",
        {"optimization": {"learning_rate": 0.0007}},
    )
    assert abs(job_cfg.optimization.learning_rate - 0.0007) < 1e-12
    assert job_cfg.training_manager.uri  # kept from boot


def test_regression_ledger_stamps_job_model_id_not_session():
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mhsa",
            model_instance_id="tm-brain",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=10**9, checkpoint_on_local_best=False),
        )
        assert engine._ledger_model_instance_id() == "tm-brain"
        engine.close()


def test_regression_job_scoped_exits_when_lease_released():
    hb = _StubHB()
    hb.job_bound = True
    hb.bound_model_id = "tm-brain"
    hb.job_id = "job-1"
    ticks = {"n": 0}
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mhsa",
            model_instance_id="tm-brain",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=10**9, checkpoint_on_local_best=False),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )

        def step() -> bool:
            ticks["n"] += 1
            return True  # keep ticking until lease drops

        engine.set_external_step(step)
        engine.adopt_claimed_job({"job_id": "job-1", "model_id": "tm-brain"})
        assert engine._run_authorized is True

        def _go() -> None:
            engine.run(job_scoped=True)

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        deadline = time.time() + 2.0
        while time.time() < deadline and ticks["n"] < 2:
            time.sleep(0.02)
        assert ticks["n"] >= 2
        hb.job_bound = False
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert engine._work_lease_active is False
        engine.close()


def test_regression_job_scoped_stays_on_false_tick_until_lease_ends():
    """Continuous train: one False external_step must not end the job lease."""
    hb = _StubHB()
    hb.job_bound = True
    hb.bound_model_id = "tm-brain"
    hb.job_id = "job-hold"
    hb.idle_sleep_s = 0.02
    ticks = {"n": 0}
    with tempfile.TemporaryDirectory() as tmp:
        ledger = TrainingLedger(
            store=FileLedgerStore(tmp),
            branch_id="main",
            architecture_id="mhsa",
            model_instance_id="tm-brain",
        )
        engine = TrainingEngine(
            ledger=ledger,
            config=LedgerConfig(checkpoint_every_steps=10**9, checkpoint_on_local_best=False),
            manager_heartbeat=hb,  # type: ignore[arg-type]
        )

        def step() -> bool:
            ticks["n"] += 1
            return False  # no progress — must park inside lease

        engine.set_external_step(step)
        engine.adopt_claimed_job({"job_id": "job-hold", "model_id": "tm-brain"})

        def _go() -> None:
            engine.run(job_scoped=True)

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.15)
        assert t.is_alive(), "job_scoped exited on False tick while lease held"
        assert ticks["n"] >= 1
        hb.job_bound = False
        t.join(timeout=2.0)
        assert not t.is_alive()
        engine.close()


def test_regression_pool_loop_dispatches_closed_loop(monkeypatch):
    """Closed-loop job.config must call run_closed_loop_claimed_job, not supervised."""
    hb = _StubHB()
    hb.claim_jobs.append(
        {
            "job_id": "j-cl",
            "model_id": "tm-brain",
            "kind": "train",
            "config": {
                "closed_loop": {"sigma": 0.1, "max_steps": 10, "batch_size": 4, "canvas": [1, 28, 28]},
                "meta": {"pipeline_name": "closed_loop_draw_agent"},
            },
        }
    )
    seen: dict[str, Any] = {}
    supervised = {"n": 0}

    def fake_cl(heartbeat, job, *, boot_tm=None):
        seen["model_id"] = job.get("model_id")
        seen["job_id"] = job.get("job_id")
        seen["boot_tm"] = boot_tm
        del heartbeat

    def run_supervised(cfg, heartbeat, job):
        del cfg, heartbeat, job
        supervised["n"] += 1

    monkeypatch.setattr(
        "examples.closed_loop_draw.run_lease.run_closed_loop_claimed_job",
        fake_cl,
    )

    stop = {"n": 0}

    def should_stop() -> bool:
        stop["n"] += 1
        return bool(hb.acks) or stop["n"] > 5

    run_pool_worker_loop(
        boot_yaml="config/config.yaml",
        boot_cfg=object(),  # type: ignore[arg-type]
        hb=hb,
        run_supervised_job=run_supervised,
        should_stop=should_stop,
        boot_tm={"enabled": True, "uri": "http://boot"},
    )
    assert supervised["n"] == 0
    assert seen.get("model_id") == "tm-brain"
    assert seen.get("job_id") == "j-cl"
    assert len(hb.acks) == 1
    assert len(hb.fails) == 0


def test_regression_run_closed_loop_lease_stamps_job_model_id(monkeypatch, tmp_path):
    """run_closed_loop_claimed_job builds ledger with job.model_id (not pool id)."""
    from examples.closed_loop_draw import run_lease

    hb = _StubHB()
    hb.job_bound = True
    hb.job_id = "j1"
    hb.bound_model_id = "tm-brain"
    stamped: dict[str, Any] = {}

    class _FakeAgent:
        def __init__(self, cfg, *, heartbeat=None):
            self.cfg = cfg
            self.hb = heartbeat
            self._branch_id = "main"
            self._boot_cfg = cfg

        def on_claim_config(self, job):
            del job
            return True

        def train_tick(self) -> bool:
            return False

        def on_engine_start_resume(self, cmd):
            del cmd
            return True

        def on_engine_pause(self, cmd):
            del cmd

        def on_engine_cancel(self, cmd):
            del cmd

        def on_engine_restore(self, cmd):
            del cmd
            return False

        def pause_gate(self) -> bool:
            return False

        def on_release_config(self) -> None:
            return None

        def bind_engine_stop(self, stop) -> None:
            del stop

        def close(self) -> None:
            return None

    real_build = run_lease.build_closed_loop_engine

    def wrap(agent, *, model_instance_id, ledger_dir, claim_inside_engine=True):
        stamped["model_instance_id"] = model_instance_id
        stamped["ledger_dir"] = str(ledger_dir)
        eng = real_build(
            agent,
            model_instance_id=model_instance_id,
            ledger_dir=Path(tmp_path) / "ledger",
            claim_inside_engine=claim_inside_engine,
        )
        assert eng.ledger.model_instance_id == "tm-brain"

        def _run_once(**_kwargs):
            # Don't park forever in job_scoped idle — this test only stamps ids.
            return {}

        eng.run = _run_once  # type: ignore[method-assign]
        return eng

    monkeypatch.setattr(run_lease, "DrawStudentAgent", _FakeAgent)
    monkeypatch.setattr(run_lease, "build_closed_loop_engine", wrap)
    monkeypatch.setattr(run_lease, "_REPO", Path(tmp_path))

    job = {
        "job_id": "j1",
        "model_id": "tm-brain",
        "config": {
            "closed_loop": {"sigma": 0.1, "max_steps": 10, "batch_size": 4, "canvas": [1, 28, 28]},
            "meta": {"output_dir": str(tmp_path / "out")},
            "optimization": {"learning_rate": 0.001, "seed": 0},
        },
    }
    run_lease.run_closed_loop_claimed_job(hb, job, boot_tm={"enabled": True, "uri": "http://t"})
    assert stamped["model_instance_id"] == "tm-brain"
