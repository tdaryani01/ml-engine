# testing/test_config.py
"""Production config smoke tests — catch YAML typos before runtime/benchmarks."""
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_loader import load_production_config
from config.constants import EngineBackend, ModelType
from utils.runtime import load_runtime_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"

PRODUCTION_CONFIGS = (
    CONFIG_DIR / "config.yaml",
    CONFIG_DIR / "config_28.yaml",
    CONFIG_DIR / "config_pad2.yaml",
    CONFIG_DIR / "config_mhsa_sanity.yaml",
)


def _assert_yaml_parses(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict), f"{path.name}: expected top-level mapping"
    return raw


def test_production_yaml_files_parse() -> None:
    for path in PRODUCTION_CONFIGS:
        assert path.is_file(), f"missing production config: {path}"
        _assert_yaml_parses(path)
    print(f"[PASSED] YAML parse: {', '.join(p.name for p in PRODUCTION_CONFIGS)}")


def test_production_config_hydrates() -> None:
    for path in PRODUCTION_CONFIGS:
        cfg = load_production_config(str(path))
        assert cfg.meta.pipeline_name
        assert cfg.ingestion.data_file_path
        assert cfg.optimization.num_threads >= 1
    print(f"[PASSED] load_production_config: {len(PRODUCTION_CONFIGS)} file(s)")


def test_production_config_cnn_invariants() -> None:
    """CNN production defaults live in config_28 / config_pad2 (config.yaml is MHSA)."""
    for name in ("config_28.yaml", "config_pad2.yaml"):
        cfg = load_production_config(str(CONFIG_DIR / name))
        assert cfg.architecture.model_type == ModelType.CNN, name
        assert cfg.architecture.backend == EngineBackend.NATIVE, name
        cnn = cfg.architecture.cnn
        assert cnn is not None, name
        if isinstance(cnn, dict):
            assert cnn.get("input_shape"), name
            assert cnn.get("spatial_pipeline"), name
        else:
            assert cnn.input_shape, name
            assert cnn.spatial_pipeline, name
    print("[PASSED] config_28/pad2: CNN + native backend invariants")


def test_production_config_mhsa_invariants() -> None:
    cfg = load_production_config(str(CONFIG_DIR / "config.yaml"))
    assert cfg.architecture.model_type == ModelType.MHSA
    assert cfg.architecture.backend == EngineBackend.NATIVE
    mhsa = cfg.architecture.mhsa
    assert mhsa is not None
    if isinstance(mhsa, dict):
        assert int(mhsa.get("d_model", 0)) > 0
        assert int(mhsa.get("num_heads", 0)) > 0
        assert int(mhsa.get("max_seq_len", 0)) > 0
        assert int(mhsa.get("action_dim", 0)) > 0
    else:
        assert mhsa.d_model > 0
        assert mhsa.num_heads > 0
        assert mhsa.max_seq_len > 0
        assert mhsa.action_dim > 0
        assert isinstance(mhsa.use_input_proj, bool)
    print("[PASSED] config.yaml: MHSA + native backend invariants")


def test_production_config_ledger_section() -> None:
    cfg = load_production_config(str(CONFIG_DIR / "config.yaml"))
    assert cfg.ledger.path
    assert cfg.ledger.branch_id
    assert cfg.ledger.checkpoint_every_steps >= 1
    assert isinstance(cfg.ledger.native_async_submit, bool)
    print("[PASSED] config.yaml: ledger section hydrates")


def test_runtime_yaml_parses() -> None:
    path = CONFIG_DIR / "runtime.yaml"
    raw = _assert_yaml_parses(path)
    assert "env" in raw
    assert "blas_threads" in raw
    print("[PASSED] runtime.yaml: YAML parse")


def test_runtime_settings_load_default_paths() -> None:
    settings = load_runtime_settings()
    assert settings.num_threads >= 1
    assert settings.platform in ("windows", "linux")
    assert isinstance(settings.env, dict)
    print(f"[PASSED] load_runtime_settings: {settings.num_threads} threads, platform={settings.platform}")


def test_runtime_threads_fallback_to_config_when_unset() -> None:
    """No OMP_* / num_threads in runtime.yaml => use config.yaml optimization.num_threads."""
    import tempfile

    cfg = yaml.safe_load((CONFIG_DIR / "config.yaml").read_text(encoding="utf-8"))
    expected = int(cfg["optimization"]["num_threads"])

    with tempfile.TemporaryDirectory() as td:
        rt_path = Path(td) / "runtime.yaml"
        rt_path.write_text(
            "\n".join(
                [
                    "env:",
                    '  OMP_DYNAMIC: "false"',
                    "blas_threads:",
                    "  native: null",
                    "  numpy: null",
                    "  im2col_gemm: null",
                    "linux: {}",
                    "windows: {}",
                    "docker: {}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        settings = load_runtime_settings(runtime_path=rt_path)
        assert settings.num_threads == expected
        assert settings.effective_omp_thread_limit() == expected
        assert settings.env["OMP_NUM_THREADS"] == str(expected)
        assert settings.env["OMP_THREAD_LIMIT"] == str(expected)
    print(f"[PASSED] runtime threads fallback: num_threads={expected}")


def test_runtime_threads_override_from_runtime_yaml() -> None:
    """When runtime.yaml sets num_threads / omp_thread_limit, those win over config.yaml."""
    import tempfile

    cfg = yaml.safe_load((CONFIG_DIR / "config.yaml").read_text(encoding="utf-8"))
    config_threads = int(cfg["optimization"]["num_threads"])
    override_threads = config_threads + 3
    override_limit = config_threads + 1

    with tempfile.TemporaryDirectory() as td:
        rt_path = Path(td) / "runtime.yaml"
        rt_path.write_text(
            "\n".join(
                [
                    f"num_threads: {override_threads}",
                    f"omp_thread_limit: {override_limit}",
                    "env:",
                    '  OMP_DYNAMIC: "false"',
                    "blas_threads:",
                    "  native: null",
                    "  numpy: null",
                    "  im2col_gemm: null",
                    "linux: {}",
                    "windows: {}",
                    "docker: {}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        settings = load_runtime_settings(runtime_path=rt_path)
        assert settings.num_threads == override_threads
        assert settings.effective_omp_thread_limit() == override_limit
        # The exported team never exceeds the exported limit (a team above the limit hangs an OpenMP OpenBLAS GEMM).
        assert settings.env["OMP_NUM_THREADS"] == str(min(override_threads, override_limit))
        assert settings.env["OMP_THREAD_LIMIT"] == str(override_limit)
    print(
        f"[PASSED] runtime threads override: "
        f"num_threads={override_threads} limit={override_limit}"
    )


def test_ledger_native_async_submit_from_config() -> None:
    """native_async_submit lives under config.yaml ledger, not runtime/env.

    Unit-tests both states on LedgerSettings directly; does not assert a
    specific value for the current config.yaml, since that's a deployment
    choice, not a schema invariant.
    """
    from config.schema import LedgerSettings

    disabled = LedgerSettings(native_async_submit=False)
    assert disabled.native_async_submit is False
    enabled = LedgerSettings(native_async_submit=True)
    assert enabled.native_async_submit is True
    default = LedgerSettings()
    assert default.native_async_submit is False  # schema default, not config.yaml
    print("[PASSED] ledger native_async_submit: both states hydrate via schema")


def _tm_payload(**overrides) -> dict:
    """A complete, authoritative TM job ``config`` mapping (MHSA)."""
    payload = {
        "meta": {
            "pipeline_name": "tm_job",
            "stage": "dev",
            "suppress_logging": True,
            "logging_level": "warning",
            "output_dir": "out",
        },
        "ingestion": {
            "source_mode": "csv",
            "data_file_path": "data/samples/mhsa/cue_recall_quick.npz",
            "feature_names": "auto",
            "splits": {"train": 0.7, "val": 0.15},
            "drain_on_empty": False,
            "val_queue_name": "",
            "amqp_url": "",
            "queue_name": "",
        },
        "architecture": {
            "model_type": "mhsa",
            "backend": "native",
            "num_classes": 4,
            "hidden_layers": [],
            "p_dropout": 0.0,
            "use_batch_norm": False,
            "bn_momentum": 0.9,
            "mhsa": {"d_model": 64, "num_heads": 4, "max_seq_len": 32, "action_dim": 4},
        },
        "optimization": {
            "optimizer": "adam",
            "epochs_full_dataset": 1,
            "steps_streaming": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "lr_scheduler": "none",
            "scheduler_decay_rate": 0.98,
            "scheduler_epochs_per_drop": 10,
            "scheduler_drop_ratio": 0.5,
            "early_stopping_enabled": False,
            "patience": 10,
            "min_delta": 1e-4,
            "gradient_clipping_max_norm": 5.0,
            "num_threads": 4,
        },
        "regularization": {"lam_l1": 0.0, "lam_l2": 0.0, "sparsity_tolerance": 1e-5},
        "transformations": {"fourier_expansion": {"enabled": False, "num_frequencies": 4}},
        "persistence": {"load_saved_model": False, "model_asset_path": "x.npz"},
        "diagnostics": {
            "enabled": False,
            "metric_to_plot": "loss",
            "save_raw_logs": False,
            "figure_width": 8,
            "figure_height": 6,
            "plot_style": "default",
            "output_format": "png",
        },
    }
    payload.update(overrides)
    return payload


def _geom(obj, name):
    """Read a geometry field from either a schema dataclass or a raw mapping."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name)


def test_local_config_has_no_schema_template() -> None:
    """Standalone hydration is untouched: no TM template leaks into local YAML."""
    cfg = load_production_config(str(CONFIG_DIR / "config.yaml"))
    assert cfg.architecture.schema_template is None
    print("[PASSED] local YAML: schema_template absent (standalone unchanged)")


def test_tm_config_strict_rejects_partial_payload() -> None:
    from config.config_loader import TMConfigError, parse_tm_production_config

    try:
        parse_tm_production_config({"optimization": {"learning_rate": 0.0003}})
    except TMConfigError as exc:
        assert "architecture" in str(exc)
    else:
        raise AssertionError("strict TM parse must reject a partial payload")
    print("[PASSED] TM strict: partial payload rejected with TMConfigError")


def test_tm_config_strict_requires_mhsa_geometry() -> None:
    from config.config_loader import TMConfigError, parse_tm_production_config

    payload = _tm_payload()
    payload["architecture"] = {
        k: v for k, v in payload["architecture"].items() if k != "mhsa"
    }
    try:
        parse_tm_production_config(payload)
    except TMConfigError as exc:
        assert "mhsa" in str(exc)
    else:
        raise AssertionError("missing MHSA geometry must raise TMConfigError")
    print("[PASSED] TM strict: missing MHSA geometry rejected")


def test_tm_config_retains_host_identity() -> None:
    from config.config_loader import parse_tm_production_config

    cfg = parse_tm_production_config(
        _tm_payload(),
        host_identity={
            "enabled": True,
            "uri": "http://boot-tm",
            "kind": "engine",
            "instance_id": "host-7",
            "m2m": {"client_id": "x"},
        },
    )
    assert cfg.training_manager.enabled is True
    assert cfg.training_manager.uri == "http://boot-tm"
    assert cfg.training_manager.instance_id == "host-7"
    assert cfg.training_manager.m2m == {"client_id": "x"}
    print("[PASSED] TM strict: host identity retained from boot")


def test_tm_config_schema_template_supplies_geometry() -> None:
    from config.config_loader import parse_tm_production_config

    payload = _tm_payload()
    payload["architecture"] = {
        k: v for k, v in payload["architecture"].items() if k != "mhsa"
    }
    payload["schema_template"] = {
        "d_model": 128,
        "num_heads": 8,
        "max_seq_len": 64,
        "action_dim": 4,
        "ffn_mult": 2,
        "num_layers": 3,
    }
    cfg = parse_tm_production_config(payload)
    mhsa = cfg.architecture.mhsa
    assert mhsa is not None
    assert (
        int(_geom(mhsa, "d_model")),
        int(_geom(mhsa, "num_heads")),
        int(_geom(mhsa, "max_seq_len")),
    ) == (128, 8, 64)
    assert int(_geom(mhsa, "ffn_mult")) == 2
    assert int(_geom(mhsa, "num_layers")) == 3
    assert cfg.architecture.schema_template is not None
    assert cfg.architecture.schema_template.d_model == 128
    print("[PASSED] TM strict: schema_template supplies/overlays MHSA geometry")


def test_config_source_detection_payload_markers() -> None:
    from config.config_source import (
        ConfigSource,
        resolve_config_source,
        tm_payload_declares_authority,
    )

    assert tm_payload_declares_authority({"schema_template": {}}) is True
    assert tm_payload_declares_authority({"architecture": {"model_type": "mhsa"}}) is True
    assert tm_payload_declares_authority({"optimization": {"learning_rate": 1e-3}}) is False
    assert resolve_config_source(payload={"optimization": {}}) == ConfigSource.LOCAL
    assert (
        resolve_config_source(payload={"config_version": 3}) == ConfigSource.TRAINING_MANAGER
    )
    print("[PASSED] config_source: payload markers resolve provenance explicitly")


CONFIG_TESTS = [
    test_production_yaml_files_parse,
    test_production_config_hydrates,
    test_production_config_cnn_invariants,
    test_production_config_mhsa_invariants,
    test_production_config_ledger_section,
    test_runtime_yaml_parses,
    test_runtime_settings_load_default_paths,
    test_runtime_threads_fallback_to_config_when_unset,
    test_runtime_threads_override_from_runtime_yaml,
    test_ledger_native_async_submit_from_config,
    test_local_config_has_no_schema_template,
    test_tm_config_strict_rejects_partial_payload,
    test_tm_config_strict_requires_mhsa_geometry,
    test_tm_config_retains_host_identity,
    test_tm_config_schema_template_supplies_geometry,
    test_config_source_detection_payload_markers,
]


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING PRODUCTION CONFIG SMOKE TESTS ")
    print("=" * 60)
    failed = 0
    for fn in CONFIG_TESTS:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {fn.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"[FAILURE] {failed}/{len(CONFIG_TESTS)} failed.")
        sys.exit(1)
    print(f"[SUCCESS] All {len(CONFIG_TESTS)} config smoke tests passed.")
    print("=" * 60)


def test_schema_template_carries_input_dim_through_typed_mhsa_config():
    """Decoupled geometry must survive the typed parser (no silent drop)."""
    from config.constants import ModelType
    from config.schema import ArchitectureConfig, MHSAConfig, SchemaTemplate

    base = MHSAConfig(d_model=32, num_heads=4, max_seq_len=9, action_dim=5)
    assert base.input_dim is None
    arch = ArchitectureConfig(
        model_type=ModelType.MHSA, num_classes=5, hidden_layers=[], mhsa=base
    )
    tmpl = SchemaTemplate.from_mapping({"input_dim": 37, "num_heads": 2})
    out = tmpl.apply_to_architecture_config(arch)
    assert out.mhsa.input_dim == 37
    assert out.mhsa.num_heads == 2
    # Untouched fields keep their configured values.
    assert out.mhsa.d_model == 32 and out.mhsa.max_seq_len == 9
