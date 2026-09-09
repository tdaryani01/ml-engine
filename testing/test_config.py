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
    CONFIG_DIR / "config_pad2.yaml",
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
    cfg = load_production_config(str(CONFIG_DIR / "config.yaml"))
    assert cfg.architecture.model_type == ModelType.CNN
    assert cfg.architecture.backend == EngineBackend.NATIVE
    cnn = cfg.architecture.cnn
    assert cnn is not None
    if isinstance(cnn, dict):
        assert cnn.get("input_shape")
        assert cnn.get("spatial_pipeline")
    else:
        assert cnn.input_shape
        assert cnn.spatial_pipeline
    print("[PASSED] config.yaml: CNN + native backend invariants")


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
        assert settings.env["OMP_NUM_THREADS"] == str(override_threads)
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


CONFIG_TESTS = [
    test_production_yaml_files_parse,
    test_production_config_hydrates,
    test_production_config_cnn_invariants,
    test_production_config_ledger_section,
    test_runtime_yaml_parses,
    test_runtime_settings_load_default_paths,
    test_runtime_threads_fallback_to_config_when_unset,
    test_runtime_threads_override_from_runtime_yaml,
    test_ledger_native_async_submit_from_config,
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
