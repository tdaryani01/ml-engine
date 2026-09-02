"""Runtime threading policy: config-driven env + threadpoolctl."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_str = str(PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from config.constants import EngineBackend

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_RUNTIME_PATH = PROJECT_ROOT / "config" / "runtime.yaml"

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_DEVICE_THREAD_LIMIT",
)

_FIT_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "KMP_DEVICE_THREAD_LIMIT",
)

_VALID_PLATFORMS = ("linux", "windows")

_CONV_BACKENDS = frozenset({EngineBackend.NATIVE, EngineBackend.IM2COL_GEMM})


def detect_platform() -> str:
    return "windows" if sys.platform == "win32" else "linux"


def unified_omp_active() -> bool:
    """True when bin/libopenblas.dll shares LLVM OpenMP with conv_kernels (USE_OPENMP=1)."""
    raw = os.environ.get("ML_ENGINE_UNIFIED_OMP")
    if raw is not None:
        val = raw.strip().lower()
        if val in ("0", "false", "no", "off"):
            return False
        if val in ("1", "true", "yes", "on"):
            return True
    try:
        from utils.conv_dispatch import bootstrap_im2col_gemm_runtime, native_blas_unified_omp
        bootstrap_im2col_gemm_runtime()
        return native_blas_unified_omp()
    except Exception:
        return False


@dataclass(frozen=True)
class RuntimeSettings:
    num_threads: int
    platform: str
    env: Dict[str, str] = field(default_factory=dict)
    blas_threads: Dict[str, Optional[int]] = field(default_factory=dict)
    docker: Dict[str, object] = field(default_factory=dict)

    def blas_threads_for(self, backend: EngineBackend) -> int:
        key_map = {
            EngineBackend.NATIVE: "native",
            EngineBackend.NUMPY: "numpy",
            EngineBackend.IM2COL_GEMM: "im2col_gemm",
        }
        key = key_map[backend]
        raw = self.blas_threads.get(key)
        if raw is None:
            return self.num_threads
        return int(raw)

    def omp_threads_for(self, backend: EngineBackend) -> int:
        """LLVM OpenMP thread count (shared with OpenBLAS for conv backends)."""
        return self.num_threads

    def process_env(self) -> Dict[str, str]:
        """Process-wide env before NumPy/SciPy first touch.

        USE_OPENMP=1 OpenBLAS shares LLVM OMP — set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS.
        MKL/numba stay serial to avoid extra pools.
        """
        merged = dict(self.env)
        n = str(self.num_threads)
        merged.setdefault("OMP_NUM_THREADS", n)
        merged.setdefault("OMP_THREAD_LIMIT", n)
        merged.setdefault("OPENBLAS_NUM_THREADS", n)
        serial = "1"
        for key in (
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            merged.setdefault(key, serial)
        return merged


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_num_threads(
    config_path: Path,
    *,
    num_threads: Optional[int] = None,
) -> int:
    if num_threads is not None:
        return int(num_threads)
    cfg = _load_yaml(config_path)
    return int(cfg.get("optimization", {}).get("num_threads", 4))


def _merge_env_sections(raw: Mapping[str, object], platform: str) -> Dict[str, str]:
    if platform not in _VALID_PLATFORMS:
        raise ValueError(f"platform must be one of {_VALID_PLATFORMS}, got {platform!r}")

    shared = raw.get("env") or {}
    if not isinstance(shared, Mapping):
        raise ValueError("runtime.env must be a mapping")

    platform_values = raw.get(platform) or {}
    if not isinstance(platform_values, Mapping):
        raise ValueError(f"runtime.{platform} must be a mapping")

    merged: Dict[str, str] = {}
    for section in (shared, platform_values):
        for key, value in section.items():
            merged[str(key)] = str(value)
    return merged


def load_runtime_settings(
    config_path: Optional[Path] = None,
    runtime_path: Optional[Path] = None,
    *,
    num_threads: Optional[int] = None,
    platform: Optional[str] = None,
) -> RuntimeSettings:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    rt_path = Path(runtime_path) if runtime_path else DEFAULT_RUNTIME_PATH
    raw = _load_yaml(rt_path)
    threads = _resolve_num_threads(cfg_path, num_threads=num_threads)
    resolved_platform = platform or detect_platform()

    blas_section = raw.get("blas_threads") or {}
    if not isinstance(blas_section, Mapping):
        raise ValueError("runtime.blas_threads must be a mapping")

    docker_section = raw.get("docker") or {}
    if not isinstance(docker_section, Mapping):
        raise ValueError("runtime.docker must be a mapping")

    return RuntimeSettings(
        num_threads=threads,
        platform=resolved_platform,
        env=_merge_env_sections(raw, resolved_platform),
        blas_threads={str(k): (None if v is None else int(v)) for k, v in blas_section.items()},
        docker=dict(docker_section),
    )


def apply_process_env(
    settings: RuntimeSettings,
    *,
    overwrite: bool = False,
    if_unset: bool = False,
) -> Dict[str, str]:
    """Apply runtime env before native libs / BLAS first touch."""
    applied: Dict[str, str] = {}
    for key, value in settings.process_env().items():
        if if_unset and key in os.environ:
            continue
        if overwrite or if_unset or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _shared_omp_fit(backend: EngineBackend) -> bool:
    """native / im2col+gemm: one LLVM OMP pool for conv, im2col, and GEMM."""
    return backend in _CONV_BACKENDS


def _apply_fit_thread_env(settings: RuntimeSettings, backend: EngineBackend) -> Dict[str, str]:
    """Backend-specific thread caps during fit."""
    omp = settings.omp_threads_for(backend)
    shared = _shared_omp_fit(backend)
    serial = "1"
    overrides = {
        "OMP_NUM_THREADS": str(omp),
        "OMP_THREAD_LIMIT": str(omp),
        "OPENBLAS_NUM_THREADS": str(omp if shared else settings.blas_threads_for(backend)),
    }
    if omp > 1:
        # LLVM OpenMP on Windows honors KMP_DEVICE_THREAD_LIMIT; =1 causes OMP warning #96.
        overrides["KMP_DEVICE_THREAD_LIMIT"] = str(omp)
    blas_env = serial if shared else str(settings.blas_threads_for(backend))
    overrides.update({
        "MKL_NUM_THREADS": blas_env,
        "NUMEXPR_NUM_THREADS": blas_env,
        "VECLIB_MAXIMUM_THREADS": blas_env,
    })
    if backend in _CONV_BACKENDS:
        overrides["NUMBA_NUM_THREADS"] = serial
    for key, value in overrides.items():
        os.environ[key] = value
    return overrides


def _sync_native_dll_threads(settings: RuntimeSettings, backend: EngineBackend) -> int:
    from utils.conv_dispatch import sync_im2col_parallel_cap, sync_native_thread_policy

    omp = settings.omp_threads_for(backend)
    sync_native_thread_policy(omp)
    sync_im2col_parallel_cap(omp)
    return omp


def _sync_im2col_parallel_cap(settings: RuntimeSettings, backend: EngineBackend) -> int:
    from utils.conv_dispatch import sync_im2col_parallel_cap
    return sync_im2col_parallel_cap(settings.omp_threads_for(backend))


def _query_native_dll_omp() -> Optional[int]:
    try:
        from utils.conv_dispatch import _load_conv_dll
        lib = _load_conv_dll()
        if lib is not None and hasattr(lib, "get_omp_threads"):
            return int(lib.get_omp_threads())
    except Exception:
        pass
    return None


def _query_native_im2col_cap() -> Optional[int]:
    try:
        from utils.conv_dispatch import _load_conv_dll
        lib = _load_conv_dll()
        if lib is not None and hasattr(lib, "get_im2col_parallel_cap"):
            return int(lib.get_im2col_parallel_cap())
    except Exception:
        pass
    return None


def _query_native_unified_omp() -> Optional[bool]:
    try:
        return unified_omp_active()
    except Exception:
        return None


def log_runtime_settings(
    settings: RuntimeSettings,
    backend: EngineBackend,
    *,
    prefix: str = "[Runtime]",
) -> None:
    blas = settings.blas_threads_for(backend)
    omp = settings.omp_threads_for(backend)
    shared = _shared_omp_fit(backend)
    openblas_fit = omp if shared else blas
    if settings.platform == "windows":
        kmp_limit = os.environ.get("KMP_DEVICE_THREAD_LIMIT", "?")
        tune = (
            f"KMP_BLOCKTIME={os.environ.get('KMP_BLOCKTIME', '?')} "
            f"KMP_AFFINITY={os.environ.get('KMP_AFFINITY', '?')} "
            f"KMP_DEVICE_THREAD_LIMIT={kmp_limit}"
        )
    else:
        tune = (
            f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', '?')} "
            f"GOMP_SPINCOUNT={os.environ.get('GOMP_SPINCOUNT', '?')}"
        )
    dll_omp = _query_native_dll_omp()
    dll_omp_s = str(dll_omp) if dll_omp is not None else "n/a"
    im2col_cap = _query_native_im2col_cap()
    im2col_cap_s = str(im2col_cap) if im2col_cap is not None else "n/a"
    unified = _query_native_unified_omp()
    unified_s = str(unified) if unified is not None else "n/a"
    if shared and unified:
        policy = "shared_omp"
    elif shared:
        policy = "conv_omp"
    else:
        policy = backend.value
    msg = (
        f"{prefix} platform={settings.platform} num_threads={settings.num_threads} "
        f"backend={backend.value} policy={policy} omp_during_fit={omp} "
        f"openblas_during_fit={openblas_fit} scipy_blas_during_fit={1 if shared else blas} "
        f"dll_omp={dll_omp_s} im2col_cap={im2col_cap_s} unified_omp={unified_s} {tune}"
    )
    print(msg)
    logger.info(msg)
    if logger.isEnabledFor(logging.DEBUG):
        try:
            pools = threadpool_info()
        except Exception:
            pools = []
        for entry in pools:
            logger.debug(
                "%s pool user_api=%s internal_api=%s num_threads=%s",
                prefix,
                entry.get("user_api"),
                entry.get("internal_api"),
                entry.get("num_threads"),
            )


@contextmanager
def training_threadpool(
    settings: RuntimeSettings,
    backend: EngineBackend,
    *,
    blas_threads: Optional[int] = None,
    omp_threads: Optional[int] = None,
) -> Iterator[None]:
    """Scope BLAS/OpenMP/Numba pools for training.

    Policy (OMP_MAX_ACTIVE_LEVELS=1):
      - native / im2col+gemm: shared LLVM OMP + bin/libopenblas; scipy wheel BLAS pinned to 1
      - numpy: omp=blas=num_threads
    """
    omp = settings.omp_threads_for(backend) if omp_threads is None else int(omp_threads)
    if backend in _CONV_BACKENDS:
        from utils.conv_dispatch import bootstrap_im2col_gemm_runtime
        bootstrap_im2col_gemm_runtime()
    if blas_threads is not None:
        blas = int(blas_threads)
    else:
        blas = settings.blas_threads_for(backend)

    prev_numba_threads: Optional[int] = None
    if backend in _CONV_BACKENDS:
        import numba

        prev_numba_threads = numba.get_num_threads()
        numba.set_num_threads(1)

    prev_env = {key: os.environ.get(key) for key in _FIT_THREAD_ENV_KEYS}
    _apply_fit_thread_env(settings, backend)
    _sync_native_dll_threads(settings, backend)

    try:
        if backend in _CONV_BACKENDS:
            from utils.conv_dispatch import sync_openblas_thread_policy

            # Wheel scipy/numpy OpenBLAS — serial; conv/GEMM use shared DLL pool.
            with threadpool_limits(limits={"openblas": 1, "blas": 1, "mkl": 1}):
                _sync_native_dll_threads(settings, backend)
                sync_openblas_thread_policy(omp)
                yield
        elif blas == omp:
            with threadpool_limits(limits=omp):
                yield
        else:
            with threadpool_limits(limits={"openblas": blas, "blas": blas, "openmp": omp}):
                yield
    finally:
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if prev_numba_threads is not None:
            import numba

            numba.set_num_threads(prev_numba_threads)


def configure_runtime(
    backend: EngineBackend,
    *,
    config_path: Optional[Path] = None,
    runtime_path: Optional[Path] = None,
    num_threads: Optional[int] = None,
    overwrite_env: bool = False,
    if_unset_env: bool = True,
    log: bool = True,
) -> RuntimeSettings:
    """Load runtime.yaml, apply process env, optionally log effective policy."""
    settings = load_runtime_settings(
        config_path=config_path,
        runtime_path=runtime_path,
        num_threads=num_threads,
    )
    apply_process_env(settings, overwrite=overwrite_env, if_unset=if_unset_env)
    if backend in _CONV_BACKENDS:
        from utils.conv_dispatch import bootstrap_im2col_gemm_runtime
        bootstrap_im2col_gemm_runtime()
        import numba
        numba.set_num_threads(1)
    _apply_fit_thread_env(settings, backend)
    _sync_native_dll_threads(settings, backend)
    if log:
        log_runtime_settings(settings, backend)
    return settings


def get_docker_section(runtime_path: Optional[Path] = None) -> dict:
    return load_runtime_settings(runtime_path=runtime_path).docker


def get_benchmark_runner_argv(runtime_path: Optional[Path] = None) -> list[str]:
    docker_cfg = get_docker_section(runtime_path)
    runner = docker_cfg.get("benchmark_runner")
    if runner is None:
        return ["python", "-u", "benchmarks/run_benchmarks_docker.py"]
    if not isinstance(runner, list) or not runner:
        raise ValueError("runtime.docker.benchmark_runner must be a non-empty list")
    return [str(part) for part in runner]


def format_shell_exports(env: Mapping[str, str]) -> str:
    lines = []
    for key, value in env.items():
        escaped = value.replace("'", "'\"'\"'")
        lines.append(f"export {key}='{escaped}'")
    return "\n".join(lines)


def format_docker_args(env: Mapping[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in env.items())


def format_runner_shell(runtime_path: Optional[Path] = None) -> str:
    return " ".join(shlex.quote(part) for part in get_benchmark_runner_argv(runtime_path))


def _parse_overrides(values: Optional[list[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not values:
        return overrides
    for item in values:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        overrides[key] = value
    return overrides


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit runtime env from config/runtime.yaml")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--platform", choices=_VALID_PLATFORMS, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runtime", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("docker-args", "shell-exports", "json", "runner-shell", "docker-json"),
        default="docker-args",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--if-unset", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    overrides = _parse_overrides(args.override)

    if args.format == "docker-json":
        json.dump(get_docker_section(args.runtime), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.format == "runner-shell":
        sys.stdout.write(format_runner_shell(args.runtime))
        sys.stdout.write("\n")
        return 0

    settings = load_runtime_settings(
        config_path=args.config,
        runtime_path=args.runtime,
        num_threads=args.threads,
        platform=args.platform or ("linux" if args.format in ("docker-args", "shell-exports") else None),
    )
    env = settings.process_env()
    for key, value in overrides.items():
        env[key] = value

    if args.format == "shell-exports":
        if args.if_unset:
            env = {key: value for key, value in env.items() if key not in os.environ}
        sys.stdout.write(format_shell_exports(env))
        if env:
            sys.stdout.write("\n")
        return 0

    if args.format == "docker-args":
        sys.stdout.write(format_docker_args(env))
        if env:
            sys.stdout.write("\n")
        return 0

    json.dump(env, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
