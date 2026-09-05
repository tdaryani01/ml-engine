"""Runtime threading policy: config-driven env + threadpoolctl."""
from __future__ import annotations

import argparse
import ctypes
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
    omp_thread_limit: Optional[int] = None

    def effective_omp_thread_limit(self) -> int:
        """Hard OpenMP cap (OMP_THREAD_LIMIT). Defaults to num_threads."""
        if self.omp_thread_limit is not None:
            return int(self.omp_thread_limit)
        return self.num_threads

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

        OPENBLAS_NUM_THREADS is process-global: numpy's and scipy's own,
        separately-vendored OpenBLAS builds read it too (verified via
        threadpoolctl -- this process loads three distinct OpenBLAS instances:
        ours, numpy's, and scipy's). Our own bin/libopenblas.so does not need
        this env var at all -- its thread count is set via a direct runtime
        call (sync_openblas_thread_policy -> openblas_set_num_threads on our
        loaded handle), independent of the environment. Mirroring OMP_NUM_THREADS
        into this var here, before numpy/scipy have even been imported, makes
        their separate pthreads pools latch onto that thread count at their own
        first lazy init; a later per-backend cap back to 1 (threadpool_limits)
        does not kill the already-spawned OS threads, leaving them alive as
        low-utilization stragglers for the rest of the process (measured: a
        3-thread config produced 8 live OS threads in uProf). Keep this var
        serial here; conv backends never raise it, and non-conv backends that
        legitimately want BLAS parallelism (e.g. NUMPY) get the correct value
        later from _apply_fit_thread_env, before their own fit() runs.
        """
        merged = dict(self.env)
        n = str(self.num_threads)
        limit = str(self.effective_omp_thread_limit())
        serial = "1"
        # Caller/runtime.yaml may already have set these; only fill gaps.
        merged.setdefault("OMP_NUM_THREADS", n)
        merged.setdefault("OMP_THREAD_LIMIT", limit)
        merged.setdefault("OPENBLAS_NUM_THREADS", serial)
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


def _parse_positive_int(raw: object, *, field_name: str) -> int:
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


def _optional_positive_int(raw: object, *, field_name: str) -> Optional[int]:
    if raw is None:
        return None
    return _parse_positive_int(raw, field_name=field_name)


def _resolve_num_threads(
    config_path: Path,
    runtime_raw: Mapping[str, object],
    env: Mapping[str, str],
    *,
    num_threads: Optional[int] = None,
) -> int:
    """Resolve OMP/worker thread count.

    Priority:
      1. explicit load_runtime_settings(num_threads=...) / CLI --threads
      2. runtime.yaml top-level num_threads (if provided)
      3. runtime.yaml env/platform OMP_NUM_THREADS (if provided)
      4. config.yaml optimization.num_threads (default 4)
    """
    if num_threads is not None:
        return _parse_positive_int(num_threads, field_name="num_threads")

    rt_threads = _optional_positive_int(
        runtime_raw.get("num_threads"),
        field_name="runtime.num_threads",
    )
    if rt_threads is not None:
        return rt_threads

    env_omp = env.get("OMP_NUM_THREADS")
    if env_omp is not None and str(env_omp).strip() != "":
        return _parse_positive_int(env_omp, field_name="runtime env OMP_NUM_THREADS")

    cfg = _load_yaml(config_path)
    return _parse_positive_int(
        cfg.get("optimization", {}).get("num_threads", 4),
        field_name="optimization.num_threads",
    )


def _resolve_omp_thread_limit(
    runtime_raw: Mapping[str, object],
    env: Mapping[str, str],
) -> Optional[int]:
    """Optional OMP_THREAD_LIMIT override; None means mirror num_threads."""
    rt_limit = _optional_positive_int(
        runtime_raw.get("omp_thread_limit"),
        field_name="runtime.omp_thread_limit",
    )
    if rt_limit is not None:
        return rt_limit

    env_limit = env.get("OMP_THREAD_LIMIT")
    if env_limit is not None and str(env_limit).strip() != "":
        return _parse_positive_int(env_limit, field_name="runtime env OMP_THREAD_LIMIT")

    return None


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
    resolved_platform = platform or detect_platform()
    env = _merge_env_sections(raw, resolved_platform)
    threads = _resolve_num_threads(
        cfg_path, raw, env, num_threads=num_threads
    )
    omp_limit = _resolve_omp_thread_limit(raw, env)

    blas_section = raw.get("blas_threads") or {}
    if not isinstance(blas_section, Mapping):
        raise ValueError("runtime.blas_threads must be a mapping")

    docker_section = raw.get("docker") or {}
    if not isinstance(docker_section, Mapping):
        raise ValueError("runtime.docker must be a mapping")

    # Keep resolved values in env so process_env / fit scope agree.
    env = dict(env)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_THREAD_LIMIT"] = str(
        omp_limit if omp_limit is not None else threads
    )

    return RuntimeSettings(
        num_threads=threads,
        platform=resolved_platform,
        env=env,
        blas_threads={str(k): (None if v is None else int(v)) for k, v in blas_section.items()},
        docker=dict(docker_section),
        omp_thread_limit=omp_limit,
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


@contextmanager
def current_thread_omp_threads(num_threads: int) -> Iterator[None]:
    """Temporarily set libgomp's nthreads-var for the current Python thread.

    This is intentionally not `configure_native_threads()`: that API updates the
    process-wide native policy. OpenMP's `omp_set_num_threads` updates the
    current task's nthreads-var, which lets validation/predict run serially on
    the Python thread while a persistent native async worker owns the parallel
    conv team.
    """
    if sys.platform == "win32":
        yield
        return

    try:
        libgomp = ctypes.CDLL("libgomp.so.1")
        libgomp.omp_get_max_threads.restype = ctypes.c_int
        libgomp.omp_set_num_threads.argtypes = [ctypes.c_int]
        previous = int(libgomp.omp_get_max_threads())
    except Exception:
        yield
        return

    libgomp.omp_set_num_threads(max(1, int(num_threads)))
    try:
        yield
    finally:
        libgomp.omp_set_num_threads(max(1, previous))


def _shared_omp_fit(backend: EngineBackend) -> bool:
    """True only when native/im2col+gemm actually share one LLVM OMP pool with GEMM.

    Requires both: backend is a conv backend, AND bin/libopenblas was built with
    USE_OPENMP=1 and is loaded (native_blas_unified_omp()). If bin/libopenblas.so
    is missing or not unified (e.g. Docker image built without it), numpy's own
    wheel-vendored OpenBLAS is a *second*, uncoordinated pthread pool -- mirroring
    the OMP thread count into it double-counts hardware threads (measured: a
    4-thread config produced 7 live OS threads in that case).
    """
    if backend not in _CONV_BACKENDS:
        return False
    from utils.conv_dispatch import native_blas_unified_omp
    return native_blas_unified_omp()


def _apply_fit_thread_env(settings: RuntimeSettings, backend: EngineBackend) -> Dict[str, str]:
    """Backend-specific thread caps during fit."""
    omp = settings.omp_threads_for(backend)
    omp_limit = settings.effective_omp_thread_limit()
    is_conv_backend = backend in _CONV_BACKENDS
    serial = "1"
    # OPENBLAS_NUM_THREADS is a *process-global* env var: every OpenBLAS build
    # loaded in the process reads it at its own first lazy init, not just ours.
    # This process has up to three separate OpenBLAS instances (verified via
    # threadpoolctl): our own bin/libopenblas.so (openmp-threaded, unified with
    # the conv OMP pool when native_blas_unified_omp() is True) plus numpy's and
    # scipy's independently-vendored copies (pthreads-threaded, never unified).
    # Our own library's thread count is set via a direct runtime call
    # (sync_openblas_thread_policy -> openblas_set_num_threads on our loaded
    # handle, see blas_dynamic.cpp) and does NOT need this env var at all. If we
    # mirror `omp` into it here, numpy's/scipy's separate pthreads pools read
    # the same value at their own lazy init and spawn that many OS threads;
    # threadpool_limits() later caps their *active* thread count back to 1, but
    # the already-spawned idle pthreads don't get killed -- they stick around
    # as extra, low-utilization OS threads for the rest of the process
    # (measured: 3 configured OMP threads -> 8 live threads in uProf, with the
    # extra ones tracking 1:1 with numpy's + scipy's separate OpenBLAS pools).
    # Keep this env var serial always; it never legitimately needs to be >1.
    if is_conv_backend:
        openblas_env = serial
        blas_env = serial
    else:
        openblas_env = str(settings.blas_threads_for(backend))
        blas_env = openblas_env
    overrides = {
        "OMP_NUM_THREADS": str(omp),
        "OMP_THREAD_LIMIT": str(omp_limit),
        "OPENBLAS_NUM_THREADS": openblas_env,
    }
    if omp > 1:
        # LLVM OpenMP on Windows honors KMP_DEVICE_THREAD_LIMIT; =1 causes OMP warning #96.
        overrides["KMP_DEVICE_THREAD_LIMIT"] = str(omp)
    overrides.update({
        "MKL_NUM_THREADS": blas_env,
        "NUMEXPR_NUM_THREADS": blas_env,
        "VECLIB_MAXIMUM_THREADS": blas_env,
    })
    if is_conv_backend:
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
    is_conv_backend = backend in _CONV_BACKENDS
    # For conv backends, process-global BLAS env vars stay serial so numpy/scipy
    # wheel BLAS pools cannot spawn extra pthreads. Our own bin/libopenblas.so,
    # when present and unified, is controlled by a direct native runtime call.
    openblas_fit = omp if (is_conv_backend and shared) else (1 if is_conv_backend else blas)
    scipy_blas_fit = 1 if is_conv_backend else blas
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
        f"openblas_during_fit={openblas_fit} scipy_blas_during_fit={scipy_blas_fit} "
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
        try:
            import numba
        except ImportError:
            numba = None  # type: ignore[assignment]
        if numba is not None:
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
        try:
            import numba
            numba.set_num_threads(1)
        except ImportError:
            pass
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
