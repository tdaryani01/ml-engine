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

import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from config.constants import EngineBackend

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

_VALID_PLATFORMS = ("linux", "windows")


def detect_platform() -> str:
    return "windows" if sys.platform == "win32" else "linux"


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

    def process_env(self) -> Dict[str, str]:
        merged = dict(self.env)
        n = str(self.num_threads)
        for key in _THREAD_ENV_KEYS:
            merged.setdefault(key, n)
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


def log_runtime_settings(
    settings: RuntimeSettings,
    backend: EngineBackend,
    *,
    prefix: str = "[Runtime]",
) -> None:
    blas = settings.blas_threads_for(backend)
    if settings.platform == "windows":
        tune = (
            f"KMP_BLOCKTIME={os.environ.get('KMP_BLOCKTIME', '?')} "
            f"KMP_AFFINITY={os.environ.get('KMP_AFFINITY', '?')}"
        )
    else:
        tune = (
            f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', '?')} "
            f"GOMP_SPINCOUNT={os.environ.get('GOMP_SPINCOUNT', '?')}"
        )
    msg = (
        f"{prefix} platform={settings.platform} num_threads={settings.num_threads} "
        f"backend={backend.value} blas_during_fit={blas} {tune}"
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
) -> Iterator[None]:
    """Scope BLAS/OpenMP pools for training; native keeps BLAS at 1 by default."""
    omp = settings.num_threads
    blas = settings.num_threads if blas_threads is not None else settings.blas_threads_for(backend)
    if blas == omp:
        with threadpool_limits(limits=omp):
            yield
    else:
        with threadpool_limits(limits={"openblas": blas, "blas": blas, "openmp": omp}):
            yield


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
