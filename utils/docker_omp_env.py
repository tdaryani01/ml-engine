"""Load Docker benchmark config from config/docker_omp.yaml."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "docker_omp.yaml"
VALID_TARGETS = ("pytorch", "custom")
_ENV_SECTIONS = ("shared", "pytorch", "custom")


def _resolve_num_threads(num_threads: Optional[int]) -> int:
    if num_threads is not None:
        return int(num_threads)
    env_val = os.environ.get("OMP_NUM_THREADS")
    if env_val:
        return int(env_val)
    return 4


def _substitute(value: str, num_threads: int) -> str:
    return value.replace("{num_threads}", str(num_threads))


def load_docker_config(config_path: Optional[Path] = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


load_docker_omp_config = load_docker_config


def get_docker_section(config_path: Optional[Path] = None) -> dict:
    cfg = load_docker_config(config_path)
    docker_cfg = cfg.get("docker") or {}
    if not isinstance(docker_cfg, Mapping):
        raise ValueError("config section 'docker' must be a mapping")
    return dict(docker_cfg)


def get_benchmark_runner_argv(config_path: Optional[Path] = None) -> List[str]:
    docker_cfg = get_docker_section(config_path)
    runner = docker_cfg.get("benchmark_runner")
    if runner is None:
        return ["python", "-u", "benchmarks/run_benchmarks_docker.py"]
    if not isinstance(runner, list) or not runner:
        raise ValueError("docker.benchmark_runner must be a non-empty list")
    return [str(part) for part in runner]


def get_onednn_verbose_default(config_path: Optional[Path] = None) -> int:
    env = build_docker_omp_env("pytorch", config_path=config_path)
    return int(env.get("ONEDNN_VERBOSE", "0"))


def build_docker_omp_env(
    target: str,
    num_threads: Optional[int] = None,
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    cfg = load_docker_config(config_path)
    threads = _resolve_num_threads(num_threads)

    merged: Dict[str, str] = {}
    for section in ("shared", target):
        section_values = cfg.get(section) or {}
        if not isinstance(section_values, Mapping):
            raise ValueError(f"config section {section!r} must be a mapping")
        for key, raw_value in section_values.items():
            merged[str(key)] = _substitute(str(raw_value), threads)

    if overrides:
        for key, value in overrides.items():
            merged[str(key)] = _substitute(str(value), threads)

    return merged


def apply_docker_omp_env(
    target: str,
    num_threads: Optional[int] = None,
    *,
    overwrite: bool = False,
    if_unset: bool = False,
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    env = build_docker_omp_env(
        target,
        num_threads=num_threads,
        config_path=config_path,
        overrides=overrides,
    )
    for key, value in env.items():
        if if_unset and key in os.environ:
            continue
        if overwrite or if_unset or key not in os.environ:
            os.environ[key] = value
    return env


def format_shell_exports(env: Mapping[str, str]) -> str:
    lines = []
    for key, value in env.items():
        escaped = value.replace("'", "'\"'\"'")
        lines.append(f"export {key}='{escaped}'")
    return "\n".join(lines)


def format_docker_args(env: Mapping[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in env.items())


def format_dockerfile_env(env: Mapping[str, str]) -> str:
    return "\n".join(f"ENV {key}={value}" for key, value in env.items())


def format_runner_shell(config_path: Optional[Path] = None) -> str:
    return " ".join(shlex.quote(part) for part in get_benchmark_runner_argv(config_path))


def _parse_overrides(values: Optional[Iterable[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not values:
        return overrides
    for item in values:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        overrides[key] = value
    return overrides


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit Docker benchmark env/settings from config/docker_omp.yaml"
    )
    parser.add_argument("--target", choices=VALID_TARGETS)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("docker-args", "shell-exports", "dockerfile", "json", "runner-shell", "docker-json"),
        default="docker-args",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override env KEY=VALUE (repeatable).",
    )
    parser.add_argument(
        "--if-unset",
        action="store_true",
        help="With shell-exports: only emit vars not already set in the environment.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    overrides = _parse_overrides(args.override)

    if args.format == "docker-json":
        json.dump(get_docker_section(args.config), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.format == "runner-shell":
        sys.stdout.write(format_runner_shell(args.config))
        sys.stdout.write("\n")
        return 0

    if not args.target:
        raise SystemExit("--target is required unless --format is runner-shell or docker-json")

    env = build_docker_omp_env(
        args.target,
        num_threads=args.threads,
        config_path=args.config,
        overrides=overrides,
    )

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

    if args.format == "dockerfile":
        sys.stdout.write(format_dockerfile_env(env))
        if env:
            sys.stdout.write("\n")
        return 0

    json.dump(env, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
