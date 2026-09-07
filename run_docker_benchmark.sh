#!/usr/bin/env bash
# Docker benchmark orchestrator (Linux port of run_docker_benchmark.ps1).
#
# Runs benchmarks inside pinned containers so measurements are comparable
# between runs: fixed cpuset, fixed OMP env from config/runtime.yaml, and
# PyTorch / custom engine in separate images so their deps cannot interact.
#
#   ./run_docker_benchmark.sh                    # All: build both + run both
#   ./run_docker_benchmark.sh build              # build pytorch + custom
#   ./run_docker_benchmark.sh build-custom       # custom image only (fast path)
#   ./run_docker_benchmark.sh run
#   ./run_docker_benchmark.sh clean
#   ./run_docker_benchmark.sh sweep              # kernel sweep, k-min..k-max
#   ./run_docker_benchmark.sh matrix             # full k=1-7 x s=1,2 x p=1,2
#   ./run_docker_benchmark.sh sample             # sampled kernels only
#
# Options:
#   --cores N              N physical cores for --cpuset-cpus (default 4).
#                          Picks one logical CPU per core (e.g. 0,2,4,6), not
#                          SMT siblings 0-(N-1) which are only N/2 real cores.
#   --onednn-verbose N     ONEDNN_VERBOSE override (0..2)             (default 0)
#   --verbose-tracing      shorthand for --onednn-verbose 1
#   --no-cache             docker build --no-cache
#   --k-min N / --k-max N  sweep kernel range                        (default 1..7)
#   --pad N                sweep pad                                  (default 1)
#   --sample-kernels "..." space/comma separated list               (default 1 3 4 7)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Prefer BuildKit when buildx is present; otherwise classic builder.
if docker buildx version >/dev/null 2>&1; then
  export DOCKER_BUILDKIT=1
else
  export DOCKER_BUILDKIT=0
  echo "[!] docker-buildx not installed; using classic builder (DOCKER_BUILDKIT=0)." >&2
  echo "    Optional: sudo apt install docker-buildx-plugin   # or docker-buildx" >&2
fi

PYTORCH_IMG="ml-engine-pytorch-bench:latest"
CUSTOM_IMG="ml-engine-custom-bench:latest"
RUNTIME_SCRIPT="$ROOT/utils/runtime.py"
CONFIG_PATH="$ROOT/config/config.yaml"

ACTION="all"
CORES=4
ONEDNN_VERBOSE=0
VERBOSE_TRACING=0
NO_CACHE=0
K_MIN=1
K_MAX=7
PAD=1
SAMPLE_KERNELS="1 3 4 7"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "[ERROR] python3 not found" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    all|All|build|Build|build-custom|Build-custom|run|Run|clean|Clean|sweep|Sweep|matrix|Matrix|sample|Sample)
      ACTION="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"; shift ;;
    --action) ACTION="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --cores) CORES="$2"; shift 2 ;;
    --onednn-verbose) ONEDNN_VERBOSE="$2"; shift 2 ;;
    --verbose-tracing) VERBOSE_TRACING=1; shift ;;
    --no-cache) NO_CACHE=1; shift ;;
    --k-min) K_MIN="$2"; shift 2 ;;
    --k-max) K_MAX="$2"; shift 2 ;;
    --pad) PAD="$2"; shift 2 ;;
    --sample-kernels) SAMPLE_KERNELS="${2//,/ }"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$ACTION" in
  all|build|build-custom|run|clean|sweep|matrix|sample) ;;
  *) echo "[ERROR] unknown action: $ACTION" >&2; exit 1 ;;
esac

if (( ONEDNN_VERBOSE < 0 || ONEDNN_VERBOSE > 2 )); then
  echo "[ERROR] --onednn-verbose must be 0..2" >&2
  exit 1
fi
if (( CORES < 1 )); then
  echo "[ERROR] --cores must be >= 1" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is not installed." >&2
  echo "  Install Docker Engine, then add your user to the docker group:" >&2
  echo "    sudo apt install docker.io" >&2
  echo "    sudo usermod -aG docker \"\$USER\" && newgrp docker" >&2
  exit 1
fi

# Active shells keep the group set from when they started. If docker.sock is
# group-writable and this session lacks docker, re-exec under sg docker so the
# caller still only runs one command.
if [[ -z "${ML_ENGINE_DOCKER_SG:-}" ]] && ! docker info >/dev/null 2>&1; then
  _docker_err="$(docker info 2>&1 || true)"
  if printf '%s' "$_docker_err" | grep -qi 'permission denied'; then
    _have_docker_group=0
    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
      _have_docker_group=1
    fi
    if (( _have_docker_group == 0 )) && getent group docker 2>/dev/null | grep -Eq "(^|:)${USER}(,|$)"; then
      echo "[!] docker group is configured but inactive in this shell; re-running under sg docker" >&2
      export ML_ENGINE_DOCKER_SG=1
      _quoted=()
      for _a in "$@"; do
        _quoted+=("$(printf %q "$_a")")
      done
      exec sg docker -c "cd $(printf %q "$ROOT") && exec $(printf %q "$0") ${_quoted[*]}"
    fi
  fi
fi

# One logical CPU per physical core (SMT-aware). "0-3" on this Ryzen is only
# two cores (siblings 0-1 and 2-3); we want e.g. 0,2,4,6 for --cores 4.
CPUSET="$("$PYTHON" - "$CORES" <<'PY'
import os
import sys

def parse_siblings(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out

need = max(1, int(sys.argv[1]))
base = "/sys/devices/system/cpu"
try:
    present = sorted(
        int(name[3:])
        for name in os.listdir(base)
        if name.startswith("cpu") and name[3:].isdigit()
    )
except OSError:
    present = list(range(need))

chosen: list[int] = []
seen_cores: set[frozenset[int]] = set()
for cpu in present:
    path = f"{base}/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path, encoding="utf-8") as fh:
            sibs = frozenset(parse_siblings(fh.read()))
    except OSError:
        sibs = frozenset([cpu])
    if sibs in seen_cores:
        continue
    seen_cores.add(sibs)
    chosen.append(min(sibs))
    if len(chosen) >= need:
        break

if len(chosen) < need:
    # Topology incomplete; fall back to contiguous logical CPUs.
    chosen = list(range(need))
print(",".join(str(c) for c in chosen))
PY
)"

if [[ ! -f "$RUNTIME_SCRIPT" ]]; then
  echo "[ERROR] Missing runtime loader: $RUNTIME_SCRIPT" >&2
  exit 1
fi

DIAG_REL="$(
  "$PYTHON" - <<'PY'
import sys
sys.path.insert(0, ".")
from utils.runtime import get_docker_section
print(get_docker_section().get("diagnostics_dir", "benchmark_diagnostics"))
PY
)"
DIAG_DIR="$ROOT/$DIAG_REL"
mkdir -p "$DIAG_DIR"

# OMP thread count follows config.yaml (same as PowerShell Get-ConfigThreadCount).
# --cores only controls --cpuset-cpus.
THREAD_COUNT="$(grep -oP '^\s*num_threads:\s*\K\d+' "$CONFIG_PATH" 2>/dev/null | head -1 || true)"
if [[ -z "$THREAD_COUNT" ]]; then
  THREAD_COUNT="$CORES"
fi
if [[ "$THREAD_COUNT" != "$CORES" ]]; then
  echo "[!] config num_threads=$THREAD_COUNT differs from --cores $CORES; using config value for OMP env." >&2
fi

VERBOSE_LEVEL="$ONEDNN_VERBOSE"
if (( VERBOSE_TRACING == 1 && VERBOSE_LEVEL == 0 )); then
  VERBOSE_LEVEL=1
fi

test_docker_endpoint() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  local err
  err="$(docker info 2>&1 || true)"
  echo "[!] Docker daemon unresponsive. Re-evaluating default context..." >&2
  docker context use default >/dev/null 2>&1 || true
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  err="$(docker info 2>&1 || true)"
  if printf '%s' "$err" | grep -qi 'permission denied'; then
    echo "[ERROR] permission denied talking to docker.sock" >&2
    echo "  Your account is likely missing an active docker group in this shell." >&2
    echo "  Fix once: newgrp docker   OR open a new terminal after usermod -aG docker" >&2
    exit 1
  fi
  if printf '%s' "$err" | grep -qi 'no such file or directory\|Cannot connect\|Is the docker daemon running'; then
    echo "[ERROR] Docker daemon is not running." >&2
    echo "  Start it with: sudo systemctl start docker" >&2
    exit 1
  fi
  echo "[ERROR] Docker daemon is not reachable." >&2
  echo "  docker info said:" >&2
  printf '%s\n' "$err" >&2
  exit 1
}

stop_existing_benchmark_containers() {
  local ids
  ids="$(
    {
      docker ps -q --filter "ancestor=$PYTORCH_IMG" 2>/dev/null || true
      docker ps -q --filter "ancestor=$CUSTOM_IMG" 2>/dev/null || true
    } | sort -u | tr '\n' ' '
  )"
  if [[ -n "${ids// /}" ]]; then
    echo "[!] Stopping leftover benchmark container(s): $ids"
    # shellcheck disable=SC2086
    docker stop $ids >/dev/null || true
  fi
  docker rm -f \
    ml-engine-bench-pytorch \
    ml-engine-bench-custom \
    ml-engine-bench-sweep \
    ml-engine-bench-matrix \
    ml-engine-bench-sample \
    >/dev/null 2>&1 || true
}

write_runtime_env_file() {
  local env_file="$DIAG_DIR/.runtime.env"
  local -a override_args=()
  if (( VERBOSE_LEVEL != 0 )); then
    override_args+=(--override "ONEDNN_VERBOSE=$VERBOSE_LEVEL")
  fi
  if ! "$PYTHON" "$RUNTIME_SCRIPT" --threads "$THREAD_COUNT" --platform linux \
      --format docker-args "${override_args[@]+"${override_args[@]}"}" > "$env_file"; then
    echo "[ERROR] Failed to load config/runtime.yaml" >&2
    exit 1
  fi
  printf '%s' "$env_file"
}

runtime_profile_summary() {
  "$PYTHON" - "$RUNTIME_SCRIPT" "$THREAD_COUNT" <<'PY'
import json, subprocess, sys
script, threads = sys.argv[1], sys.argv[2]
out = subprocess.run(
    [sys.executable, script, "--threads", threads, "--platform", "linux", "--format", "json"],
    capture_output=True, text=True, check=True,
).stdout
env = json.loads(out)
print(f"wait={env.get('OMP_WAIT_POLICY', 'default')}, spin={env.get('GOMP_SPINCOUNT', 'default')}")
PY
}

onednn_display() {
  if (( VERBOSE_LEVEL != 0 )); then
    printf '%s' "$VERBOSE_LEVEL"
    return
  fi
  "$PYTHON" - "$RUNTIME_SCRIPT" "$THREAD_COUNT" <<'PY'
import json, subprocess, sys
script, threads = sys.argv[1], sys.argv[2]
out = subprocess.run(
    [sys.executable, script, "--threads", threads, "--platform", "linux", "--format", "json"],
    capture_output=True, text=True, check=True,
).stdout
print(json.loads(out).get("ONEDNN_VERBOSE", "0"))
PY
}

# Usage:
#   invoke_docker_run NAME IMAGE ENV_FILE [docker-run-extra...] -- [container-cmd...]
# Everything before the first bare "--" is extra docker-run flags (e.g. --entrypoint).
# Everything after is passed to the container (entryoint may rewrite it).
invoke_docker_run() {
  local name="$1" image="$2" env_file="$3"
  shift 3
  local -a extra=() cmd=()
  local seen_sep=0
  for arg in "$@"; do
    if [[ "$arg" == "--" && $seen_sep -eq 0 ]]; then
      seen_sep=1
      continue
    fi
    if (( seen_sep == 0 )); then
      extra+=("$arg")
    else
      cmd+=("$arg")
    fi
  done

  docker run --rm \
    --name "$name" \
    "--cpuset-cpus=$CPUSET" \
    "--env-file=$env_file" \
    "${extra[@]+"${extra[@]}"}" \
    -v "$ROOT/config:/workspace/config" \
    -v "$ROOT/data:/workspace/data" \
    -v "$DIAG_DIR:/workspace/$DIAG_REL" \
    "$image" \
    "${cmd[@]+"${cmd[@]}"}"
}

build_flags=()
if (( NO_CACHE == 1 )); then
  build_flags+=(--no-cache)
fi

build_custom_container() {
  echo
  echo "[+] Verifying Docker context and endpoint connectivity..."
  test_docker_endpoint
  echo
  echo "[+] Building Custom Engine Isolated Image (scripts/Dockerfile.custom)..."
  docker build "${build_flags[@]+"${build_flags[@]}"}" \
    -f scripts/Dockerfile.custom -t "$CUSTOM_IMG" .
  echo
  echo "[OK] Custom benchmark container image successfully built."
}

build_containers() {
  echo
  echo "[+] Verifying Docker context and endpoint connectivity..."
  test_docker_endpoint
  echo
  echo "[+] Building PyTorch Isolated Image (scripts/Dockerfile.pytorch)..."
  docker build "${build_flags[@]+"${build_flags[@]}"}" \
    -f scripts/Dockerfile.pytorch -t "$PYTORCH_IMG" .
  build_custom_container
  echo
  echo "[OK] Both benchmark container images successfully built."
}

banner() {
  echo
  echo "=================================================================="
  echo "  $1"
  echo "  Hardware Allocation  : $CORES Dedicated Cores (cpuset: $CPUSET)"
  echo "  OpenMP Thread Count  : $THREAD_COUNT"
  [[ -n "${2:-}" ]] && echo "  $2"
  [[ -n "${3:-}" ]] && echo "  $3"
  echo "=================================================================="
}

run_benchmarks() {
  test_docker_endpoint
  stop_existing_benchmark_containers
  local env_file
  env_file="$(write_runtime_env_file)"

  banner "DOCKER CONVERGENCE BENCHMARK ORCHESTRATOR - ISOLATED RUN" \
    "Runtime OMP Profile  : $(runtime_profile_summary)" \
    "oneDNN Verbose Level : $(onednn_display) (config/runtime.yaml)"

  echo
  echo "[+] Executing PyTorch Isolated Benchmark Container..."
  # Entrypoint sees --target=* (starts with '-') and prepends the runner.
  invoke_docker_run ml-engine-bench-pytorch "$PYTORCH_IMG" "$env_file" \
    -- --target=pytorch

  echo
  echo "[+] Executing Custom Engine Isolated Benchmark Container..."
  invoke_docker_run ml-engine-bench-custom "$CUSTOM_IMG" "$env_file" \
    -- --target=custom
}

run_kernel_sweep() {
  test_docker_endpoint
  stop_existing_benchmark_containers
  local env_file log_name
  env_file="$(write_runtime_env_file)"
  log_name="kernel_sweep_pad${PAD}_k${K_MIN}-${K_MAX}.log"

  banner "DOCKER KERNEL SWEEP - GENERIC FALLBACK ONLY (k=${K_MIN}-${K_MAX}, pad=${PAD})" \
    "Log file             : $DIAG_DIR/$log_name"

  echo
  echo "[+] Executing kernel sweep in Custom Engine container..."
  invoke_docker_run ml-engine-bench-sweep "$CUSTOM_IMG" "$env_file" \
    --entrypoint python -- \
    -u benchmarks/sweep_kernel_pad.py \
    --k-min "$K_MIN" --k-max "$K_MAX" --pad "$PAD" \
    --output "/workspace/$DIAG_REL/$log_name"

  echo
  echo "[OK] Kernel sweep complete. Log: $DIAG_DIR/$log_name"
}

run_matrix_sweep() {
  test_docker_endpoint
  stop_existing_benchmark_containers
  local env_file stamp log_name
  env_file="$(write_runtime_env_file)"
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_name="conv_matrix_k1-7_s1-2_p1-2_$stamp.log"

  banner "DOCKER CONV MATRIX SWEEP (k=1-7, stride=1,2, pad=1,2)" \
    "Log file             : $DIAG_DIR/$log_name"

  echo
  echo "[+] Executing full conv matrix in Custom Engine container..."
  invoke_docker_run ml-engine-bench-matrix "$CUSTOM_IMG" "$env_file" \
    --entrypoint python -- \
    -u benchmarks/sweep_kernel_pad.py \
    --full-matrix \
    --output "/workspace/$DIAG_REL/$log_name"

  echo
  echo "[OK] Conv matrix sweep complete. Log: $DIAG_DIR/$log_name"
}

run_sample_sweep() {
  test_docker_endpoint
  stop_existing_benchmark_containers
  local env_file stamp kernel_tag log_name
  env_file="$(write_runtime_env_file)"
  stamp="$(date +%Y%m%d_%H%M%S)"
  kernel_tag="$(echo "$SAMPLE_KERNELS" | tr ' ' '-')"
  log_name="conv_sample_k${kernel_tag}_s1-2_p1-2_$stamp.log"

  banner "DOCKER CONV SAMPLE SWEEP (kernels=$kernel_tag, stride=1,2, pad=1,2)" \
    "Log file             : $DIAG_DIR/$log_name"

  # shellcheck disable=SC2206
  local -a kernel_args=($SAMPLE_KERNELS)

  echo
  echo "[+] Executing sample conv sweep in Custom Engine container..."
  invoke_docker_run ml-engine-bench-sample "$CUSTOM_IMG" "$env_file" \
    --entrypoint python -- \
    -u benchmarks/sweep_kernel_pad.py \
    --kernels "${kernel_args[@]}" \
    --strides 1 2 \
    --pads 1 2 \
    --output "/workspace/$DIAG_REL/$log_name"

  echo
  echo "[OK] Sample conv sweep complete. Log: $DIAG_DIR/$log_name"
}

clean_containers() {
  echo
  echo "[+] Pruning benchmark containers and dangling build stages..."
  docker image rm -f "$PYTORCH_IMG" "$CUSTOM_IMG" >/dev/null 2>&1 || true
  docker builder prune -f
  echo "[OK] Benchmark artifacts cleaned."
}

case "$ACTION" in
  build)         build_containers ;;
  build-custom)  build_custom_container ;;
  run)           run_benchmarks ;;
  clean)         clean_containers ;;
  sweep)         build_custom_container; run_kernel_sweep ;;
  matrix)        build_custom_container; run_matrix_sweep ;;
  sample)        build_custom_container; run_sample_sweep ;;
  all)           build_containers; run_benchmarks ;;
esac
