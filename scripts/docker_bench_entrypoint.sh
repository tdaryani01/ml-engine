#!/bin/sh
set -e

TARGET="${DOCKER_OMP_TARGET:?DOCKER_OMP_TARGET must be set (pytorch or custom)}"

if [ -n "${DOCKER_OMP_NUM_THREADS:-}" ]; then
  THREADS="${DOCKER_OMP_NUM_THREADS}"
else
  THREADS="$(python - <<'PY'
import yaml
from pathlib import Path

path = Path("/workspace/config/config.yaml")
try:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    print(int(cfg.get("optimization", {}).get("num_threads", 4)))
except Exception:
    print(4)
PY
)"
fi

eval "$(python /workspace/utils/docker_omp_env.py --target "$TARGET" --threads "$THREADS" --format shell-exports --if-unset)"

# docker run IMAGE --target=... replaces CMD; restore default benchmark command.
if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
  set -- python -u benchmarks/run_benchmarks_docker.py "$@"
fi

exec "$@"
