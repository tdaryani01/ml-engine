#!/usr/bin/env bash
# profile_engine.sh — Linux port of profile_engine.ps1 (AMD uProf 5.3)
#
# Analysis types:
#   Hotspots | SourceDisasm | Assess | ConcurrencyBound | Memory | Cache |
#   MemAlloc | BackwardDiag | Threading
#
# Memory  -> AMDuProfPcm hardware counters (IPC, L1/L2/DRAM, AVX, branches)
# Cache   -> AMDuProfCLI --config memory (IBS OP / false-sharing; GUI Cache Analysis)
#
# Usage:
#   ./profile_engine.sh
#   ./profile_engine.sh --analysis Memory
#   ./profile_engine.sh --analysis Cache
#   ./profile_engine.sh --analysis Assess --system-wide
#   ./profile_engine.sh --analysis Hotspots --python-bpf
#   ./profile_engine.sh --analysis MemAlloc
#   ./profile_engine.sh --cprofile --target run_pipeline.py
#   ./profile_engine.sh -- -- .venv/bin/python benchmarks/benchmark_cnn.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

UPROF_ROOT="${UPROF_ROOT:-/opt/AMDuProf_5.3-521}"
UPROF_CLI="${UPROF_CLI:-$UPROF_ROOT/bin/AMDuProfCLI}"
UPROF_PCM="${UPROF_PCM:-$UPROF_ROOT/bin/AMDuProfPcm}"
CONFIG_DIR="${UPROF_ROOT}/bin/Data/Config"

TARGET_SCRIPT="${TARGET_SCRIPT:-run_pipeline.py}"
ANALYSIS="${ANALYSIS:-Memory}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.AMDuProf/AMDuProf}"
REPORT_DIR="${REPORT_DIR:-$ROOT/uprof_reports}"
ACTIVE_CORES="${ACTIVE_CORES:-4}"
PCM_DURATION="${PCM_DURATION:-30}"
CPROFILE_OUT="${CPROFILE_OUT:-train.prof}"

SYSTEM_WIDE=0
INCLUDE_DISASM=0
ENABLE_CPROFILE=0
CHECK_CONTIGUITY=0
MOCK_DX_EDGES=0
PYTHON_BPF=0
TARGET_CMD=()

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    PYTHON="python3"
  fi
fi

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --analysis|-a) ANALYSIS="$2"; shift 2 ;;
    --target|-t) TARGET_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --output-dir|-o) OUTPUT_DIR="$2"; shift 2 ;;
    --report-dir) REPORT_DIR="$2"; shift 2 ;;
    --pcm-duration) PCM_DURATION="$2"; shift 2 ;;
    --cores) ACTIVE_CORES="$2"; shift 2 ;;
    --system-wide) SYSTEM_WIDE=1; shift ;;
    --disasm|--include-disasm) INCLUDE_DISASM=1; shift ;;
    --cprofile) ENABLE_CPROFILE=1; shift ;;
    --cprofile-out) CPROFILE_OUT="$2"; shift 2 ;;
    --check-contiguity) CHECK_CONTIGUITY=1; shift ;;
    --mock-dx-edges) MOCK_DX_EDGES=1; shift ;;
    --python-bpf) PYTHON_BPF=1; shift ;;
    --)
      shift
      TARGET_CMD=("$@")
      break
      ;;
    *)
      echo "[ERROR] unknown arg: $1 (use --help)" >&2
      exit 1
      ;;
  esac
done

case "$ANALYSIS" in
  Hotspots|SourceDisasm|Assess|ConcurrencyBound|Memory|Cache|DataAccess|MemAlloc|BackwardDiag|Threading) ;;
  *)
    echo "[ERROR] bad --analysis '$ANALYSIS'" >&2
    echo "  Hotspots|SourceDisasm|Assess|ConcurrencyBound|Memory|Cache|DataAccess|MemAlloc|BackwardDiag|Threading" >&2
    exit 1
    ;;
esac

BIN_DIR="$ROOT/bin"
SRC_DIR="$ROOT/src/native"
SO_PATH="$BIN_DIR/conv_kernels.so"
ZEN3_CFG="$ROOT/uprof_configs/0x19_0x5.conf"

if [[ ! -x "$UPROF_CLI" ]]; then
  echo "[ERROR] AMDuProfCLI not found: $UPROF_CLI" >&2
  echo "  Set UPROF_ROOT to your install (default /opt/AMDuProf_5.3-521)" >&2
  exit 1
fi

ensure_zen3_pcm_config() {
  mkdir -p "$(dirname "$ZEN3_CFG")"
  [[ -f "$ZEN3_CFG" ]] && return 0
  local base="$CONFIG_DIR/0x19_0x4.conf"
  if [[ ! -f "$base" ]]; then
    echo "[ERROR] missing base PCM config: $base" >&2
    exit 1
  fi
  echo "[INIT] Writing $ZEN3_CFG (model 0x50-0x5f from 0x19_0x4)"
  sed 's/modellow="40"/modellow="50"/; s/modelhigh="4f"/modelhigh="5f"/' "$base" > "$ZEN3_CFG"
}

check_native_so() {
  if [[ ! -f "$SO_PATH" ]]; then
    echo "[WARN] missing $SO_PATH — run ./build_native.sh release-symbols" >&2
    return 0
  fi
  if file "$SO_PATH" | grep -qi 'not stripped'; then
    echo "  Native symbols:     $SO_PATH (unstripped)"
  else
    echo "  Native symbols:     $SO_PATH (stripped — prefer ./build_native.sh release-symbols)"
  fi
}

ensure_zen3_pcm_config

EXEC_ARGS=()
if ((${#TARGET_CMD[@]} > 0)); then
  EXEC_ARGS=("${TARGET_CMD[@]}")
elif [[ "$ENABLE_CPROFILE" -eq 1 ]]; then
  EXEC_ARGS=(-m cProfile -o "$CPROFILE_OUT" "$TARGET_SCRIPT")
else
  EXEC_ARGS=("$TARGET_SCRIPT")
fi

echo "=================================================================="
echo "        AMD Performance Profiler (Linux): $ANALYSIS"
echo "=================================================================="
if [[ "$SYSTEM_WIDE" -eq 1 ]]; then
  echo "  Scope:              System-Wide (--system-wide)"
else
  echo "  Scope:              Target Application Only"
fi
echo "  Python:             $PYTHON"
echo "  Executing:          $PYTHON ${EXEC_ARGS[*]}"
echo "  Active Cores hint:  $ACTIVE_CORES"
echo "  Output dir:         $OUTPUT_DIR"
echo "  Report dir:         $REPORT_DIR"
check_native_so
if [[ "$MOCK_DX_EDGES" -eq 1 ]]; then
  export BWD_DX_MOCK_EDGES=1
  echo "  BWD_DX_MOCK_EDGES:  1"
fi
echo "=================================================================="
echo

rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR" "$OUTPUT_DIR"

if [[ "$CHECK_CONTIGUITY" -eq 1 ]]; then
  echo "[PRE-CHECK] NumPy / contiguity smoke..."
  "$PYTHON" - <<'PY'
import numpy as np
print(f"NumPy Version: {np.__version__}")
print("Contiguity rules: C-Contiguous buffers have stride[-1] == itemsize and zero pointer gaps.")
PY
  echo
fi

# ---------------------------------------------------------------------------
# TRACK A: Memory -> AMDuProfPcm (same counter families as Windows script)
# ---------------------------------------------------------------------------
if [[ "$ANALYSIS" == "Memory" ]]; then
  if [[ ! -x "$UPROF_PCM" ]]; then
    echo "[ERROR] AMDuProfPcm not found: $UPROF_PCM" >&2
    exit 1
  fi
  echo "[1/3] Launching AMDuProf PCM (Zen3 config, ${PCM_DURATION}s / until app exits)..."
  echo "  Config: $ZEN3_CFG"
  echo "  Counters: IPC, AVX GFLOPs, L1/L2 DC miss, DRAM fills, SSE/AVX stalls, mispred branches, Eff Freq"

  set +e
  "$UPROF_PCM" \
    -i "$ZEN3_CFG" \
    -a \
    -d "$PCM_DURATION" \
    -O "$REPORT_DIR" \
    -w "$ROOT" \
    -- \
    "$PYTHON" "${EXEC_ARGS[@]}"
  pcm_rc=$?
  set -e
  if [[ $pcm_rc -ne 0 ]]; then
    echo "[WARN] AMDuProfPcm exit=$pcm_rc (perf_event_paranoid / capabilities?)" >&2
    echo "  Try: sudo sysctl kernel.perf_event_paranoid=1" >&2
  fi

  echo
  echo "[2/3] PARSING PCM HARDWARE COUNTERS..."
  echo "========================================================================================================"
  "$PYTHON" - "$REPORT_DIR" <<'PY'
import csv, glob, os, re, sys

report_dir = sys.argv[1]
share_path = os.path.join(report_dir, "SHARE.txt")
files = sorted(
    glob.glob(os.path.join(report_dir, "**", "*.csv"), recursive=True),
    key=os.path.getmtime,
    reverse=True,
)
if not files:
    print("No PCM CSV generated.")
    raise SystemExit(0)
path = files[0]
print(f"Report: {path}\n")
lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
header_idx = -1
for i, line in enumerate(lines):
    if "Retired Instructions" in line and "IPC" in line:
        header_idx = i
        break
if header_idx < 0:
    for line in lines[:25]:
        print(line)
    raise SystemExit(0)

rows = list(csv.DictReader(lines[header_idx:]))
# Short labels for terminal width
cols = [
    ("IPC", "IPC (Sys + User)"),
    ("AVX GFLOPs", "Retired SSE/AVX Flops(GFLOPs)"),
    ("Util%", "Utilization (%)"),
    ("L1miss", "L1 DC Miss (pti)"),
    ("L2hit", "L2 Hit from DC Miss (pti)"),
    ("L2miss", "L2 Miss from DC Miss (pti)"),
    ("DRAM", "DC Fills From Local Memory (pti)"),
    ("AVX stall", "Mixed SSE/AVX Stalls (pti)"),
    ("MisBr", "Retired Branches Mispredicted (pti)"),
    ("MHz", "Eff Freq (MHz)"),
]

def num(v):
    if v is None:
        return 0.0
    s = str(v).strip().replace(",", "")
    if not s or s.lower() in {"nan", "n/a", "-"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else 0.0

def row_vals(row):
    out = []
    for _, key in cols:
        v = row.get(key, "")
        if key == "Mixed SSE/AVX Stalls (pti)" and (v is None or str(v).strip() == ""):
            v = "0.00"
        out.append(v if v is not None else "")
    return out

def is_all_zero(row):
    vals = []
    for _, key in cols:
        vals.append(num(row.get(key)))
    # Drop idle/empty samples: everything ~0 and util~0
    return all(abs(v) < 1e-9 for v in vals)

# Prefer Package/Core id columns if present
id_keys = [k for k in (rows[0].keys() if rows else []) if k and re.search(r"core|pkg|package|ccx|cpu|socket|time", k, re.I)]

alive = [r for r in rows if not is_all_zero(r)]
# Sort by util then IPC descending so useful cores float up
alive.sort(key=lambda r: (num(r.get("Utilization (%)")), num(r.get("IPC (Sys + User)"))), reverse=True)

widths = [max(len(lab), 8) for lab, _ in cols]
id_w = 10
print("PCM CORE METRICS  (idle/all-zero rows omitted)")
print("=" * 110)
hdr = f"{'ID':<{id_w}} " + " ".join(f"{lab:>{w}}" for (lab, _), w in zip(cols, widths))
print(hdr)
print("-" * len(hdr))

share_lines = ["# uProf SHARE — Memory (PCM)", "", "Idle/all-zero rows removed.", "", "```", hdr, "-" * len(hdr)]

shown = 0
for row in alive[:24]:
    rid = ""
    for k in id_keys:
        if row.get(k) not in (None, ""):
            rid = str(row.get(k))
            break
    if not rid:
        rid = str(shown)
    cells = row_vals(row)
    line = f"{rid[:id_w]:<{id_w}} " + " ".join(
        f"{str(c)[:w]:>{w}}" for c, w in zip(cells, widths)
    )
    print(line)
    share_lines.append(line)
    shown += 1

print("-" * len(hdr))
print(f"Shown {shown} active rows / {len(rows)} total (dropped {len(rows) - len(alive)} all-zero)")
share_lines.append("-" * len(hdr))
share_lines.append(f"Shown {shown} / {len(rows)} (dropped {len(rows) - len(alive)} zeros)")
share_lines.append("```")
share_lines.append("")

with open(share_path, "w", encoding="utf-8") as f:
    f.write("\n".join(share_lines) + "\n")
print(f"\nPaste file: {share_path}")
PY
  echo
  echo "NOTE: Memory = PCM counters only (no function hotspots)."
  echo "      Cache/IBS false-sharing: ./profile_engine.sh --analysis Cache"
  exit 0
fi

# ---------------------------------------------------------------------------
# TRACK B: AMDuProfCLI collect + report
# ---------------------------------------------------------------------------
case "$ANALYSIS" in
  Hotspots|SourceDisasm|BackwardDiag|ConcurrencyBound) COLLECT_PRESET=hotspots ;;
  Assess|MemAlloc) COLLECT_PRESET=assess ;;
  Cache) COLLECT_PRESET=memory ;;
  DataAccess) COLLECT_PRESET=data_access ;;
  Threading) COLLECT_PRESET=threading ;;
  *) COLLECT_PRESET=assess ;;
esac

USE_DISASM=0
if [[ "$ANALYSIS" == "SourceDisasm" || "$ANALYSIS" == "BackwardDiag" || "$INCLUDE_DISASM" -eq 1 ]]; then
  USE_DISASM=1
fi

COLLECT_ARGS=(collect --config "$COLLECT_PRESET")
if [[ "$SYSTEM_WIDE" -eq 1 ]]; then
  COLLECT_ARGS+=(--system-wide)
fi
if [[ "$COLLECT_PRESET" == "hotspots" || "$COLLECT_PRESET" == "threading" ]]; then
  COLLECT_ARGS+=(-g)
fi
if [[ "$PYTHON_BPF" -eq 1 ]]; then
  if [[ "$COLLECT_PRESET" != "hotspots" ]]; then
    echo "[WARN] --python-bpf only applies to Hotspots; ignoring for $ANALYSIS" >&2
  else
    COLLECT_ARGS+=(--sampling-mode bpf --python)
  fi
fi
COLLECT_ARGS+=(-w "$ROOT" -o "$OUTPUT_DIR" "$PYTHON" "${EXEC_ARGS[@]}")

echo "[1/4] Running AMDuProf Collection ($COLLECT_PRESET)..."
echo "  ${UPROF_CLI} ${COLLECT_ARGS[*]}"
START_TS=$(date +%s.%N)
set +e
"$UPROF_CLI" "${COLLECT_ARGS[@]}"
collect_rc=$?
set -e
END_TS=$(date +%s.%N)
WALL_SEC=$("$PYTHON" -c "print(round(float('$END_TS')-float('$START_TS'), 3))")

if [[ $collect_rc -ne 0 ]]; then
  echo "[ERROR] AMDuProf collection failed (exit $collect_rc)" >&2
  exit "$collect_rc"
fi

SESSION_DIR="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'AMDuProf-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [[ -z "${SESSION_DIR:-}" || ! -d "$SESSION_DIR" ]]; then
  echo "[ERROR] no AMDuProf-* session under $OUTPUT_DIR" >&2
  exit 1
fi

echo "[2/4] Generating reports from: $SESSION_DIR"
SUMMARY_CSV="$REPORT_DIR/summary_report.csv"
DETAIL_CSV="$REPORT_DIR/detail_report.csv"

PATH_ARGS=(--bin-path "$BIN_DIR" --symbol-path "$BIN_DIR")
if [[ -d "$SRC_DIR" ]]; then
  PATH_ARGS+=(--src-path "$SRC_DIR")
else
  PATH_ARGS+=(--src-path "$ROOT")
fi

"$UPROF_CLI" report -i "$SESSION_DIR" --report-output "$SUMMARY_CSV" --show-sample-count "${PATH_ARGS[@]}"
if [[ "$USE_DISASM" -eq 1 ]]; then
  "$UPROF_CLI" report -i "$SESSION_DIR" --report-output "$DETAIL_CSV" \
    --disasm --disasm-style intel --show-sample-count "${PATH_ARGS[@]}"
fi

if [[ "$ANALYSIS" == "Cache" ]]; then
  CACHE_CSV="$REPORT_DIR/cache_memory_view.csv"
  set +e
  "$UPROF_CLI" report -i "$SESSION_DIR" --view memory --report-output "$CACHE_CSV" \
    --show-sample-count "${PATH_ARGS[@]}"
  # richer IBS miss-rate views (same session)
  for view in ibs_op_ld ibs_op_ld_lat ibs_op_ls_overview; do
    "$UPROF_CLI" report -i "$SESSION_DIR" --view "$view" \
      --report-output "$REPORT_DIR/${view}.csv" \
      --show-sample-count "${PATH_ARGS[@]}" 2>/dev/null || true
  done
  set -e
fi

if [[ "$ANALYSIS" == "Assess" || "$ANALYSIS" == "DataAccess" ]]; then
  set +e
  for view in dc_focus dc_assess dtlb_focus triage_assess; do
    "$UPROF_CLI" report -i "$SESSION_DIR" --view "$view" \
      --report-output "$REPORT_DIR/${view}.csv" \
      --show-sample-count "${PATH_ARGS[@]}" 2>/dev/null || true
  done
  set -e
fi

echo
echo "[3/4] SYSTEM & PIPELINE METRICS ($ANALYSIS):"
echo "========================================================================================================"
echo "Wall time (collect): ${WALL_SEC}s"

if [[ -f "$SUMMARY_CSV" ]]; then
  "$PYTHON" "$ROOT/scripts/uprof_report_summary.py" "$SUMMARY_CSV" \
    --analysis "$ANALYSIS" --wall-sec "$WALL_SEC" \
    --share-out "$REPORT_DIR/SHARE.txt"
fi

if [[ "$ANALYSIS" == "BackwardDiag" ]]; then
  echo
  echo "BACKWARD DIAG CHECKLIST:"
  echo "  1. Sources / disasm -> conv_fallback.cpp: process_bwd_dx_tile vs process_dw_nci_task"
  echo "  2. Re-run --analysis Memory for core L1/L2/DRAM PCM"
  echo "  3. Re-run --analysis Cache for IBS false-sharing"
  echo "  4. Re-run --cprofile for Python vs native split"
  echo "  Detail report: $DETAIL_CSV"
fi

if [[ "$ANALYSIS" == "MemAlloc" ]]; then
  echo
  echo "========================================================================================================"
  echo "           PYTHON TRACEMALLOC: HEAP ALLOCATION & BUFFER BREAKDOWN"
  echo "========================================================================================================"
  "$PYTHON" - "$TARGET_SCRIPT" <<'PY'
import tracemalloc, runpy, sys
tracemalloc.start(25)
target = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(target, run_name="__main__")
finally:
    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n  Current Live Allocated Heap:  {current / (1024*1024):>10.3f} MB")
    print(f"  Peak Dynamic Heap Footprint:  {peak / (1024*1024):>10.3f} MB")
    print("-" * 100)
    print("\nTOP 15 ALLOCATING CALL SITES & SIZES:")
    print("-" * 100)
    print(f"{'Source File / Line':<60} {'Total Size (KB)':<18} {'Count':<10}")
    print("-" * 90)
    for stat in snapshot.statistics("lineno")[:15]:
        print(f"{str(stat.traceback):<60} {stat.size / 1024:>14.2f} KB {stat.count:>10}")
PY
fi

if [[ "$ENABLE_CPROFILE" -eq 1 && -f "$CPROFILE_OUT" ]]; then
  echo
  echo "========================================================================================================"
  echo "                         PYTHON CPROFILE BREAKDOWN ($CPROFILE_OUT)"
  echo "========================================================================================================"
  "$PYTHON" - "$CPROFILE_OUT" <<'PY'
import pstats, os, sys
from collections import defaultdict
prof = sys.argv[1]
p = pstats.Stats(prof)
print("\nTOP 10 PYTHON MODULES (AGGREGATED SELF-TIME):")
print("-" * 80)
mod_tot, mod_cum, mod_calls = defaultdict(float), defaultdict(float), defaultdict(int)
for (fn, ln, func), (cc, nc, tt, ct, callers) in p.stats.items():
    m = os.path.basename(fn) if fn != "~" else "<built-in>"
    mod_tot[m] += tt
    mod_cum[m] = max(mod_cum[m], ct)
    mod_calls[m] += nc
print(f"{'Module / File':<35} {'Self Time (s)':<16} {'Max CumTime (s)':<16} {'Total Calls':<12}")
print("-" * 80)
for m, tt in sorted(mod_tot.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{m:<35} {tt:<16.4f} {mod_cum[m]:<16.4f} {mod_calls[m]:<12d}")
print("\nTOP 10 PYTHON FUNCTIONS (SORTED BY SELF-TIME):")
print("-" * 80)
p.strip_dirs().sort_stats("tottime").print_stats(10)
PY
fi

echo
echo "[4/4] Done. Session: $SESSION_DIR"
echo "      Reports: $REPORT_DIR"
echo "      Paste me:  $REPORT_DIR/SHARE.txt"
