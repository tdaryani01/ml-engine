#!/usr/bin/env bash
# Build conv_kernels.so (g++ / clang++).
# Windows DLL build is unchanged: .\build_native.ps1
#
#   Artifacts: build/native/conv_kernels.so
#   Runtime:   bin/conv_kernels.so copied on every successful build
#
#   ./build_native.sh                 # release (default)
#   ./build_native.sh release
#   ./build_native.sh release-symbols  # -g + same opts (perf tools)
#   ./build_native.sh release-noinline # -g + optimized, no inlining/LTO (uProf attribution)
#   ./build_native.sh release-contract-profile # diagnostic per-OMP-thread contract timing
#   ./build_native.sh debug
#   ./build_native.sh release --run-tests
#
# Override compiler: CXX=clang++ ./build_native.sh
# Override CPU tune:  MARCH=haswell ./build_native.sh   (default: native)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/build/native"
BIN_DIR="$ROOT/bin"
BUILT_SO="$BUILD_DIR/conv_kernels.so"
OUT_SO="$BIN_DIR/conv_kernels.so"

MODE="${1:-release}"
RUN_TESTS=0
if [[ "${2:-}" == "--run-tests" ]] || [[ "${1:-}" == "--run-tests" ]]; then
  RUN_TESTS=1
fi
if [[ "${1:-}" == "--run-tests" ]]; then
  MODE=release
fi

case "$MODE" in
  debug|release|release-symbols|release-noinline|release-contract-profile) ;;
  *)
    echo "[build] unknown mode: $MODE (use debug|release|release-symbols|release-noinline)" >&2
    exit 1
    ;;
esac

CXX="${CXX:-}"
if [[ -z "$CXX" ]]; then
  if command -v g++ >/dev/null 2>&1; then
    CXX=g++
  elif command -v clang++ >/dev/null 2>&1; then
    CXX=clang++
  else
    echo "[build] g++ or clang++ required:" >&2
    echo "  sudo apt install build-essential libomp-dev" >&2
    exit 1
  fi
fi

# Prefer g++ for .cpp; gcc alone is wrong for this tree.
if [[ "$(basename "$CXX")" == "gcc" ]]; then
  echo "[build] refusing CXX=gcc — use g++ or clang++" >&2
  exit 1
fi

SOURCES=(
  src/native/conv_fallback.cpp
  src/native/conv_onednn_fwd.cpp
  src/native/conv_dispatcher.cpp
  src/native/omp_config.cpp
  src/native/im2col.cpp
  src/native/im2col_telemetry.cpp
  src/native/blas_dynamic.cpp
  src/native/conv_im2col_gemm.cpp
  src/native/contract_runner.cpp
)

ONEDNN_ROOT="${ONEDNN_ROOT:-$ROOT/third_party/onednn_install}"
ONEDNN_INC="$ONEDNN_ROOT/include"
ONEDNN_LIB="$ONEDNN_ROOT/lib"
if [[ ! -f "$ONEDNN_LIB/libdnnl.so" ]]; then
  echo "[build] missing $ONEDNN_LIB/libdnnl.so" >&2
  echo "[build] build oneDNN first (ninja):" >&2
  echo "[build]   cmake -G Ninja -S oneDNN -B oneDNN/build-ninja -DCMAKE_BUILD_TYPE=Release -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=third_party/onednn_install && cmake --build oneDNN/build-ninja -j && cmake --install oneDNN/build-ninja" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$BIN_DIR"

# Host ISA: default -march=native so AVX2/FMA/BMI match this CPU (AMD uProf box).
# Set MARCH=x86-64-v3 or haswell for portable binaries.
MARCH="${MARCH:-native}"

COMMON=(
  -std=c++17
  -shared
  -fPIC
  -fvisibility=hidden
  -fvisibility-inlines-hidden
  "-march=${MARCH}"
  -mtune=native
  -mavx2
  -mfma
  -fopenmp
  -I.
  -Isrc/native
  -I"$ONEDNN_INC"
  -pthread
  -pipe
)

# Hot-path math: match MSVC /O2 /Oi /Ot /Ox /GL /fp:fast /arch:AVX2 intent.
# -O3 + LTO ≈ /Ox+/GL+/LTCG; -ffast-math ≈ /fp:fast for FMADD-heavy conv.
RELEASE_BASE_OPTS=(
  -O3
  -DNDEBUG
  -ffast-math
  -fno-math-errno
  -funroll-loops
  -ftree-vectorize
  -fomit-frame-pointer
  -fno-plt
)

RELEASE_OPTS=(
  "${RELEASE_BASE_OPTS[@]}"
  -flto=auto
)

# Profiling artifact: optimized math, but no inlining/LTO so uProf can attribute
# hot native helpers instead of collapsing them into callers.
RELEASE_NOINLINE_OPTS=(
  "${RELEASE_BASE_OPTS[@]}"
  -DML_ENGINE_NO_FORCEINLINE=1
  -fno-inline
  -fno-inline-functions
  -fno-inline-small-functions
  -fno-inline-functions-called-once
  -fno-ipa-cp
  -fno-ipa-sra
)

case "$MODE" in
  release)
    CXXFLAGS=("${RELEASE_OPTS[@]}")
    LABEL="Release (O3 no-inline fast-math ${MARCH})"
    ;;
  release-symbols)
    CXXFLAGS=("${RELEASE_OPTS[@]}" -g -fno-omit-frame-pointer)
    LABEL="Release + symbols (O3 LTO fast-math ${MARCH})"
    ;;
  release-noinline)
    CXXFLAGS=("${RELEASE_NOINLINE_OPTS[@]}" -g -fno-omit-frame-pointer)
    LABEL="Release no-inline + symbols (O3 fast-math ${MARCH})"
    ;;
  release-contract-profile)
    CXXFLAGS=("${RELEASE_OPTS[@]}" -DML_ENGINE_PROFILE_CONTRACT_THREADS=1)
    LABEL="Release + diagnostic per-thread contract timing (O3 LTO fast-math ${MARCH})"
    ;;
  debug)
    CXXFLAGS=(-O0 -g -D_DEBUG -fno-omit-frame-pointer)
    LABEL="Debug"
    ;;
esac

echo "[build] $LABEL"
echo "[build] cxx: $CXX"
echo "[build] out: $BUILT_SO"

# Drop stale objects / so before link.
rm -f "$BUILT_SO"
find "$BUILD_DIR" -maxdepth 1 -name '*.o' -delete 2>/dev/null || true

# Compile each TU into build/native, then link with LTO when enabled.
OBJS=()
for src in "${SOURCES[@]}"; do
  base="$(basename "$src" .cpp)"
  obj="$BUILD_DIR/${base}.o"
  echo "[build] compile $src"
  "$CXX" "${COMMON[@]}" "${CXXFLAGS[@]}" -c "$ROOT/$src" -o "$obj"
  OBJS+=("$obj")
done

echo "[build] link $BUILT_SO"
LINK_EXTRA=(-fopenmp -ldl -pthread -L"$ONEDNN_LIB" -ldnnl -Wl,-rpath,"$BIN_DIR")
# LTO needs the same -flto flag on the link line.
if [[ " ${CXXFLAGS[*]} " == *" -flto=auto "* ]] || [[ " ${CXXFLAGS[*]} " == *" -flto "* ]]; then
  LINK_EXTRA+=(-flto=auto)
fi
"$CXX" "${COMMON[@]}" "${CXXFLAGS[@]}" "${OBJS[@]}" -o "$BUILT_SO" "${LINK_EXTRA[@]}"

if [[ ! -f "$BUILT_SO" ]]; then
  echo "[build] expected output missing: $BUILT_SO" >&2
  exit 1
fi

cp -f "$BUILT_SO" "$OUT_SO"
# Runtime: ship libdnnl next to conv_kernels.so (rpath=$BIN_DIR)
cp -f "$ONEDNN_LIB/libdnnl.so" "$BIN_DIR/libdnnl.so"
cp -af "$ONEDNN_LIB/libdnnl.so."* "$BIN_DIR/" 2>/dev/null || true
echo "[build] Wrote $BUILT_SO"
echo "[build] Copied -> $OUT_SO"
echo "[build] Copied oneDNN -> $BIN_DIR/libdnnl.so"

# Optional OpenBLAS from scripts/build_openblas.sh (Linux .so).
# Windows still uses bin/libopenblas.dll from build_native.ps1 / build_openblas.ps1.
if [[ -f "$ROOT/build/openblas/libopenblas.so" ]]; then
  cp -f "$ROOT/build/openblas/libopenblas.so" "$BIN_DIR/libopenblas.so"
  if [[ -f "$ROOT/build/openblas/openblas_build.json" ]]; then
    cp -f "$ROOT/build/openblas/openblas_build.json" "$BIN_DIR/openblas_build.json"
  fi
  echo "[build] Copied OpenBLAS -> $BIN_DIR/libopenblas.so"
elif [[ -f "$BIN_DIR/libopenblas.so" ]]; then
  echo "[build] using existing $BIN_DIR/libopenblas.so"
else
  echo "[build] optional missing: libopenblas.so (run scripts/build_openblas.sh when needed)"
fi

# Smoke: exported symbols Python expects
if command -v nm >/dev/null 2>&1; then
  NEED=(
    configure_native_threads
    direct_conv_block_forward_avx2
    im2col_avx2
    run_contract_training_step
  )
  for sym in "${NEED[@]}"; do
    if ! nm -D --defined-only "$OUT_SO" 2>/dev/null | grep -E " ${sym}$" >/dev/null; then
      echo "[build] warning: symbol not found in dynamic table: $sym" >&2
    else
      echo "[build] export ok: $sym"
    fi
  done
fi

echo "[build] bin:"
ls -la "$BIN_DIR" | sed 's/^/[build] /'

if [[ "$RUN_TESTS" -eq 1 ]]; then
  python3 "$ROOT/run_tests.py"
fi
