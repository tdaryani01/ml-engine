#!/usr/bin/env bash
# Build OpenBLAS with USE_OPENMP=1.
#   Artifacts: build/openblas/libopenblas.so (+ openblas_build.json)
#   Runtime:   bin/libopenblas.so copied from build/openblas on every successful build
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="$ROOT/third_party"
SRC="$THIRD/OpenBLAS"
ARTIFACT="$ROOT/build/openblas"
TAG="${OPENBLAS_TAG:-v0.3.28}"
TARGET="${OPENBLAS_TARGET:-HASWELL}"
BIN="$ROOT/bin"

mkdir -p "$BIN" "$THIRD" "$ARTIFACT"
if [[ ! -d "$SRC" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/OpenMathLib/OpenBLAS.git "$SRC"
fi

make -C "$SRC" -j"$(nproc)" BINARY=64 USE_OPENMP=1 TARGET="$TARGET" NOFORTRAN=1 \
  BUILD_SHARED_LIBS=1 CFLAGS="-O2 -fopenmp" libs

SO="$(find "$SRC" -maxdepth 1 -name 'libopenblas*.so*' | head -n1)"
cp -f "$SO" "$ARTIFACT/libopenblas.so"
cat > "$ARTIFACT/openblas_build.json" <<EOF
{"use_openmp": true, "target": "$TARGET", "tag": "$TAG"}
EOF
cp -f "$ARTIFACT/libopenblas.so" "$BIN/libopenblas.so"
cp -f "$ARTIFACT/openblas_build.json" "$BIN/openblas_build.json"
echo "[openblas] Wrote $ARTIFACT/libopenblas.so"
echo "[openblas] Copied -> $BIN/libopenblas.so (USE_OPENMP=1)"
