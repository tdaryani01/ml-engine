#!/usr/bin/env bash
# Build OpenBLAS with USE_OPENMP=1.
#   Artifacts: build/openblas/libopenblas.so (+ openblas_build.json)
#   Runtime:   bin/libopenblas.so copied from build/openblas on every successful build
#
# Notes:
# - Prefer a fresh Linux checkout (Windows trees often have CRLF scripts that break ./c_check).
# - We build the static archive (`libs`) then link a shared .so. OpenBLAS `make shared`
#   re-runs netlib LAPACK and fails on newer GCC (callback prototype errors).
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
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 --branch "$TAG" https://github.com/OpenMathLib/OpenBLAS.git "$SRC"
  else
    # No system git (common on fresh Ubuntu installs) — use the release tarball.
    ver="${TAG#v}"
    tarball="$THIRD/openblas-${TAG}.tar.gz"
    url="https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/${TAG}.tar.gz"
    echo "[openblas] git missing; downloading $url"
    wget -q "$url" -O "$tarball"
    tar -xzf "$tarball" -C "$THIRD"
    mv "$THIRD/OpenBLAS-${ver}" "$SRC"
    rm -f "$tarball"
  fi
fi

# Detect CRLF c_check from a Windows copy and refuse — rebuild from a clean tree.
if file "$SRC/c_check" 2>/dev/null | grep -qi 'CRLF'; then
  echo "[openblas] $SRC/c_check has CRLF line endings (Windows copy)." >&2
  echo "[openblas] Remove third_party/OpenBLAS and re-run this script." >&2
  exit 1
fi

make -C "$SRC" -j"$(nproc)" BINARY=64 USE_OPENMP=1 TARGET="$TARGET" NOFORTRAN=1 \
  CFLAGS="-O2 -fopenmp" libs

# Prefer an already-built .so; otherwise link one from the static archive.
SO="$(find "$SRC" -maxdepth 1 -name 'libopenblas*.so*' ! -name '*.a' | head -n1 || true)"
if [[ -z "$SO" || ! -f "$SO" ]]; then
  A="$(find "$SRC" -maxdepth 1 -name 'libopenblas*.a' | head -n1 || true)"
  if [[ -z "$A" || ! -f "$A" ]]; then
    echo "[openblas] no libopenblas.a / .so after make libs" >&2
    exit 1
  fi
  SO="$SRC/libopenblas.so"
  echo "[openblas] linking shared $SO from $(basename "$A")"
  cc -shared -o "$SO" -Wl,--whole-archive "$A" -Wl,--no-whole-archive -fopenmp -lm -lpthread -ldl
fi

cp -f "$SO" "$ARTIFACT/libopenblas.so"
cat > "$ARTIFACT/openblas_build.json" <<EOF
{"use_openmp": true, "target": "$TARGET", "tag": "$TAG"}
EOF
cp -f "$ARTIFACT/libopenblas.so" "$BIN/libopenblas.so"
cp -f "$ARTIFACT/openblas_build.json" "$BIN/openblas_build.json"
echo "[openblas] Wrote $ARTIFACT/libopenblas.so"
echo "[openblas] Copied -> $BIN/libopenblas.so (USE_OPENMP=1)"
