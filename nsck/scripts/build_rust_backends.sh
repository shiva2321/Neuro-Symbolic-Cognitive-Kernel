#!/usr/bin/env bash
# build_rust_backends.sh
# Build and install all three NSCK Rust backends (hypervec_rs, snn_rs, societal_rs).
#
# Usage (from repo root):
#   bash nsck/scripts/build_rust_backends.sh
#
# Requirements:
#   - Rust toolchain (rustc + cargo) — https://rustup.rs/
#   - maturin >= 1.0: pip install maturin
#   - Python 3.9+

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NSCK_ROOT="${REPO_ROOT}/nsck"

echo "=== NSCK V30: Building Rust Backends ==="
echo "Repo root : ${REPO_ROOT}"
echo ""

# Detect pip
PIP_BIN="$(command -v pip || command -v pip3 || true)"
if [ -z "${PIP_BIN}" ]; then
  echo "ERROR: pip not found. Please install Python and pip first."
  exit 1
fi

# Ensure maturin is installed
if ! command -v maturin &>/dev/null; then
  echo "Installing maturin..."
  "${PIP_BIN}" install --user "maturin>=1.0,<2.0" -q
fi

build_backend() {
  local name="$1"
  local dir="$2"
  echo ""
  echo "--- Building ${name} ---"
  cd "${dir}"
  maturin build --release 2>&1 | grep -E "Finished|Built wheel|error|warning: unused" || true
  wheel=$(ls target/wheels/*.whl 2>/dev/null | head -1)
  if [ -n "${wheel}" ]; then
    "${PIP_BIN}" install --user --force-reinstall "${wheel}" -q
    echo "  Installed: ${wheel##*/}"
  else
    echo "  ERROR: No wheel found in ${dir}/target/wheels/"
    exit 1
  fi
}

build_backend "hypervec_rs" "${NSCK_ROOT}/rust_vsa"
build_backend "snn_rs"      "${NSCK_ROOT}/rust_snn"
build_backend "societal_rs" "${NSCK_ROOT}/rust_societal"

echo ""
echo "=== Verifying imports ==="
python3 -c "
import hypervec_rs; print('  hypervec_rs OK — HyperVector:', type(hypervec_rs.HyperVector(42)))
import snn_rs;      print('  snn_rs OK')
import societal_rs; print('  societal_rs OK')
print()
print('All Rust backends active!')
"

echo ""
echo "=== Done ==="
echo "Run: PYTHONPATH=nsck python -m pytest nsck/tests/ -q"
