# NSCK Rust Build Guide

This document explains how to build the optional Rust extensions for NSCK.
The Python fallback shim works without Rust; the Rust build provides
significant speedups for VSA and SNN operations.

---

## Prerequisites

| Tool | Minimum Version | Install |
|------|----------------|---------|
| Rust | 1.75 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| maturin | 1.4 | `pip install maturin` |
| Python | 3.9 | system / conda / pyenv |
| unzip | any | `apt install unzip` (Linux) |

---

## Quick Build (Verified — February 2026)

```bash
# From repo root
pip install maturin

# 1. Build hypervec_rs (VSA + concurrent memory)
cd nsck/rust_vsa
maturin build --release
unzip -o target/wheels/hypervec_rs-*.whl "hypervec_rs/hypervec_rs*" -d /tmp/hv
cp /tmp/hv/hypervec_rs/*.so ../hypervec_rs.so

# 2. Build snn_rs (Spiking Neural Network)
cd ../rust_snn
maturin build --release
unzip -o target/wheels/snn_rs-*.whl "snn_rs/snn_rs*" -d /tmp/snn
cp /tmp/snn/snn_rs/*.so ../snn_rs.so
```

After a successful build the `.so` files appear in:
```
nsck/hypervec_rs.so   ← VSA engine + concurrent memory (gitignored)
nsck/snn_rs.so        ← Spiking Neural Network layer (gitignored)
```

NSCK auto-detects the Rust extensions at import time via the shim
(`python/core/vsa/hypervec_shim.py`).  If extensions are absent, the pure
Python fallback is used transparently.

---

## Detailed Steps

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
rustup update
```

Verify:
```bash
rustc --version   # rustc 1.75.0 (or later)
cargo --version
```

### 2. Install maturin

```bash
pip install "maturin>=1.4"
```

### 3. Build `hypervec_rs`

```bash
cd nsck/rust_vsa
maturin build --release
```

Extract the `.so` from the built wheel:
```bash
unzip -o target/wheels/hypervec_rs-*.whl "hypervec_rs/hypervec_rs*" -d /tmp/hv
cp /tmp/hv/hypervec_rs/*.so ../hypervec_rs.so
```

`hypervec_rs` provides:
- `HyperVector` — 10 240-bit VSA vector (XOR bind, bundle, cosine similarity)
- `HyperVectorRegistry` — fast nearest-neighbour lookup
- `SemanticMemoryConcurrent` / `EpisodicMemoryConcurrent` — thread-safe Rust memory
- `CognitiveWorkerPool` — Rayon parallel task execution
- `PersistentStorage` — SQLite-backed HV persistence (via rusqlite)
- `AsyncCognitiveRuntime` — Tokio async runtime for concurrent decisions
- **`bundle_hvs`** *(V4 new)* — correct N-vector majority-vote bundle
- **`lsh_bucket`** *(V4 new)* — LSH bucket key for ProceduralMemory O(1) lookup
- **`spreading_activation_step`** *(V4 new)* — one step of graph spreading activation hot path

> **As of V4, `NSCK_USE_RUST=1` is the default. Rust is expected to be built.**

Verify Rust is active:
```python
python -c "import hypervec_rs; print('Rust OK:', hypervec_rs.HyperVector(1).bits[:5])"
```

### 4. Build `snn_rs`

```bash
cd nsck/rust_snn
maturin build --release
unzip -o target/wheels/snn_rs-*.whl "snn_rs/snn_rs*" -d /tmp/snn
cp /tmp/snn/snn_rs/*.so ../snn_rs.so
```

`snn_rs` provides:
- `LIFLayer` / `SnnCore` — Leaky Integrate-and-Fire neuron layer
- `StdpEngine` — spike-timing-dependent plasticity
- `HebbianMatrix` — Hebbian weight matrix
- `ConceptMapper` — SNN → HV concept mapping
- `RateCoder` — rate-coded spike encoder

### 5. Verify

```python
import sys
sys.path.insert(0, 'nsck')      # run from repo root

import hypervec_rs
print("hypervec_rs classes:", [c for c in dir(hypervec_rs) if not c.startswith('_')])

import snn_rs
print("snn_rs classes:", [c for c in dir(snn_rs) if not c.startswith('_')])

# Confirm shim uses Rust
from nsck.python.core.vsa import hypervec_shim as shim
# prints: >> [VSA] Using Rust Accelerator (hypervec_rs) [10-100x Performance]
```

---

## Build in CI (GitHub Actions)

```yaml
- name: Install Rust
  uses: dtolnay/rust-toolchain@stable

- name: Install maturin
  run: pip install maturin

- name: Build hypervec_rs
  run: |
    cd nsck/rust_vsa
    maturin build --release
    unzip -o target/wheels/hypervec_rs-*.whl "hypervec_rs/hypervec_rs*" -d /tmp/hv
    cp /tmp/hv/hypervec_rs/*.so ../hypervec_rs.so

- name: Build snn_rs
  run: |
    cd nsck/rust_snn
    maturin build --release
    unzip -o target/wheels/snn_rs-*.whl "snn_rs/snn_rs*" -d /tmp/snn
    cp /tmp/snn/snn_rs/*.so ../snn_rs.so

- name: Run tests with Rust
  run: python -m pytest nsck/tests/ --tb=short -q
```

---

## Performance Impact

Verified on x86-64 Linux (rustc 1.93.1, Python 3.12, February 2026):

| Operation | Python fallback | Rust extension | Speedup |
|-----------|----------------|----------------|---------|
| VSA bind (XOR) | ~800 K ops/s | **~2.6 M ops/s** | **3×** |
| VSA similarity | ~125 K ops/s | **~2.7 M ops/s** | **22×** |
| VSA bundle | ~21 K ops/s | **~1.0 M ops/s** | **50×** |
| Memory query (1K) | ~15 ms | ~0.6 ms | **25×** |

> **Note**: V4 modules (`bundle_hvs`, `lsh_bucket`, `spreading_activation_step`) are pure Rust
> and do not require additional Python dependencies. VSA operations inside enrichment modules
> accelerate when Rust is active.

---

## Running Tests with Rust

```bash
# Run the full test suite (Rust active)
NSCK_USE_RUST=1 python -m pytest nsck/tests/ --tb=short -q

# Run only Rust-specific tests
python -m pytest nsck/tests/unit/rust/ nsck/tests/unit/vsa/test_hypervec_parity.py -v

# Run V4 full-system tests
NSCK_USE_RUST=1 python -m pytest nsck/tests/integration/test_v4_full_system.py -v

# Run V4 benchmarks
python nsck/eval/vsa_capability_benchmark.py
python -m pytest nsck/tests/benchmarks/test_v17_benchmarks.py -v --benchmark-disable
```

Expected results with Rust active (February 2026):
- **1 607 tests** pass (1521 main + 86 Rust-specific)
- 2 stochastic image/audio similarity tests may flap (known, non-blocking)
- 3 xfailed (known VSA/NLU limitations)

---

## Cargo Check (without Python linking)

`cargo test` fails due to PyO3 linking requirements. Use instead:

```bash
cd nsck/rust_vsa && cargo check --lib
cd nsck/rust_snn && cargo check --lib
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `maturin: command not found` | `pip install maturin` |
| `linker 'cc' not found` | `apt install build-essential` |
| `unzip: command not found` | `apt install unzip` |
| `.so not found at runtime` | Re-run build steps; copy `.so` to `nsck/` |
| Python fallback active unexpectedly | Ensure `nsck/hypervec_rs.so` exists; check `sys.path` includes `nsck/` |
| `cargo test` fails with PyO3 link error | Expected — use `cargo check --lib` instead |
| Build takes long on first run | Rust compiles all deps from scratch; subsequent builds are incremental |

---

## See Also

- `nsck/docs/RUST_API_REFERENCE.md` — full Rust extension API reference
- `nsck/docs/ARCHITECTURE.md` — system architecture overview
- `nsck/docs/V4_RELEASE_REPORT.md` — V4 implementation report

---

*Updated for NSCK V4, February 2026. Verified on rustc 1.75+ / Python 3.9+ / x86-64 Linux.*
