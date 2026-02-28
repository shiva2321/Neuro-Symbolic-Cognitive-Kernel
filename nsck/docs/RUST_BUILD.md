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

Optional:
- **CUDA toolkit** (for GPU SNN): install via NVIDIA or conda-forge.

---

## Quick Build

```bash
# From repo root
cd nsck/rust/hypervec_rs
maturin develop --release

cd ../snn_rs
maturin develop --release
```

After a successful build the `.so` files appear in:
```
nsck/python/core/vsa/hypervec_rs.so
nsck/python/core/training/snn_rs.so
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
cd nsck/rust/hypervec_rs
maturin develop --release
```

`hypervec_rs` provides:
- `HyperVector` — VSA vector with XOR bind, bundle, cosine similarity
- `HyperVectorRegistry` — fast nearest-neighbour lookup
- Parallel batch operations (Rayon)

### 4. Build `snn_rs`

```bash
cd ../snn_rs
maturin develop --release
```

`snn_rs` provides:
- `LIFNeuronLayer` — Leaky Integrate-and-Fire layer
- `STDPLearner` — spike-timing-dependent plasticity
- `ConceptMapper` — SNN → HV concept mapping

### 5. Verify

```python
from python.core.vsa.hypervec_shim import HyperVec
import python.core.vsa.hypervec_shim as shim
print(shim._BACKEND)   # should print "rust" if build succeeded
```

---

## Build in CI (GitHub Actions)

```yaml
- name: Install Rust
  uses: dtolnay/rust-toolchain@stable

- name: Build Rust extensions
  run: |
    pip install maturin
    cd nsck/rust/hypervec_rs && maturin develop --release
    cd ../snn_rs && maturin develop --release
```

---

## Performance Impact

| Operation | Python fallback | Rust extension | Speedup |
|-----------|----------------|----------------|---------|
| VSA bind (XOR) | ~800 K ops/s | ~4.1 M ops/s | **5×** |
| VSA similarity | ~125 K ops/s | ~3.8 M ops/s | **30×** |
| VSA bundle | ~21 K ops/s | ~1.4 M ops/s | **65×** |
| Memory query (1K) | ~15 ms | ~0.6 ms | **25×** |

> **Note**: V17 enrichment modules (`CausalEnricher`, `PerceptualEnricher`,
> `SemanticEnricher`, `GlassBoxTracer`, `CrossModalEnricher`) are pure Python
> and do not require Rust. They achieve > 1 M ops/s on a modern CPU without
> Rust.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `maturin: command not found` | `pip install maturin` |
| `linker 'cc' not found` | Install build tools: `apt install build-essential` |
| `CUDA not found` | Install CUDA toolkit or disable GPU features |
| `.so not found at runtime` | Re-run `maturin develop --release` in the correct directory |
| Python fallback active unexpectedly | Check `shim._BACKEND` == `"python"` — rebuild Rust |

---

## See Also

- `nsck/docs/RUST_API_REFERENCE.md` — full Rust extension API reference
- `nsck/docs/ARCHITECTURE.md` — system architecture overview

---

*Updated for NSCK V17, April 2026.*
