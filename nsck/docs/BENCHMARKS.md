# NSCK Benchmarks — V5

Benchmarks are shown for the **Python fallback** (active when Rust crates are not compiled) and the **Rust backend** (when compiled via maturin). The Python fallback is the default in most CI environments.

## Python Fallback (default — no compiled Rust)

| Operation | Python Fallback | Notes |
|---|---|---|
| HyperVector XOR bind (10240-bit) | ~2.1 µs | Pure Python / NumPy |
| HyperVector similarity | ~10.7 µs | Pure Python / NumPy |
| HyperVector bundle (pair) | ~64.5 µs | Pure Python / NumPy |

## Rust Backend Speedups (when compiled)

When Rust crates (`rust_vsa`, `rust_snn`, `rust_societal`) are compiled via maturin, operations accelerate significantly. These are target estimates — actual numbers depend on hardware.

| Operation | Python Fallback | Rust (estimated) | Speedup |
|---|---|---|---|
| HyperVector XOR bind (10240-bit) | ~2.1 µs | ~0.1–0.4 µs | **5–20×** |
| HyperVector similarity | ~10.7 µs | ~0.25–1 µs | **10–40×** |
| HyperVector bundle (pair) | ~64.5 µs | ~0.45–1.5 µs | **40–85×** |
| bundle_hvs (N=10, D=10240) | — | ~0.18 ms | — |
| lsh_bucket (D=10240, n_bits=16) | — | ~0.04 ms | — |
| spreading_activation_step (1K nodes, 5K edges) | ~12 ms | ~0.8 ms | **~15×** |
| parallel_semantic_search (10K concepts, k=10) | ~45 ms | ~1.2 ms | **~37×** |
| SNN step (1024 neurons) | ~4 ms | ~0.8 ms | **~5×** |

## Rust Backend Operations

### bundle_hvs

Correct N-vector majority-vote bundling (rust_vsa).

```python
import hypervec_rs
result = hypervec_rs.bundle_hvs([a.bits, a.bits, a.bits, b.bits])
# majority vote: 3 copies of a vs 1 of b → result ≈ a
```

**Complexity:** O(N · D), N = number of vectors, D = 10240 bits = 160 u64 blocks

### lsh_bucket

LSH bucket computation for ProceduralMemory O(1) lookup (rust_vsa).

```python
key = hypervec_rs.lsh_bucket(hv.bits, n_bits=16, seed=0xDEAD)
```

### spreading_activation_step

One graph spreading activation step — hot path in SemanticMemory (rust_vsa).

```python
new_activation = hypervec_rs.spreading_activation_step(
    activation={"concept_a": 1.0},
    edges=[("concept_a", "concept_b", 0.8)],
    decay=0.7,
    max_frontier=200
)
```

## V5 Societal Operations (rust_societal)

The `rust_societal` crate accelerates societal HV operations — spreading activation across hierarchical city-model layers, Hodge Laplacian steps, and percolation monitoring. Run `eval/bench_spread_activation.py` for live measurements.

## Rust vs Python Backend Selection

```python
import os
os.environ["NSCK_USE_RUST"] = "1"

# Verify Rust is loaded:
import hypervec_rs
print("Rust OK:", hypervec_rs.HyperVector(1).bits[:5])
```

The shim (`python/core/vsa/hypervec_shim.py`) auto-detects compiled extensions at import time. If absent, the pure-Python/NumPy fallback is used transparently with a log warning.

## Running Your Own Benchmarks

```bash
# Full benchmark suite
python nsck/tests/benchmarks/full_architecture_benchmark.py

# Spread activation benchmark (Rust vs Python)
python nsck/eval/bench_spread_activation.py
```
