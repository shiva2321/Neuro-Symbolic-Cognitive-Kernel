# NSCK Benchmarks — V4

> All benchmarks run with `NSCK_USE_RUST=1` (default as of V4) on a commodity CPU (Intel Core i7, 8 cores, 16 GB RAM).

## Summary

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| HyperVector XOR bind (10240-bit) | ~120 µs | ~6 µs | **20×** |
| HyperVector similarity | ~18 µs | ~0.25 µs | **72×** |
| HyperVector bundle (pair) | ~30 µs | ~0.45 µs | **67×** |
| bundle_hvs (N=10, D=10240) | ~3.2 ms | ~0.18 ms | **18×** |
| lsh_bucket (D=10240, n_bits=16) | ~0.9 ms | ~0.04 ms | **22×** |
| spreading_activation_step (1K nodes, 5K edges) | ~12 ms | ~0.8 ms | **15×** |
| parallel_semantic_search (10K concepts, k=10) | ~45 ms | ~1.2 ms | **37×** |
| SNN step (1024 neurons) | ~4 ms | ~0.8 ms | **5×** |

## V4 New Operations (Rust Default-On)

### bundle_hvs

Correct N-vector majority-vote bundling. Replaces the previous pair-wise OR approximation.

```python
import hypervec_rs
a = hypervec_rs.HyperVector(1)
b = hypervec_rs.HyperVector(2)
result = hypervec_rs.bundle_hvs([a.bits, a.bits, a.bits, b.bits])
# result is close to a (majority vote: 3 copies of a vs 1 of b)
```

**Complexity:** O(N · D) where N = number of vectors, D = dimension (10240 bits = 160 u64 blocks)

### lsh_bucket

LSH bucket computation for ProceduralMemory O(1) lookup.

```python
key = hypervec_rs.lsh_bucket(hv.bits, n_bits=16, seed=0xDEAD)
```

**Complexity:** O(n_bits · D/64)

### spreading_activation_step

One graph spreading activation step — hot path in SemanticMemory.

```python
new_activation = hypervec_rs.spreading_activation_step(
    activation={"concept_a": 1.0},
    edges=[("concept_a", "concept_b", 0.8)],
    decay=0.7,
    max_frontier=200
)
```

**Complexity:** O(E) where E = number of edges in frontier

## Rust vs Python Backend Selection

```python
import os
os.environ["NSCK_USE_RUST"] = "1"  # default in V4

# Verify Rust is loaded:
import hypervec_rs
print("Rust OK:", hypervec_rs.HyperVector(1).bits[:5])
```

## Running Your Own Benchmarks

```bash
# Full benchmark suite
NSCK_USE_RUST=1 python nsck/tests/benchmarks/full_architecture_benchmark.py

# V4-specific benchmarks
NSCK_USE_RUST=1 python -m pytest nsck/tests/benchmarks/test_v17_benchmarks.py -v
```
