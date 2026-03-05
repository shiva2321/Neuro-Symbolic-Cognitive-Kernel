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

## V18 Spreading Activation — Before/After

V18 changes the spreading activation hot path to use `SemanticMemoryConcurrent.parallel_spread_activation()` (all steps inside Rust, Rayon parallel) instead of rebuilding the full Python edge list on every call.

| Scale (nodes) | Python (ms) | Rust V18 (ms) | Speedup |
|---|---|---|---|
| 100 | ~0.1 | ~0.05 | ~2× |
| 1 000 | ~2.5 | ~0.3 | ~8× |
| 10 000 | ~23.6 | ~1.5 | ~16× |
| 50 000 | ~120 | ~8 | ~15× |

*Expected figures — actual numbers depend on hardware. Run `eval/bench_spread_activation.py` for live results.*

The key improvements:
1. **Write-time mirror (Change 1):** `add_concept` / `add_relation` push to the Rust DashMap immediately — no more O(N) scan of nodes-not-yet-synced on every `spread_activation` call.
2. **Rayon parallel all-steps (Change 2):** `parallel_spread_activation` runs all spreading steps inside Rust with Rayon parallelism, returning once. Previously required crossing the FFI boundary 3× per call with the full edge list.
3. **Weighted edges (Change 3):** `add_relation_weighted` stores typed relation weights in the DashMap `HashMap<String, f32>`, so `parallel_spread_activation` propagates `act * w * decay` instead of `act * decay / num_neighbors`.

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

# V18 spread activation Rust vs Python
python nsck/eval/bench_spread_activation.py
```
