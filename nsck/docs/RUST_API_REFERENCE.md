# Rust Backend API Reference — NSCK V13

Rust extensions for NSCK, built with [PyO3](https://pyo3.rs/) and
[maturin](https://github.com/PyO3/maturin). Two compiled shared-object modules
provide **5–85× speedup** over pure-Python equivalents on hot-path VSA and SNN
operations.

| Extension | Size | Contents |
|-----------|------|----------|
| `hypervec_rs.so` | 4.3 MB | VSA operations, memory backends, parallel utilities |
| `snn_rs.so` | 1.1 MB | SNN neuron layers, STDP learning, concept mapping |

### Quick performance snapshot

| Operation | Rust ops/s | Python ops/s | Speedup |
|-----------|-----------|-------------|---------|
| VSA bind (XOR) | 4,109,142 | 803,079 | **5.1×** |
| VSA similarity | 3,793,104 | 124,658 | **30.4×** |
| VSA bundle | 1,354,342 | 20,685 | **65.5×** |
| Memory query (1 K concepts) | 0.56 ms | — | — |
| SNN LIF step | 0.017 ms | — | — |

---

## Table of Contents

1. [Build Instructions](#build-instructions)
2. [hypervec\_rs API](#hypervec_rs-api)
   - [HyperVector](#hypervector)
   - [HyperVectorRegistry](#hypervectorregistry)
   - [SemanticMemoryConcurrent](#semanticmemoryconcurrent)
   - [EpisodicMemoryConcurrent](#episodicmemoryconcurrent)
   - [Episode](#episode)
   - [CognitiveWorkerPool](#cognitiveworkerpool)
   - [PersistentStorage](#persistentstorage)
   - [ActivationAccumulator](#activationaccumulator)
   - [AsyncCognitiveRuntime](#asynccognitiveruntime)
3. [Parallel & Utility Functions](#parallel--utility-functions)
4. [snn\_rs API](#snn_rs-api)
   - [SnnCore](#snncore)
   - [LIFLayer](#liflayer)
   - [StdpEngine](#stdpengine)
   - [HebbianMatrix](#hebbianmatrix)
   - [ConceptMapper](#conceptmapper)
   - [RateCoder](#ratecoder)
5. [Python Shim Layer](#python-shim-layer)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Thread Safety](#thread-safety)
8. [Troubleshooting](#troubleshooting)

---

## Build Instructions

```bash
pip install maturin

# Build hypervec_rs
cd nsck/rust_vsa && maturin build --release

# Build snn_rs
cd nsck/rust_snn && maturin build --release

# Extract .so files from wheels
unzip -o nsck/rust_vsa/target/wheels/*.whl "hypervec_rs/*" -d /tmp/rv
cp /tmp/rv/hypervec_rs/*.so nsck/hypervec_rs.so

unzip -o nsck/rust_snn/target/wheels/*.whl "snn_rs/*" -d /tmp/rs
cp /tmp/rs/snn_rs/*.so nsck/snn_rs.so
```

For iterative development without the wheel extraction step:

```bash
cd nsck/rust_vsa && maturin develop --release
cd nsck/rust_snn && maturin develop --release
```

---

## hypervec\_rs API

All types below live in the `hypervec_rs` compiled module and are re-exported
through `python.core.vsa.hypervec_shim`.

### HyperVector

10,240-bit binary vector stored as **160 × u64** blocks.

#### Constructor

```python
HyperVector(seed: Optional[int] = None)
```

- `seed` — deterministic PRNG seed. Omit for a random vector.

```python
HyperVector.zero()           # all-zero vector
HyperVector.from_u64_words(words: List[int])  # rebuild from raw blocks
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `xor` | `(other) → HyperVector` | XOR binding (VSA ⊗) |
| `bundle` | `(other) → HyperVector` | Majority-vote bundling (VSA ⊕) |
| `weighted_bundle` | `(other, weight, seed=None) → HyperVector` | Weighted superposition |
| `permute` | `(shift) → HyperVector` | Circular bit rotation (temporal encoding) |
| `permute_inverse` | `(shift) → HyperVector` | Inverse rotation |
| `similarity` | `(other) → float` | Hamming similarity in [0, 1] |
| `negate` | `() → HyperVector` | VSA negation (XOR with fixed role vector) |
| `lsh_hash` | `(seed, n_bits) → int` | Locality-sensitive hash signature |

Pickle support: `__getstate__()` / `__setstate__(state)`.

#### Example

```python
from python.core.vsa import hypervec_shim as hv

a = hv.HyperVector(seed=1)
b = hv.HyperVector(seed=2)

bound   = a.xor(b)                      # bind
bundled = a.bundle(b)                    # bundle
weighted = a.weighted_bundle(b, 0.7)     # 70 % a, 30 % b
t1      = a.permute(1)                   # temporal shift
sim     = a.similarity(b)               # ≈ 0.50 for random pair
neg     = a.negate()                     # negation
```

---

### HyperVectorRegistry

Thread-safe named-vector store backed by `DashMap`.

```python
HyperVectorRegistry()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `(name: str, hv)` | Insert / overwrite |
| `get` | `(name: str) → Optional[HyperVector]` | Lookup by name |
| `contains` | `(name: str) → bool` | Membership test |
| `remove` | `(name: str) → bool` | Delete entry |
| `clear` | `()` | Remove all entries |

---

### SemanticMemoryConcurrent

Thread-safe semantic memory with concept vectors **and** a directed relation
graph.

```python
SemanticMemoryConcurrent()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_concept` | `(name, hv)` | Store concept vector |
| `get_concept` | `(name) → Optional[HV]` | Retrieve vector |
| `add_relation` | `(source, target)` | Add directed edge |
| `get_neighbors` | `(concept) → List[str]` | Outgoing edges |
| `get_incoming_neighbors` | `(concept) → List[str]` | Incoming edges |
| `parallel_semantic_search` | `(query_hv, k) → List[(str, float)]` | Top-k similarity search |
| `parallel_spread_activation` | `(sources, depth, decay) → Dict[str, float]` | Spreading activation |
| `hybrid_search` | `(query_hv, start_concepts, k) → List[(str, float)]` | VSA + graph search |
| `get_activated_concepts` | `(threshold) → List[(str, float)]` | Above-threshold concepts |
| `concept_count` / `relation_count` | `() → int` | Counts |
| `get_all_concepts` | `() → List[str]` | All concept names |
| `get_stats` | `() → Dict[str, int]` | Statistics dict |
| `clear` | `()` | Wipe everything |

#### Example

```python
sem = hv.SemanticMemoryConcurrent()
sem.add_concept("dog", hv.HyperVector(seed=1))
sem.add_concept("cat", hv.HyperVector(seed=2))
sem.add_relation("dog", "animal")

results = sem.parallel_semantic_search(hv.HyperVector(seed=1), k=3)
# [("dog", 1.0), ("cat", 0.51)]
```

---

### EpisodicMemoryConcurrent

Thread-safe episodic memory with a fixed-capacity hot tier.

```python
EpisodicMemoryConcurrent(max_hot_size: int = 10000)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_episode` | `(ep) → Optional[Episode]` | Store; returns evicted episode if full |
| `batch_add_episodes` | `(eps) → List[Episode]` | Bulk insert |
| `parallel_knn_search` | `(query_hv, k, task_tag=None) → List[Episode]` | KNN recall |
| `batch_knn_search` | `(queries, k, task_tag=None) → List[List[Episode]]` | Batch KNN |
| `get_recent_episodes` | `(task_tag, n) → List[Episode]` | Last *n* for a task |
| `get_all_episodes` | `() → List[Episode]` | Full dump |
| `get_high_impact_episodes` | `(threshold, limit) → List[Episode]` | Filter by impact score |
| `search_by_task` | `(task_tag) → List[Episode]` | All episodes for task |
| `search_by_reward` | `(min_reward, max_reward) → List[Episode]` | Reward-range filter |
| `size` | `() → int` | Hot-tier count |
| `get_stats` | `() → Dict[str, Any]` | Statistics |
| `clear` | `()` | Remove all |

---

### Episode

Immutable episode record.

```python
Episode(timestamp, task_tag, situation_hv, action, outcome, reward, impact_score=0.0)
```

Read-only properties: `timestamp`, `task_tag`, `situation_hv`, `action`,
`outcome`, `reward`, `impact_score`.

---

### CognitiveWorkerPool

Rayon-backed thread pool that shares semantic and episodic memory across
workers.

```python
CognitiveWorkerPool(semantic_memory, episodic_memory, num_workers=4)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `submit_task` | `(task_type: str, data: dict)` | Enqueue work item |
| `get_result` | `(timeout: float) → Optional[dict]` | Blocking dequeue |
| `shutdown` | `()` | Drain queue and join workers |

### PersistentStorage

SQLite-backed disk persistence with batched writes.

```python
PersistentStorage(db_path: str, batch_size: int = 100)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `store_hypervector` | `(name, hv)` | Write HV to DB |
| `load_hypervector` | `(name) → Optional[HV]` | Read HV |
| `store_episode` | `(ep)` | Write episode |
| `load_episodes` | `(task_tag, limit) → List[Episode]` | Read episodes |
| `commit` | `()` | Force-flush pending writes |
| `close` | `()` | Close DB connection |

### ActivationAccumulator

Thread-safe running accumulator for spreading-activation values.

```python
ActivationAccumulator()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_activation` | `(concept, value)` | Additive update |
| `get_activation` | `(concept) → float` | Current value |
| `decay` | `(factor)` | Multiply all values |
| `top_k` | `(k) → List[(str, float)]` | Highest-activation concepts |
| `clear` | `()` | Reset to zero |

### AsyncCognitiveRuntime

Tokio-backed async runtime wrapping memory searches.

```python
AsyncCognitiveRuntime(semantic_memory, episodic_memory)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `semantic_search_async` | `(query, k) → List[(str, float)]` | Non-blocking semantic search |
| `episodic_recall_async` | `(query, k, task=None) → List[Episode]` | Non-blocking episodic recall |

---

## Parallel & Utility Functions

Free functions exported from `hypervec_rs` (available via `hypervec_shim`).

```python
parallel_bundle(vectors: List[HyperVector]) -> HyperVector
```
Majority-vote bundle across an arbitrary number of vectors (rayon parallel).

```python
parallel_similarity_search(query, candidates, k) -> List[(int, float)]
```
Top-k similarity search over a candidate list. Returns `(index, similarity)`.

```python
batch_parallel_similarity_search(queries, candidates, k) -> List[List[(int, float)]]
```
One search per query, all parallelised.

```python
batch_similarity_matrix(vectors: List[HyperVector]) -> List[float]
```
N×N pairwise similarity matrix returned as a flat row-major list.

```python
run_semantic_search_async(memory, query_hv, k) -> List[(str, float)]
```
Async semantic search wrapper (runs on Tokio).

```python
weber_fechner_compress(values: List[float], base: float = math.e) -> List[float]
```
Logarithmic (Weber–Fechner) compression of a value list.

---

## snn\_rs API

All types below live in the `snn_rs` compiled module and are re-exported
through `python.core.perception.snn_shim`.

### SnnCore

Full-cycle spiking neural network simulator (input → LIF → STDP → output).

```python
SnnCore(input_dim, snn_size, tau, v_rest, v_reset, v_thresh,
        refractory_period, dt, stdp_lr, tau_stdp, a_plus, a_minus,
        seed=0, weber_fechner=False)
```

| Method / Property | Signature | Description |
|-------------------|-----------|-------------|
| `simulate` | `(sensory_input, n_steps, learn) → List[List[float]]` | Run perceive loop |
| `normalize_weights` | `()` | L2-normalise weight rows |
| `get_weights` / `set_weights` | `() → List[float]` / `(w)` | Flat weight matrix access |
| `get_stats` | `() → List[(str, float)]` | Runtime statistics |
| `weber_fechner` | getter/setter | Toggle log compression |
| `input_dim`, `snn_size` | getters | Dimensions |
| `stdp_updates`, `weight_updates` | getters | Counters |

### LIFLayer

Leaky Integrate-and-Fire neuron layer (rayon-parallelised per-neuron updates).

```python
LIFLayer(n_neurons, tau, v_rest, v_reset, v_thresh, refractory_period, dt)
```

| Method / Property | Signature | Description |
|-------------------|-----------|-------------|
| `step` | `(input_current: List[float]) → List[float]` | One timestep; returns spike vector |
| `reset` | `()` | Reset membrane potentials |
| `get_spike_train` | `() → List[List[float]]` | Full spike history |
| `n_neurons`, `dt` | getters | Layer parameters |

### StdpEngine

Spike-Timing-Dependent Plasticity weight updater.

```python
StdpEngine(input_dim, snn_size, stdp_lr, tau_stdp, a_plus, a_minus)
```

| Method / Property | Signature | Description |
|-------------------|-----------|-------------|
| `apply` | `(input_spikes, output_spikes) → List[float]` | Compute weight deltas |
| `advance_time` | `(dt)` | Increment internal clock |
| `reset` | `()` | Clear spike times |
| `updates`, `current_time` | getters | Counters / clock |

### HebbianMatrix

Oja's-rule Hebbian weight matrix.

```python
HebbianMatrix(input_dim: int, output_dim: int, lr: float)
```

### ConceptMapper

Jaccard-similarity–based pattern → concept recogniser.

```python
ConceptMapper()
```

### RateCoder

Converts spike trains to firing rates and active-neuron masks.

```python
RateCoder()
```

`HebbianMatrix`, `ConceptMapper`, and `RateCoder` are registered via the
`hebbian` and `concept` sub-modules of `snn_rs`.

---

## Python Shim Layer

Three thin shim modules try importing the compiled Rust extension and fall back
to pure-Python implementations when the `.so` is unavailable.

| Shim file | Rust module | Fallback | Exports |
|-----------|-------------|----------|---------|
| `python/core/vsa/hypervec_shim.py` | `hypervec_rs` | `HyperVectorPy` | `HyperVector`, memory types, parallel functions |
| `python/core/perception/snn_shim.py` | `snn_rs` | `PythonSnnCore` | `SnnCore`, `LIFLayer`, `StdpEngine`, `HebbianMatrix`, `ConceptMapper`, `RateCoder` |
| `python/core/vsa/rust_concurrent_shim.py` | `hypervec_rs` | Python wrappers | `SemanticMemoryConcurrent`, `EpisodicMemoryConcurrent`, `CognitiveWorkerPool`, `batch_similarity_matrix` |

Check which backend is active:

```python
from python.core.vsa import hypervec_shim as hv
print(hv.__backend__)          # "Rust" or "Python"
print(hv.get_backend_info())   # detailed status dict

from python.core.perception import snn_shim
print(snn_shim.USE_RUST)       # True / False
```

The shim interface is identical regardless of backend, so application code
never needs conditional imports.

---

## Performance Benchmarks

Measured on x86-64, single-socket, with Rust compiled in `--release` mode.

| Operation | Rust | Python | Speedup |
|-----------|------|--------|---------|
| VSA bind (XOR) | 4,109,142 ops/s | 803,079 ops/s | **5.1×** |
| VSA similarity | 3,793,104 ops/s | 124,658 ops/s | **30.4×** |
| VSA bundle | 1,354,342 ops/s | 20,685 ops/s | **65.5×** |
| Memory query (1 K concepts) | 0.56 ms | — | — |
| SNN LIF step | 0.017 ms | — | — |

Parallel functions (`parallel_bundle`, `batch_parallel_similarity_search`,
etc.) scale near-linearly up to the number of physical cores.

---

## Thread Safety

| Type | Safe | Strategy |
|------|------|----------|
| `HyperVector` | ✅ Immutable | Copy-on-write; no locking needed |
| `HyperVectorRegistry` | ✅ | DashMap (lock-free concurrent map) |
| `SemanticMemoryConcurrent` | ✅ | DashMap for concepts; RwLock for graph |
| `EpisodicMemoryConcurrent` | ✅ | RwLock (read-biased) |
| `ActivationAccumulator` | ✅ | DashMap |
| `CognitiveWorkerPool` | ✅ | MPSC channels between Python and Rayon |
| `AsyncCognitiveRuntime` | ✅ | Tokio multi-thread runtime |
| `PersistentStorage` | ⚠️ Single-writer | SQLite serialised writes |

All `*Concurrent` types can be shared across Python threads (and across Rust
worker threads internally) without additional locking.

---

## Troubleshooting

### Rust extension not found

```
ImportError: cannot import name 'hypervec_rs'
```

The `.so` was not placed on the Python path. Either rebuild with
`maturin develop --release` (installs into the active venv) or copy the `.so`
manually:

```bash
cp /tmp/rv/hypervec_rs/*.so nsck/hypervec_rs.so
```

### Wrong Python version / ABI mismatch

```
ImportError: ... undefined symbol: _Py_Dealloc
```

Rebuild with the same Python that will run the code:

```bash
python -m pip install maturin
cd nsck/rust_vsa && maturin build --release --interpreter $(which python)
```

### Debugging Rust panics

```bash
RUST_BACKTRACE=1 python my_script.py
```

### TypeError from Rust

PyO3 validates argument types at the boundary. A helpful `TypeError` is raised
when the wrong type is passed:

```python
sem.add_concept(123, "not_a_hv")
# TypeError: argument 'name': 'int' object cannot be converted to 'PyString'
```

### Performance is no better than Python

Make sure the Rust extension was compiled in **release** mode
(`--release` flag). Debug builds are 10–50× slower than release.

---

**Last Updated:** June 2025
**NSCK Version:** V13
**Rust Backend Version:** 13.0.0
