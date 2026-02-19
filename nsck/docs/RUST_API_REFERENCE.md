# Rust Backend API Reference

Complete reference for all Rust-accelerated modules in NSCK.

**Performance**: 10-100x faster than pure Python for memory operations, similarity search, and bundling.

---

## Table of Contents

- [Core Types](#core-types)
  - [HyperVector](#hypervector)
  - [Episode](#episode)
- [Memory Systems](#memory-systems)
  - [SemanticMemoryConcurrent](#semanticmemoryconcurrent)
  - [EpisodicMemoryConcurrent](#episodicmemoryconcurrent)
- [Cognitive Infrastructure](#cognitive-infrastructure)
  - [HyperVectorRegistry](#hypervectorregistry)
  - [ActivationAccumulator](#activationaccumulator)
  - [CognitiveWorkerPool](#cognitiveworkerpool)
- [Storage](#storage)
  - [PersistentStorage](#persistentstorage)
- [Async Runtime](#async-runtime)
  - [AsyncCognitiveRuntime](#asynccognitiveruntime)
- [Parallel Functions](#parallel-functions)
- [Type Mappings](#type-mappings)

---

## Core Types

### HyperVector

**10,240-dimensional binary hypervector** (stored as 160 x u64 blocks).

#### Constructor

```python
HyperVector(seed: Optional[int] = None) -> HyperVector
```

**Args**:
- `seed`: Random seed for reproducible vector generation (optional)

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

# Random vector
hv1 = hrs.HyperVector()

# Deterministic vector
hv2 = hrs.HyperVector(seed=42)

# Zero vector
hv_zero = hrs.HyperVector.zero()
```

#### Methods

| Method | Signature | Description | Performance |
|--------|-----------|-------------|-------------|
| `xor` | `xor(other: HV) -> HV` | XOR binding | ~50ns |
| `bundle` | `bundle(other: HV) -> HV` | Majority bundling | ~200ns |
| `weighted_bundle` | `weighted_bundle(other: HV, weight: float, seed: Optional[int]) -> HV` | Weighted bundling | ~500ns |
| `permute` | `permute(shift: int) -> HV` | Circular rotation (temporal encoding) | ~200ns |
| `permute_inverse` | `permute_inverse(shift: int) -> HV` | Inverse permutation | ~200ns |
| `similarity` | `similarity(other: HV) -> float` | Hamming similarity [0,1] | ~100ns |
| `lsh_hash` | `lsh_hash(seed: int, n_bits: int) -> int` | LSH signature | ~5µs |

**Example**:
```python
a = hrs.HyperVector(seed=1)
b = hrs.HyperVector(seed=2)

# Binding: A ⊗ B
bound = a.xor(b)

# Bundling: A + B
bundled = a.bundle(b)

# Weighted: 0.7A + 0.3B
weighted = a.weighted_bundle(b, weight=0.7)

# Temporal encoding
t0 = a.permute(0)  # time step 0
t1 = a.permute(1)  # time step 1

# Similarity
sim = a.similarity(b)  # ~0.5 (random vectors)
```

---

### Episode

**Immutable episode data structure** for episodic memory.

#### Constructor

```python
Episode(
    timestamp: float,
    task_tag: str,
    situation_hv: HyperVector,
    action: str,
    outcome: str,
    reward: float,
    impact_score: float = 0.0
) -> Episode
```

**Args**:
- `timestamp`: Unix timestamp (seconds since epoch)
- `task_tag`: Task identifier string
- `situation_hv`: Situation encoded as hypervector
- `action`: Action taken (string description)
- `outcome`: Outcome description
- `reward`: Reward value (float, typically [-1, 1])
- `impact_score`: Importance score (default: 0.0)

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs
import time

situation = hrs.HyperVector(seed=100)

episode = hrs.Episode(
    timestamp=time.time(),
    task_tag="robot_navigation",
    situation_hv=situation,
    action="move_forward",
    outcome="success",
    reward=1.0,
    impact_score=0.8
)

print(episode.task_tag)     # "robot_navigation"
print(episode.reward)       # 1.0
print(episode.impact_score) # 0.8
```

---

## Memory Systems

### SemanticMemoryConcurrent

**Thread-safe semantic memory** with concurrent concept storage and graph relations.

#### Constructor

```python
SemanticMemoryConcurrent() -> SemanticMemoryConcurrent
```

**No arguments** - creates empty semantic memory.

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

sem = hrs.SemanticMemoryConcurrent()
```

#### Methods

| Method | Signature | Description | Thread-Safe |
|--------|-----------|-------------|-------------|
| `add_concept` | `add_concept(name: str, hv: HV)` | Store concept HV | ✅ Yes |
| `get_concept` | `get_concept(name: str) -> Optional[HV]` | Retrieve concept HV | ✅ Yes |
| `add_relation` | `add_relation(source: str, target: str)` | Add directed edge | ✅ Yes |
| `get_neighbors` | `get_neighbors(concept: str) -> List[str]` | Get outgoing edges | ✅ Yes |
| `get_incoming_neighbors` | `get_incoming_neighbors(concept: str) -> List[str]` | Get incoming edges | ✅ Yes |
| `parallel_semantic_search` | `parallel_semantic_search(query_hv: HV, k: int) -> List[Tuple[str, float]]` | Top-k similarity search | ✅ Yes |
| `parallel_spread_activation` | `parallel_spread_activation(sources: List[str], depth: int, decay: float) -> Dict[str, float]` | Spreading activation | ✅ Yes |
| `hybrid_search` | `hybrid_search(query_hv: HV, start_concepts: List[str], k: int) -> List[Tuple[str, float]]` | Combined VSA + graph search | ✅ Yes |
| `get_activated_concepts` | `get_activated_concepts(threshold: float) -> List[Tuple[str, float]]` | Get concepts above threshold | ✅ Yes |
| `concept_count` | `concept_count() -> int` | Number of concepts | ✅ Yes |
| `relation_count` | `relation_count() -> int` | Number of relations | ✅ Yes |
| `get_all_concepts` | `get_all_concepts() -> List[str]` | All concept names | ✅ Yes |
| `get_stats` | `get_stats() -> Dict[str, int]` | Memory statistics | ✅ Yes |
| `clear` | `clear()` | Remove all data | ✅ Yes |

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

sem = hrs.SemanticMemoryConcurrent()

# Add concepts
dog = hrs.HyperVector(seed=1)
cat = hrs.HyperVector(seed=2)
animal = hrs.HyperVector(seed=3)

sem.add_concept("dog", dog)
sem.add_concept("cat", cat)
sem.add_concept("animal", animal)

# Add relations
sem.add_relation("dog", "animal")
sem.add_relation("cat", "animal")

# Query (NOTE: method name is parallel_semantic_search, not search)
results = sem.parallel_semantic_search(dog, k=3)
# [(name, similarity), ...]
print(results)  # [("dog", 1.0), ("cat", 0.51), ("animal", 0.49)]

# Graph traversal
neighbors = sem.get_neighbors("dog")
print(neighbors)  # ["animal"]

# Spreading activation
activation = sem.parallel_spread_activation(["dog"], depth=2, decay=0.9)
print(activation)  # {"dog": 1.0, "animal": 0.9, ...}

# Hybrid search (VSA + graph)
hybrid_results = sem.hybrid_search(dog, start_concepts=["dog"], k=5)

# Statistics
print(f"Concepts: {sem.concept_count()}, Relations: {sem.relation_count()}")
```

**Performance**: 
- `add_concept`: ~1µs (lock-free write)
- `parallel_semantic_search`: ~100µs for 1000 concepts (parallel)

---

### EpisodicMemoryConcurrent

**Thread-safe episodic memory** with hot-tier caching (recent episodes).

#### Constructor

```python
EpisodicMemoryConcurrent(max_hot_size: int = 10000) -> EpisodicMemoryConcurrent
```

**Args**:
- `max_hot_size`: Maximum number of recent episodes in hot tier (default: 10,000)

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

# Store last 5000 episodes in hot tier
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=5000)
```

#### Methods

| Method | Signature | Description | Thread-Safe |
|--------|-----------|-------------|-------------|
| `add_episode` | `add_episode(ep: Episode) -> Optional[Episode]` | Add episode, returns evicted if full | ✅ Yes |
| `batch_add_episodes` | `batch_add_episodes(eps: List[Episode]) -> List[Episode]` | Add multiple episodes | ✅ Yes |
| `parallel_knn_search` | `parallel_knn_search(query_hv: HV, k: int, task_tag: Optional[str]) -> List[Episode]` | KNN search for similar episodes | ✅ Yes |
| `batch_knn_search` | `batch_knn_search(queries: List[HV], k: int, task_tag: Optional[str]) -> List[List[Episode]]` | Batch KNN search | ✅ Yes |
| `get_recent_episodes` | `get_recent_episodes(task_tag: str, n: int) -> List[Episode]` | Get last N episodes for task | ✅ Yes |
| `get_all_episodes` | `get_all_episodes() -> List[Episode]` | Get all episodes | ✅ Yes |
| `get_high_impact_episodes` | `get_high_impact_episodes(threshold: float, limit: int) -> List[Episode]` | Get episodes by impact score | ✅ Yes |
| `search_by_task` | `search_by_task(task_tag: str) -> List[Episode]` | Get all episodes for task | ✅ Yes |
| `search_by_reward` | `search_by_reward(min_reward: float, max_reward: float) -> List[Episode]` | Filter by reward range | ✅ Yes |
| `size` | `size() -> int` | Number of episodes in hot tier | ✅ Yes |
| `get_stats` | `get_stats() -> Dict[str, Any]` | Memory statistics | ✅ Yes |
| `clear` | `clear()` | Remove all episodes | ✅ Yes |

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs
import time

epi = hrs.EpisodicMemoryConcurrent(max_hot_size=1000)

# Add episodes
for i in range(10):
    situation = hrs.HyperVector(seed=i)
    ep = hrs.Episode(
        timestamp=time.time() + i,
        task_tag="task_a",
        situation_hv=situation,
        action=f"action_{i}",
        outcome="success",
        reward=1.0,
        impact_score=0.5 + i * 0.05
    )
    evicted = epi.add_episode(ep)
    if evicted:
        print(f"Evicted old episode: {evicted.timestamp}")

# Recall similar situations (NOTE: method is parallel_knn_search)
query = hrs.HyperVector(seed=5)
similar_eps = epi.parallel_knn_search(query, k=3, task_tag="task_a")
for ep in similar_eps:
    print(f"Action: {ep.action}, Reward: {ep.reward}")

# Get recent episodes (NOTE: method is get_recent_episodes)
recent = epi.get_recent_episodes("task_a", n=5)
print(f"Last 5 episodes: {[ep.action for ep in recent]}")

# High-impact episodes
high_impact = epi.get_high_impact_episodes(threshold=0.7, limit=5)
print(f"High-impact count: {len(high_impact)}")

# Search by reward
good_episodes = epi.search_by_reward(min_reward=0.8, max_reward=1.0)

# Statistics
stats = epi.get_stats()
print(f"Total episodes: {epi.size()}")
```

**Performance**:
- `add_episode`: ~2µs (lock on write)
- `parallel_knn_search`: ~500µs for 1000 episodes (parallel KNN)

---

## Cognitive Infrastructure

### HyperVectorRegistry

**Thread-safe registry** for named hypervectors (like a concurrent hashmap).

#### Constructor

```python
HyperVectorRegistry() -> HyperVectorRegistry
```

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

registry = hrs.HyperVectorRegistry()
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `register(name: str, hv: HV)` | Store HV |
| `get` | `get(name: str) -> Optional[HV]` | Retrieve HV |
| `contains` | `contains(name: str) -> bool` | Check if exists |
| `remove` | `remove(name: str) -> bool` | Remove HV |
| `clear` | `clear()` | Remove all |

**Example**:
```python
registry = hrs.HyperVectorRegistry()

# Register
registry.register("concept_a", hrs.HyperVector(seed=1))
registry.register("concept_b", hrs.HyperVector(seed=2))

# Retrieve
hv = registry.get("concept_a")
if hv:
    print(f"Found: {hv}")

# Check
if registry.contains("concept_a"):
    print("Concept A exists")

# Remove
registry.remove("concept_b")
```

---

### ActivationAccumulator

**Thread-safe accumulator** for spreading activation values.

#### Constructor

```python
ActivationAccumulator() -> ActivationAccumulator
```

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

acc = hrs.ActivationAccumulator()
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_activation` | `add_activation(concept: str, value: float)` | Add to activation |
| `get_activation` | `get_activation(concept: str) -> float` | Get current value |
| `decay` | `decay(factor: float)` | Multiply all by factor |
| `top_k` | `top_k(k: int) -> List[Tuple[str, float]]` | Get most activated |
| `clear` | `clear()` | Reset all to 0.0 |

**Example**:
```python
acc = hrs.ActivationAccumulator()

# Spreading activation
acc.add_activation("concept_a", 1.0)
acc.add_activation("concept_b", 0.5)
acc.add_activation("concept_a", 0.2)  # Accumulates: now 1.2

# Get activation
val = acc.get_activation("concept_a")
print(f"Activation: {val}")  # 1.2

# Decay (e.g., 5% per step)
acc.decay(0.95)

# Top activated
top = acc.top_k(k=3)
print(top)  # [("concept_a", 1.14), ("concept_b", 0.475), ...]
```

**Use Case**: Global workspace competition, attention mechanisms.

---

### CognitiveWorkerPool

**Multi-threaded worker pool** for parallel cognitive operations.

#### Constructor

```python
CognitiveWorkerPool(
    semantic_memory: SemanticMemoryConcurrent,
    episodic_memory: EpisodicMemoryConcurrent,
    num_workers: int = 4
) -> CognitiveWorkerPool
```

**Args**:
- `semantic_memory`: Shared semantic memory instance
- `episodic_memory`: Shared episodic memory instance
- `num_workers`: Number of worker threads (default: 4, recommended: CPU cores)

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

sem = hrs.SemanticMemoryConcurrent()
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=1000)

# 8-worker pool
pool = hrs.CognitiveWorkerPool(sem, epi, num_workers=8)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `submit_task` | `submit_task(task_type: str, data: dict)` | Submit task to queue |
| `get_result` | `get_result(timeout: float) -> Optional[dict]` | Get result (blocking) |
| `shutdown` | `shutdown()` | Stop all workers |

**Example**:
```python
pool = hrs.CognitiveWorkerPool(sem, epi, num_workers=4)

# Submit similarity search tasks
for i in range(100):
    query = hrs.HyperVector(seed=i)
    pool.submit_task("similarity_search", {
        "query": query,
        "k": 10
    })

# Collect results
for _ in range(100):
    result = pool.get_result(timeout=1.0)
    if result:
        print(f"Found {len(result['matches'])} matches")

pool.shutdown()
```

**Performance**: Near-linear scaling up to number of physical cores.

---

## Storage

### PersistentStorage

**SQLite-backed persistent storage** for hypervectors and episodes.

#### Constructor

```python
PersistentStorage(
    db_path: str,
    batch_size: int = 100
) -> PersistentStorage
```

**Args**:
- `db_path`: Path to SQLite database file (created if doesn't exist)
- `batch_size`: Number of items to batch before auto-commit (default: 100)

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

# Create database
store = hrs.PersistentStorage("my_brain.db", batch_size=200)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `store_hypervector` | `store_hypervector(name: str, hv: HV)` | Save HV to DB |
| `load_hypervector` | `load_hypervector(name: str) -> Optional[HV]` | Load HV from DB |
| `store_episode` | `store_episode(ep: Episode)` | Save episode to DB |
| `load_episodes` | `load_episodes(task_tag: str, limit: int) -> List[Episode]` | Load episodes |
| `commit` | `commit()` | Force commit pending writes |
| `close` | `close()` | Close connection |

**Example**:
```python
store = hrs.PersistentStorage("brain.db", batch_size=50)

# Store concepts
for i in range(1000):
    hv = hrs.HyperVector(seed=i)
    store.store_hypervector(f"concept_{i}", hv)

# Auto-commits every 50 items, or force:
store.commit()

# Load concept
hv = store.load_hypervector("concept_42")

# Store episodes
import time
for i in range(100):
    ep = hrs.Episode(
        timestamp=time.time() + i,
        task_tag="task_a",
        situation_hv=hrs.HyperVector(seed=i),
        action=f"action_{i}",
        outcome="success",
        reward=1.0
    )
    store.store_episode(ep)

# Load episodes
episodes = store.load_episodes("task_a", limit=10)

store.close()
```

**Performance**: 
- Write: ~10µs per item (batched)
- Read: ~50µs per item (indexed)

---

## Async Runtime

### AsyncCognitiveRuntime

**Tokio-based async runtime** for non-blocking cognitive operations.

#### Constructor

```python
AsyncCognitiveRuntime(
    semantic_memory: SemanticMemoryConcurrent,
    episodic_memory: EpisodicMemoryConcurrent
) -> AsyncCognitiveRuntime
```

**Args**:
- `semantic_memory`: Semantic memory instance
- `episodic_memory`: Episodic memory instance

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

sem = hrs.SemanticMemoryConcurrent()
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=1000)

runtime = hrs.AsyncCognitiveRuntime(sem, epi)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `semantic_search_async` | `semantic_search_async(query: HV, k: int) -> List[Tuple[str, float]]` | Async search |
| `episodic_recall_async` | `episodic_recall_async(query: HV, k: int, task: Optional[str]) -> List[Episode]` | Async recall |

**Example**:
```python
runtime = hrs.AsyncCognitiveRuntime(sem, epi)

# Non-blocking search
query = hrs.HyperVector(seed=42)
results = runtime.semantic_search_async(query, k=10)
print(results)

# Non-blocking recall
episodes = runtime.episodic_recall_async(query, k=5, task="robot_nav")
```

**Use Case**: Prevents blocking main thread during inference.

---

## Parallel Functions

### parallel_similarity_search

**Parallel similarity search** across a collection of hypervectors.

```python
parallel_similarity_search(
    query: HyperVector,
    candidates: List[HyperVector],
    k: int
) -> List[Tuple[int, float]]
```

**Args**:
- `query`: Query hypervector
- `candidates`: List of candidate hypervectors
- `k`: Number of top results to return

**Returns**: List of `(index, similarity)` tuples, sorted by similarity (descending)

**Example**:
```python
from python.core.vsa import hypervec_shim as hrs

query = hrs.HyperVector(seed=0)
candidates = [hrs.HyperVector(seed=i) for i in range(1000)]

# Find top 10 most similar
results = hrs.parallel_similarity_search(query, candidates, k=10)
for idx, sim in results:
    print(f"Candidate {idx}: similarity = {sim:.3f}")
```

**Performance**: ~100µs for 1000 candidates (8 cores)

---

### batch_parallel_similarity_search

**Batch parallel search** for multiple queries.

```python
batch_parallel_similarity_search(
    queries: List[HyperVector],
    candidates: List[HyperVector],
    k: int
) -> List[List[Tuple[int, float]]]
```

**Args**:
- `queries`: List of query hypervectors
- `candidates`: List of candidate hypervectors
- `k`: Top-k per query

**Returns**: List of result lists (one per query)

**Example**:
```python
queries = [hrs.HyperVector(seed=i) for i in range(10)]
candidates = [hrs.HyperVector(seed=i+100) for i in range(1000)]

batch_results = hrs.batch_parallel_similarity_search(queries, candidates, k=5)

for query_idx, results in enumerate(batch_results):
    print(f"Query {query_idx}:")
    for idx, sim in results:
        print(f"  Candidate {idx}: {sim:.3f}")
```

**Performance**: ~1ms for 10 queries × 1000 candidates (8 cores)

---

### parallel_bundle

**Parallel bundling** (majority vote) across multiple vectors.

```python
parallel_bundle(vectors: List[HyperVector]) -> HyperVector
```

**Args**:
- `vectors`: List of hypervectors to bundle (2+)

**Returns**: Bundled hypervector (majority vote per bit)

**Example**:
```python
vectors = [hrs.HyperVector(seed=i) for i in range(100)]

# Majority vote: prototype of 100 vectors
prototype = hrs.parallel_bundle(vectors)
```

**Performance**: ~50µs for 100 vectors (parallel bit counting)

---

### run_semantic_search_async

**Async wrapper** for semantic search.

```python
run_semantic_search_async(
    semantic_memory: SemanticMemoryConcurrent,
    query_hv: HyperVector,
    k: int
) -> List[Tuple[str, float]]
```

**Args**:
- `semantic_memory`: Memory instance
- `query_hv`: Query vector
- `k`: Top-k results

**Returns**: List of `(concept_name, similarity)` tuples

**Example**:
```python
sem = hrs.SemanticMemoryConcurrent()
# ... add concepts ...

query = hrs.HyperVector(seed=42)
results = hrs.run_semantic_search_async(sem, query, k=10)
```

---

## Type Mappings

### Python ↔ Rust Types

| Python Type | Rust Type | Notes |
|-------------|-----------|-------|
| `HyperVector` | `HyperVector` | 160 × u64 blocks |
| `int` | `i64`, `usize` | Auto-converted |
| `float` | `f64` | Auto-converted |
| `str` | `String` | UTF-8 encoded |
| `List[T]` | `Vec<T>` | Converted via PyO3 |
| `Optional[T]` | `Option<T>` | `None` ↔ `None` |
| `Dict[K, V]` | `HashMap<K, V>` | Converted |

### Thread Safety

| Class | Thread-Safe? | Locking Strategy |
|-------|--------------|------------------|
| `HyperVector` | ✅ Yes (immutable) | None (copy-on-write) |
| `SemanticMemoryConcurrent` | ✅ Yes | DashMap (lock-free) |
| `EpisodicMemoryConcurrent` | ✅ Yes | RwLock (read-heavy) |
| `HyperVectorRegistry` | ✅ Yes | DashMap (lock-free) |
| `ActivationAccumulator` | ✅ Yes | DashMap (lock-free) |
| `CognitiveWorkerPool` | ✅ Yes | MPSC channels |
| `PersistentStorage` | ⚠️ Single-writer | SQLite lock |
| `AsyncCognitiveRuntime` | ✅ Yes | Tokio runtime |

---

## Performance Summary

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| HV Creation | ~5µs | ~200ns | **25x** |
| XOR Binding | ~10µs | ~50ns | **200x** |
| Bundling | ~50µs | ~200ns | **250x** |
| Similarity | ~10µs | ~100ns | **100x** |
| Semantic Search (1K) | ~10ms | ~100µs | **100x** |
| Episodic Recall (1K) | ~25ms | ~500µs | **50x** |
| Parallel Bundle (100) | ~2ms | ~50µs | **40x** |

**Hardware:** 8-core CPU, hyperthreading enabled

---

## Best Practices

### 1. Use Rust for Hot Paths

```python
# ✅ GOOD: Use Rust for performance-critical operations
results = hrs.parallel_similarity_search(query, candidates, k=10)

# ❌ BAD: Python loop for similarity search
results = [(i, query.similarity(c)) for i, c in enumerate(candidates)]
```

### 2. Batch Operations

```python
# ✅ GOOD: Batch search (single parallelization overhead)
results = hrs.batch_parallel_similarity_search(queries, candidates, k=5)

# ❌ BAD: Multiple serial searches
results = [hrs.parallel_similarity_search(q, candidates, k=5) for q in queries]
```

### 3. Reuse Memory Instances

```python
# ✅ GOOD: Shared memory across components
sem = hrs.SemanticMemoryConcurrent()
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=10000)
pool = hrs.CognitiveWorkerPool(sem, epi, num_workers=8)

# ❌ BAD: Creating multiple instances
sem1 = hrs.SemanticMemoryConcurrent()
sem2 = hrs.SemanticMemoryConcurrent()  # Isolated, can't share
```

### 4. Graceful Degradation

```python
from python.core.vsa import hypervec_shim as hrs

# Always works (Rust or Python fallback)
if hrs.__backend__ == "Rust":
    print("Using 10-100x faster Rust backend")
else:
    print("Using Python fallback (slower but portable)")
```

---

## FAQ

### Q: How do I check if Rust backend is available?

```python
from python.core.vsa import hypervec_shim as hrs

print(f"Backend: {hrs.__backend__}")  # "Rust" or "Python"

# Check specific class
if hrs.SemanticMemoryConcurrent is not None:
    print("Rust semantic memory available")
```

### Q: Can I mix Rust and Python HyperVectors?

**Yes!** They implement the same interface:

```python
rust_hv = hrs.HyperVector(seed=1)
python_hv = hypervec_py.HyperVectorPy(bits=...)

# Both work the same
result = rust_hv.xor(python_hv)  # ✅ Works
```

### Q: What happens if I pass wrong argument types?

Rust will raise `TypeError` with helpful message:

```python
sem = hrs.SemanticMemoryConcurrent()
sem.add_concept(123, "not_a_hypervector")  # ❌ TypeError

# Correct:
sem.add_concept("concept", hrs.HyperVector())  # ✅ OK
```

### Q: How do I debug Rust panics?

Set `RUST_BACKTRACE=1`:

```bash
RUST_BACKTRACE=1 python my_script.py
```

### Q: How do I rebuild the Rust extension?

```bash
cd nsck/rust_vsa
cargo build --release
maturin develop --release
```

---

## See Also

- [NSCK V3 Enhancements](NSCK_V3_ENHANCEMENTS.md) - Implementation details
- [Architecture](ARCHITECTURE.md) - System architecture
- [Formulas](FORMULAS.md) - VSA mathematics
- [Module Reference](MODULE_REFERENCE.md) - Python modules

---

**Last Updated:** February 19, 2026  
**NSCK Version:** V3  
**Rust Backend Version:** 0.3.0
