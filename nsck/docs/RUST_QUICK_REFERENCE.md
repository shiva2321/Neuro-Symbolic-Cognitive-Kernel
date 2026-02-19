# NSCK Rust Backend Quick Reference

**Fast lookup for correct method signatures**

---

## Import

```python
from python.core.vsa import hypervec_shim as hrs
```

---

## Core Classes

### HyperVector

```python
# Create
hv = hrs.HyperVector(seed=42)          # Random with seed
hv_zero = hrs.HyperVector.zero()       # All zeros

# Operations
result = hv1.xor(hv2)                   # Binding
result = hv1.bundle(hv2)                # Bundling (random tie-break)
result = hv1.weighted_bundle(hv2, weight=0.7, seed=None)  # Weighted
result = hv.permute(shift=5)            # Temporal encoding
result = hv.permute_inverse(shift=5)    # Inverse permutation
similarity = hv1.similarity(hv2)        # Returns float [0, 1]
hash_val = hv.lsh_hash(seed=42, n_bits=32)  # LSH signature
```

---

## Memory Systems

### SemanticMemoryConcurrent

```python
# Create
sem = hrs.SemanticMemoryConcurrent()

# Add concepts
sem.add_concept("dog", hv)

# Retrieve
hv = sem.get_concept("dog")             # Returns HV or None

# Relations
sem.add_relation("dog", "animal")       # Directed edge
neighbors = sem.get_neighbors("dog")    # List[str]
incoming = sem.get_incoming_neighbors("animal")  # List[str]

# Search
results = sem.parallel_semantic_search(query_hv, k=10)
# Returns: List[Tuple[str, float]] = [(name, similarity), ...]

# Spreading activation
activation = sem.parallel_spread_activation(
    start_concepts=["dog", "cat"],
    steps=3,                # Number of spreading steps
    decay=0.9,              # Decay per step
    min_activation=0.1,     # Min threshold
    bidirectional=True      # Spread along incoming edges too
)
# Returns: Dict[str, float] = {"dog": 1.0, "animal": 0.9, ...}

# Hybrid search (VSA + graph)
results = sem.hybrid_search(query_hv, start_concepts=["dog"], k=10)

# Get activated concepts above threshold
activated = sem.get_activated_concepts(threshold=0.5)
# Returns: List[Tuple[str, float]]

# Stats
count = sem.concept_count()             # int
count = sem.relation_count()            # int
names = sem.get_all_concepts()          # List[str]
stats = sem.get_stats()                 # Dict[str, int]

# Clear
sem.clear()                              # Remove all
```

### EpisodicMemoryConcurrent

```python
# Create
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=10000)

# Create episode
import time
ep = hrs.Episode(
    timestamp=time.time(),
    task_tag="robot_nav",
    situation_hv=hv,
    action="move_forward",
    outcome="success",
    reward=1.0,
    impact_score=0.8
)

# Add episode
evicted = epi.add_episode(ep)           # Returns Episode or None (if evicted)

# Batch add
evicted_list = epi.batch_add_episodes([ep1, ep2, ep3])

# Search by similarity (KNN)
episodes = epi.parallel_knn_search(
    query_hv=query,
    k=10,
    task_tag="robot_nav"                # Optional: filter by task
)
# Returns: List[Episode]

# Batch KNN search
batch_results = epi.batch_knn_search(
    queries=[hv1, hv2, hv3],
    k=5,
    task_tag=None                        # Optional
)
# Returns: List[List[Episode]]

# Get recent episodes
recent = epi.get_recent_episodes(task_tag="robot_nav", n=10)

# Get all
all_eps = epi.get_all_episodes()

# Filter by impact
high_impact = epi.get_high_impact_episodes(threshold=0.7, limit=10)

# Search by task
task_eps = epi.search_by_task(task_tag="robot_nav")

# Search by reward
good_eps = epi.search_by_reward(min_reward=0.5, max_reward=1.0)

# Stats
size = epi.size()                        # int: number of episodes
stats = epi.get_stats()                  # Dict[str, Any]

# Clear
epi.clear()
```

---

## Infrastructure

### HyperVectorRegistry

```python
reg = hrs.HyperVectorRegistry()

reg.register("concept_a", hv)           # Store
hv = reg.get("concept_a")               # Retrieve (None if missing)
exists = reg.contains("concept_a")      # bool
removed = reg.remove("concept_a")       # bool (True if existed)
reg.clear()                              # Remove all
```

### ActivationAccumulator

```python
acc = hrs.ActivationAccumulator()

acc.add_activation("concept_a", 1.0)    # Accumulate (adds to existing)
val = acc.get_activation("concept_a")   # float (0.0 if missing)
acc.decay(factor=0.95)                   # Multiply all by 0.95
top = acc.top_k(k=5)                     # List[Tuple[str, float]]
acc.clear()                              # Reset all
```

### CognitiveWorkerPool

```python
sem = hrs.SemanticMemoryConcurrent()
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=1000)

pool = hrs.CognitiveWorkerPool(
    semantic_memory=sem,
    episodic_memory=epi,
    num_workers=8                        # Default: 4
)

# Use worker pool for parallel tasks
pool.submit_task(task_type="search", data={...})
result = pool.get_result(timeout=1.0)   # Returns dict or None
pool.shutdown()
```

### PersistentStorage

```python
store = hrs.PersistentStorage(
    db_path="brain.db",
    batch_size=100                       # Default: 100
)

store.store_hypervector("concept_1", hv)
hv = store.load_hypervector("concept_1")  # None if missing

store.store_episode(episode)
episodes = store.load_episodes(task_tag="robot_nav", limit=100)

store.commit()                           # Force write pending batch
store.close()
```

### AsyncCognitiveRuntime

```python
sem = hrs.SemanticMemoryConcurrent()
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=1000)

runtime = hrs.AsyncCognitiveRuntime(
    semantic_memory=sem,
    episodic_memory=epi
)

# Non-blocking operations
results = runtime.semantic_search_async(query_hv, k=10)
episodes = runtime.episodic_recall_async(query_hv, k=5, task="robot_nav")
```

---

## Parallel Functions

### parallel_similarity_search

```python
query = hrs.HyperVector(seed=0)
candidates = [hrs.HyperVector(seed=i) for i in range(1000)]

results = hrs.parallel_similarity_search(
    query=query,
    candidates=candidates,
    k=10
)
# Returns: List[Tuple[int, float]] = [(index, similarity), ...]
```

### batch_parallel_similarity_search

```python
queries = [hrs.HyperVector(seed=i) for i in range(5)]
candidates = [hrs.HyperVector(seed=i+100) for i in range(1000)]

batch_results = hrs.batch_parallel_similarity_search(
    queries=queries,
    candidates=candidates,
    k=10
)
# Returns: List[List[Tuple[int, float]]]
# batch_results[i] = results for queries[i]
```

### parallel_bundle

```python
vectors = [hrs.HyperVector(seed=i) for i in range(100)]

bundled = hrs.parallel_bundle(vectors=vectors)
# Returns: HyperVector (majority vote across all vectors)
```

### run_semantic_search_async

```python
sem = hrs.SemanticMemoryConcurrent()
# ... add concepts ...

results = hrs.run_semantic_search_async(
    semantic_memory=sem,
    query_hv=query,
    k=10
)
# Returns: List[Tuple[str, float]]
```

---

## Helper Functions

### get_rust_help

```python
# Get documentation for Rust classes
doc = hrs.get_rust_help("CognitiveWorkerPool")
print(doc)

# Check backend
print(hrs.__backend__)  # "Rust" or "Python"
```

---

##Common Patterns

### Creating and Querying Semantic Network

```python
sem = hrs.SemanticMemoryConcurrent()

# Build ontology
concepts = {
    "dog": hrs.HyperVector(seed=1),
    "cat": hrs.HyperVector(seed=2),
    "animal": hrs.HyperVector(seed=3),
    "mammal": hrs.HyperVector(seed=4),
}

for name, hv in concepts.items():
    sem.add_concept(name, hv)

# Add hierarchy
sem.add_relation("dog", "mammal")
sem.add_relation("cat", "mammal")
sem.add_relation("mammal", "animal")

# Query
results = sem.parallel_semantic_search(concepts["dog"], k=5)
print(results)  # [("dog", 1.0), ("cat", 0.51), ...]

# Spread activation from "dog"
activation = sem.parallel_spread_activation(
    start_concepts=["dog"],
    steps=2,
    decay=0.9,
    min_activation=0.1,
    bidirectional=False
)
print(activation)  # {"dog": 1.0, "mammal": 0.9, "animal": 0.81, ...}
```

### Episodic Memory with Filtering

```python
import time
epi = hrs.EpisodicMemoryConcurrent(max_hot_size=1000)

# Add experiences
for i in range(100):
    situation = hrs.HyperVector(seed=i)
    ep = hrs.Episode(
        timestamp=time.time() + i,
        task_tag="maze_task" if i < 50 else "nav_task",
        situation_hv=situation,
        action=f"action_{i % 4}",
        outcome="success" if i % 2 == 0 else "failure",
        reward=1.0 if i % 2 == 0 else -0.5,
        impact_score=0.1 + i * 0.01
    )
    epi.add_episode(ep)

# Find similar situations
query = hrs.HyperVector(seed=25)
similar = epi.parallel_knn_search(query, k=5, task_tag="maze_task")

# Get high-impact episodes
important = epi.get_high_impact_episodes(threshold=0.8, limit=10)

# Get successful episodes
successes = epi.search_by_reward(min_reward=0.5, max_reward=1.0)

print(f"Total: {epi.size()}, Similar: {len(similar)}, Important: {len(important)}")
```

### Parallel Similarity Search

```python
# Generate test data
query = hrs.HyperVector(seed=0)
database = [hrs.HyperVector(seed=i) for i in range(10000)]

# Single query
results = hrs.parallel_similarity_search(query, database, k=100)
top_10 = results[:10]
print(f"Top 10: {[(idx, f'{sim:.3f}') for idx, sim in top_10]}")

# Multiple queries
queries = [hrs.HyperVector(seed=i) for i in range(10)]
batch_results = hrs.batch_parallel_similarity_search(queries, database, k=50)

for i, results in enumerate(batch_results):
    print(f"Query {i}: found {len(results)} results, top sim = {results[0][1]:.3f}")
```

---

## Performance Tips

1. **Use Rust backend for hot paths**: Check `hrs.__backend__ == "Rust"`
2. **Batch operations**: Use `batch_*` methods for multiple queries
3. **Reuse memory instances**: Create once, use many times
4. **Set appropriate hot tier size**: Balance memory vs. speed
5. **Use correct parameter names**: `steps` not `depth`, `max_hot_size` not `capacity`

---

## Error Handling

```python
# Always handle None returns
hv = sem.get_concept("unknown")
if hv is None:
    print("Concept not found")

# Check backend availability
if hrs.SemanticMemoryConcurrent is None:
    print("Rust backend not available, using Python fallback")
    # Use Python implementation

# Type errors
try:
    sem.add_concept(123, "not_a_hv")  # Wrong types
except TypeError as e:
    print(f"Type error: {e}")
```

---

## Full Documentation

- **Comprehensive API**: [RUST_API_REFERENCE.md](RUST_API_REFERENCE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Enhancements**: [NSCK_V3_ENHANCEMENTS.md](NSCK_V3_ENHANCEMENTS.md)

---

**Last Updated:** February 19, 2026  
**NSCK Version:** V3.0  
**Rust Backend:** 0.3.0
