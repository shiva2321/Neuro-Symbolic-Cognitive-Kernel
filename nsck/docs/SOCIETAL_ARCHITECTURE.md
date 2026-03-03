# NSCK V26 — Societal Hypervector Knowledge Representation (SHVKR)

## Architecture Overview

The **Societal Hypervector Knowledge Representation** (SHVKR) system extends
NSCK's flat Vector Symbolic Architecture (VSA) with a dynamic, self-organising
*society* of knowledge atoms.  Each atom — a **LivingHyperVector** (LHV) — is
a binary hypervector enriched with:

- A **lifecycle** (birth epoch, activation level, decay dynamics).
- **Social bonds** to peers (strength, type, formation / dissolution rules).
- A **domain path** in a hierarchical knowledge taxonomy.
- A **community membership** (Leiden-style clustering).
- A **topological persistence** score (simplified H₀ filtration).

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        NSCKSubstrate                            │
│                                                                 │
│  process() / ingest()                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────┐     ┌──────────────────────────────┐  │
│  │  CognitiveEngine     │     │  SocietalContextRouter       │  │
│  │  (existing NSCK)     │     │  ├── nearest_neighbors()     │  │
│  └──────────────────────┘     │  ├── activate_concept()      │  │
│                               │  └── route() → ctx dict      │  │
│                               └──────────────────────────────┘  │
│                                         │                       │
│                               ┌─────────▼──────────────────┐   │
│                               │      SocietyManager         │   │
│                               │  ┌─ register/unregister     │   │
│                               │  ├─ auto_bond()             │   │
│                               │  ├─ leiden_cluster()        │   │
│                               │  ├─ percolation_threshold() │   │
│                               │  ├─ domain_tree()           │   │
│                               │  └─ step_epoch()            │   │
│                               └──────────┬─────────────────┘   │
│                                          │                      │
│                               ┌──────────▼─────────────────┐   │
│                               │  LivingHyperVector[]        │   │
│                               │  ├─ concept_id: str         │   │
│                               │  ├─ hv: HyperVector         │   │
│                               │  ├─ domain_path: List[str]  │   │
│                               │  ├─ activation: float       │   │
│                               │  ├─ bonds: Dict[id, Bond]   │   │
│                               │  └─ cluster_id: int         │   │
│                               └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. `LivingHyperVector` (`python/core/societal/living_hypervector.py`)

The atomic unit of societal knowledge.

| Attribute | Type | Description |
|-----------|------|-------------|
| `concept_id` | `str` | Unique identifier |
| `hv` | `HyperVector` | Binary VSA representation (10 240 bits) |
| `domain_path` | `List[str]` | Hierarchical domain address |
| `role` | `str` | `"domain_root"` / `"hub"` / `"bridge"` / `"leaf"` |
| `activation` | `float ∈ [0,1]` | Current salience |
| `birth_epoch` | `int` | Logical clock at creation |
| `_bonds` | `Dict[str, Bond]` | Weighted edges to peers |
| `_cluster_id` | `Optional[int]` | Community label from Leiden |
| `_topo_persistence` | `float` | Persistent homology lifetime |

**Key operations:**

```python
lhv.form_bond(peer, epoch=t, bond_type="similarity")
lhv.dissolve_bond(peer_id)
lhv.activate(delta=0.3, epoch=t)
lhv.decay_activation(rate=0.05)
lhv.spread_activation(peers_dict, spread_factor=0.4, epoch=t)
lhv.compute_topo_persistence(peers_dict)
lhv.to_dict()  # JSON-serialisable snapshot
```

### 2. `Bond` (`python/core/societal/living_hypervector.py`)

A weighted, typed edge between two LHVs.

```python
@dataclass
class Bond:
    peer_id: str
    strength: float          # ∈ [0, 1]
    bond_type: str           # "similarity" | "causal" | "hierarchical" | ...
    formed_epoch: int
    last_active_epoch: int
```

Bond lifecycle:
1. **Formation** — when `sim(a, b) >= bond_threshold`.
2. **Reinforcement** — `bond.reinforce(delta=0.05)` on co-activation.
3. **Decay** — `bond.decay(rate=0.01)` per epoch.
4. **Dissolution** — when `bond.strength < min_strength`.

### 3. `SocietyManager` (`python/core/societal/society_manager.py`)

Manages the full population of LHVs.

```python
mgr = SocietyManager(
    bond_threshold=0.65,
    max_bonds=8,
    bond_decay_rate=0.01,
    auto_cluster_interval=10,
)
mgr.register(lhv)
mgr.auto_bond()                       # O(N²) pairwise bonding
mgr.leiden_cluster(resolution=1.0)   # community detection
mgr.percolation_threshold()          # connectivity phase transition
mgr.step_epoch()                     # advance clock, decay, spread
mgr.domain_tree()                    # nested domain dict
```

**Leiden-style clustering:** Greedy modularity maximisation.  Each node is
initially its own community; nodes move to neighbours' communities when
modularity gain δQ > 0.  Iterates until convergence or N×5 steps.

Modularity Q is computed as:

```
Q = (1/2m) Σᵢⱼ [Aᵢⱼ - γ·kᵢkⱼ/2m] δ(cᵢ, cⱼ)
```

where m = total edge weight, kᵢ = node degree, γ = resolution parameter.

**Multi-resolution clustering:**

```python
results = mgr.multi_resolution_cluster([0.5, 1.0, 1.5, 2.0])
```

Higher γ → finer-grained communities.

### 4. `SocietalContextRouter` (`python/core/societal/societal_context_router.py`)

Integrates societal context into every substrate query.

```python
router = SocietalContextRouter(manager=mgr, top_k=5, min_similarity=0.55)
ctx = router.route(query_hv, task_tag="my_task")
# ctx = {
#   "matches": [{"concept_id": "...", "similarity": 0.72}, ...],
#   "top_concept": "quantum_field",
#   "top_similarity": 0.72,
#   "active_cluster": 3,
#   "cluster_concepts": ["quantum_field", "superposition", ...],
#   "activated_count": 3,
#   "epoch": 15,
# }
```

### 5. `SubstrateResult.societal_context`

Every `SubstrateResult` now carries a `societal_context` field (None unless
`enable_societal=True` and `init_societal_world()` has been called).

```python
result = substrate.ingest("The electron has spin 1/2", "physics_task")
print(result.societal_context["top_concept"])       # e.g. "electron"
print(result.societal_context["active_cluster"])    # e.g. 2
```

### 6. `NSCKConfig.societal()` preset

```python
cfg = NSCKConfig.societal()
# → enable_societal=True, bond_threshold=0.65, max_bonds=8, ...
substrate = NSCKSubstrate(config=cfg)
mgr = substrate.init_societal_world(concepts=[...])
```

---

## Transplant Pipeline Integration

`TransplantPipeline.societal_transplant()` extends the standard 6-stage
transplant pipeline by:

1. Running the standard `run()` pipeline (harvest → project → calibrate →
   validate → integrate → save).
2. Wrapping each projected token HV as a `LivingHyperVector` in the supplied
   `SocietyManager`.
3. Running `auto_bond()` to form similarity-based bonds among the new concepts.
4. Running `leiden_cluster()` to assign community membership.

```python
pipeline = TransplantPipeline()
report = pipeline.societal_transplant(
    model=bert_model,
    domain_name="nlp_concepts",
    strategy="svd_factored",
    societal_manager=mgr,
    bond_threshold=0.65,
    cluster_resolution=1.0,
)
```

---

## Dashboard API

```
GET  /societal/status          → health metrics
GET  /societal/domain_tree     → hierarchical domains
GET  /societal/communities     → current cluster listing
POST /societal/register        → {concept_id, domain_path, role}
POST /societal/query           → {concept_id, top_k}
POST /societal/bond            → {concept_a, concept_b, bond_type, strength}
POST /societal/step            → advance one epoch
GET  /societal/cluster         → ?resolution=1.0
GET  /societal/percolation     → percolation threshold
GET  /societal/topo_health     → full topological report
```

Start the dashboard:

```python
from python.core.api.societal_dashboard import SocietalDashboard
dash = SocietalDashboard()
dash.run(port=8080)
```

---

## Backwards Compatibility

All changes are additive and behind feature flags:

| Flag | Default | Effect |
|------|---------|--------|
| `NSCKConfig.enable_societal` | `False` | No societal processing |
| All `societal_*` config fields | sensible defaults | Controlled via `societal()` preset |

`SubstrateResult.societal_context` is `None` by default (no regression).

The transplant `run()` method is unchanged; `societal_transplant()` is a
new method that delegates to `run()` internally.

---

## Performance Characteristics

| Operation | Complexity | Backend |
|-----------|-----------|---------|
| `_hv_cosine_sim(a, b)` | O(D/64) | Hamming via Rust/Python |
| `auto_bond(N concepts)` | O(N²) | Python |
| `leiden_cluster(N, M bonds)` | O(N²) worst case | Python |
| `nearest_neighbors(query, k)` | O(N·D/64) | Python/Rust |
| `percolation_threshold(N)` | O(N²) | Python |
| `step_epoch(N)` | O(N·bonds_per_node) | Python |

For large societies (N > 1000), consider using `candidates` subsets in
`auto_bond()` and pre-clustering before routing.

---

## Test Coverage

220 unit tests in `nsck/tests/unit/societal/test_societal_hvs.py` covering:

- Bond lifecycle (formation, reinforcement, decay, dissolution)
- LHV activation dynamics (spike, decay, spreading)
- Serialisation roundtrips
- Topological persistence computation
- SocietyManager registration, bonding, clustering, percolation
- Domain tree and hierarchical queries
- Epoch stepping and auto-clustering
- SocietalContextRouter routing and feedback
- NSCKConfig societal preset and all flags
- SubstrateResult.societal_context field
- NSCKSubstrate.init_societal_world() and lazy initialisation
- Integration / regression scenarios
- Edge cases and stress tests

---

*NSCK V26 · March 2026 · Societal Hypervector Knowledge Representation*
