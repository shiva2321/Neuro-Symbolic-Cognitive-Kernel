# NSCK V5 Societal API Reference

## 1. Core Societal Package

### `python.core.societal`

```python
from python.core.societal import (
    LivingHyperVector,
    ValenceEngine,
    KnowledgeNeighborhood,
    KnowledgeDomain,
    SocietalKnowledgeWorld,
)
```

---

## 2. LivingHyperVector

```python
class LivingHyperVector(concept_id: str, hv: Any, metadata: dict = None)
```

### Properties
| Property | Type | Description |
|----------|------|-------------|
| `concept_id` | `str` | Unique identifier |
| `hv` | `HyperVector` | Underlying hypervector |
| `age` | `int` | Ticks since creation |
| `stability` | `float` | [0,1] crystallization |
| `stability_class` | `str` | `volatile|active|stable|crystallized` |
| `valence` | `float` | Motivational charge |
| `activation` | `float` | Current activation [0,1] |
| `activation_history` | `List[float]` | Recent activations (len≤32) |
| `domain_affinities` | `Dict[str,float]` | Domain name → score |
| `primary_domain` | `str|None` | Domain with highest affinity |
| `neighborhood_id` | `str|None` | Assigned neighborhood |
| `electronegativity` | `float` | Centrality proxy |
| `bonds` | `Dict[str,float]` | Bonded concept → strength |
| `hybridization_state` | `str` | `free|bonded|hybridized` |
| `ewc_protection` | `float` | EWC protection score |
| `provenance` | `dict` | Origin metadata |

### Methods
```python
# Update activation; records to history
lhv.update_activation(value: float) -> None

# Advance one world tick (age++, decay activation, update stability)
lhv.tick() -> None

# Recompute stability_class from stability value
lhv.update_stability_class() -> None

# Add or update a bond
lhv.add_bond(other_id: str, strength: float) -> None

# Remove a bond
lhv.remove_bond(other_id: str) -> None

# Get domain affinity (0.0 if unknown)
lhv.get_affinity(domain: str) -> float

# Set domain affinity, update primary_domain
lhv.set_affinity(domain: str, score: float) -> None

# Return a context-conditioned copy (bundled with context_hv)
lhv.hybridize(context_hv: Any) -> LivingHyperVector

# Cosine similarity to another LHV
lhv.cosine_similarity_to(other: LivingHyperVector) -> float

# Serialize to dict (without raw HV bits)
lhv.to_dict() -> dict

# Reconstruct from dict + raw HV
LivingHyperVector.from_dict(d: dict, hv: Any) -> LivingHyperVector
```

---

## 3. ValenceEngine

```python
class ValenceEngine(bond_threshold=0.30, break_threshold=0.10, max_bonds_per_concept=20)
```

### Methods
```python
# Compute bond strength (0.0–1.0)
engine.compute_bond_strength(lhv_a, lhv_b) -> float

# Attempt to form a bond; returns strength if formed, else None
engine.try_form_bond(lhv_a, lhv_b) -> Optional[float]

# Break bond if below break_threshold; returns True if broken
engine.try_break_bond(lhv_a, lhv_b) -> bool

# Initialize bonds from precomputed similarity matrix
engine.bulk_initialize(concepts: List[LHV], sim_matrix: np.ndarray) -> int

# Update hybridization_state based on bond count
engine.update_hybridization(lhv, context_hv) -> None

# Retrieve top-k concepts by similarity (+optional context blending)
engine.context_conditioned_retrieval(
    query_hv, concepts: List[LHV],
    context_hv=None, top_k=10
) -> List[Tuple[str, float]]
```

---

## 4. KnowledgeNeighborhood

```python
class KnowledgeNeighborhood(neighborhood_id: str = None)
```

### Methods
```python
nbhd.add_concept(concept_id: str, lhv=None) -> None
nbhd.remove_concept(concept_id: str) -> None
nbhd.elect_anchor(concepts_dict: Dict[str, LHV]) -> Optional[str]
nbhd.compute_border_concepts(concepts_dict, world_graph=None) -> Set[str]
nbhd.merge_with(other: KnowledgeNeighborhood) -> KnowledgeNeighborhood
nbhd.stats() -> dict
```

---

## 5. KnowledgeDomain

```python
class KnowledgeDomain(name: str, domain_id: str = None, percolation_threshold=0.30)
```

### Methods
```python
domain.add_neighborhood(nbhd: KnowledgeNeighborhood) -> None
domain.remove_neighborhood(nbhd_id: str) -> None
domain.update_city_hall(neighborhoods: dict, concepts: dict) -> None
domain.compute_bridge_score(concept_id: str, concepts_dict: dict) -> float
domain.stats() -> dict
```

---

## 6. SocietalKnowledgeWorld

```python
class SocietalKnowledgeWorld(config=None, bond_threshold=0.30, break_threshold=0.10)
```

### Properties
| Property | Type | Description |
|----------|------|-------------|
| `concepts` | `Dict[str, LHV]` | All registered concepts |
| `neighborhoods` | `Dict[str, KnowledgeNeighborhood]` | All neighborhoods |
| `domains` | `Dict[str, KnowledgeDomain]` | All domains |
| `tick_count` | `int` | Current tick number |

### Methods
```python
# Register a concept (update HV if already exists)
world.register_concept(concept_id: str, hv, metadata: dict = None) -> LHV

# Activate + spread to bonded neighbors
world.activate_concept(concept_id: str, strength: float = 1.0) -> Optional[LHV]

# Query top-k by similarity
world.query(
    query_hv, top_k=10,
    context_hv=None, domain_filter=None
) -> List[Tuple[str, float]]

# Form a neighborhood from concept list
world.form_neighborhood(concept_ids: List[str], neighborhood_id=None) -> KnowledgeNeighborhood

# Form a domain from neighborhood list
world.form_domain(
    neighborhood_ids: List[str], domain_name: str, domain_id=None
) -> KnowledgeDomain

# Execute one societal tick
world.run_societal_tick() -> None

# Get primary domain of a concept
world.get_concept_domain(concept_id: str) -> Optional[str]

# Return full statistics dict
world.stats_report() -> dict
```

---

## 7. Emergence Modules

### SpectralLaplacianRG

```python
from python.core.societal.emergence import SpectralLaplacianRG

rg = SpectralLaplacianRG(n_eigenvectors=32, similarity_threshold=0.30, coarse_grain_ratio=0.50)

# Full pipeline
result = rg.run(concepts: Dict[str, LHV]) -> {
    supernodes: Dict[str, List[str]],
    supernode_graph: Dict[str, dict],
    spectral_gap: float,
    eigenvalues: List[float],
    n_concepts: int, n_supernodes: int
}
```

### PercolationMonitor

```python
from python.core.societal.emergence import PercolationMonitor

monitor = PercolationMonitor(bond_threshold=0.30, min_cluster_fraction=0.10)

monitor.monitor_tick(concepts) -> {
    is_transitioning: bool,
    giant_fraction: float,
    transition_type: str,  # emergence|merge|split|stable
    n_components: int, description: str
}

monitor.scan_threshold(concepts, thresholds=[0.0, ..., 1.0]) -> List[dict]
```

### ZipfValidator

```python
from python.core.societal.emergence import ZipfValidator

validator = ZipfValidator(min_concepts=10)

validator.validate(concepts) -> {
    is_zipf_like: bool, alpha: float, r_squared: float,
    entropy: float, health_score: float, description: str
}
```

---

## 8. Navigation

### SocietalHNSW

```python
from python.core.societal.navigation import SocietalHNSW

hnsw = SocietalHNSW(ef_construction=200, M=16, seed=42)
hnsw.build(world: SocietalKnowledgeWorld) -> None

hnsw.query(query_hv, top_k=10, domain_filter=None, context_hv=None) -> List[Tuple[str, float]]
hnsw.cross_domain_route(query_hv, source_domain, target_domain, top_k=5) -> List[Tuple[str, float]]
hnsw.stats() -> dict
```

---

## 9. TDA Health Monitor

```python
from python.core.societal.tda import TDAHealthMonitor

tda = TDAHealthMonitor(n_landmarks=50, max_edge_length=0.70, alert_threshold=0.30)

tda.run_analysis(concepts) -> {
    beta0: int, beta1: int, beta2: int,
    persistence_entropy: float, health_score: float,
    n_landmarks: int, n_witnesses: int, n_edges: int,
    n_persistence_pairs: int, alerts: List[str]
}

# Background daemon
tda.schedule_background(world, interval=60.0) -> None
tda.stop_background() -> None
```

---

## 10. Context Router

```python
from python.core.societal.routing import SocietalContextRouter

router = SocietalContextRouter(world, border_zone_radius=2, spreading_decay=0.30, max_hops=3)

router.detect_domain(query_hv) -> Tuple[Optional[str], float]

router.border_zone_routing(query_hv, domain_id) -> List[Tuple[str, float]]

router.domain_weighted_spreading_activation(
    seed_concept_id, domain_id=None, n_hops=3, top_k=20
) -> Dict[str, float]

router.route(query_hv, top_k=10) -> {
    domain_id: Optional[str], confidence: float,
    concepts: List[Tuple[str, float]],
    border_zone_active: bool, routing_trace: dict
}

router.update_coalition_scores(coalitions, world_stats) -> List
```

---

## 11. Transplant Extension

```python
from python.core.societal.transplant_extension import SocietalTransplantStage

stage = SocietalTransplantStage(world, n_domain_clusters=10, seed=42)

result = stage.run_stage7(
    transplant_report,
    embeddings: Dict[str, np.ndarray],
    domain_name: str,
    codebook: Optional[Dict[str, HV]] = None
) -> {
    n_concepts_registered: int,
    n_neighborhoods: int,
    n_bonds_formed: int,
    domain_id: str,
    societal_stats: dict,
    delta: dict
}
```

---

## 12. V5 Dashboard

### Start

```bash
python nsck_vision/dashboard/v5_app.py --port 5092 [--debug]
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v5/analyze` | Single image (base64 JSON) |
| `POST` | `/v5/batch_analyze` | Multiple images (JSON list or multipart) |
| `GET` | `/v5/societal/stats` | World statistics |
| `GET` | `/v5/societal/health` | TDA + Zipf + Percolation health |
| `GET` | `/v5/societal/domains` | Domain cards |
| `GET` | `/v5/benchmarks` | Latest benchmark results |
| `GET` | `/health` | API health check |

### Batch Analyze Request Format

```json
[
    {"image": "<base64-encoded-bytes>", "label": "photo1.jpg"},
    {"image": "<base64-encoded-bytes>", "label": "photo2.png"}
]
```

### Batch Analyze Response Format

```json
{
    "n_images": 2,
    "overall_confidence": 0.74,
    "results": [
        {
            "label": "photo1.jpg",
            "width": 640, "height": 480,
            "overall_confidence": 0.82,
            "claims": [
                {"claim": "Image loaded successfully", "confidence": 0.99, "rating": "high"},
                {"claim": "Scene contains varied visual content", "confidence": 0.75, "rating": "high"},
                ...
            ],
            "elapsed_ms": 4.2
        }
    ],
    "societal_stats": { ... }
}
```
