# NSCK V5: Societal Hypervector Knowledge Representation — Architecture

## Overview

NSCK V5 transforms the flat Vector Symbolic Architecture (VSA) space of previous versions into a hierarchical, emergent "societal world" of hypervectors. Each hypervector becomes a *LivingHyperVector* — a dynamic agent with social and chemical properties — and concepts self-organize into *KnowledgeNeighborhoods* and *KnowledgeDomains* analogous to districts and cities.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SocietalKnowledgeWorld                       │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  KnowledgeDomain│  │KnowledgeDomain   │  │  KnowledgeDomain│ │
│  │  "Science"     │  │"Arts"            │  │  "Technology"  │  │
│  │  ┌─────────┐   │  │  ┌───────────┐   │  │  ┌──────────┐  │  │
│  │  │Nbhd_A1  │   │  │  │ Nbhd_C1   │   │  │  │ Nbhd_E1  │  │  │
│  │  │(Physics)│   │  │  │ (Music)   │   │  │  │ (AI)     │  │  │
│  │  │ [LHVs]  │   │  │  │ [LHVs]    │   │  │  │ [LHVs]   │  │  │
│  │  └─────────┘   │  │  └───────────┘   │  │  └──────────┘  │  │
│  │  city_hall_hv  │  │  city_hall_hv    │  │  city_hall_hv  │  │
│  └────────────────┘  └──────────────────┘  └────────────────┘  │
│              ↕ bonds                ↕ bonds                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │           LivingHyperVector Pool (concept atoms)       │     │
│  │  cat(age=42,stab=0.8,val=0.2,bonds={dog:0.7,...})     │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
  SocietalHNSW      SpectralLaplacianRG    TDAHealthMonitor
  (navigation)      (emergence)            (health)
         ↑
  SocietalContextRouter
  (GWT integration)
```

---

## Module Map

| Module | Path | Purpose |
|--------|------|---------|
| `LivingHyperVector` | `nsck/python/core/societal/living_hypervector.py` | Dynamic HV agent |
| `ValenceEngine` | `nsck/python/core/societal/valence_engine.py` | Bond chemistry |
| `KnowledgeNeighborhood` | `nsck/python/core/societal/knowledge_neighborhood.py` | District structure |
| `KnowledgeDomain` | `nsck/python/core/societal/knowledge_neighborhood.py` | City structure |
| `SocietalKnowledgeWorld` | `nsck/python/core/societal/societal_world.py` | World orchestrator |
| `SpectralLaplacianRG` | `nsck/python/core/societal/emergence/spectral_rg.py` | Coarse-graining |
| `PercolationMonitor` | `nsck/python/core/societal/emergence/percolation.py` | Phase transitions |
| `ZipfValidator` | `nsck/python/core/societal/emergence/zipf_validator.py` | Power-law health |
| `SocietalHNSW` | `nsck/python/core/societal/navigation/societal_hnsw.py` | Hierarchical ANN |
| `TDAHealthMonitor` | `nsck/python/core/societal/tda/tda_monitor.py` | Topological health |
| `SocietalContextRouter` | `nsck/python/core/societal/routing/context_router.py` | GWT routing |
| `SocietalTransplantStage` | `nsck/python/core/societal/transplant_extension.py` | Stage 7 seeding |
| Rust backend | `nsck/rust_vsa/src/societal.rs` | Fast property store + HNSW |
| V5 Dashboard | `nsck_vision/dashboard/v5_app.py` | Web UI |

---

## LivingHyperVector Properties

```
concept_id       — unique string identifier
hv               — raw hypervector (Python HyperVectorPy or Rust HyperVector)
age              — ticks since creation
stability        — [0,1] crystallization level
stability_class  — volatile | active | stable | crystallized
valence          — [-1,1] motivational charge
activation       — [0,1] current activation level
activation_history — ring buffer of recent activations (len=32)
domain_affinities  — {domain_name: affinity_score}
primary_domain   — domain with highest affinity
neighborhood_id  — which neighborhood this concept belongs to
electronegativity — centrality proxy (bonds × stability)
bonds            — {concept_id: bond_strength}
hybridization_state — free | bonded | hybridized
ewc_protection   — elastic weight consolidation protection
provenance       — {created_at, method, source_domain}
embedding        — original dense embedding (optional np.ndarray)
```

---

## Societal Tick Pipeline

Each call to `SocietalKnowledgeWorld.run_societal_tick()` executes:

```
1. LHV.tick() for every concept
   → age++, activation -= DECAY (0.05), stability += 0.001
   → update stability_class and electronegativity

2. (every 10 ticks) Bond updates per neighborhood:
   → try_break_bond(a, b) for all intra-neighborhood pairs
   → try_form_bond(a, b) for unconnected pairs

3. (every 50 ticks) Anchor re-election per neighborhood:
   → elect_anchor: pick highest electronegativity concept

4. (every 25 ticks) City-hall rebuild per domain:
   → bundle all anchor HVs → new city_hall_hv
```

---

## Spectral Renormalization Group Pipeline

```
concepts → build_similarity_graph()
         → NxN cosine similarity matrix (bipolar HV dot products)
         → threshold at similarity_threshold (default 0.3)
         ↓
         → compute_laplacian()
         → L = I - D^{-½} A D^{-½} (normalized Laplacian)
         ↓
         → eigendecompose()
         → (λ₀ ≤ λ₁ ≤ ... ≤ λ_{N-1}, eigenvectors)
         ↓
         → spectral_gap = λ₁ - λ₀
         → coarse_grain(): k-means on top eigenvectors → supernodes
         → build_supernode_graph(): cross-supernode bond summary
```

---

## TDA Health Pipeline

```
concepts → select_landmarks()  [maxmin sampling]
         ↓
         → build_witness_complex()
         → For each witness: find 2 nearest landmarks
         → Add edge (l₁, l₂) if both within max_edge_length
         ↓
         → compute_betti_numbers()
         → β₀ = components (union-find)
         → β₁ = V - E + β₀  (independent cycles)
         → β₂ = 0
         ↓
         → compute_persistence_diagram()
         → Add bond edges by decreasing weight, track merges
         ↓
         → health_score = 0.5·(1/β₀) + 0.3·min(1, β₁/threshold) + 0.2·entropy/3
```

---

## Rust Backend (societal.rs)

Three classes exposed via PyO3:

| Class | Purpose |
|-------|---------|
| `LivingHvStore` | Concurrent `Arc<RwLock<HashMap>>` property store. `tick_all(decay)` atomically ages all concepts. |
| `SocietalHnswRs` | Greedy NSW over `Vec<f32>`. Pre-normalises vectors; supports `add_item`, `query`. |
| `SpectralRgRs` | Static helpers: `cosine_similarity_matrix(vecs)`, `degree_vector(adj)`. |

Access via `nsck/python/core/societal/rust_shim.py` with transparent fallback.

---

## Integration Points

### GWT / CognitiveEngine

`SocietalContextRouter.update_coalition_scores(coalitions, stats)` boosts `base_salience` of coalitions whose content text mentions currently activated concepts (+0.1 per match).

### TransplantPipeline Stage 7

`SocietalTransplantStage.run_stage7(report, embeddings, domain_name)`:
1. Assigns stability by frequency rank
2. k-means clusters embeddings → sub-domain neighborhoods
3. Registers each token as LHV
4. Forms neighborhoods from clusters
5. Initializes bonds by cosine similarity
6. Forms a domain

---

## V5 Dashboard

```
POST /v5/analyze          — single image, per-claim accuracy ratings (8 claims)
POST /v5/batch_analyze    — batch images (JSON base64 or multipart)
GET  /v5/societal/stats   — world stats (n_concepts, stability dist, etc.)
GET  /v5/societal/health  — TDA + Zipf + percolation
GET  /v5/societal/domains — domain cards with neighborhoods
GET  /v5/benchmarks       — latest benchmark JSON
```

Run: `python nsck_vision/dashboard/v5_app.py --port 5092`
