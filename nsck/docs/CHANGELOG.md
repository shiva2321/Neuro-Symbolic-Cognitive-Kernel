# NSCK Changelog

## V5 — Societal Knowledge World (March 2026)

- **SocietalKnowledgeWorld** hierarchical memory substrate (Towns → Cities → States → Countries → Continents)
- **LivingHyperVector** — active state tracking (energy, stability, social valence) per concept
- **Spectral Renormalization Group** — coarse-graining of large knowledge graphs via spectral eigendecomposition
- **Hodge Laplacians (L0–L2)** — higher-order topological analysis for triadic closure and latent flow detection
- **TDA Persistent Homology** — real-time semantic health monitoring via Betti numbers (β₀, β₁)
- **PercolationMonitor** — autonomous domain emergence via cluster density and phase transitions
- **ThoughtTrace enrichment** (V31) — richer decision trace annotations
- **rust_societal** — third Rust crate for societal HV operations
- **SOCIETAL coalition** added to GlobalWorkspace broadcast

## V4 — VSA-NLU, KnowledgeSeeder, LSH Memory, Rust Default-On (February 2026)

- **VSANLUEngine** (`language/vsa_nlu.py`) — VSA-based intent classifier (7 intents, entity extraction)
- **KnowledgeSeeder** (`bootstrap/knowledge_seeder.py`) — YAML domain bootstrapper; `navigation.yaml` + `scheduling.yaml` kits
- **ProceduralMemory LSH** — 16-bit LSH bucket index, O(1) lookup, threshold 0.72
- **SemanticMemory hot cache** — 256-entry LRU cache; HNSW default-on
- **EWC-aware rule pruning** — `gwt_win_count` + `ewc_importance` composite score
- `bundle_hvs`, `lsh_bucket`, `spreading_activation_step` added to `rust_vsa`
- `imagine_rollout()` multi-step planning; auto-cache skills on positive reward
- `NSCK_USE_RUST=1` default-on

## V18 — Concurrent Memory (February 2026)

- **SemanticMemoryConcurrent** — Rayon parallel spreading activation (all steps inside Rust)
- Write-time mirror: `add_concept` / `add_relation` push to Rust DashMap immediately
- Weighted edges: `add_relation_weighted` stores typed relation weights

## V17 — Societal Context (February 2026)

- **SocietalContextRouter** — integrates societal hierarchy into SemanticMemory queries
- Societal coalition added to GWT broadcast set

## V16 — Security, EWC, Eval Suite (February 2026)

- **SafetyVerifier** enhancements — declarative safety properties, veto gates
- **EWC (Elastic Weight Consolidation)** — prevents catastrophic forgetting of important rules
- Evaluation harness (`eval/`) with 15 scripts

## V15 — Model Transplantation (February 2026)

- **TransplantPipeline** — 5-stage pipeline: Harvest → Project → Calibrate → Validate → Integrate
- **ModelHarvester** — extracts embedding matrices from any pre-trained neural model
- **SVDFactoredProjector** (default), RandomProjector, LearnedProjector
- **STDPCalibrator** — SNN fine-tuning of initial codebook
- **TransplantValidator** — Spearman ρ, Recall@10/50, ARI metrics

## V14 — Rich Perception (February 2026)

- Rich image/audio/video adapters with PerceptionDistiller quality tracking
- Rust spreading activation wiring (`semantic_memory_shim.py`)
- Knowledge Packs (`.kp` files) for portable domain knowledge
- Scale validation benchmarks

## V13 — Universal Substrate (February 2026)

- **UniversalHVEncoder** — multi-modal signal → HV with statistics
- **SignalIngestor** — raw signal pre-processing
- **PatternGeneralizer** — clusters HVs into abstract prototypes
- **ConformalWrapper** — calibrated uncertainty bounds
- KLE uncertainty in GlobalWorkspace
- Cross-modal associative memory
- StagedRecall (fast → deep), ConceptDriftDetector, Homeostasis
