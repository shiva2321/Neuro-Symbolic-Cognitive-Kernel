# NSCK Architecture — V4 (Neuro-Symbolic Cognitive Kernel)

> **Version 4 (V4) — February 2026**
>
> This document is the definitive reference for NSCK's internal design.
> It covers the seven-layer architecture, data-flow pipeline,
> Global Workspace Theory (GWT) competition, dual-process routing,
> VSA / SNN subsystems, memory architecture, Rust acceleration,
> the V15 Model Transplantation Pipeline, V17 enrichment modules,
> and the full configuration system.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Seven-Layer Architecture](#2-seven-layer-architecture)
3. [Data Flow](#3-data-flow)
4. [Global Workspace Theory (GWT)](#4-global-workspace-theory-gwt)
5. [Dual-Process Architecture](#5-dual-process-architecture)
6. [VSA Engine](#6-vsa-engine)
7. [SNN Subsystem](#7-snn-subsystem)
8. [Memory Architecture](#8-memory-architecture)
9. [Perception Pipeline](#9-perception-pipeline)
10. [Rust Acceleration](#10-rust-acceleration)
11. [Configuration](#11-configuration)
12. [V15 Model Transplantation Pipeline](#12-v15-model-transplantation-pipeline)
13. [V17 Enrichment Modules](#13-v17-enrichment-modules)
14. [Seeding Subsystem](#14-seeding-subsystem)
15. [V4 Additions](#15-v4-additions)
16. [Design Boundaries, Known Gaps, and Trade-offs](#16-design-boundaries-known-gaps-and-trade-offs)

---

## 1. Overview

**NSCK** (Neuro-Symbolic Cognitive Kernel) is a cognitive architecture that
fuses three computational paradigms into a single decision loop:

| Paradigm | Implementation |
|---|---|
| **Vector-Symbolic Architecture (VSA)** | Binary 10 240-bit hypervectors for distributed representation |
| **Spiking Neural Network (SNN)** | LIF neurons + STDP learning for temporal pattern recognition |
| **Global Workspace Theory (GWT)** | Coalition competition + broadcast for conscious access |

### Core Design Principles

| Principle | Meaning |
|---|---|
| **Neuro-symbolic fusion** | Every concept exists as *both* a hypervector and a symbolic predicate |
| **Glass-box transparency** | Every decision produces a machine-readable `Explanation` trace |
| **Modality-agnostic perception** | All inputs enter through the `PerceptPacket` contract |
| **Biologically motivated** | GWT consciousness, Hebbian/STDP learning, LIF neurons, active inference |

---

## 2. Seven-Layer Architecture

```
Input → Perception → GWT Broadcast → CognitiveEngine
      → Memory (Semantic+Episodic+Procedural) → Learning → Output
```

```
Layer 6 — Executive      Metacognition · SelfModel · TheoryOfMind · Safety
Layer 5 — Language        Parser · ConstructionGrammar · VSANLUEngine · NLG · Dialogue
Layer 4 — Learning        Hebbian · Q-Learning · RuleInduction · ActiveInference · Curiosity · Conformal
Layer 3 — Reasoning       Causal · Rules · Planning · Analogy · Spatial · Math · Beliefs · imagine_rollout()
Layer 2 — Memory          Semantic (KG + HNSW + hot cache) · Episodic (Rust) · Procedural (LSH O(1))
Layer 1 — Perception      10 Adapters · SignalIngestor · UniversalHVEncoder · SNN (auto-grounded V4)
Layer 0 — Substrate       VSA Engine (10 240-bit HVs) · GWT · Dual-Process · Rust backends (default-on V4)
```

### Layer 0 — Substrate

| Component | File | Role |
|---|---|---|
| `HyperVector` | `core/vsa/hypervec_shim.py` → `hypervec_rs` | 10 240-bit binary VSA vector |
| `FHRRVector` | `core/vsa/fhrr.py` | Complex-phasor VSA variant |
| `GlobalWorkspace` | `core/reasoning/global_workspace.py` | Coalition competition + broadcast |
| `NSCKConfig` | `core/integration/config.py` | Central configuration |

### Layer 1 — Perception

| Adapter | File | Accepts |
|---|---|---|
| `TextAdapter` | `core/adapters/text_adapter.py` | `str` |
| `DictStateAdapter` | `core/adapters/dict_state_adapter.py` | `dict` |
| `NumericAdapter` | `core/adapters/numeric_adapter.py` | `int / float` |
| `NumericSequenceAdapter` | `core/adapters/numeric_sequence_adapter.py` | `list[float]` |
| `SNNAdapter` | `core/adapters/snn_adapter.py` | spike trains |
| `MultimodalFuser` | `core/adapters/multimodal_fuser.py` | mixed |
| `StreamProcessor` | `core/adapters/stream_processor.py` | streaming data |
| `ImageAdapter` | `core/adapters/image_adapter.py` | `np.ndarray` (image) |
| `AudioAdapter` | `core/adapters/audio_adapter.py` | audio signal |
| `VideoAdapter` | `core/adapters/video_adapter.py` | video frames |

**V4:** SNN auto-grounding — `register_concepts_from_memory()` called at
`NSCKSubstrate.__init__()` to bridge spike patterns to semantic predicates.

### Layer 2 — Memory

| Store | File | Backend |
|---|---|---|
| `SemanticMemory` | `core/memory/semantic_memory.py` | NetworkX KG + HNSW (default-on V4) + `_hot_cache` LRU 256 |
| `EpisodicMemory` | `core/memory/episodic_memory.py` | Python + `EpisodicMemoryConcurrent` (Rust) |
| `ProceduralMemory` | `core/memory/procedural_memory.py` | LSH 16-bit bucket index (V4), threshold 0.72 |
| `CrossModalAssociativeMemory` | `core/memory/cross_modal_associative_memory.py` | HV bind across modalities |
| `ConceptDriftDetector` | `core/memory/concept_drift_detector.py` | Tracks semantic drift |

### Layer 3 — Reasoning

| Module | File | Technique |
|---|---|---|
| `CausalReasoner` | `core/reasoning/causal_reasoning.py` | `CausalGraph` + intervention / counterfactual |
| `RuleLearner` | `core/reasoning/rule_learner.py` | EWC-aware rule pruning (V4) |
| `STRIPSPlanner` | `core/reasoning/planner.py` | Forward-search STRIPS planning |
| `AnalogyEngine` | `core/reasoning/analogy.py` | Structure-mapping over HV similarity |
| `SpatialReasoner` | `core/reasoning/spatial_reasoning.py` | Position codebook + relation inference |
| `MathReasoner` | `core/reasoning/math_reasoning.py` | FPE codebook + expression evaluation |
| `BeliefScorer` | `core/reasoning/belief_revision.py` | Bayesian belief update |
| `imagine_rollout()` | `core/reasoning/cognitive_engine.py` | Multi-step forward simulation (V4) |

### Layer 4 — Learning

| Module | File |
|---|---|
| `VSAHebbianLearner` | `core/learning/hebbian.py` |
| Q-learning policy | embedded in `CognitiveEngine` |
| `RuleLearner` (induction + EWC pruning V4) | `core/reasoning/rule_learner.py` |
| `ActiveInferenceLearner` | `core/learning/active_inference.py` |
| `CuriosityModule` | `core/learning/curiosity.py` |
| `ConformalWrapper` | `core/learning/conformal_wrapper.py` |
| `PatternGeneralizer` | `core/learning/pattern_generalizer.py` |
| `MetaLearner` | `core/learning/meta_learning.py` |

### Layer 5 — Language

| Module | File |
|---|---|
| `LeftCornerParser` | `core/language/parser.py` |
| `ConstructionMatcher` | `core/language/construction_grammar.py` |
| `VSANLUEngine` *(V4 new — primary)* | `core/language/vsa_nlu.py` |
| `NgramNLU` *(fallback)* | `core/language/ngram_nlu.py` |
| `NLGEngine` | `core/language/nlg.py` |
| `FluentResponseComposer` | `core/language/fluent_nlg.py` |
| `DialogueManager` | `core/language/dialogue_manager.py` |
| `BrillPosTagger` | `core/language/pos_tagger.py` |

### Layer 6 — Executive

| Module | File |
|---|---|
| `MetacognitiveEngine` | `core/cognitive/metacognition.py` |
| `SafetyGate` | `core/cognitive/metacognition.py` |
| `SelfModel` | `core/cognitive/self_model.py` |
| `TheoryOfMind` | `core/cognitive/theory_of_mind.py` |
| `EmotionSystem` | `core/cognitive/emotion_system.py` |
| `SafetyGateVerifier` | `core/cognitive/safety_verifier.py` |

---

## 3. Data Flow

```
 Raw input (str | dict | ndarray | PerceptPacket | …)
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │  NSCKSubstrate.process()                                 │
 │  ├─ SignalIngestor: normalise / window / route           │
 │  ├─ Detect modality → select Adapter                     │
 │  └─ Adapter.encode() → PerceptPacket                     │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │  CognitiveEngine.decide(percept)                         │
 │                                                          │
 │  0. V4 ProceduralMemory fast-path (LSH O(1))             │
 │     └─ if similarity(state_hv, skill.context_hv) ≥ 0.72  │
 │        → return cached (action, confidence) immediately  │
 │                                                          │
 │  1. Grounding: extract active_predicates                 │
 │                                                          │
 │  2. Build coalitions from 7 sources:                     │
 │     RULES | MEMORY | EXPLORATION | Q_LEARNING            │
 │     PLANNER | MATH | EXTERNAL                            │
 │                                                          │
 │  3. ActiveInference adjusts salience                     │
 │  4. Dual-Process routing                                 │
 │  5. GWT competition → winner broadcast                   │
 │  6. Safety gate check → possible veto                    │
 │  7. Return CognitiveState                                │
 └──────────────────────────────────────────────────────────┘
```

---

## 4. Global Workspace Theory (GWT)

File: `core/reasoning/global_workspace.py`

```python
@dataclass
class Coalition:
    source:            str
    content:           Any
    base_salience:     float
    relevance:         float
    affect_match:      float
    sender_confidence: float

    activation = base_salience + relevance + affect_match
                 + 0.5 × sender_confidence
```

### Mental Rehearsal (V4)

Before committing to the winner the workspace can *rehearse* via
`imagine_rollout()` (V4 multi-step):

1. `imagine_rollout(state_hv, action_sequence)` → `(total_reward, is_plan_safe)`
2. Compare predicted states against registered **danger vectors**.
3. If `similarity ≥ veto_threshold` (0.75) → **VETO**: halve salience, retry.
4. Unsafe plans have coalition salience halved (0.75 → 0.375).

---

## 5. Dual-Process Architecture

File: `core/reasoning/cognitive_engine.py`

```
               ┌─────────────────────────────┐
               │  System 1 — Fast (≤ 1 ms)   │
               │  ProceduralMemory (LSH O(1)) │
               │  modules: RULES,             │
               │           Q_LEARNING,        │
               │           EXPLORATION        │
               └─────────┬───────────────────┘
                         │ confidence ≥ 0.75?
                   yes ──┤── no
                   │     │
            return │     ▼
                   │  ┌─────────────────────────────┐
                   │  │  System 2 — Slow (> 1 ms)   │
                   │  │  adds: MEMORY (episodic)     │
                   │  │        PLANNER (STRIPS)      │
                   │  └─────────────────────────────┘
                   ▼
              GWT competition on full coalition set
```

---

## 6. VSA Engine

### 6.1 Representation

- **Dimensionality:** 10 240 bits (stored as 160 × `u64`).
- **Type:** Binary (Multiply-Add-Permute family).

| Op | Impl | Semantics |
|---|---|---|
| **Bind** | XOR | Role–filler binding |
| **Bundle** | Majority vote | Superposition / set union |
| **Bundle N-vectors** | `bundle_hvs()` Rust (V4) | Correct majority-vote over N vectors |
| **Permute** | Bit rotation | Sequence / order encoding |
| **Similarity** | 1 − (Hamming / dim) | Cosine-like ∈ [0, 1] |

### 6.2 Rust Backend — `hypervec_rs`

Compiled extension (`hypervec_rs.so`, 4.3 MB). Built with **maturin + PyO3**; uses **rayon**.

**Existing exports:** `HyperVector`, `HyperVectorRegistry`, `parallel_bundle`,
`SemanticMemoryConcurrent`, `EpisodicMemoryConcurrent`, `CognitiveWorkerPool`, `PersistentStorage`

**V4 new exports:**

| Function | Signature | Purpose |
|---|---|---|
| `bundle_hvs` | `(Vec<Vec<u8>>) -> Vec<u8>` | Correct N-vector majority-vote bundle |
| `lsh_bucket` | `(Vec<u64>, u32, u64) -> u32` | LSH bucket for ProceduralMemory O(1) |
| `spreading_activation_step` | `(activation, edges, decay, max_frontier) -> Dict` | One step of graph spreading activation |

---

## 7. SNN Subsystem

### 7.1 Neuron Model — LIF

Leaky Integrate-and-Fire with configurable `tau`, `v_thresh`, `v_reset`,
`refractory_period`, and `dt`.

### 7.2 Learning — STDP

Spike-Timing Dependent Plasticity via `StdpEngine` (Rust).

### 7.3 V4 SNN Auto-Grounding

`SNNPerceptionModule.register_concepts_from_memory(semantic_memory)` is called
at `NSCKSubstrate.__init__()`. Spike patterns now resolve to named predicates
via nearest-neighbor HV lookup (cleanup memory pattern).

### 7.4 Rust Backend — `snn_rs`

Compiled extension (`snn_rs.so`, 1.1 MB).

| Export | File |
|---|---|
| `SnnCore` | `lib.rs` |
| `LIFLayer` | `lib.rs` |
| `StdpEngine` | `lib.rs` |
| `HebbianMatrix` | `hebbian.rs` |
| `ConceptMapper` | `concept.rs` |
| `RateCoder` | `concept.rs` |

---

## 8. Memory Architecture

### 8.1 Semantic Memory *(V4 updated)*

File: `core/memory/semantic_memory.py`

- **Knowledge graph** backed by `networkx.DiGraph`.
- **HV index** via HNSW (default-on in V4; was behind config flag).
- **Hot cache:** `_hot_cache` LRU stores 256 most-recently-activated concept HVs.
  Updated by `spread_activation()` → `_update_hot_cache()`. Cache hits avoid
  full dict lookup.
- **Spreading activation** hot path uses `spreading_activation_step` Rust free-fn
  (V4) via `semantic_memory_shim.py`.

### 8.2 Episodic Memory

File: `core/memory/episodic_memory.py`

- Case-based: stores `(situation_hv, action, reward, predicates)`.
- Rust backend `EpisodicMemoryConcurrent` for lock-free concurrent reads.
- Default capacity: **10 000** episodes.

### 8.3 Procedural Memory *(V4 updated)*

File: `core/memory/procedural_memory.py`

```python
@dataclass
class Skill:
    context_hv:    HyperVector
    action:        str
    reward:        float
    access_count:  int   = 0
    last_accessed: float = now()
    label:         str | None = None
```

- **V4 fast-path:** 16-bit LSH bucket index → O(1) average candidate lookup.
  Familiarity threshold **0.72** (was 0.85).
- **V4 auto-population:** `CognitiveEngine.learn()` automatically caches skills
  on every positive-reward experience.
- LRU eviction when `len(skills) > max_skills` (default 500).

### 8.4 Cross-Modal Associative Memory

File: `core/memory/cross_modal_associative_memory.py`

Binds entity HVs from different modalities using `ModalityBinding` records.

---

## 9. Perception Pipeline

### 9.1 PerceptPacket — Universal Contract

File: `core/types/percept_packet.py`

```python
@dataclass(frozen=True)
class PerceptPacket:
    modality:          str
    timestamp:         float
    situation_hv:      HyperVector
    entity_hvs:        dict[str, HyperVector]
    relation_hvs:      list[tuple[s, p, o, HV]]
    active_predicates: frozenset[str]
    confidence:        float
    raw_state:         dict | None
    adapter_name:      str
    adapter_trace:     dict
```

---

## 10. Rust Acceleration

| Extension | File | Size | Default |
|---|---|---|---|
| `hypervec_rs.so` | `rust_vsa/` | 4.3 MB | **On** (V4) |
| `snn_rs.so` | `rust_snn/` | 1.1 MB | **On** (V4) |

As of V4, `NSCK_USE_RUST=1` is the default. If the `.so` files are missing,
pure-Python fallbacks activate transparently.

### Dispatch Logic

```python
# semantic_memory_shim.py — V4: captured at import, not per call
_rust_step_fn = getattr(hypervec_rs, 'spreading_activation_step', None)

def spread_activation_fast(concept_graph, ...):
    if _rust_step_fn is not None:
        return _rust_step_fn(...)
    return None  # Python fallback
```

### Performance

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| HyperVector XOR bind | ~120 µs | ~6 µs | **20×** |
| HyperVector similarity | ~18 µs | ~0.25 µs | **72×** |
| bundle_hvs (N=10, D=10240) | ~3.2 ms | ~0.18 ms | **18×** |
| lsh_bucket (n_bits=16) | ~0.9 ms | ~0.04 ms | **22×** |
| spreading_activation_step | ~12 ms | ~0.8 ms | **15×** |
| SNN step (1 024 neurons) | ~4 ms | ~0.8 ms | **5×** |

---

## 11. Configuration

File: `core/integration/config.py`

### Factory Methods

| Factory | Behaviour |
|---|---|
| `NSCKConfig.minimal()` | All V3+ flags off — original behaviour, zero extra cost |
| `NSCKConfig.research()` | All V3+V4 capability flags on — maximum feature exploration |
| `NSCKConfig.production()` | Stable + performance flags on, experimental off |
| `NSCKConfig.rich()` | Research flags + bridge perception mode (`perception_mode="bridge"`) |
| `NSCKConfig.for_scale(n_concepts)` | Auto-tunes `memory_capacity` and enables HNSW at ≥10 000 concepts |
| `NSCKConfig.seeded()` | Enables ConceptNet + auto-seeding from `.kp` pack |
| `NSCKConfig.transplant()` | All transplant flags enabled (`svd_factored`, 10 calibration epochs) |
| `NSCKConfig.v17()` | All V17 enrichment and glass-box tracing flags enabled |
| `NSCKConfig.from_env()` | Reads `NSCK_DEVICE`, `NSCK_LR`, `NSCK_MODEL_PATH` from environment |

### Key Fields

```python
# Hardware
device: str = "cpu"                              # "cpu" or "cuda"

# VSA
enable_dual_process: bool = False
system1_confidence_threshold: float = 0.75
enable_hnsw_index: bool = False                  # on in production() + research()

# Memory
memory_capacity: int = 2500
episode_capacity: int = 10000
procedural_familiarity_threshold: float = 0.72   # V4: was 0.85

# Rule Learning
min_rule_support: int = 5
min_rule_confidence: float = 0.7
enable_ewc: bool = False                         # EWC-aware pruning
ewc_lambda: float = 1000.0

# Transplant (V15)
enable_transplant: bool = False
transplant_strategy: str = "svd_factored"        # "random" | "learned" | "svd_factored"
transplant_calibration_epochs: int = 10
transplant_validation_threshold: float = 0.80
transplant_svd_components: int = 128
transplant_fpe_bins: int = 256
transplant_rho_threshold: float = 0.80           # Spearman ρ gate
transplant_recall10_threshold: float = 0.70      # Recall@10 gate
transplant_recall50_threshold: float = 0.60      # Recall@50 gate
transplant_ari_threshold: float = 0.65           # ARI gate

# V17 Enrichment
enable_causal_enrichment: bool = False
enable_perceptual_enrichment: bool = False
enable_semantic_enrichment: bool = False
enable_crossmodal_enrichment: bool = False
causal_enrichment_n_context: int = 3
perceptual_enricher_window: int = 8
semantic_enrichment_add_inverses: bool = True

# Seeding (V14)
enable_seeding: bool = False
seed_conceptnet_pack: str = ""
knowledge_packs: List[str] = []                  # paths to .kp files loaded at substrate init

# Rich Perception (V14)
perception_mode: str = "internal"                # "internal" | "bridge"
distillation_threshold: float = 0.80

# HNSW
semantic_hnsw_m: int = 16
semantic_hnsw_ef: int = 50
```

---

## 12. V15 Model Transplantation Pipeline

File: `core/transplant/` (5 modules)

V15 enables NSCK to absorb learned knowledge from any pre-trained neural network —
BERT, GPT, ViT, Whisper, CLIP, etc. — by converting their embedding spaces into
NSCK's native 10 240-bit binary hypervector space.

### Pipeline Stages

```
External Model (BERT / GPT / ViT / Whisper / CLIP / …)
           │
    ┌──────▼──────┐
    │  Harvester  │  core/transplant/harvester.py
    │             │  → HarvestResult(embeddings, vocab_mapping,
    └──────┬──────┘    model_type, embedding_dim, vocab_size, source_model)
           │
    ┌──────▼──────┐
    │  Projector  │  core/transplant/projector.py
    │             │  → Dict[token → HyperVector]  (codebook)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Calibrator │  core/transplant/calibrator.py  (optional STDP fine-tune)
    │             │  → CalibratedResult(codebook, snn_weights, quality_curve)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Validator  │  core/transplant/validator.py
    │             │  → TransplantReport(spearman_rho, recall@10, recall@50, ari, passed)
    └──────┬──────┘
           │ report.passed == True?
    ┌──────▼──────┐
    │  Integrate  │  inject into SemanticMemory + add similarity relations
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │    Save     │  persist as KnowledgePack (optional)
    └─────────────┘
```

### Harvesting Strategies

`ModelHarvester.harvest(model, method="auto")` supports four methods:

| Method | Probes | Use case |
|---|---|---|
| `"auto"` | LM attrs → vision attrs → encoder-decoder → named params | Default |
| `"embedding_layer"` | `model.embeddings`, `embed_tokens`, `wte`, `word_embeddings` | Transformer LM |
| `"named_params"` | Scans all 2-D weight matrices for largest (N > d) parameter | Any model |
| `"forward_hook"` | Registers hook on first detectable embedding layer, runs dummy forward pass | When weight access fails |

Model types detected: `"transformer_lm"`, `"transformer_vision"`, `"encoder_decoder"`, `"generic"`, `"error"`.

### Projection Strategies

| Strategy | Class | Algorithm | Similarity preservation |
|---|---|---|---|
| `"random"` | `RandomProjector` | Johnson–Lindenstrauss: `sign(e · P)`, `P ~ N(0,1)` | Spearman ρ ≈ 0.6 |
| `"learned"` | `LearnedProjector` | Gradient descent on `(cos_sim − hamming_sim)²` over sampled pairs | ρ ≈ 0.6 after 5 epochs |
| `"svd_factored"` *(default)* | `SVDFactoredProjector` | SVD to 128 components + FPE codebook encoding (matches `image_adapter.py` pattern) | Cluster structure preserved; continuous ρ ≈ 0.1 |

**SVDFactoredProjector seed formulas** (mirror `image_adapter.py` / `audio_adapter.py`):
- Codebook HVs: `HyperVector(i*31 + cb_seed_base)` where `cb_seed_base = j*31 + 7777` for component `j`, bin `i` — mirrors `image_adapter.py`'s `i*31 + 7777` with per-component offset
- Role HVs: `HyperVector((j*1013 + 5003) % 2**32)`

### STDP Calibration

`STDPCalibrator` refines the codebook by running pairs of embeddings through
`PythonSnnCore` and re-encoding via spike rates:

1. Simulate N random pairs through SNN for 10 steps with `learn=True`
2. Re-encode all tokens: `bits = (mean_firing_rate ≥ threshold)`
3. Measure Recall@10; if improvement > `min_improvement`: save new codebook
4. Early stopping after `patience` epochs without improvement

### Quality Metrics (TransplantReport)

| Metric | Threshold | Meaning |
|---|---|---|
| `spearman_rho` | ≥ 0.80 | Rank correlation between cosine and Hamming similarity on 2 000 random pairs |
| `recall_at_10` | ≥ 0.70 | Fraction of true top-10 cosine neighbours found in HV top-10 |
| `recall_at_50` | ≥ 0.60 | Same at top-50 |
| `ari` | ≥ 0.65 | Adjusted Rand Index between k-means clustering in embedding vs HV space |

Report also lists `worst_concepts`, `best_concepts` (per-token Recall@10), and
`calibration_quality_curve` (one float per STDP epoch).

### Integration

When `report.passed == True` and `cognitive_engine` is provided:
- All `codebook[token]` HVs are injected into `SemanticMemory.concept_hvs`
- Concept pairs with `similarity > 0.7` get a `"similar_to"` relation
- Capped at `_RELATION_TOKEN_CAP = 500` tokens to bound O(n²) cost

### NSCKSubstrate API

```python
config = NSCKConfig.transplant()
substrate = NSCKSubstrate(config)
report = substrate.transplant(
    model=bert_model,
    domain_name="language",
    strategy="svd_factored",       # default
    calibration_epochs=10,
    save_pack_path="my_domain.kp", # optional
)
print(report.passed, report.spearman_rho, report.recall_at_10)
```

---

## 13. V17 Enrichment Modules

V17 adds five enrichment modules and a glass-box tracer. Enabled individually
or all at once via `NSCKConfig.v17()`.

### CausalEnricher (`core/reasoning/causal_enricher.py`)

Enriches causal triples `(cause, effect, strength)` with semantic context:
1. Look up `n_context` nearest-neighbour concepts for each endpoint in `SemanticMemory`
2. Bundle co-occurring context into an enriched causal HV
3. Store enriched triple back for future queries
4. Return `CausalTrace(cause, effect, strength, context_concepts, enrichment_steps)`

Key methods: `enrich(cause, effect, strength)`, `enrich_chain(chain, base_strength)`

Enable: `NSCKConfig(enable_causal_enrichment=True, causal_enrichment_n_context=3)`

### PerceptualEnricher (`core/perception/perceptual_enricher.py`)

Post-processes `PerceptPacket` after the modality adapter, before GWT broadcast:
1. Normalise the situation HV (binarise near-threshold dimensions)
2. Add a temporal context HV derived from a rolling window (`window_size=8`) of recent packets
3. Tag packet with a confidence score (fraction of HV dims above threshold)
4. Return `EnrichedPercept(original_modality, confidence, temporal_ctx_available, tags)`

Enable: `NSCKConfig(enable_perceptual_enrichment=True, perceptual_enricher_window=8)`

### SemanticEnricher (`core/memory/semantic_enricher.py`)

Lightweight post-processing after concept insertion into `SemanticMemory`:
1. Infer missing property concepts via spreading activation
2. Add inverse relations automatically (`is_a → sub_class_of`, `has_part → part_of`, `causes → caused_by`, etc.)
3. Strengthen frequently co-queried concept pairs by rebundling their HVs
4. Return `EnrichmentReport(concepts_enriched, inverse_relations_added, bundles_strengthened)`

Enable: `NSCKConfig(enable_semantic_enrichment=True, semantic_enrichment_add_inverses=True)`

### GlassBoxTracer (`core/cognitive/glass_box_tracer.py`)

Captures a step-by-step human-readable trace of every decision:

```python
tracer = GlassBoxTracer()
with tracer.span("perception"):
    tracer.record("TextAdapter", "encoded 'hello world'", confidence=0.9)
with tracer.span("reasoning"):
    tracer.record("CognitiveEngine", "selected rule R42", confidence=0.75)

trace = tracer.export()   # → DecisionTrace
print(tracer.format_trace(trace))
```

Key types: `TraceEntry(span, module, message, confidence, timestamp_ms)`,
`DecisionTrace(decision_id, entries, start_ms, end_ms, elapsed_ms)`.

Key methods: `begin_decision()`, `end_decision()`, `record()`, `span()`,
`export()`, `last_trace()`, `history()`, `format_trace()`.

Enable: `NSCKConfig(enable_glass_box_tracer=True)` or via `NSCKConfig.v17()`

### CrossModalEnricher (`core/memory/crossmodal_enricher.py`)

Enriches `CrossModalAssociativeMemory` by:
1. Automatically linking modality-specific concept HVs to a shared cross-modal anchor
2. Computing pairwise similarity between anchors to detect cross-modal concept clusters
3. Generating a `CrossModalEnrichmentReport(anchors_created, links_added, clusters_detected)`

Enable: `NSCKConfig(enable_crossmodal_enrichment=True)`

### V17 Config Preset

```python
cfg = NSCKConfig.v17()
# Equivalent to:
cfg.enable_causal_enrichment = True
cfg.enable_perceptual_enrichment = True
cfg.enable_semantic_enrichment = True
cfg.enable_crossmodal_enrichment = True
cfg.enable_glass_box_tracer = True
```

---

## 14. Seeding Subsystem

Two seeding pathways exist: the **KnowledgeSeeder** (V4 — YAML domain kits)
and the **SemanticSeeder** (V14 — ConceptNet / BERT embedding seeding).

### KnowledgeSeeder (`core/bootstrap/knowledge_seeder.py`)

Declarative YAML-based domain bootstrapping:

```python
from python.core.bootstrap.knowledge_seeder import KnowledgeSeeder
seeder = KnowledgeSeeder()
n = seeder.seed_from_yaml("nsck/python/core/bootstrap/domain_kits/navigation.yaml", engine)
```

Domain YAML format:
```yaml
concepts:
  - name: room
    properties: {type: place}
relations:
  - [room, connects_to, corridor]
rules:
  - if: [at_location=room]
    then: navigate
    confidence: 0.9
```

Ships with: `bootstrap/domain_kits/navigation.yaml`, `bootstrap/domain_kits/scheduling.yaml`

### SemanticSeeder (`core/seeding/semantic_seeder.py`)

Higher-level seeder orchestrating ConceptNet and BERT seeding:

| Method | Description |
|---|---|
| `seed_from_conceptnet_pack(substrate, pack_path)` | Load a ConceptNet `.kp` pack into semantic memory |
| `seed_from_bert(substrate, model_name, n_concepts)` | Harvest BERT embeddings and transplant top-n concepts |
| `post_seed_enrich(substrate)` | Run SemanticEnricher after seeding |
| `full_seed(substrate, ...)` | Combined ConceptNet + BERT + post-enrich in one call |

### ConceptNetLoader (`core/seeding/conceptnet_loader.py`)

Parses ConceptNet CSV assertions into `(concept, relation, concept)` triples.
`load_from_csv(path, language="en", max_concepts=50000)` → iterable of triples.

### BertSeeder (`core/seeding/bert_seeder.py`)

`BertSeeder.seed(substrate, model_name, n_concepts)` uses `ModelHarvester` +
`SVDFactoredProjector` to inject BERT vocabulary HVs into semantic memory.

### KnowledgePack (`core/integration/knowledge_pack.py`) *(V14)*

Portable, serialisable domain bundle (gzip+JSON, `schema_version=2`):

```python
pack = KnowledgePack(name="my_domain")
pack.add_concept("room", {"type": "place"}, hv=room_hv)
pack.add_relation("room", "connects_to", "corridor")
pack.add_causal_link("fire", "evacuate", strength=0.95)
pack.save("my_domain.kp")

# Later:
loaded = KnowledgePack.load("my_domain.kp")
stats = loaded.inject_into(cognitive_engine)
# → {"concepts": 1, "relations": 1, "causal_links": 1}
```

HVs serialise as base64-encoded int8 arrays for security (no pickle).

---

## 15. V4 Additions

### VSANLUEngine (replaces NgramNLU as primary)

`VSANLUEngine` (`core/language/vsa_nlu.py`) classifies intent using
`DistributionalCodebook` word HVs and 7 intent prototype bundles:
`question`, `command`, `statement`, `greeting`, `farewell`, `exclamation`, `negation`.

`intent* = argmax_k cosine(encode_sentence(text), prototype_k)`

### KnowledgeSeeder

`KnowledgeSeeder` (`core/bootstrap/knowledge_seeder.py`) provides
declarative domain bootstrapping from YAML domain kits. Ships with
`navigation.yaml` and `scheduling.yaml`.

`engine.seed_domain("navigation.yaml")` → injects concepts, rules, causal edges,
and procedural skills in one call.

### LSH ProceduralMemory (threshold 0.72, 16-bit LSH)

`ProceduralMemory` replaces O(N) linear scan with 16-bit LSH bucket index.
`key = lsh_bucket(hv.bits, n_bits=16, seed=0xDEAD)` → 65 536 buckets.
Familiarity threshold: **0.72** (was 0.85).

### SemanticMemory Hot Cache

`SemanticMemory._hot_cache` (LRU, 256 entries) stores top-K concept HVs by
access frequency. Populated during `spread_activation()`. Cache hits avoid
full dict lookup.

### SNN Auto-Grounding

`NSCKSubstrate.__init__()` calls
`SNNPerceptionModule.register_concepts_from_memory(semantic_memory)`,
closing the SNN→predicate bridge. Spike patterns resolve to named predicates
via nearest-neighbor HV lookup.

### EWC-Aware Rule Pruning

`Rule` gains `gwt_win_count: int = 0` and `ewc_importance: float = 0.0`.
`decide()` increments `ewc_importance` when RULES coalition wins.
`prune_rules()` uses composite score `confidence × (1 + ewc_importance)`,
protecting frequently-winning rules from pruning.

---

## 16. Design Boundaries, Known Gaps, and Trade-offs

Not everything here is a "limitation" in the defect sense. This section
distinguishes **intentional design choices** from **genuine implementation gaps**
and **known trade-offs**, so readers understand *why* each behaviour exists.

---

### Design Boundaries (intentional architectural choices)

These are not defects. They are deliberate decisions that make NSCK what it is.

| Boundary | What it means | Why it's intentional |
|---|---|---|
| **VSA-based NLU only** | `VSANLUEngine` uses prototype HV matching — no semantic parsing, no neural language models built in | Preserves glass-box transparency; every NLU decision traces back to a cosine similarity over symbolic prototypes. No gradient tensors, no hidden states. The module's own docstring says "No gradients, no transformers." |
| **No built-in gradient learning** | NSCK has zero backprop-trained models internally | The architecture is explicitly neuro-symbolic: all learning is Hebbian, STDP, rule induction, or Q-learning. Gradient models are *intentionally external*. The V15 Transplant Pipeline is the designed bridge for injecting knowledge from gradient-trained models into NSCK's HV space. |
| **Rust is the fast path, Python is the correctness path** | The Rust `spreading_activation_step` computes per-edge weighted activation (relation weights + stigmergy are pre-computed by the shim and passed as edge weights). Both Rust and Python paths produce the same mathematical result — Rust is simply faster. | Performance without sacrificing semantics. The Python path remains as a verified fallback. |

---

### Known Implementation Gaps

These are genuine TODOs where a feature is incomplete or not yet wired up.

| Gap | Exact behaviour | What it means in practice |
|---|---|---|
| **SNN grounding fires once** | `SNNPerceptionModule.register_concepts_from_memory()` is called only during `NSCKSubstrate.__init__()` | Concepts added to `SemanticMemory` after substrate construction are not automatically registered in the SNN. Call `substrate.engine.snn_perception.register_concepts_from_memory(substrate.engine.semantic_memory)` manually after bulk concept additions. |
| **HNSW index is not persisted** | `SemanticMemory.save()` serialises concepts, HVs, and graph edges as gzip+JSON — the HNSW index object itself is not serialised | After `load()`, the HNSW index is rebuilt lazily on the first `add_concept()` call. This is automatic but adds latency after a cold load on large memories. |

---

### Scalability Characteristics

These depend on which backend is active and how the system is configured.

| Characteristic | Python-only path | With Rust backends (`NSCK_USE_RUST=1`, default) |
|---|---|---|
| **Semantic memory scale** | Practical ceiling ≈ 10 000 concepts (NetworkX graph + linear similarity scan) | `SemanticMemoryConcurrent` (Rust DashMap + Rayon parallel search) scales to 100 K+ concepts; `for_scale(n)` enables HNSW automatically at ≥ 10 000 |
| **Spreading activation throughput** | ~83 ops/s at 1 000 nodes / 5 000 edges | **V18:** `parallel_spread_activation` (all steps Rayon-parallel, weighted edges) — ~1 000 ops/s at 1K nodes, ~667 ops/s at 10K nodes (~16× faster). Rust DashMap is kept in sync at write time (no lazy O(N) scan per call). |
| **Parallel similarity search** | Linear scan — O(N·D) | Rust `parallel_semantic_search` via Rayon — O(N·D/cores) with HNSW ANN shortcut |

**V18 SemanticMemory write-time sync diagram:**

```
add_concept() / add_relation()
    ├── NetworkX DiGraph (graph queries, persistence, Python fallback)
    └── SemanticMemoryConcurrent DashMap (Rust, immediate O(1) mirror, spread_activation fast path)
              └── add_relation_weighted(src, tgt, w(rel))   ← typed relation weight stored
```

Previously, Rust sync happened lazily inside `spread_activation_fast()` (O(N) scan per call).
V18 pushes the sync to write time, so `spread_activation()` always finds a current DashMap.

---

### Known Trade-offs (strategy-specific, by design)

| Trade-off | Detail |
|---|---|
| **SVDFactoredProjector: cluster structure vs. continuous rank** | Reduces to 128 principal components then FPE-quantises each. Preserves cluster membership (ARI ≥ 0.65 target) but sacrifices continuous cosine rank (Spearman ρ ≈ 0.1 on random pairs). **RandomProjector** achieves ρ ≈ 0.6 if continuous rank matters more than interpretable components. See `TRANSPLANT_REPORT.md` for strategy comparison. |
| **STDP calibration is stochastic** | Random pair sampling in `STDPCalibrator` means two runs may produce slightly different codebooks. This is fundamental to STDP learning, not a bug. The `patience` parameter and `min_improvement` threshold bound the variance. Calibration is optional (`calibration_epochs=0` skips it). |
| **Transplant quality gates may reject** | `TransplantReport.passed` is `False` when any of ρ, Recall@10, Recall@50, or ARI fall below thresholds. Lower the thresholds in `NSCKConfig.transplant()` or switch to `"random"` strategy if the default `"svd_factored"` strategy fails a domain. |

---

*Document generated for NSCK V4, February 2026.*

---

## V18 · Semantic HV Bootstrap

### Problem Solved

Prior to V18, all word concept HVs were seeded from `hash(word) % 2**32` — purely random
10,240-bit vectors with no semantic geometry. This made every downstream VSA operation
(similarity search, spreading activation, analogy, episodic recall) semantically blind,
producing `sim("brain", "memory") ≈ 0.50` (random chance).

### Solution Architecture

V18 introduces `SemanticBootstrapper` — a tiered, zero-dependency fallback chain:

| Tier | Strategy | Source | Quality | Deps |
|------|----------|--------|---------|------|
| 0 | `prebuilt` | Pre-saved .pkl codebook | Highest | None |
| 1 | `bridge` | sentence-transformers/all-MiniLM-L6-v2 | High | sentence-transformers |
| 2 | `hf_corpus` | HuggingFace dataset download | Medium | internet |
| 3 | `corpus` | BUILTIN_CORPUS co-occurrence | Baseline | None |

`strategy="auto"` tries Tier 1, falls back to Tier 3 if sentence-transformers is not
installed.  The bootstrapper **never raises an exception** — worst case is Tier 3.

### Key Files

| File | Role |
|------|------|
| `python/core/language/semantic_bootstrap.py` | `SemanticBootstrapper` — tier controller |
| `python/core/language/cognitive_vocabulary.py` | 500+ word curated vocabulary |
| `python/core/language/distributional_semantics.py` | `build_default(strategy=)`, `build_semantic()` |

### Measured Improvements

| Metric | Before V18 | After V18 (bridge) |
|--------|-----------|-------------------|
| sim(brain, memory) | ~0.51 (random) | ~0.85 |
| sim(king, queen) | ~0.50 (random) | ~0.81 |
| SemanticMemory.query recall | ~random | semantically coherent |
| spread_activation quality | ~random | semantically guided |

### Usage

```python
from python.core.integration.config import NSCKConfig
from python.core.substrate import NSCKSubstrate

# Uses bootstrap automatically
config = NSCKConfig.semantic()
substrate = NSCKSubstrate(config)
```

Or standalone:

```python
from python.core.language.semantic_bootstrap import SemanticBootstrapper
cb = SemanticBootstrapper.build_codebook(strategy="auto")
sim = cb.similarity("brain", "memory")   # → semantically meaningful
```
