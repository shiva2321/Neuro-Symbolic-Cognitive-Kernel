# NSCK Architecture — Neuro-Symbolic Cognitive Kernel

> **Version 13 (V13)** — February 2026
>
> This document is the definitive reference for NSCK's internal design.
> It covers the seven-layer architecture, data-flow pipeline,
> Global Workspace Theory (GWT) competition, dual-process routing,
> VSA / SNN subsystems, memory architecture, and Rust acceleration.

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
12. [Limitations](#12-limitations)

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
Layer 6 — Executive      Metacognition · SelfModel · TheoryOfMind · Safety
Layer 5 — Language        Parser · ConstructionGrammar · NgramNLU · NLG · Dialogue
Layer 4 — Learning        Hebbian · Q-Learning · RuleInduction · ActiveInference · Curiosity · Conformal
Layer 3 — Reasoning       Causal · Rules · Planning · Analogy · Spatial · Math · Beliefs
Layer 2 — Memory          Semantic (KG + HNSW) · Episodic (Rust) · Procedural · CrossModal
Layer 1 — Perception      10 Adapters · SignalIngestor · UniversalHVEncoder · SNN
Layer 0 — Substrate       VSA Engine (10 240-bit HVs) · GWT · Dual-Process · Rust backends
```

### Layer 0 — Substrate

**Purpose.** Provide the fundamental computational primitives that every
higher layer depends on: hypervector algebra, the GWT broadcast bus,
and the dual-process router.

| Component | File | Role |
|---|---|---|
| `HyperVector` | `core/vsa/hypervec_shim.py` → `hypervec_rs` | 10 240-bit binary VSA vector |
| `FHRRVector` | `core/vsa/fhrr.py` | Complex-phasor VSA variant |
| `GlobalWorkspace` | `core/reasoning/global_workspace.py` | Coalition competition + broadcast |
| `NSCKConfig` | `core/integration/config.py` | Central configuration (dual-process flag, thresholds) |

**Interfaces ↑ Layer 1:** Perception adapters call `HyperVector.bind / bundle`
to build `PerceptPacket.situation_hv`.

---

### Layer 1 — Perception

**Purpose.** Convert arbitrary raw input into a uniform `PerceptPacket`
that downstream layers can consume without knowing the original modality.

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

**V13 additions:**
- `SignalIngestor` (`core/perception/signal_ingestor.py`) — normalisation,
  windowing, and routing of raw signals before adapter dispatch.
- `UniversalHVEncoder` (`core/vsa/universal_hv_encoder.py`) — signal-agnostic
  statistical encoder that works across all modalities.

**Interfaces ↑ Layer 2:** `PerceptPacket.entity_hvs` feed into
`SemanticMemory` look-ups; ↑ Layer 3: `active_predicates` feed the rule engine.

---

### Layer 2 — Memory

**Purpose.** Persistent and working memory stores that ground symbols
and recall past experience.

| Store | File | Backend |
|---|---|---|
| `SemanticMemory` | `core/memory/semantic_memory.py` | NetworkX KG + `_NSWIndex` (HNSW when available) |
| `EpisodicMemory` | `core/memory/episodic_memory.py` | Python + `EpisodicMemoryConcurrent` (Rust) |
| `ProceduralMemory` | `core/memory/procedural_memory.py` | In-memory skill cache (`Skill` dataclass) |
| `CrossModalAssociativeMemory` | `core/memory/cross_modal_associative_memory.py` | HV bind across modalities |
| `ConceptDriftDetector` | `core/memory/concept_drift_detector.py` | Tracks semantic drift |

**Interfaces ↑ Layer 3:** Episodic recall proposes `MEMORY` coalitions;
semantic similarity underpins analogy; ↓ Layer 1: perception writes new
episodes.

---

### Layer 3 — Reasoning

**Purpose.** Symbolic and hybrid inference over grounded predicates and
hypervectors.

| Module | File | Technique |
|---|---|---|
| `CausalReasoner` | `core/reasoning/causal_reasoning.py` | `CausalGraph` + intervention / counterfactual |
| `RuleLearner` | `core/reasoning/rule_learner.py` | Predicate-condition → action rules |
| `STRIPSPlanner` | `core/reasoning/planner.py` | Forward-search STRIPS planning |
| `AnalogyEngine` | `core/reasoning/analogy.py` | Structure-mapping over HV similarity |
| `SpatialReasoner` | `core/reasoning/spatial_reasoning.py` | Position codebook + relation inference |
| `MathReasoner` | `core/reasoning/math_reasoning.py` | FPE codebook + expression evaluation |
| `BeliefScorer` | `core/reasoning/belief_revision.py` | Bayesian belief update |

**Interfaces ↑ Layer 4:** Rule learner invokes Hebbian strengthening;
↑ Layer 6: Safety gate may veto any proposed action.

---

### Layer 4 — Learning

**Purpose.** Online adaptation without gradient-based training.

| Module | File |
|---|---|
| `VSAHebbianLearner` | `core/learning/hebbian.py` |
| Q-learning policy | embedded in `CognitiveEngine` |
| `RuleLearner` (induction) | `core/reasoning/rule_learner.py` |
| `ActiveInferenceLearner` | `core/learning/active_inference.py` |
| `CuriosityModule` | `core/learning/curiosity.py` |
| `ConformalWrapper` | `core/learning/conformal_wrapper.py` |
| `PatternGeneralizer` | `core/learning/pattern_generalizer.py` |
| `MetaLearner` | `core/learning/meta_learning.py` |

**Interfaces ↓ Layer 3:** Active inference adjusts coalition salience;
curiosity injects `EXPLORATION` proposals.

---

### Layer 5 — Language

**Purpose.** Natural-language understanding and generation (rule-based /
n-gram; no deep learning).

| Module | File |
|---|---|
| `LeftCornerParser` | `core/language/parser.py` |
| `ConstructionMatcher` | `core/language/construction_grammar.py` |
| `NgramNLU` | `core/language/ngram_nlu.py` |
| `NLGEngine` | `core/language/nlg.py` |
| `FluentResponseComposer` | `core/language/fluent_nlg.py` |
| `DialogueManager` | `core/language/dialogue_manager.py` |
| `BrillPosTagger` | `core/language/pos_tagger.py` |

**Interfaces ↓ Layer 1:** `TextAdapter` uses language module for predicate
extraction; ↑ Layer 6: dialogue state informs metacognition.

---

### Layer 6 — Executive

**Purpose.** Self-monitoring, safety enforcement, and theory-of-mind.

| Module | File |
|---|---|
| `MetacognitiveEngine` | `core/cognitive/metacognition.py` |
| `SafetyGate` | `core/cognitive/metacognition.py` |
| `SelfModel` | `core/cognitive/self_model.py` |
| `TheoryOfMind` | `core/cognitive/theory_of_mind.py` |
| `EmotionSystem` | `core/cognitive/emotion_system.py` |
| `SafetyGateVerifier` | `core/cognitive/safety_verifier.py` |

**Interfaces ↓ Layer 3:** Safety gate can veto any coalition winner;
self-model tracks confidence calibration history.

---

## 3. Data Flow

```
 Raw input (str | dict | ndarray | PerceptPacket | …)
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │  NSCKSubstrate.process()                                 │
 │  ├─ SignalIngestor (V13): normalise / window / route     │
 │  ├─ Detect modality → select Adapter                     │
 │  └─ Adapter.encode() → PerceptPacket                     │
 │     (situation_hv, entity_hvs, active_predicates, …)     │
 └──────────────────────────────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │  CognitiveEngine.decide(percept)                         │
 │                                                          │
 │  0. V13 ProceduralMemory fast-path check                 │
 │     └─ if similarity(state_hv, skill.context_hv) ≥ 0.85 │
 │        → return cached (action, confidence) immediately  │
 │                                                          │
 │  1. Grounding: extract active_predicates                 │
 │                                                          │
 │  2. Build coalitions from 7 sources:                     │
 │     ┌────────────────────────────────────────────────┐   │
 │     │ RULES        — matching predicate rules        │   │
 │     │ MEMORY       — episodic recall (System 2)      │   │
 │     │ EXPLORATION  — curiosity-driven action          │   │
 │     │ Q_LEARNING   — reward-based policy              │   │
 │     │ PLANNER      — STRIPS goal-directed (System 2)  │   │
 │     │ MATH         — MathReasoner proposals           │   │
 │     │ EXTERNAL     — metacognition / SNN coalitions   │   │
 │     └────────────────────────────────────────────────┘   │
 │                                                          │
 │  3. ActiveInference adjusts salience:                    │
 │     c.base_salience += weight × (0.5 − free_energy)     │
 │                                                          │
 │  4. Dual-Process routing (§5)                            │
 │                                                          │
 │  5. GWT competition → winner broadcast (§4)              │
 │                                                          │
 │  6. Safety gate check → possible veto                    │
 │                                                          │
 │  7. Return CognitiveState                                │
 │     (chosen_action, confidence, explanation,             │
 │      kle_uncertainty, uncertainty_bounds, system_used)   │
 └──────────────────────────────────────────────────────────┘
```

**CognitiveState** fields (abridged):

| Field | Type | Meaning |
|---|---|---|
| `chosen_action` | `str` | Winning action label |
| `confidence` | `float` | Normalised activation of winner |
| `explanation` | `Explanation` | Glass-box trace |
| `system_used` | `str` | `"system_1"` or `"system_2"` |
| `kle_uncertainty` | `float` | Shannon entropy of activation distribution |
| `uncertainty_bounds` | `tuple` | Conformal prediction interval *(V13)* |
| `active_predicates` | `list[str]` | Grounded predicates for this tick |

---

## 4. Global Workspace Theory (GWT)

File: `core/reasoning/global_workspace.py`

### 4.1 Coalition Structure

```python
@dataclass
class Coalition:
    source:            str    # e.g. "RULES", "Q_LEARNING"
    content:           Any    # proposed action / information
    base_salience:     float  # intrinsic loudness  [0–1]
    relevance:         float  # context match        [0–1]
    affect_match:      float  # drive/emotion match  [0–1]
    sender_confidence: float  # proposer confidence   [0–1]

    activation = base_salience + relevance + affect_match
                 + 0.5 × sender_confidence          # max ≈ 3.5
```

`_effective_activation` adds a **mission-focus bonus** (0.0–0.2) on top,
biasing competition toward the current goal.

### 4.2 Competition

1. All coalitions ranked by `_effective_activation`.
2. Winner takes the workspace; its `content` is broadcast.
3. Registered `WorkspaceModule` instances receive `receive_broadcast(content)`.

### 4.3 KLE Uncertainty

```
activations  = [eff_act(c) for c in proposals]
probs        = normalise(activations)            # sum-to-1
KLE          = − Σ pᵢ · log(pᵢ)                 # Shannon entropy
```

High KLE → multiple strong competitors → uncertain decision.
Returned in `CognitiveState.kle_uncertainty`.

### 4.4 Mental Rehearsal

Before committing to the winner the workspace can *rehearse*:

1. `WorldModel.imagine(state_hv, action_hv)` → predicted next state.
2. Compare predicted state against registered **danger vectors**.
3. If `similarity ≥ veto_threshold` (0.75) → **VETO**: halve salience,
   remove candidate, retry with next-best (up to 3 cycles).
4. If all candidates vetoed → emergency `ACTION_STAY`.

Veto events are logged in `rehearsal_log` for telemetry.

---

## 5. Dual-Process Architecture

File: `core/reasoning/cognitive_engine.py`

```
               ┌─────────────────────────────┐
               │  System 1 — Fast (≤ 1 ms)   │
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

- **System 1** builds *quick* coalitions (rules, Q-table, exploration),
  runs a preliminary GWT competition, and normalises activation
  (`activation / 2.0`).  If confidence ≥ `system1_confidence_threshold`
  (default **0.75**), the answer is returned *without* consulting
  episodic memory or the planner.

- **System 2** adds MEMORY and PLANNER coalitions and runs full
  competition.

- Controlled by `NSCKConfig.enable_dual_process` (default `False`).
  `CognitiveState.system_used` records which path fired.

---

## 6. VSA Engine

### 6.1 Representation

- **Dimensionality:** 10 240 bits (stored as 160 × `u64`).
- **Type:** Binary (Multiply-Add-Permute family).
- **Operations:**

| Op | Impl | Semantics |
|---|---|---|
| **Bind** | XOR | Role–filler binding |
| **Bundle** | Majority vote | Superposition / set union |
| **Permute** | Bit rotation | Sequence / order encoding |
| **Similarity** | 1 − (Hamming / dim) | Cosine-like ∈ [0, 1] |

### 6.2 FHRR Mode

`FHRRVector` (`core/vsa/fhrr.py`) uses **complex-phasor** hypervectors
for continuous-valued binding (element-wise multiply on the unit circle).

### 6.3 Rust Backend — `hypervec_rs`

Compiled extension (`hypervec_rs.so`, 4.3 MB).  Built with
**maturin + PyO3**; uses **rayon** for data-parallelism.

**Key exports:**

| Class / Function | Module |
|---|---|
| `HyperVector` | `lib.rs` |
| `HyperVectorRegistry` | `concurrent.rs` |
| `parallel_bundle` | `concurrent.rs` |
| `batch_parallel_similarity_search` | `concurrent.rs` |
| `SemanticMemoryConcurrent` | `semantic.rs` |
| `EpisodicMemoryConcurrent` | `episodic.rs` |
| `CognitiveWorkerPool` | `worker_pool.rs` |
| `PersistentStorage` | `persistence.rs` |

**Fallback:** When `.so` is unavailable, `hypervec_shim.py` transparently
loads the pure-Python `HyperVectorPy` from `hypervec_py.py`.
Benchmark: Rust path is **5–85× faster** depending on operation.

---

## 7. SNN Subsystem

### 7.1 Neuron Model — LIF

Leaky Integrate-and-Fire with configurable `tau`, `v_thresh`,
`v_reset`, `refractory_period`, and `dt`.

### 7.2 Learning — STDP

Spike-Timing Dependent Plasticity: potentiation when pre fires before
post; depression when post fires before pre.

### 7.3 Coding Modes

| Mode | Class | Encoding |
|---|---|---|
| **Rate** | `RateCoder` | Value → firing rate |
| **Temporal** | `TemporalCoder` (Python) | Value → precise spike time |

### 7.4 Rust Backend — `snn_rs`

Compiled extension (`snn_rs.so`, 1.1 MB).  PyO3 + rayon.

| Export | File |
|---|---|
| `SnnCore` | `lib.rs` |
| `LIFLayer` | `lib.rs` |
| `StdpEngine` | `lib.rs` |
| `HebbianMatrix` | `hebbian.rs` |
| `ConceptMapper` | `concept.rs` |
| `RateCoder` | `concept.rs` |

### 7.5 VSA ↔ SNN Bridge

`core/perception/vsa_snn_bridge.py` converts between spike patterns
and hypervectors (`SpikeEncoding`, `HVtoSpikeDecoder`), enabling
Layer 0 (VSA) and Layer 1 (SNN) to share representations.

---

## 8. Memory Architecture

### 8.1 Semantic Memory

File: `core/memory/semantic_memory.py`

- **Knowledge graph** backed by `networkx.DiGraph`.
- **HV index** via `_NSWIndex` (navigable small-world); upgrades to
  HNSW when the library is available.
- **Concept decay:** unused concepts lose activation over time.
- **Stigmergy:** frequently co-accessed concepts strengthen links.
- **Prototype building:** bundle exemplar HVs into a single prototype.

### 8.2 Episodic Memory

File: `core/memory/episodic_memory.py`

- Case-based: stores `(situation_hv, action, reward, predicates)`.
- Rust backend `EpisodicMemoryConcurrent` for lock-free concurrent
  reads.
- Similarity search: best-match recall by Hamming distance.
- Default capacity: **10 000** episodes (configurable).

### 8.3 Procedural Memory *(V13)*

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

- **Fast-path:** if `similarity(query_hv, skill.context_hv) ≥ 0.85`,
  return cached action immediately (short-circuits `CognitiveEngine.decide`).
- LRU eviction when `len(skills) > max_skills` (default 500).

### 8.4 Cross-Modal Associative Memory *(V13)*

File: `core/memory/cross_modal_associative_memory.py`

Binds entity HVs from different modalities (e.g. visual "cup" ↔
auditory "cup") using `ModalityBinding` records.

---

## 9. Perception Pipeline

### 9.1 PerceptPacket — Universal Contract

File: `core/types/percept_packet.py`

```python
@dataclass(frozen=True)
class PerceptPacket:
    modality:          str                          # "text", "dict", "image", …
    timestamp:         float
    situation_hv:      HyperVector                  # bundled scene HV
    entity_hvs:        dict[str, HyperVector]       # named entities
    relation_hvs:      list[tuple[s, p, o, HV]]     # (subj, pred, obj, triple_hv)
    active_predicates: frozenset[str]               # grounded symbols
    confidence:        float                        # [0, 1]
    raw_state:         dict | None                  # original input
    adapter_name:      str
    adapter_trace:     dict                         # adapter metadata
```

All 10 adapters produce exactly this type.  Downstream code never
inspects `modality` — it only consumes HVs and predicates.

### 9.2 V13 Additions

| Component | File | Role |
|---|---|---|
| `SignalIngestor` | `core/perception/signal_ingestor.py` | Pre-processing: normalisation, windowing, type routing |
| `UniversalHVEncoder` | `core/vsa/universal_hv_encoder.py` | Statistical features → HV, modality-agnostic |

---

## 10. Rust Acceleration

| Extension | File | Size | Crate deps |
|---|---|---|---|
| `hypervec_rs.so` | `rust_vsa/` | 4.3 MB | pyo3, rayon, dashmap, parking_lot, crossbeam, tokio |
| `snn_rs.so` | `rust_snn/` | 1.1 MB | pyo3, rayon, rand, parking_lot |

Both are built with `maturin develop --release` and loaded at import
time.  If the `.so` is missing, pure-Python fallbacks activate
transparently — no user action required.

### Performance (representative benchmarks)

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| `xor` 10 240-bit | ~12 µs | ~0.14 µs | **85×** |
| `similarity` | ~18 µs | ~0.25 µs | **72×** |
| `bundle` (pair) | ~30 µs | ~0.45 µs | **67×** |
| SNN `step` 1 024 neurons | ~4 ms | ~0.8 ms | **5×** |

---

## 11. Configuration

File: `core/integration/config.py`

`NSCKConfig` is a `@dataclass` with 75+ fields covering hardware,
hyper-parameters, feature flags, and capacity limits.

| Factory | Behaviour |
|---|---|
| `NSCKConfig.minimal()` | All V3+ flags off — original behaviour |
| `NSCKConfig.research()` | All capability flags on |
| `NSCKConfig.production()` | Stable + performance flags on, experimental off |
| `NSCKConfig.from_env()` | Reads `NSCK_DEVICE`, `NSCK_LR`, … from environment |

Critical fields:

```python
enable_dual_process: bool = False
system1_confidence_threshold: float = 0.75
memory_capacity: int = 2500
episode_capacity: int = 10000
vsa_strength: float = 5.0
confidence_threshold: float = 0.6
```

---

## 12. Limitations

| Area | Limitation |
|---|---|
| **NLU** | N-gram heuristics and pattern matching — no semantic parsing or transformers |
| **Deep learning** | Zero gradient-based models; no CNNs, transformers, or embeddings |
| **SNN FFI** | Rust ↔ Python overhead dominates at fine per-neuron granularity |
| **Scale** | Semantic memory practical ceiling ≈ **10 000** concepts |
| **Media adapters** | Image / audio / video adapters are schematic stubs, not production-grade feature extractors |
| **Cold start** | Analogy and causal reasoning require a populated memory to be useful |

---

*Document generated for NSCK V13, February 2026.*

---

## V17 — Enrichment Layer & Glass-Box Tracing

V17 adds an **Enrichment Layer** that sits between adapters/memory and the GWT
broadcast stage, plus a **GlassBoxTracer** for full decision observability.

### Enrichment Layer

```
PerceptPacket → PerceptualEnricher → EnrichedPercept → GWT
CausalTriple  → CausalEnricher    → CausalTrace     → Reasoning
Concept/Rel   → SemanticEnricher  → EnrichmentReport → SemanticMemory
ModalConcept  → CrossModalEnricher→ CMAnchor         → CrossModalMemory
```

| Module | Location | Purpose |
|--------|----------|---------|
| `CausalEnricher` | `reasoning/causal_enricher.py` | Enrich causal chains |
| `PerceptualEnricher` | `perception/perceptual_enricher.py` | Temporal ctx + confidence |
| `SemanticEnricher` | `memory/semantic_enricher.py` | Inverse relations + coquery |
| `CrossModalEnricher` | `memory/crossmodal_enricher.py` | Anchor linking + clusters |
| `GlassBoxTracer` | `cognitive/glass_box_tracer.py` | Step-by-step decision trace |

### GlassBoxTracer

Every module can call `tracer.record(module, message, confidence)` inside a
`tracer.span(name)` context to append a `TraceEntry` to the active
`DecisionTrace`. Completed traces are archived in a rolling history.

*Document updated for NSCK V17, April 2026.*
