# NSCK Architecture — V4 (Neuro-Symbolic Cognitive Kernel)

> **Version 4 (V4) — February 2026**
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
12. [V4 Additions](#12-v4-additions)
13. [Limitations](#13-limitations)

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

| Factory | Behaviour |
|---|---|
| `NSCKConfig.minimal()` | All V3+ flags off — original behaviour |
| `NSCKConfig.research()` | All capability flags on |
| `NSCKConfig.production()` | Stable + performance flags on, experimental off |
| `NSCKConfig.from_env()` | Reads `NSCK_DEVICE`, `NSCK_LR`, … from environment |

Critical fields (V4 defaults):

```python
enable_dual_process: bool = False
system1_confidence_threshold: float = 0.75
procedural_familiarity_threshold: float = 0.72  # V4: was 0.85
memory_capacity: int = 2500
episode_capacity: int = 10000
enable_hnsw_index: bool = True   # V4: default-on
hot_cache_size: int = 256        # V4 new
```

---

## 12. V4 Additions

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

## 13. Limitations

| Area | Limitation |
|---|---|
| **NLU** | VSA prototype matching — no semantic parsing or transformers |
| **Deep learning** | Zero gradient-based models; no CNNs, transformers, or embeddings |
| **SNN grounding** | Fires at startup only; new concepts added after `__init__()` require manual re-grounding |
| **HNSW persistence** | Index not persisted across save/load cycles — rebuilds on next `add_concept()` |
| **Scale** | Semantic memory practical ceiling ≈ **10 000** concepts |
| **Rust step dispatch** | `spreading_activation_step` uses uniform decay; full relation-weighted spreading uses Python path |

---

*Document generated for NSCK V4, February 2026.*
