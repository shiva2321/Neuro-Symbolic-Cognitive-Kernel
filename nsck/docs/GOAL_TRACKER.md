# NSCK Goal Tracker — AGI Vision vs Implementation

> **Version 14 (V14)** — Updated February 2026
>
> This is a living document that tracks the creator's original AGI vision against
> the actual state of implementation. It is meant to be honest, not aspirational.
> Status is assessed against what is provably running in the test suite, not
> against what is planned or theoretically possible.

---

## Creator's Vision Statement

> *"An architectural substrate which is inherently capable of learning, reasoning,
> remembrance, recall, generalization of learnings, cross-domain knowledge transfer
> after generalizations, lifelong learning and improvement, multi-modal input
> processing, glass-box transparency in its decision making, efficiency so as to
> run on commodity hardware without massive compute requirements, and developer
> extensibility so that domain experts can plug in their knowledge without needing
> AI expertise."*

This statement defines ten capabilities. Each is tracked below.

---

## Capability Status Table

| # | Capability | Status | Evidence |
|---|---|---|---|
| 1 | **Learning** | ✅ Implemented | Hebbian, Q-learning, STDP, rule induction, active inference, online |
| 2 | **Reasoning** | ✅ Implemented | Causal, rules, STRIPS planning, analogy, spatial, math, belief revision |
| 3 | **Remembrance & Recall** | ✅ Implemented | Episodic + semantic memory; HV similarity recall; Rust lock-free backend |
| 4 | **Generalization** | ✅ Implemented | `PatternGeneralizer` (V13) clusters HVs into abstract prototypes |
| 5 | **Cross-domain Transfer** | ⚠️ Partial | Analogy engine enables structural transfer; no automatic domain bridge |
| 6 | **Lifelong Learning** | ✅ Implemented | No catastrophic forgetting; tenure-based rule decay; `sleep()` consolidation |
| 7 | **Multi-modal Input** | ✅ Implemented | 10 adapters + V14 rich adapters (text/image/audio bridge); `PerceptPacket` contract |
| 8 | **Glass-box Transparency** | ✅ Implemented | Every decision yields `Explanation` + `ThoughtTrace`; zero hidden layers |
| 9 | **Efficiency** | ✅ Implemented | Rust VSA/SNN backends (5–85× speedup); runs on laptop CPU, no GPU required |
| 10 | **Developer Extensibility** | ✅ Implemented | `register_task()`, `register_encoder()`, `KnowledgePack` (V14) |

---

## Per-Capability Notes

### 1. Learning

NSCK uses five distinct online learning mechanisms simultaneously. No
gradient descent or backpropagation is used anywhere.

| Mechanism | Module | Notes |
|---|---|---|
| Hebbian weight update | `learning/hebbian.py` | Oja's rule; strengthens co-activated concepts |
| Q-learning | `reasoning/cognitive_engine.py` | Tabular policy; `feedback()` updates Q-values |
| STDP | `rust_snn/src/lib.rs` | Spike-timing plasticity; Rust-accelerated |
| Rule induction | `reasoning/rule_learner.py` | Observes predicate co-occurrences; infers If→Then rules |
| Active inference | `learning/active_inference.py` | Minimises free energy via salience adjustment |

**V14 addition:** `PerceptionDistiller` tracks whether internal encoding
quality has converged to bridge encoding quality per modality, enabling
the system to graduate away from neural bridge models when pure encoding
is sufficient.

**Honest assessment:** Learning is online and continual, but it is not
deep learning. NSCK cannot learn visual features, syntactic structures,
or complex functions from raw data the way a neural network can. Its
strengths are symbolic rule learning and episodic association.

---

### 2. Reasoning

Seven reasoning modules covering causal, deductive, spatial, mathematical,
and analogical inference. All are glass-box and produce auditable traces.

**Honest assessment:** Reasoning quality is bounded by what is in memory
and what rules have been induced. Cold-start performance is weak. The
system cannot parse novel mathematics or natural language reliably — it
uses n-gram heuristics, not semantic parsing.

---

### 3. Remembrance & Recall

- **Episodic memory:** stores `(situation_hv, action, reward)` tuples;
  recall by Hamming similarity. Capacity: 10 000 episodes.
- **Semantic memory:** knowledge graph (NetworkX) + navigable small-world
  index; Rust `SemanticMemoryConcurrent` for concurrent reads.
- **Procedural memory (V13):** `Skill` cache with 0.85-similarity fast-path.

**V14 addition:** `semantic_memory_shim.py` wires spreading activation to the
Rust `SemanticMemoryConcurrent` DashMap backend when available, improving
concurrent read performance at scale.

---

### 4. Generalization

`PatternGeneralizer` (V13) clusters HVs by similarity ≥ 0.7 and builds
majority-vote prototype vectors. Abstraction level scales with cluster size.

**Honest assessment:** Generalization is statistical clustering over HVs,
not the kind of compositional generalisation that neural networks achieve on
large corpora. It works well for structured domains; it is weak on perceptual
diversity.

---

### 5. Cross-domain Transfer

`AnalogyEngine` maps source-domain structures onto target-domain structures
via HV similarity. After a pattern is generalized into a prototype, it can be
matched against a new domain if there is structural overlap.

**Honest assessment:** Transfer is not automatic. It requires explicit domain
registration and sufficient overlap in predicate vocabulary. NSCK does not
perform zero-shot transfer to entirely novel domains.

---

### 6. Lifelong Learning

- No weight matrices to overwrite → no catastrophic forgetting by design.
- Episodic memory is a ring buffer; oldest episodes are evicted when full
  (capacity 10 000), not overwritten randomly.
- Rule tenure: rules that are not reinforced decay; rules in use strengthen.
- `sleep()` triggers offline consolidation: prototype building, drift check,
  episode replay.

---

### 7. Multi-modal Input

V14 extends V13's perception layer with three rich adapters:

| Adapter | Bridge model (optional) | Fallback |
|---|---|---|
| `RichTextAdapter` | `all-MiniLM-L6-v2` (sentence-transformers) | Char-ngram codebook |
| `RichImageAdapter` | `mobilenet_v3_small` (timm, `pretrained=False`) | Classical CV features |
| `RichAudioAdapter` | `whisper-tiny` | Classical DSP (MFCC-like) |

All three are gated behind `perception_mode` (`"pure"` / `"bridge"` /
`"hybrid"`). When the optional dependency is absent, the system falls back
silently to the pure VSA path. The `PerceptPacket` contract is unchanged.

---

### 8. Glass-box Transparency

Every call to `decide()` / `process()` returns a `CognitiveState` containing:

- `explanation` — human-readable string listing the winning coalition and
  its source.
- `ThoughtTrace` — structured record of every coalition considered, their
  salience scores, and the GWT competition result.
- `active_predicates` — the grounded symbols that drove the decision.

No latent activations, no weight matrices — every decision is traceable
to a rule, an episode, or a Q-value entry.

---

### 9. Efficiency

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| XOR (10 240 bits) | ~12 µs | ~0.14 µs | 85× |
| Similarity | ~18 µs | ~0.25 µs | 72× |
| Bundle (pair) | ~30 µs | ~0.45 µs | 67× |
| SNN step (1 024 neurons) | ~4 ms | ~0.8 ms | 5× |

Scale benchmarks (V14, pure Python, no Rust):

| Concepts | Spreading activation | Query (k=10) |
|---|---|---|
| 100 | < 1 ms | < 1 ms |
| 500 | ~2 ms | ~2 ms |
| 1 000 | ~5 ms | ~4 ms |
| 5 000 | ~25 ms | ~15 ms |

---

### 10. Developer Extensibility

```python
# Register a custom task domain
substrate.register_task("medical_triage")

# Register a custom perception adapter
substrate.register_encoder("ecg", MyECGAdapter())

# Load and inject a knowledge pack (V14)
from python.core.integration.knowledge_pack import KnowledgePack
pack = KnowledgePack.load("nsck/data/knowledge_packs/medicine.gz")
pack.inject_into(engine)
```

`NSCKConfig` exposes 75+ fields controllable at construction time or via
`NSCKConfig.from_env()`. No AI expertise is required to add new tasks.

---

## What NSCK Is Not

To be honest with the vision, there are capabilities the vision implies
that NSCK does not yet have:

| Gap | Explanation |
|---|---|
| **Deep perceptual learning** | `RichImageAdapter` uses structural features from an untrained `timm` model; it does not learn visual concepts from data |
| **Compositional language understanding** | NLU is n-gram heuristics, not a grammar-based or transformer-based semantic parser |
| **Automatic cross-domain discovery** | Transfer requires human-authored task definitions; zero-shot discovery is not implemented |
| **World-model imagination at scale** | `WorldModel.imagine()` is single-step; multi-step mental simulation is limited |
| **Self-improvement of architecture** | The architecture is fixed; NSCK can learn within its structure but cannot modify the structure |

---

*Document version: V15, February 2026. Update this table whenever a new version ships.*

---

## V15 Update — Model Transplantation

**Transplantation** bridges the gap between pretrained neural network knowledge
and NSCK's glass-box hypervector reasoning. After transplantation, NSCK "knows"
what the source model knows, but in native HV form — fully traceable.

**Capability impact**: Developer Extensibility (✅ enhanced), Knowledge Transfer (✅ enhanced).

```python
config = NSCKConfig.transplant()
substrate = NSCKSubstrate(config)
report = substrate.transplant(model=bert, domain_name="nlp")
# report.passed → True if quality thresholds met
```
