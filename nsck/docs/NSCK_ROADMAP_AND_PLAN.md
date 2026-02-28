# NSCK Roadmap & Version History

## Overview

The **Neuro-Symbolic Cognitive Kernel (NSCK)** is a Python/Rust cognitive
architecture that fuses Vector Symbolic Architectures (VSA), spiking neural
networks, global workspace theory, and symbolic reasoning into a single
substrate. Development began with a 10,240-bit binary hypervector core and has
grown over thirteen versions into a 94-module, 232-class system spanning
~36 K lines of Python and ~4.3 K lines of Rust.

This document records what was added in each version and outlines future
directions. Every entry below has been verified against the codebase.

---

## Version History

| Version | Theme | Key Additions | Tests |
|---------|-------|---------------|-------|
| **V1** | Foundation | Core VSA engine (10,240-bit binary HVs), semantic memory (knowledge graph), episodic memory (LSH-based recall) | — |
| **V2** | Foundation | Global Workspace Theory (coalition competition), basic rule learning, left-corner parser, SNN perception (LIF + Hebbian), Q-learning reward processing | — |
| **V3** | Language & Reasoning | Construction grammar, frame semantics, coreference, distributional semantics, belief revision (AGM-style), memory homeostasis, spatial reasoning, 14 new config flags | 531 (Python) / 671 (with Rust) |
| **V4** | Cognitive Extensions | Temporal reasoning, abductive reasoning, predictive processing, schema induction, PMI learning, transitive inference, prototype generalization | — |
| **V5** | Spatial + Safety | Spatial reasoner with FPE-based position encoding, safety gate for dangerous-action veto | — |
| **V6** | Quality | Negation handling in HVs, HNSW index for semantic memory, bug fixes and test improvements | — |
| **V7** | Language Quality | FluentNLG wired into DialogueManager, stop-concept filtering (53 words) for KG noise, DistributionalCodebook pre-trains on built-in corpus, HuggingFace corpus loader with offline fallback, 2 new config flags | — |
| **V8** | Multi-Modal + Active Inference | ConcurrentMultimodalScheduler, memory lifecycle (decay/prune/reconsolidation), dialogue state tracking, MathReasoner-GWT coalition, HierarchicalResonatorNetwork, MultiAgentSession, NSCK-Eval benchmarks (5 domains), ActiveInferenceLearner, 9 new config flags | 1 111 |
| **V9** | Universal Substrate | PerceptPacket universal input contract, 7 modality adapters, sleep() auto-generalisation, register_task() auto-transfer, Rule confidence history, StreamProcessor + StreamVerifier, substrate benchmarks | 999 |
| **V10** | Rust Concurrency + REST API | EmbeddingVSABridge, FHRR complex phasor VSA, Rust concurrent shim, NgramNLU (151 K sent/s), AttentionGWTBridge, RuleNeuralScorer, SafetyGateVerifier, FastAPI REST API, Rust .so rebuilt | 1 248 |
| **V11** | Continuous Learning | LSH rebuilding, continuous generalisation, NumericSequenceAdapter + TimeSeriesEncoder, cross-modal learning, extended NSCKSubstrate API | 1 177 |
| **V12** | Image + Audio | ImageAdapter (65-dim spatial/colour/Sobel FPE), AudioAdapter (23-dim MFCC/spectral FPE), both wired into NSCKSubstrate | 1 199 |
| **V13** | Uncertainty + Procedural Memory | SignalIngestor, UniversalHVEncoder, CrossModalAssociativeMemory, ProceduralMemory, ConceptDriftDetector, ConformalWrapper, PatternGeneralizer, CausalRuleAuditor, VideoAdapter, KLE uncertainty in GlobalWorkspace, NSCKSubstrate V13 ingest/feedback API | 1 437 (1 424 with Rust) |

---

## Detailed Version Notes

### V1 — Foundation (Part 1)

The project started with the three pillars that every later version builds on:

- **Core VSA engine** — 10,240-bit binary hypervectors with bind, bundle, and
  permute operations.
- **Semantic memory** — a knowledge graph that stores concept-relation-concept
  triples as bound HVs.
- **Episodic memory** — LSH-based recall that indexes experience snapshots for
  fast approximate nearest-neighbour retrieval.

### V2 — Foundation (Part 2)

V2 completed the foundational cognitive loop:

- **Global Workspace Theory (GWT)** — coalition competition decides which
  information reaches conscious broadcast.
- **Basic rule learning** — symbolic if-then rules extracted from experience.
- **Left-corner parser** — incremental syntactic analysis for natural-language
  input.
- **SNN perception** — Leaky Integrate-and-Fire neurons with Hebbian learning
  convert raw input into spike-coded HVs.
- **Q-learning reward processing** — reinforcement signal drives rule
  selection and memory consolidation.

### V3 — Language & Reasoning

V3 dramatically expanded linguistic and reasoning capabilities:

- **Construction grammar** — pattern-based syntactic analysis that maps
  directly to meaning.
- **Frame semantics** — semantic frames capture event structure (agent, action,
  patient).
- **Coreference resolution** — tracks entity mentions across discourse.
- **Distributional semantics** — co-occurrence statistics inform word-level HV
  similarity.
- **Belief revision (AGM-style)** — principled contraction and expansion of
  the belief set when new evidence contradicts existing knowledge.
- **Memory homeostasis** — automatic regulation of memory load to prevent
  runaway growth.
- **Spatial reasoning** — qualitative spatial relations (above, inside,
  near, …) encoded as HVs.
- **14 new configuration flags** added.
- **531 Python tests** pass (671 including Rust-accelerated paths).

### V4 — Cognitive Extensions

V4 broadened the reasoning repertoire:

- **Temporal reasoning** — Allen-style interval relations encoded in VSA.
- **Abductive reasoning** — inference to the best explanation from incomplete
  evidence.
- **Predictive processing** — top-down predictions compared against sensory
  input; prediction error drives learning.
- **Schema induction** — recurring event patterns compressed into reusable
  schemas.
- **PMI learning** — pointwise mutual information for collocation discovery.
- **Transitive inference** — if A > B and B > C then A > C, computed in HV
  space.
- **Prototype generalisation** — bundles of exemplars produce category
  prototypes.

### V5 — Spatial + Safety

- **Spatial reasoner** — Fractional Power Encoding (FPE) places objects in a
  continuous coordinate space inside VSA.
- **Safety gate** — vetoes dangerous actions before they reach the motor
  output, using a configurable blacklist and confidence threshold.

### V6 — Quality

- **Negation handling** — explicit NOT operation on HVs preserves
  dissimilarity semantics.
- **HNSW index** — Hierarchical Navigable Small World graph replaces brute-
  force search in semantic memory for O(log n) recall.
- Assorted **bug fixes and test improvements**.

### V7 — Language Quality

- **FluentNLG** — template-and-slot natural language generator wired into the
  `DialogueManager` for coherent replies.
- **Stop-concept filtering** — 53 high-frequency words (the, is, a, …)
  excluded from KG insertion to reduce noise.
- **DistributionalCodebook** — pre-trained on a built-in corpus at import time
  so distributional similarity works out-of-the-box.
- **HuggingFace corpus loader** — downloads training text from HuggingFace
  datasets; falls back to the built-in corpus when offline.
- **2 new configuration flags** added.

### V8 — Multi-Modal + Active Inference

The largest single release, V8 introduced concurrency, multi-agent operation,
and active inference:

- **ConcurrentMultimodalScheduler** — round-robin + priority scheduling of
  modality-specific processing pipelines.
- **Memory lifecycle** — exponential decay, pruning of stale items, and sleep-
  phase reconsolidation.
- **Dialogue state tracking** — history HV accumulates past turns; cosine
  distance detects topic shifts.
- **MathReasoner-GWT coalition** — arithmetic sub-module competes in the
  global workspace like any other specialist.
- **HierarchicalResonatorNetwork** — two-level (L1 / L2) resonator for
  compositional factorisation of bound HVs.
- **MultiAgentSession** — multiple NSCK instances share a workspace for
  cooperative problem-solving.
- **NSCK-Eval benchmarks** — standardised evaluation across 5 cognitive
  domains (language, reasoning, memory, perception, safety).
- **ActiveInferenceLearner** — free-energy minimisation drives exploration vs.
  exploitation trade-off.
- **9 new configuration flags** added.
- **1 111 tests** pass.

### V9 — Universal Substrate

V9 unified every input modality behind a single contract:

- **PerceptPacket** — dataclass that every adapter must produce; carries
  modality tag, raw data, and pre-computed HV.
- **7 modality adapters** — text, dict, numeric, SNN spike-train, multimodal
  bundle, stream, and raw-array.
- **sleep() auto-generalises** — during offline consolidation the system
  generates prototypes, fires transitive inference, and creates auto-
  abstractions.
- **register_task() auto-transfers** — registering a new task automatically
  transfers relevant prior knowledge.
- **Rule metadata** — `confidence_history`, `last_fired`, `fire_count` added
  to every rule for introspection.
- **StreamProcessor + StreamVerifier** — windowed processing of unbounded
  input streams with integrity checks.
- **substrate_benchmarks.py** — evaluation harness for substrate-level
  throughput and accuracy.
- **999 tests** pass.

### V10 — Rust Concurrency + REST API

V10 added a Rust acceleration layer and a network-accessible API:

- **EmbeddingVSABridge** — bidirectional map between dense embedding vectors
  and sparse binary HVs.
- **FHRR complex phasor VSA** — Fourier Holographic Reduced Representations
  for phase-angle encoding.
- **Rust concurrent shim** — `SemanticMemoryConcurrent`,
  `EpisodicMemoryConcurrent`, and `CognitiveWorkerPool` implemented in Rust
  behind PyO3 bindings.
- **NgramNLU** — Naive Bayes intent classifier over character n-grams;
  benchmarked at 151 K sentences/second.
- **AttentionGWTBridge** — multi-head attention re-ranks coalition candidates
  before GWT broadcast.
- **RuleNeuralScorer** — single-layer perceptron scores candidate rules for
  the reasoner.
- **SafetyGateVerifier** — formula-based (not just blacklist) safety
  verification.
- **FastAPI REST API** — `/decide`, `/learn`, `/sleep`, `/status` endpoints
  expose the full cognitive loop over HTTP.
- Both Rust shared objects (`.so`) rebuilt.
- **1 248 tests** pass.

### V11 — Continuous Learning

- **LSH rebuilding** — episodic memory hash tables rebuilt periodically to
  maintain recall quality as distribution shifts.
- **Continuous generalisation** — prototype and transitive inference run
  incrementally during normal operation, not only during sleep.
- **NumericSequenceAdapter + TimeSeriesEncoder** — encode ordered numeric
  sequences and time-series windows as HVs.
- **Cross-modal learning** — associations learned in one modality transfer to
  another via shared HV space.
- **Extended NSCKSubstrate API** — new convenience methods for inspection,
  bulk ingestion, and state export.
- **1 177 tests** pass.

### V12 — Image + Audio

- **ImageAdapter** — 65-dimensional feature vector (spatial grid positions,
  colour histogram bins, Sobel edge magnitudes) encoded via FPE into a single
  HV.
- **AudioAdapter** — 23-dimensional feature vector (13 MFCCs + 10 spectral
  features) encoded via FPE.
- Both adapters wired into `NSCKSubstrate` through the `PerceptPacket`
  contract.
- **1 199 tests** pass.

### V13 — Uncertainty + Procedural Memory (Current)

The current release adds uncertainty quantification, procedural memory, and
additional multi-modal capabilities:

- **SignalIngestor** — raw-signal pre-processing (normalisation, windowing,
  feature extraction) before adapter dispatch.
- **UniversalHVEncoder** — single entry-point encoder that selects the right
  adapter, tracks encoding statistics, and caches results.
- **CrossModalAssociativeMemory** — binds entity HVs across modalities so
  that seeing an object and hearing its name activate the same concept.
- **ProceduralMemory** — LRU skill cache that stores compiled action sequences
  for fast-path decisions, bypassing full deliberation.
- **ConceptDriftDetector** — monitors cosine stability of key concept HVs over
  time; fires an alert when drift exceeds a threshold.
- **ConformalWrapper** — wraps any point prediction with calibrated uncertainty
  bounds via split-conformal prediction.
- **PatternGeneralizer** — extracts common structure from sets of examples and
  produces generalised pattern HVs.
- **CausalRuleAuditor** — audits learned causal rules for spurious
  correlations using interventional checks.
- **VideoAdapter** — processes video as a sequence of image frames, producing
  per-frame and aggregated HVs.
- **KLE uncertainty in GlobalWorkspace** — coalition scores now carry
  Kullback-Leibler-based uncertainty estimates that modulate broadcast
  confidence.
- **CognitiveState** gains `kle_uncertainty` and `uncertainty_bounds` fields.
- **NSCKSubstrate V13 API** — `ingest()` and `feedback()` methods provide a
  streamlined data-in / reward-in interface.
- **1 437 tests** (1 424 pass when the Rust extension is loaded).

---

## Current State Summary

| Metric | Value |
|--------|-------|
| Core modules | 94 |
| Classes | 232 |
| Python LOC | ~36 000 |
| Rust LOC | ~4 300 |
| Total tests | 1 437 |
| Rust speed-up | 5–65× over pure Python |

The architecture is fully self-contained: no external neural-network weights
are required at runtime. All learning happens inside the HV space via
bundling, binding, and resonator decoding, augmented by lightweight neural
scorers (perceptron, Naive Bayes, SNN) and reinforcement signals.

---

## Future Directions

The items below are listed roughly in order of expected impact. None are
committed to a release date.

### Real NLU

The current NLU pipeline (left-corner parser + NgramNLU) handles constrained
input well but cannot compete with transformer-based models on open-domain
text. A planned **transformer integration layer** would run a small language
model (e.g. a distilled BERT) to produce parse features that the VSA substrate
can ingest, preserving the neuro-symbolic boundary while gaining modern
semantic parsing quality.

### Grounded Perception

ImageAdapter and AudioAdapter extract hand-crafted features. Integrating a
pre-trained **CNN or Vision Transformer (ViT)** as a feature front-end would
let the system ground concepts in real visual input rather than engineered
descriptors. The `EmbeddingVSABridge` already provides the dense-to-HV
mapping needed.

### Online Learning from Reasoning Errors

Today, reasoning errors are corrected only when explicit feedback is provided.
A **self-supervised error-detection loop** would compare predicted outcomes
with observed outcomes and trigger belief revision automatically, closing the
learning cycle without human intervention.

### Scale Testing Beyond 10 K Concepts

Semantic memory has been tested with hundreds of concepts. Validating
behaviour at **10 K+ concepts** — particularly HNSW recall quality, GWT
coalition latency, and memory homeostasis stability — is necessary before
production deployment.

### Pre-loaded Domain Knowledge Bases

The system currently starts with an empty knowledge graph. Shipping **curated
domain packs** (e.g. commonsense, biomedical, legal) as serialised HV bundles
would give new instances a useful starting vocabulary without online training.

### Distributed Multi-Node Operation

`MultiAgentSession` runs multiple NSCK instances in a single process.
Extending this to a **distributed setting** (separate machines, message-passing
via gRPC or NATS) would unlock horizontal scaling for large-scale cognitive
workloads.

### SNN Rust Batch Processing

The SNN perception layer is pure Python with per-spike FFI calls into Rust.
Moving the **entire spike-batch loop into Rust** and returning only the final
HV would eliminate FFI overhead and could yield an additional order-of-magnitude
speed-up.

---

*This document is maintained alongside the codebase. Update it whenever a new
version is tagged.*

---

## V17 — Enrichment Layer & Glass-Box Tracing (April 2026)

**Status**: ✅ Complete

### Deliverables

| Module | Status |
|--------|--------|
| `CausalEnricher` | ✅ Implemented |
| `PerceptualEnricher` | ✅ Implemented |
| `SemanticEnricher` | ✅ Implemented |
| `GlassBoxTracer` | ✅ Implemented |
| `CrossModalEnricher` | ✅ Implemented |
| V17 config flags + `NSCKConfig.v17()` | ✅ Implemented |
| 56 new tests (unit + integration + benchmark) | ✅ All passing |
| Capability benchmark (`vsa_capability_benchmark.py`) | ✅ Implemented |

### Key Metrics

- New modules: 5
- New config flags: 10
- New tests: 56
- All tests pass: 1,478 total (V17 baseline)
- Zero breaking changes from V16

### Design Principles

1. **Enrichment as post-processing**: All enrichers are optional decorators
   around existing modules — no changes to existing APIs.
2. **No neural dependencies**: All V17 enrichment is pure Python/numpy.
3. **Glass-box by default**: The tracer is zero-cost when disabled (`enabled=False`).
