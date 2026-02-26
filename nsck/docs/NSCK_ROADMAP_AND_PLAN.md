# NSCK Full Implementation Roadmap & Plan
## Neural-Symbolic Cognitive Kernel: Path to Fully Functional AGI

**Date:** February 2026  
**Status:** Living document — updated as milestones are completed  
**Research Sources:** VSA/HDC literature (2020–2025), SNN neuromorphics, Neuro-Symbolic AI surveys,
cross-domain knowledge transfer (AAAI 2024), resonator networks (Nature 2024), semantic role labeling.

---

## Table of Contents

1. [Where We Stand Now](#1-where-we-stand-now)
2. [What's Missing and Why It Matters](#2-whats-missing-and-why-it-matters)
3. [Research Foundations](#3-research-foundations)
4. [The Full Architecture Vision](#4-the-full-architecture-vision)
5. [Implementation Plan — Phase by Phase](#5-implementation-plan--phase-by-phase)
6. [Cross-Domain Knowledge: The Secret Weapon](#6-cross-domain-knowledge-the-secret-weapon)
7. [Technical Deep-Dives](#7-technical-deep-dives)
8. [Efficiency & Scalability Strategy](#8-efficiency--scalability-strategy)
9. [Benchmarks and Success Criteria](#9-benchmarks-and-success-criteria)
10. [Research References](#10-research-references)

---

## 1. Where We Stand Now

### ✅ What NSCK Already Has (Strengths)

| Subsystem | Status | Evidence |
|-----------|--------|---------|
| VSA Core (BSC/HRR, 10240-dim) | ✅ Operational | `hypervec_py.py`, Rust `hypervec_rs.so` |
| SNN Perception (LIF + STDP + Hebbian) | ✅ Operational | `snn_perception.py` ~874 LOC |
| VSA↔SNN Bridge | ✅ Operational | `vsa_snn_bridge.py`, `snn_integration.py` |
| Semantic Memory (concept graphs) | ✅ Operational | `semantic_memory.py` + Rust concurrent backend |
| Episodic Memory (k-NN LSH) | ✅ Operational | `episodic_memory.py` + Rust concurrent backend |
| Global Workspace (GWT competition) | ✅ Operational | `global_workspace.py` |
| Causal Reasoning (discovery + intervention) | ✅ Operational | `causal_reasoning.py` ~1233 LOC |
| Rule Learning | ✅ Operational | `rule_learner.py` ~644 LOC |
| Analogy Engine (cross-task abstract mapping) | ✅ Operational | `analogy.py` ~609 LOC |
| Left-Corner Parser (VSA syntax trees) | ✅ Operational | `parser.py` |
| Semantic Fingerprinting (SDR/sparse) | ✅ Operational | `lingua_cortex.py` |
| Continual Learning (EWC + task isolation) | ✅ Operational | `continual_learning.py` |
| Meta-Learning (MAML-style fast adaptation) | ✅ Operational | `meta_learning.py` |
| Multimodal Processor (text/image/audio/video) | ✅ Operational | `multimodal_processor.py` |
| Image Perception Adapter (classical CV + FPE) | ✅ Operational | `adapters/image_adapter.py` — V12 |
| Audio Perception Adapter (MFCC + FPE) | ✅ Operational | `adapters/audio_adapter.py` — V12 |
| NSCKSubstrate developer API | ✅ Operational | `substrate.py` — V11 |
| Emotion System | ✅ Operational | `emotion_system.py` |
| Self-Model + Theory of Mind | ✅ Operational | `self_model.py`, `theory_of_mind.py` |
| Metacognition + Safety Gate | ✅ Operational | `metacognition.py` |
| STRIPS Planner | ✅ Operational | `planner.py` |
| Brain Fusion (multi-task knowledge merge) | ✅ Operational | `brain_fusion.py` |
| Persistence (SQLite + BrainStore) | ✅ Operational | `persistence.py` |
| Explanation Generator (11-stage traces) | ✅ Operational | `explanation.py` |
| Rust Concurrent Layer (21-206× speedup) | ✅ Operational | `rust_vsa/`, `rust_snn/` |
| Test Coverage (1,199 tests, ≥99% pass) | ✅ Operational | `nsck/tests/` |

### ⚠️ Current Gaps (What We Need to Add/Improve)

| Gap | Why It Matters | Priority | Status |
|-----|---------------|----------|--------|
| Semantic Role Labeling (SRL) via Resonator Networks | True NLU requires knowing *who did what to whom* | P0 | ✅ **Done (A1)** |
| Math/Numeric Reasoning in VSA | "Numbers and math" are core input types | P0 | ✅ **Done (B1)** |
| Cross-Domain Knowledge Transfer (formal) | Generalization across domains is the hallmark of intelligence | P0 | ✅ **Done (C1)** |
| NLG: Multi-sentence discourse planning | Fluent responses need more than S-V-O templates | P1 | ✅ **Done (A2)** |
| Concurrent multimodal fusion pipeline | Simultaneous inputs (e.g., spoken + visual) not properly scheduled | P1 | ✅ **Done (V8)** |
| SNN Rust port for perception (<5ms) | Python SNN is ~51ms; need Rust for real-time use | P1 | ✅ **Done (V8 — snn_rs.so)** |
| Formal temporal reasoning (before/after/during) | Time is fundamental to language and events | P2 | ✅ **V4 Done** |
| Spatial reasoning (above/below/inside/outside) | Required for vision-language grounding | P2 | ✅ **V5 Done** |
| Negation handling in VSA | "X is NOT Y" needs explicit anti-bundling | P2 | ✅ **V4 Done** |
| Conditional logic ("if X then Y") | Conditional reasoning is fundamental to causality | P2 | ✅ **V4 Done** |
| Taxonomic/transitive closure | A is_a B + B is_a C → A is_a C | P2 | ✅ **V4 Done** |
| Prototype-based category generalization | Rosch prototype theory: bundle members into category | P2 | ✅ **V4 Done** |
| Massively extended COMMON_VERBS | ~40-60% → ~75-85% NLU coverage | P0 | ✅ **V4 Done** |
| Cross-domain research synthesis | Biology, physics, math, psychology insights for NSCK | P1 | ✅ **V4 Done** |
| Scalar implicature and pragmatics | Natural language is richer than propositional logic | P3 | ✅ **V5 Done** |
| Memory lifecycle (decay + prune) | Prevent unbounded growth; Ebbinghaus forgetting | P1 | ✅ **Done (V8)** |
| Multi-turn dialogue state tracking | Coreference + topic coherence across turns | P1 | ✅ **Done (V8)** |
| Hierarchical resonator networks | Nested predicate-argument structure (relative clauses) | P1 | ✅ **Done (V8)** |
| Multi-agent cognitive fusion | Shared belief negotiation across agents | P2 | ✅ **Done (V8)** |
| Active inference full loop | Free-energy minimisation for action selection | P1 | ✅ **Done (V8)** |
| NSCK-Eval benchmark suite | Formal evaluation beyond internal tests | P1 | ✅ **Done (V8)** |

### ✅ V8 Completed (February 2026)

**Test suite: 1,199 passed, 150 skipped, 3 xfailed.** Rust extensions (`hypervec_rs.so` + `snn_rs.so`) built and active.

| Feature | Files Changed | Impact |
|---------|--------------|--------|
| `ConcurrentMultimodalScheduler` | `multimodal_processor.py` | Parallel per-modality processing; attention-weighted 50ms coherence fusion |
| Memory lifecycle (decay + prune + reconsolidation) | `semantic_memory.py`, `homeostasis.py`, `episodic_memory.py` | Ebbinghaus forgetting; prune low-importance concepts; episodic HV blending on recall |
| Multi-turn dialogue state tracking | `dialogue_manager.py` | Rolling history HV; topic-shift detection; `get_context_hv()`; `clarification_request()` |
| MathReasoner–GWT integration | `cognitive_engine.py`, `universal_input.py` | MATH coalition (salience 0.95); `is_mathematical()` regex classifier |
| `HierarchicalResonatorNetwork` | `resonator.py` | 2-level L1/L2 factorization; `factorize_hierarchical(hv, depth=2)` |
| `MultiAgentSession` + ToM extensions | `brain_fusion.py`, `theory_of_mind.py` | `exchange_snapshots()`, `negotiate_beliefs()`, `model_other_agent()`, `perspective_take()` |
| NSCK-Eval benchmark suite | `benchmarks/` (6 new files) | bAbI, math, transfer, NLG, dialogue; `BenchmarkRunner.run_all()` |
| `ActiveInferenceLearner` | `active_inference.py`, `cognitive_engine.py`, `metacognition.py` | `F=pred_error−epi_value` wired into decide(); `SafetyGate.check_free_energy()` |
| V8 config flags (9 new) | `config.py` | `enable_active_inference`, `enable_concurrent_multimodal`, etc. |
| Rust extensions built | `nsck/*.so` | `hypervec_rs.so` + `snn_rs.so` via maturin; 1,199 tests pass |
| 140 new tests | `tests/unit/**` | 20+20+20+15+15+15+15+10 = 140 new tests |
| V8_CHANGELOG.md | `docs/V8_CHANGELOG.md` | Complete feature log |

### ✅ V5 Completed (February 2026)

| Feature | Files Changed | Impact |
|---------|--------------|--------|
| Spatial Reasoning module (`spatial_reasoning.py`) | `reasoning/spatial_reasoning.py` | 2D/3D VSA position encoding, 8 relation types |
| FPE-based position locality | `reasoning/spatial_reasoning.py` | Nearby positions → similar HVs (measurable) |
| Projective relations (above/below/left/right) | `reasoning/spatial_reasoning.py` | "What is above X?" queries |
| Topological relations (adjacent/at_same_position) | `reasoning/spatial_reasoning.py` | "Is X adjacent to Y?" |
| Metric queries (distance, find_near) | `reasoning/spatial_reasoning.py` | Euclidean distance queries |
| VSA assertion encoding | `reasoning/spatial_reasoning.py` | bind(v_ABOVE, bind(v_fig, v_ground)) |
| Pragmatics module (`pragmatics.py`) | `language/pragmatics.py` | 7 speech act types, 15 scalar scales |
| Scalar implicature (Horn scales) | `language/pragmatics.py` | "some" → implies "not all" |
| Gricean maxim checking | `language/pragmatics.py` | Quality/Quantity/Relation/Manner |
| Indirect speech act detection | `language/pragmatics.py` | "Can you…" → request (indirect) |
| Presupposition detection | `language/pragmatics.py` | "stopped X" → presupposes prior X |
| Politeness hedging detection | `language/pragmatics.py` | "please", "could you" |
| V5 config flags (2 new) | `config.py` | `enable_spatial_reasoning`, `enable_pragmatics` |
| 45 spatial tests | `tests/unit/reasoning/test_spatial_reasoning.py` | All passing |
| 45 pragmatics tests | `tests/unit/language/test_pragmatics.py` | All passing |
| Updated COGNITIVE_REPORT_CARD.md | `COGNITIVE_REPORT_CARD.md` | V5 honest assessment |

### ✅ V4 Completed (February 2026)

| Feature | Files Changed | Impact |
|---------|--------------|--------|
| Extended COMMON_VERBS (~400 forms, 18 categories) | `construction_grammar.py` | NLU coverage ~75-85% |
| Improved POS morphological classifier | `construction_grammar.py` | Better -s/-ies/-ize/-ish handling |
| Negation constructions (9 new patterns) | `construction_grammar.py` | "X is not Y", "X lacks Y", "X cannot V Y" |
| Conditional constructions (7 new patterns) | `construction_grammar.py` | "X implies Y", "X leads to Y", "if X then Y" |
| Temporal ordering constructions (7 new patterns) | `construction_grammar.py` | "X before Y", "X after Y", "since X" |
| Similarity/difference constructions (4 new patterns) | `construction_grammar.py` | "X is like Y", "X differs from Y" |
| V4 feature flags (5 new) | `config.py` | Gated by `enable_negation_handling`, etc. |
| Transitive inference | `semantic_memory.py` | A is_a B + B is_a C → A is_a C |
| Prototype generalization | `semantic_memory.py` | VSA bundle of category members |
| V4 relation weights (negation, temporal, etc.) | `semantic_memory.py` | Correct spreading activation |
| TKL role extraction for V4 constructions | `text_knowledge_learner.py` | V4 facts stored in knowledge graph |
| TKL transitive inference wiring | `text_knowledge_learner.py` | Runs every 10 sentences |
| Cross-domain research document | `docs/CROSS_DOMAIN_RESEARCH.md` | 20 domain sources, 15 actionable proposals |
| 50 new V4 unit tests | `tests/unit/language/test_v4_features.py` | All passing |

---

## 2. What's Missing and Why It Matters

### 2.1 Semantic Role Labeling — The "Who Did What" Problem

**What it is:** Extracting structured predicate-argument structure from text.
- "The dog bit the boy" → AGENT=dog, VERB=bit, PATIENT=boy
- "The ball was kicked by Maria into the net" → AGENT=Maria, THEME=ball, GOAL=net

**Why NSCK needs it:** Our current left-corner parser can identify S-V-O patterns, but misses:
- Passive voice agent recovery
- Prepositional phrase role disambiguation  
- Frame semantics (FrameNet: same concept in different contexts)
- Instrument, Location, Source, Goal, Beneficiary, Temporal roles

**How (VSA approach):**  
Resonator networks (Frady et al. 2020; Renner et al. 2024 — *Nature Machine Intelligence*) can simultaneously unbind all role-filler pairs from a single composite VSA vector:
```
S = bind(AGENT, v_dog) ⊕ bind(VERB, v_bite) ⊕ bind(PATIENT, v_boy)
Resonator recovers: (AGENT→dog, VERB→bite, PATIENT→boy)
```
This is fully VSA-native — no neural network needed.

### 2.2 Math/Numeric Reasoning — Numbers Are First-Class Citizens

**What it is:** Understanding and manipulating numerical information:
- Arithmetic: "What is 3 + 5?" → 8
- Comparison: "Is 7 > 4?" → True
- Algebra: "x + 3 = 7, find x" → x = 4
- Word problems: "If Alice has 3 apples and Bob gives her 2 more, how many does she have?" → 5

**Why NSCK needs it:** Numbers appear in virtually all real-world knowledge domains (science, finance, everyday reasoning). Without numeric reasoning, the system cannot fully understand text that contains quantitative information.

**How (VSA approach):**  
VSA can encode numbers in multiple complementary ways:
1. **Resonance Coding**: Assign each integer a unique, slightly-related hypervector using permutation chains (n+1 = permute(n))
2. **Magnitude Encoding**: Use signed bipolar vectors where inner product with a "magnitude probe" gives the value
3. **Fractional Power Encoding (FPE)**: `v_n = v_base^n` — proven to support arithmetic (Plate 2003, Gosmann & Eliasmith 2019)
4. **Symbolic Equations**: Parse equations into VSA predicate structures; apply rule-based algebra via pattern matching

### 2.3 Cross-Domain Transfer — The True Test of Intelligence

**What it is:** Using knowledge from one domain to solve problems in another.
- "Water flows downhill" → "Information propagates in a network via shortest paths" (fluid dynamics → graph theory)
- "A spring stores energy" → "A capacitor stores charge" (mechanics → electronics)

**Why NSCK needs it:** Domain-specific learning is brittle. A truly intelligent system recognizes structural isomorphisms across domains and transfers rules/patterns without retraining.

**How (VSA approach):**  
The existing `AnalogyEngine` provides a framework. We need to formalize:
1. **Relational alignment**: Find roles that play the same structural role in two domains
2. **Rule lifting**: Abstract a rule from specific domain vocabulary to domain-independent form
3. **VSA binding for abstract schemas**: `v_schema = bind(ROLE_A, v_domain1_concept) ⊕ bind(ROLE_B, v_domain2_concept)`
4. **Transfer confidence scoring**: Measure structural overlap via resonator cosine similarity

---

## 3. Research Foundations

### 3.1 VSA/HDC for Language (2020–2025 Key Papers)

| Paper | Key Contribution | NSCK Application |
|-------|-----------------|-----------------|
| Frady et al. 2020 — *Resonator Networks* | Factorization of VSA composites | SRL, multimodal scene parsing |
| Renner et al. 2024 — *Nature MI* | Hierarchical resonators for visual scene understanding | Multimodal concept extraction |
| Plate 2003 — *HRR* | Circular convolution binding | Core VSA operations |
| Gayler 2003 — *VSA review* | Binding properties | All symbolic operations |
| Gosmann & Eliasmith 2019 — *FPE* | Fractional power encoding for numbers | Math reasoning module |
| Kanerva 2009 — *Hyperdimensional Computing* | Theoretical foundations | All HDC operations |
| Mitrokhin et al. 2019 — *SNN-HDC* | SNN-based HDC classification | SNN-VSA bridge |
| Neubert et al. 2024 — *VSA survey* | State-of-art review | Architecture decisions |

### 3.2 SNN for Temporal/Sequential Processing

| Paper | Key Contribution | NSCK Application |
|-------|-----------------|-----------------|
| Mahowald et al. 1992 — *LIF* | Leaky Integrate-and-Fire model | SNN Perception |
| Bi & Poo 1998 — *STDP* | Spike-timing-dependent plasticity | Hebbian learning |
| Izhikevich 2003 — *Polychronization* | Neural group formation | Concept formation |
| Schuman et al. 2017 — *Neuromorphic survey* | Hardware overview | Efficiency strategy |
| Pfeiffer & Pfeil 2018 — *SNN learning* | Surrogate gradients | Future SNN training |

### 3.3 Cross-Domain Knowledge Transfer

| Paper | Key Contribution | NSCK Application |
|-------|-----------------|-----------------|
| Falkenhainer et al. 1989 — *SME* | Structure mapping engine | Analogy engine foundation |
| Hummel & Holyoak 1997 — *LISA* | Relationship-based analogical inference | Cross-domain transfer |
| Wang et al. 2024 AAAI — *SNN transfer* | Domain alignment loss for SNNs | SNN cross-domain |
| Bengio et al. 2012 — *Representation learning* | Domain-invariant features | Transfer strategy |

### 3.4 Cognitive Architecture Research

| Architecture | Key Insight | What NSCK Borrows |
|-------------|------------|-------------------|
| ACT-R (Anderson 1993) | Chunking, procedural + declarative | Rule learner design |
| SOAR (Laird 1987) | Problem spaces, universal subgoaling | Planning architecture |
| Global Workspace Theory (Baars 1988) | Broadcast, competition | GlobalWorkspace module |
| Semantic Pointer Architecture (Eliasmith 2013) | VSA neural implementation | Core VSA-SNN bridge |
| HTM (Hawkins 2004) | Sparse distributed representations | Lingua cortex SDRs |
| Predictive Processing (Clark 2013) | World models, top-down predictions | Metacognition module |

---

## 4. The Full Architecture Vision

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               NSCK COGNITIVE KERNEL                     │
                    │                                                         │
  ┌──────┐          │  ┌────────────────┐    ┌────────────────────────────┐  │
  │TEXT  │──────────┼─▶│  PERCEPTION    │    │   GLOBAL WORKSPACE (GWT)   │  │
  │IMAGE │──────────┼─▶│  LAYER         │───▶│   Competition + Broadcast  │  │
  │AUDIO │──────────┼─▶│                │    │   (Attention Gate)         │  │
  │MATH  │──────────┼─▶│ • Multimodal   │    └──────────────┬─────────────┘  │
  │SENSOR│──────────┼─▶│   Processor    │                   │               │
  └──────┘          │  │ • SNN (spikes) │    ┌──────────────▼─────────────┐  │
                    │  │ • Grounding    │    │   SEMANTIC WORKING MEMORY  │  │
                    │  └───────┬────────┘    │   VSA HyperVectors (10240d)│  │
                    │          │             └──────────────┬─────────────┘  │
                    │          ▼                             │               │
                    │  ┌────────────────┐                   │               │
                    │  │  LANGUAGE      │    ┌──────────────▼─────────────┐  │
                    │  │  LAYER         │    │   REASONING LAYER          │  │
                    │  │                │    │   • Causal Discovery       │  │
                    │  │ • Parser (SRL) │    │   • Rule Learning          │  │
                    │  │ • Resonator    │    │   • Analogy (Cross-Domain) │  │
                    │  │ • Semantic Map │    │   • STRIPS Planner         │  │
                    │  │ • NLG Engine   │    │   • Math Reasoning         │  │
                    │  │ • Text Learner │    │   • Context Engine         │  │
                    │  └───────┬────────┘    └──────────────┬─────────────┘  │
                    │          │                             │               │
                    │          ▼                             ▼               │
                    │  ┌──────────────────────────────────────────────────┐  │
                    │  │                 MEMORY SYSTEMS                   │  │
                    │  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │  │
                    │  │  │  Semantic   │  │  Episodic   │  │  Working │ │  │
                    │  │  │  Memory     │  │  Memory     │  │  Memory  │ │  │
                    │  │  │  (concepts) │  │  (episodes) │  │  (active)│ │  │
                    │  │  └─────────────┘  └─────────────┘  └──────────┘ │  │
                    │  └──────────────────────────────────────────────────┘  │
                    │                                                         │
                    │  ┌──────────────────────────────────────────────────┐  │
                    │  │               LEARNING LAYER                     │  │
                    │  │  Hebbian │ Continual │ Meta-Learning │ Cross-Domain│ │
                    │  └──────────────────────────────────────────────────┘  │
                    │                                                         │
                    │  ┌──────────────────────────────────────────────────┐  │
                    │  │             COGNITIVE LAYER                      │  │
                    │  │  Emotion │ Self-Model │ ToM │ Metacognition      │  │
                    │  └──────────────────────────────────────────────────┘  │
                    └─────────────────────────────────────────────────────────┘
                                              │
                              ┌───────────────▼──────────────┐
                              │     RESPONSE GENERATION       │
                              │  NLG → Fluent Natural Language│
                              └──────────────────────────────┘
```

### Information Flow

1. **Input** arrives (text/image/audio/math/sensor — sequential or simultaneous)
2. **Perception Layer** converts each modality to a HyperVector via:
   - Text → tokenize + semantic fingerprint + role binding
   - Image → HOG + color histogram + spatial quadrant encoding
   - Audio → MFCC + spectral features → HV
   - Math → FPE numeric encoding + equation parsing
   - Sensor → rate coding → SNN spikes → HV via bridge
3. **Multimodal Fusion** binds modality HVs with role vectors (TEXT_ROLE, IMAGE_ROLE...)
4. **Language Layer** applies SRL resonator to extract who/what/where/when/why
5. **Global Workspace** selects most relevant concepts for broadcast
6. **Reasoning Layer** applies causal/rule/analogy/planning on the activated concepts
7. **Memory** provides context: retrieve similar episodes, activate related concepts
8. **Learning Layer** updates weights, transfers applicable knowledge across domains
9. **NLG** generates a fluent natural-language response from the reasoning trace

---

## 5. Implementation Plan — Phase by Phase

### Phase A: Language Understanding (Priority P0) — 2–3 weeks

#### A1: VSA Semantic Role Labeling (SRL)
**File:** `nsck/python/core/language/semantic_roles.py` ✅ (implemented in this PR)

**What:** A VSA-native module that identifies semantic roles in sentences without any neural network. Uses resonator networks to factorize bound role-filler composites.

**Why:** True NLU requires knowing WHO did WHAT to WHOM, WHERE, WHEN, WHY, and HOW. Without this, responses are syntactically formed but semantically shallow.

**How:**
```python
# 1. Map each word to a hypervector
# 2. Identify predicate (main verb) via POS heuristics
# 3. Bind each argument with its ThemRole vector:
#    S = bind(v_AGENT, v_subj) ⊕ bind(v_PRED, v_verb) ⊕ bind(v_PATIENT, v_obj)
# 4. Run resonator network to recover all (role, filler) pairs
# 5. Return structured SRL frame
```

**Algorithms:**
- Lexico-syntactic heuristics for initial role candidates (NomBank, FrameNet-style)
- VSA resonator for verification and disambiguation
- Confidence scoring via cosine similarity

**Tests:** `nsck/tests/unit/language/test_semantic_roles.py` ✅

---

#### A2: Enhanced NLG — Multi-Sentence Discourse
**File:** `nsck/python/core/language/nlg.py` ✅ (enhanced in this PR)

**What:** Move from single S-V-O template generation to multi-sentence discourse that can:
- Generate explanatory paragraphs (topic sentence + supporting facts)
- Use discourse connectives (therefore, because, however, also, first, then)
- Apply pronoun anaphora (avoid repeating "The dog ... the dog ... the dog")
- Produce query-type-appropriate responses (factual vs. explanatory vs. procedural)
- Handle negation ("X does not Y")

**Why:** The existing NLG produces technically correct but unnatural, stilted output. Real understanding must be demonstrated through fluent expression.

**How:**
- Discourse planner: selects relevant facts from reasoning trace, orders by coherence
- Coreference tracker: keeps entity reference chain for pronoun substitution
- Connective selector: based on semantic relation type (causal → "because/therefore", adversative → "however", additive → "also/furthermore")

---

### Phase B: Numeric & Symbolic Math (Priority P0) — 1–2 weeks

#### B1: Math Reasoning Module
**File:** `nsck/python/core/reasoning/math_reasoning.py` ✅ (implemented in this PR)

**What:** VSA-based numeric understanding and arithmetic:
- Integer and float representation via Fractional Power Encoding (FPE)
- Basic arithmetic: +, -, ×, ÷
- Comparison: <, >, ==
- Simple algebra: solve one-variable linear equations
- Math word problem parsing → equation extraction → solution

**Why:** Numbers appear in virtually all real knowledge. Without this, the system is blind to quantitative information.

**How:**
- **FPE encoding**: `v_n = v_base ** n` where ** is repeated binding
  - v_0 = identity, v_1 = v_base, v_2 = bind(v_base, v_base), ...
  - Supports smooth interpolation (v_1.5 = midpoint between v_1 and v_2)
- **Arithmetic via FPE algebra**: add(A, B) ≈ search(v_A * v_B / v_1) in number codebook
- **Comparison via similarity**: sim(v_n, v_m) decreases as |n-m| increases
- **Word problem parser**: regex + template matching to extract numbers and operations
- **Equation solver**: symbolic algebra (pure Python, no libraries needed)

---

### Phase C: Cross-Domain Knowledge Transfer (Priority P0) — 2–3 weeks

#### C1: Cross-Domain Transfer Engine
**File:** `nsck/python/core/learning/cross_domain.py` ✅ (implemented in this PR)

**What:** Formal mechanism for transferring knowledge across domains:
1. **Schema extraction**: identify abstract relational patterns in source domain
2. **Structure mapping**: find corresponding structure in target domain using VSA similarity
3. **Rule transfer**: lift domain-specific rules to domain-independent forms, apply to target
4. **Confidence calibration**: score each transferred inference by structural overlap

**Why:** This is THE test of generalization. Cross-domain transfer enables:
- Physics knowledge → financial modeling (flow rates → transaction rates)
- Biological evolution → software version management (mutation, selection, fitness)
- Navigation rules → argument structure (starting point, path, destination → premise, inference, conclusion)

**How:**
- Builds on existing `AnalogyEngine` but adds formal rule lifting
- Uses VSA role binding to create domain-independent schema vectors
- Maintains transfer confidence scores and provenance
- Integrates with `RuleLearner` to apply transferred rules in target domain

---

### Phase D: Multimodal Concurrent Fusion (Priority P1) — 2 weeks

#### D1: Concurrent Input Scheduler
**File:** Enhancement to `nsck/python/core/multimodal/multimodal_processor.py`

**What:** Priority-based scheduling for simultaneous multimodal inputs:
- Each modality has a processing pipeline that runs concurrently
- Results are timestamped and fused with attention weights
- Temporal coherence checking: inputs within 50ms window are treated as co-occurring

**Why:** Real-world perception is simultaneous. A person speaking while showing an image should result in a unified representation, not sequential processing.

**How:**
- Thread-per-modality processing with Python `concurrent.futures`
- Attention-weighted fusion: modality weight = confidence × temporal_coherence × relevance_to_context
- Temporal binding via permutation: events at time t use `permute(v, t)` encoding

---

### Phase E: SNN Performance (Priority P1) — 1 week

#### E1: Rust SNN Port
**Target:** `nsck/rust_snn/` (expand existing stub)

**What:** Port the Python SNN perception module to Rust for <5ms latency (currently ~19ms Python)

**Why:** Real-time sensory processing requires low latency. The 5ms target is needed for:
- Interactive dialogue (response within 100ms)
- Sensory fusion with video (30fps → 33ms per frame budget)
- Concurrent multi-modal processing

**How:**
- Rust LIF neuron with SIMD-vectorized integration
- Parallel thread-per-neuron-group processing
- PyO3 Python bindings matching existing `SNNPerceptionModule` API

---

### Phase F: Temporal & Spatial Reasoning (Priority P2) — 2–3 weeks

#### F1: Temporal Reasoning
**File:** `nsck/python/core/reasoning/temporal_reasoning.py`

**What:** Explicit representation and inference over time:
- Before/after/during/since/until relationships
- Temporal interval algebra (Allen's 13 relations)
- Event sequencing and duration estimation
- "Yesterday", "last week", "in 3 hours" → absolute timestamp reasoning

**How:**
- Temporal HVs: `v_t = permute(v_base, t)` where t is discrete time step
- Allen interval algebra: encode 13 relations (before, meets, overlaps, etc.) as VSA role binding
- Timeline: linear chain of bound temporal HVs

#### F2: Spatial Reasoning
**File:** `nsck/python/core/reasoning/spatial_reasoning.py`

**What:** 2D/3D spatial relationship understanding:
- Topological: inside, outside, adjacent, overlaps, contains
- Projective: above, below, left, right, in front, behind
- Metric: distance encoding using FPE on coordinate vectors

**How:**
- 2D grid encoding: x-axis and y-axis HVs via FPE, position = bind(v_x, v_y)
- Relative position: bind(RELATION_ROLE, bind(v_obj1, v_obj2))
- Image spatial features already in multimodal_processor.py — extend to symbolic queries

---

### Phase G: Negation & Pragmatics (Priority P2–P3) — 3–4 weeks

#### G1: VSA Negation
**File:** Enhancement to `hypervec_py.py` + `language_module.py`

**What:** Represent "NOT X" in VSA:
- Bipolar negation: negate(v) = -v (flip all signs in bipolar representation)
- Exclusive bundling: A ⊕ NOT(A) → near-zero vector (orthogonal to both)

**How:**
- Anti-bundling: `v_notX = XOR(v_X, v_randomNoise)` — makes v_X and v_notX orthogonal
- Negation scope tracking: "NOT (A AND B)" vs "NOT A AND B"

#### G2: Scalar Implicature
**File:** `nsck/python/core/language/pragmatics.py`

**What:** Grice's maxims and implicature:
- "Some students passed" implies "not all students passed"
- "It's warm" when outside = "It could be better" (irony via emotion system integration)

---

## 6. Cross-Domain Knowledge: The Secret Weapon

### Why Cross-Domain Transfer Makes NSCK Unique

Most AI systems learn domain by domain. NSCK's VSA core enables structural isomorphism detection across domains because:

1. **VSA is domain-agnostic**: the same binding operations work for any content
2. **Semantic similarity is structural**: sim(v_A, v_B) reflects structural overlap, not just label matching
3. **Rules are first-class vectors**: a rule like "IF X causes Y AND Y causes Z THEN X causes Z" can itself be encoded as a VSA composite and searched against the current context

### Cross-Domain Examples NSCK Should Handle

| Source Domain | Target Domain | Transferred Pattern |
|--------------|--------------|---------------------|
| Physics: F=ma | Finance: return = beta × market_move | Proportionality schema |
| Biology: predator-prey | Economics: supply-demand cycles | Oscillatory competition schema |
| Navigation: pathfinding | Logical deduction: proof search | Search in state space |
| Geometry: angle bisector | Fairness: equitable division | Bisection/balance schema |
| Fluid dynamics: flow rate | Information theory: bandwidth | Rate-limited transfer schema |
| Immune system: antibody | Security: firewall rule | Specific blocking schema |

### How to Apply Cross-Domain Knowledge in Practice

```
1. User asks: "How does compound interest work?"
2. NSCK has physics knowledge: "Snowball rolling downhill gains mass proportional to mass"
3. Cross-domain transfer detects: finance.compound_growth ↔ physics.exponential_accumulation
4. Transfer rule: "growth(t) = initial × rate^t"
5. Answer: "Compound interest grows like a snowball — each period the interest earned 
   is added to the principal, so next period's interest is larger. 
   This gives exponential growth: balance(t) = P × (1+r)^t."
```

---

## 7. Technical Deep-Dives

### 7.1 Resonator Networks for Semantic Role Labeling

**Theory (Frady et al. 2020):**

Given a compositionally-bound VSA representation:
```
S = bind(v_AGENT, v_dog) ⊕ bind(v_VERB, v_bite) ⊕ bind(v_PATIENT, v_boy)
```

A resonator network recovers each role-filler pair simultaneously using attractor dynamics:
```
r_AGENT(t+1) = normalize( unbind(S, v_AGENT) · codebook_similarity )
r_VERB(t+1)  = normalize( unbind(S, v_VERB)  · codebook_similarity )
r_PATIENT(t+1) = normalize( unbind(S, v_PATIENT) · codebook_similarity )
```

The network converges when each r_(role) points stably to the correct filler vector.

**Convergence guarantee:** With D=10240 dimensions and codebook size ≤ 10,000 items, convergence in <20 iterations with >99% recall accuracy (Frady et al. 2020 proof).

**Hierarchical extension (Renner et al. 2024):**  
For nested structures ("The dog that chased the cat bit the boy"):
- Level 1: sentence-level roles (AGENT, VERB, PATIENT)
- Level 2: relative clause roles (MODIFIER_AGENT, MODIFIER_VERB, MODIFIER_PATIENT)
- Each level uses a separate partition of the hypervector space

### 7.2 Fractional Power Encoding (FPE) for Numbers

**Theory (Gosmann & Eliasmith 2019):**

For any scalar value n, its FPE encoding is:
```python
v_n = v_base ** n  # repeated binding operation
```

Properties:
- `v_0` = identity vector (similarity 1.0 with everything)
- `v_1` = `v_base` (the primitive)
- `v_n * v_m` ≈ `v_{n+m}` (addition via binding!)
- `sim(v_n, v_m) = cos(π(n-m)/D_eff)` where D_eff is effective dimensionality
- Works for non-integers: `v_1.5` = midpoint between `v_1` and `v_2`

**Arithmetic via FPE:**
```
add(a, b) = bind(v_a, v_b) ≈ v_{a+b}  [search in number codebook]
sub(a, b) = bind(v_a, inv(v_b)) ≈ v_{a-b}
mul(a, b) = need scaling — use log(a) FPE + FPE addition → FPE of log(a×b) = log(a)+log(b)
```

**NSCK Implementation Strategy:**
1. Pre-compute FPE codebook for integers 0..1023 and common fractions
2. For arithmetic, bind operand FPEs and search codebook for nearest neighbor
3. For word problems, parse text → equation → symbolic algebra (exact) → FPE encoding

### 7.3 Concurrent Multimodal Processing

**Design principle:** Independent modality pipelines that merge at the Global Workspace

```
Text input    ──▶ [Text Pipeline]   ──▶ (HV_text, conf_text)   ──┐
Image input   ──▶ [Image Pipeline]  ──▶ (HV_image, conf_image) ──┼──▶ [Fusion] ──▶ Global Workspace
Audio input   ──▶ [Audio Pipeline]  ──▶ (HV_audio, conf_audio) ──┤
Sensor input  ──▶ [SNN Pipeline]    ──▶ (HV_sensor, conf_sens) ──┘
```

**Fusion formula:**
```
HV_fused = Σ_m (conf_m × bind(v_role_m, HV_m)) / Σ_m conf_m
```

**Temporal binding:**  
For events separated by Δt:
```
HV_at_t = permute(HV_event, round(Δt / δt))  # δt = temporal resolution
```

---

## 8. Efficiency & Scalability Strategy

### Memory Efficiency

| Component | Current | Target | Strategy |
|-----------|---------|--------|----------|
| HyperVector dim | 10,240 bits | 10,240 bits (keep) | Already optimal for GPU-free use |
| Semantic Memory | All in RAM | Tiered (hot/warm/cold) | LRU cache + SQLite overflow |
| Episodic Memory | Full copy | LSH index only | Summary vectors + timestamps |
| Rule base | Full Python dicts | Compressed VSA lattice | Rules as VSA composites |

### Compute Efficiency

| Operation | Python | Rust | Target |
|-----------|--------|------|--------|
| Similarity search (1000 items) | ~10ms | ~0.1ms | <1ms |
| SNN perception (1 step) | ~19ms | ~2ms (target) | <5ms |
| Multimodal fusion (4 modalities) | ~40ms | ~5ms (target) | <10ms |
| NLG generation (1 response) | ~5ms | — | <10ms |

### Scaling Strategy

1. **Horizontal scaling**: Multiple CognitiveEngine instances (different tasks) share a BrainFusion master
2. **Incremental learning**: New concepts added to HV codebook without retraining existing ones
3. **Knowledge pruning**: Low-confidence, rarely-accessed rules/episodes expire (importance weighting)
4. **Rust acceleration**: All hot paths ported to Rust (similarity search, SNN, bundling)

---

## 9. Benchmarks and Success Criteria

### Minimum Viable Cognitive System (MVCS)

A system that can be called "fully functional" must pass:

| Capability | Benchmark | Target | Current |
|-----------|-----------|--------|---------|
| Natural language understanding | Parse 100 diverse sentences → extract correct SRL frames | >85% | ~60% (no SRL) |
| Contextual comprehension | Answer 10 context-dependent questions about a passage | >80% | ~50% |
| Math reasoning | Solve 50 arithmetic word problems | >90% | ~0% (not implemented) |
| Cross-domain transfer | Apply 5 physics rules to financial problems | >70% | ~20% (analogy only) |
| Multimodal fusion | Correctly describe image+text combinations | >75% | ~65% |
| NLG quality | Human raters score responses ≥ 3.5/5.0 | ≥ 3.5/5 | ~2.5/5 |
| Continual learning | Learn 10 tasks without forgetting first 3 | >85% retention | ~83% (achieved) |
| Temporal reasoning | Answer 30 "before/after" questions | >80% | ~40% (no formal module) |
| Latency (text query) | End-to-end response time | <100ms | ~15ms ✅ |
| Memory footprint | Full system RAM usage | <512MB | ~85MB ✅ |

### Full Cognitive System (FCS)

The "fully functional" NSCK that the problem statement envisions:

| Capability | Benchmark | Target |
|-----------|-----------|--------|
| Multi-domain reasoning | 100-question mixed-domain test | >75% |
| Language generation fluency | BLEU-4 vs human references | >0.35 |
| Cross-domain analogy | Raven's Progressive Matrices analog | >70% |
| Simultaneous multimodal | Image+audio+text queries | >65% |
| Generalization to unseen domains | Zero-shot transfer to 3 new domains | >50% |
| Long conversation coherence | 20-turn dialogue with consistent context | >80% |

---

## 10. Research References

### Core VSA/HDC
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.
- Plate, T.A. (2003). *Holographic Reduced Representations*. CSLI Publications.
- Gayler, R.W. (2003). Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience. *Proceedings of ICCS/ASCS*.
- Frady, E.P., Kleyko, D., & Sommer, F.T. (2020). Resonator networks, 1: An efficient solution for factoring high-dimensional, distributed representations of data structures. *Neural Computation*, 32(12), 2311–2379.

### Resonator Networks
- Renner, A., Supic, L., Bharadwaj, A., Frady, E.P., Davies, M., Sommer, F.T., & Neftci, E.O. (2024). Neuromorphic visual scene understanding with resonator networks. *Nature Machine Intelligence*, 6, 551–565.

### SNN & Neuromorphics
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659–1671.
- Schuman, C.D., et al. (2022). Opportunities for neuromorphic computing algorithms and applications. *Nature Computational Science*, 2, 10–19.

### Cross-Domain Transfer
- Falkenhainer, B., Forbus, K.D., & Gentner, D. (1989). The structure-mapping engine. *Artificial Intelligence*, 41(1), 1–63.
- Wang, Y., et al. (2024). An efficient knowledge transfer strategy for spiking neural networks from static to event domain. *AAAI 2024*.

### Semantic Role Labeling
- Palmer, M., Gildea, D., & Kingsbury, P. (2005). The proposition bank: An annotated corpus of semantic roles. *Computational Linguistics*, 31(1), 71–106.
- Shi, P., & Lin, J. (2019). Simple BERT models for relation extraction and semantic role labeling. *arXiv:1904.05255*.

### Numbers in Neural Representations
- Gosmann, J., & Eliasmith, C. (2019). Vector-derived transformation encoding: An improved approach to encoding linear transformations in vectors. *Neural Computation*, 31(5), 849–869.
- Plate, T.A. (2000). Analogy retrieval and processing with distributed vector representations. *Expert Systems*, 17(1), 29–40.

### Cognitive Architectures
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Eliasmith, C., et al. (2012). A large-scale model of the functioning brain. *Science*, 338(6111), 1202–1205.
- Anderson, J.R., et al. (2004). An integrated theory of the mind. *Psychological Review*, 111(4), 1036–1060.

### Neuro-Symbolic AI (2024 Reviews)
- Sarker, M.K., et al. (2021). Neuro-symbolic artificial intelligence: Current trends. *AI Communications*, 34(3), 197–209.
- Yu, H., et al. (2024). Neuro-symbolic AI in 2024: A systematic review. *arXiv:2501.05435*.
- Neubert, P., et al. (2024). Classification using hyperdimensional computing: A review with new research directions. *Artificial Intelligence Review*, 58(2).

---

*This document is maintained alongside the codebase. Each completed phase should be checked off in the Implementation Plan and the relevant test results added to Section 9.*
