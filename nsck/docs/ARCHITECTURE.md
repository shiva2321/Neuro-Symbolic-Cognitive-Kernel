# NSCK Architecture

Complete system architecture of the Neural-Symbolic Cognitive Kernel. All diagrams, LOC counts, and arrows reflect the actual codebase — every link is a verified import or method call.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [VSA Foundation](#2-vsa-foundation)
3. [Memory Systems](#3-memory-systems)
4. [Reasoning Layer](#4-reasoning-layer)
5. [Perception Layer](#5-perception-layer)
6. [Learning Subsystem](#6-learning-subsystem)
7. [Cognitive Layer](#7-cognitive-layer)
8. [Language Layer](#8-language-layer)
9. [Integration Layer](#9-integration-layer)
10. [Central Orchestrator — CognitiveEngine](#10-central-orchestrator--cognitiveengine)
11. [Decision Loop Data Flow](#11-decision-loop-data-flow)
12. [Rust Concurrent Layer](#12-rust-concurrent-layer)
13. [nsck_ai_model Architecture](#13-nsck_ai_model-architecture)

---

## 1. System Overview

NSCK is structured as 8 subsystems, all orchestrated by a single `CognitiveEngine`. The subsystems communicate through the `CognitiveEngine` and the `GlobalWorkspace` broadcast channel.

```mermaid
graph TB
    subgraph S1["1 · VSA Foundation"]
        HV["HyperVector Engine\nhypervec_py.py · 361 LOC\nhypervec_shim.py · 320 LOC\nrust_vsa/ · 7 files"]
    end

    subgraph S2["2 · Memory"]
        Epi["EpisodicMemory\nepisodic_memory.py · 501 LOC\nHot tier (deque) + Warm tier (SQLite)\nLSH k-NN retrieval"]
        Sem["SemanticMemory\nsemantic_memory.py · 291 LOC\nNetworkX DiGraph + HV index\nSpreading activation"]
        SR["StagedRecall\nstaged_recall.py · ~134 LOC"]
    end

    subgraph S3["3 · Reasoning"]
        CE["CognitiveEngine\ncognitive_engine.py · 1,402 LOC"]
        GWT["GlobalWorkspace\nglobal_workspace.py · 312 LOC"]
        Caus["CausalDiscovery + CausalGraph\ncausal_reasoning.py · 1,233 LOC"]
        RL["RuleLearner\nrule_learner.py · 644 LOC"]
        Plan["STRIPSPlanner\nplanner.py · 296 LOC"]
        Ana["AnalogyEngine\nanalogy.py · 609 LOC"]
        Ctx["ContextEngine\ncontext_engine.py · 440 LOC"]
        Math["MathReasoner\nmath_reasoning.py\nFPE + algebra + word problems"]
        Spatial["SpatialReasoner\nspatial_reasoning.py · V5\nFPE bit-flip positions · 8 relations"]
        Temporal["TemporalReasoner\ntemporal_reasoning.py · V4\ntime points · before/after/during"]
        Abduct["AbductiveReasoner\nabductive_reasoning.py · V4\nbest explanation selection"]
        Predict["PredictiveProcessor\npredictive_processor.py · V4\nerror minimisation · precision"]
    end

    subgraph S4["4 · Perception"]
        SNN["SNNPerceptionModule\nsnn_perception.py · 874 LOC\nLIF neurons + STDP"]
        Br["VSA-SNN Bridge\nvsa_snn_bridge.py · ~461 LOC\nRate/Temporal coding"]
        Gr["GroundingVerifier\ngrounding_verifier.py · 585 LOC"]
        MM["MultimodalProcessor + ConcurrentMultimodalScheduler\nmultimodal_processor.py · 788 LOC\nHOG · color · LBP · edges\nV8: attention-weighted coherence fusion"]
    end

    subgraph S5["5 · Learning"]
        Hebb["HebbianMatrix + VSAHebbianLearner\nhebbian.py · 506 LOC\nOja's rule"]
        Cur["CuriosityModule\ncuriosity.py · 336 LOC"]
        XD["TransferEngine\ncross_domain.py\nSchema + Structure + Rule lifting"]
        Schema["SchemaInduction\nschema_induction.py · V4\npattern abstraction · slot filling"]
        PMI["PMILearner\npmi_learner.py · V4\nPointwise Mutual Information"]
        PC["PredictiveCodingModule\npredictive_coding.py · V4\nPRIOR_UNCERTAINTY=0.5"]
        AI["ActiveInferenceLearner\nactive_inference.py · V8\nfree-energy F=pred_error−epi_value\ncoalition salience bias"]
    end

    subgraph S6["6 · Cognitive"]
        Emo["EmotionSystem\nemotion_system.py · 420 LOC\nPluchtik 8 + Circumplex"]
        Meta["SafetyGate + MetacognitiveEngine\nmetacognition.py · 504 LOC"]
        Self["SelfModel\nself_model.py · ~255 LOC"]
        ToM["TheoryOfMind\ntheory_of_mind.py · 312 LOC\nV8: model_other_agent()\nV8: perspective_take()"]
    end

    subgraph S7["7 · Language"]
        TKL["TextKnowledgeLearner\ntext_knowledge_learner.py · 1,074 LOC\n_STOP_CONCEPTS 53 words (V7)\nDistributionalCodebook pre-trained (V7)"]
        LM["LanguageModule + LinguaCortex\nlanguage_module.py · 526 LOC\nlingua_cortex.py · ~260 LOC"]
        DM["DialogueManager\ndialogue_manager.py · 506 LOC\nFluentNLG wired (V7)\nV8: rolling history HV\nV8: topic-shift detection\nV8: clarification_request()"]
        UI["UniversalInput\nuniversal_input.py · 947 LOC"]
        SRL["SemanticRoleLabeler\nsemantic_roles.py\n12 thematic roles · resonator"]
        NLG["NLGEngine + DiscoursePlanner\nnlg.py\nconnectives · anaphora · steps"]
        FlNLG["FluentResponseComposer + NSCKResponseEngine\nfluent_nlg.py · V6\nRelationVerbalizer · context-aware prose"]
        POS["BrillPosTagger\npos_tagger.py · V6\n300+ lexicon · 8 suffix rules"]
        Prag["PragmaticEngine\npragmatics.py · V5\n15 Horn scales · 7 speech acts\nGricean maxims · presuppositions"]
        HFLoad["HuggingFace corpus loader\nhf_corpus_loader.py · V7\noffline fallback · streaming"]
    end

    subgraph S8["8 · Integration"]
        BS["BrainStore (SQLite)\npersistence.py · 852 LOC"]
        BF["BrainFusion + MultiAgentSession\nbrain_fusion.py · 481 LOC\nV8: exchange_snapshots()\nV8: negotiate_beliefs()"]
        EG["ExplanationGenerator\nexplanation.py · 433 LOC"]
        Conf["NSCKConfig\nconfig.py · ~120 LOC\n34 feature flags\nenable_fluent_dialogue=True (V7)\nenable_hf_corpus=False (V7)\nenable_active_inference=False (V8)\nenable_concurrent_multimodal=False (V8)"]
    end

    CE --> GWT --> RL & Caus & Plan & Ana
    CE --> Epi & Sem & SR
    CE --> HV
    CE --> SNN --> Br --> Gr
    CE --> MM
    CE --> Hebb & Cur
    CE --> Emo & Meta & Self & ToM
    CE --> TKL & LM & DM & UI
    CE --> BS & BF & EG & Conf
```

---

## 2. VSA Foundation

All knowledge in NSCK is represented as **10,240-bit binary hypervectors**. The VSA operations define an algebra over this space.

### Class diagram

```mermaid
classDiagram
    class HyperVectorPy {
        +bits: ndarray int8 10240
        +xor(other) HyperVectorPy
        +bundle(other) HyperVectorPy
        +weighted_bundle(other, weight) HyperVectorPy
        +permute(shift) HyperVectorPy
        +permute_inverse(shift) HyperVectorPy
        +negate() HyperVectorPy
        +similarity(other) float
        +cosine_similarity(other) float
        +similarity_robust(other, method) float
        +lsh_hash(seed, n_bits) int
        +from_bits(bits)  HyperVectorPy
        +zero()  HyperVectorPy
    }

    class CleanupMemory {
        +memory: Dict str ndarray
        +max_size: int
        +register(label, hv, force)
        +cleanup(noisy_hv, threshold) tuple
        +cleanup_or_keep(noisy_hv, threshold) HyperVectorPy
        +batch_register(vectors)
        +get_stats() dict
        +clear()
    }

    HyperVectorPy ..> CleanupMemory : registered in
```

**Key operations:**

| Operation | Code | Role |
|---|---|---|
| `xor(A, B)` | element-wise XOR | Binding — encodes role–filler pairs |
| `bundle(A, B)` | majority vote; tie-breaking mask seeded from input bits | Superposition — "both A and B"; deterministic for the same pair |
| `permute(k)` | `np.roll(bits, -k)` | Encodes position / temporal order |
| `negate(A)` | `xor(A, NEG_SEED_HV)` | VSA anti-bundling — inverts membership; `sim(A, negate(A))≈0.50`; `negate(negate(A))==A` |
| `similarity(A, B)` | `1 - Hamming/d` | Associative lookup distance |
| `cosine_similarity(A, B)` | bipolar dot product / d | Noise-robust similarity |

**`hypervec_shim.py`** — backend selector: tries `import hypervec_rs` (Rust); falls back to `HyperVectorPy` (NumPy). All NSCK modules import from the shim; switching backends requires no code changes.

---

## 3. Memory Systems

### 3.1 Episodic Memory — two-tier with LSH

```mermaid
graph LR
    subgraph EpisodicMemory
        Store["store(live_episode)"]
        Recall["recall(query_hv, k)"]
        LSH["LSH Index\nmultiple random projections → hash buckets"]
        Hot["Hot Tier\ndeque (recent)\nin-memory, fast"]
        Warm["Warm Tier\nSQLite via BrainStore\npersistent, large"]
        Sleep["sleep() consolidation\nhot → warm\nprune low-impact episodes"]
    end

    Store --> Hot
    Store --> LSH
    Recall --> LSH --> Hot & Warm
    Sleep --> Hot --> Warm
```

`LiveEpisode` key fields: `timestamp`, `task_tag`, `situation_hv`, `state`, `action`, `outcome`, `reward`, `emotion`, `tom_beliefs`, `image`, `impact_score`.

Retrieval: (1) hash `query_hv` into LSH buckets, (2) collect candidates from both tiers, (3) rerank by Hamming similarity, (4) return top-k.

### 3.2 Semantic Memory — graph + HV index

```mermaid
graph LR
    subgraph SemanticMemory
        Graph["NetworkX DiGraph\nconcept_graph"]
        HVIdx["HV Index\nconcept_hvs: Dict str HV"]
        RelW["Relation Weights\nis_a=0.9  has_property=0.7\ncauses/leads_to/results_in=0.6\nimplies=0.55  part_of=0.5\nsimilar_to=0.4  semantically_related=0.35"]
        SA["spread_activation(starts, steps, decay)"]
        Q["query(query_hv, k)"]
        NSW["NSW ANN Index (V6)\n_NSWIndex pure-Python fallback\nO(log n) approximate k-NN\nfalls back to exact if hnswlib absent"]
        Infer["infer_transitive(relation, max_hops)\nBFS up relation edges\nreturns count of new inferred edges"]
        Proto["build_prototypes(min_members)\nbundle all member HVs per category\nprototype_hv = argmin dist to members"]
        RustBE["Rust Backend\nSemanticMemoryConcurrent\n(parallel spreading)"]
    end

    Graph --> SA
    RelW --> SA
    HVIdx --> Q
    HVIdx --> NSW
    SA --> RustBE
    NSW --> Q
```

`add_concept(name, props)` — generates HV by binding property role–filler pairs via XOR + bundle.

`spread_activation(starts, steps=3, decay=0.7)` — iterates `steps` times; each active node propagates `activation × decay × edge_weight` to neighbours. Only top-200 activated nodes spread each step (prevents blow-up).

`get_inherited_properties(concept)` — BFS up `is_a` edges; merges properties from general → specific (specific wins).

`infer_transitive(relation_type, max_hops)` (V4) — adds implied edges: if `A→is_a→B` and `B→is_a→C`, asserts `A→is_a→C`.

`build_prototypes(min_members)` (V4) — for each category with ≥ `min_members` instances, bundles all member HVs into a single prototype vector. Based on Rosch (1973) prototype theory: bundle = central tendency of members.

---

## 4. Reasoning Layer

### 4.1 Global Workspace Theory (GWT)

```mermaid
graph TD
    subgraph Coalitions["Coalition sources"]
        Srule["Rule coalition\nRuleLearner → best matching rule"]
        Scaus["Causal coalition\nCausalReasoner → forward chain"]
        Smem["Memory coalition\nEpisodicMemory → best recalled episode"]
        Splan["Planner coalition\nSTRIPSPlanner → first step of plan"]
    end

    subgraph GW["GlobalWorkspace"]
        Comp["compete(coalitions)\nsort by coalition.activation"]
        Rehear["mental_rehearsal(winner)\nWorldModel.predict() → danger check"]
        Veto["Veto if danger_similarity > threshold\ntry next-best coalition"]
        Broad["broadcast winner to all modules"]
    end

    Srule & Scaus & Smem & Splan --> Comp
    Comp --> Rehear
    Rehear --> Veto
    Veto --> Broad
```

`Coalition.activation` = `base_salience + relevance + affect_match + sender_confidence × 0.5`

**Mental rehearsal veto** (Phase 8): before committing, the winning coalition is simulated through `WorldModel.predict()`. If the predicted next state has similarity > `danger_threshold` to a known danger vector, the coalition is vetoed and the next-best coalition is tried.

### 4.2 Causal Reasoning

```
Delta-P(cause, effect) = P(effect | cause) - P(effect | not-cause)

Computed from observed (cause, effect) co-occurrence counts with Laplace smoothing.
Works from as few as 2-5 observations.
```

`CausalGraph.forward_chain(start, max_depth)` — BFS from start following `causes` edges, multiplying strengths.

`CausalGraph.backward_chain(goal, max_depth)` — reverse BFS to find what would need to be true for `goal` to occur.

`CausalGraph.counterfactual(do_X, observe_Y)` — simulates removing cause X and predicts what would change.

### 4.3 Rule Learner

Observes `(state, action, outcome)` tuples and induces:

```
IF {predicate_1, predicate_2, ...} THEN action   [confidence = successes / support]
```

Rules progress through three tenure states: **new** → **bootstrap** → **tenured**. Tenured rules require more counter-evidence to demote (stability). Pruning removes rules below `min_success_rate=0.7`.

Integrates with `GlobalWorkspace` via `WorkspaceModule` interface — the rule coalition is built by finding the highest-confidence rule whose conditions match the current predicates.

### 4.4 STRIPS Planner

A* search over state → action → state transitions:

```
Cost:      depth (uniform cost)
Heuristic: 0 (admissible — guarantees optimal plan length)
Operators: extracted from CausalGraph (learned) + hardcoded bootstrap rules
```

`learn_operators_from_graph(graph, context)` — converts causal links into STRIPS operators: `cause` = precondition, `effect` = add effect.

### 4.5 Analogy Engine

Cross-domain structural alignment:

1. Abstract domain predicates and relations into `AbstractConcept` objects.
2. Compute HV similarity between concept pairs across domains.
3. Build `ConceptMapping` for pairs above threshold.
4. Return `Analogy` with `overall_similarity` and `reasoning_chain`.

---

## 5. Perception Layer

### 5.1 Spiking Neural Network

```mermaid
graph LR
    Input["Sensory input\narray or dict"]
    Encode["Rate or Temporal encoding\nVSA-SNN Bridge"]
    LIF["LIF Neuron Layer\nτ·dV/dt = -(V-V_rest) + I\nspike when V ≥ V_thresh"]
    STDP["STDP learning\nΔw based on spike timing"]
    Hebb2["Hebbian association\nVSAHebbianLearner"]
    ConceptHV["Concept HyperVectors"]

    Input --> Encode --> LIF --> STDP --> Hebb2 --> ConceptHV
```

**Rust acceleration** (`rust_snn/`): `LIFLayer.step()`, `SnnCore.simulate()`, and `StdpEngine.apply()` are all Rayon-parallelised. Pure-Python fallback in `snn_perception.py`.

### 5.2 Symbol Grounding

`GroundingVerifier` maps symbolic predicates to HV-based verification functions. When `CognitiveEngine` needs to check `"obstacle_ahead"`, it calls the registered predicate lambda and optionally validates via HV similarity to a grounded prototype.

### 5.3 Multimodal Processing

`MultimodalProcessor` extracts 5 feature types from images, each converted to a HV:

| Feature | Method |
|---|---|
| HOG | Histogram of Oriented Gradients |
| Color | Mean RGB/HSV per 2×2 spatial quadrant |
| LBP | Local Binary Pattern histogram |
| Edges | Sobel edge density |
| Spatial | Quadrant spatial context |

All feature HVs are XOR-bound and bundled into a single fused image HV.

---

## 6. Learning Subsystem

### 6.1 Hebbian Learning

`HebbianMatrix` — co-occurrence matrix over concept pairs.

Oja's rule (in `rust_snn/src/hebbian.rs`):

```
Δw_ij = η · (x_i · y_j - y_j² · w_ij)
```

Normalized update: prevents weight explosion via the forgetting term `y_j² · w_ij`.

### 6.2 Curiosity Module

```
novelty(s)    = 1.0 - max_similarity(encode(s), known_concept_hvs)
progress      = improvement_rate(success_rate, window=100)
explore_now   = (novelty > threshold) AND (confidence < 0.5)
             OR (learning_progress < stagnation_threshold)
```

Maintains `testable_hypotheses` — concept pairs not yet observed together — and can request specific actions to test them.

---

## 7. Cognitive Layer

### 7.1 Emotion System

Russell's Circumplex model with Plutchik's 8 basic emotions as prototypes:

```
valence  ∈ [-1.0, +1.0]   (negative ↔ positive)
arousal  ∈ [ 0.0,  1.0]   (calm ↔ excited)

emotion = argmin_e distance((valence, arousal), prototype_e)
```

Prototypes: joy (0.8, 0.7), trust (0.5, 0.2), fear (-0.7, 0.8), surprise (0.0, 0.9), sadness (-0.6, 0.2), disgust (-0.5, 0.4), anger (-0.5, 0.8), anticipation (0.3, 0.5).

Phase 2.2 adds **emotion blending** — weighted mix of multiple emotions (e.g. 60% joy + 30% anticipation) and an emotion history for mood trend detection.

### 7.2 Metacognition

`SafetyGate` — applies veto on actions violating safety constraints before GWT competition.

`MetacognitiveEngine` — tracks system-level performance; triggers sleep consolidation or task switching when performance degrades.

### 7.3 Self-Model

```
confidence(task, context) = successes(task, context) / attempts(task, context)
```

Context-aware: uses `(task_tag, context_key)` pairs. Cold-start: uses global average when fewer than `min_samples` observations. Tracks calibration error = `|confidence - actual_success_rate|`.

### 7.4 Theory of Mind

Maintains a belief model for each observed agent:

```python
tom.observe(agent_id, action, state)
tom.predict_action(agent_id, current_state)
tom.get_beliefs(agent_id)   # → inferred goal + confidence
```

---

## 8. Language Layer

### 8.1 Text Knowledge Learner (V7 pipeline)

The TKL pipeline now has 7 stages, with V4-V7 additions shown:

```mermaid
flowchart LR
    Text["Input sentence"]
    POS["BrillPosTagger (V6)\ntag_sentence()"]
    CG["ConstructionMatcher (V3/V4)\n71 constructions\nNegation/Temporal/Conditional"]
    SVO["Extract SVO triples\nheuristic NLP + SRL"]
    Stop["_STOP_CONCEPTS filter (V7)\n53 words removed from KG"]
    Concept["SemanticMemory\nadd_concept() + HV binding"]
    Rel["add_relation()\n_GENERIC_RELATION_THRESHOLD=0.62 (V7)"]
    Causal["CausalGraph\ndetect causal keywords"]
    DistSem["DistributionalCodebook (V7)\nco-occurrence · context HVs"]

    Text --> POS --> CG --> SVO --> Stop --> Concept --> Rel --> Causal
    Concept --> DistSem
```

### 8.2 LinguaCortex — semantic folding

```
word_hv(w)      = HyperVector(hash(w) % 2^32)           # deterministic base HV
context_hv(w,i) = permute(word_hv(w), i)                # position encoding
sentence_hv     = bundle(context_hv(w_1,1), ..., context_hv(w_n,n))
```

### 8.3 Universal Input

Converts any input type to a HV:

| Input type | Method |
|---|---|
| `str` | LinguaCortex semantic folding |
| `dict` | XOR-bind each key–value pair, bundle results |
| `np.ndarray` (image) | MultimodalProcessor → fused image HV |
| numeric | FPE encoding (MathReasoner) or scalar quantisation → HV |

### 8.4 Semantic Role Labeling (`semantic_roles.py`)

VSA-native SRL: identifies *who did what to whom, where, when, why, and how* without any neural network.

**12 thematic roles:** PRED, AGENT, PATIENT, THEME, RECIPIENT, INSTRUMENT, LOCATION, TEMPORAL, MANNER, CAUSE, PURPOSE, NEGATION.

**Pipeline:**
1. Tokenise and lower-case.
2. Find predicate: irregular verbs first (bit, ran, gave…) → morphological pattern (-s/-ed/-ing) → copular fallback.
3. Extract AGENT (NP before verb), PATIENT (NP after verb), PPs for LOCATION/TEMPORAL/etc.
4. Build VSA composite: `S = ⊕ bind(role_hv, filler_hv)` for each filled role.
5. Resonator verification: for each role, `estimate = S ⊕ role_hv`, find nearest filler in codebook.

**Role vectors** are deterministic seeds (60001–60012); **word vectors** are MD5-seeded — both are reproducible across sessions.

### 8.5 NLG with Discourse Planning (`nlg.py`)

`StructuralRealizer` (existing) handles single S-V-O sentences.

`DiscoursePlanner` (new) generates coherent multi-sentence paragraphs from a list of semantic frames:
- Orders frames by relation type (definitions first, causal chains next)
- Inserts discourse connectives appropriate to the relation (causes → "As a result,"; contradicts → "However,")
- Applies pronoun anaphora for repeated subjects ("fire … it …")
- Formats procedural responses as numbered steps

`NLGEngine` exposes both via `generate()` (single frame) and `generate_discourse()` (list of frames).

### 8.6 Fluent NLG (`fluent_nlg.py`) — V6/V7

`FluentResponseComposer` generates fluent English from semantic frame lists. Added in V6, wired into DialogueManager in V7.

**5 query types:** `factual` | `explanatory` | `procedural` | `causal` | `comparative`

`RelationVerbalizer` maps symbolic relation names to natural English:
- `is_a` → "is a kind of" / "is a"
- `causes` → "leads to" / "results in"
- `has_property` → "is known for its" / "is characterized by"
- `inhibits` → "prevents" / "reduces"
- `part_of` → "is part of" / "belongs to"

`NSCKResponseEngine.describe(concept, semantic_memory)`:
1. Query semantic memory for up to `max_relations` edges for the concept
2. Convert triples to frame dicts; detect query type from question word
3. `FluentResponseComposer.compose(frames, query_type, topic)` → fluent paragraph
4. Uses anaphora for repeated subjects and discourse connectives by relation type

### 8.7 BrillPosTagger (`pos_tagger.py`) — V6

Brill transformation-based POS tagger using a 300+ word lexicon and 8 suffix/prefix rules.

**Key rules:**
1. Words ending in `-ing` → VBG (gerund) unless in `_ING_NOUNS`
2. Words ending in `-ed` unless in `_ED_ADJECTIVES` → VBD
3. Words in NEGATORS → NEG tag (not, never, no, …)
4. Words in TEMPORAL_CONNECTIVES → TEMP (before, after, while, …)
5. Words in CONDITIONAL_CONNECTIVES → COND (if, unless, provided, …)
6. Words in SIMILARITY_CONNECTIVES → SIM (like, as, similarly, …)
7. Capitalized non-sentence-initial → NNP (proper noun)
8. Words ending in `-tion`/`-ness`/`-ity`/`-ment`/`-ism` → NN

`tag_sentence(sentence)` → `List[Tuple[str, str]]` (word, POS) pairs, used in TKL and CG matching.

### 8.8 Pragmatics (`pragmatics.py`) — V5

`PragmaticEngine` implements Gricean cooperative pragmatics for dialogue:

- **Horn scales** (15 total): `(all, most, many, some)`, `(always, usually, sometimes)`, `(certain, probable, possible)`, `(and, or)`, `(know, believe)`, …
- **Speech acts** (7): assertion, question, directive, commissive, expressive, declaration, threat
- **Gricean maxims**: Quantity (be informative), Quality (be truthful), Relation (be relevant), Manner (be clear)
- **Presupposition projection**: factive verbs (`know`, `realize`) project their complement as presupposed

### 8.9 DistributionalCodebook (`distributional_semantics.py`) — V3/V7

Co-occurrence based semantic similarity built from a corpus.

**V7 addition:** Pre-trained on `BUILTIN_CORPUS` (200 curated sentences) at initialization. Results: semantically related word pairs achieve similarity 0.52–0.70 vs 0.50 random baseline.

```
co_occur[w1][w2] += 1  (within 5-word window)
context_hv(w) = bundle(permute(word_hv(w2), k) for w2, k in context_window(w))
similarity(w1, w2) = context_hv(w1).similarity(context_hv(w2))
```

### 8.10 HuggingFace Corpus Loader (`hf_corpus_loader.py`) — V7

Streams text from HuggingFace datasets (FineWeb, C4, Wikipedia) for large-scale distributional pre-training. Falls back gracefully when offline.

**Config flag:** `NSCKConfig(enable_hf_corpus=True)` — disabled by default (requires internet).

---

## 8b. Math Reasoning (`math_reasoning.py`)

Pure-Python symbolic math — no neural networks, no `eval()`.

**FPE Codebook** — Fractional Power Encoding via incremental noise accumulation:

```
v_0 = base_bits                          # seed 9876543
v_n = v_{n-1} with FLIP_BITS (64) flipped deterministically
sim(v_n, v_m) decreases monotonically as |n-m| increases
sim(v_1, v_2) ≈ 0.994   sim(v_1, v_50) ≈ 0.775
```

**ExpressionEvaluator** — recursive-descent parser for `+ − × ÷ ** ()`.

**LinearSolver** — solves `ax + b = c` forms in natural notation ("x + 3 = 7" → x=4).

**WordProblemParser** — cue-word classification (add/sub/mul/div) for natural-language arithmetic problems.

---

## 8c. Cross-Domain Knowledge Transfer (`cross_domain.py`)

Formal VSA-based mechanism for lifting knowledge from a source domain and applying it in a target domain.

**Components:**

| Class | Role |
|---|---|
| `SchemaExtractor` | Groups domain rules by relation type → bundled schema HV |
| `StructureMapper` | Finds concept correspondences via HV cosine similarity |
| `RuleLifter` | Substitutes fillers using correspondences → `TransferredInference` |
| `TransferEngine` | Orchestrates all three; exposes `register_*` and `transfer()` API |

**Transfer confidence:**
```
conf(inf) = source_rule.confidence × min(src_mapping.sim, tgt_mapping.sim)
```

**Example:**
```python
engine.register_rule("physics", "force", "causes", "acceleration", 0.9)
engine.register_correspondence("physics", "force", "finance", "return", 1.0)
engine.register_correspondence("physics", "acceleration", "finance", "volatility", 1.0)
results = engine.transfer("physics", "finance")
# → "return causes volatility (conf=0.90)"
```

---

## 9. Integration Layer

### 9.1 BrainStore — SQLite Persistence

| Table | Contents |
|---|---|
| `concepts` | Concept name + serialised HV bits |
| `episodes` | Compressed episode sketches (JSON) + metadata |
| `rules` | Learned rules + support/confidence stats |
| `q_values` | Tabular Q-values `{(state_key, action): float}` |

### 9.2 BrainFusion — Multi-task Knowledge Transfer

`TaskBrain` objects (one per domain) can be merged into a `FusedBrain`. Rules and concepts learned in one domain are available to others via the shared HV space.

### 9.3 ExplanationGenerator

Produces human-readable explanations from cognitive traces:

```python
Explanation(
    winning_module = "RULE",
    action         = "wait",
    reason         = "Rule fired: IF obstacle_ahead THEN wait, confidence=0.82",
    evidence       = ["3 supporting episodes", "causal: obstacle→collision(0.9)"],
    confidence     = 0.82,
)
```

---

## 10. Central Orchestrator — CognitiveEngine

`CognitiveEngine` (`cognitive_engine.py`, 1,402 LOC) instantiates all modules and runs the cognitive loop.

**Public API:**

| Method | Returns | Description |
|---|---|---|
| `register_task(tag, predicates, actions, …)` | None | Register a new domain |
| `decide(state, available_actions, task_tag)` | `CognitiveState` | Main decision loop |
| `record_outcome(reward, task_tag, new_state)` | None | TD(0) Q-update |
| `learn(state, action, reward, next_state, task)` | None | Full learning update |
| `process_dialogue(text)` | `str` | NLU → NLG response |
| `sleep()` | None | Offline consolidation |
| `explain()` | `str` | Natural-language explanation of last decision |
| `transfer(src, tgt, state, preds)` | None | Cross-task analogy transfer |
| `get_stats()` | `dict` | Episode count, rule count, Q-value count |

**`decide()` steps (10 steps, in order):**

```
1.  encode state → situation_hv       (UniversalInput)
2.  extract active predicates          (GroundingVerifier)
3.  curiosity evaluation               (CuriosityModule)
4.  build rule coalition               (RuleLearner)
5.  build causal coalition             (CausalReasoner)
6.  build memory coalition             (EpisodicMemory — skipped in fast_mode)
7.  build planner coalition            (STRIPSPlanner — skipped in fast_mode)
8.  GWT competition                    (GlobalWorkspace.compete)
9.  safety veto check                  (SafetyGate)
10. generate explanation               (ExplanationGenerator)
11. update curiosity                   (CuriosityModule.record_visit)
12. build and return CognitiveState
```

---

## 11. Decision Loop Data Flow

```mermaid
sequenceDiagram
    participant App
    participant CE as CognitiveEngine
    participant UI as UniversalInput
    participant GV as GroundingVerifier
    participant Cur as CuriosityModule
    participant RL as RuleLearner
    participant CR as CausalReasoner
    participant EM as EpisodicMemory
    participant SP as STRIPSPlanner
    participant GW as GlobalWorkspace
    participant SG as SafetyGate
    participant EG as ExplanationGenerator

    App->>CE: decide(state, actions, task_tag)
    CE->>UI: encode(state) → situation_hv
    CE->>GV: extract_predicates(state, task_tag)
    CE->>Cur: evaluate(situation_hv) → explore?

    CE->>RL: build_coalition(preds, task_tag) → rule_coalition
    CE->>CR: build_coalition(preds) → causal_coalition
    CE->>EM: recall(situation_hv, k=5) → memory_coalition
    CE->>SP: plan(preds, goal) → planner_coalition

    CE->>GW: compete([rule, causal, memory, planner])
    GW-->>CE: winner_coalition + action

    CE->>SG: check(winner, state, task_tag)
    Note over SG: Veto if safety constraint violated

    CE->>EG: explain_action(action, state, task_tag, trace)
    EG-->>CE: Explanation

    CE->>Cur: record_visit(situation_hv, task_tag)
    CE-->>App: CognitiveState
```

---

## 12. Rust Concurrent Layer

```mermaid
graph TB
    subgraph RUST["rust_vsa/  (PyO3 extension = hypervec_rs)"]
        LIB["lib.rs\nHyperVector  Vec u64  160 blocks  ChaCha8Rng"]
        CONC["concurrent.rs\nHyperVectorRegistry  DashMap lock-free"]
        SEM2["semantic.rs\nSemanticMemoryConcurrent  Rayon parallel spreading"]
        EPI2["episodic.rs\nEpisodicMemoryConcurrent  RwLock hot tier  parallel k-NN"]
        WP["worker_pool.rs\nCognitiveWorkerPool  Rayon thread pool"]
        PERS["persistence.rs\nPersistentStorage  async SQLite"]
        ASYNC["async_runtime.rs\nAsyncCognitiveRuntime  Tokio"]
    end

    Python["Python (hypervec_shim.py)"] --> LIB & CONC & SEM2 & EPI2 & WP & PERS & ASYNC
```

**Benchmarks:**

| Operation | Speedup |
|---|---|
| HyperVector XOR / permute / similarity | 21–206× |
| Parallel similarity search (1,000 vectors) | ~50× |
| Spreading activation (500-node graph) | ~30× |
| Episodic k-NN retrieval | ~40× |

---

## 13. nsck_ai_model Architecture

```mermaid
graph TB
    subgraph ENGINE["NSCKAIEngine  ai_engine.py  2,790 LOC"]
        NSCK_MODS["NSCK core:\nSemanticMemory · EpisodicMemory\nGlobalWorkspace · EmotionSystem\nCausalGraph · SelfModel\nCuriosityModule · TextKnowledgeLearner\nMultimodalProcessor · ImageGenerator"]
        AI_WRAPPERS["AI wrappers:\nContextRetentionModule · CounterfactualReasoner\nResponseComposer · MathHandler"]
    end

    subgraph PIPELINE["Training"]
        T1["train_on_text(text)\nTKL → SemanticMemory + CausalGraph + EpisodicMemory"]
        T2["train_on_image(img, caption)\nMultimodalProcessor → cross-modal episode"]
        T3["DataPipeline\nHuggingFace WikiText streaming"]
        T4["AutonomousTrainer\ntext → QA → math → conv → image"]
    end

    subgraph CHAT["Chat pipeline  chat()"]
        C1["Encode query → HV"]
        C2["Semantic + Episodic + Spreading"]
        C3["Causal inference + GWT competition"]
        C4["ResponseComposer → 2-sentence response"]
        C5["ThoughtTrace  11 stages"]
    end

    subgraph DASH["Flask Dashboard  dashboard.py  687 LOC"]
        W1["Embedded HTML+JS UI"]
        W2["15 REST API endpoints"]
    end

    ENGINE --> PIPELINE & CHAT & DASH
```

**ThoughtTrace stages:**

| # | Stage | Responsible module |
|---|---|---|
| 1 | Input encoding | `LinguaCortex._encode_text()` |
| 2 | Emotion | `EmotionSystem.update()` |
| 3 | Concept extraction | `TextKnowledgeLearner._extract_concepts()` |
| 4 | Semantic search | `SemanticMemory.query(k=10)` |
| 5 | Episodic recall | `EpisodicMemory.recall(k=5)` |
| 6 | Spreading activation | `SemanticMemory.spread_activation()` |
| 7 | Causal inference | `CausalGraph.forward_chain()` + `backward_chain()` |
| 8 | GWT competition | `GlobalWorkspace.compete()` |
| 9 | Self-model | `SelfModel.get_confidence()` |
| 10 | Curiosity | `CuriosityModule.evaluate()` |
| 11 | Response generation | `ResponseComposer.compose()` |

---

## 14. V8 Subsystems (February 2026)

**Test suite: 1,111 passed, 5 skipped, 4 xfailed** (with Rust extensions active).

```mermaid
graph TB
    subgraph V8["V8 New Capabilities"]
        CMS["ConcurrentMultimodalScheduler\nmultimodal_processor.py\nThreadPoolExecutor · 50ms coherence window\nAttention-weighted HV fusion"]
        MLC["Memory Lifecycle\nsemantic_memory.py\ndecay_concepts(λ) · prune_below(θ)\nEpisodic reconsolidation"]
        DST["Dialogue State Tracker\ndialogue_manager.py\nRolling history HV · topic shift\nclarification_request()"]
        MathW["MathReasoner–GWT\ncognitive_engine.py + math_reasoning.py\nMATH coalition salience=0.95\nis_mathematical() regex classifier"]
        HRN["HierarchicalResonatorNetwork\nresonator.py\nL1: AGENT/VERB/PATIENT\nL2: MODIFIER roles"]
        MAS["MultiAgentSession\nbrain_fusion.py\nexchange_snapshots() · negotiate_beliefs()\nVSA consensus bundling"]
        AIF["ActiveInferenceLearner\nactive_inference.py\nF=prediction_error−epistemic_value\nCoalition salience bias · SafetyGate.check_free_energy()"]
        EVAL["NSCK-Eval Benchmarks\nbenchmarks/\nbAbI · math · transfer\nNLG · dialogue · BenchmarkRunner"]
    end

    CMS --> MLC
    DST --> MathW
    HRN --> MAS
    AIF --> EVAL
```

### V8 Config Flags Added

| Flag | Default | Purpose |
|---|---|---|
| `enable_concurrent_multimodal` | `False` | Use `ConcurrentMultimodalScheduler` |
| `enable_full_rust_snn` | `False` | Full Rust SNN backend |
| `memory_decay_lambda` | `0.01` | Ebbinghaus decay rate λ |
| `memory_prune_threshold` | `0.1` | Prune below this importance score |
| `enable_dialogue_state_tracking` | `False` | Rolling history HV in DialogueManager |
| `enable_hierarchical_srl` | `False` | Use HierarchicalResonatorNetwork |
| `enable_multi_agent` | `False` | MultiAgentSession consensus |
| `enable_active_inference` | `False` | ActiveInferenceLearner in decide() |
| `active_inference_weight` | `0.2` | Weight w for salience adjustment |

### V8 New Files

| File | Purpose |
|---|---|
| `python/core/learning/active_inference.py` | ActiveInferenceLearner — free energy loop |
| `benchmarks/babi_tasks.py` | 20 bAbI-style QA tasks |
| `benchmarks/math_word_problems.py` | 50 arithmetic word problems |
| `benchmarks/cross_domain_transfer.py` | 5 cross-domain transfer tests |
| `benchmarks/nlg_quality.py` | 10 NLG fluency checks |
| `benchmarks/dialogue_coherence.py` | Multi-turn coherence test |
| `benchmarks/runner.py` | BenchmarkRunner orchestrator |
| `tests/unit/multimodal/test_concurrent_multimodal.py` | 20 tests |
| `tests/unit/memory/test_memory_lifecycle.py` | 20 tests |
| `tests/unit/language/test_dialogue_state.py` | 20 tests |
| `tests/unit/reasoning/test_math_integration.py` | 15 tests |
| `tests/unit/vsa/test_hierarchical_resonator.py` | 15 tests |
| `tests/unit/integration/test_multi_agent.py` | 15 tests |
| `tests/unit/learning/test_active_inference_integration.py` | 15 tests |
| `tests/integration/test_benchmarks.py` | 10 tests |

---

## 14. V9 — Modality-Agnostic Cognitive Substrate

### 14.1 PerceptPacket Contract

V9 introduces a **universal percept contract** that decouples perception from cognition.
Every modality adapter converts raw input into a `PerceptPacket` before the reasoning core sees it.

```
Any Modality Input (dict/text/number/ndarray/stream/multimodal)
         ↓
   ModalityAdapter.encode(raw_input, task_tag)
         ↓
   PerceptPacket  ← frozen dataclass — universal currency
         ↓
   CognitiveEngine.decide(percept_or_dict, task_tag)
```

`decide()` accepts both `Dict[str, Any]` (backward-compatible) and `PerceptPacket`.
Dicts are auto-wrapped via the task's registered `DictStateAdapter`.

### 14.2 Adapter Layer

| Adapter | Input | Key Delegation |
|---|---|---|
| `DictStateAdapter` | `dict` | `GroundingVerifier + EpisodicMemory.create_situation_hv()` |
| `TextAdapter` | `str` | `UniversalInput.ground_text()` |
| `NumericAdapter` | `float` or `list` | `UniversalInput.ground_scalar/sequence()` |
| `SNNAdapter` | `np.ndarray` | `SNNPerceptionModule.perceive()` |
| `MultimodalFuser` | `List[PerceptPacket]` | VSA bundle + predicate union |
| `StreamProcessor` | timestamped channel readings | temporal feature extraction → StreamVerifier |

### 14.3 Generalization Pipeline (sleep())

After the existing consolidation steps, `sleep()` now automatically runs:
1. `SemanticMemory.build_prototypes(min_members=2)` — VSA prototype per category
2. `SemanticMemory.infer_transitive("is_a", 3)` + `infer_transitive("causes", 2)`
3. `AnalogyEngine.auto_discover_abstractions()` for every task pair

### 14.4 Lifelong Stability Additions

- `Rule.confidence_history` / `last_fired` / `fire_count` fields
- `_detect_rule_drift()` in `sleep()` — marks rules where recent confidence < 50% of older average
- `MemoryHomeostasis.prune_unused_rules()` — wired into `sleep()`

See `docs/NSCK_V9_SUBSTRATE.md` for the full V9 specification.

---

## 15. V10 — Intelligence Extensions

V10 adds eight new modules that extend the cognitive substrate with richer VSA operations, neural rule scoring, probabilistic NLU, attention-guided GWT competition, and a formal safety layer.

### 15.1 New Module Table

| Module | Path | Role |
|---|---|---|
| `EmbeddingVSABridge` | `vsa/vsa_embedding_bridge.py` | Project dense embeddings (e.g. sentence-transformers) to/from binary HyperVectors |
| `RustConcurrentShim` | `vsa/rust_concurrent_shim.py` | Expose Rust concurrent memory classes with Python fallbacks |
| `FHRRVector` / `FHRRMemory` | `vsa/fhrr.py` | Complex-phasor VSA for differentiable, invertible operations |
| `NgramNLU` | `language/ngram_nlu.py` | Naive Bayes n-gram intent classifier + entity extractor |
| `MultiHeadAttentionGWT` | `reasoning/attention_gwt_bridge.py` | Multi-head attention re-weighting of GWT coalition saliences |
| `RuleNeuralScorer` | `learning/rule_neural_scorer.py` | Online perceptron re-ranking of symbolic rules before GWT |
| `SafetyRuleVerifier` / `SafetyGateVerifier` | `cognitive/safety_verifier.py` | Declarative safety properties; gates decisions before execution |
| `NSCKApiServer` | `api/nsck_api.py` | FastAPI / stdlib HTTP REST wrapper around `CognitiveEngine` |

### 15.2 V10 Decision Loop

The full decision loop with V10 components:

```
CognitiveEngine.decide(state, task_tag)
  │
  ├─ PerceptPacket (V9 adapter)
  │     └─ EmbeddingVSABridge (optional text→HV enrichment)
  │
  ├─ Coalition building
  │     ├─ MATH coalition (MathReasoner)
  │     ├─ RULES coalition
  │     │     └─ RuleNeuralScorer.rank_rules()   ← V10
  │     ├─ EXPLORATION coalition (CuriosityModule)
  │     ├─ Q-LEARNING coalition
  │     ├─ PLAN coalition (STRIPSPlanner)
  │     └─ MEMORY coalition (SemanticMemory)
  │
  ├─ GWT competition
  │     └─ MultiHeadAttentionGWT.rerank()        ← V10 (optional)
  │
  ├─ Safety gate
  │     └─ SafetyGateVerifier.gate_decision()    ← V10
  │
  └─ CognitiveState (action, confidence, explanation)
```

### 15.3 FHRR + Embedding Bridge Workflow

```
Text / embedding → EmbeddingVSABridge.embed_to_hv()
                          ↓ binary HyperVector
                   VSA binding / bundling / similarity

or:

FHRRVector.encode_symbol("dog").bind(FHRRVector.encode_symbol("mammal"))
                          ↓ complex phasor
                   FHRRMemory.store("dog_is_mammal", result)
                   FHRRMemory.retrieve("dog_is_mammal")  → similarity query
```

### 15.4 Rust Concurrent Memory

`rust_concurrent_shim.py` transparently selects the backend at import time:

```
hypervec_rs.SemanticMemoryConcurrent  ← Rust (parallel rayon ops)
        OR
SemanticMemoryConcurrent (Python)     ← pure-Python fallback
```

Call `get_status()` to inspect which backend is active:

```python
from python.core.vsa.rust_concurrent_shim import get_status
print(get_status())
# {'use_rust': True, 'available_classes': [...]}
```

### 15.5 Safety Verification

```python
from python.core.cognitive.safety_verifier import SafetyRuleVerifier, SafetyProperty

verifier = SafetyRuleVerifier()
verifier.add_property(SafetyProperty("max_complexity", "len(conditions) < 20"))
report = verifier.verify_rule(my_rule)
# {'safe': True/False, 'violations': [...], 'score': 0.0-1.0}
```

Default critical properties: `no_runaway` (fire_count < 10 000) and `no_code_injection` (action not in FORBIDDEN_ACTIONS).

---

## 16. V12 — Image & Audio Perception Adapters + NSCKSubstrate

V12 adds two new modality adapters for native image and audio perception, and formalises the `NSCKSubstrate` as the recommended developer-facing API.

### 16.1 New Modules

| Module | Path | Role |
|---|---|---|
| `ImageAdapter` | `adapters/image_adapter.py` | Classical CV features → 65-dim FPE HV; predicates `IMAGE_*` |
| `AudioAdapter` | `adapters/audio_adapter.py` | DSP features (MFCC + spectral) → 23-dim FPE HV; predicates `AUDIO_*` |
| `NSCKSubstrate` | `substrate.py` | Clean public API wrapping `CognitiveEngine`; auto-routes all input types |
| `TimeSeriesEncoder` | `perception/stream_encoder.py` | FPE encoding of numeric sequences (added V11) |

### 16.2 ImageAdapter Feature Pipeline

```
np.ndarray (2D/3D)
  │
  ├─ Spatial 4×4 grid: mean + std per cell  (32 dims)
  ├─ Colour histograms: 8 bins × 3 channels  (24 dims)
  ├─ Sobel edge density                        (1 dim)
  ├─ Global mean, std, channel count           (3 dims)
  ├─ 2×2 quadrant means                        (4 dims)
  └─ Aspect ratio                              (1 dim)
        │ 65-dim feature vector
        ▼
  FPE quantisation (256 bins per dim) + role-binding + VSA bundle
        │
        ▼
  situation_hv  +  active_predicates {IMAGE_BRIGHT, IMAGE_DARK, ...}
```

### 16.3 AudioAdapter Feature Pipeline

```
np.ndarray (1-D float waveform)
  │
  ├─ 13 MFCC coefficients (mel filterbank → log → DCT)
  ├─ 4 normalised energy bands
  ├─ Zero-crossing rate
  ├─ Spectral centroid + rolloff
  ├─ RMS energy + peak amplitude + log-length
        │ 23-dim feature vector
        ▼
  FPE quantisation (256 bins per dim) + role-binding + VSA bundle
        │
        ▼
  situation_hv  +  active_predicates {AUDIO_LOUD, AUDIO_TONAL, ...}
```

### 16.4 NSCKSubstrate API

`NSCKSubstrate` is the recommended entry point for third-party developers. It wraps `CognitiveEngine` with:
- Automatic input-type routing (str → text, 2D ndarray → ImageAdapter, list → NumericSequenceAdapter, …)
- `register_encoder(modality, fn)` for custom modalities
- `process()`, `process_multimodal()`, `learn()`, `sleep()`, `remember()`
- `SubstrateResult` dataclass: `chosen_action`, `confidence`, `explanation`, `predicates`, `trace`, `modalities_processed`, `generalization_triggered`

```python
from python.core.substrate import NSCKSubstrate
import numpy as np

substrate = NSCKSubstrate()
substrate.register_task("demo")

# Text
result = substrate.process("fire detected", "demo")

# Image
img = np.zeros((64, 64, 3), dtype=np.uint8)
result = substrate.process(img, "demo")
assert "IMAGE_DARK" in result.predicates

# Audio
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
result = substrate.process_multimodal({"audio": audio, "text": "beep"}, "demo")

# Custom modality
substrate.register_encoder("thermal", lambda data, tag: my_encoder(data, tag))
```

### 16.5 Honest Limitations

| Limitation | Workaround |
|---|---|
| No rotation-invariant image perception | Pair `ImageAdapter` with `EmbeddingVSABridge` + pretrained CNN |
| No speaker/phoneme recognition | Pair `AudioAdapter` with `EmbeddingVSABridge` + wav2vec2 |
| No gradient-based learning in VSA core | Use FHRR + JAX/numpy for numerical gradients |
| NLU is keyword-match + n-gram classification | Pair with LLM adapter for open-domain NLU |

---

## §17 — V13 Universal Cognitive Substrate Architecture (February 2026)

### 17.1 New Module Graph

```
NSCKSubstrate V13
├── SignalIngestor ─────────────── Universal TypedSignal conversion (any Python input)
├── UniversalHVEncoder ─────────── Adaptive Hebbian FPE encoding; feature importance
├── CrossModalAssociativeMemory ── Modality-agnostic binding (HV XOR trick)
├── ProceduralMemory ────────────── Skill cache; fast-path decisions
├── ConceptDriftDetector ────────── Semantic memory stability monitoring
├── ConformalWrapper ─────────────── Calibrated prediction sets (split conformal)
├── CausalRuleAuditor ────────────── ILP rules × causal graph; per-rule causal_score
├── VideoAdapter + TemporalStreamEncoder ── Frame sequences with temporal decay
└── PatternGeneralizer + CrossDomainTransferPipeline ── Abstraction + transfer
```

### 17.2 V13 Decision Cycle

```
Input
  │
  ▼
SignalIngestor.ingest()           ← universal TypedSignal
  │
  ▼
UniversalHVEncoder.encode()       ← Hebbian-weighted FPE → HV
  │
  ├─ ProceduralMemory.recall_action()  ← fast-path if familiar (→ SubstrateResult)
  │
  ▼
CognitiveEngine.decide()          ← full deliberation
  │ GlobalWorkspace.compete()
  │   └─ KLE uncertainty computed  ← Shannon entropy of activations
  │
  ▼
ConformalWrapper.predict_set()    ← calibrated uncertainty bounds
  │
  ▼
SubstrateResult
  ├── chosen_action, confidence, explanation
  ├── kle_uncertainty               ← V13 glass-box
  ├── uncertainty_bounds            ← V13 conformal
  ├── encoding_stats                ← V13 encoding trace
  └── procedural_hit                ← V13 fast-path flag
```

### 17.3 CausalRuleAuditor Bridge

```
ILP Rule: condition → consequence (confidence)
  │
  ├─ CausalGraph.edges lookup (direct + indirect path)
  │
  ▼
AuditedRule
  ├── causal_score    [0,1]
  ├── combined_score  = (1-w)×conf + w×causal
  └── audit_trace     List[str]  ← glass-box
```

### 17.4 File Locations

| Module | File |
|--------|------|
| `CausalRuleAuditor` | `python/core/reasoning/causal_rule_auditor.py` |
| `SignalIngestor` | `python/core/perception/signal_ingestor.py` |
| `UniversalHVEncoder` | `python/core/vsa/universal_hv_encoder.py` |
| `CrossModalAssociativeMemory` | `python/core/memory/cross_modal_associative_memory.py` |
| `ProceduralMemory` | `python/core/memory/procedural_memory.py` |
| `ConceptDriftDetector` | `python/core/memory/concept_drift_detector.py` |
| `VideoAdapter` + `TemporalStreamEncoder` | `python/core/adapters/video_adapter.py` |
| `ConformalWrapper` | `python/core/learning/conformal_wrapper.py` |
| `PatternGeneralizer` + `CrossDomainTransferPipeline` | `python/core/learning/pattern_generalizer.py` |
