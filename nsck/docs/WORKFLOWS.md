# Workflows — NSCK V13

NSCK (Neuro-Symbolic Cognitive Kernel) is a neuro-symbolic cognitive architecture
that combines VSA hypervectors, rule learning, and global workspace theory.
Below are the step-by-step data flows for every key workflow in V13.
Classes are referenced by their actual module paths under `nsck/python/core/`.

---

## 1. Decision Loop (`NSCKSubstrate.process`)

The primary inference path from raw input to `SubstrateResult`.

```
  raw input (str / dict / list / ndarray / PerceptPacket)
       │
       ▼
  NSCKSubstrate._encode_single()
       │  selects adapter by Python type
       ▼
  Adapter.encode()  ──────────────────────────────────┐
  (TextAdapter | DictStateAdapter | NumericAdapter     │
   | ImageAdapter | AudioAdapter | MultimodalFuser)    │
       │                                               │
       ▼                                               │
  PerceptPacket                                        │
  { situation_hv, active_predicates, confidence }      │
       │                                               │
       ▼                                               │
  CognitiveEngine.decide(packet, task_tag)             │
       │                                               │
       ├─►  1. Input normalization → raw_state_dict    │
       ├─►  2. GroundingVerifier.get_active_predicates │
       ├─►  3. CuriosityModule.should_explore()        │
       │        novelty check → explore vs exploit     │
       ├─►  4. Build coalition proposals               │
       │        RULES | MEMORY | EXPLORATION           │
       │        Q_LEARNING | PLANNER | MATH | EXTERNAL │
       ├─►  5. ActiveInferenceLearner adjusts salience │
       │        by expected free energy                │
       ├─►  6. GlobalWorkspace.compete()               │
       │        → winning Coalition                    │
       ├─►  7. SafetyGateVerifier.gate_decision()      │
       │        → pass or veto                         │
       ├─►  8. Normalize action to allowed set         │
       ├─►  9. ExplanationGenerator.explain()          │
       │        → natural-language rationale           │
       └─► 10. Return CognitiveState                   │
                 │                                     │
                 ▼                                     │
           SubstrateResult (V13 fields)                │
           { chosen_action, confidence, explanation,   │
             kle_uncertainty, uncertainty_bounds,       │
             procedural_hit, encoding_stats }          │
```

### Steps in detail

| # | Component | What happens |
|---|-----------|--------------|
| 1 | `CognitiveEngine` | Extracts `raw_state` dict from `PerceptPacket` for backward-compatible predicate evaluation |
| 2 | `GroundingVerifier` | Evaluates registered predicate lambdas → `List[str]` of active predicates |
| 3 | `CuriosityModule` | Compares `situation_hv` against visited states; high novelty → EXPLORATION coalition |
| 4 | `CognitiveEngine._build_coalitions` | Each source proposes `Coalition(source, action, salience)` |
| 5 | `ActiveInferenceLearner` | Re-weights saliences by expected information gain (free energy) |
| 6 | `GlobalWorkspace.compete` | Scores = salience + relevance + affect + 0.5×confidence; threshold ≥ 0.5 |
| 7 | `SafetyGateVerifier` | Critical violations → veto; warnings logged only |
| 8 | `CognitiveEngine` | Maps winning action onto `available_actions`; falls back to explore |
| 9 | `ExplanationGenerator` | Template-based NL: *"Chose {action} because {reason}"* |
| 10 | `CognitiveEngine` | Packs everything into `CognitiveState` → wrapped as `SubstrateResult` |

---

## 2. V13 Ingest / Feedback Loop

```
  ┌──────────────────────────────────────────────────┐
  │  substrate.ingest(input, task_tag)               │
  │                                                  │
  │  1. ProceduralMemory.recall(situation_hv)        │
  │     ├── HIT  → return cached action (fast path)  │
  │     └── MISS → fall through                      │
  │                                                  │
  │  2. Full process() pipeline (§1 above)           │
  │     → SubstrateResult                            │
  └──────────────────────────────────────────────────┘
                      │
                      ▼  environment executes action
  ┌──────────────────────────────────────────────────┐
  │  substrate.feedback(action, reward, task_tag)    │
  │                                                  │
  │  3. Q-value TD(0) update                         │
  │  4. RuleLearner.observe() + induce_rules()       │
  │  5. EpisodicMemory.store(episode)                │
  │  6. ConformalWrapper.calibrate(prediction, actual)│
  │  7. ProceduralMemory.update()                    │
  │     (store skill when reward > threshold)        │
  └──────────────────────────────────────────────────┘
```

---

## 3. Sleep Consolidation (`NSCKSubstrate.sleep`)

Offline learning invoked periodically (e.g. after N episodes).

```
  sleep()
    │
    ├─► 1. EpisodicMemory.consolidate()
    │      compress + merge similar episodes; rebuild LSH indices
    │
    ├─► 2. RuleLearner.induce_rules()
    │      promote frequent (state,action,reward) patterns
    │
    ├─► 3. SemanticMemory.build_prototypes()
    │      auto-generalize concept clusters (min_members=2)
    │
    ├─► 4. SemanticMemory.infer_transitive()
    │      discover transitive relations (is_a depth 3, causes depth 2)
    │
    ├─► 5. PatternGeneralizer.generalize()         [V13]
    │      generalize patterns from recent examples
    │
    └─► 6. ConceptDriftDetector.check()            [V13]
           monitor semantic stability across tasks
```

---

## 4. Perception Pipeline (Adapter → `PerceptPacket`)

Each modality has a dedicated adapter in `nsck/python/core/adapters/`.

```
  ┌─────────────┬──────────────────────────────────────────────────────┐
  │ Modality    │ Adapter → encoding steps → PerceptPacket            │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Text        │ TextAdapter                                         │
  │             │   tokenize → VSA encode (bind role HVs) → packet    │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Dict        │ DictStateAdapter                                    │
  │             │   encode key-value pairs → bundle HVs → packet      │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Numeric     │ NumericAdapter                                      │
  │             │   FPE quantize → packet                             │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Image       │ ImageAdapter                                        │
  │             │   spatial grid + colour histogram + Sobel edges     │
  │             │   → 65-dim FPE → packet                             │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Audio       │ AudioAdapter                                        │
  │             │   MFCC + spectral features → 23-dim FPE → packet    │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ Multimodal  │ MultimodalFuser                                     │
  │             │   fuse List[PerceptPacket] → combined packet        │
  └─────────────┴──────────────────────────────────────────────────────┘
```

Common output: `PerceptPacket(situation_hv, active_predicates, confidence, modality, raw_state)`.

---

## 5. Memory Recall

### Semantic Memory

```
  query (concept name or HV)
    ├─► SemanticMemory.get_concept(name) — direct graph node lookup
    ├─► SemanticMemory.query(hv) — cosine_sim against concept_hvs → top-k
    └─► SemanticMemory.spread_activation(seed) — propagate edges, decay per hop
```

### Episodic Memory

```
  query (situation_hv)
    → LSH bucket lookup (4 tables × 10-bit hashes)
    → 1-bit neighbour probing
    → exact similarity ranking on candidates
    → top-k episodes (Rust-backed batch search)
```

### Procedural Memory (V13)

```
  query (situation_hv)
    → hash(situation_hv) exact match
    → HIT: return cached (action, reward) — no GWT needed
    → MISS: fall through to full decide() pipeline
```

---

## 6. Rule Learning and Firing

```
  ┌─ Online ────────────────────────────────────┐
  │ 1. Observe (state, action, reward) triple   │
  │ 2. RuleLearner.observe() — count patterns   │
  │ 3. RuleLearner.induce_rules()               │
  │    → Rule(condition, consequence,            │
  │           confidence, fire_count,            │
  │           confidence_history)                │
  └─────────────────────────────────────────────┘
           │
           ▼
  ┌─ Coalition building ────────────────────────┐
  │ 4. RuleNeuralScorer.rank_rules(applicable)  │
  │    perceptron re-ranks by relevance (V10)   │
  │ 5. Top rule → RULES Coalition(action, score)│
  └─────────────────────────────────────────────┘
```

After feedback, `RuleNeuralScorer.update(rule, reward)` adjusts perceptron weights.

---

## 7. Causal Discovery

```
  observe(context, causes, effects)
       │
       ▼
  Accumulate contingency tables
       │
       ├─► _calculate_delta_p()
       │     ΔP (causal strength) = P(effect|cause) - P(effect|¬cause)
       │
       ├─► _mutual_information()
       │     detect confounders via MI
       │
       ▼
  induce_graph()
       │  build CausalGraph from confident links
       │
       ▼
  CausalRuleAuditor.audit_rule()             [V13]
       combined_score = 0.6×confidence + 0.4×causal_score
       audit_trace: human-readable scoring
```

---

## 8. Analogy and Cross-Domain Transfer

```
  AnalogyEngine
    │
    ├─► 1. Receive source + target domain HVs
    ├─► 2. Structural alignment via VSA similarity
    ├─► 3. Compute functoriality score
    │        (how well structure is preserved)
    ├─► 4. MaxEnt threshold for acceptance
    │
    ▼
  CrossDomainTransferPipeline
    │
    ├─► register(hv, domain_label)
    └─► transfer(query_hv, source, target)
         → [{source_pattern, target_pattern, transfer_score}]
```

---

## 9. Language Processing

Full NLU → NLG pipeline through modules in `nsck/python/core/language/`.

```
  raw text
    │
    ├─► 1. NgramNLU.predict()            → intent classification
    ├─► 2. LeftCornerParser.parse()       → phrase tree
    ├─► 3. ConstructionGrammar.match()    → pattern matching
    ├─► 4. FrameSemantics.fill()          → frame filling
    ├─► 5. SemanticRoleLabeler.label()    → agent / patient / theme
    ├─► 6. CoreferenceResolver.resolve()  → pronoun resolution
    ├─► 7. DialogueManager.process_turn() → state tracking, topic shift
    │
    ▼
  FluentNLG / NSCKResponseEngine
    └─► response generation (context-appropriate fluent prose)
```

---

## 10. REST API Flow

Served by `NSCKApiServer` (`nsck/python/core/api/nsck_api.py`).

```
  Client                  NSCKApiServer              CognitiveEngine
    │                          │                           │
    │  POST /decide            │                           │
    │  {state, task_tag}  ───► │  handle_decide()          │
    │                          │  ───► decide(state, tag) ─┤
    │                          │  ◄─── CognitiveState      │
    │  ◄── JSON {action,       │                           │
    │       confidence,        │                           │
    │       explanation}       │                           │
    │                          │                           │
    │  POST /learn             │                           │
    │  {state, action,    ───► │  handle_learn()           │
    │   reward, task_tag}      │  ───► learn(...)     ─────┤
    │  ◄── {status: "ok"}     │                           │
    │                          │                           │
    │  POST /sleep             │                           │
    │  {}                 ───► │  handle_sleep()           │
    │                          │  ───► sleep()        ─────┤
    │  ◄── {status:            │                           │
    │   "sleep_complete"}      │                           │
    │                          │                           │
    │  GET /status             │                           │
    │                     ───► │  handle_status()          │
    │  ◄── system telemetry    │                           │
```

---

## Workflow 10 — Knowledge Bootstrapping (V4)

```
YAML Domain Kit
    │
    ▼
KnowledgeSeeder.seed_from_yaml(yaml_path, engine)
    │
    ├── semantic_concepts → engine.semantic_memory.add_concept()
    ├── causal_rules → engine.rule_learner.learned_rules[domain]
    ├── causal_graph → engine.causal_graphs[domain].add_causes()
    └── high-confidence rules → engine.procedural_memory.cache_skill()
    │
    ▼
engine.seed_domain("navigation.yaml")
    └── Returns: number of rules seeded
```

### Workflow 10 Steps:
1. Load YAML domain kit (navigation.yaml, scheduling.yaml, or custom)
2. Inject semantic concepts into SemanticMemory
3. Inject causal rules directly into RuleLearner (bypassing learn/induce cycle)
4. Build causal graph edges
5. Cache high-confidence rules (≥0.8 confidence) as ProceduralMemory skills

**Result**: Engine starts with domain knowledge, reducing cold-start from 50+ episodes to 0.

---

## Workflow 1 Update — Decision Loop with System-1 Fast-Path (V4)

The System-1 fast-path now auto-populates from `learn()`:

```
learn(state, action, reward=1.0, task_tag)
    │
    ├── [existing] rule_learner.observe(...)
    ├── [existing] episodic_memory.record(...)
    └── [V4 NEW] if reward > 0.0 AND situation_hv is not None:
            procedural_memory.cache_skill(context_hv, action, reward)
```

On subsequent `decide()` calls:
```
decide(state, task_tag)
    │
    └── [V4] Q_LEARNING coalition includes procedural recall hint
            ProceduralMemory.recall_action(query_hv) → (action, similarity, reward)
            [LSH bucket O(1) lookup, threshold 0.72]
```
