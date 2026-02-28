# Workflows — NSCK V4

NSCK (Neuro-Symbolic Cognitive Kernel) is a neuro-symbolic cognitive architecture
that combines VSA hypervectors, rule learning, and global workspace theory.
Below are the step-by-step data flows for every key workflow in V4.
Classes are referenced by their actual module paths under `nsck/python/core/`.

---

## Workflow 1: Decision Loop (V4)

The primary inference path from raw input to `SubstrateResult`.

```
  raw input (str / dict / list / ndarray / PerceptPacket)
       │
       ▼
  NSCKSubstrate._encode_single()
       │  selects adapter by Python type
       ▼
  Adapter.encode()
  (TextAdapter | DictStateAdapter | NumericAdapter
   | ImageAdapter | AudioAdapter | MultimodalFuser)
       │
       ▼
  PerceptPacket
  { situation_hv, active_predicates, confidence }
       │
       ▼
  [V4] VSANLUEngine.process(text)   ← primary NLU (replaces NgramNLU)
       │  intent + entity extraction via VSA prototype matching
       │
       ▼
  CognitiveEngine.decide(packet, task_tag)
       │
       ├─► [V4] ProceduralMemory fast-path (LSH O(1), threshold 0.72)
       │        → HIT: return cached (action, confidence) immediately
       │        → MISS: fall through to full pipeline
       │
       ├─►  1. Input normalization → raw_state_dict
       ├─►  2. GroundingVerifier.get_active_predicates
       ├─►  3. CuriosityModule.should_explore()
       ├─►  4. Build coalition proposals
       │        RULES | MEMORY | EXPLORATION
       │        Q_LEARNING | PLANNER | MATH | EXTERNAL
       ├─►  5. ActiveInferenceLearner adjusts salience
       ├─►  6. GlobalWorkspace.compete()
       ├─►  7. SafetyGateVerifier.gate_decision()
       ├─►  8. Normalize action to allowed set
       ├─►  9. ExplanationGenerator.explain()
       └─► 10. Return CognitiveState / SubstrateResult
```

---

## Workflow 2: Learning (V4 — auto-cache on positive reward)

```
  ┌──────────────────────────────────────────────────────────┐
  │  substrate.ingest(input, task_tag)                       │
  │                                                          │
  │  1. ProceduralMemory.recall(situation_hv)  [LSH O(1)]   │
  │     ├── HIT  → return cached action (fast path)          │
  │     └── MISS → fall through                              │
  │                                                          │
  │  2. Full process() pipeline (Workflow 1)                 │
  │     → SubstrateResult                                    │
  └──────────────────────────────────────────────────────────┘
                      │
                      ▼  environment executes action
  ┌──────────────────────────────────────────────────────────┐
  │  substrate.feedback(action, reward, task_tag)            │
  │  OR  engine.learn(state, action, reward)                 │
  │                                                          │
  │  3. Q-value TD(0) update                                 │
  │  4. RuleLearner.observe() + induce_rules()               │
  │     EWC-aware pruning (V4): composite score              │
  │     confidence × (1 + ewc_importance)                    │
  │  5. EpisodicMemory.store(episode)                        │
  │  6. ConformalWrapper.calibrate(prediction, actual)       │
  │  7. [V4 NEW] if reward > 0.0 AND situation_hv is not None:
  │        ProceduralMemory.cache_skill(context_hv, action, reward)
  └──────────────────────────────────────────────────────────┘
```

---

## Workflow 3: Sleep Consolidation (V4 — EWC task consolidation)

Offline learning invoked periodically (e.g. after N episodes).

```
  sleep()
    │
    ├─► 1. EpisodicMemory.consolidate()
    │      compress + merge similar episodes; rebuild LSH indices
    │
    ├─► 2. RuleLearner.induce_rules()
    │      promote frequent (state,action,reward) patterns
    │      EWC-aware pruning protects high-importance rules (V4)
    │
    ├─► 3. SemanticMemory.build_prototypes()
    │      auto-generalize concept clusters (min_members=2)
    │
    ├─► 4. SemanticMemory.infer_transitive()
    │      discover transitive relations (is_a depth 3, causes depth 2)
    │
    ├─► 5. PatternGeneralizer.generalize()
    │      generalize patterns from recent examples
    │
    ├─► 6. ConceptDriftDetector.check()
    │      monitor semantic stability across tasks
    │
    └─► 7. [V4] EWC task consolidation
           ewc_importance updated from gwt_win_count for each rule
```

---

## Workflow 4: Knowledge Seeding — NEW V4

Declarative domain bootstrapping from YAML domain kits.

```
  YAML Domain Kit (navigation.yaml / scheduling.yaml / custom)
      │
      ▼
  KnowledgeSeeder.seed_from_yaml(yaml_path, engine)
      │
      ├── semantic_concepts → SemanticMemory.add_concept()
      ├── causal_rules → RuleLearner.learned_rules[domain]
      ├── causal_graph → CausalGraph.add_causes()
      └── high-confidence rules (≥ 0.8) → ProceduralMemory.cache_skill()
      │
      ▼
  engine.seed_domain("navigation.yaml")  ← thin wrapper
      └── Returns: int (number of rules seeded)
```

### Steps in Detail

1. Load YAML domain kit (`navigation.yaml`, `scheduling.yaml`, or custom)
2. Inject semantic concepts into `SemanticMemory`
3. Inject causal rules directly into `RuleLearner` (bypassing learn/induce cycle)
4. Build causal graph edges
5. Cache high-confidence rules (≥ 0.8 confidence) as `ProceduralMemory` skills

**Result**: Engine starts with domain knowledge, reducing cold-start from 50+ episodes to 0.

---

## Workflow 5: Spreading Activation with Rust — NEW V4

The Rust `spreading_activation_step` free-function is the hot path for
`SemanticMemory.spread_activation()`.

```
  SemanticMemory.spread_activation(seed_concepts, steps, decay)
      │
      ▼
  semantic_memory_shim.spread_activation_fast(concept_graph, ...)
      │
      ├── [Rust available] _rust_step_fn(activation, edges, decay, max_frontier)
      │       → Dict[concept → activation_float]
      │       (V4: _rust_step_fn captured at import, not per call)
      │
      └── [Rust unavailable] Python graph traversal fallback
              for each step:
                  for each (u, v) edge in frontier:
                      A'[v] += decay × w(u,v) × A[u]
      │
      ▼
  _update_hot_cache(activated_concepts)   [V4 new]
      └── LRU cache of top-256 concept HVs updated
```

---

## Workflow 6: Procedural Fast-Path — V4 Enhanced

LSH bucket check enables O(1) average candidate lookup.

```
  ProceduralMemory.recall_action(query_hv)
      │
      ▼
  [V4] lsh_bucket(query_hv.bits, n_bits=16, seed=0xDEAD) → bucket_key
      │
      ▼
  Look up _lsh_index[bucket_key]  → List[Skill candidates]
      │
      ├── For each candidate:
      │       sim = similarity(query_hv, skill.context_hv)
      │       if sim ≥ 0.72 (V4 threshold, was 0.85):
      │           return (skill.action, sim, skill.reward)
      │
      └── No match → return None (fall through to full decide())
```

**Complexity:** O(1) average (LSH bucket lookup + small candidate list)
vs. O(N) linear scan in V3.

---

## Workflow 7: REST API Flow

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

*Document updated for NSCK V4, February 2026.*
