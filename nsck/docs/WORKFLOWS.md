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

## Workflow 8: Model Transplantation (V15)

Transfer learned knowledge from any pre-trained neural network into NSCK.

```
  config = NSCKConfig.transplant()
  substrate = NSCKSubstrate(config)
       │
       ▼
  substrate.transplant(
       model         = bert_model,
       domain_name   = "language",
       strategy      = "svd_factored",      # or "random" / "learned"
       calibration_epochs = 10,             # 0 = skip STDP fine-tuning
       save_pack_path = "language.kp",      # optional: persist as KnowledgePack
       cognitive_engine = substrate.engine  # optional: inject on pass
  )
       │
       ▼ TransplantPipeline.run()
  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 1 — HARVEST (harvester.py)                                │
  │   ModelHarvester().harvest(model, method="auto")                │
  │   ├── probe LM attrs: embeddings, embed_tokens, wte             │
  │   ├── probe vision attrs: patch_embed, features, conv1          │
  │   ├── probe encoder-decoder: model.encoder                      │
  │   └── fallback: largest 2-D matrix in named_parameters()        │
  │   → HarvestResult(embeddings[N×d], vocab_mapping, model_type)   │
  └──────────────────────────────────────────────────────────────┬──┘
                                                                 │
  ┌──────────────────────────────────────────────────────────────▼──┐
  │ Stage 2 — PROJECT (projector.py)                                │
  │   SVDFactoredProjector:                                         │
  │   a. SVD(E − mean) → keep k=128 principal components           │
  │   b. Per-component FPE codebook (seed=j*31+7777)                │
  │   c. Role HVs (seed=(j*1013+5003)%2^32)                        │
  │   d. Encode each token: ⊕ over all components                  │
  │   → Dict[token → HyperVector]  (codebook)                      │
  └──────────────────────────────────────────────────────────────┬──┘
                                                                 │
  ┌──────────────────────────────────────────────────────────────▼──┐
  │ Stage 3 — CALIBRATE (calibrator.py)  [optional]                 │
  │   STDPCalibrator: for each epoch:                               │
  │   a. Simulate n_pairs token pairs through PythonSnnCore         │
  │   b. Re-encode tokens: mean_firing_rate ≥ threshold → bits      │
  │   c. Measure Recall@10; keep best codebook; early-stop          │
  │   → CalibratedResult(codebook, snn_weights, quality_curve)      │
  └──────────────────────────────────────────────────────────────┬──┘
                                                                 │
  ┌──────────────────────────────────────────────────────────────▼──┐
  │ Stage 4 — VALIDATE (validator.py)                               │
  │   TransplantValidator:                                          │
  │   a. Spearman ρ on 2000 random pairs                            │
  │   b. Recall@10 and Recall@50 over 200 queries                   │
  │   c. ARI: k-means (k=10) in embedding vs HV space               │
  │   d. Per-token quality → worst/best concept lists               │
  │   passed = (ρ≥0.80 AND R@10≥0.70 AND R@50≥0.60 AND ARI≥0.65)  │
  │   → TransplantReport                                            │
  └──────────────────────────────────────────────────────────────┬──┘
                                                                 │
                                           report.passed == True?
                                     yes ──┤           no → return report
                                           │
  ┌──────────────────────────────────────────────────────────────▼──┐
  │ Stage 5 — INTEGRATE                                             │
  │   For each token in codebook:                                   │
  │     sem.add_concept(token, {domain, token})                     │
  │     sem.concept_hvs[token] = codebook[token]                    │
  │   For each pair where similarity > 0.7:                         │
  │     sem.add_relation(t_a, "similar_to", t_b)                    │
  │   (capped at 500 tokens to bound O(n²) cost)                    │
  └──────────────────────────────────────────────────────────────┬──┘
                                                                 │
  ┌──────────────────────────────────────────────────────────────▼──┐
  │ Stage 6 — SAVE (optional)                                       │
  │   KnowledgePack(name=domain_name)                               │
  │     .add_concept(), .add_relation()                             │
  │     .save(save_pack_path)                                       │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  TransplantReport returned to caller
```

### Quick-Start Example

```python
from python.core.substrate import NSCKSubstrate
from python.core.integration.config import NSCKConfig

config = NSCKConfig.transplant()
substrate = NSCKSubstrate(config)

import torch
bert = torch.hub.load(
    'huggingface/pytorch-transformers', 'model', 'bert-base-uncased'
)
report = substrate.transplant(
    model=bert, domain_name="language", strategy="svd_factored",
    calibration_epochs=0, save_pack_path="language.kp"
)
print(f"Passed: {report.passed}, ρ={report.spearman_rho:.3f}, R@10={report.recall_at_10:.3f}")
```

---

## Workflow 9: V17 Enrichment

V17 enrichment modules post-process decisions at various pipeline stages.
Enable all via `NSCKConfig.v17()`.

```
  config = NSCKConfig.v17()
  substrate = NSCKSubstrate(config)
       │
       ▼
  substrate.process(input)
       │
       ▼  (Stage A — Perceptual Enrichment)
  ┌─────────────────────────────────────────────────────────┐
  │ PerceptualEnricher.enrich(packet)                       │
  │   ├── normalise situation HV                            │
  │   ├── bundle temporal context from rolling window (8)   │
  │   └── compute confidence score                          │
  │   → EnrichedPercept(confidence, temporal_ctx_available) │
  │   Runs AFTER Adapter.encode() and BEFORE GWT broadcast  │
  └──────────────────────────────────────────────┬──────────┘
                                                 │
       ▼  (Stage B — Causal Enrichment)
  ┌─────────────────────────────────────────────────────────┐
  │ CausalEnricher.enrich(cause, effect, strength)          │
  │   ├── lookup n_context=3 nearest neighbours in SemanticMemory │
  │   ├── bundle context into enriched causal HV            │
  │   └── store enriched triple back                        │
  │   → CausalTrace(context_concepts, enrichment_steps)     │
  │   Called during CognitiveEngine causal reasoning step   │
  └──────────────────────────────────────────────┬──────────┘
                                                 │
       ▼  (Stage C — Semantic Enrichment)
  ┌─────────────────────────────────────────────────────────┐
  │ SemanticEnricher.enrich_concept(concept, relation, tgt) │
  │   ├── add inverse relation (is_a → sub_class_of, etc.)  │
  │   ├── track co-query counts                             │
  │   └── rebundle frequently co-queried pairs              │
  │   → EnrichmentReport                                    │
  │   Called after SemanticMemory.add_concept() / queries   │
  └──────────────────────────────────────────────┬──────────┘
                                                 │
       ▼  (Stage D — CrossModal Enrichment)
  ┌─────────────────────────────────────────────────────────┐
  │ CrossModalEnricher.link_modalities(anchor, pairs)       │
  │   ├── register modality-specific concepts under anchor  │
  │   └── detect cross-modal clusters (similarity > 0.7)   │
  │   → CrossModalEnrichmentReport                          │
  │   Called when multi-modal concepts are registered       │
  └──────────────────────────────────────────────┬──────────┘
                                                 │
       ▼  (Stage E — Glass-Box Tracing, every stage)
  ┌─────────────────────────────────────────────────────────┐
  │ GlassBoxTracer                                          │
  │   tracer.begin_decision()                               │
  │   with tracer.span("perception"):                       │
  │       tracer.record("TextAdapter", "encoded...", 0.9)   │
  │   with tracer.span("reasoning"):                        │
  │       tracer.record("CognitiveEngine", "rule R42", 0.75)│
  │   trace = tracer.end_decision()                         │
  │   → DecisionTrace with full step-by-step log            │
  └─────────────────────────────────────────────────────────┘
```

### GlassBoxTracer Usage

```python
from python.core.cognitive.glass_box_tracer import GlassBoxTracer

tracer = GlassBoxTracer(max_history=100, enabled=True)
decision_id = tracer.begin_decision()

with tracer.span("perception"):
    tracer.record("TextAdapter", "encoded 'navigate to kitchen'", confidence=0.92)

with tracer.span("reasoning"):
    tracer.record("ProceduralMemory", "LSH hit: navigate skill", confidence=0.87)

trace = tracer.end_decision()
print(tracer.format_trace(trace))
# Span [perception]:
#   [TextAdapter] encoded 'navigate to kitchen'  conf=0.92
# Span [reasoning]:
#   [ProceduralMemory] LSH hit: navigate skill  conf=0.87
# Elapsed: 0.3 ms
```

---

## Workflow 10: Knowledge Pack Load

Loading and injecting a pre-built domain knowledge pack at startup.

```python
config = NSCKConfig(knowledge_packs=[
    "nsck/data/knowledge_packs/conceptnet_en_50k.kp",
    "my_domain.kp",
])
substrate = NSCKSubstrate(config)
# NSCKSubstrate.__init__() auto-loads all packs:
#   for path in config.knowledge_packs:
#       pack = KnowledgePack.load(path)
#       stats = pack.inject_into(self.engine)
#       → {"concepts": N, "relations": M, "causal_links": K}
```

Manual injection:

```python
from python.core.integration.knowledge_pack import KnowledgePack

pack = KnowledgePack.load("my_domain.kp")
stats = pack.inject_into(substrate.engine)
print(stats)  # {"concepts": 50000, "relations": 200000, "causal_links": 10000}
```

---

*Document updated for NSCK V4, February 2026.*

---

## V5 Workflow Additions

### V5 Sleep Workflow — Drift Check + EWC Consolidation

In V5, the `sleep()` workflow includes two new steps:

```
sleep(task_tag)
    │
    ├── 1. PatternGeneralizer.generalize()
    │       └── L2 prototype normalization (V5) — p̂ = p / ‖p‖₂
    │           prevents prototype drift across consolidation cycles
    │
    ├── 2. ContinualLearner.consolidate_task(task_tag)  ← V5 new
    │       └── EWC importance update for task parameters
    │
    └── 3. DriftDetector.check(concept, hv)             ← V5 new
            └── snapshot top-50 concept HVs from SemanticMemory
                flags concepts that drift > threshold
```

**V5 drift check code path (substrate.py):**

```python
def sleep(self, task_tag=None):
    self._engine.sleep(task_tag)
    if self.drift_detector is not None:
        for concept, hv in list(self._engine.semantic_memory.concept_hvs.items())[:50]:
            self.drift_detector.check(concept, hv)
    return {"sleep_cycles": self._engine.stats.get("sleep_cycles", 0)}
```

---

### V5 Learn Workflow — Emotion Modulation

In V5, `learn()` calls `EmotionSystem.update_from_drives()` after the standard
Q-learning update, allowing affect state to modulate future curiosity and
exploration:

```
learn(state, action, reward, task_tag, outcome)
    │
    ├── Q-table update (reward signal)
    ├── EpisodicMemory.store(situation_hv, action, reward, outcome)
    ├── RuleLearner.observe(predicates, action, reward)
    └── EmotionSystem.update_from_drives(     ← V5 new
            drives={
                "reward": reward,
                "novelty": curiosity.compute_novelty(state_hv),
            }
        )
        → Updates valence, arousal, drives
        → get_mood() reflects reward history
```

---

*Document updated for NSCK V5, March 2026.*
