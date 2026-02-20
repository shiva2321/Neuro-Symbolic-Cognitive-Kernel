# Module Reference

Complete reference for every file and module in the NSCK codebase. Every class, method, and parameter is taken directly from the source code.

> **Keeping this document current:** This reference was generated from the codebase at the time of writing. If source files change, verify the relevant sections against the actual code. The ground truth is always the source files under `nsck/python/core/` and `nsck_ai_model/`.

---

## Table of Contents

- [VSA Foundation](#vsa-foundation)
- [Memory Systems](#memory-systems)
- [Reasoning Layer](#reasoning-layer)
- [Perception Layer](#perception-layer)
- [Learning Subsystem](#learning-subsystem)
- [Cognitive Layer](#cognitive-layer)
- [Language Layer](#language-layer)
- [Integration Layer](#integration-layer)
- [Multimodal](#multimodal)
- [Training](#training)
- [Rust Accelerators](#rust-accelerators)
- [nsck_ai_model Modules](#nsck_ai_model-modules)
- [Tests Reference](#tests-reference)

---

## VSA Foundation

### `vsa/hypervec_py.py` — 361 LOC

Core data type: 10,240-bit binary hypervectors and the cleanup associative memory.

#### `HyperVectorPy`

```python
HyperVectorPy(seed: Optional[int] = None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `from_bits` | `(bits: ndarray) → HyperVectorPy` | `HyperVectorPy` | Create HV from an existing bit array (classmethod) |
| `zero` | `() → HyperVectorPy` | `HyperVectorPy` | Create all-zero HV (classmethod) |
| `xor` | `(other: HyperVectorPy) → HyperVectorPy` | `HyperVectorPy` | Binding — element-wise XOR |
| `bundle` | `(other: HyperVectorPy) → HyperVectorPy` | `HyperVectorPy` | Superposition — majority vote with random tie-break |
| `permute` | `(shift: int) → HyperVectorPy` | `HyperVectorPy` | Circular bit-shift by `shift` positions |
| `permute_inverse` | `(shift: int) → HyperVectorPy` | `HyperVectorPy` | Inverse circular shift |
| `similarity` | `(other: HyperVectorPy) → float` | `[0, 1]` | Normalised Hamming similarity (legacy) |
| `cosine_similarity` | `(other: HyperVectorPy) → float` | `[-1, 1]` | Bipolar cosine similarity (recommended) |
| `similarity_robust` | `(other, method='cosine') → float` | `[0, 1]` | Configurable similarity (normalises cosine to [0,1]) |

**Constant:** `DIMENSION = 10240`

**Alias:** `HyperVector = HyperVectorPy` (for backward compatibility)

#### `CleanupMemory`

```python
CleanupMemory(max_size: int = 10000)
```

LRU associative memory for denoising hypervectors after binding operations.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `register` | `(label: str, hv: HyperVectorPy, force: bool = False)` | None | Register a clean atomic vector |
| `cleanup` | `(noisy_hv, threshold: float = 0.4) → tuple` | `(HV, label) or (None, None)` | Snap noisy HV to nearest known prototype |
| `cleanup_or_keep` | `(noisy_hv, threshold: float = 0.4) → HyperVectorPy` | `HyperVectorPy` | Cleanup variant — returns original if no match |
| `batch_register` | `(vectors: Dict[str, HV])` | None | Register multiple vectors at once |
| `load_from_store` | `(store)` | None | Load all concepts from BrainStore |
| `get_stats` | `() → dict` | dict | `{size, max_size, capacity_used, total_cleanups, success_rate, most_accessed}` |
| `clear` | `()` | None | Reset all memory |

#### Utility functions

| Function | Signature | Description |
|---|---|---|
| `bundle_with_cleanup` | `(vectors, cleanup_mem, threshold) → HV` | Bundle list then optionally denoise |
| `unbind_with_cleanup` | `(bound_hv, key_hv, cleanup_mem, threshold) → HV` | XOR unbind then optionally denoise |

---

### `vsa/hypervec_shim.py` — 320 LOC

Backend selector. Imports Rust extension if available; falls back to Python.

**Exported names (available from either backend):**

| Name | Type | Description |
|---|---|---|
| `HyperVector` | class | Active HV class (Rust or Python) |
| `SemanticMemoryConcurrent` | class or None | Thread-safe semantic memory (Rust only) |
| `EpisodicMemoryConcurrent` | class or None | Thread-safe episodic memory (Rust only) |
| `CognitiveWorkerPool` | class or None | Parallel computation pool (Rust only) |
| `PersistentStorage` | class or None | Disk persistence (Rust only) |
| `AsyncCognitiveRuntime` | class or None | Async runtime (Rust only) |
| `parallel_similarity_search` | func or None | Batch HV similarity search (Rust only) |

---

### `vsa/resonator.py`

`ResonatorNetwork` — iterative decoding of VSA superpositions using resonator dynamics. Used to factorize a bound HV back into its components when the codebook is known.

---

## Memory Systems

### `memory/episodic_memory.py` — 501 LOC

#### `LiveEpisode` (dataclass)

| Field | Type | Description |
|---|---|---|
| `timestamp` | float | Unix timestamp |
| `task_tag` | str | Domain identifier |
| `situation_hv` | HyperVector | Encoded state |
| `state` | Dict | Full state dict (recent only) |
| `action` | str | Chosen action |
| `outcome` | str | Result string |
| `reward` | float | Scalar reward signal |
| `emotion` | str | Emotion at time of episode |
| `tom_beliefs` | Optional[Dict] | Theory of Mind belief snapshot |
| `image` | Optional[ndarray] | Visual frame (for SNN dreaming) |
| `impact_score` | float | `|reward| + novelty` — priority for pruning |

#### `EpisodicMemory`

```python
EpisodicMemory(store: Optional[BrainStore] = None, hot_capacity: int = 500, lsh_num_tables: int = 8, lsh_key_bits: int = 16)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `store` | `(episode: LiveEpisode)` | None | Store episode in hot tier + LSH index |
| `recall` | `(query_hv: HV, k: int = 5, task_tag: str = None) → List[LiveEpisode]` | list | LSH bucket lookup + Hamming rerank |
| `sleep` | `()` | None | Consolidate hot → warm tier, prune low-impact |
| `get_stats` | `() → dict` | dict | `{hot_size, warm_size, total_stored, lsh_tables}` |
| `register_sketch_extractor` | `(task_tag, extractor)` | None | Register domain-specific state compressor |

---

### `memory/semantic_memory.py` — 291 LOC

#### `SemanticMemory`

```python
SemanticMemory(relation_weights: Optional[Dict] = None, use_rust: bool = True)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `add_concept` | `(name: str, properties: dict, hv_override=None)` | None | Add concept with bound property HVs |
| `get_concept` | `(name: str) → Optional[HV]` | HV or None | Get concept HV |
| `add_relation` | `(c1: str, rel: str, c2: str, timestamp: float = 0.0)` | None | Add directed edge with temporal validation |
| `query` | `(query_hv: HV, k: int = 5) → List[Tuple[str, float]]` | list | Top-k similar concepts |
| `spread_activation` | `(start_concepts: List[str], steps: int = 3, decay: float = 0.7) → Dict[str, float]` | dict | Weighted spreading activation |
| `get_inherited_properties` | `(concept: str) → dict` | dict | BFS up `is_a` edges + merge properties |
| `extract_schema` | `(concept: str) → dict` | dict | Full schema: `{name, properties, is_a, parts, causes, caused_by}` |
| `reset` | `()` | None | Clear all concepts and relations |
| `save` | `(filepath: str)` | None | Pickle to disk |
| `load` | `(filepath: str)` | None | Load from pickle |

**Default relation weights:** `is_a=0.9, has_property=0.7, causes=0.6, part_of=0.5, similar_to=0.4`

---

### `memory/staged_recall.py`

`StagedRecall` — multi-stage recall that first checks semantic memory, then episodic memory, then applies spreading activation across the combined results. Returns a ranked list of evidence items.

---

## Reasoning Layer

### `reasoning/cognitive_engine.py` — 1,402 LOC

The central orchestrator. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full decision loop.

#### `CognitiveEngine`

```python
CognitiveEngine(config: Optional[NSCKConfig] = None, store: Optional[BrainStore] = None, fast_mode: bool = False)
```

**Key attributes:**

| Attribute | Type | Description |
|---|---|---|
| `q_values` | dict | Tabular Q-values `{(state_key, action): float}` |
| `episodic_memory` | EpisodicMemory | Episode storage |
| `semantic_memory` | SemanticMemory | Concept graph |
| `global_workspace` | GlobalWorkspace | GWT competition arena |
| `causal_graph` | CausalGraph | Learned causal model |
| `rule_learner` | RuleLearner | Symbolic rule store |
| `self_model` | SelfModel | Performance tracker |
| `curiosity` | CuriosityModule | Novelty-driven exploration |
| `emotion_system` | EmotionSystem | Affective state |
| `trace_history` | List[dict] | Last `max_trace_history` decision traces |

**Public API:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `register_task` | `(tag, predicates, actions, causal_graph=None, planner=None)` | None | Register domain |
| `decide` | `(state: dict, available_actions: List[str], task_tag: str) → CognitiveState` | CognitiveState | Full decision loop |
| `record_outcome` | `(reward: float, task_tag: str, new_state: dict = None)` | None | TD(0) Q-update |
| `learn` | `(state, action, reward, next_state, task_tag)` | None | Full learning update |
| `process_dialogue` | `(text: str) → str` | str | Natural language response |
| `sleep` | `()` | None | Offline consolidation |
| `explain` | `() → str` | str | Explanation of last decision |
| `why_not` | `(rejected_action: str) → str` | str | Why an action was not chosen |
| `counterfactual` | `(query: str) → str` | str | Counterfactual reasoning |
| `transfer` | `(src_task, tgt_task, state, predicates)` | None | Cross-domain analogy |
| `get_stats` | `() → dict` | dict | System statistics |
| `get_allowed_actions` | `(task_tag: str) → List[str]` | list | Registered actions for task |
| `set_mission_goal` | `(goal: str)` | None | Set planner objective |

#### `CognitiveState` (dataclass)

| Field | Type | Description |
|---|---|---|
| `task_tag` | str | Domain |
| `situation_hv` | Optional[HV] | Encoded state |
| `active_predicates` | List[str] | True predicates in this state |
| `chosen_action` | str | Winning action |
| `confidence` | float | Estimated confidence |
| `self_confidence` | float | Self-model confidence |
| `exploration_mode` | bool | Whether curiosity triggered exploration |
| `explanation` | Optional[Explanation] | Human-readable reasoning trace |
| `trace` | dict | Raw trace dict for telemetry |

#### `Proposal` (dataclass)

| Field | Type | Description |
|---|---|---|
| `module` | str | Source module name |
| `action` | str | Proposed action |
| `content` | str | Detail string |
| `salience` | float | Intrinsic priority score |

---

### `reasoning/global_workspace.py` — 312 LOC

#### `Coalition` (dataclass)

| Field | Type | Description |
|---|---|---|
| `source` | str | Module name |
| `content` | Any | Proposed action or information |
| `base_salience` | float | Intrinsic loudness [0.0–1.0] |
| `relevance` | float | Match with current context/goal |
| `affect_match` | float | Match with emotional drives |
| `sender_confidence` | float | Module's self-reported confidence |
| `activation` | property | `base_salience + relevance + affect_match + confidence×0.5` |

#### `GlobalWorkspace`

```python
GlobalWorkspace(danger_threshold: float = 0.8)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `compete` | `(coalitions: List[Coalition]) → Optional[Coalition]` | Coalition or None | Sort by activation, apply mental rehearsal veto, return winner |
| `broadcast` | `(winner: Coalition)` | None | Send winner content to all registered modules |
| `register_module` | `(module: WorkspaceModule)` | None | Add module to broadcast list |
| `get_rehearsal_stats` | `() → dict` | dict | Veto count and veto rate |

#### `RehearsalEvent` (dataclass)

Records each mental rehearsal veto event: `timestamp`, `vetoed_source`, `vetoed_action`, `danger_similarity`, `cycle`.

---

### `reasoning/causal_reasoning.py` — 1,233 LOC

#### `CausalLink` (dataclass)

| Field | Type | Description |
|---|---|---|
| `cause` | str | Cause predicate |
| `effect` | str | Effect predicate |
| `relation` | CausalRelation | CAUSES / PREVENTS / ENABLES / REQUIRES |
| `strength` | float | P(effect|cause) in [0.0, 1.0] |
| `context` | Optional[str] | Task-specific context |
| `evidence_count` | int | Number of supporting observations |

#### `CausalGraph`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `add_link` | `(link: CausalLink)` | None | Add causal relationship |
| `forward_chain` | `(start: str, max_depth: int = 5) → List[CausalChain]` | list | Follow causes forward |
| `backward_chain` | `(goal: str, max_depth: int = 5) → List[CausalChain]` | list | Trace back to root causes |
| `counterfactual` | `(do_X: str, observe_Y: str) → CounterfactualResult` | CounterfactualResult | Simulate removing X |
| `get_causes` | `(effect: str) → List[CausalLink]` | list | Direct causes of an effect |
| `get_effects` | `(cause: str) → List[CausalLink]` | list | Direct effects of a cause |

#### `CausalDiscovery`

```python
CausalDiscovery()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `observe` | `(context: str, causes: List[str], effects: List[str])` | None | Record one observation |
| `induce_graph` | `(context: str, min_confidence: float = 0.5) → CausalGraph` | CausalGraph | Build graph from observations |
| `register_outcome_keywords` | `(positive: set, negative: set)` | None | Extend outcome keyword sets |
| `generate_hypotheses` | `(observations: list) → List[str]` | list | Suggest causal hypotheses |
| `design_experiment` | `() → dict` | dict | Suggest test to distinguish hypotheses |

**Delta-P formula:** `ΔP(c→e) = P(e|c) - P(e|¬c)` with Laplace smoothing.

#### `CausalReasoner`

Wraps `CausalGraph` + `CausalDiscovery` for high-level reasoning queries.

---

### `reasoning/rule_learner.py` — 644 LOC

#### `RuleLearner` (implements `WorkspaceModule`)

```python
RuleLearner(verifier: GroundingVerifier, store: BrainStore = None, min_support: int = 5, min_confidence: float = 0.3, min_success_rate: float = 0.7, max_rules_per_task: int = 50)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `observe` | `(state: dict, action: str, reward: float, task_tag: str)` | None | Record experience |
| `induce_rules` | `(task_tag: str)` | None | Promote candidates that exceed `min_support` |
| `get_best_rule` | `(active_preds: Set[str], task_tag: str) → Optional[Rule]` | Rule or None | Best rule for current predicates |
| `build_coalition` | `(active_preds, task_tag) → Optional[Coalition]` | Coalition or None | Build GWT coalition from best rule |
| `prune_rules` | `(task_tag: str)` | None | Remove rules below `min_success_rate` |
| `receive_broadcast` | `(content)` | None | `WorkspaceModule` interface |
| `get_stats` | `() → dict` | dict | Rule counts per task |

**Tenure states:** `new` (< `min_support`) → `bootstrap` (supported but not tenured) → `tenured` (stable, high confidence).

---

### `reasoning/planner.py` — 296 LOC

#### `STRIPSPlanner`

```python
STRIPSPlanner(operators: List[Rule] = None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `plan` | `(start_state: FrozenSet[str], goal_state: FrozenSet[str]) → List[PlanStep]` | list | A* search for action sequence |
| `set_operators` | `(rules: List[Rule])` | None | Update available operators |
| `learn_operators_from_graph` | `(graph: CausalGraph, context: str = None)` | None | Extract STRIPS operators from causal graph |
| `set_causal_reasoner` | `(reasoner: CausalReasoner)` | None | Link to causal reasoner |

---

### `reasoning/analogy.py` — 609 LOC

#### `AnalogyEngine`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `find_analogy` | `(source_domain, target_domain) → Optional[Analogy]` | Analogy or None | Find structural mapping between domains |
| `transfer_knowledge` | `(analogy: Analogy, source_concept: str) → Optional[str]` | str or None | Map concept to target domain |
| `register_domain` | `(domain_name: str, concepts: List[AbstractConcept])` | None | Register domain for analogy search |
| `get_reasoning_chain` | `(analogy: Analogy) → List[str]` | list | Human-readable mapping chain |

---

### `reasoning/context_engine.py` — 440 LOC

`ContextEngine` — maintains a sliding window of recent predicates and HVs for context-aware reasoning. Used internally by `CognitiveEngine` for carry-over reasoning across `decide()` calls.

---

### `reasoning/math_reasoning.py`

**Math/numeric reasoning module.** No neural networks. No `eval()`. Pure symbolic computation.

#### `FPECodebook`

Fractional Power Encoding via incremental noise accumulation.

```python
FPECodebook(max_int: int = 1023)
```

| Method | Signature | Returns |
|---|---|---|
| `encode` | `(n: int | float) → HyperVector` | HV for numeric value |
| `similarity` | `(a, b) → float` | cosine similarity between encodings |
| `find_nearest` | `(hv, candidates?) → int` | nearest integer in codebook |

**Property:** `sim(v_n, v_m)` decreases monotonically as `|n-m|` increases.

#### `ExpressionEvaluator`

Safe recursive-descent arithmetic parser supporting `+ − × ÷ ** ()` and unary minus.

```python
ExpressionEvaluator().evaluate("(3 + 5) * 2")   # → 16.0
```

#### `LinearSolver`

Solves one-variable linear equations in natural notation.

```python
LinearSolver().solve("x + 3 = 7")   # → {'x': 4.0}
LinearSolver().solve("2x - 1 = 5")  # → {'x': 3.0}
```

#### `WordProblemParser`

Extracts arithmetic operations from natural-language word problems via cue-word classification (add/sub/mul/div).

#### `MathReasoner`

Top-level API combining all math sub-components.

```python
MathReasoner(fpe_max_int: int = 1023)
```

| Method | Signature | Returns |
|---|---|---|
| `solve_expression` | `(expr: str) → float` | exact arithmetic result |
| `solve_algebra` | `(equation: str) → Dict[str, float]` | variable → value |
| `solve_word_problem` | `(text: str) → dict` | `{answer, expression, explanation}` |
| `compare` | `(a, b) → dict` | `{less, equal, greater, difference, fpe_similarity}` |
| `magnitude_similarity` | `(a, b) → float` | FPE cosine similarity |
| `encode_number` | `(n) → HyperVector` | FPE HV for n |
| `is_math_query` | `(text) → bool` | heuristic detection |
| `extract_numbers` | `(text) → List[float]` | all numeric values in text |
| `verbalize_answer` | `(result) → str` | natural-language sentence |

---

### `perception/snn_perception.py` — 874 LOC

#### `SNNPerceptionModule`

```python
SNNPerceptionModule(input_dim: int = 100, n_neurons: int = 200, n_concepts: int = 50, use_rust: bool = True)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `perceive` | `(input_data: ndarray or dict) → HyperVector` | HV | Full perception pipeline → concept HV |
| `learn_from_episode` | `(input_data, label: str)` | None | STDP-supervised learning |
| `get_concept_hv` | `(concept_id: int) → HV` | HV | Get HV for a formed concept |
| `reset` | `()` | None | Reset all neuron states |

Internal components:

| Class | Description |
|---|---|
| `LIFNeuronLayer` | Leaky Integrate-and-Fire layer — `step(input_current) → spike_mask` |
| `SimpleConceptMapper` | Maps activation patterns to concept HVs via Jaccard similarity |

---

### `perception/vsa_snn_bridge.py`

| Class | Description |
|---|---|
| `RateCoder` | Encodes scalars/arrays as spike rates — `encode(data) → spikes` |
| `TemporalCoder` | Encodes values as spike timing — `encode(data) → spike_times` |
| `VSASNNBridge` | Bidirectional: HV → spikes and spikes → HV |

---

### `perception/grounding_verifier.py` — 585 LOC

`GroundingVerifier` — maps symbolic predicates to verification lambdas and HV prototypes.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `register_predicate` | `(name: str, verifier: callable, prototype_hv: HV = None)` | None | Register predicate |
| `verify` | `(state: dict, predicate: str) → bool` | bool | Check predicate truth in state |
| `extract_active` | `(state: dict, registered: List[str]) → Set[str]` | set | All true predicates |
| `get_grounding_hv` | `(predicate: str) → Optional[HV]` | HV or None | HV prototype for predicate |

---

### `perception/symbol_grounding.py`

`SymbolGrounding` — VSA-based symbol grounding that binds predicate names to prototype HVs, enabling graded truth values via cosine similarity.

---

### `perception/snn_shim.py`

Backend selector for SNN operations (mirrors `hypervec_shim.py`): tries `import snn_rs`, falls back to `SNNPerceptionModule`.

---

## Learning Subsystem

### `learning/hebbian.py` — 506 LOC

#### `HebbianMatrix`

```python
HebbianMatrix(n_concepts: int, learning_rate: float = 0.01)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `update` | `(pre: int, post: int)` | None | Hebbian update: `w[pre,post] += lr` |
| `update_oja` | `(pre: int, post: int, output: float)` | None | Oja's rule normalised update |
| `get_association` | `(concept: int) → ndarray` | ndarray | Activation weights from concept |
| `decay` | `(rate: float = 0.001)` | None | Apply weight decay |

#### `VSAHebbianLearner`

```python
VSAHebbianLearner(semantic_memory: SemanticMemory, learning_rate: float = 0.01)
```

Learns HV-level associations from co-occurrence of concepts in semantic memory.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `learn_from_episode` | `(episode: LiveEpisode)` | None | Extract concept pairs and update matrix |
| `get_similar_concepts` | `(concept: str, k: int = 5) → List[Tuple[str, float]]` | list | Top-k associated concepts |

---

### `learning/curiosity.py` — 336 LOC

#### `ExplorationDecision` (dataclass)

Fields: `should_explore: bool`, `novelty_score: float`, `learning_progress: float`, `reason: str`, `testable_hypotheses: List[Tuple[str, str]]`.

#### `CuriosityModule`

```python
CuriosityModule(novelty_threshold: float = 0.7, progress_window: int = 100, stagnation_threshold: float = 0.01)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `evaluate` | `(situation_hv: HV, task_tag: str = None) → ExplorationDecision` | ExplorationDecision | Should we explore? |
| `record_visit` | `(situation_hv: HV, task_tag: str)` | None | Update novelty memory after decision |
| `record_outcome` | `(success: bool, task_tag: str)` | None | Update learning progress |
| `get_hypotheses` | `() → List[Tuple[str, str]]` | list | Untested concept pairs |
| `add_hypothesis` | `(concept_a: str, concept_b: str)` | None | Add testable hypothesis |

---

### `learning/meta_learning.py`

`MetaLearner` — MAML-inspired rapid task adaptation: tracks which predicates and actions generalise across tasks.

---

### `learning/continual_learning.py`

`ContinualLearner` — manages catastrophic forgetting prevention via episodic replay buffers and elastic weight consolidation on the Q-table.

---

### `learning/cross_domain.py`

**Cross-domain knowledge transfer engine.** Lifts rules from a source domain and applies them in a target domain using VSA structural similarity — no neural network required.

#### `DomainConcept` (dataclass)

`domain: str`, `name: str`, `hv: HyperVector`

#### `DomainRelation` (dataclass)

`domain`, `source`, `relation`, `target`, `confidence: float`, `triple_hv: HyperVector`

The `triple_hv` is `src_hv XOR rel_hv XOR tgt_hv`.

#### `ConceptCorrespondence` (dataclass)

`source_domain`, `source_concept`, `target_domain`, `target_concept`, `structural_similarity: float`

#### `TransferredInference` (dataclass)

`source_domain`, `source_rule`, `target_domain`, `target_source`, `target_relation`, `target_target`, `confidence`, `provenance: str`

Method: `summary() → str`

#### `SchemaExtractor`

Groups domain relations by relation type and bundles their `triple_hv`s into a schema HV for each type.

| Method | Signature | Returns |
|---|---|---|
| `extract` | `(relations, concepts) → Dict[str, HV]` | schema HVs keyed by relation type |

#### `StructureMapper`

Finds concept correspondences between source and target domains.

| Method | Signature | Returns |
|---|---|---|
| `map` | `(src_concepts, tgt_concepts, explicit?, threshold) → List[ConceptCorrespondence]` | correspondences |

When `explicit` is provided, it is returned directly (bypasses auto-discovery).

#### `RuleLifter`

Substitutes fillers in source-domain rules using a concept mapping.

| Method | Signature | Returns |
|---|---|---|
| `lift` | `(relations, correspondences, target_domain, confidence_threshold) → List[TransferredInference]` | transferred inferences |

Confidence formula: `source_rule.confidence × min(src_corr.similarity, tgt_corr.similarity)`.

#### `TransferEngine`

Top-level orchestrator.

```python
engine = TransferEngine(confidence_threshold=0.55)
```

| Method | Signature | Returns |
|---|---|---|
| `register_concept` | `(domain, name, seed?) → DomainConcept` | registered concept |
| `register_rule` | `(domain, src, relation, tgt, confidence) → DomainRelation` | registered relation |
| `register_correspondence` | `(src_domain, src_concept, tgt_domain, tgt_concept, sim) → ConceptCorrespondence` | declared mapping |
| `transfer` | `(source_domain, target_domain) → List[TransferredInference]` | sorted by confidence ↓ |
| `discover_correspondences` | `(source_domain, target_domain, threshold) → List[ConceptCorrespondence]` | auto-discovered |
| `apply` | `(inferences, rule_learner) → int` | count of injected rules |
| `domain_summary` | `(domain) → dict` | `{domain, num_concepts, num_relations, …}` |
| `list_domains` | `() → List[str]` | sorted domain names |

---

### `cognitive/emotion_system.py` — 420 LOC

#### `EmotionSystem`

```python
EmotionSystem()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `update` | `(drives: dict) → str` | str | Update state from homeostatic drives, return dominant emotion name |
| `get_emotion_info` | `() → dict` | dict | `{name, valence, arousal, blend, history}` |
| `get_affect_bias` | `(action: str) → float` | float | Affective bias for GWT coalition scoring |
| `get_history` | `() → List[dict]` | list | Last `HISTORY_LIMIT` emotion states |

**Circumplex prototypes** (valence, arousal): joy (0.8, 0.7), trust (0.5, 0.2), fear (-0.7, 0.8), surprise (0.0, 0.9), sadness (-0.6, 0.2), disgust (-0.5, 0.4), anger (-0.5, 0.8), anticipation (0.3, 0.5).

---

### `cognitive/metacognition.py` — 504 LOC

#### `SafetyGate`

```python
SafetyGate(constraints: List[str] = None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `check` | `(coalition: Coalition, state: dict, task_tag: str) → Coalition` | Coalition | Veto or pass coalition |
| `add_constraint` | `(constraint: str)` | None | Add safety constraint predicate |

#### `MetacognitiveEngine`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `evaluate` | `(state: CognitiveState)` | None | Update performance metrics |
| `should_sleep` | `() → bool` | bool | True if performance degraded below threshold |
| `should_switch_task` | `() → bool` | bool | True if stuck on current task |
| `get_report` | `() → str` | str | Human-readable metacognitive summary |

---

### `cognitive/self_model.py`

#### `SelfModel`

```python
SelfModel()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `update_confidence` | `(task_tag: str, actual_success: bool)` | None | Update running success ratio |
| `get_confidence` | `(task_tag: str, state: dict = None) → float` | float | Context-aware confidence estimate |
| `get_calibration_error` | `(task_tag: str) → float` | float | `|confidence - actual_success_rate|` |
| `get_identity_hv` | `() → HV` | HV | Stable HV representation of "self" |

---

### `cognitive/theory_of_mind.py` — 312 LOC

#### `TheoryOfMind`

```python
TheoryOfMind()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `observe` | `(agent_id: str, action: str, state: dict)` | None | Record observed agent behaviour |
| `predict_action` | `(agent_id: str, current_state: dict) → str` | str | Predict next action |
| `get_beliefs` | `(agent_id: str) → dict` | dict | `{inferred_goal, confidence, model}` |
| `update_belief` | `(agent_id: str, key: str, value: Any)` | None | Manually update belief |

---

## Language Layer

### `language/text_knowledge_learner.py` — 1,074 LOC

#### `TextKnowledgeLearner`

```python
TextKnowledgeLearner(semantic_memory: SemanticMemory = None, causal_graph: CausalGraph = None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `learn` | `(text: str) → dict` | dict | Full learning pipeline — returns `{concepts, relations, causal_links}` |
| `query` | `(query: str, k: int = 10) → List[Tuple[str, float]]` | list | Find relevant stored facts |
| `_extract_concepts` | `(text: str) → List[str]` | list | Tokenise and filter stop words |
| `_extract_relations` | `(text: str) → List[Tuple[str, str, str]]` | list | SVO triple extraction |
| `_detect_causal` | `(text: str) → List[CausalLink]` | list | Keyword-based causal link detection |
| `get_learned_facts` | `() → List[str]` | list | All learned sentences |

**Causal keywords detected:** causes, leads to, results in, produces, triggers, prevents, stops, reduces, enables, requires.

---

### `language/lingua_cortex.py`

#### `LinguaCortex`

Semantic folding encoder:

```python
LinguaCortex()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `encode_word` | `(word: str) → HV` | HV | Deterministic HV from word hash |
| `encode_text` | `(text: str) → HV` | HV | Sentence HV via positional bundling |
| `encode_context` | `(text: str, context: str) → HV` | HV | Context-aware encoding |

---

### `language/language_module.py` — 526 LOC

#### `LanguageModule`

```python
LanguageModule(semantic_memory: SemanticMemory, episodic_memory: EpisodicMemory)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `process_input` | `(text: str) → dict` | dict | Parse + encode + extract predicates |
| `generate_response` | `(query_hv: HV, context: dict) → str` | str | Retrieve + compose natural language response |
| `update_vocabulary` | `(words: List[str])` | None | Register new words in vocabulary |

---

### `language/dialogue_manager.py` — 506 LOC

`DialogueManager` — discourse state machine tracking dialogue acts (question, assertion, confirmation, clarification), speaker intent, and grounding status.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `process_turn` | `(utterance: str) → str` | str | Update state, return response |
| `get_state` | `() → dict` | dict | Current dialogue state |
| `reset` | `()` | None | Reset conversation |

---

### `language/universal_input.py` — 947 LOC

#### `UniversalInput`

Converts any input type to a HV for unified processing.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `encode` | `(data: Union[str, dict, ndarray, float]) → HV` | HV | Auto-dispatch by type |
| `encode_text` | `(text: str) → HV` | HV | LinguaCortex encoding |
| `encode_state` | `(state: dict) → HV` | HV | Bind key–value pairs |
| `encode_image` | `(image: ndarray) → HV` | HV | MultimodalProcessor encoding |
| `encode_numeric` | `(value: float) → HV` | HV | Scalar quantisation |

---

### `language/semantic_roles.py`

**VSA Semantic Role Labeling.** Extracts thematic roles from natural-language sentences using lexico-syntactic heuristics and a VSA resonator — no neural network.

#### `SRLFrame` (dataclass)

| Field | Type | Default |
|---|---|---|
| `pred` | `str` | `""` |
| `agent` | `str` | `""` |
| `patient` | `str` | `""` |
| `theme` | `str` | `""` |
| `recipient` | `str` | `""` |
| `instrument` | `str` | `""` |
| `location` | `str` | `""` |
| `temporal` | `str` | `""` |
| `manner` | `str` | `""` |
| `cause` | `str` | `""` |
| `purpose` | `str` | `""` |
| `negation` | `bool` | `False` |
| `confidence` | `float` | `0.0` |

Method: `to_dict() → dict` (omits empty optional fields).

#### `SemanticRoleLabeler`

```python
SemanticRoleLabeler()
```

| Method | Signature | Returns |
|---|---|---|
| `label` | `(sentence: str) → SRLFrame` | role frame for one sentence |
| `label_batch` | `(sentences: List[str]) → List[SRLFrame]` | frames for multiple sentences |
| `update_codebook` | `(words: List[str]) → None` | add words to resonator codebook |

**Role vector seeds:** deterministic seeds 60001–60012 (one per role); word vectors use MD5-hash seeds — reproducible across sessions.

**Processing pipeline:**
1. Tokenise + lowercase.
2. Find predicate: irregular past tenses → morphological pattern → copular fallback.
3. Extract AGENT (pre-verb NP), PATIENT (post-verb NP, skipping PPs).
4. Parse prepositional phrases → LOCATION / TEMPORAL / RECIPIENT / INSTRUMENT / etc.
5. Build VSA composite: `S = ⊕ bind(role_hv, filler_hv)`.
6. Resonator verification: `estimate = S ⊕ role_hv`, match against codebook.

---

### `language/nlg.py` — enhanced

#### `StructuralRealizer` (existing)

Single-sentence grammar engine.

| Method | Signature | Returns |
|---|---|---|
| `pluralize` | `(noun: str) → str` | morphological plural |
| `conjugate` | `(verb, person, number, tense) → str` | conjugated form |
| `add_article` | `(noun) → str` | "a/an/the" + noun |
| `realize_sentence` | `(subj, relation, obj, tense?) → str` | complete sentence |
| `realize_chain` | `(chain: List) → str` | causal chain narrative |

#### `DiscoursePlanner` (new)

Multi-sentence paragraph generator from a list of semantic frames.

```python
DiscoursePlanner().plan(frames, query_type="factual", topic="")
```

**Query types:** `"factual"` | `"explanatory"` | `"procedural"` | `"causal"` | `"comparative"`

**Frame schema:** `{"subject": str, "relation": str, "object": str, "tense"?: str, "negate"?: bool, "importance"?: float}`

Features:
- Discourse connectives by relation type (`causes` → "As a result,"; `contradicts` → "However,")
- Pronoun anaphora for repeated subjects
- Numbered step formatting for procedural queries
- Importance-weighted frame ordering

#### `NLGEngine` (new)

Thin orchestrator exposing both old and new APIs.

| Method | Signature | Returns |
|---|---|---|
| `generate` | `(category, data) → str` | single-frame sentence (backward-compatible) |
| `generate_discourse` | `(frames, query_type?, topic?) → str` | multi-sentence paragraph |
| `generate_causal_chain` | `(chain) → str` | causal chain narrative |

---

### `integration/persistence.py` — 852 LOC

#### `BrainStore`

```python
BrainStore(db_path: str = "nsck_brain.db")
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `save_concept` | `(name: str, hv: HV)` | None | Persist concept HV to SQLite |
| `load_concepts` | `() → List[Concept]` | list | Load all concepts |
| `save_episode` | `(episode: Episode)` | None | Persist compressed episode |
| `load_episodes` | `(task_tag: str = None, limit: int = 1000) → List[Episode]` | list | Load episodes |
| `save_rule` | `(rule: Rule)` | None | Persist learned rule |
| `load_rules` | `(task_tag: str = None) → List[Rule]` | list | Load rules |
| `save_q_values` | `(q_values: dict)` | None | Persist Q-table |
| `load_q_values` | `() → dict` | dict | Load Q-table |
| `close` | `()` | None | Close SQLite connection |

#### `Episode` (dataclass)

Fields: `id`, `timestamp`, `task_tag`, `state_sketch` (dict), `action`, `outcome`, `reward`, `hv_bits`.

#### `Rule` (dataclass)

Fields: `id`, `task_tag`, `conditions` (FrozenSet[str]), `action`, `confidence`, `support`, `successes`, `tenure`.

---

### `integration/brain_fusion.py` — 481 LOC

#### `TaskBrain`

```python
TaskBrain(task_tag: str)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `add_concept` | `(name: str, hv: HV)` | None | Add concept |
| `add_rule` | `(conditions: FrozenSet, action: str, priority: int)` | None | Add rule |
| `get_rules` | `() → List[Rule]` | list | All rules |

#### `BrainFusion`

```python
BrainFusion()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `register_brain` | `(brain: TaskBrain)` | None | Add task brain |
| `fuse` | `() → FusedBrain` | FusedBrain | Merge all task brains |
| `transfer` | `(src_tag: str, tgt_tag: str, concept: str) → Optional[HV]` | HV or None | Transfer knowledge across tasks |

---

### `integration/explanation.py` — 433 LOC

#### `ExplanationGenerator`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `explain_action` | `(action, state, task_tag, trace) → Explanation` | Explanation | Build human-readable explanation |
| `explain_rule` | `(rule: Rule) → str` | str | Rule explanation in natural language |
| `explain_causal` | `(chain: CausalChain) → str` | str | Causal chain explanation |

#### `Explanation` (dataclass)

Fields: `winning_module`, `action`, `reason`, `evidence` (List[str]), `confidence`.

---

### `integration/config.py`

#### `NSCKConfig`

Configuration dataclass with defaults for all tunable parameters:

| Parameter | Default | Description |
|---|---|---|
| `hv_dimension` | 10240 | HyperVector bit width |
| `episodic_hot_capacity` | 500 | Hot-tier episode limit |
| `lsh_num_tables` | 8 | LSH table count |
| `gwt_danger_threshold` | 0.8 | Mental rehearsal veto threshold |
| `curiosity_novelty_threshold` | 0.7 | Novelty → exploration threshold |
| `rule_min_support` | 5 | Minimum observations to promote rule |
| `rule_min_confidence` | 0.3 | Minimum confidence to promote rule |
| `q_learning_rate` | 0.1 | TD(0) learning rate |
| `q_discount` | 0.95 | Reward discount factor |

---

### `integration/knowledge_integration.py`

`KnowledgeIntegration` — merges multiple knowledge sources (SemanticMemory, CausalGraph, RuleLearner) into a unified queryable knowledge base with conflict resolution via timestamp ordering.

---

## Multimodal

### `multimodal/multimodal_processor.py` — 788 LOC

#### `MultimodalInput` (dataclass)

Fields: `text: Optional[str]`, `image: Optional[ndarray]`, `audio: Optional[ndarray]`, `structured_data: Optional[dict]`.

#### `MultimodalProcessor`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `process` | `(input: MultimodalInput) → ProcessedInput` | ProcessedInput | Extract all features → fused HV |
| `process_image` | `(image: ndarray) → ModalityResult` | ModalityResult | HOG + color + LBP + edges + spatial |
| `process_text` | `(text: str) → ModalityResult` | ModalityResult | LinguaCortex encoding |
| `fuse` | `(results: List[ModalityResult]) → HV` | HV | XOR-bind and bundle all modality HVs |

#### `ProcessedInput`

Fields: `fused_hv`, `modality_results`, `extracted_concepts`, `confidence`.

---

### `multimodal/image_generator.py` — 575 LOC

`ImageGenerator` — generates pixel images from concept HVs using learned feature templates.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `train_from_examples` | `(examples: List[Tuple[str, ndarray]], max_examples: int)` | None | Learn concept → visual mapping |
| `generate` | `(concept: str, config: GenerationConfig) → ndarray` | ndarray | Generate image for concept |
| `describe` | `(image: ndarray) → List[str]` | list | Reverse: image → concept labels |

---

## Training

### `training/snn_training.py`

`SNNTrainer` — supervised SNN training pipeline with configurable epochs, batch size, and STDP learning rates.

### `training/vsa_trainer.py`

`VSATrainer` — trains HV encodings by adjusting similarity thresholds to separate positive/negative concept pairs.

### `training/snn_benchmarks.py`

Benchmark suite measuring SNN inference throughput (QPS), STDP convergence rate, and Hebbian learning stability.

---

## Rust Accelerators

### `rust_vsa/src/lib.rs`

**`HyperVector`** (PyO3 class, `hypervec_rs` module)

| Method | Signature | Returns |
|---|---|---|
| `new` | `(seed: Optional[u64])` | `HyperVector` |
| `xor` | `(&self, other: &HV) → HV` | `HyperVector` |
| `bundle` | `(&self, other: &HV) → HV` | `HyperVector` |
| `permute` | `(&self, shift: i64) → HV` | `HyperVector` |
| `permute_inverse` | `(&self, shift: i64) → HV` | `HyperVector` |
| `similarity` | `(&self, other: &HV) → f64` | `float` |
| `cosine_similarity` | `(&self, other: &HV) → f64` | `float` |
| `from_bits` | `(bits: &[i8])` | `HyperVector` |
| `zero` | `()` | `HyperVector` |

Storage: `Vec<u64>` (160 blocks × 64 bits = 10,240 bits). RNG: `ChaCha8Rng`.

**Module-level functions:**

| Function | Description |
|---|---|
| `parallel_similarity_search(query, corpus, k)` | Rayon parallel top-k search |
| `parallel_bundle(vectors)` | Rayon parallel superposition |

### `rust_vsa/src/concurrent.rs`

`HyperVectorRegistry` — `DashMap<String, HyperVector>` for lock-free concurrent concept storage.

### `rust_vsa/src/semantic.rs`

`SemanticMemoryConcurrent` — parallel spreading activation using Rayon work-stealing.

### `rust_vsa/src/episodic.rs`

`EpisodicMemoryConcurrent` — `RwLock<Vec<Episode>>` hot tier + parallel k-NN retrieval.

### `rust_vsa/src/worker_pool.rs`

`CognitiveWorkerPool` — Rayon `ThreadPool` with configurable parallelism.

### `rust_vsa/src/persistence.rs`

`PersistentStorage` — async SQLite persistence using `rusqlite`.

### `rust_vsa/src/async_runtime.rs`

`AsyncCognitiveRuntime` — Tokio-based async executor for non-blocking I/O.

### `rust_snn/src/lib.rs`

`LIFLayer` — Rayon-parallelised LIF neuron step.
`SnnCore` — full SNN simulation loop.
`StdpEngine` — STDP weight update (parallel over pre-post pairs).

### `rust_snn/src/hebbian.rs`

`HebbianMatrix` — Oja's rule outer-product, Rayon-parallelised rows.

### `rust_snn/src/concept.rs`

`ConceptMapper` — Jaccard similarity per concept, Rayon-parallelised.

---

## nsck_ai_model Modules

### `ai_engine.py` — 2,790 LOC

Main entry point. See [nsck_ai_model/README.md](../../nsck_ai_model/README.md) for full reference.

**Key constants:**

| Constant | Value | Description |
|---|---|---|
| `DIMENSION` | 10240 | HyperVector width |
| `MAX_CONTEXT` | 20 | Conversation history turns |
| `SIMILARITY_THRESHOLD` | 0.35 | Minimum similarity for retrieval |
| `HASH_SEED_MODULO` | 2³² | Seed range for deterministic HVs |
| `MAX_POSITION_SHIFT` | 64 | Maximum permutation for positional encoding |
| `_FUNCTION_WORDS` | frozenset | Grammatical stop words (the only built-in list) |

**`ThoughtTrace`** — container for 11 `TraceStep` objects.

**`TraceStep`** (dataclass): `stage: str`, `summary: str`, `data: dict`, `elapsed_ms: float`.

**`ResponseGenerator`** — internal class implementing the IDF-weighted candidate scoring and Jaccard deduplication.

---

### `context_retention.py` — 412 LOC

**`ConversationTurn`** (dataclass): `turn_id`, `user_input`, `response`, `concepts`, `entities`, `timestamp`, `hypervector`.

**`ContextRetentionModule`** — see [nsck_ai_model/README.md](../../nsck_ai_model/README.md).

---

### `counterfactual_reasoner.py` — 525 LOC

**`HypotheticalState`** (dataclass): `state_id`, `description`, `modified_facts`, `original_facts`, `consequences`, `confidence`.

**`CounterfactualScenario`** (dataclass): `query`, `original_state`, `hypothetical_state`, `comparison`.

**`CounterfactualReasoner`** — 12+ pattern detection for "what if" queries including: "what if", "suppose", "assume", "had … not", "would have", "could have", "if … then", "hypothetically", "imagine", "in the event", "unless", "provided that".

---

### `math_handler.py` — 1,126 LOC

**`MathResult`** (dataclass): `question`, `answer`, `numeric_value`, `steps`, `confidence`, `method`.

**`MathHandler`** — symbolic math evaluation. No neural networks.

**`LearnableOperatorDetector`** — detects custom operator keywords learned from training data.

**`is_math_question(text: str) → bool`** — standalone detector.

---

### `response_composer.py` — 505 LOC

**`ResponseComposer`**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `compose` | `(candidates: List[str], query: str, max_sentences: int = 2) → str` | str | IDF-score → Jaccard-dedup → top-N sentences |
| `score_candidate` | `(candidate: str, query: str) → float` | float | IDF-weighted relevance score |
| `deduplicate` | `(sentences: List[str], threshold: float = 0.5) → List[str]` | list | Jaccard similarity deduplication |

---

### `dashboard.py` — 687 LOC

Flask app with 15 endpoints. See [nsck_ai_model/README.md](../../nsck_ai_model/README.md) for full API reference.

---

### `data_pipeline.py` — 245 LOC

**`DataPipeline`** — HuggingFace WikiText streaming + built-in seed corpus.

---

### `data_fetcher.py` — 808 LOC

**Sample dataclasses:** `TextSample`, `QASample`, `MathSample`, `ImageTextSample`, `ConversationSample`.

**`DataFetcher`** — multi-modal sample fetcher with configurable sources.

**`stream_rich_corpus(max_samples: int) → Iterator[TextSample]`** — yields samples from all configured sources.

---

### `evaluator.py` — 504 LOC

**`EvaluationReport`** (dataclass): per-domain scores, overall F1, overall accuracy, latency stats.

**`NSCKEvaluator`** — 8 evaluation domains: geography, science, animals, technology, math, reasoning, conversation, image understanding.

---

### `autonomous_trainer.py` — 831 LOC

**`TrainingReport`** (dataclass): phases completed, samples processed, evaluation scores per phase, total time.

**`AutonomousTrainer`** — multi-phase orchestrator with per-phase evaluation and checkpoint saving.

---

### `telemetry_monitor.py` — 591 LOC

**`TelemetryMonitor`** — tracks QPS, mean/p95 latency, token throughput, anomaly rate (latency > 3σ).

---

## Tests Reference

### NSCK Unit Tests (`nsck/tests/unit/`)

| File | Class(es) | What it tests |
|---|---|---|
| `vsa/test_hypervec_parity.py` | `TestHyperVecParity` | Rust ↔ Python HV operation parity |
| `memory/test_cleanup_memory.py` | `TestCleanupMemory` | CleanupMemory registration, denoising, LRU eviction |
| `reasoning/test_causal_discovery.py` | `TestCausalDiscovery` | ΔP discovery, spurious correlation rejection |
| `reasoning/test_chaining.py` | `TestCausalChaining` | Forward/backward chain traversal |
| `reasoning/test_counterfactuals.py` | `TestCounterfactuals` | Counterfactual simulation |
| `reasoning/test_planning.py` | `TestPlanning` | A* plan generation, operator learning |
| `reasoning/test_plan_execution.py` | `TestPlanExecution` | Plan step execution and state transitions |
| `reasoning/test_rule_learner_interface.py` | `TestRuleLearnerInterface` | WorkspaceModule interface compliance |
| `reasoning/test_logic_bridge.py` | `TestLogicBridge` | VSA ↔ symbolic logic bridging |
| `reasoning/test_theory_formation.py` | `TestTheoryFormation` | Rule induction from observations |
| `reasoning/test_math_reasoning.py` | `TestMathReasoner`, `TestFPECodebook`, `TestExpressionEvaluator`, `TestLinearSolver`, `TestWordProblemParser` | FPE encoding, expression evaluation, algebra, word problems |
| `cognitive/test_context_engine.py` | `TestContextEngine` | Context window management |
| `cognitive/test_metacognition.py` | `TestMetacognition` | MetacognitiveEngine performance tracking |
| `cognitive/test_metacognitive_veto.py` | `TestMetacognitiveVeto` | SafetyGate veto logic |
| `cognitive/test_self_model.py` | `TestSelfModel` | Confidence calibration |
| `cognitive/test_integrated_metacognition.py` | `TestIntegratedMetacognition` | Full metacognition integration |
| `language/test_lingua.py` | `TestLingua` | LinguaCortex encoding properties |
| `language/test_semantic_roles.py` | `TestSemanticRoleLabeler`, `TestSRLFrame`, `TestVSAHVFunctions` | SRL pipeline: predicate finding, role extraction, resonator, negation, anaphora |
| `language/test_nlg_discourse.py` | `TestStructuralRealizerBackCompat`, `TestDiscoursePlanner`, `TestNLGEngine` | Discourse planning: connectives, anaphora, procedural, negation |
| `learning/test_text_knowledge_learner.py` | `TestTextKnowledgeLearner` | SVO extraction, causal detection |
| `learning/test_cross_domain.py` | `TestTransferEngine`, `TestSchemaExtractor`, `TestStructureMapper`, `TestRuleLifter` | Cross-domain transfer: registration, mapping, lifting, confidence ordering |
| `perception/test_semantic_folding.py` | `TestSemanticFolding` | Semantic folding properties |
| `perception/test_semantic_roles.py` | `TestSemanticRoles` | Role-filler binding and unbinding |
| `integration_core/test_brain_fusion.py` | `TestBrainFusion` | Multi-task knowledge transfer |
| `integration_core/test_global_workspace.py` | `TestGlobalWorkspace` | Coalition competition, veto |

### NSCK Integration Tests (`nsck/tests/integration/`)

| File | Focus |
|---|---|
| `test_system_capabilities.py` | End-to-end capability proofs (VSA, Memory, Rules, Causal, Fusion, Meta, Planning, Efficiency) |
| `test_cognitive_wiring.py` | All modules connected through CognitiveEngine |
| `test_cross_module.py` | Cross-module data consistency |
| `test_phase5.py` | Phase 5 integration (language + reasoning) |
| `test_phase8_mental_rehearsal.py` | Mental rehearsal veto with WorldModel |
| `test_phase8_permutation.py` | Permutation properties at integration level |
| `test_phase8_universal_input.py` | UniversalInput with all types |
| `test_stability.py` | Long-run numerical stability |
| `test_phase2_old.py` | Phase 2 regression tests |

### NSCK Architecture Tests (`nsck/tests/core_architecture/`)

| File | Focus |
|---|---|
| `test_reasoning.py` | Causal reasoning + GWT + rule induction |
| `test_learning.py` | Hebbian, curiosity, continual learning |
| `test_snn_integration.py` | SNN perception with cognitive engine |
| `run_comprehensive_tests.py` | All architecture tests in one run |

### NSCK Experiments (`nsck/tests/experiments/`)

| File | Focus |
|---|---|
| `belief_revision_test.py` | Belief revision via temporal ordering in SemanticMemory |
| `text_reasoning_test.py` | Text-grounded reasoning queries |
| `transitive_test.py` | Transitive inference through causal/semantic chains |
| `verify_f1.py` | F1 evaluation against a labelled test set |

### NSCK Regression (`nsck/tests/regression/`)

| File | Focus |
|---|---|
| `test_bug_fixes.py` | Regression tests for known-fixed bugs |

### AI Model Tests (`nsck_ai_model/tests/`)

| File | Tests | Focus |
|---|---|---|
| `test_ai_engine.py` | 77 | Engine internals, ThoughtTrace structure, encoding, edge cases |
| `test_production.py` | 46 | End-to-end: QA, reasoning, conversation, emotion, performance |
| `test_multimodal.py` | — | Image training and description pipeline |
