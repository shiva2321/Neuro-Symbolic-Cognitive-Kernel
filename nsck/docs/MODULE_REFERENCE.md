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
| `bundle` | `(other: HyperVectorPy) → HyperVectorPy` | `HyperVectorPy` | Superposition — majority vote; tie-breaking mask seeded deterministically from both inputs (same pair → same result) |
| `weighted_bundle` | `(other: HyperVectorPy, weight: float, seed=None) → HyperVectorPy` | `HyperVectorPy` | Weighted superposition — `weight∈[0,1]` biases toward `self`; `weight=0.5` equals plain bundle |
| `permute` | `(shift: int) → HyperVectorPy` | `HyperVectorPy` | Circular bit-shift by `shift` positions |
| `permute_inverse` | `(shift: int) → HyperVectorPy` | `HyperVectorPy` | Inverse circular shift |
| `negate` | `() → HyperVectorPy` | `HyperVectorPy` | VSA anti-bundling negation: `XOR(self, NEG_SEED_HV)` where `NEG_SEED=0xDEADBEEFCAFEBABE`. Result: `sim(hv, negate(hv))≈0.50` (near-orthogonal); `negate(negate(hv))==hv` (involution — double negation recovers original). Both Rust and Python implementations. |
| `similarity` | `(other: HyperVectorPy) → float` | `[0, 1]` | Normalised Hamming similarity (legacy) |
| `cosine_similarity` | `(other: HyperVectorPy) → float` | `[-1, 1]` | Bipolar cosine similarity (recommended) |
| `similarity_robust` | `(other, method='cosine') → float` | `[0, 1]` | Configurable similarity (normalises cosine to [0,1]) |
| `lsh_hash` | `(seed: int, n_bits: int) → int` | `int` | Locality-sensitive hash — stable fingerprint for approximate nearest-neighbour bucketing |

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
| `query` | `(query_hv: HV, k: int = 5) → List[Tuple[str, float]]` | list | Top-k similar concepts via NSW ANN (V6) or exact scan |
| `spread_activation` | `(start_concepts: List[str], steps: int = 3, decay: float = 0.7) → Dict[str, float]` | dict | Weighted spreading activation |
| `get_inherited_properties` | `(concept: str) → dict` | dict | BFS up `is_a` edges + merge properties |
| `extract_schema` | `(concept: str) → dict` | dict | Full schema: `{name, properties, is_a, parts, causes, caused_by}` |
| `infer_transitive` | `(relation_type: str = 'is_a', max_hops: int = 3) → int` | int | V4: BFS up relation edges to close transitive chains; returns count of new inferred edges |
| `build_prototypes` | `(category_relation: str = 'is_a', min_members: int = 2) → Dict[str, HV]` | dict | V4: Bundle all member HVs per category (Rosch 1973 prototype theory); returns `{category → prototype_hv}` |
| `reset` | `()` | None | Clear all concepts and relations |
| `save` | `(filepath: str)` | None | Pickle to disk |
| `load` | `(filepath: str)` | None | Load from pickle |

**V6 NSW ANN index:** `SemanticMemory` maintains a pure-Python `_NSWIndex` (Navigable Small World) for approximate nearest-neighbour queries. Falls back to exact scan when `hnswlib` is not installed. NSW index is rebuilt incrementally as concepts are added.

**Default relation weights:** `is_a=0.9, has_property=0.7, causes=0.6, leads_to=0.6, results_in=0.6, implies=0.55, part_of=0.5, similar_to=0.4, semantically_related=0.35`

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

**Integration note:** Pass the engine's shared `CausalGraph` via `causal_graph=` so causal facts learned from text flow directly into `CausalReasoner` without a manual sync step. When `causal_graph=None`, TKL creates its own isolated graph.

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

### `language/fluent_nlg.py` — V6 (added Feb 2026)

Fluent natural-language response generation from semantic frames.

#### `FluentResponseComposer`

```python
FluentResponseComposer()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `compose` | `(frames, topic, query_type, max_sentences) → str` | str | Multi-sentence fluent response from frame list |

**query_type options:** `"factual"` \| `"explanatory"` \| `"procedural"` \| `"causal"` \| `"comparative"`

Each query type uses a different opening frame and connective pattern:
- `factual`: "Understanding X requires examining…"
- `causal`: "To explain X:" + "As a result," connectives
- `explanatory`: "X can be understood in the following way."
- `procedural`: Numbered step list
- `comparative`: "By comparison," / "Unlike,"

#### `RelationVerbalizer`

Maps symbolic relations to English predicates:

| Relation | Surface form |
|---|---|
| `is_a` / `categorization` | "is a kind of" / "is" |
| `causes` / `leads_to` | "leads to" / "results in" |
| `has_property` | "is known for its" / "is characterized by" |
| `inhibits` / `prevents` | "prevents" / "reduces" |
| `part_of` | "is part of" / "belongs to" |
| `requires` | "requires" / "depends on" |
| `contains` | "contains" / "includes" |
| `comparison` | "relates to" |
| `co_occurs` | "often appears alongside" |

#### `NSCKResponseEngine`

High-level response engine — the preferred public API.

```python
NSCKResponseEngine()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `respond` | `(frames, topic, query_type, max_sentences) → str` | str | Fluent paragraph from frame list |
| `answer_query` | `(query: str, facts: List[Tuple], topic: str) → str` | str | Answer NL question from (s,r,o) triples; auto-detects query type |
| `describe` | `(concept: str, semantic_memory, max_relations: int) → str` | str | Look up concept relations and describe in fluent prose |

**V7:** `DialogueManager` routes all response methods through `NSCKResponseEngine` instead of template strings.

---

### `language/pos_tagger.py` — V6 (added Feb 2026)

Brill transformation-based POS tagger.

#### `BrillPosTagger`

```python
BrillPosTagger()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `tag_sentence` | `(sentence: str) → List[Tuple[str, str]]` | list | Tag each word with POS |
| `tag_word` | `(word: str, context: List[str]) → str` | str | Tag single word using lexicon + rules |

**POS tags used:**

| Tag | Meaning | Examples |
|---|---|---|
| `NN` | Noun | brain, memory, cell |
| `NNP` | Proper noun | Einstein, Earth |
| `VB` / `VBD` / `VBG` | Verb forms | run, ran, running |
| `JJ` | Adjective | large, important |
| `RB` | Adverb | quickly, very |
| `NEG` | Negator | not, never, no |
| `TEMP` | Temporal connective | before, after, while |
| `COND` | Conditional | if, unless, provided |
| `SIM` | Similarity | like, as, similarly |

**Lexicon:** 300+ common English words with hand-assigned tags.

**Suffix rules (8):**
1. `-ing` → VBG (unless in `_ING_NOUNS`)
2. `-ed` → VBD (unless in `_ED_ADJECTIVES`)
3. `-tion` / `-ness` / `-ity` / `-ment` / `-ism` → NN
4. `-ful` / `-less` / `-ous` / `-ical` / `-ial` / `-ual` → JJ
5. `-ly` → RB
6. `-ize` / `-ise` → VB (unless in `_ING_NOUNS`)
7. Starts with capital + not sentence-initial → NNP
8. Single digit or pure number → CD

---

### `language/pragmatics.py` — V5 (added Feb 2026)

Gricean cooperative pragmatics for dialogue.

#### `PragmaticEngine`

```python
PragmaticEngine()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `classify_speech_act` | `(utterance: str) → str` | str | One of 7 speech-act labels |
| `generate_implicature` | `(utterance: str) → List[str]` | list | List of pragmatic implicatures |
| `check_maxims` | `(utterance: str, context: dict) → dict` | dict | Gricean maxim violations |
| `project_presuppositions` | `(utterance: str) → List[str]` | list | Factive presuppositions |

**Speech acts (7):** assertion, question, directive, commissive, expressive, declaration, threat

**Horn scales (15):** `(all, most, many, some)`, `(always, usually, sometimes, rarely)`, `(certain, probable, possible)`, `(and, or)`, `(know, believe, think)`, …

**Gricean maxims:**
- **Quantity:** Be as informative as required, not more
- **Quality:** Do not assert what you believe to be false
- **Relation:** Be relevant
- **Manner:** Avoid obscurity, ambiguity, verbosity

**Presupposition triggers:** factive verbs (`know`, `realize`, `regret`), change-of-state verbs, definite descriptions

---

### `language/distributional_semantics.py` — V3/V7

Co-occurrence based semantic similarity.

#### `DistributionalCodebook`

```python
DistributionalCodebook(window_size: int = 5)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `train` | `(sentences: List[str])` | None | Build co-occurrence counts from corpus |
| `get_context_hv` | `(word: str) → Optional[HV]` | HV or None | Context vector for word |
| `similarity` | `(w1: str, w2: str) → float` | float | Cosine of context HVs |
| `save` | `(path: str)` | None | Serialize to disk |
| `load` | `(path: str)` | None | Load from disk |

**V7 change:** `DistributionalCodebook` is pre-trained on `BUILTIN_CORPUS` (200 carefully selected sentences) at NSCK initialization when `enable_distributional_semantics=True`. This gives above-random similarity for well-known word pairs.

**Context HV construction:**
```
context_hv(w) = bundle(permute(word_hv(w'), k) for (w', offset=k) in window(w, ±window_size))
```

---

### `language/hf_corpus_loader.py` — V7 (added Feb 2026)

HuggingFace dataset streaming for large-scale corpus ingestion.

#### `HFCorpusLoader`

```python
HFCorpusLoader(dataset_name: str = "HuggingFaceFW/fineweb", max_sentences: int = 100_000)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `stream_sentences` | `(max: int) → Iterator[str]` | iterator | Yield clean sentences from HF dataset |
| `load_offline_fallback` | `() → List[str]` | list | Return BUILTIN_CORPUS when offline |
| `is_available` | `() → bool` | bool | Check if HF datasets library + internet available |

**Enabled by:** `NSCKConfig(enable_hf_corpus=True)`. Disabled by default — requires `datasets` library and internet.

**Offline fallback:** Returns `BUILTIN_CORPUS` (200 sentences hardcoded in the module) when `datasets` is not installed or network is unavailable.

---

### `language/construction_grammar.py` — V3/V4 (expanded)

Construction grammar pattern matching for English NLU.

#### `ConstructionMatcher`

```python
ConstructionMatcher()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `match` | `(tokens: List[str]) → List[ConstructionMatch]` | list | Match all applicable constructions |
| `get_all_constructions` | `() → List[Construction]` | list | List all 71 constructions |

**71 constructions** (V3+V4):

| Category | Examples |
|---|---|
| Core SVO | `"X V Y"` → agent-action-patient |
| Copular | `"X is Y"`, `"X is a Y"`, `"X is an Y"` |
| Possessive | `"X has Y"`, `"X contains Y"` |
| Causal | `"X causes Y"`, `"X leads to Y"`, `"X results in Y"` |
| Similarity | `"X is like Y"`, `"X resembles Y"` |
| Negation (V4) | `"X is not Y"`, `"X does not V"`, `"X cannot Y"` |
| Temporal (V4) | `"After X, Y"`, `"Before X, Y"`, `"While X, Y"`, `"When X, Y"` |
| Conditional (V4) | `"If X then Y"`, `"Unless X, Y"`, `"Provided X, Y"` |
| Instrumental | `"X V Y using Z"`, `"X V Y by Z"` |
| Comparative | `"X is more ADJ than Y"`, `"X is ADJ-er than Y"` |

**Key constants:**
- `_MIN_ISH_SUFFIX_LEN = 6` — minimum word length for `-ish` suffix rule
- `NEGATION_WORDS` — `frozenset` of negators
- `TEMPORAL_CONNECTIVES` — `frozenset` of temporal markers
- `CONDITIONAL_CONNECTIVES` — `frozenset` of conditionals
- `COMMON_VERBS` — 400+ verb forms for direct matching

---

### `reasoning/spatial_reasoning.py` — V5 (added Feb 2026)

VSA-based spatial relation encoding using Fixed-Point Encoding (FPE) bit-flip.

#### `PositionCodebook`

```python
PositionCodebook(dims: int = 3)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `encode_position` | `(coords: Tuple[float, ...]) → HV` | HV | Encode n-dimensional position as HV |
| `decode_position` | `(hv: HV) → Tuple[float, ...]` | tuple | Approximate position from HV |

**FPE bit-flip encoding:**
```
pos_hv(x) = base_hv
  XOR flip(base_hv, AXIS_FLIP_BITS × |x_normalized|)   # flip bits proportional to magnitude
```
- `_AXIS_FLIP_BITS = 50` — bits flipped per unit of movement along each axis
- `_NEGATIVE_STEP_OFFSET = 100_000` — large integer offset for negative coordinates (avoids 0-cross confusion)
- `_MIN_QUERY_SIMILARITY = 0.4` — minimum similarity threshold for relation queries

**Why FPE and not permute?** Python `permute()` gives ~0.50 similarity regardless of shift distance — it's not monotone. FPE bit-flip is monotone: adjacent positions are more similar than distant ones.

#### `SpatialReasoner`

```python
SpatialReasoner(config: NSCKConfig)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `assert_spatial` | `(obj1: str, obj2: str, relation: str) → None` | None | Assert VSA spatial relationship |
| `query_relation` | `(obj1: str, obj2: str) → str` | str | Query relationship between two objects |
| `get_all_assertions` | `() → List[Tuple]` | list | All asserted (obj1, rel, obj2) triples |

**8 relations:** `above`, `below`, `left`, `right`, `inside`, `outside`, `near`, `far`

---

### `reasoning/temporal_reasoning.py` — V4 (added Feb 2026)

Temporal reasoning over ordered events and intervals.

#### `TemporalReasoner`

```python
TemporalReasoner()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `add_event` | `(event: str, time: float)` | None | Add timestamped event |
| `before` | `(e1: str, e2: str) → bool` | bool | Is e1 before e2? |
| `after` | `(e1: str, e2: str) → bool` | bool | Is e1 after e2? |
| `during` | `(event: str, interval: Tuple) → bool` | bool | Is event within interval? |
| `get_timeline` | `() → List[Tuple[str, float]]` | list | All events sorted by time |

---

### `reasoning/abductive_reasoning.py` — V4 (added Feb 2026)

Abductive reasoning — inference to the best explanation.

#### `AbductiveReasoner`

```python
AbductiveReasoner(causal_graph: CausalGraph, semantic_memory: SemanticMemory)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `abduce` | `(observation: str, max_explanations: int = 5) → List[Explanation]` | list | Return best explanations for an observation |
| `score_explanation` | `(hypothesis: str, observation: str) → float` | float | Score: plausibility × parsimony |

**Selection criteria:** Plausibility (causal strength), parsimony (shortest chain), and coherence (consistent with known facts).

---

### `reasoning/predictive_processor.py` — V4 (added Feb 2026)

Predictive processing and active inference engine.

#### `PredictiveProcessor`

```python
PredictiveProcessor(config: NSCKConfig)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `predict` | `(context_hv: HV) → HV` | HV | Generate prediction from prior |
| `compute_error` | `(prediction: HV, observation: HV) → float` | float | Prediction error (1 - similarity) |
| `update_prior` | `(error: float, observation: HV)` | None | Bayesian prior update |

**Key constants:**
- `PRIOR_UNCERTAINTY = 0.5` — initial prior precision
- Update rule: Bayesian: `posterior ∝ likelihood × prior`

---

### `learning/schema_induction.py` — V4 (added Feb 2026)

Schema induction from repeated conceptual patterns.

#### `SchemaInducer`

```python
SchemaInducer(semantic_memory: SemanticMemory)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `induce` | `(episodes: List[dict], min_support: int) → List[Schema]` | list | Extract recurring patterns |
| `apply_schema` | `(schema: Schema, context: dict) → dict` | dict | Fill schema slots from context |

---

### `learning/pmi_learner.py` — V4 (added Feb 2026)

Pointwise Mutual Information learner for co-occurrence patterns.

#### `PMILearner`

```python
PMILearner()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `observe` | `(items: List[str])` | None | Record co-occurrence |
| `pmi` | `(a: str, b: str) → float` | float | `log P(a,b) / (P(a)×P(b))` |
| `top_associates` | `(word: str, k: int = 10) → List[Tuple]` | list | Top-k highest PMI partners |

**Formula:** `PMI(a,b) = log₂(P(a,b) / (P(a) × P(b)))`

Positive PMI = a and b co-occur more than chance. Clamped to `[0, ∞)` as PPMI.

---

### `learning/predictive_coding.py` — V4 (added Feb 2026)

Predictive coding module for error-minimisation learning.

#### `PredictiveCodingModule`

```python
PredictiveCodingModule(prior_uncertainty: float = PRIOR_UNCERTAINTY)
```

| Constant | Value | Description |
|---|---|---|
| `PRIOR_UNCERTAINTY` | 0.5 | Initial prior precision weight |

---

### `learning/active_inference.py` — V4 (added Feb 2026)

Active inference learner — selects actions to minimise expected free energy.

#### `ActiveInferenceLearner`

```python
ActiveInferenceLearner(config: NSCKConfig)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `select_action` | `(observations: List[HV], actions: List[str]) → str` | str | Select action minimising expected FE |
| `update_beliefs` | `(observation: HV)` | None | Update posterior beliefs |

**Key constants:**
- `MIN_TEMP = 0.1` — minimum softmax temperature
- `MAX_TEMP = 5.0` — maximum softmax temperature (exploration)

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

Configuration dataclass with defaults for all tunable parameters. 25 feature flags control optional subsystems.

**Core numeric parameters:**

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

**Feature flags (25 total):**

| Flag | Default | Version | Description |
|---|---|---|---|
| `enable_snn` | True | V1 | Spiking neural network perception |
| `enable_vsa` | True | V1 | VSA hypervector operations |
| `enable_sleep` | True | V1 | Offline consolidation |
| `enable_construction_grammar` | False | V3 | 71 CG constructions |
| `enable_frame_semantics` | False | V3 | VerbNet-style verb frames |
| `enable_coreference` | False | V3 | Pronoun resolution |
| `enable_contextual_encoding` | False | V3 | Context-aware LinguaCortex |
| `enable_free_energy_beliefs` | False | V3 | Belief revision via FE |
| `enable_distributional_semantics` | False | V3 | Co-occurrence context HVs |
| `enable_incremental_concept_refinement` | False | V3 | Incremental concept update |
| `enable_dual_process` | False | V3 | System 1 / System 2 routing |
| `enable_conceptual_blending` | False | V3 | Analogy blend HVs |
| `enable_hnsw_index` | False | V3 | HNSW/NSW ANN for fast query |
| `enable_homeostasis` | False | V3 | Memory homeostasis regulation |
| `enable_stigmergy` | False | V3 | Pheromone-weighted spreading |
| `enable_auto_categories` | False | V3 | Automatic concept categorisation |
| `enable_negation_handling` | False | V4 | Negation as first-class operator |
| `enable_temporal_reasoning` | False | V4 | Temporal connectives + ordering |
| `enable_conditional_logic` | False | V4 | If/unless/provided logic |
| `enable_transitive_inference` | False | V4 | Transitive is_a/causes closure |
| `enable_prototype_generalization` | False | V4 | Category prototype bundling |
| `enable_spatial_reasoning` | False | V5 | FPE spatial positions + 8 relations |
| `enable_pragmatics` | False | V5 | Gricean maxims + speech acts |
| `enable_fluent_dialogue` | **True** | V7 | FluentNLG wired into DialogueManager |
| `enable_hf_corpus` | False | V7 | HuggingFace corpus pre-training |

**Factory methods:**

| Method | What it enables |
|---|---|
| `NSCKConfig.minimal()` | Core only — SNN + VSA + sleep |
| `NSCKConfig.research()` | All 25 flags enabled |
| `NSCKConfig.from_env()` | Read flags from environment variables |

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

---

## V8 New Classes and Methods

*Added in V8 (February 2026). Full test suite: **1,111 passed, 5 skipped, 4 xfailed**.*

---

### `multimodal/multimodal_processor.py` — `ConcurrentMultimodalScheduler`

```python
ConcurrentMultimodalScheduler(
    max_workers: int = 4,
    coherence_window_ms: float = 50.0,
    context_engine=None,
    semantic_memory=None,
)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `process_concurrent` | `(inp: MultimodalInput) → ProcessedInput` | `ProcessedInput` | Process all modalities in parallel; fuse within coherence window using attention-weighted HV: `Σ(conf_m · bind(role_m, HV_m)) / Σconf_m` |
| `close` | `() → None` | `None` | Shut down the ThreadPoolExecutor |

---

### `memory/semantic_memory.py` — Memory Lifecycle Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `decay_concepts` | `(lambda_decay: float = 0.01) → int` | `int` | Exponential Ebbinghaus decay: `importance *= exp(−λ · hours_since_access)`. Returns count of concepts updated |
| `prune_below` | `(threshold: float = 0.1) → int` | `int` | Remove all concepts with `importance_score < threshold`. Returns count removed |

`add_concept()` now initialises `access_count=0`, `last_accessed=time.time()`, `importance_score=1.0` on every new node. `get_concept()` increments `access_count` and updates `last_accessed`.

---

### `language/dialogue_manager.py` — Dialogue State Tracking

```python
DialogueManager(cognitive_engine, language_module, causal_service=None, config=None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `get_context_hv` | `() → Optional[HyperVector]` | `HyperVector or None` | Returns current dialogue history HV `d_t = bundle(permute(d_{t-1}), utterance_hv)` |
| `clarification_request` | `(term: str) → str` | `str` | Returns `"Could you clarify what you mean by {term}?"` |

When `config.enable_dialogue_state_tracking=True`, each `process_turn()` call updates the rolling history HV, checks for topic shift (cosine sim < 0.3), and registers new entities via `EntityRegister`.

---

### `language/universal_input.py` — `UniversalInput.is_mathematical`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `is_mathematical` | `(text: str) → bool` | `bool` | Returns `True` if text matches any of 7 arithmetic/word-problem regex patterns |

---

### `vsa/resonator.py` — `HierarchicalResonatorNetwork`

```python
HierarchicalResonatorNetwork(codebooks: Dict[str, SemanticMemory], verbose: bool = False)
```

**L1 roles:** `AGENT`, `VERB`, `PATIENT`, `THEME`, `INSTRUMENT`  
**L2 roles:** `MODIFIER_AGENT`, `MODIFIER_VERB`, `MODIFIER_PATIENT`

| Method | Signature | Returns | Description |
|---|---|---|---|
| `factorize_hierarchical` | `(composite_hv, depth: int = 2) → Dict` | `{"L1": …, "L2": …}` | Factorize at two levels of abstraction |
| `factorize` | `(target_s, max_iter=20, convergence_threshold=0.65) → Dict` | `Dict` | Compatibility wrapper — factorizes at L1 |

---

### `integration/brain_fusion.py` — `MultiAgentSession`

```python
MultiAgentSession(engines=None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `add_engine` | `(engine) → int` | `int` | Add engine and return its index |
| `exchange_snapshots` | `() → Dict[int, HyperVector]` | `Dict` | Each agent bundles up to 10 SemanticMemory HVs into a summary |
| `negotiate_beliefs` | `(topic_hv) → HyperVector` | `HyperVector` | Majority-vote bundle of all agent belief proposals about `topic_hv` |
| `__len__` | `() → int` | `int` | Number of engines in session |

---

### `cognitive/theory_of_mind.py` — V8 Additions

| Method | Signature | Returns | Description |
|---|---|---|---|
| `model_other_agent` | `(agent_id: str, observed_actions: list) → MentalStateModel` | `MentalStateModel` | Update model of `agent_id`'s intentions from observed actions |
| `perspective_take` | `(topic_hv, agent_id: str) → dict` | `dict` | Return `agent_id`'s believed state about `topic_hv` |

---

### `learning/active_inference.py` — `ActiveInferenceLearner`

```python
ActiveInferenceLearner(curiosity_module=None, safety_threshold: float = 0.7)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `prediction_error` | `(action: str, state_hv) → float` | `[0, 1]` | Hamming distance between predicted and actual next state HV |
| `epistemic_value` | `(action: str, state_hv) → float` | `[0, 1]` | Curiosity/novelty value from `CuriosityModule` |
| `free_energy` | `(action: str, state_hv) → float` | `float` | `F = prediction_error − epistemic_value` |
| `should_veto` | `(action: str, state_hv) → bool` | `bool` | Returns `True` if `F > safety_threshold` |
| `update_world_model` | `(state_hv, action: str, next_state_hv) → None` | `None` | Predict next state via bundle of observed transitions |

**CognitiveEngine integration:** `decide()` calls `active_inference.free_energy(c.content, situation_hv)` for each coalition and adjusts `c.base_salience += w × (0.5 − F)` (default `w=0.2`). `record_outcome()` calls `update_world_model()`.

---

### `cognitive/metacognition.py` — V8 Additions

| Method | Signature | Returns | Description |
|---|---|---|---|
| `SafetyGate.check_free_energy` | `(action: str, state_hv, active_inference_learner) → bool` | `bool` | `True` if action passes free energy check (F ≤ threshold) |

---

### `reasoning/cognitive_engine.py` — V8 Additions

| Method | Signature | Returns | Description |
|---|---|---|---|
| `_solve_math` | `(text: str) → Optional[Dict]` | `{"answer", "expression", "explanation"} or None` | Route text through MathReasoner; returns result dict |
| `get_belief_summary` | `(topic_hv) → HyperVector` | `HyperVector` | Bundle of up to 3 SemanticMemory HVs as agent belief summary |

---

### `benchmarks/runner.py` — `BenchmarkRunner`

```python
BenchmarkRunner()
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `run_all` | `() → Dict[str, float]` | `{"babi_tasks": %, "math_word_problems": %, …}` | Run all 5 benchmarks and print tabular summary |

**Benchmark modules:** `babi_tasks` (20 tasks), `math_word_problems` (50 problems), `cross_domain_transfer` (5 tasks), `nlg_quality` (10 prompts), `dialogue_coherence` (multi-turn). Each exposes a `run_*_benchmark(engine=None) → float` function returning 0–100%.

---

## V9 — Substrate Modules

### `python/core/types/percept_packet.py` — `PerceptPacket`

Frozen dataclass. Universal input contract for all modality adapters.

| Field | Type | Description |
|---|---|---|
| `modality` | str | "text", "numeric", "dict", "snn", "multimodal", "stream" |
| `situation_hv` | HyperVector | Bundled situation representation |
| `active_predicates` | frozenset\[str\] | Grounded symbolic predicates |
| `entity_hvs` | dict | Named concept → HV |
| `relation_hvs` | list | (subj, pred, obj, triple_hv) |
| `confidence` | float | Encoding confidence |
| `raw_state` | dict\|None | Original raw input |
| `adapter_name` | str | Name of producing adapter |
| `adapter_trace` | dict | Glass-box metadata |

| Method | Signature | Description |
|---|---|---|
| `make` | `(modality, situation_hv, active_predicates, **kw) → PerceptPacket` | Convenience factory with defaults |

### `python/core/types/modality_adapter.py` — `ModalityAdapter`

Abstract base class. One abstract method: `encode(raw_input, task_tag) → PerceptPacket`.

### `python/core/adapters/dict_state_adapter.py` — `DictStateAdapter`

Wraps `GroundingVerifier + EpisodicMemory.create_situation_hv()`. Default adapter registered automatically by `register_task()`.

### `python/core/adapters/text_adapter.py` — `TextAdapter`

Delegates to `UniversalInput.ground_text()`.

### `python/core/adapters/numeric_adapter.py` — `NumericAdapter`

Delegates to `UniversalInput.ground_scalar()` or `ground_sequence()` depending on input type.

### `python/core/adapters/snn_adapter.py` — `SNNAdapter`

Wraps `SNNPerceptionModule.perceive()` with the same lazy-STDP scheduling as `perceive_and_decide()`.

### `python/core/adapters/multimodal_fuser.py` — `MultimodalFuser`

| Method | Signature | Description |
|---|---|---|
| `fuse` | `(packets: List[PerceptPacket]) → PerceptPacket` | VSA bundle of situation HVs, predicate union, averaged confidence |

### `python/core/adapters/stream_processor.py` — `StreamProcessor`, `StreamVerifier`

**StreamProcessor:**

| Method | Signature | Description |
|---|---|---|
| `ingest` | `(channel, value, timestamp?)` | Add data point to channel |
| `ready` | `() → bool` | True when any channel has ≥ window_size points |
| `emit` | `(adapter, task_tag) → PerceptPacket` | Compute temporal features and emit packet |
| `_extract_features` | `() → dict` | mean, min, max, trend, rate, anomaly per channel |

**StreamVerifier** extends `GroundingVerifier` — auto-generates `RISING_X`, `FALLING_X`, `ANOMALY_X` predicates from feature dicts.

### `eval/substrate_benchmarks.py`

| Function | Returns | Description |
|---|---|---|
| `benchmark_learning_curve(engine, task, gen, n)` | `[(ep, rate)]` | Per-episode success rate (rolling 10-episode avg) |
| `benchmark_transfer(engine, src, tgt, gen, ...)` | `{zero_shot_rate, few_shot_rate, delta}` | Zero-shot and few-shot transfer rates |
| `benchmark_lifelong(engine, tasks, gens, ...)` | `{peak_rates, final_rates, forgetting_ratio}` | Peak success + catastrophic forgetting measurement |
| `benchmark_efficiency(engine, task, gen, n)` | `{avg_ms, p95_ms, max_ms}` | Decision latency statistics |

---

## V10 — Intelligence Extension Modules

### `vsa/vsa_embedding_bridge.py` — `EmbeddingVSABridge`

```python
EmbeddingVSABridge(dim_in=768, hv_dim=10240, seed=42)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `embed_to_hv` | `(embedding: np.ndarray) → HyperVector` | `HyperVector` | Project dense vector to binary HV space via random projection |
| `hv_to_embed` | `(hv: HyperVector) → np.ndarray` | `np.ndarray` | Pseudo-inverse projection from HV bits back to embedding space |
| `batch_embed_to_hv` | `(embeddings: List[np.ndarray]) → List[HyperVector]` | `List[HyperVector]` | Batch conversion |
| `similarity_in_embed_space` | `(hv1, hv2) → float` | `float` | Cosine similarity measured in embedding space |
| `encode_text` | `(text: str) → HyperVector` | `HyperVector` | Text→HV via sentence-transformers or n-gram fallback |
| `try_load_sentence_transformer` | `(model_name: str) → bool` | `bool` | Attempt to load sentence-transformers model |

---

### `vsa/fhrr.py` — `FHRRVector`, `FHRRMemory`

```python
FHRRVector(dim=1024, seed=None)
FHRRMemory(dim=1024)
```

**FHRRVector:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `bind` | `(other) → FHRRVector` | `FHRRVector` | Element-wise complex multiplication |
| `unbind` | `(other) → FHRRVector` | `FHRRVector` | Conjugate multiplication (inverse of bind) |
| `bundle` | `(others: List[FHRRVector]) → FHRRVector` | `FHRRVector` | Sum phasors then renormalise |
| `similarity` | `(other) → float` | `float` | Cosine similarity of magnitude vectors |
| `gradient_similarity` | `(other) → float` | `float` | Differentiable similarity via real/imag parts |
| `encode_scalar` | `(x: float) → FHRRVector` | `FHRRVector` | Golden-ratio deterministic scalar encoding |
| `encode_symbol` | `(name: str, dim=1024) → FHRRVector` | `FHRRVector` | Hash-based deterministic symbol encoding |
| `to_gradient_input` | `() → np.ndarray` | `np.ndarray` | `[real, imag]` float array for gradient frameworks |
| `from_gradient_output` | `(arr, dim) → FHRRVector` | `FHRRVector` | Reconstruct from `[real, imag]` float array |

**FHRRMemory:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `store` | `(key: str, vector: FHRRVector)` | — | Bind key→vector and bundle into composite memory |
| `retrieve` | `(key: str) → FHRRVector` | `FHRRVector` | Unbind key from composite memory trace |
| `cleanup` | `(query, codebook) → (name, sim)` | `Tuple[str, float]` | Find nearest match in codebook |

---

### `vsa/rust_concurrent_shim.py` — Concurrent Memory Shim

Exports (Rust if available, Python fallback otherwise):

| Name | Type | Description |
|---|---|---|
| `SemanticMemoryConcurrent` | class | Concept store with `add_concept`, `get_concept`, `parallel_semantic_search`, `concept_count` |
| `EpisodicMemoryConcurrent` | class | Episode buffer with `add_episode`, `get_recent_episodes`, `search_by_task`, `size` |
| `Episode` | dataclass | `timestamp, task_tag, situation_hv, action, outcome, reward, impact_score` |
| `HyperVectorRegistry` | class | Named HV registry: `register(name, hv)`, `get(name)` |
| `PersistentStorage` | class | On-disk HV persistence: `store_hypervector`, `load_hypervector` |
| `parallel_bundle` | function | `(hvs: list) → HV` — bundle list of HVs |
| `batch_parallel_similarity_search` | function | `(query, hvs, k=5) → List[(idx, sim)]` |
| `batch_similarity_matrix` | function | `(hvs: list) → List[List[float]]` |
| `get_status` | function | `() → {use_rust: bool, available_classes: list}` |

---

### `language/ngram_nlu.py` — `NgramNLU`

```python
NgramNLU(n=2)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `train` | `(sentences, labels)` | — | Train Naive Bayes on labeled sentences |
| `classify` | `(text: str) → List[(label, prob)]` | `List[Tuple[str,float]]` | Ranked intent labels with probabilities |
| `extract_intent` | `(text: str) → Tuple[str, float]` | `(label, confidence)` | Top intent and confidence |
| `extract_entities` | `(text: str) → List[(entity, type)]` | `List[Tuple[str,str]]` | Named entities with type heuristic |

Default intent labels: `question`, `command`, `statement`, `greeting`, `farewell`.
Module constant `INTENT_LABELS` lists all defaults.

---

### `reasoning/attention_gwt_bridge.py` — `MultiHeadAttentionGWT`, `GWTAttentionBridge`

```python
MultiHeadAttentionGWT(n_heads=4, key_dim=64)
GWTAttentionBridge(n_heads=4, key_dim=64)
```

**MultiHeadAttentionGWT:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `compute_attention` | `(query_vec, coalitions) → Dict[str, float]` | `{source: weight}` | Compute per-coalition attention weights |

**GWTAttentionBridge:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `rerank` | `(coalitions: List[Coalition], situation_hv) → List[Coalition]` | `List[Coalition]` | Adjust `base_salience` by attention weight; returns re-ranked coalitions |

---

### `learning/rule_neural_scorer.py` — `RuleNeuralScorer`, `RuleFeaturizer`

```python
RuleNeuralScorer(n_features=6, hidden=16)
RuleFeaturizer()
```

**RuleFeaturizer:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `featurize` | `(rule) → np.ndarray` | `ndarray(6,)` | Extract `[confidence, support, fire_ratio, complexity, trend, has_task]` |

**RuleNeuralScorer:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `score` | `(rule) → float` | `float [0,1]` | Forward-pass score for one rule |
| `rank_rules` | `(rules: List) → List` | `List` | Rules sorted by score descending |
| `batch_score` | `(rules: List) → List[float]` | `List[float]` | Scores for all rules |
| `update` | `(rule, reward, lr=0.01)` | — | Online MSE gradient update |
| `save` | `(path: str)` | — | Save weights to `.npz` file |
| `load` | `(path: str)` | — | Load weights from `.npz` file |

---

### `cognitive/safety_verifier.py` — `SafetyRuleVerifier`, `SafetyGateVerifier`, `SafetyProperty`

```python
SafetyProperty(name, formula, severity="critical")
SafetyRuleVerifier()
SafetyGateVerifier(config=None)
```

**SafetyProperty:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `check` | `(rule) → (bool, str)` | `Tuple[bool, str]` | Evaluate formula in restricted namespace |

Available formula variables: `confidence`, `support`, `conditions`, `action`, `fire_count`, `FORBIDDEN_ACTIONS`, `len`.

**SafetyRuleVerifier:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `add_property` | `(prop: SafetyProperty)` | — | Register additional safety property |
| `verify_rule` | `(rule) → dict` | `{safe, violations, score, critical_violations}` | Check all properties against rule |
| `verify_ruleset` | `(rules: List) → dict` | `{total_rules, safe_rules, unsafe_rules, total_violations, results}` | Verify a list of rules |

**SafetyGateVerifier:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `gate_decision` | `(action, confidence, active_rules) → (bool, str)` | `Tuple[bool, str]` | Allow or block decision based on safety + confidence |

---

### `api/nsck_api.py` — `NSCKApiServer`

```python
NSCKApiServer(config=None)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `handle_decide` | `(state: dict, task_tag: str) → dict` | `{action, confidence, explanation, active_predicates}` or `{error, ...}` | One cognitive cycle |
| `handle_learn` | `(state, action, reward, task_tag, outcome) → dict` | `{status}` | Record learning update |
| `handle_sleep` | `(task_tag?) → dict` | `{status}` | Trigger offline consolidation |
| `handle_status` | `() → dict` | `{status, tasks, decisions, config}` | System health |
| `create_app` | `() → FastAPI or None` | FastAPI app or None | Build FastAPI app if available |
| `run` | `(host="127.0.0.1", port=8000)` | — | Start HTTP server (FastAPI+uvicorn or stdlib fallback) |

---

## 17. V12 — ImageAdapter, AudioAdapter, NSCKSubstrate

---

### `adapters/image_adapter.py` — `ImageAdapter`

```python
ImageAdapter()
```

Encodes 2D (H×W) or 3D (H×W×C) numpy image arrays into `PerceptPacket` using a 65-dimensional feature vector (spatial-grid statistics, colour histograms, Sobel edge density) encoded via FPE.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `encode` | `(image: np.ndarray, task_tag: str) → PerceptPacket` | `PerceptPacket(modality="image")` | Extract features → FPE HV + predicates |

**Predicates emitted:** `IMAGE_BRIGHT`, `IMAGE_DARK`, `IMAGE_UNIFORM`, `IMAGE_HIGH_CONTRAST`, `IMAGE_DETAILED`, `IMAGE_SMOOTH`, `IMAGE_COLOR`, `IMAGE_GRAYSCALE`, `IMAGE_DEGENERATE`.

**Module-level helpers:**

| Symbol | Description |
|---|---|
| `_extract_image_features(img)` | Returns `np.ndarray` of 65 features |
| `_features_to_hv(feature_vec)` | FPE: quantise each dim → bind with role HV → bundle |
| `_get_descriptors(img, feature_vec)` | Returns `List[str]` of human-readable descriptors |
| `_CODEBOOK` | Module-level list of 256 HVs (seeds `i*31+7777`) |

---

### `adapters/audio_adapter.py` — `AudioAdapter`

```python
AudioAdapter()
```

Encodes 1-D float audio waveforms into `PerceptPacket` using a 23-dimensional feature vector (13 MFCC + 4 energy bands + ZCR + spectral centroid/rolloff + RMS/peak/log-length) encoded via FPE.

| Method | Signature | Returns | Description |
|---|---|---|---|
| `encode` | `(audio: np.ndarray, task_tag: str) → PerceptPacket` | `PerceptPacket(modality="audio")` | Extract DSP features → FPE HV + predicates |

**Class attribute:** `SAMPLE_RATE = 16000` (assumed default sample rate).

**Predicates emitted:** `AUDIO_LOUD`, `AUDIO_QUIET`, `AUDIO_NOISY`, `AUDIO_TONAL`, `AUDIO_HIGH_FREQ`, `AUDIO_LOW_FREQ`, `AUDIO_NARROW_BAND`, `AUDIO_WIDE_BAND`, `AUDIO_SIGNAL`, `AUDIO_SILENT`, `AUDIO_UNKNOWN`.

**Module-level helpers:**

| Symbol | Description |
|---|---|
| `_extract_audio_features(audio, sample_rate)` | Returns `np.ndarray` of 23 features |
| `_features_to_hv(feature_vec)` | FPE: quantise each dim → bind with role HV → bundle |
| `_get_audio_descriptors(feature_vec)` | Returns `List[str]` of descriptors |
| `_AUDIO_CODEBOOK` | Module-level list of 256 HVs (seeds `i*37+8888`) |

---

### `substrate.py` — `NSCKSubstrate`, `SubstrateResult`

```python
NSCKSubstrate(config: NSCKConfig = None)
SubstrateResult  # frozen dataclass
```

`NSCKSubstrate` is the recommended high-level API for all third-party code. It wraps `CognitiveEngine` with automatic input-type routing, a plugin encoder registry, and cross-modal learning.

**SubstrateResult fields:**

| Field | Type | Description |
|---|---|---|
| `chosen_action` | `str` | Selected action from GWT competition |
| `confidence` | `float` | Decision confidence (0.0–1.0) |
| `explanation` | `str` | Human-readable reasoning trace |
| `predicates` | `Set[str]` | Active symbolic predicates |
| `trace` | `Dict[str, Any]` | Full glass-box trace |
| `modalities_processed` | `List[str]` | Which modalities were active |
| `generalization_triggered` | `bool` | Whether generalisation ran this cycle |

**NSCKSubstrate methods:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `register_task` | `(task_tag: str)` | — | Register a new task/domain |
| `process` | `(input_data, task_tag, available_actions?) → SubstrateResult` | `SubstrateResult` | Process any input: `str` → text, 2D+ndarray → `ImageAdapter`, 1D list/array → `NumericSequenceAdapter`, `dict` → `DictStateAdapter`, `PerceptPacket` → direct |
| `process_multimodal` | `(inputs: Dict[str, Any], task_tag) → SubstrateResult` | `SubstrateResult` | Fuse multiple modalities simultaneously; keys `"image"` and `"audio"` routed to respective adapters |
| `learn` | `(state, action, reward, task_tag, outcome)` | — | Record a (state, action, reward) experience |
| `sleep` | `(task_tag?)` | `dict` | Offline consolidation + generalisation |
| `remember` | `(query, task_tag?, top_k) → List[dict]` | `List[dict]` | Recall similar past episodes |
| `register_encoder` | `(modality_name: str, encoder_fn)` | — | Register custom encoder `fn(data, task_tag) → PerceptPacket` |
| `get_knowledge` | `(concept: str) → dict` | `{concept, known, similar}` | Query semantic memory |
| `get_stats` | `() → dict` | stats dict | System statistics |

**Input routing in `process()`:**

| Input type | Routed to |
|---|---|
| `str` | `DictStateAdapter` (via `{"text": ...}`) |
| `np.ndarray` with `ndim >= 2` | `ImageAdapter` |
| `np.ndarray` with `ndim == 1` | `NumericSequenceAdapter` |
| `list/tuple` of numbers | `NumericSequenceAdapter` |
| `dict` | `DictStateAdapter` |
| `PerceptPacket` | Direct pass-through |

**Modality routing in `_encode_single()` / `process_multimodal()`:**

| Modality key | Adapter |
|---|---|
| `"image"` | `ImageAdapter` |
| `"audio"` | `AudioAdapter` |
| `str` data | Text path |
| `list` / `ndarray` data | `NumericSequenceAdapter` |
| `dict` data | `DictStateAdapter` |
