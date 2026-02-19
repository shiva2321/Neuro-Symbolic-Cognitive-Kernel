# Module Reference

Per-module API reference for the NSCK core. All classes, methods, and parameters are taken directly from the source code.

---

## Table of Contents

- [VSA Foundation](#vsa-foundation)
- [Reasoning Layer](#reasoning-layer)
- [Memory Systems](#memory-systems)
- [Perception Layer](#perception-layer)
- [Learning Subsystem](#learning-subsystem)
- [Cognitive Layer](#cognitive-layer)
- [Language Layer](#language-layer)
- [Integration Layer](#integration-layer)

---

## VSA Foundation

### HyperVectorPy (`vsa/hypervec_py.py` — 361 LOC)

The core data type representing 10,240-bit binary vectors.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `HyperVectorPy(seed: int)` | Create deterministic HV from uint32 seed |
| `xor` | `xor(other) → HyperVectorPy` | Binding operation (element-wise XOR) |
| `bundle` | `bundle(other) → HyperVectorPy` | Superposition (majority vote) |
| `permute` | `permute(n: int = 1) → HyperVectorPy` | Circular shift by n positions |
| `similarity` | `similarity(other) → float` | Normalized Hamming similarity [0, 1] |
| `cosine_similarity` | `cosine_similarity(other) → float` | Cosine similarity [-1, 1] |
| `weighted_bundle` | `weighted_bundle(other, w) → HyperVectorPy` | Weight-biased bundle |
| `lsh_hash` | `lsh_hash(seed, bits) → int` | Locality-sensitive hash |

### CleanupMemory (`vsa/hypervec_py.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `add(name: str, hv: HyperVectorPy)` | Register a clean prototype |
| `cleanup` | `cleanup(noisy_hv) → (str, float)` | Find nearest prototype |

### hypervec_shim (`vsa/hypervec_shim.py` — 320 LOC)

Backend selector. Tries Rust (`hypervec_rs`) first, falls back to `HyperVectorPy`.

**Exports (available via either backend):**
- `HyperVector` — The active HV class
- `SemanticMemoryConcurrent` — Thread-safe semantic memory (Rust only)
- `EpisodicMemoryConcurrent` — Thread-safe episodic memory (Rust only)
- `CognitiveWorkerPool` — Parallel computation pool (Rust only)
- `PersistentStorage` — Disk persistence (Rust only)
- `AsyncCognitiveRuntime` — Async runtime (Rust only)
- `parallel_similarity_search` — Parallel batch similarity (Rust only)

---

## Reasoning Layer

### CognitiveEngine (`reasoning/cognitive_engine.py` — 1,188 LOC)

The central orchestrator. Instantiates all modules and runs the cognitive loop.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `decide` | `decide(state, available_actions, task_tag)` | `CognitiveState` | Main decision loop (see Workflows) |
| `learn` | `learn(state, action, reward, next_state, task_tag)` | `None` | Store episode, induce rules, update causal graph |
| `record_outcome` | `record_outcome(state, action, reward, next_state, task_tag)` | `None` | TD(0) Q-update + confidence update |
| `register_task` | `register_task(tag, predicates, actions, ...)` | `None` | Register a domain |
| `sleep` | `sleep()` | `None` | Offline consolidation |
| `explain` | `explain()` | `str` | Natural language explanation of last decision |
| `get_stats` | `get_stats()` | `dict` | Episode count, rule count, etc. |
| `set_mission_goal` | `set_mission_goal(goal)` | `None` | Set current mission objective |

**Key attributes:**
- `q_values: dict` — Tabular Q-values `{(state_key, action): float}`
- `rules: list` — Learned rules from RuleLearner
- `global_workspace: GlobalWorkspace` — GWT competition arena
- `episodic_memory: EpisodicMemory` — Episode storage
- `semantic_memory: SemanticMemory` — Concept graph
- `self_model: SelfModel` — Confidence tracker
- `curiosity: CuriosityModule` — Exploration module

### GlobalWorkspace (`reasoning/global_workspace.py` — 312 LOC)

LIDA-lite Global Workspace Theory implementation.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `compete` | `compete(coalitions, focused_module, mission_goal)` | `Coalition` | Run coalition competition |
| `compete_with_rehearsal` | `compete_with_rehearsal(coalitions, world_model, ...)` | `Coalition` | Competition + mental simulation |
| `register_module` | `register_module(module: WorkspaceModule)` | `None` | Add broadcast subscriber |
| `broadcast` | `broadcast(content)` | `None` | Broadcast winner to all subscribers |
| `register_danger` | `register_danger(name, hv)` | `None` | Register danger pattern for veto |

### Coalition (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Module name (EXTERNAL, RULES, Q_LEARNING, etc.) |
| `action` | `str` | Proposed action |
| `salience` | `float` | Base salience [0, 1] |
| `relevance` | `float` | Context relevance [0, 1] |
| `confidence` | `float` | Module's confidence [0, 1] |
| `content` | `dict` | Arbitrary metadata |

### STRIPSPlanner (`reasoning/planner.py` — 296 LOC)

A* search over STRIPS operators.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `plan` | `plan(initial_state, goal, max_steps)` | `List[str]` | A* plan from initial to goal |
| `learn_operators_from_graph` | `learn_operators_from_graph(graph)` | `None` | Extract STRIPS ops from CausalGraph |
| `set_reasoner` | `set_reasoner(reasoner)` | `None` | Attach CausalReasoner for effects |
| `decompose_goal` | `decompose_goal(goal)` | `List[Set[str]]` | Hierarchical goal decomposition |

### CausalDiscovery (`reasoning/causal_reasoning.py` — 954 LOC)

Statistical causal discovery using ΔP.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `observe` | `observe(context, causes, effects)` | `None` | Record an observation |
| `induce_graph` | `induce_graph()` | `CausalGraph` | Build graph from accumulated stats |
| `incremental_update` | `incremental_update()` | `None` | Update existing graph |

### CausalGraph

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_causes` | `add_causes(cause, effect, strength)` | `None` | Add causal link |
| `add_prevents` | `add_prevents(cause, effect, strength)` | `None` | Add preventive link |
| `get_effects` | `get_effects(concept)` | `List[str]` | Forward chain |
| `get_causes` | `get_causes(concept)` | `List[str]` | Backward chain |

### CausalReasoner

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_immediate_effects` | `get_immediate_effects(concept)` | `list` | Direct effects |
| `trace_chain` | `trace_chain(start, max_depth)` | `CausalChain` | DFS forward chain |
| `counterfactual` | `counterfactual(premise, question)` | `CounterfactualResult` | What-if analysis |

### RuleLearner (`reasoning/rule_learner.py` — 624 LOC)

Rule induction from observation. Implements `WorkspaceModule` for GWT broadcasts.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `observe` | `observe(state, action, reward, task_tag)` | `None` | Record action in context |
| `induce_rules` | `induce_rules()` | `List[Rule]` | Promote candidates to rules |
| `get_applicable_rules` | `get_applicable_rules(active_preds, task_tag)` | `List[Rule]` | Match rules to current state |
| `receive_broadcast` | `receive_broadcast(content)` | `None` | GWT broadcast handler |

### AnalogyEngine (`reasoning/analogy.py` — 528 LOC)

Analogical reasoning and cross-domain transfer.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `register_abstract` | `register_abstract(name, slots)` | `None` | Register abstract schema |
| `register_domain` | `register_domain(domain, mappings)` | `None` | Register domain mappings |
| `find_analogy` | `find_analogy(source, target)` | `Analogy` | Find structural alignment |
| `transfer_rule` | `transfer_rule(rule, source_domain, target_domain)` | `Rule` | Transfer rule across domains |

### ContextEngine (`reasoning/context_engine.py` — 440 LOC)

Polysemy disambiguation using VSA + semantic memory.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `register_meaning` | `register_meaning(word, sense, prototype_hv)` | `None` | Register a word sense |
| `disambiguate` | `disambiguate(word, context_hvs)` | `DisambiguatedMeaning` | Select best sense for context |

---

## Memory Systems

### EpisodicMemory (`memory/episodic_memory.py` — 398 LOC)

Two-tier episodic memory with LSH indexing.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `record` | `record(episode: LiveEpisode)` | `None` | Store new episode |
| `recall_similar` | `recall_similar(query_hv, k, task_tag)` | `List[LiveEpisode]` | LSH + exact-match retrieval |
| `create_situation_hv` | `create_situation_hv(predicates)` | `HyperVector` | Bundle predicates into HV |
| `sample` | `sample(k, task_tag)` | `List[LiveEpisode]` | Random sample for replay |

### LiveEpisode (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `state` | `dict` | Full state snapshot |
| `action` | `str` | Action taken |
| `reward` | `float` | Reward received |
| `next_state` | `dict` | Resulting state |
| `situation_hv` | `HyperVector` | VSA encoding of state |
| `timestamp` | `float` | When it happened |
| `task_tag` | `str` | Which task |
| `emotion` | `str` | Emotional label (optional) |

### SemanticMemory (`memory/semantic_memory.py` — 217 LOC)

NetworkX graph + VSA dual representation.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_concept` | `add_concept(name, properties, hv_override)` | `None` | Add concept node |
| `add_relation` | `add_relation(source, relation, target, weight)` | `None` | Add typed edge |
| `query` | `query(query_hv, k)` | `List[(str, float)]` | VSA similarity search |
| `spread_activation` | `spread_activation(start, steps, decay)` | `dict` | Iterative activation spread |
| `get_inherited_properties` | `get_inherited_properties(concept)` | `dict` | Property inheritance via IS-A |

### StagedRecall (`memory/staged_recall.py` — 134 LOC)

Multi-level memory retrieval.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `query` | `query(query_hv, top_k, threshold)` | `List[(str, float)]` | L0 → L2 → L3 cascading search |

---

## Perception Layer

### SNNPerceptionModule (`perception/snn_perception.py` — 561 LOC)

Spiking Neural Network with LIF neurons and STDP learning.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `perceive` | `perceive(sensory_input)` | `dict` | Full pipeline: input → spikes → concept HV |
| `process_timestep` | `process_timestep(input_current)` | `np.ndarray` | Single LIF update + spikes |

### LIFNeuronLayer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neurons` | 128 | Number of LIF neurons |
| `tau_m` | 20.0 | Membrane time constant (ms) |
| `v_threshold` | 1.0 | Spike threshold |
| `v_reset` | 0.0 | Post-spike reset |
| `dt` | 1.0 | Timestep (ms) |

### GroundingVerifier (`perception/grounding_verifier.py` — 585 LOC)

Maps symbolic predicates to state-dependent checks.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `register_predicate` | `register_predicate(name, check_fn, task_tag)` | `None` | Register predicate lambda |
| `register_action` | `register_action(name, fn, task_tag)` | `None` | Register action handler |
| `get_active_predicates` | `get_active_predicates(state, task_tag)` | `List[str]` | Evaluate all predicates |
| `verify_predicate` | `verify_predicate(name, state, task_tag)` | `GroundingResult` | Check single predicate |
| `verify_rule` | `verify_rule(rule, state, task_tag)` | `GroundingResult` | Check rule applicability |

### VSA-SNN Bridge (`perception/vsa_snn_bridge.py` — 461 LOC)

Bidirectional spike ↔ HyperVector conversion.

| Class | Method | Description |
|-------|--------|-------------|
| `RateCoder` | `encode(spike_train)` | Spike rates → bundled neuron HVs |
| `TemporalCoder` | `encode(spike_train)` | First-spike times → permuted neuron HVs |
| `HVtoSpikeDecoder` | `decode(target_hv)` | HyperVector → spike pattern |
| `ConceptMapper` | `recognize(hv)` | Match HV against known concepts |
| `ConceptMapper` | `learn_from_spikes(...)` | Learn new concepts from spike patterns |

---

## Learning Subsystem

### HebbianMatrixNumPy (`learning/hebbian.py` — 506 LOC)

Oja's rule with reward modulation and eligibility traces.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `hebbian_update` | `hebbian_update(pre, post, reward)` | `None` | Three-factor Hebbian update |
| `get_weights` | `get_weights()` | `np.ndarray` | Current weight matrix |
| `modulate` | `modulate(reward)` | `None` | Apply reward modulation |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | — | Pre-synaptic dimension |
| `output_dim` | — | Post-synaptic dimension |
| `lr` | 0.01 | Learning rate (α) |
| `eligibility_decay` | 0.9 | Trace decay (λ) |

### VSAHebbianLearner

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update_associations` | `update_associations(concept_a, concept_b, reward)` | `None` | Learn association between concept HVs |
| `get_association` | `get_association(concept)` | `HyperVector` | Retrieve associated concept |

### CuriosityModule (`learning/curiosity.py` — 336 LOC)

VSA-based novelty detection and exploration control.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `should_explore` | `should_explore(situation_hv, task_tag, confidence)` | `ExplorationDecision` | Explore or exploit? |
| `compute_novelty` | `compute_novelty(hv, task_tag)` | `float` | Novelty score [0, 1] |
| `record_visit` | `record_visit(hv, task_tag)` | `None` | Mark state as visited |
| `record_outcome` | `record_outcome(task_tag, reward)` | `None` | Update learning progress |

---

## Cognitive Layer

### SafetyGate (`cognitive/metacognition.py` — 504 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `is_safe` | `is_safe(action, state, task_tag)` | `bool` | Static safety check |

### MetacognitiveEngine

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `infer` | `infer(facts, task_tag)` | `InferenceResult` | Confidence-monitored inference |
| `detect_conflicts` | `detect_conflicts(results)` | `List[Conflict]` | Cross-module conflict detection |

### SelfModel (`cognitive/self_model.py` — 255 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_confidence` | `get_confidence(task_tag)` | `float` | Current confidence estimate |
| `update_confidence` | `update_confidence(task_tag, outcome)` | `None` | Update from outcome |
| `predict_success` | `predict_success(task_tag)` | `float` | Predicted success probability |
| `update` | `update(task_tag, predicted, actual, action, reward, state)` | `None` | Full update |
| `get_calibration_error` | `get_calibration_error()` | `float` | Mean |predicted - actual| |
| `get_improvement_trend` | `get_improvement_trend(task_tag)` | `str` | "improving"/"stable"/"declining" |

### TheoryOfMind (`cognitive/theory_of_mind.py` — 312 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update_agent_perspective` | `update_agent_perspective(agent_id, state)` | `None` | Update agent's mental model |
| `predict_action` | `predict_action(agent_id)` | `str` | Predict agent's next action |
| `detect_false_belief` | `detect_false_belief(agent_id, reality)` | `bool` | Sally-Anne style detection |
| `recursive_belief` | `recursive_belief(a, b, concept)` | `Any` | "A believes B believes X" |

### EmotionSystem (`cognitive/emotion_system.py` — 420 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update_from_drives` | `update_from_drives(drives, reward)` | `None` | Update valence/arousal |
| `recognize_emotion_from_text` | `recognize_emotion_from_text(text)` | `str` | Text → emotion label |
| `get_emotion_blend` | `get_emotion_blend()` | `dict` | Current emotion distribution |
| `get_mood` | `get_mood()` | `str` | Dominant emotion |

---

## Language Layer

### LanguageModule (`language/language_module.py` — 484 LOC)

Dual-mode NLU: LLM (llama_cpp + phi-3) when available, otherwise 300-line rule-based NLU.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `understand` | `understand(text)` | `dict` | Parse text → structured intent |
| `generate` | `generate(intent_data)` | `str` | Intent → natural language |

### DialogueManager (`language/dialogue_manager.py` — 145 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `respond` | `respond(user_text)` | `str` | Process user input, return response |
| `process_turn` | `process_turn(text)` | `str` | Alias for respond |

### TextKnowledgeLearner (`language/text_knowledge_learner.py` — 1,071 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `learn_from_text_file` | `learn_from_text_file(path)` | `LearningSession` | Ingest text file → knowledge |
| `learn_from_text` | `learn_from_text(text)` | `LearningSession` | Ingest text string → knowledge |
| `get_session_summary` | `get_session_summary()` | `dict` | Facts learned, relations, etc. |

### UniversalInput (`language/universal_input.py` — 947 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `encode` | `encode(value)` | `HyperVector` | Any input → 10,240-bit HV |

Handles: `str`, `int`, `float`, `bool`, `dict`, `list`, `None`. Text uses 4-component encoding (keyword + n-gram + word-order + phrase-structure).

---

## Integration Layer

### NSCKConfig (`integration/config.py` — 73 LOC)

Dataclass holding all hyperparameters. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `hv_dimension` | 10240 | HyperVector dimension |
| `novelty_threshold` | 0.5 | Curiosity exploration threshold |
| `gwt_threshold` | 0.5 | GWT attention threshold |
| `q_learning_rate` | 0.1 | TD(0) α |
| `q_discount` | 0.9 | TD(0) γ |
| `epsilon_start` | 0.3 | Initial exploration rate |
| `epsilon_min` | 0.1 | Minimum exploration rate |

### BrainStore (`integration/persistence.py` — 852 LOC)

SQLite persistence for all long-term storage.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `save_rule` | `save_rule(rule)` | `None` | Persist a learned rule |
| `load_rules` | `load_rules(task_tag)` | `List[Rule]` | Load task-specific rules |
| `record_episode` | `record_episode(episode)` | `None` | Persist episode |
| `flush_episodes` | `flush_episodes(episodes)` | `None` | Batch persist |
| `prune_old_episodes` | `prune_old_episodes(task_tag, max_count)` | `None` | Remove old episodes |
| `save_concept` | `save_concept(concept)` | `None` | Persist semantic concept |
| `checkpoint` | `checkpoint()` | `None` | Full state save |

### BrainFusion (`integration/brain_fusion.py` — 481 LOC)

Multi-task brain management.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_or_create_brain` | `get_or_create_brain(task_tag)` | `TaskBrain` | Get task-specific brain |
| `FusedBrain.resolve_rules` | `resolve_rules(facts, task_tag)` | `List[QueryResult]` | Cross-brain rule resolution |

### ExplanationGenerator (`integration/explanation.py` — 433 LOC)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `explain_action` | `explain_action(action, coalitions, winner, ...)` | `Explanation` | Generate NL explanation |
