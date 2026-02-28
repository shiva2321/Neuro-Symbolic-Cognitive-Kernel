# Module Reference — NSCK V14

Complete reference for every module in the NSCK codebase. 100 core modules, 238 classes across 12 subsystems. Every class and method listed here is taken directly from source code.

> **Ground truth is always the source files** under `nsck/python/core/`, `nsck/api/`, and `nsck_ai_model/`.

---

## Table of Contents

- [Substrate](#substrate)
- [VSA](#vsa)
- [Perception](#perception)
- [Memory](#memory)
- [Reasoning](#reasoning)
- [Learning](#learning)
- [Language](#language)
- [Cognitive](#cognitive)
- [Integration](#integration)
- [Adapters](#adapters)
- [Types](#types)
- [API](#api)
- [Multimodal](#multimodal)
- [Training](#training)

---

## Substrate

### `python/core/substrate.py`

**NSCKSubstrate** — Top-level orchestrator. Wires perception, memory, reasoning, and learning into a single pipeline.

| Method | Description |
|--------|-------------|
| `process(input_data, task_tag, available_actions)` | Full perceive → reason → act cycle. Returns `SubstrateResult`. |
| `process_multimodal(inputs, task_tag, available_actions)` | Multi-modal variant of `process()`. |
| `ingest(input_data, task_tag)` | V13 API — encode and store without acting. |
| `feedback(action, reward, task_tag)` | V13 API — reward signal for learning. |
| `learn(task_tag)` | Trigger offline learning (rule induction, Hebbian updates). |
| `sleep(task_tag)` | Consolidation cycle (episodic replay, prototype building, drift check). |
| `remember(query, task_tag)` | Query memory systems by HV similarity. |
| `register_task(task_tag)` | Register a new task context. |
| `register_encoder(modality, adapter)` | Register a modality adapter. |
| `get_knowledge(task_tag)` | Retrieve current knowledge state. |
| `get_stats()` | Return system-wide statistics. |

**SubstrateResult** (dataclass):
`chosen_action`, `confidence`, `explanation`, `predicates`, `trace`, `modalities_processed`, `generalization_triggered`, `kle_uncertainty`, `uncertainty_bounds`, `encoding_stats`, `procedural_hit`.

---

## VSA

`python/core/vsa/`

### `hypervec_py.py`

**HyperVectorPy** — Pure-Python binary hypervector (10 000-bit default).

| Method | Description |
|--------|-------------|
| `xor(other)` | Binding via component-wise XOR. |
| `bundle(other)` | Bundling via majority vote. |
| `similarity(other)` | Normalised Hamming similarity. |
| `cosine_similarity(other)` | Cosine similarity on ±1 encoding. |
| `similarity_robust(other)` | Robust similarity with noise tolerance. |
| `permute(n)` | Cyclic bit shift by `n` positions. |
| `permute_inverse(n)` | Inverse cyclic shift. |
| `negate()` | Bit-flip negation. |
| `lsh_hash(n_planes)` | Locality-sensitive hash. |
| `weighted_bundle(others, weights)` | Weighted majority bundling. |
| `from_bits(bits)` | Construct from explicit bit array. |
| `zero()` | All-zero vector. |

**CleanupMemory** — Nearest-neighbour auto-associative memory.

| Method | Description |
|--------|-------------|
| `register(label, hv, force)` | Store a prototype. |
| `cleanup(noisy_hv, threshold)` | Return nearest stored label. |
| `cleanup_or_keep()` | Clean up or return input if below threshold. |
| `batch_register(items)` | Bulk registration. |
| `load_from_store()` | Load from persistent store. |
| `get_stats()` | Memory statistics. |
| `clear()` | Reset memory. |

### `hypervec_shim.py`

Transparent shim: imports Rust `hypervec_rs` if available, falls back to `HyperVectorPy`. Exposes `HyperVectorRegistry`, `HyperVector`, etc.

### `fhrr.py`

**FHRRVector** — Complex-phasor hypervector for continuous binding.

| Method | Description |
|--------|-------------|
| `bind(other)` | Element-wise phasor multiplication. |
| `unbind(other)` | Inverse binding (conjugate multiply). |
| `bundle(other)` | Element-wise phasor averaging. |
| `similarity(other)` | Cosine similarity of phasors. |
| `gradient_similarity(other)` | Differentiable similarity. |
| `encode_scalar(value)` | Map a scalar to a phasor HV. |
| `encode_symbol(label)` | Map a symbol to a random phasor HV. |

**FHRRMemory** — Associative store over FHRR vectors: `store()`, `retrieve()`, `cleanup()`.

### `vsa_embedding_bridge.py`

**EmbeddingVSABridge** — Dense vector ↔ VSA conversion.

| Method | Description |
|--------|-------------|
| `embed_to_hv(embedding)` | Dense → binary HV via random projection. |
| `hv_to_embed(hv)` | Binary HV → dense approximation. |
| `batch_embed_to_hv(embeddings)` | Batch conversion. |
| `similarity_in_embed_space(hv1, hv2)` | Compare HVs in embedding space. |
| `encode_text(text)` | Text → HV via embedding model. |

### `resonator.py`

**ResonatorNetwork** — Iterative factorization of bundled HVs.

| Method | Description |
|--------|-------------|
| `factorize(target_s, max_iter, threshold)` | Decompose a composite HV into factors. |
| `reset_estimates()` | Reset internal state. |

**HierarchicalResonatorNetwork** — L1/L2 hierarchical factorization.

| Method | Description |
|--------|-------------|
| `factorize(composite_hv)` | Single-level factorization. |
| `factorize_hierarchical(composite_hv, depth)` | Multi-level recursive factorization. |

### `universal_hv_encoder.py`

**UniversalHVEncoder** (V13) — Encode multi-modal signals with statistics.

| Method | Description |
|--------|-------------|
| `encode(signal, label)` | Encode a signal into an HV. |
| `encode_with_stats()` | Encode with encoding statistics. |
| `hebbian_update()` | Update encoder weights via Hebbian rule. |
| `get_top_features()` | Return most informative features. |
| `reset_weights()` | Reset learned weights. |

### `rust_concurrent_shim.py`

Wrappers for Rust-accelerated exports (falls back to Python):
- **SemanticMemoryConcurrent**
- **EpisodicMemoryConcurrent**
- **CognitiveWorkerPool**

---

## Perception

`python/core/perception/`

### `snn_perception.py`

**SimpleConceptMapper** — Map spike patterns to named concepts.

| Method | Description |
|--------|-------------|
| `register_concept(label, hv)` | Register a concept prototype. |
| `recognize_pattern(hv)` | Classify a pattern by nearest concept. |
| `label_concept(label)` | Get label for a concept. |
| `get_concept_label()` | Return current concept label. |
| `get_concept_hv()` | Return concept HV. |

**LIFNeuronLayer** — Leaky integrate-and-fire neuron layer: `step(input_current)`, `reset()`, `get_spike_train()`.

**PythonSnnCore** / **PythonStdpEngine** — Pure-Python SNN simulation and STDP learning.

**SNNPerceptionModule** — Full SNN perception pipeline.

| Method | Description |
|--------|-------------|
| `perceive(sensory_input, duration_ms, learn)` | Run SNN on sensory input. |
| `update_weights()` | Apply STDP weight updates. |
| `get_stats()` | Pipeline statistics. |
| `reset_stats()` | Reset counters. |

**SNNWorkspaceModule** — GWT-compatible SNN wrapper: `process(sensory_data) → (HyperVector, float)`, `get_status()`.

### `snn_shim.py`

SNN backend selector — imports Rust `snn_rs` if available, falls back to Python.

### `snn_integration.py`

**SNNWorkspaceAdapter** — Integrates SNN perception with Global Workspace Theory.

### `signal_ingestor.py`

**SignalIngestor** (V13) — Raw signal pre-processing.

| Method | Description |
|--------|-------------|
| `ingest(data, label)` | Detect type and produce `TypedSignal`. |

### `symbol_grounding.py`

Symbol grounding — HV → predicate extraction.

### `grounding_verifier.py`

**GroundingVerifier** — Verify predicate groundings.

| Method | Description |
|--------|-------------|
| `register_predicate(name, fn)` | Register a predicate function. |
| `register_action(name, fn)` | Register an action validator. |
| `verify_predicate(name, state)` | Check a predicate against state. |
| `verify_action(name, state)` | Check an action is valid. |
| `verify_rule(rule, state)` | Verify a full rule. |
| `get_active_predicates(state)` | Return all true predicates for state. |

Factory functions: `create_snake_verifier()`, `create_pong_verifier()`, `create_maze_verifier()`, `create_collector_verifier()`.

### `vsa_snn_bridge.py`

Bridge between SNN spike patterns and VSA hypervectors.

- **RateCoder**: `encode(spike_train, time_window_ms) → SpikeEncoding`
- **TemporalCoder**: `encode(spike_train, dt_ms) → SpikeEncoding`
- **HVtoSpikeDecoder**: `decode(target_hv, n_steps)`
- **ConceptMapper**: `register_concept()`, `recognize()`, `learn_from_spikes()`

### `stream_encoder.py`

**StreamEncoder** — Encode temporal streams.

---

## Memory

`python/core/memory/`

### `semantic_memory.py`

**SemanticMemory** — Graph-structured concept store with spreading activation.

| Method | Description |
|--------|-------------|
| `add_concept(label, hv)` | Store a concept. |
| `get_concept(label)` | Retrieve a concept HV. |
| `add_relation(src, rel, tgt)` | Add a typed edge. |
| `query(hv, top_k)` | Nearest-neighbour concept lookup. |
| `spread_activation(start, steps)` | Propagate activation through graph. |
| `mark_path(nodes)` | Stigmergic path marking. |
| `evaporate_stigmergy()` | Decay stigmergy traces. |
| `get_stigmergy(node)` | Read stigmergy value. |
| `extract_schema(concepts)` | Extract relational schema. |
| `build_prototypes()` | Build category prototypes from instances. |
| `infer_transitive(rel)` | Infer transitive closure of a relation. |
| `get_inherited_properties(concept)` | Retrieve properties via IS-A hierarchy. |
| `decay_concepts()` | Time-based concept decay. |
| `prune_below(threshold)` | Remove low-activation concepts. |
| `save(path)` / `load(path)` | Persist / restore memory. |

### `episodic_memory.py`

**EpisodicMemory** — Temporal episode store.

| Method | Description |
|--------|-------------|
| `reset()` | Clear all episodes. |
| `record(episode)` | Store a new episode. |
| `sample(n)` | Random sample of episodes. |
| `recall_similar(query_hv, top_k)` | Similarity-based recall. |
| `recall_recent(n)` | Recency-based recall. |
| `retrieve_salient(threshold)` | Retrieve high-salience episodes. |
| `recall_by_outcome(outcome)` | Filter by outcome label. |
| `recall_by_reward(min_reward)` | Filter by reward threshold. |
| `get_statistics()` | Episode store statistics. |

### `procedural_memory.py`

**ProceduralMemory** (V13) — Skill cache with fast-path lookup.

| Method | Description |
|--------|-------------|
| `cache_skill(action, context_hv, reward)` | Store a skill. |
| `recall_action(context_hv)` | Fast-path skill retrieval. |
| `get_statistics()` | Cache statistics. |

### `cross_modal_associative_memory.py`

**CrossModalAssociativeMemory** (V13) — Bind entities across modalities.

| Method | Description |
|--------|-------------|
| `bind(modality1, modality2, label)` | Cross-modal binding. |
| `recall(query_hv)` | Retrieve associated HV. |
| `recall_by_label(label)` | Label-based retrieval. |
| `get_statistics()` | Binding statistics. |

### `concept_drift_detector.py`

**ConceptDriftDetector** (V13) — Monitor semantic stability.

| Method | Description |
|--------|-------------|
| `snapshot(concept, hv)` | Record baseline HV. |
| `check(concept, hv)` | Compare against baseline → `DriftEvent`. |
| `update_snapshot(concept, hv)` | Update baseline. |
| `get_drift_report()` | Summary of all drift events. |
| `get_alarmed_concepts()` | Concepts exceeding drift threshold. |
| `get_statistics()` | Detector statistics. |

### `homeostasis.py`

**MemoryHomeostasis** — Balance memory systems.

| Method | Description |
|--------|-------------|
| `regulate(memory)` | Run homeostatic regulation cycle. |
| `prune_unused_rules()` | Remove stale rules. |

### `staged_recall.py`

**StagedRecall** — Multi-stage memory retrieval.

| Method | Description |
|--------|-------------|
| `update_cache(key, value)` | Update recall cache. |
| `query(query_hv, top_k, threshold)` | Multi-stage recall. |

### `semantic_memory_shim.py` *(V14)*

Auto-selects Rust or Python backend for spreading activation, using the same pattern as `hypervec_shim.py`.

| Function | Description |
|----------|-------------|
| `spread_activation_fast(concept_graph, start_concepts, relation_weights, stigmergy, steps, decay)` | Attempt Rust-accelerated spreading activation. Returns `Dict[str, float]` on success, `None` if Rust unavailable or sync fails. |

The caller (`SemanticMemory.spread_activation`) falls through to the Python path when `None` is returned. When `hypervec_rs.SemanticMemoryConcurrent` is importable, the shim syncs concept nodes from the NetworkX graph to the Rust DashMap backend before running activation.

---

## Reasoning

`python/core/reasoning/`

### `cognitive_engine.py`

**CognitiveEngine** — Central decision-making hub. Orchestrates GWT, rule learning, causal reasoning, and metacognition.

| Method | Description |
|--------|-------------|
| `register_task(task_tag)` | Set up a task context. |
| `perceive_and_decide(percept, available_actions)` | Full perception → decision pipeline. |
| `decide(situation_hv, predicates, available_actions)` | Choose an action given state. |
| `decide_multimodal(inputs, available_actions)` | Multi-modal decision variant. |
| `record_outcome(action, reward, task_tag)` | Feed reward signal. |
| `learn(task_tag)` | Trigger offline learning. |
| `transfer(source_tag, target_tag)` | Cross-domain transfer. |
| `process_dialogue(utterance)` | Language understanding + response. |
| `explain(query_type)` | Generate explanation (action, state, etc.). |
| `why_not(action)` | Contrastive explanation. |
| `counterfactual(altered_state)` | Counterfactual reasoning. |
| `enable_neural_rule_scoring()` | Enable perceptron-based rule ranking. |
| `get_allowed_actions(task_tag)` | Retrieve registered actions. |
| `get_belief_summary()` | Current belief state summary. |
| `set_mission_goal(goal)` | Set high-level goal. |
| `sleep(task_tag)` | Consolidation cycle. |
| `get_stats()` | Engine statistics. |
| `get_workspace_telemetry()` | GWT telemetry. |
| `export_traces()` | Export decision traces. |
| `register_broadcaster(fn)` | Register GWT broadcast listener. |
| `get_causal_telemetry()` | Causal graph telemetry. |

**CognitiveState** (dataclass):
`task_tag`, `situation_hv`, `active_predicates`, `chosen_action`, `confidence`, `self_confidence`, `exploration_mode`, `explanation`, `trace`, `construction_match`, `frame_fill`, `coreference_chain`, `belief_revision`, `system_used`, `system_1_confidence`, `homeostasis_actions`, `kle_uncertainty`, `uncertainty_bounds`.

### `global_workspace.py`

**GlobalWorkspace** — GWT coalition competition and broadcast.

| Method | Description |
|--------|-------------|
| `register_module(module)` | Register a `WorkspaceModule`. |
| `compete(proposals)` | Select winning coalition. |
| `broadcast(content)` | Broadcast to all modules. |
| `compete_with_rehearsal(proposals)` | Competition with episodic rehearsal. |
| `register_danger(label)` | Register a danger signal (veto). |
| `get_status()` | Workspace status. |
| `get_kle_uncertainty()` | KL-divergence uncertainty estimate. |
| `get_recent_vetoes()` | Recent safety vetoes. |

**WorkspaceModule** (ABC) — `receive_broadcast()`.
**Coalition** (dataclass) — `activation` property.

### `causal_reasoning.py`

**CausalDiscovery** — Observational causal induction.

| Method | Description |
|--------|-------------|
| `observe(state, action, outcome)` | Record an observation. |
| `detect_confounders()` | Identify potential confounders. |
| `induce_graph()` | Build causal graph from observations. |
| `incremental_update()` | Update graph incrementally. |
| `get_hypotheses()` | Return current hypotheses. |
| `update_strengths()` | Recompute link strengths. |

**CausalGraph** — Directed causal graph.

| Method | Description |
|--------|-------------|
| `add_link(src, tgt, strength)` | Add a causal link. |
| `add_causes(src, tgt)` | Add a positive causal edge. |
| `add_prevents(src, tgt)` | Add a preventive edge. |
| `forward_chain(start)` | Forward causal chain. |
| `backward_chain(end)` | Backward causal chain. |
| `find_path(src, tgt)` | Find causal path. |
| `get_immediate_effects(node)` | Direct effects. |
| `get_immediate_causes(node)` | Direct causes. |
| `prune_redundant()` | Remove redundant links. |

**CausalReasoner** — `predict_effects()`, `simulate_counterfactual()`, `explain_why()`, `predict_outcome_from_graph()`, `counterfactual()`, `why_not()`.

**CausalSchema** — Abstract causal template.

**TheoryModule** — `register_abstractions()`, `abstract_term()`, `form_theories()`, `predict_from_theory()`.

**EnhancedCausalDiscovery** — `generate_hypotheses()`, `design_experiment()`, `update_hypothesis()`, `suggest_next_experiment()`, `get_hypothesis_statistics()`.

### `causal_rule_auditor.py`

**CausalRuleAuditor** (V13) — Audit causal rules for consistency.

### `causal_interface.py`

**CausalInterface** — Abstract interface for causal reasoning.

### `causal_service_impl.py`

**CausalServiceImpl** — Service-layer implementation of `CausalInterface`.

### `rule_learner.py`

**RuleLearner** (`WorkspaceModule`) — Induce symbolic rules from observations.

| Method | Description |
|--------|-------------|
| `observe(state, action, outcome)` | Record an observation. |
| `induce_rules()` | Generate rules from data. |
| `validate_rules()` | Prune low-confidence rules. |
| `prune_rules()` | Remove stale rules. |
| `get_rules()` | Return all rules. |
| `get_applicable_rules(state)` | Rules matching current state. |
| `propose()` | GWT proposal. |
| `update(broadcast)` | GWT broadcast handler. |

**Rule** (dataclass): `confidence_history`, `last_fired`, `fire_count`.

### `attention_gwt_bridge.py`

**AttentionGWTBridge** (V10) — Multi-head attention for GWT coalition reranking.

### `planner.py`

**STRIPSPlanner** — STRIPS-style goal-directed planning.

| Method | Description |
|--------|-------------|
| `set_operators(operators)` | Define planning operators. |
| `learn_operators_from_graph(graph)` | Induce operators from causal graph. |
| `plan(initial, goal)` | BFS forward search. |
| `decompose_goal(goal)` | Hierarchical goal decomposition. |
| `plan_hierarchical(initial, goal)` | HTN-style planning. |
| `simulate_sequence(initial, actions)` | Simulate an action sequence. |

### `analogy.py`

**AnalogyEngine** — Structural alignment and cross-domain transfer.

| Method | Description |
|--------|-------------|
| `register_domain(name, concepts, relations)` | Register a source/target domain. |
| `register_abstract(name, schema)` | Register an abstract schema. |
| `lift_to_abstract(domain)` | Lift domain to abstract level. |
| `ground_to_domain(abstract, target)` | Ground abstract to target domain. |
| `find_analogy(source, target)` | Find structural alignment. |
| `transfer_rule(rule, source, target)` | Transfer a rule across domains. |
| `adapt_state(state, source, target)` | Adapt a state representation. |
| `zero_shot_action(source, target)` | Zero-shot action transfer. |
| `blend(domain_a, domain_b)` | Conceptual blending. |

### `math_reasoning.py`

**MathReasoner** — GWT coalition module for mathematical queries.

### `spatial_reasoning.py`

**SpatialReasoner** — Spatial relation reasoning.

### `belief_revision.py`

**BeliefScorer** — `free_energy()`, `should_revise()`.

**BeliefRevisionEngine** — AGM-style belief revision.

### `context_engine.py`

**ContextEngine** — Context tracking and disambiguation.

| Method | Description |
|--------|-------------|
| `register_meaning(word, context, hv)` | Register context-dependent meaning. |
| `disambiguate(word, context_hv)` | Context-aware word disambiguation. |
| `learn_from_feedback(word, correct_meaning)` | Update from feedback. |
| `infer_context_domain(state)` | Infer current domain. |
| `get_statistics()` | Engine statistics. |

---

## Learning

`python/core/learning/`

### `hebbian.py`

**HebbianMatrixNumPy** — `forward()`, `hebbian_update()`, `get_weights()`, `set_weights()`, `get_stats()`.

**VSAHebbianLearner** — Hebbian learning on HV associations.

| Method | Description |
|--------|-------------|
| `update_associations(hv_a, hv_b)` | Strengthen association. |
| `spread_activation(hv, steps)` | Propagate activation. |
| `get_associated_concepts(hv)` | Retrieve associated concepts. |
| `get_stats()` | Learner statistics. |

### `curiosity.py`

**CuriosityModule** — Surprise-driven exploration (free energy).

| Method | Description |
|--------|-------------|
| `compute_novelty(hv)` | Novelty score via prototype distance. |
| `compute_learning_progress()` | Learning progress estimate. |
| `compute_free_energy_surprise(hv)` | Free energy surprise signal. |
| `update_prototype(hv)` | Update novelty prototype. |
| `record_outcome(outcome)` | Record exploration outcome. |
| `record_visit(state)` | Increment visit count. |
| `get_visit_count(state)` | Return visit count. |
| `set_subgoals(goals)` | Set intrinsic subgoals. |
| `should_explore(confidence)` | Explore/exploit decision. |
| `get_exploration_action(actions)` | Select exploratory action. |
| `get_statistics()` | Exploration statistics. |

**ExplorationDecision** (dataclass).

### `active_inference.py`

**ActiveInferenceLearner** — Free energy minimisation.

| Method | Description |
|--------|-------------|
| `prediction_error(predicted, observed)` | Compute prediction error. |
| `epistemic_value(action)` | Expected information gain. |
| `free_energy(state)` | Variational free energy. |
| `should_veto(action)` | Safety veto via free energy. |
| `update_world_model(observation)` | Update generative model. |

### `rule_neural_scorer.py`

**RuleFeaturizer** — `featurize(rule)`.

**RuleNeuralScorer** (V10) — Perceptron-based rule ranking.

| Method | Description |
|--------|-------------|
| `score(rule)` | Score a rule. |
| `update(rule, target)` | Update scorer weights. |
| `batch_score(rules)` | Score multiple rules. |
| `rank_rules(rules)` | Rank rules by score. |
| `save(path)` / `load(path)` | Persist / restore scorer. |

### `conformal_wrapper.py`

**ConformalWrapper** (V13) — Calibrated uncertainty bounds via conformal prediction.

| Method | Description |
|--------|-------------|
| `calibrate(scores, labels)` | Calibrate on held-out data. |
| `predict_set(score)` | Return prediction set at coverage level. |
| `uncertainty_bound(score)` | Return uncertainty interval. |
| `is_calibrated()` | Check calibration status. |
| `get_statistics()` | Wrapper statistics. |

### `pattern_generalizer.py`

**PatternGeneralizer** (V13) — Generalise patterns from examples.

| Method | Description |
|--------|-------------|
| `observe(pattern)` | Record a new pattern instance. |
| `get_mature_patterns()` | Return patterns with sufficient evidence. |
| `match(input)` | Match input against generalisations. |
| `get_statistics()` | Generaliser statistics. |

**CrossDomainTransferPipeline** — `register()`, `transfer()`, `get_transfer_log()`.

### `perception_distiller.py` *(V14)*

**PerceptionDistiller** — Tracks convergence between bridge and internal HV encoding quality per modality. Enables a principled criterion for retiring optional neural bridge models.

| Method | Description |
|--------|-------------|
| `observe(modality, bridge_hv, internal_hv)` | Record one quality observation; returns current similarity score. |
| `is_graduated(modality)` | Return `True` when rolling avg ≥ `threshold` over last 100 observations. |
| `get_report()` | Return per-modality dict with `observations`, `current_quality`, `avg`, `graduated`. |

Constructor: `PerceptionDistiller(threshold=0.80)`. Threshold controlled by `NSCKConfig.distillation_threshold`.

### `cross_domain.py`

**TransferEngine** — Cross-domain knowledge transfer.

| Method | Description |
|--------|-------------|
| `register_concept(domain, concept)` | Register a domain concept. |
| `register_rule(domain, rule)` | Register a domain rule. |
| `register_correspondence(src, tgt, mapping)` | Register cross-domain mapping. |
| `transfer(source, target)` | Execute transfer. |
| `discover_correspondences(src, tgt)` | Auto-discover mappings. |
| `apply(transferred, target_state)` | Apply transferred knowledge. |
| `domain_summary(domain)` | Summarise a domain. |
| `list_domains()` | List registered domains. |

Supporting classes: `DomainConcept`, `DomainRelation`, `ConceptCorrespondence`, `TransferredInference`, `SchemaExtractor`, `StructureMapper`, `RuleLifter`.

### `cross_modal.py`

**CrossModalCorrelationLearner** — Learn associations across modalities.

| Method | Description |
|--------|-------------|
| `observe(modality_a, modality_b)` | Record co-occurrence. |
| `predict_modality(source_hv, target_modality)` | Predict target from source. |
| `get_correlation_strength(a, b)` | Correlation estimate. |
| `get_statistics()` | Learner statistics. |

### `continual_learning.py`

**ContinualLearner** — Lifelong learning without catastrophic forgetting.

| Method | Description |
|--------|-------------|
| `register_task(task_tag)` | Register a new task. |
| `compute_importance(params)` | Fisher information for EWC. |
| `ewc_loss(params)` | Elastic weight consolidation loss. |
| `bind_to_task(hv, task_tag)` | Context-dependent binding. |
| `consolidate_task(task_tag)` | Consolidate after training. |
| `retrieve_task(task_tag)` | Retrieve task-specific knowledge. |
| `measure_forgetting()` | Measure backward transfer. |
| `get_statistics()` | Learner statistics. |

### `meta_learning.py`

**MetaLearner** — Learning to learn.

| Method | Description |
|--------|-------------|
| `add_task(task)` | Register a `MetaTask`. |
| `adapt(task)` | Fast adaptation to new task. |
| `meta_update()` | Meta-level parameter update. |
| `select_strategy(task)` | Choose best learning strategy. |
| `update_strategy_performance(strategy, result)` | Update strategy stats. |
| `learn_binding_pattern(pattern)` | Learn HV binding pattern. |
| `get_statistics()` | Meta-learner statistics. |

---

## Language

`python/core/language/`

### `parser.py`

**LeftCornerParser** — Parse sentences into phrase trees: `reset()`, `get_vector()`.

### `language_module.py`

**LanguageModule** — Main language interface (LLM or mock): `understand()`, `generate()`.

### `vsa_language_module.py`

**VSALanguageModule** — VSA-based neuro-symbolic language processing.

### `ngram_nlu.py`

**NgramNLU** — Fast n-gram intent classifier (143K sent/s).

| Method | Description |
|--------|-------------|
| `train(examples, labels)` | Train on labelled data. |
| `predict(text)` | Return top intent. |
| `predict_distribution(text)` | Return intent distribution. |
| `extract_intent(text)` | Structured intent extraction. |
| `extract_entities(text)` | Named entity extraction. |

**NgramNLUAdapter** — GWT adapter for NgramNLU.

### `fluent_nlg.py`

**NSCKResponseEngine** — Natural language generation for cognitive states.

### `nlg.py`

**StructuralRealizer**, **NLGEngine**, **DiscoursePlanner** — NLG pipeline components.

### `dialogue_manager.py`

**DialogueManager** — Dialogue state tracking, history HV, topic shift detection: `reset()`, `process_turn()`, `resolve_anaphora()`.

### `construction_grammar.py`

**Construction**, **ConstructionMatch**, **ConstructionMatcher** — Pattern-based grammar matching.

### `frame_semantics.py`

**Frame**, **FrameLibrary** — FrameNet-style semantic frames.

### `coreference.py`

**EntityMention**, **EntityRegister** — Pronoun and entity coreference resolution.

### `semantic_roles.py`

**SRLFrame**, **SemanticRoleLabeler** — Agent/patient/theme semantic role labelling.

### `pragmatics.py`

**PragmaticsEngine** — Speech acts, implicature, pragmatic inference.

### `distributional_semantics.py`

**DistributionalCodebook** — Co-occurrence statistics, pre-trained on built-in corpus.

### `text_knowledge_learner.py`

**TextKnowledgeLearner**, **LearningSession** — Extract knowledge from text.

### `pos_tagger.py`

**BrillPosTagger** — Part-of-speech tagging.

### `universal_input.py`

**UniversalInput** — Normalise diverse input formats.

### `lingua_cortex.py`

**SemanticFingerprint** — `union()`, `intersection()`, `overlap()`, `jaccard()`.
**SemanticMap**, **SemanticFoldingTrainer** — High-level language cortex with semantic folding.

### `control.py`

**CognitiveController** — Language control module.

### `hf_corpus_loader.py`

HuggingFace corpus loader with offline fallback.

### `compositional_semantics_backup.py`

Legacy backup of compositional semantics module.

---

## Cognitive

`python/core/cognitive/`

### `metacognition.py`

**MetacognitiveEngine** — Monitor reasoning confidence, trigger System 2.

| Method | Description |
|--------|-------------|
| `compute_confidence(state)` | Estimate decision confidence. |
| `detect_conflict(proposals)` | Detect conflicting proposals. |
| `infer_from_facts(facts)` | Logical inference. |
| `infer(query)` | General inference. |
| `check_cycle(graph)` | Detect reasoning cycles. |

### `self_model.py`

**SelfModel** — Agent self-representation (identity HV).

| Method | Description |
|--------|-------------|
| `update_confidence(outcome)` | Update calibration from outcome. |
| `get_confidence()` | Current confidence estimate. |
| `predict_success(action)` | Predict action success probability. |
| `update(state)` | Update self-model from state. |
| `get_identity()` | Return identity HV. |
| `get_capability()` | Return capability profile. |
| `get_calibration_error()` | ECE metric. |
| `get_stats()` | Self-model statistics. |
| `get_improvement_trend()` | Performance trend. |

### `theory_of_mind.py`

**TheoryOfMind** — Model other agents' beliefs.

| Method | Description |
|--------|-------------|
| `get_or_create_model(agent_id)` | Get/create agent model. |
| `update_agent_perspective(agent_id, observation)` | Update agent's perspective. |
| `predict_action(agent_id)` | Predict what agent will do. |
| `detect_false_belief(agent_id)` | Detect false belief (Sally-Anne). |
| `recursive_belief(agent_id, depth)` | Recursive belief modelling. |
| `model_other_agent(agent_id, state)` | Full agent modelling. |
| `perspective_take(agent_id)` | Adopt agent's perspective. |

### `emotion_system.py`

**EmotionSystem** — Affect-based drive modulation.

| Method | Description |
|--------|-------------|
| `reset()` | Reset emotional state. |
| `update_from_drives(drives)` | Update emotions from drives. |
| `get_emotion_blend()` | Current emotion mixture. |
| `get_mood()` | Aggregate mood. |
| `get_emotional_trajectory()` | Emotion history. |
| `recognize_emotion_from_text(text)` | Text emotion recognition. |
| `get_emotion_info()` | Emotion system info. |

### `safety_verifier.py`

**SafetyRuleVerifier** — Formula-based safety checks: `add_property()`, `verify_rule()`, `verify_ruleset()`.

**SafetyGateVerifier** (V10) — GWT safety gate: `gate_decision()`.

---

## Integration

`python/core/integration/`

### `config.py`

**NSCKConfig** — All hyperparameters and feature flags.

| Factory | Description |
|---------|-------------|
| `minimal()` | Minimal configuration for testing. |
| `research()` | Research configuration with all features. |
| `production()` | Production configuration. |
| `from_env()` | Load from environment variables. |
| `rich()` *(V14)* | `research()` + `perception_mode="bridge"`. Use when optional deep-learning deps are available. |
| `for_scale(n_concepts)` *(V14)* | Auto-tunes `memory_capacity` and `enable_hnsw_index` for the expected concept count. |

**V14 config fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `perception_mode` | `str` | `"pure"` | `"pure"` / `"bridge"` / `"hybrid"` — adapter selection |
| `text_bridge_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformer model name |
| `image_bridge_model` | `str` | `"mobilenet_v3_small"` | timm model name |
| `audio_bridge_model` | `str` | `"whisper-tiny"` | Whisper model name |
| `bridge_cache_embeddings` | `bool` | `True` | Cache bridge embeddings in memory |
| `bridge_dim` | `int` | `384` | Input dimension for `EmbeddingVSABridge` |
| `distillation_threshold` | `float` | `0.80` | `PerceptionDistiller` graduation threshold |
| `knowledge_packs` | `List[str]` | `[]` | Paths to `.gz` packs loaded on substrate init |

### `brain_fusion.py`

**BrainFusion** — `register_brain()`, `align_concepts()`, `fuse()`.

**FusedBrain** — `resolve_rules()`, `query()`, `forward_chain_multi()`.

**ConcurrentMultimodalScheduler** — Multi-modal orchestration.

### `explanation.py`

**ExplanationGenerator** — Natural language explanations.

| Method | Description |
|--------|-------------|
| `explain_action(action, state)` | Explain why an action was chosen. |
| `explain_rejection(action, state)` | Explain why an action was rejected. |
| `explain_state(state)` | Describe current state. |
| `explain_goal(goal)` | Explain goal pursuit. |
| `explain_contrastive(action_a, action_b)` | Contrastive explanation. |

**Explanation** (dataclass).

### `persistence.py`

**BrainStore** — Save/load system state.

| Method | Description |
|--------|-------------|
| `save_rule(rule)` | Persist a rule. |
| `load_rules()` | Load all rules. |
| `save_concept(label, hv)` | Persist a concept. |
| `load_concept(label)` | Load a concept. |
| `record_episode(episode)` | Persist an episode. |
| `load_recent_episodes(n)` | Load recent episodes. |
| `create_checkpoint()` | Full system checkpoint. |
| `export_brain(path)` | Export entire brain state. |
| `import_brain(path)` | Import brain state. |

### `knowledge_integration.py`

**KnowledgeIntegrator** — Merge knowledge sources.

| Method | Description |
|--------|-------------|
| `process_input(input)` | Process and integrate input. |
| `query_knowledge(query)` | Query integrated knowledge. |
| `query_relation(src, rel)` | Query by relation. |
| `correct_knowledge(label, correction)` | Correct stored knowledge. |
| `learn_from_feedback(feedback)` | Update from feedback. |
| `apply_knowledge_to_new_domain(source, target)` | Cross-domain application. |
| `get_statistics()` | Integration statistics. |

### `knowledge_pack.py` *(V14)*

**KnowledgePack** — Portable, serialisable bundle of domain concepts, relations, and causal links. Stored as gzip-compressed pickle.

| Method | Description |
|--------|-------------|
| `add_concept(name, properties, hv=None)` | Add a concept with optional pre-computed HV. |
| `add_relation(src, rel, dst)` | Add a directed relation triple. |
| `add_causal_link(cause, effect, strength=1.0)` | Add a causal association with strength ∈ [0, 1]. |
| `save(path)` | Serialise to `path` (gzip pickle). |
| `load(path)` | Class method — deserialise from `path`. |
| `inject_into(engine)` | Inject into a `CognitiveEngine`'s `SemanticMemory`. Returns `{"concepts": N, "relations": N, "causal_links": N}`. |

Constructor: `KnowledgePack(name="unnamed")`. Packs are loaded on substrate init when `NSCKConfig.knowledge_packs` contains their paths.

---

## Adapters

`python/core/adapters/` — All adapters produce `PerceptPacket` via `encode()`.

| File | Class | Notes |
|------|-------|-------|
| `text_adapter.py` | **TextAdapter** | Text → HV encoding. |
| `dict_state_adapter.py` | **DictStateAdapter** | Dict/JSON state → HV. |
| `numeric_adapter.py` | **NumericAdapter** | Scalar → HV. |
| `numeric_sequence_adapter.py` | **NumericSequenceAdapter** | Numeric sequence → HV. |
| `snn_adapter.py` | **SNNAdapter** | SNN spike output → HV. |
| `image_adapter.py` | **ImageAdapter** | Spatial grid + colour histogram + Sobel FPE (65 dims). |
| `audio_adapter.py` | **AudioAdapter** | MFCC + spectral FPE (23 dims). |
| `video_adapter.py` | **VideoAdapter** (V13) | Temporal stream encoding. Also: `TemporalStreamEncoder`. |
| `multimodal_fuser.py` | **MultimodalFuser** | Fuse multiple `PerceptPacket`s: `fuse()`. |
| `stream_processor.py` | **StreamProcessor** | `ingest()`, `ready()`, `emit()`. **StreamVerifier**: `get_active_predicates()`. |

**V14 Rich Adapters** — Gated behind `NSCKConfig.perception_mode`. All fall back silently to the pure VSA path when optional dependencies are absent.

| File | Class | Bridge (optional dep) | Fallback |
|------|-------|-----------------------|----------|
| `rich_text_adapter.py` | **RichTextAdapter** *(V14)* | `sentence-transformers` (`all-MiniLM-L6-v2`) | Char-ngram `EmbeddingVSABridge` |
| `rich_image_adapter.py` | **RichImageAdapter** *(V14)* | `timm` + `torch` (`mobilenet_v3_small`, `pretrained=False`) | Existing `ImageAdapter` |
| `rich_audio_adapter.py` | **RichAudioAdapter** *(V14)* | `whisper` (`whisper-tiny`) | Existing `AudioAdapter` |

All three accept `config: NSCKConfig` in their constructor and produce standard `PerceptPacket` output. The `adapter_trace` field includes `encoding_method` (`"sentence_transformer"` / `"char_ngram"` / `"hash"` / `"timm"` / `"classical_cv"` / `"whisper"` / `"classical_dsp"`) and `latency_ms`.

---

## Types

`python/core/types/`

### `percept_packet.py`

**PerceptPacket** (dataclass) — Universal perception output.

Fields: `modality`, `timestamp`, `situation_hv`, `entity_hvs`, `relation_hvs`, `active_predicates`, `confidence`, `raw_state`, `adapter_name`, `adapter_trace`.

Factory: `make()`.

### `modality_adapter.py`

**ModalityAdapter** (ABC) — Abstract base for all adapters. Single abstract method: `encode()`.

---

## API

`nsck/api/`

### `nsck_api.py`

**NSCKApiServer** — FastAPI REST endpoints (falls back to stdlib `http.server`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | Run decision cycle. |
| `/learn` | POST | Trigger learning. |
| `/sleep` | POST | Trigger consolidation. |
| `/status` | GET | System status. |

Methods: `handle_decide()`, `handle_learn()`, `handle_sleep()`, `handle_status()`, `create_app()`, `run()`.

---

## Multimodal

`python/core/multimodal/`

### `multimodal_processor.py`

**MultimodalProcessor** — `process()`, `supported_modalities()`. Internal: `_process_text()`, `_process_image()`, `_process_audio()`, `_process_video()`.

**ConcurrentMultimodalProcessor** — `process()` (thread-pool variant).

**ConcurrentMultimodalScheduler** — `process_concurrent()`, `close()`.

Data classes: `ModalityResult`, `MultimodalInput`, `ProcessedInput`.

### `image_generator.py`

**ImageGenerator** — VSA-driven image generation.

| Method | Description |
|--------|-------------|
| `train_from_examples(examples)` | Learn visual concepts. |
| `generate(concept_hv)` | Generate from concept. |
| `get_statistics()` | Generator statistics. |

**ConceptVisualMemory** — `learn_association()`, `retrieve_visual()`, `retrieve_similar_visual()`.

**VisualFeatures** — `to_array()`, `from_array()`, `random()`.

---

## Training

`python/core/training/`

### `vsa_trainer.py`

**VSATrainer** — `train()`, `save_checkpoint()`.

### `snn_training.py`

**SNNTrainer** — Full SNN training pipeline.

| Method | Description |
|--------|-------------|
| `train_epoch_unsupervised(data)` | Unsupervised STDP epoch. |
| `train_epoch_supervised(data, labels)` | Supervised epoch. |
| `train_epoch_reinforcement(env)` | RL epoch. |
| `evaluate(data, labels)` | Evaluation. |
| `train(config)` | Full training loop. |
| `save_checkpoint(path)` / `load_checkpoint(path)` | Persist / restore. |

**SNNDataset** — `batch_iterator()`.

Data classes: `TrainingConfig`, `TrainingMetrics`.

### `snn_benchmarks.py`

**BenchmarkSuite** — `run_all()`, `benchmark_pattern_classification()`, `benchmark_temporal_sequences()`, `print_summary()`.

**BenchmarkResult** (dataclass).

---

## V17 Enrichment Modules

### `python/core/reasoning/causal_enricher.py`

**CausalEnricher** — enrich causal triples with semantic context.

| Method/Property | Description |
|-----------------|-------------|
| `enrich(cause, effect, strength)` | Returns `CausalTrace` with context concepts. |
| `enrich_chain(chain, base_strength)` | Pairwise enrichment with 0.9^i decay. |
| `enrichment_count` | Total enrichments performed. |

**CausalTrace** (dataclass): `cause`, `effect`, `strength`, `context_concepts`, `enrichment_steps`.

---

### `python/core/perception/perceptual_enricher.py`

**PerceptualEnricher** — temporal context + confidence for PerceptPackets.

| Method/Property | Description |
|-----------------|-------------|
| `enrich(packet)` | Returns `EnrichedPercept`. |
| `reset()` | Clear temporal history window. |
| `enrich_count` | Total enrichments performed. |

**EnrichedPercept** (dataclass): `original_modality`, `confidence`, `temporal_ctx_available`, `tags`, `metadata`.

---

### `python/core/memory/semantic_enricher.py`

**SemanticEnricher** — inverse-relation inference and co-query tracking.

| Method/Property | Description |
|-----------------|-------------|
| `enrich_concept(concept, relation, target)` | Single triple enrichment. |
| `enrich_bulk(triples)` | Batch enrichment. |
| `coquery_stats()` | Returns `{(concept, target): count}` dict. |
| `total_enrichments` | Total enrichment calls. |

**EnrichmentReport** (dataclass): `concepts_enriched`, `inverse_relations_added`, `bundles_strengthened`, `steps`.

**INVERSE_RELATION_MAP**: `is_a→sub_class_of`, `has_part→part_of`, `causes→caused_by`, `used_for→uses`, `at_location→location_of`.

---

### `python/core/cognitive/glass_box_tracer.py`

**GlassBoxTracer** — step-by-step decision trace.

| Method/Property | Description |
|-----------------|-------------|
| `begin_decision(decision_id)` | Start a new trace. |
| `end_decision()` | Finalise and archive trace; returns `DecisionTrace`. |
| `record(module, message, confidence, **metadata)` | Append a `TraceEntry`. |
| `span(name)` | Context manager for grouping entries. |
| `export()` | Return active (in-progress) trace. |
| `last_trace()` | Most recent completed trace. |
| `history()` | All archived traces. |
| `format_trace(trace)` | Static; human-readable string. |

**DecisionTrace** (dataclass): `decision_id`, `entries`, `start_ms`, `end_ms`, `elapsed_ms`.

**TraceEntry** (dataclass): `span`, `module`, `message`, `confidence`, `timestamp_ms`, `metadata`.

---

### `python/core/memory/crossmodal_enricher.py`

**CrossModalEnricher** — anchor-based cross-modal linking.

| Method/Property | Description |
|-----------------|-------------|
| `link_modalities(anchor, pairs)` | Register modality concepts under anchor. |
| `detect_clusters()` | Find anchors in 2+ modalities. |
| `summary()` | Dict with `total_links`, `anchor_count`, `anchors`. |
| `total_links` | Total modality links registered. |
| `anchor_count` | Number of distinct anchors. |

**CrossModalEnrichmentReport** (dataclass): `anchors_created`, `links_added`, `clusters_detected`, `steps`.

---

## V4 New and Updated Modules

### VSANLUEngine (`python/core/language/vsa_nlu.py`)

**Purpose**: VSA-based intent classifier and entity extractor. Primary NLU engine as of V4.

**Methods**:
- `classify(text: str) -> Tuple[str, float]`: Returns (intent, confidence) using HV similarity to prototypes
- `extract_entities(text: str) -> List[Tuple[str, str]]`: Returns (entity, type) via capitalization + concept lookup
- `encode_sentence(text: str) -> HyperVector`: Encode whole sentence as HV for downstream use
- `process(text: str) -> Dict`: NgramNLU-compatible output dict

**Intent classes**: question, command, statement, greeting, farewell, exclamation, negation

**Dependencies**: DistributionalCodebook (auto-built from BUILTIN_CORPUS)

---

### KnowledgeSeeder (`python/core/bootstrap/knowledge_seeder.py`)

**Purpose**: Bootstrap a domain from a declarative YAML spec.

**Methods**:
- `seed_from_yaml(yaml_path: str, engine: CognitiveEngine) -> int`: Returns number of rules seeded

**YAML format**: See `python/core/bootstrap/domain_kits/navigation.yaml` for spec.

**Injection targets**:
1. `engine.semantic_memory.add_concept()` — semantic concepts
2. `engine.rule_learner.learned_rules[domain]` — causal rules
3. `engine.causal_graphs[domain].add_causes()` — causal graph edges
4. `engine.procedural_memory.cache_skill()` — high-confidence rules cached as skills

---

### ProceduralMemory (V4 Updates) (`python/core/memory/procedural_memory.py`)

**V4 changes**:
- Familiarity threshold: **0.72** (was 0.85)
- New `_lsh_buckets: Dict[int, List[int]]` — 16-bit LSH bucket index
- `_compute_lsh(bits, n_bits, seed)` — LSH key computation
- `recall_action()` — checks LSH bucket first (O(1)), falls back to full scan if empty
- Auto-populated by `CognitiveEngine.learn()` on positive rewards

---

### SemanticMemory (V4 Updates) (`python/core/memory/semantic_memory.py`)

**V4 changes**:
- `_hnsw_enabled: bool = True` (was False; now default-on)
- `_hot_cache: Dict[str, (HyperVector, float)]` — 256-entry LRU hot cache
- `_HOT_CACHE_SIZE: int = 256` — configurable via `config.semantic_hot_cache_size`
- `_update_hot_cache(activations)` — populates hot cache from top activated concepts
- `spread_activation()` — now calls `_update_hot_cache()` on completion
