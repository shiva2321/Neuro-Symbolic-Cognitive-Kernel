# NSCK V8 Changelog

## Overview
NSCK V8 adds 9 new capabilities to the Neural-Symbolic Cognitive Kernel, enhancing multimodal processing, memory management, dialogue understanding, reasoning, and multi-agent coordination.

## New Features

### 1. Config Flags (config.py)
- `enable_concurrent_multimodal`: Enables ConcurrentMultimodalScheduler
- `enable_full_rust_snn`: Enables full Rust SNN backend
- `memory_decay_lambda`: Exponential decay rate for concept importance
- `memory_prune_threshold`: Prune concepts below this importance score
- `enable_dialogue_state_tracking`: Multi-turn dialogue state HV tracking
- `enable_hierarchical_srl`: Hierarchical semantic role labeling
- `enable_multi_agent`: Multi-agent cognitive fusion
- `enable_active_inference`: Active inference free energy loop
- `active_inference_weight`: Weight for active inference salience adjustment

### 2. ConcurrentMultimodalScheduler (multimodal_processor.py)
- Wraps `MultimodalProcessor` with `ThreadPoolExecutor`
- Parallel per-modality processing
- Coherence window filtering (default 50ms)
- Attention-weighted fusion: `HV_fused = Σ_m(conf_m × bind(role_m, HV_m)) / Σ_m conf_m`
- `process_concurrent(inp)` and `close()` API

### 3. Memory Lifecycle (semantic_memory.py, homeostasis.py)
- `add_concept()` now tracks `access_count`, `last_accessed`, `importance_score`
- `get_concept()` updates access metadata
- `decay_concepts(lambda_decay)`: exponential decay of importance scores
- `prune_below(threshold)`: removes low-importance concepts
- `MemoryHomeostasis.regulate()` optionally calls decay/prune

### 4. Multi-Turn Dialogue State Tracking (dialogue_manager.py)
- `DialogueManager.__init__` accepts `config=None`
- Rolling dialogue history HyperVector via permuted bundling
- Topic shift detection (similarity < 0.3)
- `get_context_hv()` returns current history HV
- `clarification_request(term)` helper

### 5. MathReasoner in CognitiveEngine (cognitive_engine.py, universal_input.py)
- `UniversalInput.is_mathematical(text)` pattern matcher
- `CognitiveEngine._solve_math(text)` helper
- MATH coalition with 0.95 salience for math queries
- `get_belief_summary(topic_hv)` for multi-agent use

### 6. Hierarchical Resonator Networks (resonator.py)
- `HierarchicalResonatorNetwork`: 2-level factorization
- L1 (sentence): AGENT, VERB, PATIENT, THEME, INSTRUMENT
- L2 (clause): MODIFIER_AGENT, MODIFIER_VERB, MODIFIER_PATIENT
- `factorize_hierarchical(hv, depth)` and compatibility `factorize()` method

### 7. Multi-Agent Cognitive Fusion (brain_fusion.py, theory_of_mind.py)
- `MultiAgentSession`: manages multiple engines
- `exchange_snapshots()`: semantic memory HV summaries
- `negotiate_beliefs(topic_hv)`: consensus bundling
- `TheoryOfMind.model_other_agent()`: infer beliefs from actions
- `TheoryOfMind.perspective_take()`: agent perspective on topic

### 8. NSCK-Eval Benchmark Suite (benchmarks/)
- `babi_tasks.py`: 20 bAbI-style QA tasks
- `math_word_problems.py`: 50 arithmetic problems
- `cross_domain_transfer.py`: 5 transfer tasks
- `nlg_quality.py`: 10 NLG fluency checks
- `dialogue_coherence.py`: multi-turn coherence
- `runner.py`: BenchmarkRunner orchestrator

### 9. Active Inference Full Loop (active_inference.py)
- `ActiveInferenceLearner`: free energy minimization
- `prediction_error()`, `epistemic_value()`, `free_energy()`
- `should_veto()`: safety gate integration
- `update_world_model()`: predictive model updates
- Wired into `CognitiveEngine.decide()` (coalition salience)
- Wired into `CognitiveEngine.record_outcome()` (model updates)
- `SafetyGate.check_free_energy()` helper

## Rust Extensions
- `hypervec_rs`: Rust VSA operations
- `snn_rs`: Rust SNN backend

## Tests Added
- `tests/unit/multimodal/test_concurrent_multimodal.py` (20+ tests)
- `tests/unit/memory/test_memory_lifecycle.py` (20+ tests)
- `tests/unit/language/test_dialogue_state.py` (20+ tests)
- `tests/unit/reasoning/test_math_integration.py` (15+ tests)
- `tests/unit/vsa/test_hierarchical_resonator.py` (15+ tests)
- `tests/unit/integration/test_multi_agent.py` (15+ tests)
- `tests/unit/learning/test_active_inference_integration.py` (15+ tests)
- `tests/integration/test_benchmarks.py` (10+ tests)
