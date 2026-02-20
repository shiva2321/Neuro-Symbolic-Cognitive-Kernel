# Testing Reference

All tests in the NSCK codebase: what each file tests, how it tests it, and why. Tests are organised into four layers: unit, integration, architecture, and experiments.

---

## Table of Contents

- [How to Run Tests](#how-to-run-tests)
- [Test Overview](#test-overview)
- [NSCK Unit Tests](#nsck-unit-tests)
- [NSCK Integration Tests](#nsck-integration-tests)
- [NSCK Architecture Tests](#nsck-architecture-tests)
- [NSCK Experiments](#nsck-experiments)
- [NSCK Regression Tests](#nsck-regression-tests)
- [nsck_ai_model Tests](#nsck_ai_model-tests)
- [Rust VSA Property Tests](#rust-vsa-property-tests)
- [Python Core Tests](#python-core-tests)
- [Testing Philosophy](#testing-philosophy)

---

## How to Run Tests

```bash
# All NSCK tests
cd nsck
pytest tests/ python/core/tests/ -q

# Unit tests only (fast)
pytest tests/unit/ -q

# Integration tests
pytest tests/integration/ -q

# Architecture tests
pytest tests/core_architecture/ -q

# AI model tests
python -m pytest nsck_ai_model/tests/ -v

# Rust property tests
cd nsck/rust_vsa && cargo test --release
```

---

## Test Overview

| Layer | Location | Files | Test methods | Description |
|---|---|---|---|---|
| Unit | `nsck/tests/unit/` | 25 files | ~199 | Module-level isolation tests |
| Integration | `nsck/tests/integration/` | 10 files | ~290 | Cross-module and system tests |
| Architecture | `nsck/tests/core_architecture/` | 4 files | ~63 | Architecture capability verification |
| Experiments | `nsck/tests/experiments/` | 4 files | ~10 | Behavioural validation scenarios |
| Regression | `nsck/tests/regression/` | 1 file | 6 | Known-fixed bug coverage |
| AI model | `nsck_ai_model/tests/` | 3 files | 123 | Chat engine: unit + production |
| Rust | `nsck/rust_vsa/tests/` | 1 file | 11 | Property-based VSA correctness |
| Python core | `nsck/python/core/tests/` | 3 files | ~40 | Module-internal tests |

**Total: ~597 test methods** across the codebase.

---

## NSCK Unit Tests

Unit tests target individual classes in isolation, using minimal or mocked dependencies.

---

### `unit/vsa/test_hypervec_parity.py` — 6 tests

**What:** Verifies that the Rust `hypervec_rs` extension produces mathematically identical results to the Python `HyperVectorPy` reference implementation.

**How:**
1. Creates identical HVs in both backends using `from_bits()` (bypasses RNG differences).
2. Applies XOR, bundle, permute, similarity operations.
3. Compares results with `np.testing.assert_array_equal`.

**Why:** The Rust backend is a performance optimisation. Users and the rest of the system must be able to trust that switching backends doesn't change results. Any discrepancy would invalidate all speedup measurements.

**Notable test:** `test_seed_determinism` is marked `@pytest.mark.xfail` — it's known that the Python (`PCG64`) and Rust (`ChaCha8`) RNGs produce different sequences from the same seed. This is an accepted limitation documented in `vsa/test_hypervec_parity.py` (see the `xfail` marker comment: "RNG algorithms differ (PCG64 vs ChaCha8)"). All other parity tests use `from_bits()` to create identical inputs, bypassing the RNG difference entirely.

---

### `unit/memory/test_cleanup_memory.py` — 17 tests

**What:** Tests `CleanupMemory` — the LRU associative memory used to denoise VSA operations.

**How:**
- Registers clean HVs; verifies retrieval by exact match and by similarity threshold.
- Adds random bit flips to simulate VSA noise; verifies cleanup snaps back to nearest prototype.
- Tests LRU eviction when `max_size` is exceeded.
- Tests `force=True` overwrite behaviour.
- Tests the `bundle_with_cleanup` and `unbind_with_cleanup` utility functions.
- Tests `load_from_store` with a mock `BrainStore`.

**Why:** CleanupMemory is the error-correction mechanism for all VSA operations. Without it, noise accumulates exponentially with operation depth. These tests ensure that (1) exact matches are always recovered, (2) the threshold parameter works correctly, (3) the LRU policy is correct, and (4) the store integration works.

**Key test:** `test_bind_unbind_with_noise` — flips 50 bits (0.5% noise), then verifies that unbind + cleanup recovers the original with ≥99% similarity. This proves the denoising chain works end-to-end.

---

### `unit/reasoning/test_causal_discovery.py` — 2 tests

**What:** Tests `CausalDiscovery.induce_graph()` — ΔP-based causal learning.

**How:**
- `test_strong_causality`: feeds 10 SWITCH_ON→LIGHT_ON and 10 SWITCH_OFF→LIGHT_OFF observations; expects SWITCH_ON→LIGHT_ON to be discovered with ΔP=1.0.
- `test_spurious_correlation`: feeds both CLAP and WAIT followed by BIRD_CHIRPS; expects ΔP(CLAP→BIRD_CHIRPS) ≈ 0 (rejected as spurious).

**Why:** Distinguishing true causation from correlation is the core function of `CausalDiscovery`. These minimal tests prove the ΔP formula is correctly implemented and that the Laplace-smoothed probability estimates behave as expected.

---

### `unit/reasoning/test_chaining.py` — 2 tests

**What:** Tests `CausalGraph.forward_chain()` and `backward_chain()`.

**How:** Builds a 3-node chain (A→B→C) with known strengths; verifies that:
- Forward chain from A finds B and C.
- Chain strength decreases along the path (product of link strengths).
- Backward chain from C traces back to A.

**Why:** Causal chains are the primary output of causal reasoning. These tests verify that the BFS traversal is correct and that strength propagation follows the product rule.

---

### `unit/reasoning/test_counterfactuals.py` — 2 tests

**What:** Tests `CausalGraph.counterfactual()` and `CounterfactualReasoner`.

**How:** Sets up a causal graph; applies a counterfactual intervention (do_X = remove a cause); verifies the predicted outcome changes.

**Why:** Counterfactual reasoning is one of the most cognitively demanding capabilities. These tests ensure the do-operator simulation produces different outcomes when a cause is removed.

---

### `unit/reasoning/test_planning.py` — 1 test

**What:** Tests `STRIPSPlanner.plan()`.

**How:** Registers operators for a simple 2-room navigation problem; calls `plan(start, goal)`; verifies the returned plan reaches the goal state.

**Why:** A* planning is used to create multi-step action sequences. The test verifies that the planner correctly identifies applicable operators and produces a valid state sequence.

---

### `unit/reasoning/test_plan_execution.py` — 1 test

**What:** Tests plan execution — applying a plan step-by-step through a simulated state transition function.

**How:** Executes each step of a generated plan; checks that the state transitions correctly after each step.

**Why:** A plan is only useful if it can be executed correctly. This test connects planning to action execution.

---

### `unit/reasoning/test_rule_learner_interface.py` — 1 test

**What:** Tests that `RuleLearner` implements the `WorkspaceModule` interface correctly.

**How:** Calls `receive_broadcast()` and verifies no exceptions are raised; checks that `build_coalition()` returns a valid `Coalition` object when conditions are met.

**Why:** `RuleLearner` must integrate with the Global Workspace as a `WorkspaceModule`. This test ensures the interface contract is met.

---

### `unit/reasoning/test_logic_bridge.py` — 4 tests

**What:** Tests the VSA ↔ symbolic logic bridge: converting predicate truth values to HV operations and back.

**How:** Registers predicates; encodes a state as an HV; decodes predicates from the HV; checks round-trip fidelity.

**Why:** The VSA-logic bridge is the mechanism by which symbolic predicates are represented as HVs. Errors here would corrupt all predicate-based reasoning.

---

### `unit/reasoning/test_theory_formation.py` — 3 tests

**What:** Tests `RuleLearner` rule induction from observation sequences.

**How:** Feeds `(state, action, outcome)` tuples with a specific pattern; verifies that the induced rule correctly captures the pattern with the expected confidence.

**Why:** Rule induction is one of the core learning mechanisms. These tests verify that the frequency counting and confidence threshold are working correctly.

---

### `unit/cognitive/test_context_engine.py` — 9 tests

**What:** Tests `ContextEngine` — the sliding window of recent predicates and HVs.

**How:** Adds states to the context window; queries for recent predicates; verifies the window size limit; tests the decay of older context.

**Why:** Context carries information across `decide()` calls. Without correct context management, the engine would treat every decision as independent.

---

### `unit/cognitive/test_metacognition.py` — 5 tests

**What:** Tests `MetacognitiveEngine` performance tracking and threshold detection.

**How:** Records sequences of successes and failures; verifies that `should_sleep()` returns True when performance degrades below threshold.

**Why:** Metacognition triggers offline consolidation. Incorrect thresholds would cause either premature or delayed sleep, harming performance.

---

### `unit/cognitive/test_metacognitive_veto.py` — 3 tests

**What:** Tests `SafetyGate` veto logic.

**How:** Registers a safety constraint; creates a coalition that violates it; verifies the coalition is vetoed.

**Why:** Safety is the highest-priority property of the system. These tests ensure that no amount of activation can bypass a registered safety constraint.

---

### `unit/cognitive/test_self_model.py` — 3 tests

**What:** Tests `SelfModel` confidence calibration.

**How:** Records sequences of successes and failures; verifies that `get_confidence()` converges to the empirical success rate; tests cold-start behaviour.

**Why:** Calibrated confidence is used in the GWT coalition scoring and exploration decisions. Incorrect calibration would lead to overconfident or underconfident behaviour.

---

### `unit/cognitive/test_integrated_metacognition.py` — 2 tests

**What:** Tests `SafetyGate` and `MetacognitiveEngine` working together inside `CognitiveEngine`.

**How:** Runs a full `decide()` cycle with a safety-violating action available; verifies the violating action is never selected.

**Why:** Integration test for the metacognitive veto path through the full engine.

---

### `unit/language/test_lingua.py` — 1 test

**What:** Tests `LinguaCortex` encoding properties.

**How:** Encodes two sentences; verifies (1) identical sentences produce identical HVs (determinism), (2) different sentences produce dissimilar HVs (quasi-orthogonality).

**Why:** LinguaCortex is the primary text→HV encoder. Determinism ensures reproducibility; quasi-orthogonality ensures distinct concepts are distinguishable.

---

### `unit/learning/test_text_knowledge_learner.py` — 16 tests

**What:** Tests `TextKnowledgeLearner` — concept extraction, relation detection, causal keyword detection, and query.

**How:**
- Feeds sentences; counts new concepts in SemanticMemory.
- Checks that SVO relations create graph edges.
- Tests causal keyword detection ("causes", "prevents", etc.).
- Tests `query()` returns relevant concepts.

**Why:** TextKnowledgeLearner is the primary knowledge ingestion mechanism for the AI model. All downstream reasoning depends on correct concept and relation extraction.

---

### `unit/learning/test_cross_domain.py` — 15 tests

**What:** Tests the cross-domain knowledge transfer engine (`TransferEngine`, `SchemaExtractor`, `StructureMapper`, `RuleLifter`).

**How:**
- Registers concepts and rules in a source domain.
- Declares explicit correspondences or lets `discover_correspondences` find them.
- Calls `transfer()` and checks returned `TransferredInference` objects: correct relation labels, confidence ordering, non-empty target fillers.
- Tests edge cases: no correspondences → empty result; single auto-discovered correspondence → valid mapping.

**Why:** Cross-domain transfer is the primary mechanism for generalising knowledge across domains. These tests verify that the structural mapping and rule lifting steps produce correct inferences without corrupting source domain knowledge.

---

### `unit/language/test_semantic_roles.py` — 24 tests

**What:** Tests the `SemanticRoleLabeler` and related VSA helper functions.

**How:**
- Labels sentences with known role structures; checks `frame.pred`, `frame.agent`, `frame.patient`, `frame.location`, `frame.temporal`, `frame.manner`, `frame.negation`.
- Tests predicate finding: irregular verb takes priority over morphological pattern.
- Tests resonator verification: codebook update doesn't corrupt frame.
- Tests VSA helpers: determinism of word/role HVs; different words produce different HVs.

**Why:** SRL is the entry point for structured language understanding. Incorrect role extraction would corrupt all downstream causal and semantic reasoning that depends on thematic structure.

---

### `unit/language/test_nlg_discourse.py` — 19 tests

**What:** Tests `DiscoursePlanner` and `NLGEngine` (including backward compatibility with `StructuralRealizer`).

**How:**
- Passes single and multi-frame lists to `DiscoursePlanner.plan()` and checks: connective insertion, pronoun anaphora, procedural numbering, negation surface form, and multi-frame paragraph generation.
- Tests `NLGEngine.generate_discourse()` and `generate_causal_chain()` for correct multi-sentence output.
- Tests backward compatibility: `NLGEngine.generate("factual", ...)` still produces a single sentence.

**Why:** NLG quality is the primary user-facing output of the system. These tests ensure that discourse planning produces coherent, well-connected paragraphs without breaking the existing single-sentence generation API.

---

### `unit/perception/test_semantic_folding.py` — 2 tests

**What:** Tests that semantic folding preserves important properties.

**How:** Encodes word HVs; verifies that (1) the same word always produces the same HV, (2) the XOR binding of two words is quasi-orthogonal to both.

**Why:** Semantic folding is the foundation of text representation. Property violations would corrupt all text-based reasoning.

---

### `unit/perception/test_semantic_roles.py` — 10 tests

**What:** Tests role-filler binding and unbinding for semantic role representation (subject, verb, object, etc.).

**How:** Binds a role HV with a filler HV via XOR; verifies unbinding recovers the filler; tests that the bound vector is quasi-orthogonal to both.

**Why:** Role-filler binding is used to represent structured knowledge (e.g., "capital(France)=Paris"). Incorrect unbinding would corrupt all structured queries.

---

### `unit/reasoning/test_math_reasoning.py` — 37 tests

**What:** Tests all math reasoning components: `FPECodebook`, `ExpressionEvaluator`, `LinearSolver`, `WordProblemParser`, `MathReasoner`.

**How:**
- `FPECodebook`: monotone similarity (adjacent numbers more similar than distant), determinism, boundary integers.
- `ExpressionEvaluator`: arithmetic with precedence, parentheses, unary minus, division by zero guard.
- `LinearSolver`: standard `x + b = c`, coefficient `ax`, RHS constant, negative solution.
- `WordProblemParser`: cue-word classification for all four operations.
- `MathReasoner`: end-to-end solve/compare/verbalize, `is_math_query` detection.

**Why:** Math reasoning must be exact and reproducible. These tests verify that no rounding error, parser edge case, or encoding collision slips through.

---

### `unit/integration_core/test_brain_fusion.py` — 6 tests

**What:** Tests `BrainFusion` — merging knowledge across task domains.

**How:** Creates two `TaskBrain` objects with different rules; fuses them; verifies that rules from both brains are available in the fused result; tests knowledge transfer.

**Why:** BrainFusion enables multi-task learning. These tests ensure knowledge transfer doesn't corrupt or lose existing knowledge.

---

### `unit/integration_core/test_global_workspace.py` — 1 test

**What:** Tests that `GlobalWorkspace.compete()` selects the highest-activation coalition.

**How:** Creates coalitions with known activation scores; verifies the winner is the highest-activation coalition.

**Why:** The GWT competition is the central decision mechanism. An incorrect competition result would produce wrong actions.

---

## NSCK Integration Tests

Integration tests exercise multiple modules together, verifying that the modules interact correctly and that emergent system behaviour is correct.

---

### `integration/test_system_capabilities.py` — 53 tests

**What:** Comprehensive capability proof tests — the most important test file in the codebase. Each test proves that NSCK can (or explicitly documents that it cannot) perform a specific capability.

**How:**

| Test class | What it proves |
|---|---|
| `TestVSACoreOperations` | XOR orthogonality, XOR invertibility, bundle similarity, permute invertibility, similarity accuracy, weighted bundle, LSH determinism, O(n) time complexity |
| `TestMemorySystems` | EpisodicMemory stores and recalls, LSH retrieval speed, capacity |
| `TestRuleLearning` | Rules learned from experience, confidence calibration, rule pruning |
| `TestCausalReasoning` | ΔP discovery, chain traversal, counterfactual, no spurious correlations |
| `TestBrainFusion` | TaskBrain creation, FusedBrain merging, cross-task transfer |
| `TestMetacognition` | MetacognitiveEngine tracking, SafetyGate veto |
| `TestPlanning` | STRIPSPlanner A* search, operator learning from CausalGraph |
| `TestEfficiency` | O(n) scaling for HV ops, memory usage bounds |
| `TestKnownLimitations` | Explicitly tests known limitations (documents what NSCK cannot yet do) |
| `TestCrossModuleIntegration` | CognitiveEngine wires all modules correctly |

**Why:** This file serves as the official capability specification. Each test is both documentation and verification. When a new capability is added, a test here proves it. When a limitation is resolved, the `xfail` marker is removed.

---

### `integration/test_cognitive_wiring.py` — 19 tests

**What:** Tests that all modules are correctly wired through `CognitiveEngine`.

**How:** Instantiates a full `CognitiveEngine`; calls `register_task()`, `decide()`, `record_outcome()`, `learn()`, `sleep()`; verifies each call produces expected outputs and updates internal state correctly.

**Why:** The CognitiveEngine is the integration point. Wiring tests ensure no module is accidentally disconnected or incorrectly initialised.

---

### `integration/test_cross_module.py` — 1 test

**What:** Tests that data written by one module is correctly readable by another.

**How:** Writes a concept via `TextKnowledgeLearner`; reads it via `SemanticMemory.query()`; verifies the same HV is returned.

**Why:** Cross-module data consistency is essential. If modules use different internal representations, integration silently fails.

---

### `integration/test_phase5.py` — 6 tests

**What:** Phase 5 integration — language processing combined with reasoning.

**How:** Feeds text to `TextKnowledgeLearner`; runs a causal query; verifies the causal link is available for reasoning.

**Why:** Phase 5 represents the point where language learning feeds into the reasoning pipeline. These tests verify the handoff is correct.

---

### `integration/test_phase8_mental_rehearsal.py` — 10 tests

**What:** Tests the mental rehearsal veto mechanism (Phase 8 addition).

**How:** Registers a danger HV; creates a WorldModel that predicts a dangerous next state for a specific action; verifies that `GlobalWorkspace.compete()` vetoes that action and selects the next-best coalition.

**Why:** Mental rehearsal is a critical safety feature. These tests verify that the veto fires correctly, that it tries alternative coalitions, and that it falls back to the default action when all coalitions are vetoed.

---

### `integration/test_phase8_permutation.py` — 10 tests

**What:** Tests permutation properties at the integration level (multiple modules using permuted HVs).

**How:** Creates permuted HVs in one module; passes them to another module; verifies the permutation is preserved correctly through the pipeline.

**Why:** Permutation encodes positional information. If modules strip or corrupt the permutation, word-order encoding is lost.

---

### `integration/test_phase8_universal_input.py` — 17 tests

**What:** Tests `UniversalInput` with all supported input types.

**How:** Passes strings, dicts, images (ndarray), and numerics to `UniversalInput.encode()`; verifies each produces a valid HV; verifies that different inputs produce different HVs.

**Why:** `UniversalInput` is the entry point for all inputs. Incorrect encoding for any type would silently corrupt all reasoning for that input type.

---

### `integration/test_stability.py` — 3 tests

**What:** Tests numerical stability over long operation sequences.

**How:** Runs 1000 consecutive XOR/bundle/permute operations; verifies that similarity values remain in [0, 1] and do not drift or overflow.

**Why:** Long-running deployments would accumulate floating-point errors without stability. These tests catch numerical drift early.

---

### `integration/test_phase2_old.py` — 6 tests

**What:** Phase 2 regression tests — SNN perception and VSA-SNN bridge.

**How:** Creates a `SNNPerceptionModule`; feeds synthetic sensory inputs; verifies output HVs are valid.

**Why:** Phase 2 introduced the SNN perception layer. These tests are kept as regression coverage to ensure subsequent changes don't break SNN integration.

---

### `integration/test_realworld_capabilities.py` — 145 tests

**What:** Full-stack real-world capability test suite covering every layer of the NSCK core architecture (Python + Rust), including cross-domain knowledge transfer.

**How:** Organised as 20 test classes, each targeting a concrete subsystem or integration boundary:

| Class | Subsystem | Coverage |
|---|---|---|
| `TestVSACore` | `HyperVectorPy` | XOR invertibility, deterministic bundle, weighted bundle, LSH hash, similarity range |
| `TestRustHyperVectorRegistry` | `HyperVectorRegistry` (Rust) | Register, lookup, concurrent reads, shim compat |
| `TestRustSemanticMemoryConcurrent` | `SemanticMemoryConcurrent` (Rust) | add/query, parallel spreading activation |
| `TestRustEpisodicMemoryConcurrent` | `EpisodicMemoryConcurrent` (Rust) | store, k-NN recall, hot-tier eviction |
| `TestRustActivationAccumulator` | `ActivationAccumulator` (Rust) | accumulate, top-k, merge |
| `TestRustPersistentStorage` | `PersistentStorage` (Rust, SQLite) | HV round-trip, concurrent writes |
| `TestRustCognitiveWorkerPool` | `CognitiveWorkerPool` (Rust, Tokio) | parallel task dispatch |
| `TestRustSNN` | `snn_rs` (Rust) | LIFLayer, StdpEngine, SnnCore, HebbianMatrix, RateCoder |
| `TestPythonSemanticMemory` | `SemanticMemory` (Python) | concept storage, spreading activation, inheritance |
| `TestPythonEpisodicMemory` | `EpisodicMemory` (Python) | record, recall, similarity-based retrieval |
| `TestCausalGraphAndReasoner` | `CausalGraph` + `CausalReasoner` | multi-hop chains, counterfactuals, interventions |
| `TestGlobalWorkspace` | `GlobalWorkspace` | coalition competition, broadcast, working memory |
| `TestAnalogyEngine` | `AnalogyEngine` | source–target mapping, stem-bridges |
| `TestSTRIPSPlanner` | `STRIPSPlanner` | goal regression, plan validity |
| `TestCognitiveModules` | `EmotionSystem`, `SelfModel`, `CuriosityModule` | valence update, calibration, novelty decay |
| `TestTextKnowledgeLearner` | `TextKnowledgeLearner` | SVO extraction, shared causal graph wiring |
| `TestNLGEngine` | `NLGEngine` + `DiscoursePlanner` | fluent generation, connective sequencing |
| `TestSemanticRoleLabeler` | `SemanticRoleLabeler` | 12 thematic roles, resonator decoding |
| `TestCrossDomainKnowledgeTransfer` | `TransferEngine` + `AnalogyEngine` | multi-domain learn → analogy bridge → spreading activation crosses domain boundary |
| `TestIntegrationInvariants` | All | Shared-object identity, relation-weight coverage, Rust–Python interop |

**Why:** Validates the full architecture under realistic inputs (multi-sentence corpora, multi-step causal chains, multi-domain analogies) rather than isolated unit conditions. Each test class maps to a concrete architecture layer and can be run independently. The cross-domain transfer tests are the primary regression guard for the TKL ↔ AnalogyEngine ↔ SemanticMemory integration path.

---

## NSCK Architecture Tests

Architecture tests verify that the entire system, across all modules, satisfies specific capability claims.

---

### `core_architecture/test_learning.py` — 16 tests

**What:** Tests that the system genuinely learns and retains knowledge.

**How:**
| Test class | What it proves |
|---|---|
| `TestConceptAccumulation` | Concepts accumulate as text is fed in; learned concepts are queryable; similar concepts cluster |
| `TestRelationLearning` | Relations are extracted and stored; is_a, causes, has_property edges exist |
| `TestCausalLearning` | Causal links learned from text; forward chain traversal works |
| `TestMemoryRetention` | Learned concepts persist across multiple query calls |
| `TestIncrementalLearning` | Each new text adds new knowledge without erasing previous knowledge |

**Why:** Learning retention is the most fundamental requirement. Without it, NSCK is just a rule-based system. These tests prove knowledge accumulates.

---

### `core_architecture/test_reasoning.py` — 18 tests

**What:** Tests that the system reasons over its learned knowledge.

**How:**
| Test class | What it proves |
|---|---|
| `TestCausalReasoning` | Forward chain, backward chain, confidence propagation |
| `TestGlobalWorkspace` | Coalition competition, winner selection, broadcast |
| `TestRuleLearning` | Rules induced from observation, pruning |
| `TestAnalogy` | Cross-domain structural mapping |
| `TestPlanning` | A* search, plan execution |

**Why:** Reasoning over learned knowledge is what distinguishes NSCK from a database. These tests verify the reasoning chain is connected to the learning pipeline.

---

### `core_architecture/test_snn_integration.py` — 29 tests

**What:** Tests the Spiking Neural Network perception module and its integration with the cognitive pipeline.

**How:**
- Tests LIF neuron layer: spike generation, refractory period, membrane potential dynamics.
- Tests STDP learning: weight changes for causal and anti-causal spike pairs.
- Tests VSA-SNN bridge: rate encoding, temporal encoding, HV output.
- Tests `SNNPerceptionModule.perceive()` end-to-end.
- Tests Rust SNN backend (if available) produces identical results to Python SNN.

**Why:** The SNN is the biological perception front-end. These 29 tests are the most detailed in the test suite because SNN dynamics have many subtle failure modes (numerical stability, boundary conditions, refractory period off-by-one errors).

---

### `core_architecture/run_comprehensive_tests.py`

Orchestrates all architecture tests in a single run with summary reporting. Not a pytest file — run directly: `python tests/core_architecture/run_comprehensive_tests.py`.

---

## NSCK Experiments

Behavioural validation scenarios that test complex emergent behaviour, not individual functions.

---

### `experiments/belief_revision_test.py`

**What:** Tests belief revision — the ability to update beliefs when newer contradictory information arrives.

**Scenario:** Feed text stating "The sky is blue" (timestamped T=1000). Then feed "The sky is green" (timestamped T=2000). Query: should return green (the more recent belief).

**How:** Uses `SemanticMemory.add_relation()` with timestamps; the `timestamp` parameter causes newer information to overwrite older information for the same relation.

**Why:** Belief revision is a fundamental requirement for a system operating in a changing world. A system that never updates its beliefs is unreliable. This test verifies the temporal ordering mechanism works correctly.

---

### `experiments/text_reasoning_test.py`

**What:** Tests end-to-end text-grounded reasoning.

**Scenario:** Feed a paragraph about a domain; ask a question that requires combining multiple facts.

**How:** Measures whether the answer contains concepts from the relevant fact chains.

**Why:** Text-grounded reasoning is the primary use case for the AI model. This test validates that the full pipeline (text → concepts → spreading activation → causal chains → response) produces relevant outputs.

---

### `experiments/transitive_test.py`

**What:** Tests transitive inference through causal/semantic chains.

**Scenario:** A→B, B→C; query: does A lead to C?

**How:** Builds a 3-node causal chain; queries the end node; verifies that the multi-hop path is discovered.

**Why:** Transitive inference is required for complex reasoning. Without it, the system can only answer questions about directly observed facts, not inferred ones.

---

### `experiments/verify_f1.py`

**What:** Measures F1 score on a labelled question-answer test set.

**How:** Feeds training sentences; queries with test questions; tokenises both expected and actual answers; computes precision, recall, and F1.

**Why:** F1 provides an objective, reproducible benchmark for knowledge retrieval quality. It allows comparison across code changes.

---

## NSCK Regression Tests

### `regression/test_bug_fixes.py` — 6 tests

**What:** Regression tests for specific bugs that were found and fixed.

**How:** Each test reproduces the exact input that triggered the original bug; verifies the bug does not recur.

**Why:** Regression tests prevent bugs from being reintroduced when code is refactored. Each test here represents a real bug that was found in production.

---

## nsck_ai_model Tests

### `nsck_ai_model/tests/test_ai_engine.py` — 77 tests

The main unit and integration test file for the AI engine.

**Fixtures:**
- `trained_engine` (module scope) — engine trained on 24 seed sentences covering geography, science, animals, and technology.
- `fresh_engine` (function scope) — untrained engine for testing learning.

**Test classes and what they verify:**

| Class | Tests | Focus |
|---|---|---|
| `TestNSCKModuleIntegration` | 14 | Verifies that all NSCK core modules (SemanticMemory, EpisodicMemory, GlobalWorkspace, EmotionSystem, CausalReasoner, SelfModel, CuriosityModule, TextKnowledgeLearner, HyperVector) are actually instantiated and populated inside `NSCKAIEngine` — not just imported |
| `TestKnowledgeQA` | 10 | End-to-end QA: capital of France, capital of Germany, dogs are mammals, Earth orbits Sun, rain causes flooding, Python is a programming language, Shakespeare, water molecule, unknown topic fallback |
| `TestGlassBoxTrace` | 8 | ThoughtTrace structure: 11 stages present, all NSCK modules recorded, timing data in each step, unique trace ID, GWT winner recorded, self-model confidence, curiosity novelty score |
| `TestResponseQuality` | 6 | Response format: grammatical structure, relevance, no duplicate sentences, not a question, confidence is float, latency < 2s |
| `TestEmotionSystem` | 4 | Emotion in response dict, emotion blend percentages, valence/arousal in range, mood history |
| `TestTraining` | 5 | `train_on_text()` returns stats dict, populates SemanticMemory, populates TextKnowledgeLearner facts, builds sentence index |
| `TestImageTraining` | — | `train_on_image()` processes image + caption, stores cross-modal episode |
| `TestEdgeCases` | — | Empty input, very long input, unicode, numeric-only input, repeated queries |

**Key assertion patterns:**
```python
# Verify NSCK module is actually used (not a mock)
assert isinstance(engine.semantic_memory, SemanticMemory)
assert len(engine.semantic_memory.concept_hvs) > 0

# Verify trace has all 11 stages
stage_names = [s.stage for s in result['trace'].steps]
assert 'input_encoding' in stage_names
assert 'gwt_competition' in stage_names
assert 'response_generation' in stage_names

# Verify QA produces relevant response
result = engine.chat("What is the capital of France?")
assert 'paris' in result['response'].lower()
```

---

### `nsck_ai_model/tests/test_production.py` — 46 tests

Production-readiness validation — scenarios that reflect real-world usage.

**Test classes:**

| Class | Tests | Focus |
|---|---|---|
| `TestKnowledgeQA` | 10 | Production QA with wider training set (8 geographies, 8 science, 5 animals, 4 technology facts) |
| `TestNSCKArchitecture` | 7 | All NSCK modules populated; SemanticMemory has concepts; spreading activation works; emotion classifies; self-model tracks |
| `TestTraceability` | 4 | Trace has 11 stages; records NSCK modules; has timing data; JSON-serialisable |
| `TestNaturalLanguage` | 5 | Response starts with capital letter; ends with period; not a bare question; relevant to query; no duplicates |
| `TestEdgeCases` | 5 | Empty input, XSS injection (verifies no HTML tags in output), very long input (>1000 chars), Unicode, numeric-only |
| `TestPerformance` | 3 | Single chat < 1s; single train < 1s; burst 10 queries < 5s |
| `TestSystemStats` | 2 | Stats include NSCK module counts; export includes concepts + relations + facts |
| `TestDashboardAPI` | 10 | All 10 Flask endpoints: health, chat, train, stats, knowledge, concepts, relations, rules, emotion, logs |

**Performance assertions:**
```python
def test_chat_latency_under_1s(self, model):
    start = time.time()
    model.chat("What is water?")
    elapsed = time.time() - start
    assert elapsed < 1.0

def test_burst_10_queries_under_5s(self, model):
    start = time.time()
    for _ in range(10):
        model.chat("What is the capital of France?")
    assert time.time() - start < 5.0
```

**Dashboard tests** use Flask's test client:
```python
@pytest.fixture
def client():
    from nsck_ai_model.dashboard import app
    app.config['TESTING'] = True
    return app.test_client()

def test_health(self, client):
    r = client.get('/api/health')
    data = json.loads(r.data)
    assert data['status'] == 'ok'
```

---

### `nsck_ai_model/tests/test_multimodal.py`

**What:** Tests image training and description pipeline.

**How:**
- Creates a synthetic test image (numpy array).
- Calls `engine.train_on_image(image, caption)`.
- Calls `engine.describe_image(image)`.
- Verifies the description contains concepts from the caption.

**Why:** The multimodal pipeline requires `MultimodalProcessor` and `ImageGenerator` to work together. This test verifies the cross-modal episode is stored and retrievable.

---

## Rust VSA Property Tests

### `nsck/rust_vsa/tests/integration_test.rs` — 11 property tests

Using the `proptest` crate for property-based testing (random inputs, verified invariants).

| Test | Property verified |
|---|---|
| `test_determinism` | Same seed → same HV bits |
| `test_xor_reversibility` | `xor(xor(A, B), B) == A` for all A, B |
| `test_permute_reversibility` | `permute_inverse(permute(A, k), k) == A` for all A, k |
| `test_similarity_symmetry` | `sim(A, B) == sim(B, A)` for all A, B |
| `test_similarity_range` | `sim(A, B) ∈ [0, 1]` for all A, B |
| `test_bundle_validity` | `sim(bundle(A, B), A) > 0.6` for all A, B |
| `test_weighted_bundle` | Weighted bundle biases toward dominant operand |
| `test_lsh_stability` | LSH hash is consistent for same input |
| `test_parallel_search_consistency` | Parallel similarity search matches sequential search |
| `test_registry_concurrency` | Concurrent insertions don't corrupt `HyperVectorRegistry` |
| `test_activation_accumulation` | `ActivationAccumulator` correctly accumulates partial sums |

**How to run:**
```bash
cd nsck/rust_vsa
cargo test --release
```

**Why property tests?** Random input generation finds edge cases that handwritten tests miss. The `proptest` framework shrinks failing inputs to minimal examples, making debugging easier.

---

## Python Core Tests

### `python/core/tests/real_world_eval.py` — ~40 assertions

**What:** Real-world evaluation of the full cognitive pipeline.

**How:** Runs against a curated set of text passages from the `nsck/data/` directory (belief revision, sound physics, xylophone/planets). Measures:
- Concept recall accuracy
- Relation extraction F1
- Causal inference accuracy
- Response relevance

**Why:** Real-world text is messier than synthetic test data. This evaluation catches failure modes that unit tests miss.

### `python/core/tests/test_system_integration.py`

System-level integration test: instantiates a complete `CognitiveEngine`, registers a domain, runs 50 decision cycles, and verifies that the system learns and improves over time.

### `python/core/tests/test_transparency.py`

Transparency test: for each `decide()` call, verifies that the returned `CognitiveState` contains a non-empty `explanation` object with a human-readable reason string.

---

## Testing Philosophy

### Why test VSA mathematical properties?

VSA operations underlie every other system component. A bug in XOR or bundle would silently corrupt everything above it. Mathematical property tests (especially the `test_hypervec_parity.py` suite) catch these bugs before they propagate.

### Why test with real text?

Synthetic tests can be "gamed" by implementations that special-case the test inputs. Real text (from `nsck/data/` and HuggingFace) ensures the system handles natural variation.

### Why document known limitations?

The `TestKnownLimitations` class in `test_system_capabilities.py` explicitly tests for behaviours the system cannot yet do (marked `@pytest.mark.xfail`). This turns limitations into tracked items rather than silent failures, and makes it clear when a limitation has been resolved (the xfail marker is removed, the test passes).

### Why assert on glass-box trace?

The ThoughtTrace tests (`TestGlassBoxTrace`) verify not just the final answer but the reasoning process. A system that produces the right answer by the wrong mechanism is not trustworthy. These tests ensure every reasoning stage is correctly populated and reported.

### Why performance tests?

NSCK is intended for real-time decision loops (e.g., games, robotics). A system that takes 10 seconds per decision is not deployable. Performance tests (`< 1s per chat`, `< 5s for 10 queries`) catch performance regressions early.
