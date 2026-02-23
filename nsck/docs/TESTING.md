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
# All NSCK tests (from repo root)
python -m pytest nsck/tests/ -q

# Unit tests only (fast, ~20s)
python -m pytest nsck/tests/unit/ -q

# Integration tests
python -m pytest nsck/tests/integration/ -q

# V3-specific tests only
python -m pytest nsck/tests/unit/language/ nsck/tests/unit/memory/ \
  nsck/tests/unit/reasoning/ nsck/tests/integration/test_full_pipeline_v3.py -q

# Rust backend tests (requires hypervec_rs.so + snn_rs.so built)
python -m pytest nsck/tests/unit/rust/ -q

# Architecture tests
python -m pytest nsck/tests/core_architecture/ -q

# Rust property tests (Cargo)
cd nsck/rust_vsa && cargo test --release
```

> **Note:** Run from the repository root, not from inside `nsck/`. The `pytest.ini` sets `pythonpath = nsck`, so imports like `from python.core...` resolve correctly.

---

## Test Overview

| Layer | Location | Files | Tests | Description |
|---|---|---|---|---|
| Unit — core | `nsck/tests/unit/vsa/` `cognitive/` `learning/` | 9 files | ~115 | Core module isolation |
| Unit — language (V3+) | `nsck/tests/unit/language/` | 9 files | ~105 | CG, frames, coreference, distributional, NLG, SRL, pragmatics, V4 features |
| Unit — memory | `nsck/tests/unit/memory/` | 4 files | ~24 | Homeostasis, stigmergy, HNSW, cleanup |
| Unit — reasoning (V3+) | `nsck/tests/unit/reasoning/` | 11 files | ~110 | Belief revision, dual process, blending, spatial, math |
| Unit — V6 features | `nsck/tests/unit/test_v6_features.py` | 1 file | **86** | FluentNLG, Brill POS, NSW ANN, multimodal, concurrent, negate |
| Unit — V7 features | `nsck/tests/unit/test_v7_features.py` | 1 file | **54** | KG stop-concept filter, FluentNLG wired, distributional init |
| Unit — Rust backends | `nsck/tests/unit/rust/` | 1 file | **81** | HV math, cross-backend parity, SNN, pipeline |
| Integration | `nsck/tests/integration/` | 12 files | ~315 | Cross-module + V3 pipeline + real-world |
| Architecture | `nsck/tests/core_architecture/` | 4 files | ~63 | Architecture capability verification |
| Experiments | `nsck/tests/experiments/` | 4 files | ~10 | Behavioural scenarios |
| Regression | `nsck/tests/regression/` | 1 file | 6 | Known-fixed bug coverage |
| Rust (Cargo) | `nsck/rust_vsa/tests/` | 1 file | 11 | Property-based VSA correctness |

**Total: ~951 tests passing** in `nsck/tests/` (78 test files).

**Actual pytest run results (verified Feb 2026):**

| Build condition | `python -m pytest nsck/tests/ -q` result |
|---|---|
| Python-only (no Rust .so) | **807 passed**, 150 skipped, 3 xfailed |
| With Rust .so (hypervec_rs + snn_rs) | **951 passed**, 5 skipped, 4 xfailed |

The 150 skipped tests (Python-only) are all in `nsck/tests/unit/rust/test_rust_backends.py` and `nsck/tests/unit/vsa/test_hypervec_parity.py` — they skip gracefully when the Rust `.so` files are not in `nsck/`.

The additional `xfailed` test with Rust is `test_seed_determinism` in `test_hypervec_parity.py` — Rust uses ChaCha8 RNG, Python uses PCG64; seeded construction produces different bit sequences (accepted, documented limitation).

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

## V3–V7 Unit Tests

These tests were added in V3 through V7 and cover all new modules. They skip gracefully when optional dependencies (Rust, hnswlib, torch) are not installed.

---

### `unit/language/test_construction_grammar.py` — 8 tests

**What:** Tests `ConstructionMatcher` against diverse English constructions.

**How:** Feeds tokenised sentences and verifies that the correct construction name and role fillers are returned. Covers SVO, copular `is`, `is a`, possessive `has`, causative `causes`, containment `contains`.

**Why:** Construction grammar is the backbone of V3 NLU. These tests ensure each construction template matches correctly.

---

### `unit/language/test_frame_semantics.py` — 6 tests

**What:** Tests `FrameLibrary` verb lookup and `Frame.fill()` / `Frame.extract_filler()` roundtrips.

**How:** Calls `get_frame_for_verb()` for verbs like "bought", "said", "caused"; fills a frame with HV role fillers; extracts fillers and checks similarity ≥ 0.4.

---

### `unit/language/test_coreference.py` — 7 tests

**What:** Tests `EntityRegister` pronoun resolution.

**How:** Registers named entities with gender/animacy/number features. Resolves pronouns (`he`, `she`, `it`, `they`, `this`). Checks register overflow (max 10 entities).

---

### `unit/language/test_distributional_semantics.py` — 5 tests

**What:** Tests `DistributionalCodebook` co-occurrence building and retrieval.

**How:** Builds a codebook from a small corpus; checks co-occurring words produce higher similarity; tests `save()`/`load()` roundtrip.

---

### `unit/memory/test_homeostasis.py` — 6 tests

**What:** Tests `MemoryHomeostasis.regulate()` — edge pruning and stale eviction.

---

### `unit/memory/test_stigmergy.py` — 7 tests

**What:** Tests `mark_path()`, `evaporate_stigmergy()`, and stigmergy-weighted `spread_activation()`.

**How:** Marks a preferred path 10× with reward=1.0 and an alternative once. Checks preferred-path pheromone >> alternative. Verifies `evaporate_stigmergy()` decay.

---

### `unit/memory/test_hnsw_memory.py` — 4 tests

**What:** Tests HNSW index graceful fallback when `hnswlib` is not installed.

---

### `unit/reasoning/test_belief_revision.py` — 7 tests

**What:** Tests `BeliefScorer.free_energy()` and `should_revise()` decision logic.

**How:** Verifies FE is higher for contested beliefs (high contradictions); verifies revision decision is correct.

---

### `unit/reasoning/test_dual_process.py` — 8 tests

**What:** Tests System 1 / System 2 routing in `CognitiveEngine.decide()`.

**How:** `threshold=0.0` → expects `system_used="system_1"`. `threshold=0.99` → expects `system_used="system_2"`.

---

### `unit/reasoning/test_conceptual_blending.py` — 9 tests

**What:** Tests `AnalogyEngine.blend()` and `functor_quality()`.

**How:** Creates two small concept domains; verifies blend HV is non-zero; verifies `functor_quality()` = 1.0 on a perfect mapping.

---

### `unit/language/test_pragmatics.py` — 45 tests (V5)

**What:** Tests `PragmaticEngine` — Gricean maxims, scalar implicatures, speech-act classification, and presupposition projection.

**How:**
- Feeds utterances to `classify_speech_act()`; verifies correct category (assertion, question, directive, promise, threat, warning).
- Generates scalar implicatures: "some → not all", "possible → not certain", "warm → not hot".
- Tests Gricean maxim violations: quantity-based reasoning.
- Tests presupposition projection: "The king of France is bald" → presupposes "France has a king".

**Why:** Pragmatic reasoning is needed for any natural dialogue system that must handle indirect speech acts. These tests verify the 15 Horn scales, 7 speech acts, and Gricean implicature logic are all correctly implemented.

---

### `unit/reasoning/test_spatial_reasoning.py` — 45 tests (V5)

**What:** Tests `SpatialReasoner` and `PositionCodebook` — VSA-based spatial relation encoding.

**How:**
- Encodes positions with `PositionCodebook` using FPE bit-flip encoding; verifies adjacent positions are more similar than distant ones.
- Tests `assert_spatial(obj1, obj2, relation)` for 8 relations: above, below, left, right, inside, outside, near, far.
- Tests `query_relation(obj1, obj2)` returns correct relation string.
- Tests boundary cases: `_MIN_QUERY_SIMILARITY=0.4` threshold, `_NEGATIVE_STEP_OFFSET=100_000` avoidance.

**Why:** Spatial reasoning requires that position HVs have a monotone similarity gradient. FPE bit-flip is used instead of permute because Python permute() gives ~0.5 regardless of distance — verified and documented in `_AXIS_FLIP_BITS=50`.

---

### `unit/language/test_v4_features.py` — 50 tests (V4)

**What:** Tests all V4 language enhancements: negation handling, temporal connectives, conditional logic, transitive inference, and prototype generalization.

**How:**
- Negation: `"The dog is not friendly"` → `NEGATION` tag present in parsed tokens; bound HV uses `negate()`.
- Temporal: `"After the rain, the flowers bloom"` → `TEMPORAL_CONNECTIVE` construction matched; `"after"` as TEMP tag.
- Conditional: `"If it rains, the ground gets wet"` → `CONDITIONAL` construction; antecedent/consequent separated.
- Transitive: adds `A is_a B`, `B is_a C`; calls `SemanticMemory.infer_transitive()`; verifies `A is_a C` inferred.
- Prototype: adds 3 members with `is_a Animal`; calls `build_prototypes()`; verifies prototype HV similarity to each member > 0.6.

**Why:** V4 addressed 5 specific gaps: negation as first-class linguistic operator, temporal ordering, conditional reasoning, inheritance chains, and category prototypes. These tests ensure each gap is properly filled.

---

### `unit/test_v6_features.py` — 86 tests (V6)

**What:** Comprehensive tests for all 6 V6 capabilities.

| Section | Tests | Coverage |
|---|---|---|
| FluentNLG | 20 | `FluentResponseComposer`, `NSCKResponseEngine`, relation verbalization, anaphora |
| BrillPosTagger | 14 | 300+ lexicon, suffix/prefix rules, unknown word handling |
| VSA negate() | 12 | Rust+Python negate, orthogonality, idempotence, `negate_negate==original` |
| DistributionalCodebook | 12 | PMI-weighted co-occurrence, context HV similarity, serialization |
| ConcurrentMultimodal | 14 | Parallel image+text+state processing, thread safety |
| NSW ANN fallback | 14 | `_NSWIndex` HNSW fallback, k-NN correctness without hnswlib |

All 86 tests pass with or without Rust `.so`.

---

### `unit/test_v7_features.py` — 54 tests (V7)

**What:** Tests for all V7 additions.

| Section | Tests | Coverage |
|---|---|---|
| KG stop-concept filter | 12 | `_STOP_CONCEPTS` frozenset (53 words), `_GENERIC_RELATION_THRESHOLD=0.62`, no filter noise in graph |
| FluentNLG wired into DM | 14 | `DialogueManager` uses `NSCKResponseEngine` for all response paths |
| Distributional pre-training | 10 | `DistributionalCodebook` on BUILTIN_CORPUS at init, similarity improvement |
| HF corpus loader | 10 | `hf_corpus_loader.py` offline fallback, `enable_hf_corpus=True/False` flag |
| Config flags | 8 | `enable_fluent_dialogue`, `enable_hf_corpus` present in `NSCKConfig` |

All 54 tests pass with Python-only backend.

---

### `unit/rust/test_rust_backends.py` — 81 tests

**What:** Comprehensive Rust backend validation across 8 sections.

| Section | Tests | Coverage |
|---|---|---|
| A: HV math | 11 | self-sim, XOR reversibility/commutativity, bundle symmetry, permute, LSH, weighted bundle |
| B: Cross-backend parity | 6 | XOR/permute/similarity Rust==Python; cross-type similarity |
| C: SemanticMemory Rust | 8 | add/query, spreading, stigmergy, belief revision, incremental refinement, latency |
| D: EpisodicMemory Rust | 4 | record/recall, LSH, 200-episode stress |
| E: SNN Rust classes | 25 | LIFLayer, HebbianMatrix, SnnCore, ConceptMapper, RateCoder, StdpEngine |
| F: V3 pipeline Rust | 7 | CG, coreference, distributional, frame, dual-process, homeostasis, research config |
| G: Real-world Rust | 9 | science, geography, biography+coreference, causality, contradiction, edge inputs |
| H: Scalability Rust | 7 | 10K HV creation, 1K similarity, 5K memory, 500-ring spreading, Rust vs Python |

All 81 tests are skipped gracefully when Rust `.so` files are not present in `nsck/`.

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

### `integration/test_full_pipeline_v3.py` — 9 tests

**What:** End-to-end tests for all V3 features wired through `TextKnowledgeLearner` and `SemanticMemory`.

**How:**
- `test_belief_revision_in_semantic_memory` — adds a relation then adds a contradicting relation 10×; verifies contradiction_count > 0.
- `test_coreference_resolution` — feeds a sentence with "he" after registering "John"; verifies coreference chain is populated.
- `test_frame_semantics_verb_lookup` — passes a sentence with "bought" to TKL; verifies a COMMERCIAL_TRANSACTION frame or relation was stored.
- Six further tests cover construction grammar coverage, dual-process routing, stigmergy preference, homeostasis sleep integration, and research config end-to-end.

**Why:** These are the primary regression guard for V3. If any integration wire breaks, these tests catch it before the unit tests do.

---

### `integration/test_real_world_v3.py` — 5 tests

**What:** Real-world capability tests using production-like text inputs.

**How:**
- Wikipedia paragraph about photosynthesis → query "What does photosynthesis produce?"
- Science chain: "Water is H2O", "H2O contains hydrogen", "Hydrogen is an element" → multi-hop query
- 3-sentence news snippet → entity and relation extraction accuracy
- Multi-turn coreference: "Tell me about dogs. Are they loyal? What do they eat?" → "they" → "dogs"
- Contradiction test: "The Earth is flat." (1×) vs "The Earth is round." (5×) → round belief wins

**Why:** Validates the system on naturalistic inputs, not just curated fixtures.

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
