# NSCK V4 Testing Guide

## Quick Start

Run the full test suite from the repository root:

```bash
NSCK_USE_RUST=1 python -m pytest nsck/tests/ -q
```

> **As of V4, `NSCK_USE_RUST=1` is the default.** Rust is expected to be built.

**Current results** (with Rust backends `hypervec_rs.so` + `snn_rs.so`):

| Metric     | Value                          |
|------------|--------------------------------|
| Collected  | 1443+ tests across 84+ files   |
| Passed     | 1430+                          |
| Failed     | 2 (stochastic — see below)     |
| Skipped    | 7                              |
| Xfailed    | 4                              |
| Run time   | ~14 seconds                    |

---

## 1. How to Run Tests

All commands assume you are in the repository root.

```bash
# Full suite (V4: Rust default-on)
NSCK_USE_RUST=1 python -m pytest nsck/tests/ -q

# Unit tests only (~8s)
python -m pytest nsck/tests/unit/ -q

# Integration tests
python -m pytest nsck/tests/integration/ -q

# V4-specific tests
python -m pytest nsck/tests/integration/test_v4_full_system.py -v

# Specific subsystem
python -m pytest nsck/tests/unit/reasoning/ -q
python -m pytest nsck/tests/unit/memory/ -q

# Rust backend tests (requires .so files built)
python -m pytest nsck/tests/unit/rust/ -q

# V4-specific tests
python -m pytest nsck/tests/integration/test_v4_full_system.py -v

# V13 substrate tests (baseline)
python -m pytest nsck/tests/unit/test_v13_universal_substrate.py -q

# Verbose output with short tracebacks
python -m pytest nsck/tests/ -v --tb=short
```

---

## 2. Test Directory Structure

```
nsck/tests/
├── unit/                    # Fast isolated tests
│   ├── cognitive/           # metacognition, self_model, context_engine,
│   │                        # integrated_metacognition, metacognitive_veto
│   ├── integration_core/    # brain_fusion, global_workspace
│   ├── language/            # nlg, construction_grammar, semantic_roles,
│   │                        # distributional, lingua, frame, pragmatics,
│   │                        # coreference, dialogue, v4_features,
│   │                        # test_vsa_nlu.py (V4 new)
│   ├── learning/            # cross_domain, text_knowledge, active_inference,
│   │                        # cross_modal
│   ├── memory/              # stigmergy, homeostasis, memory_lifecycle,
│   │                        # cleanup_memory, lsh_rebuild, hnsw_memory,
│   │                        # test_procedural_lsh.py (V4), test_semantic_hot_cache.py (V4)
│   ├── multimodal/          # concurrent_multimodal
│   ├── perception/          # stream_encoder, semantic_roles, semantic_folding
│   ├── reasoning/           # math, belief_revision, counterfactuals,
│   │                        # conceptual_blending, causal_discovery,
│   │                        # rule_learner, planning, plan_execution,
│   │                        # chaining, continuous_generalization,
│   │                        # theory_formation, dual_process, logic_bridge,
│   │                        # spatial, math_integration
│   ├── rust/                # rust_backends
│   ├── vsa/                 # hypervec_parity, hierarchical_resonator
│   ├── test_v6_features.py
│   ├── test_v7_features.py
│   ├── test_v10_extensions.py
│   ├── test_v13_universal_substrate.py
│   ├── test_substrate_api.py
│   ├── test_image_audio_adapters.py
│   ├── test_rust_backend.py
│   └── test_cross_disciplinary.py
├── integration/             # Cross-module tests
│   ├── test_system_capabilities.py
│   ├── test_cognitive_wiring.py
│   ├── test_e2e_substrate.py
│   ├── test_v9_substrate.py
│   ├── test_e2e_all_input_types.py
│   ├── test_cross_module.py
│   ├── test_realworld_capabilities.py
│   ├── test_benchmarks.py
│   ├── test_stability.py
│   ├── test_v4_full_system.py  (V4 new — 6 classes)
│   └── (several others)
├── core_architecture/       # Architecture-level tests
│   ├── test_learning.py
│   ├── test_reasoning.py
│   └── test_snn_integration.py
├── experiments/             # Research experiments
│   ├── transitive_test.py
│   ├── belief_revision_test.py
│   ├── text_reasoning_test.py
│   └── verify_f1.py
├── regression/              # Bug fix regression tests
│   └── test_bug_fixes.py
└── benchmarks/              # Performance benchmarks
    └── full_architecture_benchmark.py
```

### What lives where

| Directory            | Purpose                                                    |
|----------------------|------------------------------------------------------------|
| `unit/`              | Fast, isolated tests for individual modules. No external dependencies. |
| `integration/`       | Cross-module pipelines: wiring, end-to-end flows, stability. |
| `core_architecture/` | Tests that validate high-level architecture invariants (learning, reasoning, SNN integration). |
| `experiments/`       | Exploratory research scripts. Not part of CI — run manually. |
| `regression/`        | Tests that pin fixes for specific bugs so they never recur. |
| `benchmarks/`        | Performance-oriented tests and profiling scripts.          |

Version-feature test files (`test_v6_features.py`, `test_v7_features.py`, etc.) live
at the `unit/` top level and exercise the features introduced in that release.

---

## 3. Building Rust Backends for Testing

The Rust-accelerated VSA and SNN backends are optional but recommended. Tests
that require them will skip automatically if the `.so` files are absent.

```bash
pip install maturin

# Build the VSA backend
cd nsck/rust_vsa && maturin build --release && cd ../..

# Build the SNN backend
cd nsck/rust_snn && maturin build --release && cd ../..

# Extract shared objects from wheels
unzip -o nsck/rust_vsa/target/wheels/*.whl "hypervec_rs/*" -d /tmp/rv
cp /tmp/rv/hypervec_rs/*.so nsck/hypervec_rs.so

unzip -o nsck/rust_snn/target/wheels/*.whl "snn_rs/*" -d /tmp/rs
cp /tmp/rs/snn_rs/*.so nsck/snn_rs.so
```

After this, `nsck/tests/unit/rust/` and `test_rust_backend.py` will exercise
the compiled backends alongside their pure-Python equivalents.

---

## 4. Current Test Results

Snapshot taken with Rust backends enabled on a standard CI runner.

```
1443+ collected, 1430+ passed, 2 failed, 7 skipped, 4 xfailed in ~14s
```

### V4 Test Classes (`test_v4_full_system.py`)

| Class | Key checks |
|---|---|
| `TestProceduralFastPath` | LSH lookup wired, skills populated after positive reward, threshold == 0.72 |
| `TestSemanticHotCache` | `_hot_cache` dict exists, HNSW enabled by default, cache updated after spread_activation |
| `TestNLU` | VSANLUEngine classifies intent, confidence ≥ 0.0, entity extraction, process() dict output |
| `TestBundleMajorityVote` | bundle_hvs majority correct, result closer to majority input |
| `TestImagination` | imagine_rollout() returns tuple, multi-step works |
| `TestKnowledgeSeeder` | seed_from_yaml() returns int, navigation domain seeded, rules > 0 |

### Stochastic Failures (2)

These two tests compare cosine similarity of random high-dimensional vectors and
occasionally land on the wrong side of the threshold:

| Test                                       | Reason                              |
|--------------------------------------------|-------------------------------------|
| `test_similar_images_produce_similar_hvs`  | Random HV similarity is borderline  |
| `test_similar_tones_produce_similar_hvs`   | Random HV similarity is borderline  |

They are not bugs — they reflect inherent variance in stochastic representations.
Re-running typically clears them.

### Skipped Tests (7)

| Count | Reason                                          |
|-------|-------------------------------------------------|
| 3     | `torch` not installed                           |
| 1     | `WorldModel` archived (module removed)          |
| 1     | Rust cross-backend parity (requires both `.so`) |
| 2     | `hnswlib` fallback path (optional dependency)   |

### Expected Failures — xfail (4)

These document known limitations and are not regressions:

| xfail                          | Why it is expected to fail                       |
|--------------------------------|--------------------------------------------------|
| No gradient learning           | VSA/SNN substrate does not support backprop      |
| No real NLU                    | Language pipeline uses construction grammar, not transformer NLU |
| No rotation invariance         | Spatial encoder does not yet handle rotations    |
| RNG determinism diff           | Minor platform-level RNG divergence across OS/NumPy versions |

---

## 5. Testing Philosophy

1. **Tests must be fast.** The full suite runs in under 15 seconds. If a new
   test takes more than 1 second in isolation, reconsider its scope.

2. **Unit tests are truly isolated.** No network, no filesystem side effects,
   no external services. They import only from `nsck`.

3. **Integration tests exercise real pipelines.** They wire multiple modules
   together and verify end-to-end behavior — not individual functions.

4. **xfail tests document known limitations honestly.** Rather than hiding
   gaps, we mark them with `@pytest.mark.xfail` and explain *why* in the
   test docstring. If an xfail starts passing, pytest will flag it so we can
   promote it.

5. **No mocking of core VSA/SNN operations.** The hypervector and spiking
   neural network primitives are the foundation of the system. Tests must
   exercise the real implementations — mocking them would defeat the purpose.

6. **Regression tests pin bug fixes.** Every non-trivial bug fix should come
   with a test in `regression/` that reproduces the original failure.

---

## 6. Writing New Tests

### Where to put your test

- Testing a single module? → `unit/<subsystem>/`
- Testing how modules interact? → `integration/`
- Fixing a bug? → `regression/test_bug_fixes.py` (or a new file in `regression/`)
- Adding a benchmark? → `benchmarks/`

### Conventions

- File names: `test_<topic>.py`
- Test functions: `test_<what_it_checks>()`
- Use plain `assert` — no need for `self.assertEqual` unless you prefer class-based tests.
- Group related tests in a class prefixed with `Test`:

```python
class TestBeliefRevision:
    def test_contradictory_evidence_triggers_revision(self):
        agent = make_agent()
        agent.believe("sky is green")
        agent.observe("sky is blue")
        assert agent.belief("sky is blue") > agent.belief("sky is green")

    def test_revision_preserves_unrelated_beliefs(self):
        agent = make_agent()
        agent.believe("water is wet")
        agent.observe("sky is blue")
        assert agent.belief("water is wet") > 0.5
```

### Marking special cases

```python
import pytest

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gradient_feature():
    ...

@pytest.mark.xfail(reason="rotation invariance not implemented")
def test_rotated_image_similarity():
    ...
```

### Running a single test during development

```bash
python -m pytest nsck/tests/unit/reasoning/test_belief_revision.py::TestBeliefRevision::test_contradictory_evidence -v
```

---

## 7. CI Integration

The test suite is designed to run in CI without special infrastructure:

- No GPU required.
- No network access needed.
- Rust backends are optional — tests degrade gracefully with skips.
- The full run fits within a 60-second CI timeout with margin to spare.

If CI reports a failure, check the stochastic tests first (Section 4). If those
are the only failures, re-running the job is the correct response.

---

*Last updated for NSCK V4.*

---

## 8. V5 Integration Tests

V5 adds a new integration test file that covers all newly wired modules.

### Running V5 Tests

```bash
# Using make (recommended)
cd nsck && make test-v5

# Direct pytest
python -m pytest nsck/tests/integration/test_v5_end_to_end.py -v --tb=short
```

### V5 Test File

`tests/integration/test_v5_end_to_end.py` — 6 test classes covering all V5 wiring:

| Class | Coverage |
|-------|----------|
| `TestTextDecisionLoop` | Text input → VSANLUEngine → ContextEngine → decide() |
| `TestMultimodalFusion` | Multi-modal input fusion and decision |
| `TestSleepConsolidation` | sleep() → prototypes built, drift detector triggered |
| `TestCrossDomainTransfer` | Cross-domain transfer via analogy engine |
| `TestProceduralFastPath` | ProceduralMemory LSH cache hit |
| `TestFullLifecycle` | 100-cycle full lifecycle: process → feedback → sleep |

### Expected Results

```
nsck/tests/integration/test_v5_end_to_end.py
  6 passed in ~13s
```

### V4 + V5 Combined Run

```bash
# Run both V4 and V5 integration tests
python -m pytest nsck/tests/integration/test_v4_full_system.py \
                 nsck/tests/integration/test_v5_end_to_end.py -v
```

---

*Last updated for NSCK V5.*
