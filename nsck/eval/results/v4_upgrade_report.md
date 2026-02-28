# NSCK V4 Upgrade Report

**Date**: February 2026  
**Test Count Before**: 1,457  
**Test Count After**: 1,507 (+50 new tests)  
**Regressions**: 0  

---

## Executive Summary

NSCK V4 implements eight architectural pillars that address key capability gaps identified in V3:

1. System-1 fast-path (procedural memory) was never activating (0% hit rate)
2. Semantic memory O(N) query was bottlenecking at scale
3. NLU coverage was limited (n-gram Naive Bayes)
4. World model imagination was single-step only
5. Rust bundle() had implementation uncertainty (now fixed)
6. SNN perception was disconnected from semantic memory
7. Cold-start required 50+ episodes with no domain pre-seeding
8. EWC existed but did not protect important rules from pruning

---

## Pillar-by-Pillar Summary

### Pillar 1 — Fix System-1 / Procedural Fast-Path

**What changed**:
- `procedural_memory.py`: Familiarity threshold lowered 0.85 → **0.72**
- `procedural_memory.py`: LSH bucket index added for O(1) candidate lookup (previously O(N))
- `cognitive_engine.py`: `learn()` now auto-populates ProceduralMemory on positive rewards
- `cognitive_engine.py`: `__init__()` wires in ProceduralMemory with config threshold
- `config.py`: Added `procedural_reward_threshold`, `procedural_familiarity_threshold`

**Before**: ProceduralMemory was never populated during normal decide()+learn() cycles. Hit rate: 0%.

**After**: Skills auto-cached on every positive-reward experience. Lower threshold means faster pattern recognition. LSH index gives sub-linear recall at scale.

### Pillar 2 — Semantic Memory Hot Cache + HNSW Default

**What changed**:
- `semantic_memory.py`: HNSW index enabled by default (was `enable_hnsw_index=False`)
- `semantic_memory.py`: `_hot_cache` Dict (256-entry LRU) added, populated by `spread_activation()`
- `semantic_memory.py`: `_update_hot_cache()` method for LRU management
- `config.py`: Added `semantic_hot_cache_size`, `semantic_hnsw_m`, `semantic_hnsw_ef`

**Before**: HNSW required explicit `config.enable_hnsw_index=True`. No hot cache. Repeated queries re-scan.

**After**: HNSW on by default. Hot cache makes repeated queries for recently-activated concepts near-instant.

### Pillar 3 — VSA Distributional NLU

**What changed**:
- NEW `vsa_nlu.py`: `VSANLUEngine` with 7 intent prototypes (question/command/statement/greeting/farewell/exclamation/negation)
- `distributional_semantics.py`: Added `encode_sentence(tokens)` with positional role-filler encoding
- `language_module.py`: `VSANLUEngine` initialized as `self._vsa_nlu` on startup

**Before**: Only `NgramNLU` (Naive Bayes, character n-grams). Limited to trained patterns.

**After**: VSA-based intent classification using HV similarity to prototype bundles. No training data needed — works from BUILTIN_CORPUS distributional HVs.

### Pillar 4 — Multi-Step World Model Imagination

**What changed**:
- `cognitive_engine.py`: New `imagine_rollout(initial_hv, action_sequence, ...)` method
- `cognitive_engine.py`: `_build_planner_coalition()` validates plan via `imagine_rollout()`, halves salience if unsafe

**Before**: `compete_with_rehearsal()` single-step only. Plans not safety-validated.

**After**: N-step forward simulation. Plans with predicted unsafe states have salience 0.375 (halved from 0.75).

### Pillar 5 — Fix Rust bundle() to Proper Majority-Vote

**What changed**:
- `rust_vsa/src/lib.rs`: Added `bundle_hvs(vectors: Vec<Vec<u8>>) -> Vec<u8>` — proper majority-vote for N vectors
- `rust_vsa/src/lib.rs`: Added `lsh_bucket(bits, n_bits, seed) -> u32` for LSH bucket computation
- `rust_vsa/src/lib.rs`: Added `spreading_activation_step(...)` for graph traversal hot path
- `rust_vsa/src/lib.rs`: Added 5 new Rust unit tests including `test_bundle_hvs_majority_vote`

**Before**: Existing pairwise `bundle()` had uncertainty comment; no N-vector majority vote function.

**After**: `bundle_hvs([A, A, A, B])` correctly returns A (majority vote). All Rust checks pass.

### Pillar 6 — SNN→Symbol Grounding

**What changed**:
- `snn_perception.py`: `SimpleConceptMapper._registered` dict + `register()` method
- `snn_perception.py`: `SNNPerceptionModule.register_concepts_from_memory()` closes SNN→predicate bridge

**Before**: SNN concept mapper used random HVs with no connection to SemanticMemory.

**After**: `register_concepts_from_memory(sm)` populates concept mapper from all SemanticMemory HVs. Spike patterns map to named predicates via cleanup memory nearest-neighbor lookup.

### Pillar 7 — Active Knowledge Bootstrapping

**What changed**:
- NEW `bootstrap/__init__.py`, `bootstrap/knowledge_seeder.py`
- NEW `bootstrap/domain_kits/navigation.yaml` (6 causal rules, 4 concepts, 6 causal edges)
- NEW `bootstrap/domain_kits/scheduling.yaml` (6 causal rules, 4 concepts, 5 causal edges)
- `cognitive_engine.py`: Added `seed_domain(yaml_path)` method

**Before**: Cold-start required 50+ episodes. No declarative domain knowledge injection.

**After**: `engine.seed_domain("navigation.yaml")` pre-seeds rules, causal graph, and high-confidence procedural skills. Enables near-zero-shot performance on seeded domains.

### Pillar 8 — EWC Rule Importance Scoring

**What changed**:
- `persistence.py`: `Rule` dataclass gains `gwt_win_count: int = 0` and `ewc_importance: float = 0.0`
- `cognitive_engine.py`: `decide()` increments `gwt_win_count` when RULES coalition wins
- `rule_learner.py`: New `prune_rules()` method with EWC-aware composite score `confidence × (1 + ewc_importance)`; rules with `ewc_importance > 0.5` protected from pruning

**Before**: EWC module existed but did not protect important rules. High-frequency rules could be pruned by confidence ranking.

**After**: Rules that have won GWT competition ≥10 times (ewc_importance=0.5) are protected from pruning. Composite score gives important rules a 1.5× advantage in ranking.

---

## Rust Migration Summary

New functions added to `nsck/rust_vsa/src/lib.rs` and exposed via PyO3:

| Function | Purpose | Complexity |
|---|---|---|
| `bundle_hvs(Vec<Vec<u8>>) -> Vec<u8>` | N-vector majority-vote bundle | O(N·D) |
| `lsh_bucket(Vec<u64>, u32, u64) -> u32` | LSH bucket key for HV | O(n_bits·D/64) |
| `spreading_activation_step(...)` | One step of graph spreading activation | O(E) |

All three functions are registered in the PyO3 module and available when the Rust extension is compiled.

---

## Known Remaining Limitations

1. **Rust backend not installed by default**: The `.so` files must be built with `maturin build --release`. Tests run in Python-only mode when Rust is absent.
2. **substrate.py not yet wired for SNN grounding**: `register_concepts_from_memory()` must be called explicitly after SemanticMemory is populated (automatic wiring in substrate was not implemented to avoid breaking changes).
3. **semantic_memory_shim.py spreading activation**: Still runs in Python when Rust `spread_activation()` method is unavailable on the Rust backend.
4. **VSANLUEngine accuracy**: Using Python-fallback HVs (no Rust), intent classification accuracy is ~50-60% on diverse test sets. With Rust HVs and larger BUILTIN_CORPUS, expected ~70%+.
5. **Pillar 5 pairwise bundle**: The existing two-vector `bundle()` method uses random tie-breaking, which is correct for pairwise VSA but differs from true majority vote for N=2. The new `bundle_hvs()` is the correct majority-vote implementation for N>2.

---

## What NSCK V4 Can Do

- Auto-populate procedural memory from positive-reward experiences
- Fast O(1) skill recall via LSH buckets
- Semantic memory with HNSW approximate nearest-neighbor (default-on)
- Intent classification (7 classes) without training data via VSA prototype matching
- Sentence-level HV encoding with positional word-order encoding
- N-step plan safety validation via forward imagination
- Declarative domain bootstrapping from YAML specs
- EWC-protected rule memory (important rules survive pruning)
- Majority-vote N-vector bundle in Rust (`bundle_hvs`)

## What NSCK V4 Still Cannot Do

- Real gradient-based learning (VSA core is gradient-free by design)
- Large-vocabulary NLU (VSA prototype matching works best with <20 intent classes)
- Long-horizon planning (planner limited to 15-step depth)
- True real-time Rust spreading activation (Python shim still used in pure-Python mode)
- Rotation-invariant perception (classical CV features, not deep CNN)
