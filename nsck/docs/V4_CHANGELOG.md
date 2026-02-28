# NSCK V4 Changelog

> **Version 4.0** — February 2026
> 34 files changed, ~2000 lines added

## New Modules

| Module | Path | Purpose |
|---|---|---|
| `VSANLUEngine` | `python/core/language/vsa_nlu.py` | VSA-based intent classifier (replaces NgramNLU) |
| `KnowledgeSeeder` | `python/core/bootstrap/knowledge_seeder.py` | YAML domain bootstrapper |
| `navigation.yaml` | `python/core/bootstrap/domain_kits/navigation.yaml` | Navigation domain kit |
| `scheduling.yaml` | `python/core/bootstrap/domain_kits/scheduling.yaml` | Scheduling domain kit |

## Updated Modules

| Module | Change |
|---|---|
| `memory/procedural_memory.py` | LSH bucket index (16-bit), threshold 0.72 (was 0.85) |
| `memory/semantic_memory.py` | HNSW default-on, `_hot_cache` LRU (256 entries) |
| `memory/semantic_memory_shim.py` | `_rust_step_fn` captured at import; used per spreading-activation step |
| `reasoning/rule_learner.py` | EWC-aware pruning: `gwt_win_count` + `ewc_importance` composite score |
| `reasoning/cognitive_engine.py` | `imagine_rollout()`, planner safety validation, auto-cache skills on positive reward |
| `integration/persistence.py` | `Rule` gains `gwt_win_count: int = 0` and `ewc_importance: float = 0.0` |
| `integration/config.py` | V4 feature flags added |
| `substrate.py` | SNN auto-grounding via `register_concepts_from_memory()` at init |

## Rust Additions (rust_vsa/src/lib.rs)

| Function | Signature | Purpose |
|---|---|---|
| `bundle_hvs` | `(Vec<Vec<u8>>) -> Vec<u8>` | Correct N-vector majority-vote bundle |
| `lsh_bucket` | `(Vec<u64>, u32, u64) -> u32` | LSH bucket for ProceduralMemory |
| `spreading_activation_step` | `(activation, edges, decay, max_frontier) -> Dict` | One step of spreading activation |

## New Tests

| Test File | Classes |
|---|---|
| `tests/integration/test_v4_full_system.py` | TestProceduralFastPath, TestSemanticHotCache, TestNLU, TestBundleMajorityVote, TestImagination, TestKnowledgeSeeder |
| `tests/unit/language/test_vsa_nlu.py` | VSANLUEngine unit tests |
| `tests/unit/bootstrap/test_knowledge_seeder.py` | KnowledgeSeeder unit tests |
| `tests/unit/memory/test_procedural_lsh.py` | LSH index tests |
| `tests/unit/memory/test_semantic_hot_cache.py` | Hot cache tests |

## Bug Fixes

| Bug | Fix |
|---|---|
| `bundle()` used OR-approximation for N>2 vectors | Replaced with mathematically correct majority-vote in `bundle_hvs` |
| Rust shim `spreading_activation_step` not called | `_rust_step_fn` now captured at module import, not lazily per call |
| SNN perception not grounded to semantic concepts | `register_concepts_from_memory()` called in `NSCKSubstrate.__init__()` |

## Breaking Changes

None. All V4 changes are backward-compatible with V3. The procedural memory threshold change (0.85 → 0.72) increases hit rate but does not break existing code.
