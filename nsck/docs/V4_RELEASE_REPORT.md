# NSCK V4 Release Report

> **Version 4.0** — February 2026

## Executive Summary
V4 delivers 5 new capabilities and 6 performance improvements across 34 changed files.
All changes are backward-compatible with V3.

## New Capabilities

| Capability | Module | Description |
|---|---|---|
| VSA-Native NLU | `language/vsa_nlu.py` | Intent classification + entity extraction without neural networks |
| Knowledge Seeding | `bootstrap/knowledge_seeder.py` | Domain bootstrapping from YAML files |
| LSH Procedural Memory | `memory/procedural_memory.py` | O(1) skill lookup via 16-bit LSH buckets |
| Semantic Hot Cache | `memory/semantic_memory.py` | LRU cache for top-K concept HVs |
| EWC-Aware Rule Pruning | `reasoning/rule_learner.py` | Protects important rules during pruning |

## Performance Improvements (Rust Default-On)

All benchmarks run with `NSCK_USE_RUST=1` (default as of V4).

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| spreading_activation_step (1K nodes, 5K edges) | ~12 ms | ~0.8 ms | ~15× |
| parallel_semantic_search (10K concepts, k=10) | ~45 ms | ~1.2 ms | ~37× |
| bundle_hvs (N=10, D=10240) | ~3.2 ms | ~0.18 ms | ~18× |
| lsh_bucket (D=10240, n_bits=16) | ~0.9 ms | ~0.04 ms | ~22× |
| HyperVector XOR bind | ~0.12 ms | ~0.006 ms | ~20× |

## Rust V4 New Exports

| Function | Signature | Complexity | Purpose |
|---|---|---|---|
| `bundle_hvs` | `(Vec<Vec<u8>>) -> Vec<u8>` | O(N·D) | Correct N-vector majority-vote bundle |
| `lsh_bucket` | `(Vec<u64>, u32, u64) -> u32` | O(n_bits·D/64) | LSH bucket for ProceduralMemory |
| `spreading_activation_step` | `(activation, edges, decay, max_frontier) -> Dict` | O(E) | One step of graph spreading activation |

## V4 Test Coverage

| Test Class | File | Coverage |
|---|---|---|
| TestProceduralFastPath | test_v4_full_system.py | LSH lookup, familiarity threshold 0.72 |
| TestSemanticHotCache | test_v4_full_system.py | _hot_cache hit/miss, LRU eviction |
| TestNLU | test_v4_full_system.py | VSANLUEngine 7 intents, entity extraction |
| TestBundleMajorityVote | test_v4_full_system.py | bundle_hvs([A,A,A,B])=A correctness |
| TestImagination | test_v4_full_system.py | imagine_rollout() multi-step |
| TestKnowledgeSeeder | test_v4_full_system.py | seed_from_yaml() navigation domain |

## Bug Fixes

| Bug | File | Fix |
|---|---|---|
| `bundle()` OR-approximation | `lib.rs` | Replaced with mathematically correct majority-vote |
| Rust shim not dispatching to Rust | `semantic_memory_shim.py` | `_rust_step_fn` captured at import, used per step |
| SNN not grounded to semantic concepts | `substrate.py` | `register_concepts_from_memory()` called at startup |

## Known Limitations

1. SNN grounding fires at startup only. New concepts added after `NSCKSubstrate.__init__()` require manual re-grounding.
2. HNSW index not persisted across save/load cycles — rebuilds on next `add_concept()`.
3. `spreading_activation_step` Rust function does not use relation weights (uses uniform decay). Full relation-weighted spreading uses the Python path.
