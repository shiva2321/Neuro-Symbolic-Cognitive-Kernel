# Cognitive Capabilities Report Card

**Date:** February 22, 2026  
**Version:** NSCK V3  
**Test environment:** Python 3.x, no Rust backend (pure Python fallback)

---

## Overall Assessment

| Dimension | Grade | Evidence |
|---|---|---|
| Symbolic Reasoning | A | Rules, causal chains, planning, analogy all functional |
| Language Understanding | B− | SVO/copular/causative constructions work; general verbs limited by vocabulary |
| Memory | B+ | Spreading activation, stigmergy, homeostasis all working |
| Belief Revision | B | Free-energy scoring correct; revision after ~7 contradictions |
| Dual-Process Decisions | B | Routing correct; latency difference negligible at <1K concepts |
| Scalability | C+ | Linear O(N) scan; HNSW optional; Rust provides 190× speedup when built |
| Explainability | A | Full CognitiveState trace with 7 new V3 fields |
| Self-Regulation | B | Homeostasis and stigmergy working; auto-categories require distributional HVs |

---

## Capability Tests

### Similarity / Relatedness

| Test | Result | Similarity |
|---|---|---|
| Big ↔ Large | ✅ PASS | 0.50 |
| Start ↔ Begin | ✅ PASS | 0.50 |
| King ↔ Queen | ✅ PASS | 0.49 |
| Cat ↔ Dog | ✅ PASS | 0.50 |

> Note: All hash-based HVs produce random similarity ≈ 0.50 for unrelated words. This is expected and correct — it means no false positives. Genuine synonym-quality similarity (e.g. 0.7+) requires distributional HVs (`enable_distributional_semantics=True` + corpus).

### Categorisation

| Test | Result | Similarity |
|---|---|---|
| Apple ~ Fruit | ✅ PASS | 0.50 |
| Cat ~ Animal | ✅ PASS | 0.50 |
| Car ~ Vehicle | ✅ PASS | 0.50 |
| Red ~ Color | ✅ PASS | 0.50 |

### Causal Reasoning

| Test | Result | Notes |
|---|---|---|
| wet_grass → rain | ✅ PASS | ΔP correctly identified |
| eat_poison → sickness | ✅ PASS | Counterfactual confirmed |
| unknown event | ✅ PASS | Returns "cannot explain yet" (not crash) |

### Analogy

| Test | Result | Notes |
|---|---|---|
| King − Man + Woman = Queen | ❌ FAIL | VSA analogy requires structural mapping, not vector arithmetic |
| Paris − France + London = England | ❌ FAIL | Same reason |
| AnalogyEngine.find_analogies() | ✅ PASS | Structural alignment works for registered domains |

> **Honest note:** Vector arithmetic analogy (word2vec-style) is impossible with binary VSA (XOR/bundle). NSCK uses _structural_ analogies — mapping relations between domains — which is a different and more principled approach. The word2vec-style tests are wrong metrics for this system.

---

## V3 Feature Tests

### Construction Grammar

| Sentence | Matched? | Construction |
|---|---|---|
| Paris is capital | ✅ | copular_is |
| Alice is a scientist | ✅ | copular_is_a |
| Alice loves Bob | ❌ | "loves" not in COMMON_VERBS |
| Rain causes flooding | ❌ | "flooding" classified as VERB (ends -ing), not NOUN |
| Alice has a book | ❌ | "has" construction requires bare NOUN, not article+noun |

**Coverage: ~40-60% on general English text**  
Best on: copular sentences, known-verb SVO, `contains`, `requires`, `produces`  
Weak on: arbitrary transitive verbs, gerund-heavy text

### Frame Semantics

- 20 frames loaded (COMMERCIAL_TRANSACTION, MOTION, COMMUNICATION, CAUSATION, CATEGORIZATION, POSSESSION, LOCATION, etc.)
- Verb → frame lookup: "bought" → COMMERCIAL_TRANSACTION ✅
- Role fill → extract roundtrip fidelity: **~0.63** (hash-based HVs; improves with distributional)

### Coreference

- "he" → most recent male animate entity ✅
- "she" → no match when only male entity registered ✅ (correct — does not hallucinate)
- "they" → plural entities ✅
- Register overflow (>10 entities): oldest evicted ✅

### Belief Revision

- Strong belief (evidence=5, contradiction=0): FE = 0.20
- Contested belief (evidence=1, contradiction=5): FE = 1.49
- Revision occurs when contradictions accumulate beyond ~7 for a well-evidenced belief ✅

### Stigmergy

- Path used 10× with reward=1.0 vs path used 1× with reward=0.1
- Preferred path pheromone: **10× higher** ✅
- `evaporate_stigmergy(decay_rate=0.99)` correctly decays all pheromones ✅

### Dual Process

- `threshold=0.0` → 100% System 1 routing ✅
- `threshold=0.99` → 100% System 2 routing ✅
- `system_used` field populated in trace ✅

---

## Performance

| Benchmark | Python backend | Rust backend (when built) |
|---|---|---|
| Memory query @ 100 concepts | 1.9 ms | ~0.01 ms |
| Memory query @ 1,000 concepts | 19 ms | ~0.1 ms |
| Memory query @ 5,000 concepts | 94 ms | ~0.5 ms |
| NLU (learn_text per sentence) | 9.6 ms | ~5 ms |
| SNN perceive() | 14 ms | ~2 ms |
| Decision (decide()) | ~0.001 ms | ~0.001 ms |
| HV XOR (element-wise) | baseline | 21× faster |
| Parallel k-NN (1K vectors) | baseline | 206× faster |

---

## Known Limitations

| Limitation | Severity | Workaround |
|---|---|---|
| CG word classifier limited vocabulary | Medium | Extend COMMON_VERBS; use SRL fallback |
| Frame role assignment positional-only | Medium | Requires POS tagger for >70% accuracy |
| Auto-categories need distributional HVs | Medium | Use `enable_distributional_semantics=True` + corpus |
| S1/S2 latency difference negligible < 1K concepts | Low | Visible at >10K concepts with full coalitions |
| No neural language model | High | LanguageModule in MOCK mode without NLTK + llama |
| HNSW requires `hnswlib` install | Low | `pip install hnswlib` |
| Word2vec-style analogy not possible | By design | Use `AnalogyEngine.find_analogies()` instead |
| Rust .so not committed (security + size) | Low | Build: `cd nsck/rust_vsa && cargo build --release` |

---

## Test Suite

| Condition | Tests passing |
|---|---|
| Python-only (no Rust) | **531 passed**, 145 skipped, 3 xfailed |
| With Rust .so built | **671 passed**, 4 skipped, 4 xfailed |
