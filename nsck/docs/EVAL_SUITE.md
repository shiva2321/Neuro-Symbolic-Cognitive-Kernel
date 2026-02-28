# NSCK Evaluation Suite (NSCK-ES)

> **Version 1.1** — updated in NSCK V4

NSCK-ES is the official benchmark for the Neuro-Symbolic Cognitive Kernel.
It produces a single composite score in **[0, 1]** that measures five
orthogonal dimensions of cognitive competence.

---

## Composite Score Formula

```
NSCK-ES = 0.30 · T1  +  0.20 · T2  +  0.20 · T3  +  0.15 · T4  +  0.15 · T5
```

All task scores are clamped to [0, 1] before weighting.

---

## Task Descriptions

### T1 — Semantic QA (weight 0.30)

**What it measures:** Ability to retrieve factual knowledge via spreading
activation across four question categories.

| Category | Sub-weight | n | Description |
|----------|-----------|---|-------------|
| Direct | 0.40 | 25 | Teach `a → b`; query `a`; expect `b` activated at depth 2 |
| Property | 0.30 | 25 | Concept with property; query concept; expect property concept activated at depth 1 |
| Causal | 0.20 | 25 | Teach `cause → effect`; query cause; expect effect activated at depth 2 |
| Multi-hop | 0.10 | 25 | Chain `a → b → c`; query `a`; expect `c` activated at depth 3 |

**Total QA pairs:** 100  
**Scoring:** `T1 = Σ sub_weight · (correct_i / total_i)`

---

### T2 — Generalization (weight 0.20)

**What it measures:** Ability to transfer structural patterns learned in one
domain to a different domain without explicit retraining.

**Protocol:**
1. Teach 5 scenarios in domain A (animals): `a_src –is_a→ a_dst`
2. Teach the same structural pattern in domain B (vehicles): `b_src –is_a→ b_dst`
3. Query `b_src`; expect `b_dst` reachable at depth 2

**Scoring:** `T2 = successes / 5`

---

### T3 — Lifelong Retention (weight 0.20)

**What it measures:** Resistance to catastrophic forgetting when training
sequentially on multiple tasks.

**Protocol:**
1. Teach Task A: 20 concepts (`life_a_0` … `life_a_19`)
2. Measure Task A recall: `recall_before = |known_A| / 20`
3. Teach Task B: 20 different concepts (`life_b_0` … `life_b_19`)
4. Re-measure Task A recall: `recall_after`

**Scoring:**
```
forgetting_ratio = max(0, (recall_before − recall_after) / recall_before)
T3 = 1 − forgetting_ratio
```

`T3 = 1.0` means zero forgetting (expected for NSCK's symbolic SemanticMemory).

---

### T4 — Cross-Modal Recall (weight 0.15)

**What it measures:** Ability to retrieve a concept originally encoded in one
modality when queried through a different anchor.

**Protocol (10 pairs):**
1. Add concept `modal_concept_i` with `type=cross_modal`
2. Add anchor `modal_anchor_i` and relation `anchor –represents→ concept`
3. Spreading activation from anchor at depth 2; expect concept activated

**Scoring:** `T4 = correct / 10`

---

### T5 — Causal Reasoning (weight 0.15)

**What it measures:** Ability to trace multi-hop causal chains and correctly
reject unrelated negative pairs.

**Dataset (20 chains):**

| Type | Count | Description |
|------|-------|-------------|
| 2-hop positive | 8 | `a –causes→ b`; expect `b` reachable |
| 3-hop positive | 7 | `a –causes→ y –causes→ z`; expect `z` reachable |
| Negative | 5 | Isolated concept pairs with no causal link; expect NOT reachable |

**Scoring:** `T5 = (correct_positive + correct_negative) / 20`

---

## Running the Suite

### Quick Start

```python
from eval.nsck_eval_suite import NSCKEvalSuite

suite = NSCKEvalSuite()
results = suite.run_all()
print(f"NSCK-ES: {results['nsck_es']:.3f}")
# NSCK-ES: 1.000
```

### Individual Tasks

```python
from eval.tasks import t1_semantic_qa, t2_generalization, t3_lifelong
from eval.tasks import t4_cross_modal, t5_causal_reasoning

print(t1_semantic_qa.run())   # 1.0
print(t3_lifelong.run())      # 1.0  (symbolic memory never forgets)
```

### Custom Engine

```python
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.integration.config import NSCKConfig

engine = CognitiveEngine(config=NSCKConfig.research(), persistence_path=":memory:")
suite = NSCKEvalSuite(engine=engine)
results = suite.run_all()
```

### Save Report

```python
suite.save_report("eval/results/v16_report.json")
```

Output JSON:
```json
{
  "t1": 1.0,
  "t2": 1.0,
  "t3": 1.0,
  "t4": 1.0,
  "t5": 1.0,
  "nsck_es": 1.0,
  "version": "1.0",
  "elapsed_s": 0.42
}
```

---

## Regression Gate

`eval/regression_gate.py` provides a `check_regression(verbose=False)` function
that asserts NSCK-ES ≥ a stored baseline.  Integrate into CI:

```python
from eval.regression_gate import check_regression

ok = check_regression(verbose=True)
assert ok, "NSCK-ES regression detected!"
```

---

## Score Interpretation

| NSCK-ES | Interpretation |
|---------|----------------|
| 1.00 | Perfect on all five tasks |
| ≥ 0.90 | Excellent — production-ready cognition |
| 0.70–0.90 | Good — minor gaps in generalisation or causal reasoning |
| 0.50–0.70 | Acceptable — significant room for improvement |
| < 0.50 | Poor — investigate SemanticMemory or spreading activation configuration |

---

## Design Notes

- **All tasks use only `SemanticMemory` and spreading activation** — no
  gradient-based components, so scores are deterministic and reproducible.
- **T3 scores 1.0 by design** for the symbolic memory layer; it serves as a
  regression guard against accidental deletion bugs.
- **T5 negative cases** explicitly test false-positive suppression — a concept
  graph must not spuriously activate unrelated nodes.
- Each task accepts an optional `engine` argument for custom configurations.

---

## Version History

| Version | NSCK Release | Notes |
|---------|-------------|-------|
| 1.0 | V16 | Initial release — 5 tasks, composite formula |
| 1.1 | V4 | Rust default-on; V4 Rust benchmark section added |

See `nsck/docs/V4_CHANGELOG.md` for full V4 changes.

---

## V4 Rust vs Python Benchmark

> All benchmarks default to `NSCK_USE_RUST=1` as of V4.

```bash
NSCK_USE_RUST=1 python -m pytest nsck/tests/benchmarks/test_v17_benchmarks.py -v
```

| Operation | Python | Rust | Speedup |
|---|---|---|---|
| spreading_activation_step (1K nodes, 5K edges) | ~12 ms | ~0.8 ms | ~15× |
| bundle_hvs (N=10, D=10240) | ~3.2 ms | ~0.18 ms | ~18× |
| lsh_bucket (D=10240, n_bits=16) | ~0.9 ms | ~0.04 ms | ~22× |
| parallel_semantic_search (10K concepts, k=10) | ~45 ms | ~1.2 ms | ~37× |

Run eval suite with Rust backend:

```python
import os
os.environ["NSCK_USE_RUST"] = "1"   # default in V4

from eval.nsck_eval_suite import NSCKEvalSuite
results = NSCKEvalSuite().run_all()
print(f"NSCK-ES: {results['nsck_es']:.3f}")
```
