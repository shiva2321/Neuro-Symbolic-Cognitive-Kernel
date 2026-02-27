# NSCK V16 Changelog

## V16.0.0 — Security, Lifelong Learning, Evaluation & Semantic Seeding (2026-03)

V16 delivers four focused initiatives, all **backward-compatible** with V15.

---

## Initiative 1: KnowledgePack Pickle Security (gzip+JSON)

### Motivation

`KnowledgePack.save()` / `.load()` previously serialised data with Python
`pickle` inside a gzip container.  `pickle.loads()` executes arbitrary Python
code, which is a critical security risk when loading packs from untrusted
sources.

### Changes

| File | Change |
|------|--------|
| `python/core/integration/knowledge_pack.py` | `save()` now writes gzip-compressed JSON (`schema_version: 2`). `load()` tries JSON first; falls back to pickle with a `DeprecationWarning` for V14/V15 legacy files. |
| `python/core/vsa/hypervec_shim.py` | `_hv_to_json()` / `_json_to_hv()` helpers added for base64-encoded hypervector serialisation. |

### API

```python
# New format (V16): gzip-compressed JSON
pack.save("biology.kp")       # writes gzip+JSON, schema_version=2
pack = KnowledgePack.load("biology.kp")   # auto-detects format

# Legacy files still load (with DeprecationWarning):
pack = KnowledgePack.load("old_v15_file.kp")   # DeprecationWarning emitted
```

### Schema

```json
{
  "schema_version": 2,
  "name": "...",
  "concepts": [["name", {props}, "hv_base64_or_null"], ...],
  "relations": [["src", "rel", "dst"], ...],
  "causal_links": [["cause", "effect", strength], ...]
}
```

### New Tests

| File | Tests |
|------|-------|
| `tests/unit/test_knowledge_pack_json.py` | 8 tests |

---

## Initiative 2: EWC Wiring into CognitiveEngine

### Motivation

NSCK's symbolic memory (SemanticMemory) never forgets by design.  V16 adds
**Elastic Weight Consolidation (EWC)** as an optional, feature-flagged
mechanism to protect important task knowledge from being overwritten during
multi-task lifelong learning in neural components.

### Changes

| File | Change |
|------|--------|
| `python/core/learning/continual_learning.py` | New `ContinualLearner` class: `register_task()`, `compute_importance()`, `ewc_loss()`, `get_protected_concepts()` |
| `python/core/reasoning/cognitive_engine.py` | `_continual_learner` field; `register_task()`; EWC fields in `decide()` trace (`ewc_protected_concepts`, `ewc_loss`); `sleep()` consolidates tasks |
| `python/core/integration/config.py` | `enable_ewc: bool = False`, `ewc_lambda: float = 100.0`, `ewc_consolidate_interval: int = 10`; `NSCKConfig.research()` enables EWC; `NSCKConfig.minimal()` disables it |

### API

```python
from python.core.integration.config import NSCKConfig
from python.core.reasoning.cognitive_engine import CognitiveEngine

cfg = NSCKConfig()
cfg.enable_ewc = True
cfg.ewc_lambda = 100.0
engine = CognitiveEngine(config=cfg, persistence_path=":memory:")

engine.register_task("task_a")
for step in range(100):
    cs = engine.decide(state, task_tag="task_a")
    engine.learn(state, cs.chosen_action, reward=1.0, task_tag="task_a")

engine.sleep()  # triggers EWC consolidation

# Task B training won't overwrite Task A importance weights
engine.register_task("task_b")
```

### New Tests

| File | Tests |
|------|-------|
| `tests/unit/learning/test_ewc_wiring.py` | 10 tests |
| `tests/integration/test_lifelong_no_forgetting.py` | 2 tests |

---

## Initiative 3: NSCK Evaluation Suite (NSCK-ES)

### Motivation

NSCK lacked a single reproducible benchmark number comparable to established
AI benchmarks.  NSCK-ES defines a **composite score in [0, 1]** across five
cognitive task dimensions.

### Tasks

| ID | Name | Weight | Description |
|----|------|--------|-------------|
| T1 | Semantic QA | 0.30 | 100 QA pairs across 4 categories (direct, property, causal, multi-hop) |
| T2 | Generalization | 0.20 | 5 cross-domain structural transfer scenarios |
| T3 | Lifelong Retention | 0.20 | Measure catastrophic forgetting after sequential task training |
| T4 | Cross-Modal Recall | 0.15 | 10 anchor→concept pairs tested via spreading activation |
| T5 | Causal Reasoning | 0.15 | 20 causal chains (2-hop positive, 3-hop positive, negative) |

**NSCK-ES** = 0.30·T1 + 0.20·T2 + 0.20·T3 + 0.15·T4 + 0.15·T5

### New Modules

| Module | Location |
|--------|----------|
| `NSCKEvalSuite` | `eval/nsck_eval_suite.py` |
| `t1_semantic_qa` | `eval/tasks/t1_semantic_qa.py` |
| `t2_generalization` | `eval/tasks/t2_generalization.py` |
| `t3_lifelong` | `eval/tasks/t3_lifelong.py` |
| `t4_cross_modal` | `eval/tasks/t4_cross_modal.py` |
| `t5_causal_reasoning` | `eval/tasks/t5_causal_reasoning.py` |
| `regression_gate` | `eval/regression_gate.py` |

### API

```python
from eval.nsck_eval_suite import NSCKEvalSuite

suite = NSCKEvalSuite()
results = suite.run_all()
# {'t1': 1.0, 't2': 1.0, 't3': 1.0, 't4': 1.0, 't5': 1.0,
#  'nsck_es': 1.0, 'version': '1.0', 'elapsed_s': ...}

suite.save_report("eval/results/v16_report.json")
```

### New Tests

| File | Tests |
|------|-------|
| `tests/unit/eval/test_nsck_eval_suite.py` | 13 tests |

---

## Initiative 4: BERT + ConceptNet Semantic Seeding

### Motivation

NSCK's semantic memory starts empty.  Bootstrapping it with commonsense
knowledge (ConceptNet) or contextual embeddings (BERT) dramatically reduces
the number of training steps needed to reach competent performance on new tasks.

### New Modules

| Module | Location | Description |
|--------|----------|-------------|
| `ConceptNetLoader` | `python/core/seeding/conceptnet_loader.py` | Parses ConceptNet TSV exports; filters by language and `min_weight`; maps relations to NSCK canonical names; emits a `KnowledgePack` |
| `SemanticSeeder` | `python/core/seeding/semantic_seeder.py` | Injects a `KnowledgePack` into an `NSCKSubstrate`; runs `post_seed_enrich()` for transitive `is_a` closure |
| `BertSeeder` | `python/core/seeding/bert_seeder.py` | Uses the V15 transplant pipeline to absorb BERT embeddings into NSCK HV space (requires `transformers` + PyTorch) |

### Relation Map

| ConceptNet Relation | NSCK Canonical |
|--------------------|----------------|
| `IsA` | `is_a` |
| `Causes` | `causes` |
| `CapableOf` | `capable_of` |
| `UsedFor` | `used_for` |
| `HasPart` | `has_part` |
| `AtLocation` | `at_location` |
| `RelatedTo` | `related_to` |
| `Antonym` | `antonym` |
| `PartOf` | `part_of` |
| `MotivatedByGoal` | `motivated_by` |

### Configuration

```python
cfg = NSCKConfig.seeded()   # new preset
# cfg.enable_seeding = True
# cfg.seed_conceptnet_pack = "data/knowledge_packs/conceptnet_en.kp"
```

### API

```python
from python.core.seeding.conceptnet_loader import ConceptNetLoader
from python.core.seeding.semantic_seeder import SemanticSeeder

# 1. Build a KnowledgePack from ConceptNet TSV
loader = ConceptNetLoader()
pack = loader.load_from_csv("conceptnet_en.csv", min_weight=2.0)
pack.save("data/knowledge_packs/cn_en.kp")

# 2. Seed a substrate
from python.core.substrate import NSCKSubstrate
substrate = NSCKSubstrate(config=NSCKConfig())
seeder = SemanticSeeder()
counts = seeder.seed_from_conceptnet_pack(substrate, "data/knowledge_packs/cn_en.kp")
# {'concepts': 3215, 'relations': 5820, 'causal_links': 412}

seeder.post_seed_enrich(substrate)   # transitive is_a closure

# 3. BERT seeding (optional — requires transformers)
from python.core.seeding.bert_seeder import BertSeeder
bert_seeder = BertSeeder()
bert_seeder.seed(substrate)
```

### New Tests

| File | Tests |
|------|-------|
| `tests/unit/seeding/test_conceptnet_loader.py` | 8 tests |
| `tests/unit/seeding/test_bert_seeder.py` | 1 test |
| `tests/unit/seeding/test_semantic_seeder.py` | 5 tests |
| `tests/integration/test_seeded_substrate.py` | 1 test |

---

## Summary

| Initiative | New Modules | New Tests |
|-----------|-------------|-----------|
| 1. KnowledgePack Pickle Security | 0 new (modifications to existing) | 8 |
| 2. EWC Wiring | 1 (`ContinualLearner`) | 12 |
| 3. NSCK-ES | 7 | 13 |
| 4. Semantic Seeding | 3 | 15 |
| **Total** | **11** | **48** |

### Test Count

- V15 baseline: 1,375 passed
- V16 total: **1,420 passed**, 152 skipped, 3 xfailed

### Migration Guide

- No breaking changes. All V15 APIs remain unchanged.
- Legacy gzip+pickle `.kp` files continue to load with a `DeprecationWarning`; re-save with `.save()` to migrate.
- `enable_ewc` defaults to `False` — existing behaviour is unchanged.
- EWC requires no changes to call sites; add `cfg.enable_ewc = True` to opt in.

### Previous Versions

See `nsck/docs/V15_CHANGELOG.md` for V15 Model Transplantation changes.
