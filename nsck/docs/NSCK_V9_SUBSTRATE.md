# NSCK V9 — Modality-Agnostic Cognitive Substrate

## Overview

V9 transforms NSCK from a collection of cognitive modules into a **true modality-agnostic cognitive substrate** — inherently capable of learning, reasoning, remembrance, recall, generalisation, cross-domain transfer, and lifelong learning for **any input type**.

### Core Constraint

> **No neural nets in the understanding/reasoning/memory/decision core.**
> Neural components (SNN) are only allowed as optional perception adapters at the edge.
> The core remains purely VSA + symbolic.

---

## What Changed in V9

### STEP 1 — Universal PerceptPacket Contract

Every input to the cognitive engine is now normalised into a **PerceptPacket** before the reasoning core sees it. This decouples perception from cognition.

```
raw input (any modality)
        ↓
  ModalityAdapter.encode()
        ↓
  PerceptPacket  ←── frozen dataclass
        ↓
  CognitiveEngine.decide()
```

**PerceptPacket fields:**

| Field | Type | Description |
|---|---|---|
| `modality` | str | "text", "numeric", "dict", "snn", "multimodal", "stream" |
| `timestamp` | float | Unix timestamp |
| `situation_hv` | HyperVector | Bundled summary HV |
| `entity_hvs` | dict | Named concept → HV |
| `relation_hvs` | list | (subj, pred, obj, triple_hv) tuples |
| `active_predicates` | frozenset | Grounded symbolic predicates |
| `confidence` | float | 0.0–1.0 encoding confidence |
| `raw_state` | dict\|None | Original state for backward compat |
| `adapter_name` | str | Which adapter produced this |
| `adapter_trace` | dict | Glass-box adapter metadata |

**Backward compatibility:** `decide(state_dict, task_tag)` still works unchanged — dicts are auto-wrapped via `DictStateAdapter`.

**New adapters:**

| Class | File | Wraps |
|---|---|---|
| `DictStateAdapter` | `adapters/dict_state_adapter.py` | `GroundingVerifier + EpisodicMemory.create_situation_hv()` |
| `TextAdapter` | `adapters/text_adapter.py` | `UniversalInput.ground_text()` |
| `NumericAdapter` | `adapters/numeric_adapter.py` | `UniversalInput.ground_scalar/sequence()` |
| `SNNAdapter` | `adapters/snn_adapter.py` | `SNNPerceptionModule.perceive()` |

### STEP 2 — Automatic Generalisation in sleep()

`sleep()` now **automatically invokes** generalisation steps that were previously only callable manually:

1. **Prototype building** — `SemanticMemory.build_prototypes(min_members=2)` — builds VSA prototype HVs for every category in semantic memory.
2. **Transitive inference** — `infer_transitive("is_a", max_hops=3)` + `infer_transitive("causes", max_hops=2)` — closes transitive chains.
3. **Cross-task auto-abstraction** — for every pair of tasks being consolidated, `AnalogyEngine.auto_discover_abstractions()` is called to find structural alignments.

New stats tracked: `prototypes_built`, `transitive_inferences`, `auto_abstractions`.

### STEP 3 — Automatic Cross-Domain Transfer on register_task()

When `NSCKConfig.enable_auto_transfer = True` (default), calling `register_task(new_tag)` automatically:
1. Collects concept HVs from all existing tasks with learned rules.
2. Calls `AnalogyEngine.auto_discover_abstractions()` between each existing task and the new task.
3. Calls `AnalogyEngine.transfer_rule()` for each learned rule.
4. Injects translated rules into `RuleLearner` via new `add_transferred_rule()` method.

New stat tracked: `auto_transfers`.

### STEP 4 — Multimodal Fusion

`MultimodalFuser.fuse(packets)` combines a list of `PerceptPacket` objects:
- VSA bundle of all situation HVs
- Merge entity HVs (same key → bundle)
- Union all predicates
- Average confidence

`CognitiveEngine.decide_multimodal(inputs, task_tag)` is a convenience method that encodes each input and fuses them automatically.

### STEP 5 — Lifelong Stability

Three additions to strengthen long-running stability:

1. **Rule drift tracking** — `Rule` dataclass now has `confidence_history: List[float]`, `last_fired: float`, `fire_count: int`. `observe()` appends to `confidence_history` every 10 observations.

2. **Drift detection** — `_detect_rule_drift()` is called during `sleep()`. If a rule's recent confidence average falls below 50% of its older average, it is marked with `source="drifting:..."`.

3. **Unused rule pruning** — `MemoryHomeostasis.prune_unused_rules()` removes rules with `fire_count == 0` that are old enough (past the idle cutoff). Wired into `sleep()`.

### STEP 6 — Evaluation Harness

`nsck/eval/substrate_benchmarks.py` provides four benchmark functions:

```python
from eval.substrate_benchmarks import (
    benchmark_learning_curve,
    benchmark_transfer,
    benchmark_lifelong,
    benchmark_efficiency,
)
```

| Function | Returns |
|---|---|
| `benchmark_learning_curve(engine, task, gen, n_episodes)` | `[(episode, success_rate)]` |
| `benchmark_transfer(engine, src, tgt, gen, ...)` | `{zero_shot_rate, few_shot_rate, delta}` |
| `benchmark_lifelong(engine, task_seq, gens, ...)` | `{peak_rates, final_rates, forgetting_ratio}` |
| `benchmark_efficiency(engine, task, gen, n)` | `{avg_ms, p95_ms, max_ms}` |

### STEP 7 — Stream Processing

`StreamProcessor` ingests timestamped sensor readings and emits `PerceptPacket` objects with temporal features:

```python
sp = StreamProcessor(window_size=10)
sp.ingest("temperature", 23.5, timestamp)
if sp.ready():
    packet = sp.emit(adapter, task_tag)
    engine.decide(packet, task_tag)
```

Temporal features computed per channel:
- `mean`, `min`, `max`
- `trend` (linear coefficient)
- `rate` (rate of change per second)
- `anomaly` (> 2σ from mean)

`StreamVerifier` is a `GroundingVerifier` subclass that auto-generates symbolic predicates:
- `RISING_X` — positive trend on channel X
- `FALLING_X` — negative trend on channel X
- `ANOMALY_X` — anomalous value on channel X

---

## New Config Flags

| Flag | Default | Description |
|---|---|---|
| `enable_auto_transfer` | `True` | Auto-transfer rules on `register_task()` |

---

## New Public APIs

### CognitiveEngine

```python
# STEP 1: decide() now accepts PerceptPacket OR dict (backward-compatible)
cs = engine.decide(state_dict, task_tag)          # existing — unchanged
cs = engine.decide(percept_packet, task_tag)       # new V9 path

# STEP 4: multimodal convenience
cs = engine.decide_multimodal([input1, input2], task_tag)

# STEP 1: register adapter with task
engine.register_task(task_tag, verifier=v, adapter=my_adapter)
```

### PerceptPacket

```python
from python.core.types.percept_packet import PerceptPacket

packet = PerceptPacket.make(
    modality="text",
    situation_hv=hv,
    active_predicates=frozenset(["A"]),
    confidence=0.9,
    adapter_name="MyAdapter",
)
```

### RuleLearner

```python
# STEP 3: inject a transferred rule
rule_learner.add_transferred_rule(
    task_tag="target",
    condition=frozenset(["PRED_A"]),
    consequence="ACTION_X",
    source_task="source",
    source_confidence=0.8,
)
```

### MemoryHomeostasis

```python
# STEP 5: prune unused rules
homeostasis.prune_unused_rules(rule_learner, max_idle_episodes=200)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Any Modality Input                     │
│  (dict, text, number, ndarray, sensor stream, multimodal) │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │     ModalityAdapter      │  ← STEP 1
           │  (dict/text/num/snn/...)  │
           └──────────────┬───────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │       PerceptPacket      │  ← Universal contract
           │  situation_hv            │
           │  active_predicates       │
           │  entity_hvs              │
           │  confidence              │
           └──────────────┬───────────┘
                          │
                ┌─────────┴──────────┐
                │                    │
                ▼                    ▼
   ┌──────────────────┐   ┌─────────────────────┐
   │  MultimodalFuser │   │  CognitiveEngine     │
   │  (STEP 4)        │   │  decide() / learn()  │
   └────────┬─────────┘   └──────────┬──────────┘
            └────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │         sleep()          │  ← STEP 2: generalization
           │  build_prototypes()      │
           │  infer_transitive()      │
           │  auto_discover_abstractions()
           │  _detect_rule_drift()    │  ← STEP 5
           │  prune_unused_rules()    │
           └──────────────────────────┘
```

---

## File Index

| File | Purpose |
|---|---|
| `python/core/types/percept_packet.py` | PerceptPacket frozen dataclass |
| `python/core/types/modality_adapter.py` | ModalityAdapter ABC |
| `python/core/adapters/dict_state_adapter.py` | Dict → PerceptPacket |
| `python/core/adapters/text_adapter.py` | Text → PerceptPacket |
| `python/core/adapters/numeric_adapter.py` | Numeric → PerceptPacket |
| `python/core/adapters/snn_adapter.py` | SNN → PerceptPacket |
| `python/core/adapters/multimodal_fuser.py` | Fuse multiple PerceptPackets |
| `python/core/adapters/stream_processor.py` | Stream → PerceptPacket + StreamVerifier |
| `eval/substrate_benchmarks.py` | Evaluation harness |

---

## Testing

New integration tests: `tests/integration/test_v9_substrate.py` (32 tests).

All 967 V1–V8 tests continue to pass unchanged.
