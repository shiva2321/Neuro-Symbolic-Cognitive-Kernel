# NSCK V5 API Reference

Complete reference for `NSCKSubstrate` — the primary public interface for NSCK.

Source: `nsck/python/core/substrate.py`

---

## Table of Contents

- [NSCKSubstrate](#nscksubstrate)
- [SubstrateResult](#substrateresult)
- [NSCKConfig](#nsckconfig)
- [Examples](#examples)

---

## NSCKSubstrate

```python
from python.core.substrate import NSCKSubstrate
from python.core.integration.config import NSCKConfig

substrate = NSCKSubstrate(config=None)
```

Top-level orchestrator. Wires perception, memory, reasoning, and learning into
a single pipeline.

### Constructor

```python
NSCKSubstrate(config: Optional[NSCKConfig] = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `NSCKConfig \| None` | `None` | Configuration object. If None, uses `NSCKConfig()` defaults. |

---

### `register_task`

```python
substrate.register_task(task_tag: str) -> None
```

Register a new task/domain. Must be called before `process()` or `feedback()` for a new task (though `process()` will auto-register if needed).

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_tag` | `str` | Unique identifier for the task/domain (e.g. `"navigation"`, `"qa"`). |

---

### `process`

```python
substrate.process(
    input_data: Any,
    task_tag: str,
    available_actions: Optional[List[str]] = None,
) -> SubstrateResult
```

Full perceive → reason → act cycle. Accepts any input type.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_data` | `str \| dict \| np.ndarray \| list \| PerceptPacket` | Input to process. |
| `task_tag` | `str` | Task context. Auto-registered if new. |
| `available_actions` | `List[str] \| None` | Optional whitelist of valid actions. |

**Returns:** `SubstrateResult`

**Input routing:**
- `str` → text adapter (VSANLUEngine intent classification)
- `dict` → dict state adapter (predicate extraction)
- `np.ndarray` (2D/3D) → image adapter (FPE encoding)
- `list` or `np.ndarray` (1D) → numeric sequence adapter
- `PerceptPacket` → passed directly to engine

---

### `process_multimodal`

```python
substrate.process_multimodal(
    inputs: Dict[str, Any],
    task_tag: str,
    available_actions: Optional[List[str]] = None,
) -> SubstrateResult
```

Process multiple input modalities simultaneously, fusing them before reasoning.

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `Dict[str, Any]` | Map of modality name → data (e.g. `{"text": "...", "image": array}`). |
| `task_tag` | `str` | Task context. |
| `available_actions` | `List[str] \| None` | Optional action whitelist. |

**Returns:** `SubstrateResult`

---

### `ingest`

```python
substrate.ingest(
    input_data: Any,
    task_tag: str,
    available_actions: Optional[List[str]] = None,
) -> SubstrateResult
```

V13 API — encode and decide with procedural fast-path. Checks `ProceduralMemory`
first; if a cached skill matches the encoded context (similarity ≥ 0.72), returns
immediately without full deliberation.

Equivalent to `process()` with procedural pre-check.

---

### `feedback`

```python
substrate.feedback(
    action: str,
    reward: float,
    task_tag: str,
    state: Optional[Any] = None,
    outcome: str = "neutral",
) -> None
```

Record an outcome, update Q-learning, Hebbian weights, and skill cache.

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | `str` | The action that was taken. |
| `reward` | `float` | Reward signal (`1.0` = positive, `0.0` = neutral, `-1.0` = negative). |
| `task_tag` | `str` | Task context. |
| `state` | `Any \| None` | The state in which the action was taken. Used for learning. |
| `outcome` | `str` | String outcome label (e.g. `"success"`, `"failure"`, `"neutral"`). |

---

### `learn`

```python
substrate.learn(
    state: Any,
    action: str,
    reward: float,
    task_tag: str,
    outcome: str = "neutral",
) -> None
```

Trigger online learning from a `(state, action, reward)` experience. Calls Q-table
update, `EpisodicMemory.store()`, `RuleLearner.observe()`, and (V5) `EmotionSystem.update_from_drives()`.

---

### `sleep`

```python
substrate.sleep(task_tag: Optional[str] = None) -> Dict[str, Any]
```

Trigger offline consolidation:
1. `PatternGeneralizer.generalize()` — cluster HVs into prototypes (L2 normalized, V5)
2. `ContinualLearner.consolidate_task()` — EWC consolidation (V5)
3. `DriftDetector.check()` — snapshot top-50 concept HVs (V5)

**Returns:** `{"sleep_cycles": int}`

---

### `remember`

```python
substrate.remember(
    query: Any,
    task_tag: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict]
```

Recall similar past experiences from episodic memory by HV similarity.

**Returns:** List of dicts with keys: `task`, `action`, `outcome`, `reward`, `timestamp`.

---

### `register_encoder`

```python
substrate.register_encoder(modality_name: str, encoder_fn: Callable) -> None
```

Register a custom encoder for a new input modality.

| Parameter | Type | Description |
|-----------|------|-------------|
| `modality_name` | `str` | Modality name (e.g. `"lidar"`, `"eeg"`). |
| `encoder_fn` | `Callable[[Any, str], PerceptPacket]` | Function that encodes data to `PerceptPacket`. |

**Example:**
```python
def my_lidar_encoder(data, task_tag):
    from python.core.types.percept_packet import PerceptPacket
    import python.core.vsa.hypervec_shim as hv_mod
    hv = hv_mod.HyperVector(hash(str(data)) % 2**32)
    return PerceptPacket.make(modality="lidar", situation_hv=hv, active_predicates=frozenset())

substrate.register_encoder("lidar", my_lidar_encoder)
```

---

### `get_knowledge`

```python
substrate.get_knowledge(concept: str) -> Dict[str, Any]
```

Query the semantic knowledge graph for a concept.

**Returns:** `{"concept": str, "known": bool, "similar": List[str]}`

---

### `get_stats`

```python
substrate.get_stats() -> Dict[str, Any]
```

Return comprehensive system statistics including engine stats, registered tasks,
decision counter, cross-modal stats, and drift detector state.

---

### `transplant`

```python
substrate.transplant(
    model,
    domain_name: str,
    strategy: str = "lsa",
    calibration_epochs: int = 5,
    save_pack_path: Optional[str] = None,
    cognitive_engine=None,
) -> TransplantReport
```

V15 feature: absorb knowledge from a pre-trained neural model into NSCK's
semantic memory via the Harvest → Project → Calibrate → Validate → Integrate pipeline.

---

### `engine` (property)

```python
substrate.engine  # → CognitiveEngine
```

Expose the underlying `CognitiveEngine` for advanced use.

---

## SubstrateResult

```python
@dataclass
class SubstrateResult:
    chosen_action: str
    confidence: float
    explanation: str
    predicates: Set[str]
    trace: Dict[str, Any]
    modalities_processed: List[str]
    generalization_triggered: bool
    kle_uncertainty: Optional[float] = None
    uncertainty_bounds: Optional[tuple] = None
    encoding_stats: Optional[Dict[str, Any]] = None
    procedural_hit: bool = False
    societal_context: Optional[Dict[str, Any]] = None
```

| Field | Type | Description |
|-------|------|-------------|
| `chosen_action` | `str` | The selected action. |
| `confidence` | `float` | Confidence in [0, 1]. |
| `explanation` | `str` | Human-readable explanation of why this action was chosen. |
| `predicates` | `Set[str]` | Active predicates extracted from the input. |
| `trace` | `Dict[str, Any]` | Machine-readable decision trace. |
| `modalities_processed` | `List[str]` | Which modalities were used. |
| `generalization_triggered` | `bool` | Whether `PatternGeneralizer` fired. |
| `kle_uncertainty` | `float \| None` | KL-divergence uncertainty estimate. |
| `uncertainty_bounds` | `tuple \| None` | Conformal prediction interval. |
| `encoding_stats` | `Dict \| None` | Per-modality encoding statistics. |
| `procedural_hit` | `bool` | True if result came from procedural cache (fast-path). |
| `societal_context` | `Dict \| None` | V26 societal context if enabled. |

---

## NSCKConfig

```python
from python.core.integration.config import NSCKConfig

config = NSCKConfig()                    # defaults
config = NSCKConfig.transplant()         # transplant mode
config = NSCKConfig(enable_ewc=True)     # custom flags
```

Key configuration flags:

| Flag | Default | Description |
|------|---------|-------------|
| `enable_ewc` | `True` | Enable Elastic Weight Consolidation |
| `enable_seeding` | `False` | Enable `KnowledgeSeeder` at startup |
| `enable_hnsw_index` | `True` | HNSW index for semantic memory |
| `enable_causal_enrichment` | `False` | `CausalEnricher` enrichment |
| `enable_glass_box_tracer` | `False` | `GlassBoxTracer` decision tracing |
| `perception_mode` | `"pure"` | `"pure"` / `"bridge"` / `"hybrid"` |
| `knowledge_packs` | `[]` | List of knowledge pack paths to load at startup |

---

## Examples

### Minimal decision loop

```python
from python.core.substrate import NSCKSubstrate

substrate = NSCKSubstrate()
result = substrate.process({"text": "move to goal"}, "nav")
print(result.chosen_action, result.confidence)
```

### Full lifecycle with feedback

```python
from python.core.substrate import NSCKSubstrate

substrate = NSCKSubstrate()
substrate.register_task("robot")

for i in range(50):
    state = {"x": i % 10, "y": i // 10, "obstacle": i % 7 == 0}
    result = substrate.process(state, "robot")
    reward = 1.0 if not state["obstacle"] else -0.5
    substrate.feedback(result.chosen_action, reward, "robot", state)

substrate.sleep("robot")
stats = substrate.get_stats()
print(f"Rules induced: {stats.get('rules_induced', 0)}")
```

### Custom encoder

```python
import numpy as np
from python.core.substrate import NSCKSubstrate
from python.core.types.percept_packet import PerceptPacket
import python.core.vsa.hypervec_shim as hv_mod

substrate = NSCKSubstrate()

def temperature_encoder(data, task_tag):
    seed = int(data * 1000) % (2**32)
    hv = hv_mod.HyperVector(seed)
    predicates = frozenset(["hot"] if data > 37 else ["normal"])
    return PerceptPacket.make("temperature", hv, predicates)

substrate.register_encoder("temperature", temperature_encoder)
result = substrate.process_multimodal(
    {"temperature": 38.5, "text": "patient has fever"},
    "medical"
)
print(result.chosen_action)
```
