# NSCK V14 Changelog

> **Release date:** February 2026
> **Baseline:** V13 — 1,282 tests passing
> **V14 result:** 1,311 passed, 152 skipped, 3 xfailed (pure-Python run, no Rust)

V14 is a focused release delivering six work packages (WP-1 through WP-6).
There are no breaking changes to the `NSCKSubstrate` public API.
All new features are off by default (`perception_mode = "pure"`).

---

## Table of Contents

1. [WP-1: Packaging & Cleanup](#wp-1-packaging--cleanup)
2. [WP-2: Rust Spreading Activation Wiring](#wp-2-rust-spreading-activation-wiring)
3. [WP-3: Rich Perception Layer](#wp-3-rich-perception-layer)
4. [WP-4: Hybrid Distillation](#wp-4-hybrid-distillation)
5. [WP-5: Knowledge Packs](#wp-5-knowledge-packs)
6. [WP-6: Scale Validation Benchmarks](#wp-6-scale-validation-benchmarks)
7. [Configuration Changes](#configuration-changes)
8. [New Factory Methods](#new-factory-methods)
9. [Test Suite Changes](#test-suite-changes)
10. [Migration from V13](#migration-from-v13)

---

## WP-1: Packaging & Cleanup

**Files changed:**

| Change | Detail |
|---|---|
| Added `nsck/Makefile` | Developer shortcuts: `make test`, `make build-rust`, `make bench`, `make clean` |
| Deleted backup file | Removed stale `.py.bak` / archive artefact from `nsck/archive/` |

**`nsck/Makefile`** targets:

```makefile
make test        # python3 -m pytest tests/ -q
make build-rust  # cd rust_vsa && cargo build --release
make bench       # python3 eval/bench_spread_activation.py
make clean       # remove __pycache__ and .pyc files
```

---

## WP-2: Rust Spreading Activation Wiring

**New file:** `nsck/python/core/memory/semantic_memory_shim.py`

The existing `SemanticMemory.spread_activation()` method ran entirely in
Python, iterating over a `NetworkX` graph. The Rust backend
(`SemanticMemoryConcurrent`, exposed via `hypervec_rs`) has a thread-safe
DashMap that can be synced from the Python graph for concurrent reads.

`semantic_memory_shim.py` wires these together with the same auto-detect
pattern used by `hypervec_shim.py`:

```python
def spread_activation_fast(
    concept_graph,
    start_concepts: List[str],
    relation_weights: Dict[str, float],
    stigmergy: Dict,
    steps: int = 3,
    decay: float = 0.7,
) -> Optional[Dict[str, float]]:
    ...
```

**Behaviour:**
- If `hypervec_rs.SemanticMemoryConcurrent` is importable, syncs concepts from
  the NetworkX graph to the Rust DashMap and attempts the Rust path.
- Returns `None` if Rust is unavailable or sync fails; the caller falls
  through to the existing Python implementation.
- When Rust exposes a native `spread_activation` RPC in a future release,
  this shim is the integration point.

**New test file:** `nsck/tests/unit/test_rust_spread_activation.py`

---

## WP-3: Rich Perception Layer

Three new adapters providing optional neural bridge perception. All three
are guarded by `perception_mode` and fall back silently when the required
library is not installed.

### `RichTextAdapter`

**File:** `nsck/python/core/adapters/rich_text_adapter.py`

| Mode | Encoding path |
|---|---|
| `bridge` / `hybrid` + `sentence-transformers` available | `all-MiniLM-L6-v2` → `EmbeddingVSABridge` → HV |
| `bridge` / `hybrid` without library | Char-ngram codebook via `EmbeddingVSABridge.encode_text()` |
| `pure` | Original `TextAdapter` char-ngram hashing |

The `adapter_trace` of the resulting `PerceptPacket` includes
`encoding_method` (`"sentence_transformer"` / `"char_ngram"` / `"hash"`)
and `latency_ms`.

### `RichImageAdapter`

**File:** `nsck/python/core/adapters/rich_image_adapter.py`

| Mode | Encoding path |
|---|---|
| `bridge` / `hybrid` + `timm` + `torch` available | `mobilenet_v3_small` (`pretrained=False`) → feature vector → `EmbeddingVSABridge` → HV |
| Otherwise | Delegates to existing `ImageAdapter` (classical CV features) |

The `timm` model is created with `pretrained=False` to avoid network
downloads. Features are structural, not semantic. Pass `pretrained=True` in
`image_bridge_model` if a trained weight file is available.

### `RichAudioAdapter`

**File:** `nsck/python/core/adapters/rich_audio_adapter.py`

| Mode | Encoding path |
|---|---|
| `bridge` / `hybrid` + `whisper` available | `whisper-tiny` mel spectrogram → mean pooling → `EmbeddingVSABridge` → HV |
| Otherwise | Delegates to existing `AudioAdapter` (classical DSP features) |

Model name mapping: `"whisper-tiny"` → Whisper `"tiny"` model string.

### V14 Config Flags (WP-3)

```python
perception_mode: str = "pure"           # "pure" | "bridge" | "hybrid"
text_bridge_model: str = "all-MiniLM-L6-v2"
image_bridge_model: str = "mobilenet_v3_small"
audio_bridge_model: str = "whisper-tiny"
bridge_cache_embeddings: bool = True
bridge_dim: int = 384
```

**New test files:**
- `nsck/tests/unit/test_image_audio_adapters.py`
- `nsck/tests/integration/test_rich_perception_e2e.py`

---

## WP-4: Hybrid Distillation

**New file:** `nsck/python/core/learning/perception_distiller.py`

`PerceptionDistiller` tracks whether the internal (pure VSA) perception is
converging toward the bridge (neural model) perception for each modality.
When the rolling average similarity over the last 100 observations crosses
`distillation_threshold`, the modality is marked as "graduated" — the
system can switch to pure encoding without quality loss.

```python
distiller = PerceptionDistiller(threshold=0.80)
sim = distiller.observe("text", bridge_hv, internal_hv)   # returns float
distiller.is_graduated("text")                             # True when avg >= 0.80
distiller.get_report()                                     # per-modality stats
```

**New config flag:**

```python
distillation_threshold: float = 0.80
```

**Design intent:** In a deployment where `sentence-transformers` is
available during training but must be removed for inference, the distiller
provides a principled criterion for when the pure path is "good enough."

**New test file:** `nsck/tests/unit/learning/test_perception_distiller.py`

---

## WP-5: Knowledge Packs

**New file:** `nsck/python/core/integration/knowledge_pack.py`

**New directory:** `nsck/data/knowledge_packs/`

`KnowledgePack` is a serialisable bundle of domain concepts, relations, and
causal links. Domain experts can author packs without writing Python code.

```python
# Build a pack
pack = KnowledgePack(name="weather")
pack.add_concept("rain", {"type": "weather_event"})
pack.add_concept("cloud", {"type": "atmospheric"})
pack.add_relation("cloud", "precedes", "rain")
pack.add_causal_link("rain", "wet_ground", strength=0.95)
pack.save("nsck/data/knowledge_packs/weather.gz")

# Load and inject
pack = KnowledgePack.load("nsck/data/knowledge_packs/weather.gz")
counts = pack.inject_into(engine)
# counts == {"concepts": 2, "relations": 1, "causal_links": 1}
```

**Storage format:** gzip-compressed pickle. The `save` / `load` API is the
stable interface; the internal pickle format should not be relied on across
major versions.

**`NSCKConfig` integration:**

```python
knowledge_packs: List[str] = field(default_factory=list)
```

When `knowledge_packs` contains file paths, the substrate loads them on
initialisation.

**New test file:** `nsck/tests/unit/test_knowledge_packs.py`

---

## WP-6: Scale Validation Benchmarks

**New file:** `nsck/eval/scale_benchmarks.py`

**New directory:** `nsck/eval/results/`

Measures spreading activation and similarity query latency across four
concept scales (100, 500, 1 000, 5 000 nodes) to validate that NSCK scales
acceptably in the pure-Python path.

```bash
cd nsck && python eval/scale_benchmarks.py
```

Sample output:

```
 Nodes   Spread (ms)   Query (ms)
-----------------------------------
   100          0.45         0.31
   500          2.10         1.87
  1000          4.93         3.72
  5000         24.81        14.56
```

**New integration test:** `nsck/tests/integration/test_scale_validation.py`

---

## Configuration Changes

All new fields in `NSCKConfig` (`python/core/integration/config.py`):

| Field | Type | Default | Description |
|---|---|---|---|
| `perception_mode` | `str` | `"pure"` | `"pure"` / `"bridge"` / `"hybrid"` — controls adapter selection |
| `text_bridge_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for text bridge |
| `image_bridge_model` | `str` | `"mobilenet_v3_small"` | timm model for image bridge |
| `audio_bridge_model` | `str` | `"whisper-tiny"` | Whisper model for audio bridge |
| `bridge_cache_embeddings` | `bool` | `True` | Cache bridge embeddings in memory |
| `bridge_dim` | `int` | `384` | Input dimensionality for `EmbeddingVSABridge` |
| `distillation_threshold` | `float` | `0.80` | Quality threshold for `PerceptionDistiller` graduation |
| `knowledge_packs` | `List[str]` | `[]` | Paths to `.gz` packs loaded on init |

---

## New Factory Methods

Two new class methods added to `NSCKConfig`:

```python
NSCKConfig.rich() -> NSCKConfig
```
Equivalent to `NSCKConfig.research()` but sets `perception_mode="bridge"`.
Use this when optional deep-learning dependencies are available.

```python
NSCKConfig.for_scale(n_concepts: int) -> NSCKConfig
```
Auto-tunes `memory_capacity` and `enable_hnsw_index` for the expected
concept count. For example, `for_scale(10_000)` enables the HNSW index.

---

## Test Suite Changes

| File | Type | What it tests |
|---|---|---|
| `tests/unit/test_rust_spread_activation.py` | Unit | `semantic_memory_shim` Rust path / Python fallback |
| `tests/unit/test_image_audio_adapters.py` | Unit | `RichImageAdapter`, `RichAudioAdapter` pure-mode correctness |
| `tests/unit/test_knowledge_packs.py` | Unit | `KnowledgePack` save/load/inject round-trip |
| `tests/unit/learning/test_perception_distiller.py` | Unit | `PerceptionDistiller` observe, graduation, report |
| `tests/integration/test_rich_perception_e2e.py` | Integration | End-to-end `bridge` and `pure` mode substrate |
| `tests/integration/test_scale_validation.py` | Integration | Scale benchmark thresholds (5 000 concepts) |

V14 adds 29 new tests. V13 baseline: 1,282 passing. V14 result: 1,311 passing,
152 skipped, 3 xfailed. Zero regressions.

---

## Migration from V13

No changes are required for code targeting the V13 API.

- `NSCKConfig` defaults are unchanged (`perception_mode="pure"`).
- All new modules are optional; none are imported unless `perception_mode`
  is set or `knowledge_packs` is populated.
- `NSCKSubstrate` auto-detects new adapters via `register_encoder()`.

To opt in to V14 features:

```python
# Rich perception (needs sentence-transformers or timm/whisper installed)
cfg = NSCKConfig.rich()

# Scale tuning
cfg = NSCKConfig.for_scale(5000)

# Knowledge packs
cfg = NSCKConfig(knowledge_packs=["nsck/data/knowledge_packs/my_domain.gz"])
```

---

*NSCK V14 — February 2026*
