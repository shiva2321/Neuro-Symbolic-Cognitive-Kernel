# NSCK V17 Report — Enrichment Layer & Glass-Box Tracing

> **Version 17 (V17)** — April 2026

## Overview

V17 adds a dedicated **Enrichment Layer** that post-processes outputs from
the Perception, Reasoning, and Memory subsystems to produce richer, more
contextual representations without requiring any external neural-network
dependencies.

A new **GlassBoxTracer** provides full decision-trace observability, making
every NSCK decision auditable step-by-step.

---

## New Capabilities

| Module | Capability |
|--------|-----------|
| `CausalEnricher` | Enrich causal chains with semantic context; multi-hop decay |
| `PerceptualEnricher` | Temporal window confidence scoring for PerceptPackets |
| `SemanticEnricher` | Automatic inverse-relation inference; co-query tracking |
| `GlassBoxTracer` | Full step-by-step decision trace with span context managers |
| `CrossModalEnricher` | Anchor-based cross-modal linking + cluster detection |

---

## New Modules

### `python/core/reasoning/causal_enricher.py`

**CausalEnricher** enriches causal triples by:
- Looking up semantic neighbours of cause/effect concepts
- Producing a `CausalTrace` with context concepts and enrichment steps
- Supporting multi-hop chain enrichment with exponential strength decay

```python
from python.core.reasoning.causal_enricher import CausalEnricher

enricher = CausalEnricher()

# Single causal triple
trace = enricher.enrich("fire", "smoke", strength=0.9)
print(trace.context_concepts)   # [] when no semantic_memory, or semantic hits
print(trace.enrichment_steps)   # ['no_semantic_memory'] or semantic steps

# Multi-hop chain
chains = enricher.enrich_chain(["virus", "infection", "fever", "dehydration"])
for t in chains:
    print(f"  {t.cause} → {t.effect}  strength={t.strength:.2f}")
```

### `python/core/perception/perceptual_enricher.py`

**PerceptualEnricher** adds temporal context and confidence scoring to any
`PerceptPacket`:

```python
from python.core.perception.perceptual_enricher import PerceptualEnricher

enricher = PerceptualEnricher(window_size=8, confidence_threshold=0.6)

# Process multiple packets
for pkt in stream_of_packets:
    ep = enricher.enrich(pkt)
    print(f"  modality={ep.original_modality}  confidence={ep.confidence:.2f}")
    print(f"  temporal_ctx={ep.temporal_ctx_available}  tags={ep.tags}")
```

### `python/core/memory/semantic_enricher.py`

**SemanticEnricher** augments `SemanticMemory` with inverse relations and
co-query frequency tracking:

```python
from python.core.memory.semantic_enricher import SemanticEnricher

enricher = SemanticEnricher(semantic_memory=substrate.semantic_memory)

# Single concept enrichment (adds inverse: animal -sub_class_of-> dog)
report = enricher.enrich_concept("dog", "is_a", "animal")
print(report.inverse_relations_added)   # 1

# Bulk enrichment
triples = [("rain", "causes", "flood"), ("dog", "has_part", "tail")]
report = enricher.enrich_bulk(triples)
print(report.concepts_enriched)         # 2

# Co-query statistics
print(enricher.coquery_stats())
```

### `python/core/cognitive/glass_box_tracer.py`

**GlassBoxTracer** provides a full, structured decision trace:

```python
from python.core.cognitive.glass_box_tracer import GlassBoxTracer

tracer = GlassBoxTracer(max_history=100)
tracer.begin_decision("decision-001")

with tracer.span("perception"):
    tracer.record("TextAdapter", "encoded 'hello world'", confidence=0.95)

with tracer.span("reasoning"):
    tracer.record("CognitiveEngine", "selected rule R42", confidence=0.82)
    tracer.record("CausalEnricher", "enriched fire->smoke chain")

trace = tracer.end_decision()
print(GlassBoxTracer.format_trace(trace))
# === Decision decision-001 ===
#   [perception] TextAdapter: encoded 'hello world' [0.95]
#   [reasoning] CognitiveEngine: selected rule R42 [0.82]
#   [reasoning] CausalEnricher: enriched fire->smoke chain
#   elapsed: 0.1 ms
```

### `python/core/memory/crossmodal_enricher.py`

**CrossModalEnricher** links modality-specific concepts under shared anchors:

```python
from python.core.memory.crossmodal_enricher import CrossModalEnricher

enricher = CrossModalEnricher(similarity_threshold=0.7)

# Link vision + audio concepts under 'dog' anchor
report = enricher.link_modalities("dog", [
    ("vision", "dog_image"),
    ("audio", "dog_bark"),
    ("text", "dog_word"),
])
print(f"anchors: {enricher.anchor_count}, links: {enricher.total_links}")

# Detect cross-modal clusters
clusters = enricher.detect_clusters()
print(f"clusters: {clusters.clusters_detected}")  # 1 (dog in 3 modalities)
```

---

## V17 Config

```python
from python.core.integration.config import NSCKConfig

# Use the V17 preset (all enrichment + glass-box enabled)
cfg = NSCKConfig.v17()

# Or configure individually
cfg = NSCKConfig()
cfg.enable_causal_enrichment = True
cfg.enable_perceptual_enrichment = True
cfg.enable_semantic_enrichment = True
cfg.enable_glass_box_tracer = True
cfg.enable_crossmodal_enrichment = True

# Tune parameters
cfg.causal_enrichment_n_context = 5
cfg.perceptual_enricher_window = 16
cfg.glass_box_max_history = 500
cfg.crossmodal_similarity_threshold = 0.8
```

---

## Benchmarks

Run the V17 capability benchmark:

```bash
python nsck/eval/vsa_capability_benchmark.py
```

Throughput on a modern CPU:

| Benchmark | Throughput |
|-----------|-----------|
| CausalEnricher.enrich | > 1,000,000 ops/s |
| CausalEnricher.enrich_chain (4-hop) | > 270,000 ops/s |
| PerceptualEnricher.enrich | > 1,100,000 ops/s |
| SemanticEnricher.enrich_concept | > 1,500,000 ops/s |
| GlassBoxTracer (full decision) | > 160,000 decisions/s |
| CrossModalEnricher.link_modalities | > 990,000 ops/s |

Run pytest-benchmark tests:

```bash
python -m pytest nsck/tests/benchmarks/test_v17_benchmarks.py -v --benchmark-disable
```

---

## Tests

| File | Tests | Description |
|------|-------|-------------|
| `tests/unit/reasoning/test_causal_enricher.py` | 9 | CausalEnricher unit tests |
| `tests/unit/perception/test_perceptual_enricher.py` | 7 | PerceptualEnricher unit tests |
| `tests/unit/memory/test_semantic_enricher.py` | 8 | SemanticEnricher unit tests |
| `tests/unit/cognitive/test_glass_box_tracer.py` | 11 | GlassBoxTracer unit tests |
| `tests/unit/memory/test_crossmodal_enricher.py` | 8 | CrossModalEnricher unit tests |
| `tests/integration/test_v17_enrichment_e2e.py` | 7 | End-to-end V17 pipeline tests |
| `tests/benchmarks/test_v17_benchmarks.py` | 6 | pytest-benchmark tests |

Total new tests: **56**

---

## Migration Guide

V17 is fully backward-compatible with V16.

- All V17 flags default to `False` — no behaviour change unless opted in.
- Use `NSCKConfig.v17()` to enable all enrichment and tracing.
- New modules are independent; import only what you need.

---

## Changelog

See `V17_CHANGELOG.md` for the full list of file-level changes.
