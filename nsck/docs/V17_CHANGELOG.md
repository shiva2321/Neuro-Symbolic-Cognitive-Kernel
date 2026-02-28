# NSCK V17 Changelog

## V17.0.0 — Enrichment Layer & Glass-Box Tracing (2026-04)

V17 delivers a dedicated **Enrichment Layer** and **Glass-Box Tracing**, all
**backward-compatible** with V16.

---

## New Modules

### 1. CausalEnricher (`python/core/reasoning/causal_enricher.py`)

Enriches causal triples with semantic context from SemanticMemory.  Supports
multi-hop chain enrichment with exponential strength decay.

| File | Change |
|------|--------|
| `python/core/reasoning/causal_enricher.py` | New `CausalEnricher` class: `enrich()`, `enrich_chain()`, `enrichment_count` |
| `python/core/integration/config.py` | `enable_causal_enrichment`, `causal_enrichment_n_context` flags |

### 2. PerceptualEnricher (`python/core/perception/perceptual_enricher.py`)

Post-processes PerceptPackets with temporal context windowing and HV-based
confidence scoring.

| File | Change |
|------|--------|
| `python/core/perception/perceptual_enricher.py` | New `PerceptualEnricher` class: `enrich()`, `reset()`, `enrich_count` |
| `python/core/integration/config.py` | `enable_perceptual_enrichment`, `perceptual_enricher_window` flags |

### 3. SemanticEnricher (`python/core/memory/semantic_enricher.py`)

Augments SemanticMemory with automatic inverse-relation inference and
co-query frequency tracking.

| File | Change |
|------|--------|
| `python/core/memory/semantic_enricher.py` | New `SemanticEnricher` class: `enrich_concept()`, `enrich_bulk()`, `coquery_stats()` |
| `python/core/integration/config.py` | `enable_semantic_enrichment`, `semantic_enrichment_add_inverses` flags |

### 4. GlassBoxTracer (`python/core/cognitive/glass_box_tracer.py`)

Full step-by-step decision trace with nested span context managers, archived
history, and human-readable formatting.

| File | Change |
|------|--------|
| `python/core/cognitive/glass_box_tracer.py` | New `GlassBoxTracer` class: `begin_decision()`, `end_decision()`, `record()`, `span()`, `format_trace()` |
| `python/core/integration/config.py` | `enable_glass_box_tracer`, `glass_box_max_history` flags |

### 5. CrossModalEnricher (`python/core/memory/crossmodal_enricher.py`)

Anchor-based cross-modal linking with automatic cluster detection.

| File | Change |
|------|--------|
| `python/core/memory/crossmodal_enricher.py` | New `CrossModalEnricher` class: `link_modalities()`, `detect_clusters()`, `summary()` |
| `python/core/integration/config.py` | `enable_crossmodal_enrichment`, `crossmodal_similarity_threshold` flags |

---

## Configuration Changes

```python
# New flags in NSCKConfig
enable_causal_enrichment: bool = False
enable_perceptual_enrichment: bool = False
enable_semantic_enrichment: bool = False
enable_glass_box_tracer: bool = False
enable_crossmodal_enrichment: bool = False
glass_box_max_history: int = 100
causal_enrichment_n_context: int = 3
perceptual_enricher_window: int = 8
semantic_enrichment_add_inverses: bool = True
crossmodal_similarity_threshold: float = 0.7

# New factory method
cfg = NSCKConfig.v17()   # enables all enrichment + glass-box
```

---

## New Tests

| File | Tests | Description |
|------|-------|-------------|
| `tests/unit/reasoning/test_causal_enricher.py` | 9 | CausalEnricher unit tests |
| `tests/unit/perception/test_perceptual_enricher.py` | 7 | PerceptualEnricher unit tests |
| `tests/unit/memory/test_semantic_enricher.py` | 8 | SemanticEnricher unit tests |
| `tests/unit/cognitive/test_glass_box_tracer.py` | 11 | GlassBoxTracer unit tests |
| `tests/unit/memory/test_crossmodal_enricher.py` | 8 | CrossModalEnricher unit tests |
| `tests/integration/test_v17_enrichment_e2e.py` | 7 | End-to-end V17 pipeline tests |
| `tests/benchmarks/test_v17_benchmarks.py` | 6 | pytest-benchmark tests |
| **Total** | **56** | |

---

## Benchmark

| Benchmark | Throughput |
|-----------|-----------|
| CausalEnricher.enrich | > 1,000,000 ops/s |
| CausalEnricher.enrich_chain (4-hop) | > 270,000 ops/s |
| PerceptualEnricher.enrich | > 1,100,000 ops/s |
| SemanticEnricher.enrich_concept | > 1,500,000 ops/s |
| GlassBoxTracer (full decision) | > 160,000 decisions/s |
| CrossModalEnricher.link_modalities | > 990,000 ops/s |

---

## Summary

| Category | Count |
|----------|-------|
| New modules | 5 |
| New config flags | 10 |
| New factory methods | 1 (`v17()`) |
| New test files | 7 |
| New tests | 56 |

### Test Count

- V16 baseline: 1,422 passed, 63 skipped, 3 xfailed
- V17 total: **1,478 passed**, 63 skipped, 3 xfailed

### Migration Guide

- No breaking changes. All V16 APIs remain unchanged.
- All V17 flags default to `False` — no behaviour change unless opted in.
- Use `NSCKConfig.v17()` to enable all enrichment and glass-box tracing.
- New modules are independent; import only what you need.

### Previous Versions

See `nsck/docs/V16_CHANGELOG.md` for V16 changes.
