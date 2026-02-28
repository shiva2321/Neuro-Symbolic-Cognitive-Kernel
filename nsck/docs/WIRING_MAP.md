# NSCK Wiring Map — V17

This document describes how all major NSCK modules connect in the V17 system.

---

## Top-Level Data Flow

```
External Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Perception Layer                                        │
│  TextAdapter / ImageAdapter / AudioAdapter              │
│      │                                                   │
│      ▼                                                   │
│  PerceptualEnricher (V17) ──► EnrichedPercept           │
└──────────────────┬──────────────────────────────────────┘
                   │  PerceptPacket + EnrichedPercept
                   ▼
┌─────────────────────────────────────────────────────────┐
│  GWT Broadcast (Global Workspace Theory)                │
│  CognitiveBroadcast → coalition HV                      │
└──────────────────┬──────────────────────────────────────┘
                   │  Broadcast HV
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Reasoning Layer                                         │
│  CognitiveEngine.decide()                               │
│      │                                                   │
│      ├── CausalRuleAuditor                              │
│      ├── CausalEnricher (V17) ──► CausalTrace           │
│      └── GlassBoxTracer (V17)  ──► DecisionTrace        │
└──────────────────┬──────────────────────────────────────┘
                   │  DecisionState + CausalTrace
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Memory Layer                                            │
│  SemanticMemory ◄── SemanticEnricher (V17)              │
│  EpisodicMemory                                          │
│  CrossModalAssociativeMemory ◄── CrossModalEnricher (V17)│
└──────────────────┬──────────────────────────────────────┘
                   │  Updated knowledge graph
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Learning Layer                                          │
│  RuleLearner / ContinualLearner (EWC, V16)              │
└─────────────────────────────────────────────────────────┘
```

---

## V17 Enrichment Wiring

### CausalEnricher

```
CognitiveEngine.decide()
    └── causal context lookup
            └── CausalEnricher.enrich(cause, effect)
                    └── SemanticMemory.get_related(cause, n=3)  [optional]
                    └── returns CausalTrace
```

### PerceptualEnricher

```
ModalityAdapter.encode(input)
    └── returns PerceptPacket
            └── PerceptualEnricher.enrich(packet)
                    └── HV confidence scoring (numpy)
                    └── temporal window check (deque)
                    └── returns EnrichedPercept
```

### SemanticEnricher

```
substrate.add_concept(concept, properties)
    └── SemanticEnricher.enrich_concept(concept, relation, target)
            └── SemanticMemory.add_relation(target, inv_rel, concept)  [if add_inverses]
            └── coquery_counts updated
            └── returns EnrichmentReport
```

### CrossModalEnricher

```
CrossModalAssociativeMemory.register_concept(concept, modality)
    └── CrossModalEnricher.link_modalities(anchor, [(modality, concept), ...])
            └── anchor_registry updated
            └── CrossModalAssociativeMemory.register_concept(...)  [optional]
            └── returns CrossModalEnrichmentReport
```

### GlassBoxTracer

```
NSCKSubstrate / CognitiveEngine
    └── tracer.begin_decision(id)
            └── [perception span]
            │       └── tracer.record("PerceptualEnricher", ...)
            └── [reasoning span]
            │       └── tracer.record("CognitiveEngine", ...)
            │       └── tracer.record("CausalEnricher", ...)
            └── [memory span]
            │       └── tracer.record("SemanticEnricher", ...)
            └── tracer.end_decision() → DecisionTrace
```

---

## Module Dependency Graph

```
NSCKConfig
    └── NSCKSubstrate
            ├── SemanticMemory ◄── SemanticEnricher
            ├── EpisodicMemory
            ├── CrossModalAssociativeMemory ◄── CrossModalEnricher
            ├── CognitiveEngine
            │       ├── CausalRuleAuditor
            │       ├── CausalEnricher
            │       └── GlassBoxTracer
            ├── PerceptualEnricher
            └── [V16] ContinualLearner (EWC)
```

---

## Config → Module Mapping

| Config Flag | Module Enabled |
|-------------|---------------|
| `enable_causal_enrichment` | `CausalEnricher` |
| `enable_perceptual_enrichment` | `PerceptualEnricher` |
| `enable_semantic_enrichment` | `SemanticEnricher` |
| `enable_glass_box_tracer` | `GlassBoxTracer` |
| `enable_crossmodal_enrichment` | `CrossModalEnricher` |
| `enable_ewc` (V16) | `ContinualLearner` |
| `enable_seeding` (V16) | `SemanticSeeder` |
| `enable_hnsw_index` | HNSW in `SemanticMemory` |
| `perception_mode="bridge"` | Bridge adapters |

---

*Updated for NSCK V17, April 2026.*
