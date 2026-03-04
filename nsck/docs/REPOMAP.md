# NSCK V5 — Repository Navigation Map

> **~96 core Python modules · ~232 classes · ~38K LOC Python · ~4.5K LOC Rust · 1443+ tests**

| Metric | Value |
|--------|-------|
| Core Python modules | ~96 (in `python/core/`, excl. `__init__.py` and tests) |
| Classes | ~232 |
| Python core LOC | ~38,000 |
| Rust LOC | ~4,500 |
| Test files / LOC | 84+ / ~19,000+ |
| Tests | 1443+ (1430+ pass, 2 stochastic, 7 skipped, 4 xfailed) |

---

## Directory Structure

```
nsck/
├── python/core/                  94 modules, ~36K LOC
│   ├── adapters/      (10 files) Input normalization and multimodal fusion
│   ├── cognitive/      (5 files) Higher-order cognition and safety
│   ├── integration/    (5 files) Config, persistence, fusion, explanation
│   ├── language/      (21 files) NLU/NLG pipeline, dialogue, pragmatics
│   │                            V4 new: vsa_nlu.py (VSANLUEngine)
│   ├── learning/      (10 files) Hebbian, curiosity, active inference, rules
│   ├── memory/         (8 files) Semantic, episodic, procedural stores
│   │                            V4 updated: semantic_memory (hot cache LRU),
│   │                            procedural_memory (LSH O(1)), semantic_memory_shim
│   ├── multimodal/     (2 files) Cross-modal processing and image generation
│   ├── perception/     (8 files) SNN, symbol grounding, VSA-SNN bridge
│   ├── reasoning/     (14 files) Central engine, GWT, causal, planning
│   ├── training/       (3 files) VSA/SNN training and benchmarks
│   ├── types/          (2 files) PerceptPacket, ModalityAdapter protocols
│   ├── vsa/            (7 files) HyperVector ops (Python + Rust shim)
│   └── substrate.py              Public API entry point
│   (bootstrap/ V4 new)
│       ├── knowledge_seeder.py   KnowledgeSeeder — YAML domain bootstrapper
│       └── domain_kits/
│           ├── navigation.yaml   Navigation domain kit
│           └── scheduling.yaml   Scheduling domain kit
├── rust_vsa/                     Rust VSA accelerator (PyO3) → hypervec_rs.so (4.3 MB)
├── rust_snn/                     Rust SNN accelerator (PyO3) → snn_rs.so (1.1 MB)
├── api/                          FastAPI REST: /decide, /learn, /sleep, /status
├── tests/                        84+ files across unit/, integration/, core_architecture/,
│   │                             experiments/, regression/, benchmarks/
│   │                             V4 new: tests/integration/test_v4_full_system.py (6 classes)
│   │                             V5 new: tests/integration/test_v5_end_to_end.py (6 classes)
├── benchmarks/                   Benchmark runner + domain benchmarks
│   ├── realworld_harness.py      V5 new: Iris + text classification benchmark harness
│   └── run_realworld.py          V5 new: CLI entry point for real-world benchmarks
├── eval/                         Evaluation harness and end-to-end evals
├── scripts/                      Utility scripts (verify_rust, generate_v11_report)
├── examples/                     Runnable demos (quickstart, custom_module, learn_from_text, demo_snn)
├── data/                         Built-in datasets
├── docs/                         15+ documentation files
│   ├── (V4 docs)
│   ├── QUICKSTART.md             V5 new: 5-minute getting started guide
│   ├── API_REFERENCE.md          V5 new: complete NSCKSubstrate API reference
│   └── V5_CHANGELOG.md           V5 new: V5 changelog by work package
├── pyproject.toml                Project metadata and build config
└── conftest.py                   Shared pytest fixtures
```

---

## Key Entry Points

| Entry Point | File | Role |
|-------------|------|------|
| **Public API** | `python/core/substrate.py` | `NSCKSubstrate` — developer-facing interface |
| **Cognitive Hub** | `python/core/reasoning/cognitive_engine.py` | `CognitiveEngine` — central orchestrator |
| **REST API** | `api/nsck_api.py` | FastAPI server (`/decide`, `/learn`, `/sleep`, `/status`) |

---

## Subsystem Reference

### `adapters/` — 10 files · Input → PerceptPacket normalization

`text_adapter`, `dict_state_adapter`, `numeric_adapter`, `numeric_sequence_adapter`, `snn_adapter`, `multimodal_fuser`, `stream_processor`, `image_adapter` (65-dim FPE), `audio_adapter` (23-dim MFCC), `video_adapter`

### `cognitive/` — 5 files · Higher-order cognition and safety

`emotion_system` (valence/arousal), `metacognition` (safety gate + confidence), `safety_verifier`, `self_model`, `theory_of_mind`

### `integration/` — 5 files · Config, persistence, cross-cutting

`brain_fusion`, `config` (NSCKConfig + feature flags), `explanation`, `knowledge_integration`, `persistence` (BrainStore)

### `language/` — 21 files · Full NLU → dialogue → NLG pipeline

`parser`, `language_module`, `vsa_language_module`, `ngram_nlu`, **`vsa_nlu` (V4 new — VSANLUEngine, 7 intents)**, `fluent_nlg`, `nlg`, `dialogue_manager`, `construction_grammar`, `frame_semantics`, `coreference`, `semantic_roles`, `pragmatics`, `distributional_semantics`, `text_knowledge_learner`, `pos_tagger` (300+ lexicon), `universal_input`, `lingua_cortex`, `control`, `hf_corpus_loader`, `compositional_semantics_backup`

### `learning/` — 10 files · From Hebbian to meta-learning

`hebbian`, `curiosity`, `active_inference`, `rule_neural_scorer`, `conformal_wrapper`, `pattern_generalizer`, `cross_domain`, `cross_modal`, `continual_learning`, `meta_learning`

### `memory/` — 8 files · Three-store architecture + maintenance

`semantic_memory` (NSW/HNSW default-on + **hot cache V4**), `episodic_memory`, `procedural_memory` (**LSH O(1) V4**, threshold 0.72), `cross_modal_associative_memory`, `concept_drift_detector`, `homeostasis`, `staged_recall` (fast → deep), `semantic_memory_shim` (**_rust_step_fn at import V4**)

### `multimodal/` — 2 files

`multimodal_processor` (cross-modal alignment), `image_generator` (VSA-guided)

### `perception/` — 8 files · SNN and symbol grounding

`snn_perception` (LIF layers), `snn_shim` (Rust ↔ Python), `snn_integration` (SNNWorkspaceAdapter), `signal_ingestor`, `symbol_grounding`, `grounding_verifier`, `vsa_snn_bridge` (rate/temporal coding), `stream_encoder`

### `reasoning/` — 14 files · Central engine + specialized reasoners

`cognitive_engine` (**central hub**), `global_workspace` (GWT), `causal_reasoning`, `causal_rule_auditor`, `causal_interface`, `causal_service_impl`, `rule_learner`, `attention_gwt_bridge`, `planner` (STRIPS), `analogy`, `math_reasoning`, `spatial_reasoning` (8 relations), `belief_revision` (AGM), `context_engine`

### `training/` — 3 files

`vsa_trainer`, `snn_training` (STDP), `snn_benchmarks`

### `types/` — 2 files

`percept_packet` (PerceptPacket dataclass), `modality_adapter` (ModalityAdapter protocol)

### `vsa/` — 7 files · Hyperdimensional computing foundation

`hypervec_py` (pure-Python fallback), `hypervec_shim` (Rust ↔ Python selector), `fhrr`, `vsa_embedding_bridge`, `resonator`, `universal_hv_encoder`, `rust_concurrent_shim`

### `substrate.py` — Public API

`NSCKSubstrate` wraps `CognitiveEngine` and exposes `decide()`, `learn()`, `sleep()`, `explain()`.

---

## Subsystem Dependency Graph

```mermaid
graph TD
    API["api/nsck_api.py"] --> SUB["substrate.py<br/>NSCKSubstrate"]
    SUB --> CE["cognitive_engine<br/>CognitiveEngine"]
    CE --> GWT["global_workspace<br/>GlobalWorkspace"]
    CE --> CFG["config"]

    ADP["adapters (10)"] -->|"PerceptPacket"| CE

    CE --> REASON["causal · rules · planner<br/>analogy · spatial · belief<br/>math · context"]
    CE --> MEM["semantic · episodic<br/>procedural memory"]
    CE --> LEARN["hebbian · curiosity<br/>active_inference · meta"]
    CE --> LANG["language_module<br/>dialogue · NLG · NLU"]
    CE --> COG["metacognition · safety<br/>self_model · ToM · emotion"]
    CE --> PERC["snn_perception<br/>grounding_verifier"]
    CE --> INTEG["persistence · brain_fusion<br/>explanation · knowledge_integration"]

    PERC --> BRIDGE["vsa_snn_bridge"]
    ATTN["attention_gwt_bridge"] --> GWT

    CE --> SHIM["hypervec_shim"]
    MEM --> SHIM
    LEARN --> SHIM
    REASON --> SHIM
    BRIDGE --> SHIM
    PERC --> SHIM

    SHIM --> RSVSA["hypervec_rs.so<br/>4.3 MB · PyO3"]
    SHIM -.->|"fallback"| HVPY["hypervec_py"]
    PERC --> RSSNN["snn_rs.so<br/>1.1 MB · PyO3"]
```

---

## Data Flow

```mermaid
graph LR
    subgraph Inputs
        TEXT["Text"]
        DICT["State Dict"]
        IMG["Image / Audio / Video"]
        NUM["Numeric"]
    end

    ADAPT["Adapters (10)<br/>→ PerceptPacket"]
    CE["CognitiveEngine"]
    GWT_B["Global Workspace<br/>Coalition Competition"]

    subgraph MemStore["Memory"]
        SEM["Semantic"]
        EPI["Episodic"]
        PROC["Procedural"]
    end

    subgraph LearnSys["Learning"]
        HEBB["Hebbian"]
        RULES["Rule Induction"]
        AINF["Active Inference"]
    end

    subgraph Out["Outputs"]
        ACT["Action"]
        EXPL["Explanation"]
        CONF["Confidence"]
    end

    TEXT & DICT & IMG & NUM --> ADAPT
    ADAPT -->|"PerceptPacket"| CE
    CE --> GWT_B
    CE <-->|"store / recall"| SEM & EPI & PROC
    CE --> HEBB & RULES & AINF
    GWT_B --> ACT & EXPL & CONF
```

---

## Rust ↔ Python Interface

```mermaid
graph LR
    subgraph Python
        SHIM["hypervec_shim.py"]
        SSHIM["snn_shim.py"]
        MOD["94 core modules"]
    end

    subgraph Compiled["Compiled Extensions"]
        VSA_SO["hypervec_rs.so (4.3 MB)"]
        SNN_SO["snn_rs.so (1.1 MB)"]
    end

    subgraph rust_vsa["rust_vsa/ (~3K LOC)"]
        RV["HyperVector · SemanticMemoryConcurrent<br/>EpisodicMemoryConcurrent · WorkerPool<br/>PersistentStorage · AsyncRuntime"]
    end

    subgraph rust_snn["rust_snn/ (~1.3K LOC)"]
        RS["LIFLayer · StdpEngine · SnnCore<br/>HebbianMatrix · ConceptMapper · RateCoder"]
    end

    MOD --> SHIM & SSHIM
    SHIM --> VSA_SO --> RV
    SSHIM --> SNN_SO --> RS
    SHIM -.->|"fallback"| FP["hypervec_py (pure Python)"]
    SSHIM -.->|"fallback"| FS["snn_perception (pure Python)"]
```

---

## Test Organization

```
tests/                            84 files, ~19K LOC, 1437 tests
├── unit/                         Fine-grained module tests
├── integration/                  Cross-subsystem pipeline tests
├── core_architecture/            Architecture invariant checks
├── experiments/                  Research experiments and validations
├── regression/                   Regression tests for fixed bugs
└── benchmarks/                   Performance benchmarks
```

| Result | Count |
|--------|-------|
| Pass (with Rust) | 1430+ |
| Stochastic | 2 |
| Skipped | 7 |
| xfailed | 4 |
| **Total** | **1443+** |

---

## Quick Reference

```bash
NSCK_USE_RUST=1 pytest                    # run all 1443+ tests (Rust default-on)
pytest tests/integration/test_v4_full_system.py -v  # V4 tests
pytest tests/integration/test_v5_end_to_end.py -v   # V5 tests
pytest tests/unit/reasoning/             # subsystem tests
pytest tests/integration/               # pipeline tests
uvicorn nsck.api.nsck_api:app            # start REST API
python -m nsck.benchmarks.runner         # run benchmarks
make test-v5                             # V5 integration tests
make benchmark-realworld                 # real-world benchmarks
```
