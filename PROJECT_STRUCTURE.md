# NSCK Project Structure

This document provides a comprehensive overview of the repository organization, architecture, and workflows.

---

## Repository Overview

The Node_network repository contains three major projects built on the NSCK (Neuro-Symbolic Cognitive Kernel) foundation:

```
Node_network/
├── nsck-demo/              # Core NSCK cognitive kernel
├── nsck_ai_model/          # AI assistant with glass-box reasoning
├── nsck_image_gen_project/ # VSA-based image generation
├── docs/                   # Comprehensive documentation
├── tests/                  # Root-level integration tests
└── tools/                  # Launch scripts and utilities
```

---

## Project 1: NSCK Core (`nsck-demo/`)

### Architecture

```
nsck-demo/
├── python/
│   ├── core/               # Core cognitive modules
│   │   ├── hypervector.py          # VSA foundation
│   │   ├── semantic_memory.py      # Concept graph
│   │   ├── episodic_memory.py      # Experience storage
│   │   ├── global_workspace.py     # Information integration
│   │   ├── emotion_system.py       # Affective processing
│   │   ├── causal_reasoner.py      # Causal inference
│   │   ├── self_model.py           # Metacognition
│   │   ├── curiosity_module.py     # Information seeking
│   │   └── multimodal/             # Image understanding
│   ├── learning/           # Learning mechanisms
│   │   ├── text_knowledge_learner.py
│   │   ├── knowledge_consolidator.py
│   │   └── transfer_learning.py
│   └── benchmarks/         # Performance tests
├── rust_vsa/               # Rust-accelerated VSA (6-29× faster)
├── tests/                  # Unit and integration tests
│   ├── unit/               # Module tests
│   ├── integration/        # System tests
│   └── experiments/        # Research experiments
└── web/                    # Dashboard interface
```

### Core Modules

#### 1. HyperVector (VSA Foundation)
- **Type:** 10,240-bit binary vectors
- **Operations:** XOR binding, bundling, similarity
- **Backend:** Rust-accelerated (optional Python fallback)
- **Performance:** O(n) time complexity, constant space

#### 2. Semantic Memory
- **Function:** Concept graph with relations
- **Storage:** 476+ concepts across 8 domains
- **Operations:** Query, inference, abstraction
- **Test Coverage:** 123 tests

#### 3. Episodic Memory
- **Function:** Experience storage and retrieval
- **Capacity:** Unlimited with LSH indexing
- **Operations:** Store, recall, temporal queries
- **Test Coverage:** 89 tests

#### 4. Global Workspace
- **Function:** Information integration and attention
- **Mechanism:** Broadcasting to modules
- **Operations:** Focus, broadcast, competition
- **Test Coverage:** 67 tests

#### 5. Emotion System
- **Function:** Affective state tracking
- **Dimensions:** Valence, arousal, dominance
- **Operations:** Update, decay, influence
- **Test Coverage:** 54 tests

#### 6. Causal Reasoner
- **Function:** Causal inference and prediction
- **Mechanism:** Rule learning and application
- **Operations:** Infer, predict, explain
- **Test Coverage:** 78 tests

### Test Results (NSCK Core)
- **Total Tests:** 581
- **Passed:** 575 (99.0%)
- **Skipped:** 2 (Rust parity)
- **xFailed:** 4 (known limitations)
- **Coverage:** All core modules

---

## Project 2: NSCK AI Model (`nsck_ai_model/`)

### Architecture

```
nsck_ai_model/
├── ai_engine.py            # Main AI engine (glass-box)
├── dashboard.py            # Web API interface
├── train.py                # Training pipeline
├── comprehensive_benchmark.py  # 8-domain testing
├── telemetry_monitor.py    # Real-time monitoring
├── tests/
│   ├── test_ai_engine.py   # Unit tests (77 tests)
│   └── test_production.py  # Production tests (65 tests)
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   └── EVALUATION.md
└── models/                 # Saved model states
```

### Key Features

1. **Glass-Box Reasoning**
   - 11-stage thought traces
   - Explainable decisions
   - Concept activation tracking

2. **Knowledge Learning**
   - VSA-based text learning
   - No hardcoded patterns
   - Relation extraction from data

3. **Response Generation**
   - Context-aware composition
   - Fluent text generation
   - Multi-turn conversation

4. **API Endpoints**
   - `/api/chat` - Conversational interface
   - `/api/train` - Training pipeline
   - `/api/stats` - System metrics
   - `/api/knowledge` - Knowledge graph access

### Test Results (AI Model)
- **Total Tests:** 142
- **Unit Tests:** 77 (100% pass)
- **Production Tests:** 65 (100% pass)
- **Coverage:** All components

### Performance Metrics
- **Throughput:** 6,574 QPS
- **Latency:** 3.43ms average
- **Memory:** 0% growth over time
- **Confidence:** 85.95% average

---

## Project 3: Image Generation (`nsck_image_gen_project/`)

### Architecture

```
nsck_image_gen_project/
├── src/
│   ├── nsck_image_generator.py  # Main generator
│   ├── train_image_gen.py       # Training pipeline
│   └── image_demo.py            # Interactive demo
├── tests/
│   └── test_image_generation.py  # 8 tests
├── examples/                     # Example outputs
├── models/                       # Trained models
├── docs/
│   ├── IMAGE_GENERATION_README.md
│   ├── IMAGE_GENERATION_ARCHITECTURE.md
│   ├── IMAGE_GENERATION_GUIDE.md
│   └── IMAGE_GENERATION_SUMMARY.md
└── FINAL_REPORT.md
```

### Technology

**VSA + Classical CV (No Neural Networks)**
- Hypervector semantic encoding
- HOG (Histogram of Oriented Gradients)
- Color histograms
- LBP (Local Binary Patterns)
- Texture synthesis

### Test Results (Image Generation)
- **Total Tests:** 8
- **Passed:** 8 (100%)
- **Coverage:** All components

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    NSCK Cognitive Architecture               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Semantic   │  │   Episodic   │  │   Working    │      │
│  │    Memory    │  │    Memory    │  │   Memory     │      │
│  │  (Concepts)  │  │ (Episodes)   │  │  (Scratch)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────┬───────┴──────────────────┘              │
│                    │                                         │
│         ┌──────────▼─────────────┐                          │
│         │   Global Workspace     │  ◄── Attention           │
│         │  (Integration Hub)     │                          │
│         └──────────┬─────────────┘                          │
│                    │                                         │
│         ┌──────────┴─────────────┐                          │
│         │                        │                          │
│    ┌────▼────┐            ┌─────▼─────┐                    │
│    │ Emotion │            │  Causal   │                    │
│    │ System  │            │ Reasoner  │                    │
│    └────┬────┘            └─────┬─────┘                    │
│         │                        │                          │
│         └──────────┬─────────────┘                          │
│                    │                                         │
│              ┌─────▼──────┐                                 │
│              │ Self Model │                                 │
│              │(Meta-cogn.)│                                 │
│              └────────────┘                                 │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                  Hypervector Foundation                      │
│              (10,240-bit binary vectors)                     │
│         XOR • Bundle • Permute • Similarity                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Input (Text/Image)
     │
     ▼
┌────────────────┐
│   Encoding     │ ── Hypervector representation
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Memory Storage │ ── Semantic + Episodic
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  Global WS     │ ── Integration + Attention
└────────┬───────┘
         │
         ├──► Emotion System (Affective processing)
         │
         ├──► Causal Reasoner (Inference)
         │
         └──► Self Model (Metacognition)
         │
         ▼
┌────────────────┐
│ Response Gen   │ ── Output composition
└────────┬───────┘
         │
         ▼
Output (Text/Action)
```

---

## Testing Workflow

```
Development
    │
    ▼
┌──────────────────┐
│  Unit Tests      │ ── Individual module validation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Integration Tests│ ── Module interaction validation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Benchmark Tests  │ ── Performance validation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Production Tests │ ── End-to-end validation
└────────┬─────────┘
         │
         ▼
    Deployment
```

### Test Commands

```bash
# NSCK Core Tests
cd nsck-demo
python -m pytest tests/ -v

# AI Model Tests  
cd nsck_ai_model
python -m pytest tests/ -v

# Image Generation Tests
cd nsck_image_gen_project
python -m pytest tests/ -v

# All Tests
python -m pytest . -v
```

---

## CI/CD Workflow

**File:** `.github/workflows/ci.yml`

```yaml
Trigger: Push to main/develop, Pull Requests
    │
    ▼
┌──────────────────────┐
│  Setup Environment   │ ── Python 3.11, 3.12
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Install Dependencies │ ── pip install requirements.txt
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Run Test Suite    │ ── pytest nsck-demo/tests/
└──────────┬───────────┘
           │
           ▼
     ✓ Success / ✗ Failure
```

### CI Test Results
- **Status:** ✅ Passing
- **Python:** 3.11, 3.12
- **Tests:** 581 total
- **Pass Rate:** 99.0%

---

## Documentation Index

### Core Documentation (`docs/`)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design
- **[MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md)** - Complete API documentation
- **[TESTING.md](docs/TESTING.md)** - Test infrastructure and procedures
- **[TEST_RESULTS_SUMMARY.md](docs/TEST_RESULTS_SUMMARY.md)** - Latest test results
- **[FORMULAS_AND_PROOFS.md](docs/FORMULAS_AND_PROOFS.md)** - Mathematical foundations
- **[VSA_THEORY.md](docs/VSA_THEORY.md)** - Vector Symbolic Architecture theory
- **[DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md)** - Dashboard user guide
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Development guidelines

### Project Documentation
- **[NSCK AI Model](nsck_ai_model/README.md)** - AI assistant documentation
- **[Image Generation](nsck_image_gen_project/docs/)** - Image generation guides
- **[Benchmarks](docs/BENCHMARK_RESULTS.md)** - Performance benchmarks

### Quick References
- **[QUICK_TESTING_REFERENCE.md](QUICK_TESTING_REFERENCE.md)** - Quick test commands
- **[TESTING_EVALUATION_GUIDE.md](TESTING_EVALUATION_GUIDE.md)** - Evaluation procedures
- **[CHANGELOG.md](CHANGELOG.md)** - Project history

---

## Quick Start Commands

### Launch Dashboard
```bash
python launch_dashboard.py
# Access: http://localhost:5000
```

### Run AI Model
```bash
cd nsck_ai_model
python -m nsck_ai_model.dashboard --train seed --port 5090
```

### Generate Image
```bash
cd nsck_image_gen_project
python src/train_image_gen.py
python src/image_demo.py
```

### Run Tests
```bash
# All tests
python -m pytest -v

# Specific project
python -m pytest nsck-demo/tests/ -v
python -m pytest nsck_ai_model/tests/ -v
python -m pytest nsck_image_gen_project/tests/ -v
```

---

## Performance Metrics Summary

| Metric | NSCK Core | AI Model | Image Gen |
|--------|-----------|----------|-----------|
| **Tests** | 581 | 142 | 8 |
| **Pass Rate** | 99.0% | 100% | 100% |
| **Throughput** | - | 6,574 QPS | - |
| **Latency** | - | 3.43ms | - |
| **Memory** | Constant | 0% growth | Stable |
| **Speedup** | 6-29× (Rust) | - | - |

---

## Dependencies

### Core Requirements
```
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
rustworkx>=0.14.0
pytest>=7.0.0
```

### Optional Requirements
```
flask>=2.3.0         # For dashboards
rich>=13.0.0         # For terminal UI
psutil>=5.9.0        # For monitoring
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Contributing

See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for development guidelines.

---

## License

See [LICENSE](LICENSE) file for details.

