# NSCK — Neuro-Symbolic Cognitive Kernel

![Project Status](https://img.shields.io/badge/status-experimental_prototype-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Rust](https://img.shields.io/badge/rust-1.70+_(optional)-orange)
![Tests](https://img.shields.io/badge/tests-165+_passing-brightgreen)

An experimental research prototype for exploring neuro-symbolic AI. NSCK combines Vector Symbolic Architecture (VSA) with symbolic rule-based reasoning and neural spiking networks, designed for energy, processing, and memory efficiency. Runs on CPU only — no GPU required.

> **Note:** This is an active research project, not production software. See [Known Limitations](#known-limitations) for what does not work yet.

---

## Table of Contents

- [What This Is](#what-this-is)
- [What Actually Works](#what-actually-works)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)
- [Key Modules](#key-modules)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Efficiency Design](#efficiency-design)
- [Documentation](#documentation)
- [License](#license)

---

## What This Is

NSCK is a research prototype that explores how neuro-symbolic methods can be combined into a unified cognitive architecture. The core representation uses **10,240-bit binary hypervectors** (Vector Symbolic Architecture), enabling efficient XOR binding, bundling, and similarity search — all in O(n) time.

The system integrates:

- **Symbolic reasoning** — rule induction, causal graphs, STRIPS-style planning
- **Neural components** — spiking neural networks (SNN), intrinsic curiosity modules (ICM)
- **Cognitive scaffolding** — episodic memory, metacognition, world models

All components are designed to work together through a central cognitive engine, but each can be tested and used independently.

---

## What Actually Works

The following capabilities are implemented and verified by the test suite (165+ tests passing):

| Capability | Description |
|---|---|
| **VSA Core** | XOR binding, bundling, similarity search — all O(n) time on 10,240-bit binary hypervectors |
| **Episodic Memory** | Store and retrieve experiences using VSA-based hashing with LSH bucketing |
| **Rule Learning** | Automatic rule induction from experience (frequency-based) |
| **Causal Reasoning** | Causal graph construction, forward/backward chaining, counterfactual reasoning |
| **Brain Fusion** | Multi-task knowledge organization with concept promotion across task boundaries |
| **Metacognition** | Confidence scoring and conflict detection between competing actions |
| **Planning** | STRIPS-style planning with causal reasoner integration, spatial navigation |
| **Intrinsic Motivation** | ICM for curiosity-driven exploration (~22K params, lightweight) |
| **World Model** | Forward simulation with sparse random projection bottleneck (128-dim) |
| **Game Environments** | Snake, Pong, Maze (grid-based test environments) |

---

## Known Limitations

These are documented, honest limitations of the current system:

1. **No real language understanding** — the system operates on symbols, not semantics
2. **No pixel-level perception** — uses structured game state, not raw images
3. **No gradient-based learning in VSA core** — the VSA layer is not differentiable
4. **No continual learning from raw data** — requires pre-extracted predicates
5. **Causal discovery needs sufficient observation data** — sparse data yields incomplete graphs
6. **No emotions, consciousness, or social cognition** — these are long-term research goals, not current features
7. **No real-world robustness** — only tested in simple grid-world environments

---

## Project Structure

```
Node_network/
├── nsck-demo/
│   ├── python/           # Core system (78 modules, ~20,700 lines)
│   ├── tests/            # Test suite (53 test files, 165+ tests passing)
│   ├── rust_vsa/         # Rust VSA extension (optional, Python fallback available)
│   ├── web/              # Web dashboard assets
│   ├── conftest.py       # Test configuration
│   └── pyproject.toml    # Project metadata
├── docs/
│   ├── AGENT_INSTRUCTIONS.md    # Development governance
│   ├── COMPLETE_MODULE_ANALYSIS.md  # Module-by-module analysis
│   ├── IMPLEMENTATION_ROADMAP.md    # Technical implementation guide
│   └── RUST_VSA_ANALYSIS.md    # Rust VSA deep-dive
├── research/             # Research reports and references
├── ROADMAP_TO_AGI.md     # Long-term development roadmap
├── requirements.txt      # Python dependencies
└── pytest.ini            # Test configuration
```

---

## Key Modules

| Module | Purpose |
|---|---|
| `cognitive_engine.py` | Central orchestrator — integrates all subsystems |
| `brain_fusion.py` | Multi-task knowledge organization with VSA concepts |
| `causal_reasoning.py` | Causal graph construction and inference |
| `metacognition.py` | Self-monitoring, confidence scoring, conflict detection |
| `episodic_memory.py` | VSA-based experience storage and retrieval |
| `rule_learner.py` | Frequency-based symbolic rule induction |
| `planner.py` | STRIPS-style goal planning |
| `curiosity.py` | Curiosity-driven exploration using VSA novelty detection |
| `intrinsic_motivation.py` | ICM for prediction-error based curiosity (~22K params) |
| `world_model.py` | Forward simulation with sparse projection (128-dim bottleneck) |
| `universal_encoder.py` | Multi-modal input encoding (visual/temporal/conceptual) |
| `snn_qat.py` | Spiking neural network with ternary quantization (128-dim) |
| `hypervec_shim.py` | VSA backend (Rust extension with Python fallback) |
| `hypervec_py.py` | Pure Python VSA implementation (10,240-bit binary vectors) |

---

## Installation

### Prerequisites

- **Python 3.11+**
- **4 GB RAM** minimum
- **No GPU required** — all computation runs on CPU
- **Optional:** Rust 1.70+ (for VSA acceleration via the Rust extension)

### Setup

```bash
git clone https://github.com/shiva2321/Node_network.git
cd Node_network
pip install -r requirements.txt
```

---

## Running Tests

```bash
python -m pytest nsck-demo/tests/ -v
```

Some tests require optional dependencies (`torch`, `flask`, `snntorch`). Core tests run without these.

---

## Efficiency Design

NSCK is explicitly designed to avoid heavyweight computation:

- **Binary hypervectors:** O(n) XOR/bundle operations instead of O(n²) matrix multiplication
- **Sparse random projection:** 128-dim bottleneck in the world model vs 20,480-dim full state
- **Lightweight ICM:** Pooling + linear layers (~22K params) instead of Conv2d layers (150K+ params)
- **Compact SNN:** 128-dim shared layers with ternary quantization
- **No dense matrix multiplication** in core VSA operations

---

## Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | This file — project overview and quick start |
| [ROADMAP_TO_AGI.md](ROADMAP_TO_AGI.md) | Long-term development roadmap (7 phases) |
| [docs/COMPLETE_MODULE_ANALYSIS.md](docs/COMPLETE_MODULE_ANALYSIS.md) | Detailed analysis of all 78 modules |
| [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) | Technical implementation guide |
| [docs/RUST_VSA_ANALYSIS.md](docs/RUST_VSA_ANALYSIS.md) | Rust VSA performance analysis |
| [docs/AGENT_INSTRUCTIONS.md](docs/AGENT_INSTRUCTIONS.md) | Development governance rules |

---

## License

See repository for license details.
