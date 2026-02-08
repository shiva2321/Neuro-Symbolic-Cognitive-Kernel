# NSCK — Neuro-Symbolic Cognitive Kernel

![Project Status](https://img.shields.io/badge/status-Phase_1_Complete-green)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Rust](https://img.shields.io/badge/rust-1.70+_(optional)-orange)
![Tests](https://img.shields.io/badge/tests-230+_passing-brightgreen)

An experimental research prototype for exploring neuro-symbolic AI. NSCK combines Vector Symbolic Architecture (VSA) with symbolic rule-based reasoning and neural spiking networks, designed for energy, processing, and memory efficiency. Runs on CPU only — no GPU required.

**Latest Update:** Phase 1 (Neural Learning Engine) complete! Added multi-task learning with gradient surgery and rule extraction from neural policies. See [Phase 1 Completion Report](docs/PHASE1_COMPLETION_REPORT.md).

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

The following capabilities are implemented and verified by the test suite (230+ tests passing):

### Phase 0 Foundation (Verified)
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
| **RL Engine (A2C/PPO)** | Advantage Actor-Critic and PPO with GAE for neural policy training |
| **Neural-Symbolic Bridge** | Arbitration between neural policy and symbolic rules with safety overrides |
| **SNN Training Pipeline** | Hebbian learning, concept activation mapping (SNN→VSA binding) |
| **Continual Learning** | EWC, PackNet, Progressive Neural Networks, Memory Replay |
| **Meta-Learning** | MAML and Reptile for rapid few-shot task adaptation |
| **Game Environments** | Snake, Pong, Maze (grid-based test environments) |

### Phase 1: Neural Learning Engine (Complete)
| Capability | Description |
|---|---|
| **Multi-Task Learning** | Shared encoder with task-specific heads for Snake, Pong, Maze |
| **Gradient Surgery** | Conflict resolution for multi-task gradients (Yu et al., 2020) |
| **Rule Extraction** | Extract symbolic rules from trained neural policies (ECLAIRE/DeepRED patterns) |
| **Dual Inference** | Combined neural (fast) + symbolic (safe) decision-making with safety overrides |

See [Phase 1 Completion Report](docs/PHASE1_COMPLETION_REPORT.md) for details.

---

## Known Limitations

These are documented, honest limitations of the current system:

1. **No real language understanding** — the system operates on symbols, not semantics
2. **No pixel-level perception** — uses structured game state, not raw images
3. **No gradient-based learning in VSA core** — the VSA layer is not differentiable
4. **Causal discovery needs sufficient observation data** — sparse data yields incomplete graphs
5. **No real-world robustness** — only tested in simple grid-world environments
6. **RL training requires environment interaction** — A2C/PPO need live environment loops to demonstrate full capability

---

## Project Structure

```
Node_network/
├── nsck-demo/
│   ├── python/           # Core system (82 modules, ~22,000 lines)
│   ├── tests/            # Test suite (56 test files, 216+ tests passing)
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
| `rl_engine.py` | A2C/PPO reinforcement learning with neural-symbolic bridge |
| `snn_training_pipeline.py` | SNN training with Hebbian learning and concept activation mapping |
| `continual_learning.py` | EWC, PackNet, Progressive Networks, Memory Replay |
| `meta_learning.py` | MAML and Reptile for few-shot adaptation |
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
| [docs/AGENT_COORDINATION_PLAN.md](docs/AGENT_COORDINATION_PLAN.md) | Multi-agent coordination: roles, work units, guardrails, parallelism |

---

## License

See repository for license details.
