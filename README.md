# NSCK — Neuro-Symbolic Cognitive Kernel

![Project Status](https://img.shields.io/badge/status-Phase_7_Complete-green)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Rust](https://img.shields.io/badge/rust-1.70+_(optional)-orange)
![Tests](https://img.shields.io/badge/tests-178_collected_21+_verified-brightgreen)

An experimental research prototype for exploring neuro-symbolic AI. NSCK combines Vector Symbolic Architecture (VSA) with symbolic rule-based reasoning, neural spiking networks, and cross-domain transfer learning, designed for energy, processing, and memory efficiency. Runs on CPU only — no GPU required.

**Latest Update:** All 7 phases complete. Transfer learning fixed and enhanced with cross-domain knowledge consolidation, LLM translator integration (Phi3), and comprehensive test coverage. See [Architecture](docs/ARCHITECTURE.md).

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
- **Neural components** — spiking neural networks (SNN), multi-task learning with gradient surgery
- **Cognitive scaffolding** — episodic memory, metacognition, world models, self-awareness
- **Transfer learning** — cross-domain knowledge transfer via structural analogy
- **Social intelligence** — emotion processing, Theory of Mind, empathy
- **Language** — Phi3 LLM as peripheral translator (NL ↔ symbolic)

All components are designed to work together through a central cognitive engine, but each can be tested and used independently.

---

## What Actually Works

The following capabilities are implemented and verified by the test suite (178 tests collected, 21+ core tests verified passing with mathematical proofs and execution logs):

### Phase 0: Foundation
| Capability | Description |
|---|---|
| **VSA Core** | XOR binding, bundling, similarity search — all O(n) on 10,240-bit binary HVs |
| **Episodic Memory** | VSA-based experience storage with LSH bucketing |
| **Rule Learning** | Frequency-based symbolic rule induction |
| **Causal Reasoning** | Causal graphs, forward/backward chaining, counterfactuals |
| **Brain Fusion** | Multi-task knowledge organization with concept promotion |
| **Planning** | STRIPS-style planning with spatial navigation |
| **World Model** | Forward simulation with 128-dim sparse projection |

### Phase 1: Neural Learning Engine
| Capability | Description |
|---|---|
| **Multi-Task Learning** | Shared encoder with task-specific heads for Snake, Pong, Maze |
| **Gradient Surgery** | Conflict resolution for multi-task gradients |
| **Rule Extraction** | Symbolic rules extracted from neural policies |
| **Dual Inference** | Combined neural (fast) + symbolic (safe) decision-making |

### Phase 2: Perception Systems
| Capability | Description |
|---|---|
| **Multimodal Processing** | Vision, audio, language all bind to HyperVectors |

### Phase 3: Continual Learning
| Capability | Description |
|---|---|
| **EWC** | Elastic Weight Consolidation for catastrophic forgetting prevention |
| **Progressive Networks** | New columns with lateral connections per task |
| **PackNet** | Pruning & capacity allocation per task |
| **Memory Replay** | Interleaved training with experience buffer |

### Phase 4: World Models & Planning
| Capability | Description |
|---|---|
| **MPC** | Model Predictive Control for forward planning |
| **MCTS** | Monte Carlo Tree Search for exploration |
| **Hierarchical Planning** | Options framework for multi-level planning |

### Phase 5: Self-Model & Metacognition
| Capability | Description |
|---|---|
| **Self-Awareness** | Performance prediction, confidence tracking |
| **Self-Explanation** | Transparent reasoning traces |
| **Self-Improvement** | Autonomous learning rate adjustment |

### Phase 6: Social & Emotional Intelligence
| Capability | Description |
|---|---|
| **Emotion System** | Plutchik+Russell model, drive-based emotions |
| **Theory of Mind** | Agent mental modeling, Sally-Anne test passing |

### Phase 7: Integration & Transfer Learning
| Capability | Description |
|---|---|
| **Cross-Domain Transfer** | Learn in Snake, apply in Pong/Maze via structural analogy |
| **Knowledge Consolidation** | Multi-domain experiences → abstract domain-independent rules |
| **Global Rules** | Domain-independent safety rules apply across all tasks |
| **LLM Translator** | Phi3 translates system state to natural language (peripheral only) |
| **Full Integration** | All phases working as unified cognitive architecture |

---

## Known Limitations

1. **No real language understanding** — the LLM is a translator peripheral, not a reasoning engine
2. **No pixel-level perception** — uses structured game state, not raw images
3. **No gradient-based learning in VSA core** — the VSA layer is not differentiable
4. **Causal discovery needs sufficient observation data** — sparse data yields incomplete graphs
5. **No real-world robustness** — only tested in simple grid-world environments
6. **Transfer limited to structurally similar domains** — requires shared abstract concepts

---

## Project Structure

```
Node_network/
├── nsck-demo/
│   ├── python/           # Core system modules
│   ├── tests/            # Test suite (267+ tests)
│   ├── rust_vsa/         # Rust VSA extension (optional)
│   ├── web/              # Web dashboard assets
│   └── pyproject.toml    # Project metadata
├── docs/
│   ├── ARCHITECTURE.md           # System architecture & workflows
│   ├── PHASE1-7_COMPLETION_REPORT.md  # Phase completion reports
│   ├── IMPLEMENTATION_ROADMAP.md # Technical implementation guide
│   ├── COMPLETE_MODULE_ANALYSIS.md   # Module-by-module analysis
│   └── AGENT_INSTRUCTIONS.md    # Development governance
├── research/             # Research reports
├── ROADMAP_TO_AGI.md     # Long-term development roadmap
├── requirements.txt      # Python dependencies
└── pytest.ini            # Test configuration
```

---

## Key Modules

| Module | Purpose |
|---|---|
| `cognitive_engine.py` | Central orchestrator — integrates all subsystems |
| `analogy.py` | Cross-domain transfer via structural alignment |
| `train_phase7_demo.py` | Integrated system with KnowledgeStore and LLM translator |
| `testing_dashboard.py` | Comprehensive web dashboard for testing & monitoring all capabilities |
| `rule_learner.py` | Frequency-based symbolic rule induction |
| `continual_learning.py` | EWC, PackNet, Progressive Networks, Memory Replay |
| `multi_task_learning.py` | Shared encoder + task heads + gradient surgery |
| `world_model.py` | Forward simulation with sparse projection |
| `language_module.py` | Phi3 LLM translator (peripheral) |
| `emotion_system.py` | Emotion processing (Plutchik+Russell) |
| `theory_of_mind.py` | Agent mental modeling + false belief detection |
| `self_model.py` | Self-awareness and performance tracking |
| `episodic_memory.py` | VSA-based experience storage |
| `semantic_memory.py` | Concept graph with spreading activation |

---

## Installation

### Prerequisites

- **Python 3.11+**
- **4 GB RAM** minimum
- **No GPU required** — all computation runs on CPU
- **Optional:** Rust 1.70+ (for VSA acceleration)
- **Optional:** Phi3 GGUF model (for LLM translator)

### Setup

```bash
git clone https://github.com/shiva2321/Node_network.git
cd Node_network
pip install -r requirements.txt
```

---

## Running Tests

```bash
# Run core tests
python -m pytest nsck-demo/tests/ -v

# Run transfer learning tests specifically
python -m pytest nsck-demo/tests/test_transfer_learning.py -v

# Run testing dashboard tests
python -m pytest nsck-demo/tests/test_testing_dashboard.py -v
```

Some tests require optional dependencies (`torch`, `flask`, `snntorch`). Core tests run without these.

---

## Testing Dashboard

Launch the comprehensive web dashboard for interactive testing of all system capabilities:

```bash
python nsck-demo/python/testing_dashboard.py
# Open http://localhost:5051 in your browser
```

The dashboard provides:
- **💬 Chat & Test** — Submit text samples and chat with the cognitive system, observe reasoning traces and disambiguations
- **🎮 Game Simulations** — Run Snake, Pong, and Maze games simultaneously or sequentially, watch the system play and learn
- **📊 System Monitor** — Real-time emotion tracking, self-model performance, knowledge base browsing
- **📋 Logs & Export** — Full activity log with filtering, export everything to TXT or JSON

See [docs/TESTING_DASHBOARD.md](docs/TESTING_DASHBOARD.md) for detailed documentation.

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
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, workflows, mathematical foundations, and references |
| [docs/FORMULAS_AND_PROOFS.md](docs/FORMULAS_AND_PROOFS.md) | Complete mathematical formulas with derivations and test-backed proofs |
| [docs/RUN_LOGS_AND_EVIDENCE.md](docs/RUN_LOGS_AND_EVIDENCE.md) | Actual test execution logs, performance benchmarks, and concrete evidence |
| [docs/TESTING_DASHBOARD.md](docs/TESTING_DASHBOARD.md) | Testing dashboard usage, API reference, and export formats |
| [ROADMAP_TO_AGI.md](ROADMAP_TO_AGI.md) | Long-term development roadmap (7 phases) |
| [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) | Technical implementation guide |
| [docs/COMPLETE_MODULE_ANALYSIS.md](docs/COMPLETE_MODULE_ANALYSIS.md) | Detailed analysis of all modules |
| [docs/AGENT_INSTRUCTIONS.md](docs/AGENT_INSTRUCTIONS.md) | Development governance rules |
| **[docs/RESEARCH_SYNTHESIS_2026.md](docs/RESEARCH_SYNTHESIS_2026.md)** | **Latest research-backed improvements (2024-2026)** |
| **[docs/IMPLEMENTATION_PLAN_2026.md](docs/IMPLEMENTATION_PLAN_2026.md)** | **Detailed 12-week implementation plan** |
| **[docs/AGENT_TASK_ASSIGNMENTS_2026.md](docs/AGENT_TASK_ASSIGNMENTS_2026.md)** | **Agent task breakdown and assignments** |

---

## License

See repository for license details.
