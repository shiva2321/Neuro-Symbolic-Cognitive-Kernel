# NSCK — Neuro-Symbolic Cognitive Kernel

![Project Status](https://img.shields.io/badge/status-Phase_8_Complete-green)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Rust](https://img.shields.io/badge/rust-1.70+_(optional)-orange)
![Tests](https://img.shields.io/badge/tests-338+_passing-brightgreen)

An experimental research prototype for exploring neuro-symbolic AI. NSCK combines Vector Symbolic Architecture (VSA) with symbolic rule-based reasoning, neural spiking networks, and cross-domain transfer learning, designed for energy, processing, and memory efficiency. Runs on CPU only — no GPU required.

**Latest Update:** Phase 8 complete — temporal permutation, universal input layer, and mental rehearsal with veto mechanism. All 338+ tests passing. See [Architecture](docs/ARCHITECTURE.md) and [Phase History](docs/PHASE_HISTORY.md).

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

### Phase 8: Natural Language Learning (NEW)
| Capability | Description |
|---|---|
| **Text Learning** | Learn from text files using VSA-based concept extraction (NOT LLM-dependent) |
| **Concept Extraction** | Pattern-based extraction of concepts and semantic relations |
| **Hypervector Encoding** | Text → 10,240-dim hypervectors via LinguaCortex (Semantic Folding) |
| **Knowledge Storage** | Concepts stored in SemanticMemory graph + EpisodicMemory episodes |
| **Natural Language Q&A** | Answer questions based on learned knowledge with confidence scores |
| **Query System** | Semantic search + spreading activation + fact retrieval |
| **Dashboard Integration** | Upload text files, query knowledge, view statistics via web UI |

---

## Known Limitations

1. **Pattern-based language learning** — uses regex patterns for relation extraction, not deep NLU
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

| Module | Purpose | Key Classes/Methods |
|---|---|---|
| `cognitive_engine.py` | Central orchestrator — integrates all subsystems | `CognitiveEngine.decide()`, `CognitiveEngine.learn()` |
| `global_workspace.py` | GWT decision arbitration with mental rehearsal | `GlobalWorkspace.compete_with_rehearsal()` |
| `analogy.py` | Cross-domain transfer via structural alignment | `AnalogyEngine.find_analogy()`, `transfer_rule()` |
| `train_phase7_demo.py` | Integrated system with KnowledgeStore and LLM translator | Full 7-phase integration |
| `testing_dashboard.py` | Web dashboard with chat, games, monitoring, and **text learning** | Flask app on port 5051, 23 API endpoints |
| `text_knowledge_learner.py` | **NEW:** VSA-based text learning (NOT LLM-dependent) | `TextKnowledgeLearner.learn_from_text_file()`, `query_learned_knowledge()` |
| `rule_learner.py` | Frequency-based symbolic rule induction | `RuleLearner.observe()`, `induce_rules()` |
| `causal_reasoning.py` | Causal graph induction and counterfactuals | `CausalReasoner.counterfactual()`, Delta-P |
| `continual_learning.py` | EWC, PackNet, Progressive Networks, Memory Replay | `ContinualLearner.ewc_loss()`, `PackNetManager` |
| `multi_task_learning.py` | Shared encoder + task heads + gradient surgery | `GradientSurgery.project_conflicting_gradients()` |
| `world_model.py` | Forward simulation with sparse projection | `WorldModel.imagine()`, 128-dim bottleneck |
| `language_module.py` | Phi3 LLM translator (peripheral) | `LanguageModule.understand()`, `generate()` |
| `emotion_system.py` | Emotion processing (Plutchik+Russell) | `EmotionSystem.update_from_drives()`, 8 emotions |
| `theory_of_mind.py` | Agent mental modeling + false belief detection | `TheoryOfMind.detect_false_belief()` (Sally-Anne) |
| `self_model.py` | Self-awareness and performance tracking | `SelfModel.predict_success()`, context-aware |
| `episodic_memory.py` | VSA-based experience storage | `EpisodicMemory.recall_similar()` (LSH) |
| `semantic_memory.py` | Concept graph with spreading activation | `SemanticMemory.spread_activation()` (NetworkX) |
| `planner.py` | STRIPS-style goal-directed planning | `STRIPSPlanner.plan()` (BFS) |
| `hypervec_py.py` | VSA core operations (10,240-bit) | `xor()`, `bundle()`, `similarity()`, `permute()` |
| `universal_input.py` | Universal data grounding to VSA | `UniversalInput.ground()` (any modality → HV) |

**For detailed API documentation, see [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md)**

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

### API Examples

**Decision Making:**
```python
from cognitive_engine import CognitiveEngine
from config import NSCKConfig

engine = CognitiveEngine(NSCKConfig())
cognitive_state = engine.decide(
    state={"head": (5,5), "food": (8,3)},
    task_tag="snake"
)
print(f"Action: {cognitive_state.chosen_action}")
print(f"Confidence: {cognitive_state.confidence}")
print(f"Winner: {cognitive_state.trace['winner']}")
```

**Learning from Experience:**
```python
engine.learn(
    state=state_t,
    action="ACTION_RIGHT",
    reward=1.0,
    task_tag="snake",
    outcome="success",
    next_state=state_t1
)
```

**Transfer Learning:**
```python
from analogy import AnalogyEngine

engine = AnalogyEngine()
analogy = engine.find_analogy("snake", "pong")
new_cond, new_action = engine.transfer_rule(
    {"AGENT_NEAR_TARGET"}, "MOVE_FORWARD", "snake", "pong"
)
```

**Memory Retrieval:**
```python
from episodic_memory import EpisodicMemory

memory = EpisodicMemory()
similar_episodes = memory.recall_similar(query_hv, "snake", k=5)
```

**Text Learning (NEW):**
```python
from text_knowledge_learner import TextKnowledgeLearner

# Initialize learner (uses VSA, not LLM)
learner = TextKnowledgeLearner()

# Learn from text file
session = learner.learn_from_text_file("science.txt")
# → Extracts concepts, relations, stores in semantic memory

# Query learned knowledge
result = learner.query_learned_knowledge("What is photosynthesis?")
# → Returns: confidence, facts, reasoning trace
print(f"Confidence: {result['confidence']:.2f}")
print(f"Answer: {result['answer']}")
```

See [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md) for complete API reference.

---

## Running Tests

```bash
# Run core tests
python -m pytest nsck-demo/tests/ -v

# Run transfer learning tests specifically
python -m pytest nsck-demo/tests/test_transfer_learning.py -v

# Run testing dashboard tests
python -m pytest nsck-demo/tests/test_testing_dashboard.py -v

# Run text learning tests (NEW)
python -m pytest nsck-demo/tests/test_text_knowledge_learner.py -v
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
- **📚 Text Learning** — **NEW:** Upload text files, learn concepts/relations via VSA, query learned knowledge with confidence scores
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
| **Core Documentation** | |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, layer diagrams, decision cycle, and data flow |
| [docs/VSA_THEORY.md](docs/VSA_THEORY.md) | Mathematical foundations with proofs, capacity bounds, and encoding schemes |
| [docs/MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md) | Complete API reference for all 22+ modules with signatures and descriptions |
| [docs/TESTING.md](docs/TESTING.md) | Test methodology, all 85 tests documented, performance benchmarks |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Development guidelines, coding standards, and extension patterns |
| **Text Learning System** | |
| [docs/TEXT_LEARNING_USER_GUIDE.md](docs/TEXT_LEARNING_USER_GUIDE.md) | User guide for text learning: file upload, queries, API reference |
| [docs/TEXT_LEARNING_ARCHITECTURE.md](docs/TEXT_LEARNING_ARCHITECTURE.md) | Technical architecture of VSA-based text learning |
| [docs/TEXT_LEARNING_PROOF.md](docs/TEXT_LEARNING_PROOF.md) | Proof of text learning capabilities with test results |
| **Legacy Documentation** | |
| [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md) | Legacy reference - redirects to current documentation |

---

## License

See repository for license details.
