# NCGN v2.0: Brain-First Cognitive Architecture

> **A "Brain-First" AI where logical graphs drive decisions, and LLMs serve as linguistic peripherals.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ⚡ v2.0 "Brain-First" Upgrade

NCGN v2.0 inverts the typical Agentic AI paradigm. Instead of an LLM calling tools, **the Graph (Brain)** is the central operator, using the LLM only for translation and conflict resolution.

| Feature | v2.0 (New) | v7 (Legacy) |
|---------|------------|-------------|
| **Core Logic** | Explicit Graph Rules (Conflict/Decision) | LLM Reasoning |
| **Learning** | Auto-Hebbian (No LLM needed) | LLM-Dependant |
| **Dashboard** | Real-time Cytoscape + Game Panel | D3.js Static |
| **Architecture** | Logic Core + Peripherals | Monolithic Reasoner |

📖 **[Read v2.0 Architecture](docs/architecture_v2.md)**

---

## 🧠 Brain-First Design

1. **System 1 (Association)**: Fast, parallel propagation of energy through the Knowledge Graph.
2. **System 2 (Logic)**: Deterministic "Conflict Detection" and "Winner-Take-All" decision making.
3. **Peripherals**: 
   - `LinguisticProcessor`: Extracts graph triplets from text.
   - `NaturalLanguageFormatter`: Verbalizes brain state.

```python
from ncgn.brain import Brain

# Initialize Brain-First System
brain = Brain()

# Learn without LLM (Auto-Hebbian)
brain.add_concept("fire")
brain.add_concept("hot")
brain.inject_multiple({"fire": 1.0, "hot": 1.0})
brain.learn(reward=1.0) 
# -> Edge ("fire", "hot") created automatically

# Think
brain.think(steps=10)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/shiva2321/Node_network.git
cd Node_network

# Install dependencies
pip install -r requirements.txt
```

### Interactive Dashboard v2
Launch the real-time visualization with Game Integration:

```bash
python -m ui.dashboard
```
Open `http://localhost:5000` to see your brain live.

### Testing System 1
Verify that the core graph dynamics work without any LLM:

```bash
python tests/verify_v2_core.py
```

---

## 🎮 Game Integration (Phase 5)

v2.0 includes a **GameBrainBridge** that connects the cognitive graph to:
- **Corridor**: Simple 1D navigation task.
- **Snake**: 2D survival/foraging task.

The brain "sees" the game via concept injection (e.g., `see_wall` node activates) and "acts" by firing action nodes (`move_right`).

---

## 📦 Project Structure

```
ncgn/                   # v2.0 Source
├── brain.py            # Central Coordinator (Logic Core)
├── decision.py         # Winner-Take-All Logic
├── conflicts.py        # Graph Conflict Detector
├── confidence.py       # Hebbian Confirmation
├── games/              # Bridge & Environments
├── linguistic_processor.py # Parsing
└── formatter.py        # Verbalization

ui/                     # Dashboard v2
├── dashboard.py        # Flask Backend
└── templates/
    └── dashboard_v2.html # Cytoscape UI
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
