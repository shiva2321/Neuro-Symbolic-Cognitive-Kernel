# NCGN: Neuromorphic Cognitive Graph Network (Unified v6.0)

**A Neuro-Symbolic Artificial Intelligence that learns, reasons, and speaks.**

NCGN combines a biological "System 1" physics engine (Hebbian learning, thermodynamics) with a symbolic "System 2" logic controller (schema validation, planning).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Key Features

*   **Unified Cognition**: One system handles reflex games (Snake) and high-level dialogue.
*   **3-Factor Learning**: Learn from reward (Dopamine) with stability mechanisms to prevent catastrophic forgetting.
*   **Thermodynamic Regulation**: Homeostasis and Seizure Damping prevent energy explosions.
*   **Deep Reasoning**: System 2 intervenes when "Surprised" (e.g., detecting logical contradictions like "Dog Eat Metal").

---

## 📦 Installation

1.  **Clone**:
    ```bash
    git clone https://github.com/shiva2321/Node_network.git
    cd Node_network
    ```
2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    # Optional: Install spaCy model for better NLP
    python -m spacy download en_core_web_sm
    ```

---

## 🎮 Usage

### 1. **Run the Dashboard (Recommended)**
The unified interface for everything (Games, Chat, Brain Viz).

```bash
python -m ui.dashboard
```
> Open **http://localhost:5000** in your browser.

*   **Games Tab**: Train the agent to play Snake or Corridor. Watch the neuronal weights evolve.
*   **Dialogue Tab**: Chat with the system. Teach it facts ("Birds fly"). Try to confuse it ("Penguins do not fly").
*   **Brain Tab**: Visualize the firing patterns in real-time.

### 2. **Run Tests**
Verify the integrity of the system mechanics (Physics, Logic, Learning).

```bash
python -m unittest discover tests
```

---

## 🧠 Architecture Overview

The system is split into two integrated layers in `core/`:

1.  **System 1 (`core/system1.py`)**: The "Fast" Engine.
    *   **Softmax Inhibition**: Probabilistically selects active concepts.
    *   **Hebbian Plasticity**: "Neurons that fire together, wire together" (modulated by Dopamine).
2.  **System 2 (`core/system2.py`)**: The "Slow" Controller.
    *   **Schema Validation**: Checks if actions violate logical constraints (e.g., `schema_eat.json`).
    *   **Intervention**: Pauses the engine to fix errors or ask questions.

For a deep dive, see [DOCUMENTATION.md](DOCUMENTATION.md) and [CORE_ANALYSIS.md](C:\Users\shiva\.gemini\antigravity\brain\ea4af2f2-2c39-4ac3-97eb-b17c1b334fa5\CORE_ANALYSIS.md).

---

## 📄 License

MIT License. See LICENSE for details.
