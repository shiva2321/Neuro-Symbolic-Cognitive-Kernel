# NCGN Unified Architecture Documentation (v6.0)

**Version**: 6.0 (Unified)
**Type**: Neuro-Symbolic Cognitive Architecture

---

## 1. Overview

The **Neuromorphic Cognitive Graph Network (NCGN)** is a hybrid AI architecture that combines biological plausibility with symbolic reasoning. It implements a **Dual-Process Theory** of cognition:

*   **System 1 (The Engine)**: A fast, parallel, energy-based physics engine that handles association, attention, and learning. It operates on a graph of concept nodes.
*   **System 2 (The Controller)**: A slow, serial, logical supervisor that handles planning, schema validation, and error correction.

In v6.0, these systems are unified with **3-Factor Hebbian Learning** and **Thermodynamic Regulation**, allowing the system to learn from experience while remaining logically grounded.

---

## 2. Directory Structure

The unified system consolidates all core logic into the `core/` package:

```
d:/NGCN/
├── core/
│   ├── system1.py          # Physics Engine (Softmax, Thermodynamics)
│   ├── system2.py          # Logic Controller (Schemas, Intervention)
│   ├── memory.py           # Graph Data Structures (Nodes, Synapses)
│   ├── learning.py         # 3-Factor Hebbian & Dopamine
│   ├── dialogue.py         # Conversation State Machine
│   ├── query_engine.py     # Graph Traversal for QA
│   └── games/              # Integrated Environments (Snake, Corridor)
├── ui/
│   ├── dashboard.py        # Unified Flask Backend
│   └── templates/
│       └── dashboard.html  # Tabbed Dashboard Interface
└── tests/                  # Unified Test Suite
```

---

## 3. System 1: The Physics Engine

**File**: `core/system1.py`

System 1 treats the knowledge graph as a thermodynamic system. Concepts are nodes with "Energy" (0.0 to 1.0).

### Key Dynamics

1.  **Propagation**: Energy flows from active nodes to their neighbors via synapses. `Output = Input × Weight`.
2.  **Softmax Inhibition** (Replaces v5 k-WTA):
    *   Nodes are grouped into **Clusters** (e.g., MOTOR, HIDDEN).
    *   Within a cluster, nodes compete for activation energy.
    *   Formula: $P(n) = \frac{e^{(E_n/T)}}{\sum e^{(E_i/T)}}$
3.  **Thermodynamics**:
    *   **Temperature ($T$)**: dynamic parameter controlling exploration. High $T$ = random/flat; Low $T$ = determinstic/sharp.
    *   **Homeostasis**: Global metabolic caps prevent total energy from exploding. If $\sum E > Cap$, all nodes are scaled down.
4.  **Seizure Damping**: If the total system energy exceeds the safety threshold (`DEFAULT_GLOBAL_ENERGY_THRESHOLD`), the engine applies a massive damping factor (0.5x) to maintain stability.

---

## 4. Memory & Synapses

**File**: `core/memory.py`

### ConceptNode
The atom of thought.
*   `energy`: Current activation level.
*   `threshold`: Activation barrier for firing.
*   `cluster`: Group membership (MOTOR, SENSORY, HIDDEN).

### Synapse
The connection between thoughts. v6 adds critical learning fields:
*   `weight`: Connection strength ($w$).
*   `trace`: **Eligibility Trace**. Records that "Pre fired, then Post fired." Decays rapidly. Used to credit past actions for current rewards.
*   `stability`: **Consolidation Factor**. Tracks how "entrenched" a memory is. High stability prevents overwriting (Catastrophic Forgetting Prevention).

---

## 5. Learning: 3-Factor Hebbian

**File**: `core/learning.py`

The system learns via a biologically-inspired rule:
$$ \Delta W = \eta \times (1 - Stability) \times Dopamine \times Trace $$

### Components
1.  **DopamineModulator**: Calculates **Reward Prediction Error (RPE)**.
    *   $RPE = Reward_{actual} - Reward_{baseline}$ (Washout prevention).
    *   The system only learns when outcomes are *unexpectedly* good or bad.
2.  **ThreeFactorLearner**: Applies the weight update.
    *   **Trace**: "I did this action recently."
    *   **Dopamine**: "The outcome was good."
    *   **Stability**: "I already know this well, don't change much."
    *   **Result**: Weights increase for successful actions, decrease for failures.

---

## 6. System 2: The Logical Supervisor

**File**: `core/system2.py`

System 2 monitors System 1. It does not run every tick. It triggers only on **Surprise**.

### The Logic Loop
1.  **Monitor**: Checks difference between *Predicted State* and *Observed State*.
2.  **Interrupt**: If `Surprise > Threshold`, PAUSE System 1.
3.  **Diagnose**:
    *   Identify the active "Subject-Verb-Object" triple (e.g., "Dog Eat Metal").
    *   Load the relevant **EventSchema** (`eat.json`).
    *   Check logical constraints (`Metal.is_edible == False`).
4.  **Intervene**:
    *   If Violation: Trigger **LTD** (weaken the connection) or **Query User** ("Wait, dogs can't eat metal").
    *   If Valid: Trigger **LTP** (strengthen connection).

---

## 7. Interfaces: Dashboard & Dialogue

### Unified Dashboard
**File**: `ui/templates/dashboard.html`

A single-page application connecting all subsystems:
*   **Brain Tab**: Real-time D3.js force-directed graph. Shows active nodes (firing) and energy flow.
*   **Games Tab**: Controls for Snake/Corridor. Shows Reinforcement Learning metrics (Reward, RPE).
*   **Dialogue Tab**: Chat interface. Talk to the brain, teach it facts ("Sky is blue"), or ask questions.
*   **Knowledge Tab**: Bulk ingestion of text files. Staging area review for new triples.

### Running the System
```bash
# Start the unified dashboard
python -m ui.dashboard
```
Access at `http://localhost:5000`.

---

## 8. Limitations

*   **Scalability**: The Python implementation slows down significantly >10k nodes due to $O(N^2)$ potential interactions (optimized to $O(E)$ but Python overhead remains).
*   **Language**: The NLP parser is rule-based and limited to simple subject-verb-object structures. It is not an LLM.
*   **Planning Depth**: While System 2 can intervene, it lacks a deep simulation search (MCTS) for multi-step problem solving.
