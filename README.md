# NCGN: Neuromorphic Cognitive Graph Network

NCGN is a "Cognitive Control System" that operates on sparse, event-driven graph dynamics rather than dense matrix multiplication. It implements a dual-process architecture inspired by biological neural systems, where high-speed reactive processing (System 1) is overseen by a deliberative, schema-based monitor (System 2).

## 🚀 Quick Start

Ensure you have Python 3.11+ installed.

1. **Install Dependencies** (Optional - only `pytest` is used for external testing):
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run Everything**:
   Run the unified script to execute all unit tests and the "Dog Eat Metal" demo:
   ```powershell
   python run_all.py
   ```

3. **Check Results**:
   Results are saved to `ncgn_complete_results.txt`.

## 🧠 Core Philosophy

NCGN is built on three pillars:
1.  **Sparsity**: Only active nodes are processed, adhering to a biological "energy budget."
2.  **Causal Verification**: The system doesn't just predict; it validates predictions against logical schemas.
3.  **Surprise-Driven Attention**: Computational resources (System 2) are only allocated when observations violate high-confidence expectations.

## 📁 Project Structure

```
d:\NGCN\
├── core/                # The logic engine
│   ├── memory.py       # ConceptNodes, Synapses, and GraphMemory
│   ├── system1.py      # The 8-phase tick-based physics engine
│   ├── bridge.py       # Surprise monitoring & prediction tracking
│   ├── system2.py      # Deliberative diagnosis & intervention
│   └── schemas/        # Symbolic knowledge templates (JSON)
├── tests/              # Comprehensive test suite
├── main.py             # System demo and end-to-end scenarios
├── run_all.py          # Unified runner for tests and demo
└── DOCUMENTATION.md    # Detailed technical specification
```

## 🐕 The "Dog Eat Metal" Scenario

The system demonstrates its "understanding" through a classic test case:
*   **Prediction**: Based on its history, the system predicts "Meat" when a "Dog" is active.
*   **Observation**: It receives "Metal" instead.
*   **Surprise**: A high-priority interrupt is triggered because a high-confidence expectation was violated.
*   **Diagnosis**: System 2 identifies a **Schema Violation** (Metal is not edible).
*   **Response**: Rather than hallucinating or accepting the input, it generates a query for clarification.

---
*Created as part of the Advanced Agentic Coding project.*
