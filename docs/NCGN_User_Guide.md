# NCGN User Guide

How to operate, teach, and visualize the Neuromorphic Cognitive Graph Network.

## Getting Started

### Prerequisites
- Python 3.9+
- Flask, Flask-SocketIO (for dashboard)

### Installation
```powershell
pip install flask flask-socketio
```

## Running the System

### 1. The Web Dashboard
The best way to visualize System 1 dynamics.

```powershell
python run_dashboard.py
```
*   Open **http://127.0.0.1:5000**
*   **Controls**:
    *   **Inject Energy**: Click on a node to add energy.
    *   **Step**: Advance one tick.
    *   **Run**: Auto-run ticks.
    *   **Surprise Meter**: Watch this gauge. If it spikes, System 2 has intervened.

### 2. Interactive Dialogue (CLI)
Chat with the system directly to teach it.

```powershell
python -m cortex.dialogue
```

## Teaching the Brain

The system learns from **subject-verb-object** statements.

### Basic Facts
> **You**: "Dogs eat meat."
>
> **NCGN**: "✓ Learned: dogs eats meat"

### Handling Conflicts (The "Robot Dog" Scenario)
If you say something that contradicts existing schema logic (e.g., metal isn't edible):

> **You**: "Dogs eat metal."
>
> **NCGN**: "🤔 My knowledge says: Object 'metal' violates constraints: ['is_edible']. Why do you say 'dogs eats metal'?"

 You must explain *why*:

> **You**: "It's a robot dog."
>
> **NCGN**: "✓ Understood. I've learned:
>    • robot dog is a type of dog
>    • robot dog can eats metal"

### Ingesting Files
You can load bulk knowledge from text files.

1.  Create a file (e.g., `biology.txt`):
    ```text
    Birds fly.
    Penguins are birds.
    Penguins cannot fly.
    ```
2.  In the Dashboard or CLI:
    > **You**: "read biology.txt"
3.  The system will stage the concepts. Review conflicts (like "Penguins cannot fly" vs "Birds fly") and commit.

## Troubleshooting

*   **System 2 Triggered**: If the simulation pauses unexpectedly, System 2 has been triggered by high surprise. Check the console or dashboard for the Diagnosis.
*   **"Ghost" Signals**: If energy seems to disappear, check the decay rate in `core/system1.py`. Without reinforcement, thoughts naturally decay.
