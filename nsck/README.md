# NSCK Core Package

This is the main NSCK (Neural-Symbolic Cognitive Kernel) package containing the cognitive architecture.

## Package Structure

```
nsck/
├── python/core/           ← Core cognitive modules
│   ├── vsa/               ← HyperVector engine
│   ├── reasoning/         ← CognitiveEngine, GWT, Planner, Causal, Rules, Analogy
│   ├── learning/          ← Hebbian, Curiosity
│   ├── cognitive/         ← Metacognition, Self-Model, ToM, Emotions
│   ├── perception/        ← SNN, VSA-SNN Bridge, Grounding
│   ├── memory/            ← Episodic, Semantic, Staged Recall
│   ├── language/          ← NLU, Dialogue, Text Knowledge, UniversalInput
│   ├── integration/       ← Config, Persistence, Brain Fusion, Explanation
│   └── training/          ← SNN training pipelines
│
├── rust_vsa/              ← Rust VSA accelerator (PyO3)
├── tests/                 ← Comprehensive test suite
├── docs/                  ← Documentation
├── data/                  ← Test corpora
├── examples/              ← Usage examples
└── archive/               ← Historical experiments
```

## Quick Start

```python
import sys
sys.path.insert(0, '/path/to/nsck')

from python.core.reasoning.cognitive_engine import CognitiveEngine

engine = CognitiveEngine()

# Register a task domain
engine.register_task(
    task_tag="my_task",
    predicates={"obstacle": lambda s: s.get("obstacle", False)},
    actions=["move", "wait", "interact"],
)

# Decision loop
state = {"obstacle": True, "energy": 0.8}
result = engine.decide(state, ["move", "wait", "interact"], "my_task")
print(f"Action: {result.action}")
print(f"Explanation: {result.explanation}")
print(f"Confidence: {result.confidence}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design with Mermaid diagrams
- [Formulas](docs/FORMULAS.md) — All mathematical foundations
- [Workflows](docs/WORKFLOWS.md) — Decision loop and data flow
- [Module Reference](docs/MODULE_REFERENCE.md) — API reference
- [Repository Map](docs/REPOMAP.md) — Complete codebase map

## Running Tests

```bash
cd /workspaces/Node_network
pytest nsck/tests/ nsck/python/core/tests/ -q
```

## Building the Rust Accelerator

```bash
cd nsck/rust_vsa
pip install -e .
```

Provides ~10x speedup for HyperVector operations. Falls back to pure Python (NumPy) if not built.
