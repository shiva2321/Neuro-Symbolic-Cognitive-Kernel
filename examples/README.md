# Examples

This directory contains self-contained scripts that demonstrate the main capabilities of NSCK. Each script can be run from the repository root after installing dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

| Script | What it demonstrates |
|--------|----------------------|
| [`01_cognitive_engine_basic.py`](01_cognitive_engine_basic.py) | Register a domain, feed observations, and receive decisions with explanations from `CognitiveEngine` |
| [`02_semantic_memory.py`](02_semantic_memory.py) | Store concepts and relations in `SemanticMemory`, run spreading activation, and search by hypervector similarity |
| [`03_text_knowledge_learner.py`](03_text_knowledge_learner.py) | Parse plain-text sentences into SVO triples with `TextKnowledgeLearner` and inspect the resulting knowledge graph |

## Running an example

```bash
python examples/01_cognitive_engine_basic.py
python examples/02_semantic_memory.py
python examples/03_text_knowledge_learner.py
```

For deeper documentation see the [`nsck/docs/`](../nsck/docs/) directory.
