# NSCK AI Model — Genuine Cognitive AI Using the Full NSCK Architecture

A text and image AI model built on the **actual NSCK cognitive architecture** —
not a reimplementation, but a genuine integration of the existing NSCK modules
(SemanticMemory, EpisodicMemory, GlobalWorkspace, EmotionSystem, CausalReasoner,
SelfModel, CuriosityModule, TextKnowledgeLearner).

## What Makes This Different

Traditional AI models use heavy matrix multiplication (transformers, neural
networks).  This system uses **Vector Symbolic Architecture (VSA)** — 10,240-bit
binary hypervectors where all operations are XOR (binding), majority-rule
(bundling), and Hamming distance (similarity).  No matrix multiplication.
No backpropagation.  No GPU required.

## Architecture — The Actual NSCK Modules Used

| Module | Source | Purpose |
|--------|--------|---------|
| `SemanticMemory` | `nsck-demo/python/core/memory/semantic_memory.py` | Concept graph (NetworkX DiGraph) + spreading activation |
| `EpisodicMemory` | `nsck-demo/python/core/memory/episodic_memory.py` | VSA episode storage + LSH indexing for fast retrieval |
| `GlobalWorkspace` | `nsck-demo/python/core/reasoning/global_workspace.py` | LIDA-style coalition competition (consciousness model) |
| `EmotionSystem` | `nsck-demo/python/core/cognitive/emotion_system.py` | Plutchik 8 emotions + valence/arousal model |
| `CausalReasoner` | `nsck-demo/python/core/reasoning/causal_reasoning.py` | Forward/backward chaining + counterfactual reasoning |
| `SelfModel` | `nsck-demo/python/core/cognitive/self_model.py` | Confidence calibration + self-awareness |
| `CuriosityModule` | `nsck-demo/python/core/learning/curiosity.py` | Novelty detection + intrinsic motivation |
| `TextKnowledgeLearner` | `nsck-demo/python/core/language/text_knowledge_learner.py` | VSA text learning with semantic folding |
| `HyperVector` | `nsck-demo/python/core/vsa/hypervec_shim.py` | 10,240-bit binary vectors (Rust/Python) |

## Quick Start

```bash
# Install dependencies
pip install numpy flask networkx

# Train and chat (interactive)
python -m nsck_ai_model.train --mode seed --chat

# Start the monitoring dashboard
python -m nsck_ai_model.dashboard --train seed --port 5090

# Run tests
python -m pytest nsck_ai_model/tests/ -v
```

## How It Works — End to End

### Training Pipeline

1. **Text Input** → `TextKnowledgeLearner.learn_from_text()`
2. **Semantic Folding** → LinguaCortex encodes text to HyperVectors
3. **Concept Extraction** → Concepts added to `SemanticMemory` graph
4. **Relation Discovery** → Relations stored as graph edges
5. **Episodic Recording** → Episodes stored in `EpisodicMemory`
6. **Causal Learning** → Causal links added to `CausalGraph`
7. **N-gram Learning** → `ResponseGenerator` learns text patterns

### Chat Pipeline (11 Glass-Box Stages)

Every `chat()` call produces a `ThoughtTrace` with 11 stages:

| Stage | Module | What It Does |
|-------|--------|-------------|
| 1. Encode | `hypervec_shim` | Input → 10240-bit HyperVector |
| 2. Emotion | `EmotionSystem` | Classify emotional tone (Plutchik) |
| 3. Concepts | `TextKnowledgeLearner` | Extract query concepts |
| 4. Semantic Search | `SemanticMemory` | Find matching concepts by HV similarity |
| 5. Spreading Activation | `SemanticMemory` | Propagate through concept graph |
| 6. Knowledge Query | `TextKnowledgeLearner` | Query learned facts + episodes |
| 7. Causal Inference | `CausalReasoner` | Forward-chain causal rules |
| 8. Curiosity | `CuriosityModule` | Assess input novelty |
| 9. Global Workspace | `GlobalWorkspace` | Coalition competition for response |
| 10. Self-Model | `SelfModel` | Update confidence calibration |
| 11. Generate | Response assembly | Build response from learned knowledge |

### Response Assembly

- No templates, no format strings, no hardcoded patterns
- Retrieves candidate sentences from learned knowledge
- Scores by concept overlap with inverse-frequency weighting
- Deduplicates with Jaccard similarity (> 0.5 = skip)
- XSS sanitisation strips HTML tags

## Glass-Box Traceability

Every response includes a full trace showing:
- Which NSCK module contributed what
- How long each stage took
- What concepts were found
- What facts were retrieved
- Which coalition won in the GlobalWorkspace
- Confidence calibration from SelfModel
- Novelty score from CuriosityModule

Example trace output:
```json
{
  "trace_id": "a1b2c3d4e5f6",
  "query": "What is the capital of France?",
  "total_ms": 3.2,
  "step_count": 11,
  "steps": [
    {"stage": "encode", "action": "Encoded input to 10240-bit HyperVector", "duration_ms": 0.1},
    {"stage": "emotion", "action": "NSCK EmotionSystem classified input", "duration_ms": 0.05},
    {"stage": "search_semantic", "action": "Searched NSCK SemanticMemory", "duration_ms": 0.3},
    {"stage": "global_workspace", "action": "NSCK GlobalWorkspace coalition competition", "duration_ms": 0.1},
    ...
  ]
}
```

## Dashboard

The dashboard (port 5090) is wired to all NSCK backend modules:

- **Chat** — Talk to the AI and see responses + traces
- **Training** — Feed data and watch SemanticMemory graph grow
- **Traces** — Inspect full glass-box reasoning chains
- **Knowledge** — Browse concepts, relations, and facts from SemanticMemory
- **Telemetry** — Live stats from all NSCK modules
- **Logs** — Structured log stream

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with the AI |
| `/api/train` | POST | Train on text |
| `/api/train/seed` | POST | Train on seed corpus |
| `/api/train/hf` | POST | Train on HuggingFace data |
| `/api/stats` | GET | System stats from all NSCK modules |
| `/api/knowledge` | GET | Export all learned knowledge |
| `/api/knowledge/concepts` | GET | SemanticMemory concepts |
| `/api/knowledge/relations` | GET | SemanticMemory relations |
| `/api/knowledge/rules` | GET | TextKnowledgeLearner facts |
| `/api/emotion` | GET | EmotionSystem state |
| `/api/health` | GET | Health check |
| `/api/logs` | GET | Log stream |
| `/api/chat/history` | GET | Conversation history |
| `/api/reset` | POST | Reset all modules |

## Testing

```bash
# All tests (123 total)
python -m pytest nsck_ai_model/tests/ -v

# Unit tests only (77 tests)
python -m pytest nsck_ai_model/tests/test_ai_engine.py -v

# Production tests only (46 tests)
python -m pytest nsck_ai_model/tests/test_production.py -v
```

### Test Categories

- **NSCK Module Integration** (13 tests) — Verifies actual NSCK module usage
- **Knowledge QA** (10 tests) — Verified correct answers from trained model
- **Glass-Box Trace** (7 tests) — All 11 stages present and traced
- **Response Quality** (6 tests) — Grammatical, relevant, no duplicates
- **Emotion System** (4 tests) — Plutchik classification
- **Training** (7 tests) — Populates SemanticMemory, TextKnowledgeLearner
- **Conversation** (3 tests) — Multi-turn, auto-learn, history
- **Edge Cases** (6 tests) — Empty, XSS, Unicode, long input
- **System Stats** (4 tests) — All NSCK module stats
- **Performance** (4 tests) — Latency < 1s, bulk training < 5s
- **Dashboard API** (6 tests) — All endpoints working
- **More**  — ThoughtTrace, ResponseGenerator, etc.

## No Hardcoded Patterns

The system has **zero** hardcoded:
- ❌ No response templates
- ❌ No regex relation extractors
- ❌ No keyword lexicons
- ❌ No intent classifiers
- ✅ Everything learned from training data
- ✅ Only built-in: function word list (grammatical, not domain-specific)

## Files

| File | Purpose |
|------|---------|
| `ai_engine.py` | Core engine integrating all NSCK modules |
| `data_pipeline.py` | HuggingFace data loading + seed corpus |
| `train.py` | Training CLI (--mode seed/hf/full --chat) |
| `image_understanding.py` | VSA image context understanding |
| `dashboard.py` | Full monitoring dashboard (Flask) |
| `tests/test_ai_engine.py` | 77 unit tests |
| `tests/test_production.py` | 46 production tests |
| `ARCHITECTURE.md` | Detailed architecture explanation |
| `EVALUATION.md` | Test results and benchmarks |
