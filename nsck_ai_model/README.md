# NSCK AI Model

A genuine text-and-image AI model built on **Vector Symbolic Architecture (VSA)**.  
No transformers. No heavy matrix multiplication. No hardcoded patterns.  
Everything is learned autonomously from training data.

## Quick Start

```bash
# Train and start interactive chat
python -m nsck_ai_model.train --mode seed --chat

# Start the monitoring dashboard
python -m nsck_ai_model.dashboard --train seed
# Then open http://localhost:5090
```

## What It Does

The NSCK AI model understands natural language, learns from text, and responds in
sensible natural language based on what it has learned.  It can:

- **Learn** from any text — extracts concepts, relations, and patterns autonomously
- **Reason** — forward-chains causal rules, applies analogies, creates abstractions
- **Converse** — holds natural conversations grounded in its learned knowledge
- **Understand images** — encodes images into the same hypervector space as text
- **Explain itself** — every response includes a full glass-box reasoning trace

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed architecture document.

```
User Input (text or image)
    ↓
ThoughtTrace (starts recording — glass-box)
    ↓
TextEncoder (semantic folding → 10,240-dim binary hypervector)
    ↓
KnowledgeStore (concept graph + episodic memory + LSH index)
    ↓
ReasoningEngine (causal rules, spreading activation, abstraction)
    ↓
ResponseGenerator (retrieved sentences + learned n-gram NLG)
    ↓
ThoughtTrace (finalised with all steps)
    ↓
Natural Language Response + full trace
```

## Zero Hardcoded Patterns

Nothing in the engine is hardcoded:

| Component | Traditional Approach | NSCK Approach |
|---|---|---|
| Relation extraction | Regex patterns | Learned from sentence co-occurrence |
| Response generation | Template strings | Retrieved learned sentences + n-grams |
| Intent classification | Keyword lists | HV similarity to learned examples |
| Emotion detection | Sentiment lexicons | Learned valence from feedback signals |
| Category creation | Manual ontology | Autonomous abstraction from data |

## Components

| File | Purpose |
|---|---|
| `ai_engine.py` | Core engine: HyperVector, TextEncoder, KnowledgeStore, CausalRuleStore, KnowledgeAbstractor, EmotionTracker, ResponseGenerator, ThoughtTrace, NSCKAIEngine |
| `data_pipeline.py` | HuggingFace data streaming, seed corpus, training pipeline |
| `train.py` | Training script with CLI (`--mode seed/hf/full --chat`) |
| `image_understanding.py` | Image → HV encoding, cross-modal search |
| `dashboard.py` | Flask monitoring dashboard (port 5090) |
| `tests/test_ai_engine.py` | 77 tests covering all components |

## Dashboard

The dashboard at `http://localhost:5090` provides:

| Tab | What It Shows |
|---|---|
| 💬 Chat | Talk to the AI, see responses with confidence and emotion |
| 📚 Training | Feed data (seed corpus, HuggingFace, or custom text) |
| 🔍 Traces | Full glass-box reasoning chain for any query |
| 🧩 Knowledge | Browse concepts, relations, causal rules |
| 📊 Telemetry | Live stats, emotion state, system metrics |
| 📋 Logs | Structured log stream of all engine activity |

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST | Chat with the AI |
| `/api/chat/history` | GET | Get conversation history |
| `/api/train` | POST | Train on custom text |
| `/api/train/seed` | POST | Train on seed corpus |
| `/api/train/hf` | POST | Train on HuggingFace data |
| `/api/stats` | GET | System statistics |
| `/api/knowledge` | GET | Full knowledge export |
| `/api/knowledge/concepts` | GET | List concepts |
| `/api/knowledge/relations` | GET | List relations |
| `/api/knowledge/rules` | GET | List causal rules |
| `/api/emotion` | GET | Emotional state |
| `/api/logs` | GET | Activity logs |
| `/api/reset` | POST | Reset the engine |

## Testing

```bash
python -m pytest nsck_ai_model/tests/test_ai_engine.py -v
# 77 tests, all passing
```

## Python API Usage

```python
from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.data_pipeline import DataPipeline

# Create and train
engine = NSCKAIEngine()
pipeline = DataPipeline(engine)
pipeline.train_from_seed()

# Chat
result = engine.chat("What is the capital of France?")
print(result['response'])
# "Paris is the capital of France."

# Inspect reasoning trace
for step in result['trace']['steps']:
    print(f"[{step['stage']}] {step['action']}")

# System stats
stats = engine.get_system_stats()
print(f"Concepts: {stats['knowledge']['total_concepts']}")

# Image understanding
from nsck_ai_model.image_understanding import ImageUnderstanding
import numpy as np

iu = ImageUnderstanding(engine)
img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
iu.learn_image(img, "A sunset over mountains")
results = iu.search_by_text("sunset")
```
