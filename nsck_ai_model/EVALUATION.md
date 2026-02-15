# NSCK AI Engine — Evaluation Report

## Test Summary

| Test Suite | Tests | Passed | Status |
|---|---|---|---|
| Unit tests (`test_ai_engine.py`) | 77 | 77 | ✅ All Pass |
| Production tests (`test_production.py`) | 65 | 65 | ✅ All Pass |
| **Total** | **142** | **142** | **✅ All Pass** |

## Production Test Coverage

### §1 Knowledge Q&A (13 tests)
Tests factual question answering across all training domains:
- **Geography**: capitals of France, Germany, Japan
- **Science**: Sun, water, DNA, photosynthesis
- **Animals**: dogs, dolphins
- **Technology**: Python, machine learning
- **Cause/Effect**: flooding, smoking

All tests verify the response contains the correct factual keywords from
training data.

### §2 Response Quality (6 tests)
- Responses start with a capital letter
- Responses end with a period
- Responses are non-empty (> 5 chars)
- No duplicate sentences within a response
- Responses focus on the queried topic
- Same query produces consistent responses

### §3 Multi-turn Conversation (3 tests)
- **Pronoun resolution**: "What does she study?" resolves to the most
  recently discussed person
- Conversation history accumulates correctly
- Learning during conversation (auto_learn)

### §4 Glass-Box Traceability (7 tests)
- All 8 reasoning stages present in trace (encode, emotion,
  extract_concepts, search_semantic, search_episodic,
  spread_activation, causal_inference, generate)
- Timing data available for each step
- Unique trace IDs
- Inputs/outputs recorded for each step
- Confidence > 0.3 for known topics
- Confidence < 0.9 for unknown topics

### §5 Edge Cases (8 tests)
- Empty input handling
- Single character input
- Punctuation-only input
- XSS/script injection input
- Unicode input
- Very long input (200+ words)
- Repeated identical queries (10×)
- auto_learn=False doesn't train

### §6 Autonomous Learning (7 tests)
- Concept extraction (> 100 concepts from seed corpus)
- Relation learning (> 200 relations)
- Category abstraction (> 50 abstractions)
- Causal rule learning (> 30 rules)
- N-gram model training (> 50 bigram vocab)
- Forward chaining works
- Spreading activation finds related concepts

### §7 Performance (3 tests)
- Chat latency < 50ms average
- Seed corpus training < 2 seconds
- No degradation over 50 burst queries

### §8 Emotion System (2 tests)
- Emotion state included in every response
- Emotion blend values sum to ~1.0

### §9 Dashboard API (5 tests)
- `/api/health` returns status and version
- `/api/chat` rejects empty messages (400)
- `/api/train` rejects empty text (400)
- `/api/stats` returns complete structure

### §10 Image Understanding (4 tests)
- Same image → similar HVs (> 0.85 similarity)
- Different images → different HVs (< 0.8)
- Grayscale image support
- Cross-modal text→image retrieval

### §11 Data Pipeline (4 tests)
- Seed corpus has 60+ sentences
- All sentences well-formed (capital start, period end, > 10 chars)
- Pipeline trains engine to > 100 concepts, > 200 relations
- Pipeline stats track passages and time

### §12 System Lifecycle (3 tests)
- Export contains all sections
- Reset clears everything
- Retrain after reset works

## Performance Benchmarks

| Metric | Value |
|---|---|
| Chat latency (avg) | 3-8 ms |
| Seed training time | 0.3s |
| Concepts after seed | 211 |
| Relations after seed | 405 |
| Causal rules after seed | 66 |
| Abstractions after seed | 123 |
| Vocabulary size | 232 words |
| Bigram vocab | 180+ |
| Memory (approximate) | ~50 MB |

## Bugs Found & Fixed

| Bug | Severity | Status |
|---|---|---|
| Responses included previous user queries as answer text | High | ✅ Fixed |
| Duplicate sentences in responses (e.g., Eiffel Tower case) | Medium | ✅ Fixed |
| Pronoun resolution failed across turns | Medium | ✅ Fixed |
| XSS/garbage input produced random unrelated sentences | Medium | ✅ Fixed |
| Empty/punctuation-only input returned random knowledge | Low | ✅ Fixed |
| 2nd response sentence often unrelated to query | Medium | ✅ Fixed |
| Statement acknowledgment exposed internal state | Low | ✅ Fixed |
| Missing `/api/health` endpoint | Low | ✅ Added |

## Improvements Made

1. **Relevance-weighted concept scoring** — Uses inverse word frequency
   to weight rare/specific concepts higher than common ones.
2. **Fuzzy deduplication** — Jaccard similarity prevents structurally
   similar but factually different sentences from co-appearing.
3. **Episode filtering** — User queries stored as episodes are now
   filtered out of response candidates.
4. **Pronoun resolution** — Conversation context resolves "she", "he",
   "it", "they" to the most recently discussed concepts.
5. **Input validation** — Early short-circuit for empty/garbage input
   before heavy processing.
6. **n-gram validation** — Generated text is only used if it contains
   at least one query concept word.
7. **Health endpoint** — `/api/health` returns system status and version
   for monitoring.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/api/chat` | POST | Chat with AI |
| `/api/chat/history` | GET | Conversation history |
| `/api/train` | POST | Train on custom text |
| `/api/train/seed` | POST | Train on seed corpus |
| `/api/train/hf` | POST | Train on HuggingFace data |
| `/api/stats` | GET | System statistics |
| `/api/knowledge` | GET | Export all knowledge |
| `/api/knowledge/concepts` | GET | List concepts |
| `/api/knowledge/relations` | GET | List relations |
| `/api/knowledge/rules` | GET | List causal rules |
| `/api/emotion` | GET | Emotional state |
| `/api/logs` | GET | Recent logs |
| `/api/reset` | POST | Reset engine |
| `/api/health` | GET | Health check |

## Running Tests

```bash
# Unit tests (77 tests)
python -m pytest nsck_ai_model/tests/test_ai_engine.py -v

# Production tests (65 tests)
python -m pytest nsck_ai_model/tests/test_production.py -v

# All tests (142 tests)
python -m pytest nsck_ai_model/tests/ -v
```
