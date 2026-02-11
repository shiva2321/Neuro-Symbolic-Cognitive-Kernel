# NSCK Testing Dashboard

## Overview

The **NSCK Testing Dashboard** is a comprehensive web-based interface for testing, monitoring, and examining all capabilities of the NSCK cognitive system. It provides interactive tools for chatting with the system, running game simulations, observing internal cognitive states, and exporting complete logs for analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python nsck-demo/python/testing_dashboard.py

# Open in browser
# http://localhost:5051
```

The dashboard runs on port **5051** by default. Set `NSCK_TESTING_PORT` to use a different port.

---

## Features

### 💬 Chat & Test (Tab 1)

Submit text samples and chat interactively with the cognitive system.

- **Text Sample Input** — Paste or type any text for the system to process. Pre-loaded samples (Nature, Science, Story) available for quick testing.
- **Conversation** — Multi-turn chat with the system. Each response includes:
  - Natural language answer from the knowledge integration pipeline
  - Dialogue manager response for conversational context
- **Reasoning Trace** — Real-time display of the cognitive pipeline steps (encoding, memory recall, disambiguation, semantic matching).
- **Response Details** — Confidence score, emotional state, and context disambiguations.
- **Quick Stats** — Live counters for queries processed, knowledge entries, episodes, and facts learned.

### 🎮 Game Simulations (Tab 2)

Attach game simulations for the system to play, learn from, and be observed.

- **Three Games** — Snake (🐍), Pong (🏓), and Maze (🏰).
- **Sequential or Simultaneous** — Start games one at a time or launch all three at once.
- **Auto-play** — System autonomously plays using heuristic + self-model reasoning.
- **Manual Step** — Advance one step at a time for detailed observation.
- **Speed Control** — Adjustable simulation speed (50ms to 1000ms per step).
- **Live Visualization** — Canvas rendering of each game with real-time updates.
- **Per-game Logs** — Each game card shows step-by-step action log with reasoning.
- **Status Badges** — RUNNING / DONE / STOPPED indicators per game.

**How the system plays:**
- The cognitive engine uses target-seeking heuristics with exploration noise.
- Each step updates the emotion system (curiosity/hunger drives) and self-model (performance tracking).
- The self-model tracks success rates per game type, enabling the system to "know what it knows."

### 📊 System Monitor (Tab 3)

Real-time observation of all cognitive subsystems.

- **Emotional State** — Current emotion, valence, arousal, intensity with color-coded emotion blend bar.
- **Mood Analysis** — Slow-moving average of emotional state with dominant emotion and stability metric.
- **Emotion History** — Step-by-step timeline of emotional changes.
- **Self-Model Performance** — Per-task success rates, attempt counts, and trend indicators (📈 improving / 📉 declining / ➡️ stable).
- **System Statistics** — Queries, knowledge entries, episodes, facts, corrections, semantic concepts.
- **Knowledge Base Browser** — Browse all knowledge entries with source, relation, and confidence.

### 📋 Logs & Export (Tab 4)

Full activity log with filtering and export capabilities.

- **Activity Log** — Complete timestamped log of all system events (chat, game, system, input, response, errors, exports).
- **Log Filtering** — Filter by category to focus on specific event types.
- **Chat History Export** — Summary of all conversations with confidence scores.
- **Game History Export** — Summary of all game sessions with scores and status.
- **Export Options:**
  - **📄 Export TXT** — Complete text file with all system data: statistics, emotional state, emotion history, self-model performance, knowledge base, chat history, game sessions, and activity log.
  - **📥 Export JSON** — Structured JSON file with all data for programmatic analysis.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML page |
| `POST` | `/api/chat` | Send a chat message (JSON: `{message, sample_text?}`) |
| `GET` | `/api/chat/history` | Retrieve full chat history |
| `POST` | `/api/process` | Process multimodal input (form: `text, structured, task_tag`) |
| `POST` | `/api/game/start` | Start a game (JSON: `{game_type, auto_play?, speed?}`) |
| `POST` | `/api/game/step` | Step a game (JSON: `{session_id}`) |
| `POST` | `/api/game/stop` | Stop a game (JSON: `{session_id}`) |
| `GET` | `/api/game/state` | Get game state (`?session_id=...`) |
| `GET` | `/api/game/sessions` | List all game sessions |
| `GET` | `/api/monitor` | Full system monitoring snapshot |
| `GET` | `/api/stats` | System statistics |
| `GET` | `/api/logs` | Activity log |
| `GET` | `/api/knowledge` | Knowledge base entries |
| `GET` | `/api/context/stats` | Context engine statistics |
| `GET` | `/api/export/text` | Download full text export |
| `GET` | `/api/export/json` | Download full JSON export |
| `POST` | `/api/correct` | Correct knowledge (JSON: `{concept, relation, wrong_target, correct_target}`) |
| `POST` | `/api/feedback` | Provide reward feedback (JSON: `{task_tag, reward, concepts}`) |

---

## API Examples

### Chat API

**Request**:
```bash
curl -X POST http://localhost:5051/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is consciousness?",
    "sample_text": "Consciousness is the state of being aware."
  }'
```

**Response**:
```json
{
  "response": "Consciousness relates to awareness and subjective experience...",
  "dialogue_response": "That's an interesting philosophical question about awareness.",
  "reasoning_trace": [
    "Encoded input text to hypervectors",
    "Recalled 3 relevant memories",
    "Disambiguated context: philosophy",
    "Matched concept: AWARENESS with confidence 0.85"
  ],
  "confidence": 0.85,
  "emotion": "neutral",
  "emotion_blend": {
    "neutral": 0.45,
    "trust": 0.25,
    "anticipation": 0.20,
    "joy": 0.10
  },
  "context_disambiguations": ["philosophy", "cognitive_science"]
}
```

### Game Simulation API

**Start Game**:
```bash
curl -X POST http://localhost:5051/api/game/start \
  -H "Content-Type: application/json" \
  -d '{
    "game_type": "snake",
    "auto_play": true,
    "speed": 200
  }'
```

**Response**:
```json
{
  "status": "started",
  "session_id": "snake_1707627600",
  "message": "Snake game started",
  "game_type": "snake",
  "auto_play": true,
  "speed_ms": 200
}
```

**Get Game State**:
```bash
curl "http://localhost:5051/api/game/state?session_id=snake_1707627600"
```

**Response**:
```json
{
  "session_id": "snake_1707627600",
  "game_type": "snake",
  "status": "running",
  "score": 150,
  "steps": 45,
  "state": {
    "head_pos": [5, 8],
    "food_pos": [7, 8],
    "body_length": 5,
    "direction": "RIGHT"
  },
  "last_action": "ACTION_RIGHT",
  "reasoning": "Rule: {FOOD_RIGHT} → ACTION_RIGHT (confidence: 0.87)"
}
```

**Step Game Manually**:
```bash
curl -X POST http://localhost:5051/api/game/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "snake_1707627600"}'
```

**Response**:
```json
{
  "status": "stepped",
  "session_id": "snake_1707627600",
  "new_score": 160,
  "steps": 46,
  "action_taken": "ACTION_UP",
  "reward": 10,
  "reasoning": "Planner coalition won: path to food requires UP movement",
  "game_over": false
}
```

### System Monitor API

**Get Full Monitor Data**:
```bash
curl "http://localhost:5051/api/monitor"
```

**Response**:
```json
{
  "emotion": {
    "current_emotion": "joy",
    "valence": 0.72,
    "arousal": 0.58,
    "intensity": 0.65,
    "blend": {
      "joy": 0.38,
      "trust": 0.25,
      "anticipation": 0.22,
      "neutral": 0.15
    }
  },
  "mood": {
    "avg_valence": 0.68,
    "avg_arousal": 0.54,
    "dominant_emotion": "joy",
    "stability": 0.12,
    "window_size": 10
  },
  "emotion_history": [
    {"step": 1, "emotion": "neutral", "valence": 0.0, "arousal": 0.0},
    {"step": 2, "emotion": "anticipation", "valence": 0.2, "arousal": 0.6},
    {"step": 3, "emotion": "joy", "valence": 0.7, "arousal": 0.5}
  ],
  "self_model": {
    "snake": {
      "success_rate": 0.83,
      "attempts": 120,
      "trend": "improving"
    },
    "pong": {
      "success_rate": 0.71,
      "attempts": 85,
      "trend": "stable"
    }
  },
  "stats": {
    "queries_processed": 145,
    "knowledge_entries": 67,
    "episodes_stored": 205,
    "facts_learned": 12,
    "corrections_made": 3
  }
}
```

### Knowledge Base API

**Get Knowledge Entries**:
```bash
curl "http://localhost:5051/api/knowledge"
```

**Response**:
```json
{
  "entries": [
    {
      "concept": "FOOD_ABOVE",
      "relation": "CAUSES",
      "target": "ACTION_UP",
      "confidence": 0.87,
      "source": "rule_learning",
      "observations": 103
    },
    {
      "concept": "WALL_AHEAD",
      "relation": "PREVENTS",
      "target": "MOVE_FORWARD",
      "confidence": 0.95,
      "source": "safety_rules",
      "observations": 45
    }
  ],
  "total": 67
}
```

### Export API

**Export as Text**:
```bash
curl "http://localhost:5051/api/export/text" -o nsck_export.txt
```

**Export as JSON**:
```bash
curl "http://localhost:5051/api/export/json" -o nsck_export.json
```

**JSON Structure**:
```json
{
  "exported_at": "2026-02-11T04:59:16Z",
  "system_stats": {
    "queries_processed": 145,
    "knowledge_entries": 67,
    "episodes_stored": 205
  },
  "emotion": {
    "current": "joy",
    "history": [...]
  },
  "self_model": {
    "snake": {"success_rate": 0.83, "trend": "improving"},
    "pong": {"success_rate": 0.71, "trend": "stable"}
  },
  "knowledge_base": [...],
  "chat_history": [...],
  "game_sessions": [...],
  "activity_log": [...]
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Testing Dashboard (Flask)               │
│  ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐ │
│  │ Chat &    │ │  Game    │ │ System  │ │ Logs & │ │
│  │ Test      │ │  Sims    │ │ Monitor │ │ Export │ │
│  └─────┬─────┘ └────┬─────┘ └────┬────┘ └───┬────┘ │
├────────┼────────────┼────────────┼───────────┼──────┤
│        ▼            ▼            ▼           ▼      │
│  ┌──────────────────────────────────────────────┐   │
│  │         Backend API Layer (Flask)             │   │
│  └──────────────────┬───────────────────────────┘   │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │      NSCK Cognitive System Integration       │   │
│  │  ┌────────────────┐ ┌─────────────────────┐  │   │
│  │  │ Knowledge      │ │ Dialogue Manager    │  │   │
│  │  │ Integration    │ │ (Conversation)      │  │   │
│  │  └────────────────┘ └─────────────────────┘  │   │
│  │  ┌────────────────┐ ┌─────────────────────┐  │   │
│  │  │ Emotion System │ │ Self-Model          │  │   │
│  │  │ (Plutchik)     │ │ (Performance)       │  │   │
│  │  └────────────────┘ └─────────────────────┘  │   │
│  │  ┌────────────────┐ ┌─────────────────────┐  │   │
│  │  │ Game Simulator │ │ Language Module      │  │   │
│  │  │ (Snake/Pong/   │ │ (NL Translation)    │  │   │
│  │  │  Maze)         │ │                     │  │   │
│  │  └────────────────┘ └─────────────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Testing

```bash
# Run dashboard tests (25 tests)
python -m pytest nsck-demo/tests/test_testing_dashboard.py -v
```

Tests cover:
- Dashboard page loading
- Chat messaging (with and without sample text, empty rejection, history)
- Game simulations (start/step/stop for all three games, invalid type handling)
- System monitoring (emotion, self-model, stats)
- Data export (text and JSON formats)
- Knowledge correction and feedback
- Full integration flow (chat → game → monitor → export)

---

## Export Format

### Text Export

The text export contains these sections:
1. **System Statistics** — Query counts, knowledge entries, episodes, facts, corrections
2. **Emotional State** — Current emotion, valence, arousal, blend, mood
3. **Emotion History** — Step-by-step emotion trajectory
4. **Self-Model Performance** — Per-task success rates, trends, context breakdown
5. **Knowledge Base** — All knowledge entries with relations and confidence
6. **Chat History** — Full conversation log with reasoning traces and emotions
7. **Game Sessions** — All game sessions with step-by-step action logs
8. **Activity Log** — Complete timestamped event log
9. **Context Engine** — Disambiguation statistics

### JSON Export

Structured equivalent of the text export, suitable for programmatic analysis. Contains:
- `exported_at`, `system_stats`, `knowledge_base`, `emotion`, `self_model`, `chat_history`, `game_sessions`, `activity_log`, `context_engine_stats`
