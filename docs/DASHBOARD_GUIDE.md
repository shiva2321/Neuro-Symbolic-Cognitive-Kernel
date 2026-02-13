# NSCK Unified Dashboard Guide

## Overview

The NSCK Unified Scientific Dashboard is a comprehensive web-based interface for testing, monitoring, and analyzing all capabilities of the NSCK cognitive system. It features enterprise-grade structured logging designed for analysis, debugging, and sharing with reviewers.

## Quick Start

### Launch the Dashboard

```bash
# From project root
cd /workspaces/Node_network
python launch_dashboard.py
```

The dashboard will be available at `http://localhost:5000/`

### Launch Options

```bash  
# Custom port
python launch_dashboard.py --port 8080

# Allow external access (for remote review)
python launch_dashboard.py --host 0.0.0.0

# Debug mode with auto-reload
python launch_dashboard.py --debug
```

## Features

### 1. Interactive Chat & Dialogue
- Natural language interaction with the cognitive system
- Sample text processing and understanding
- Context-aware conversations
- Learned knowledge integration

### 2. Text Learning System
- Upload documents (.txt files)
- Automatic concept extraction
- Relation discovery
- Query learned knowledge
- Export learned facts

### 3. Game Simulations
- **Snake**: Classic snake game with collision detection
- **Pong**: Paddle ball game with physics
- **Maze**: Pathfinding and navigation

**Modes:**
- **Teacher ON**: System uses optimal heuristics and learns
- **Teacher OFF**: System uses only learned policy
- **Auto-Play**: Continuous game loops for extended learning

### 4. Real-Time System Monitor
- Emotion states (Plutchik model)
- Self-model confidence per task 
- Memory statistics (episodic, semantic, working)
- Global workspace competition metrics
- Brain state visualizations

### 5. Decision Traces
- Complete reasoning logs for each decision
- Module competition results
- Confidence scores and explanations
- Causal reasoning chains

### 6. Logging & Export System
**Enterprise-Grade Structured Logging**
- Multi-level categorization
- Severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured metadata for analysis
- Session-based file logging

## Logging System

### Log Categories

- **system**: Initialization, configuration, lifecycle events
- **cognitive**: Decision-making, reasoning, module interactions
- **game**: Game state changes, actions, rewards, scores
- **learning**: Knowledge acquisition, fact extraction, training
- **error**: Exceptions, failures, warnings

### Log Severity Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error events that might still allow operation
- **CRITICAL**: Critical events that may cause system failure

### Log Structure

Each log entry contains:
- **timestamp**: ISO 8601 UTC timestamp
- **category**: Event category
- **severity**: Log level
- **message**: Human-readable description
- **metadata**: Structured data (session_id, scores, actions, etc.)
- **session_age_seconds**: Time since session start

### Example Log Entry (JSON)

```json
{
  "timestamp": "2026-02-12T22:45:33.123456Z",
  "category": "game",
  "severity": "INFO",
  "message": "SNAKE step 42: RIGHT → reward=1.00, score=5",
  "metadata": {
    "session_id": "snake_1707773133000",
    "game_type": "snake",
    "step": 42,
    "action": "RIGHT",
    "reward": 1.0,
    "score": 5,
    "done": false,
    "teacher_active": true
  },
  "session_age_seconds": 15.342
}
```

## Exporting Logs for Analysis

### Via Dashboard UI

1. Navigate to the **Logs & Export** tab
2. Click one of the export buttons in the header:
   - **Export TXT**: Human-readable format for documentation
   - **Export JSON**: Machine-readable for analysis scripts
   - ** Export CSV**: Spreadsheet-compatible format

### Via API Endpoints

```bash
# Export as JSON
curl http://localhost:5000/api/logs/export/json -o logs.json

# Export as CSV  
curl http://localhost:5000/api/logs/export/csv -o logs.csv

# Export as TXT (human-readable)
curl http://localhost:5000/api/logs/export/txt -o session.txt
```

### Via Log Files

All sessions automatically create persistent log files at:
```
/workspaces/Node_network/logs/nsck_session_YYYYMMDD_HHMMSS.log
```

These files include all events in a formatted, timestamped structure.

## Sharing Logs with Reviewers

### Best Practices

**1. For Quick Reviews (Human-Readable)**
- Export as TXT format
- Share the plaintext file
- Includes session summary and all events

**2. For Deep Analysis (Machine-Readable)**
- Export as JSON format
- Provides complete structured data
- Easy to parse with analysis scripts

**3. For Spreadsheet Analysis**
- Export as CSV format
- Open in Excel, Google Sheets, or R/Python
- Filter by category, severity, time ranges

**4. For Complete Sessions**
- Locate the session log file in `/workspaces/Node_network/logs/`
- The filename includes the session timestamp
- Contains Python logger format with all details

### What to Include in Review Packages

A complete review package should include:

1. **Session TXT Export** - Overview and readability
2. **Log Statistics** - From `/api/logs/stats` endpoint
3. **Game Session Data** - If demonstrating game learning
4. **System Metrics** - From `/api/stats` endpoint
5. **Context**: Brief description of what was being tested

### Example Review Document Template

```
NSCK System Review - [Date]
================================

Session Information:
- Duration: [X] seconds
- Total Events: [Y]
- Categories: [breakdown]

Test Objectives:
- [What was being demonstrated/tested]

Key Observations:
- [Important findings from logs]

Attached Files:
- nsck_session_[timestamp].txt - Complete session log
- nsck_logs_[timestamp].json - Structured data export
- [Any other relevant files]

Notes:
- [Additional context, issues, questions]
```

## API Reference

### System & Logging

- `GET /api/stats` - Global system statistics
- `GET /api/logs` - Recent log entries (supports filtering)
- `GET /api/logs/stats` - Logging statistics
- `GET /api/logs/export/json` - Export logs as JSON
- `GET /api/logs/export/csv` - Export logs as CSV
- `GET /api/logs/export/txt` - Export logs as text

**Query Parameters for `/api/logs`:**
- `category` - Filter by category (system, cognitive, game, learning, error)
- `severity` - Filter by severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `limit` - Maximum number of entries (default: 100)

### Chat & Dialogue

- `POST /api/chat` - Send message to system
- `GET /api/chat/history` - Get conversation history

### Text Learning

- `POST /api/learn/upload` - Upload text file for learning
- `POST /api/learn/query` - Query learned knowledge
- `GET /api/learn/stats` - Learning statistics
- `GET /api/learn/export` - Export learned knowledge
- `POST /api/learn/reset` - Reset all learned knowledge

### Game Simulations

- `POST /api/game/start` - Start new game session
- `POST /api/game/step` - Advance game one step
- `POST /api/game/action` - Send manual action
- `POST /api/game/stop` - Stop auto-play
- `GET /api/game/state` - Get current game state
- `GET /api/game/sessions` - List all active sessions

### Monitoring

- `GET /api/monitor` - Real-time system telemetry
- `GET /api/decision_traces` - Get decision reasoning traces
- `GET /api/decision_stats` - Get decision-making statistics
- `GET /api/knowledge` - Get knowledge base contents
- `GET /api/context/stats` - Get context engine statistics

## Analyzing Logs

### Python Analysis Example

```python
import json

# Load exported JSON logs
with open('nsck_logs_20260212_224533.json', 'r') as f:
    data = json.load(f)

# Analyze game performance
game_events = [e for e in data['entries'] if e['category'] == 'game']
scores = [e['metadata'].get('score', 0) for e in game_events if 'score' in e['metadata']]

print(f"Average score: {sum(scores) / len(scores):.2f}")
print(f"Max score: {max(scores)}")

# Find errors
errors = [e for e in data['entries'] if e['severity'] in ['ERROR', 'CRITICAL']]
print(f"Total errors: {len(errors)}")
for error in errors:
    print(f"  {error['timestamp']}: {error['message']}")
```

### Filter Logs by Time Range

```python
from datetime import datetime, timedelta

# Filter last 5 minutes
cutoff = datetime.now() - timedelta(minutes=5)
recent = [e for e in data['entries'] 
          if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) > cutoff]
```

### Generate Statistics

```python
from collections import Counter

# Count events by category
categories = Counter(e['category'] for e in data['entries'])
print("Events by category:")
for cat, count in categories.most_common():
    print(f"  {cat}: {count}")

# Count by severity
severities = Counter(e['severity'] for e in data['entries'])
print("\nEvents by severity:")
for sev, count in severities.most_common():
    print(f"  {sev}: {count}")
```

## Troubleshooting

### Dashboard won't start

**Check Python environment:**
```bash
cd /workspaces/Node_network
.venv/bin/python launch_dashboard.py
```

**Install missing dependencies:**
```bash
pip install -r requirements.txt
pip install flask flask-cors
```

### Logs not being saved

Check log directory permissions:
```bash
ls -la /workspaces/Node_network/logs/
```

Create directory if missing:
```bash
mkdir -p /workspaces/Node_network/logs
```

### High memory usage

Logs are stored in a circular buffer (default: 10,000 entries).
To reduce memory:
1. Export logs more frequently
2. Restart dashboard to clear buffer
3. Reduce the `max_entries` parameter in `StructuredLogger.__init__()`

## Performance Notes

- **Log Buffer**: 10,000 entries in memory (auto-rotates)
- **File Logging**: All events persisted to disk
- **Game Loop**: 200ms minimum interval (configurable)
- **API Response**: < 50ms for most endpoints
- **Export Speed**: ~5000 entries/second for JSON export

## Advanced Configuration

### Custom Log Directory

Edit `unified_dashboard.py`:
```python
# In StructuredLogger.__init__()
log_dir = Path("/your/custom/path/logs")
```

### Adjust Log Retention

```python
# In __init__()
_structured_logger = StructuredLogger(max_entries=50000)  # default: 10000
```

### Change Log Level

```python
# In StructuredLogger.__init__()
fh.setLevel(logging.INFO)  # Change from DEBUG to INFO
```

## Support & Feedback

For issues, questions, or feature requests:
1. Export logs showing the issue
2. Include system metrics (`/api/stats`)
3. Document reproduction steps
4. Share via the standard review package format

---

**Last Updated**: February 2026  
**Dashboard Version**: 2.0 (Unified)  
**Compatible With**: NSCK V2
