"""
NSCK AI Dashboard — Full Monitoring & Chat Interface
=====================================================

What this does:
    A Flask web dashboard that is wired to every backend component of the
    NSCK AI engine.  It provides:

    * **Chat** — talk to the AI and see its responses
    * **Training** — feed data and watch knowledge grow in real time
    * **Traces** — inspect the full glass-box reasoning chain for any query
    * **Knowledge** — browse concepts, relations, and causal rules
    * **Telemetry** — live stats, emotion state, memory usage
    * **Logs** — structured log stream

How it works:
    The dashboard is a single Flask application with JSON API endpoints.
    The frontend is an embedded HTML page with vanilla JavaScript (no build
    step required).  All backend state comes from the ``NSCKAIEngine``.

Usage::

    python -m nsck_ai_model.dashboard                # default: port 5090
    python -m nsck_ai_model.dashboard --port 8080
    python -m nsck_ai_model.dashboard --train seed   # pre-train before start
"""

import os
import sys
import json
import time
import logging
import argparse
from collections import deque
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, Response

from nsck_ai_model.ai_engine import NSCKAIEngine
from nsck_ai_model.data_pipeline import DataPipeline

logger = logging.getLogger("nsck_ai.dashboard")

app = Flask(__name__)

# ── Lazy globals ──────────────────────────────────────────────────────────
_engine: Optional[NSCKAIEngine] = None
_pipeline: Optional[DataPipeline] = None
_log_buffer: deque = deque(maxlen=500)


def _get_engine() -> NSCKAIEngine:
    global _engine, _pipeline
    if _engine is None:
        _engine = NSCKAIEngine()
        _pipeline = DataPipeline(_engine)
    return _engine


def _get_pipeline() -> DataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline(_get_engine())
    return _pipeline


def _log(category: str, message: str, data: Optional[Dict] = None):
    entry = {
        'timestamp': time.time(),
        'category': category,
        'message': message,
        'data': data or {},
    }
    _log_buffer.append(entry)


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return _DASHBOARD_HTML


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with the AI engine."""
    body = request.get_json(force=True, silent=True) or {}
    message = body.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    engine = _get_engine()
    result = engine.chat(message)
    _log("chat", f"User: {message[:100]}", {
        "confidence": result["confidence"],
        "emotion": result["emotion"]["emotion"],
    })
    return jsonify(result)


@app.route("/api/chat/history")
def api_chat_history():
    """Get conversation history."""
    return jsonify(_get_engine().get_conversation_history())


@app.route("/api/train", methods=["POST"])
def api_train():
    """Train on text input."""
    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    engine = _get_engine()
    result = engine.train_on_text(text)
    _log("train", f"Trained on {len(text)} chars", result)
    return jsonify(result)


@app.route("/api/train/seed", methods=["POST"])
def api_train_seed():
    """Train on the built-in seed corpus."""
    pipeline = _get_pipeline()
    result = pipeline.train_from_seed()
    _log("train", "Seed corpus training complete", result)
    return jsonify(result)


@app.route("/api/train/hf", methods=["POST"])
def api_train_hf():
    """Train on HuggingFace data."""
    body = request.get_json(force=True, silent=True) or {}
    limit = body.get("limit", 200)
    pipeline = _get_pipeline()
    result = pipeline.train_from_hf(limit=limit)
    _log("train", f"HF training complete (limit={limit})", result)
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    """Get system statistics."""
    return jsonify(_get_engine().get_system_stats())


@app.route("/api/knowledge")
def api_knowledge():
    """Export all knowledge."""
    return jsonify(_get_engine().export_knowledge())


@app.route("/api/knowledge/concepts")
def api_concepts():
    """List all concepts."""
    engine = _get_engine()
    concepts = []
    for name, c in engine.knowledge.concepts.items():
        concepts.append({
            'name': name,
            'frequency': c.frequency,
            'source_count': len(c.source_sentences),
        })
    concepts.sort(key=lambda x: x['frequency'], reverse=True)
    return jsonify(concepts)


@app.route("/api/knowledge/relations")
def api_relations():
    """List all relations."""
    engine = _get_engine()
    rels = []
    for r in engine.knowledge.relations[:200]:
        rels.append({
            'source': r.source_concept,
            'target': r.target_concept,
            'sentence': r.sentence,
            'weight': r.weight,
            'evidence': r.evidence_count,
        })
    return jsonify(rels)


@app.route("/api/knowledge/rules")
def api_rules():
    """List all causal rules."""
    engine = _get_engine()
    rules = []
    for r in engine.causal_rules.rules[:100]:
        rules.append({
            'antecedent': r.antecedent,
            'consequent': r.consequent,
            'sentence': r.sentence,
            'evidence': r.evidence,
            'strength': round(r.strength, 3),
        })
    return jsonify(rules)


@app.route("/api/emotion")
def api_emotion():
    """Get current emotional state."""
    return jsonify(_get_engine().emotion.get_state())


@app.route("/api/logs")
def api_logs():
    """Get recent logs."""
    limit = request.args.get("limit", 100, type=int)
    category = request.args.get("category", None)
    logs = list(_log_buffer)
    if category:
        logs = [l for l in logs if l['category'] == category]
    return jsonify(logs[-limit:])


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the engine."""
    _get_engine().reset()
    _log("system", "Engine reset")
    return jsonify({"status": "reset"})


@app.route("/api/health")
def api_health():
    """Health check endpoint for monitoring."""
    engine = _get_engine()
    stats = engine.get_system_stats()
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "concepts": stats["knowledge"]["total_concepts"],
        "relations": stats["knowledge"]["total_relations"],
        "episodes": stats["knowledge"]["total_episodes"],
        "queries": stats["engine"]["query_count"],
    })


# ── Dashboard HTML (inline, no external dependencies) ─────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSCK AI Dashboard</title>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --text: #c9d1d9; --text2: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --orange: #d29922; --red: #f85149;
  --border: #30363d;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); }
.header { background: var(--bg2); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; color: var(--accent); }
.header .stats { font-size: 12px; color: var(--text2); margin-left: auto; }
.tabs { display: flex; border-bottom: 1px solid var(--border);
        background: var(--bg2); padding: 0 16px; }
.tab { padding: 10px 20px; cursor: pointer; color: var(--text2);
       border-bottom: 2px solid transparent; font-size: 13px; }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab:hover { color: var(--text); }
.panel { display: none; padding: 20px; max-width: 1200px; margin: 0 auto; }
.panel.active { display: block; }

/* Chat */
.chat-container { display: flex; flex-direction: column; height: calc(100vh - 140px); }
.messages { flex: 1; overflow-y: auto; padding: 10px; }
.msg { margin: 8px 0; padding: 10px 14px; border-radius: 8px; max-width: 80%; }
.msg.user { background: var(--bg3); margin-left: auto; }
.msg.ai { background: var(--bg2); border: 1px solid var(--border); }
.msg .meta { font-size: 11px; color: var(--text2); margin-top: 4px; }
.chat-input { display: flex; gap: 8px; padding: 10px 0; }
.chat-input input { flex: 1; padding: 10px; background: var(--bg2);
                    border: 1px solid var(--border); border-radius: 6px;
                    color: var(--text); font-size: 14px; }
.chat-input button { padding: 10px 20px; background: var(--accent);
                     border: none; border-radius: 6px; color: #fff;
                     cursor: pointer; font-size: 14px; }

/* Training */
.train-section { margin-bottom: 20px; }
.train-section h3 { color: var(--accent); margin-bottom: 10px; font-size: 14px; }
textarea { width: 100%; height: 100px; background: var(--bg2);
           border: 1px solid var(--border); border-radius: 6px;
           color: var(--text); padding: 10px; font-size: 13px;
           resize: vertical; }
.btn { padding: 8px 16px; background: var(--accent); border: none;
       border-radius: 6px; color: #fff; cursor: pointer; margin: 4px; font-size: 13px; }
.btn:hover { opacity: 0.9; }
.btn.secondary { background: var(--bg3); border: 1px solid var(--border); }
.result-box { background: var(--bg3); border: 1px solid var(--border);
              border-radius: 6px; padding: 12px; margin-top: 8px;
              font-size: 12px; font-family: monospace; white-space: pre-wrap;
              max-height: 300px; overflow-y: auto; }

/* Trace */
.trace-step { background: var(--bg2); border: 1px solid var(--border);
              border-radius: 6px; padding: 10px; margin: 6px 0; }
.trace-step .stage { color: var(--accent); font-weight: bold; font-size: 12px; }
.trace-step .action { color: var(--text); font-size: 13px; }
.trace-step .details { color: var(--text2); font-size: 11px; margin-top: 4px; }

/* Knowledge */
.knowledge-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card { background: var(--bg2); border: 1px solid var(--border);
        border-radius: 8px; padding: 16px; }
.card h3 { color: var(--accent); font-size: 14px; margin-bottom: 8px; }
.card-list { max-height: 400px; overflow-y: auto; }
.card-item { padding: 6px 0; border-bottom: 1px solid var(--border);
             font-size: 12px; }

/* Telemetry */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin-bottom: 20px; }
.stat-card { background: var(--bg2); border: 1px solid var(--border);
             border-radius: 8px; padding: 16px; text-align: center; }
.stat-value { font-size: 28px; font-weight: bold; color: var(--green); }
.stat-label { font-size: 11px; color: var(--text2); margin-top: 4px; }
.emotion-bar { display: flex; gap: 4px; align-items: end; height: 60px; margin-top: 8px; }
.emotion-bar .bar { flex: 1; background: var(--accent); border-radius: 3px 3px 0 0;
                    min-width: 8px; transition: height 0.3s; }
.emotion-bar .bar-label { font-size: 9px; color: var(--text2);
                          text-align: center; margin-top: 2px; }

/* Logs */
.log-entry { font-family: monospace; font-size: 11px; padding: 4px 8px;
             border-bottom: 1px solid var(--border); }
.log-entry .cat { color: var(--accent); }
.log-entry .ts { color: var(--text2); }
</style>
</head>
<body>

<div class="header">
  <h1>🧠 NSCK AI Dashboard</h1>
  <span style="font-size:12px;color:var(--text2)">Zero-Hardcode • Glass-Box • VSA-Powered</span>
  <div class="stats" id="header-stats">Loading...</div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('chat')">💬 Chat</div>
  <div class="tab" onclick="showTab('train')">📚 Training</div>
  <div class="tab" onclick="showTab('trace')">🔍 Traces</div>
  <div class="tab" onclick="showTab('knowledge')">🧩 Knowledge</div>
  <div class="tab" onclick="showTab('telemetry')">📊 Telemetry</div>
  <div class="tab" onclick="showTab('logs')">📋 Logs</div>
</div>

<!-- Chat Panel -->
<div id="panel-chat" class="panel active">
  <div class="chat-container">
    <div class="messages" id="messages"></div>
    <div class="chat-input">
      <input type="text" id="chat-input" placeholder="Type a message..."
             onkeydown="if(event.key==='Enter')sendChat()">
      <button onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<!-- Training Panel -->
<div id="panel-train" class="panel">
  <div class="train-section">
    <h3>Quick Train</h3>
    <button class="btn" onclick="trainSeed()">🌱 Train on Seed Corpus</button>
    <button class="btn secondary" onclick="trainHF()">🤗 Train on WikiText (HF)</button>
  </div>
  <div class="train-section">
    <h3>Custom Training Text</h3>
    <textarea id="train-text" placeholder="Paste text here to teach the AI..."></textarea>
    <button class="btn" onclick="trainCustom()">Train on Text</button>
  </div>
  <div class="train-section">
    <h3>Training Result</h3>
    <div class="result-box" id="train-result">No training yet.</div>
  </div>
</div>

<!-- Trace Panel -->
<div id="panel-trace" class="panel">
  <div class="train-section">
    <h3>Query with Full Trace</h3>
    <div class="chat-input">
      <input type="text" id="trace-input" placeholder="Type a query to trace..."
             onkeydown="if(event.key==='Enter')traceQuery()">
      <button onclick="traceQuery()">Trace</button>
    </div>
  </div>
  <div id="trace-response" style="margin:10px 0;padding:12px;background:var(--bg2);border-radius:6px;display:none">
  </div>
  <div id="trace-steps"></div>
</div>

<!-- Knowledge Panel -->
<div id="panel-knowledge" class="panel">
  <div class="knowledge-grid">
    <div class="card">
      <h3>Concepts <button class="btn secondary" onclick="loadConcepts()" style="float:right;padding:4px 8px;font-size:11px">Refresh</button></h3>
      <div class="card-list" id="concept-list">Click Refresh to load.</div>
    </div>
    <div class="card">
      <h3>Relations <button class="btn secondary" onclick="loadRelations()" style="float:right;padding:4px 8px;font-size:11px">Refresh</button></h3>
      <div class="card-list" id="relation-list">Click Refresh to load.</div>
    </div>
    <div class="card">
      <h3>Causal Rules <button class="btn secondary" onclick="loadRules()" style="float:right;padding:4px 8px;font-size:11px">Refresh</button></h3>
      <div class="card-list" id="rule-list">Click Refresh to load.</div>
    </div>
    <div class="card">
      <h3>Full Export <button class="btn secondary" onclick="exportKnowledge()" style="float:right;padding:4px 8px;font-size:11px">Export</button></h3>
      <div class="result-box" id="export-box" style="max-height:400px">Click Export to dump.</div>
    </div>
  </div>
</div>

<!-- Telemetry Panel -->
<div id="panel-telemetry" class="panel">
  <div class="stat-grid" id="stat-grid"></div>
  <div class="card" style="margin-top:16px">
    <h3>Emotion State</h3>
    <div id="emotion-display" style="display:flex;gap:20px;align-items:center;padding:10px 0">
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <h3>Raw Stats</h3>
    <div class="result-box" id="raw-stats">Loading...</div>
  </div>
</div>

<!-- Logs Panel -->
<div id="panel-logs" class="panel">
  <button class="btn secondary" onclick="loadLogs()">Refresh Logs</button>
  <div id="log-container" style="margin-top:10px;max-height:calc(100vh - 200px);overflow-y:auto">
  </div>
</div>

<script>
// Tab switching
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'telemetry') refreshStats();
  if (name === 'logs') loadLogs();
}

// Chat
let lastTrace = null;
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addMessage('user', msg);
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await r.json();
    lastTrace = data.trace;
    addMessage('ai', data.response,
      `confidence: ${data.confidence} | emotion: ${data.emotion.emotion} | ` +
      `latency: ${data.latency_ms}ms | trace: ${data.trace.step_count} steps`);
    refreshHeaderStats();
  } catch(e) { addMessage('ai', 'Error: ' + e.message); }
}

function addMessage(role, text, meta) {
  const div = document.getElementById('messages');
  const m = document.createElement('div');
  m.className = 'msg ' + role;
  m.innerHTML = text + (meta ? '<div class="meta">' + meta + '</div>' : '');
  div.appendChild(m);
  div.scrollTop = div.scrollHeight;
}

// Training
async function trainSeed() {
  document.getElementById('train-result').textContent = 'Training on seed corpus...';
  const r = await fetch('/api/train/seed', {method: 'POST'});
  const data = await r.json();
  document.getElementById('train-result').textContent = JSON.stringify(data, null, 2);
  refreshHeaderStats();
}

async function trainHF() {
  document.getElementById('train-result').textContent = 'Streaming from HuggingFace (may take a moment)...';
  const r = await fetch('/api/train/hf', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({limit: 200})});
  const data = await r.json();
  document.getElementById('train-result').textContent = JSON.stringify(data, null, 2);
  refreshHeaderStats();
}

async function trainCustom() {
  const text = document.getElementById('train-text').value.trim();
  if (!text) return;
  const r = await fetch('/api/train', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text})});
  const data = await r.json();
  document.getElementById('train-result').textContent = JSON.stringify(data, null, 2);
  refreshHeaderStats();
}

// Trace
async function traceQuery() {
  const input = document.getElementById('trace-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  const r = await fetch('/api/chat', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg})});
  const data = await r.json();
  // Show response
  const respDiv = document.getElementById('trace-response');
  respDiv.style.display = 'block';
  respDiv.innerHTML = '<strong>Response:</strong> ' + data.response +
    '<br><span style="color:var(--text2);font-size:12px">confidence=' + data.confidence +
    ' emotion=' + data.emotion.emotion + ' latency=' + data.latency_ms + 'ms</span>';
  // Show trace steps
  const stepsDiv = document.getElementById('trace-steps');
  stepsDiv.innerHTML = '<h3 style="color:var(--accent);margin:12px 0 8px">Reasoning Chain (' +
    data.trace.step_count + ' steps, ' + data.trace.total_ms.toFixed(1) + 'ms total)</h3>';
  data.trace.steps.forEach((step, i) => {
    const div = document.createElement('div');
    div.className = 'trace-step';
    let details = '';
    if (step.outputs) {
      details = Object.entries(step.outputs).map(([k,v]) =>
        '<br>&nbsp;&nbsp;→ <strong>' + k + ':</strong> ' + JSON.stringify(v).substring(0, 200)
      ).join('');
    }
    div.innerHTML = '<span class="stage">[' + (i+1) + '] ' + step.stage + '</span> ' +
      '<span class="action">' + step.action + '</span> ' +
      '<span style="color:var(--orange)">(' + step.duration_ms.toFixed(1) + 'ms)</span>' +
      '<div class="details">' + details + '</div>';
    stepsDiv.appendChild(div);
  });
}

// Knowledge
async function loadConcepts() {
  const r = await fetch('/api/knowledge/concepts');
  const data = await r.json();
  document.getElementById('concept-list').innerHTML = data.map(c =>
    '<div class="card-item"><strong>' + c.name + '</strong> — freq: ' +
    c.frequency + ', sources: ' + c.source_count + '</div>'
  ).join('');
}

async function loadRelations() {
  const r = await fetch('/api/knowledge/relations');
  const data = await r.json();
  document.getElementById('relation-list').innerHTML = data.map(rel =>
    '<div class="card-item"><strong>' + rel.source + '</strong> ↔ <strong>' +
    rel.target + '</strong><br><span style="color:var(--text2);font-size:11px">' +
    rel.sentence + ' (w=' + rel.weight.toFixed(1) + ', ev=' + rel.evidence + ')</span></div>'
  ).join('');
}

async function loadRules() {
  const r = await fetch('/api/knowledge/rules');
  const data = await r.json();
  document.getElementById('rule-list').innerHTML = data.map(rule =>
    '<div class="card-item">[' + rule.antecedent.join(', ') + '] → [' +
    rule.consequent.join(', ') + ']<br><span style="color:var(--text2);font-size:11px">' +
    rule.sentence + ' (str=' + rule.strength + ', ev=' + rule.evidence + ')</span></div>'
  ).join('');
}

async function exportKnowledge() {
  const r = await fetch('/api/knowledge');
  const data = await r.json();
  document.getElementById('export-box').textContent = JSON.stringify(data, null, 2);
}

// Telemetry
async function refreshStats() {
  const r = await fetch('/api/stats');
  const data = await r.json();
  const grid = document.getElementById('stat-grid');
  const stats = [
    {label: 'Concepts', value: data.knowledge.total_concepts, color: 'var(--green)'},
    {label: 'Relations', value: data.knowledge.total_relations, color: 'var(--accent)'},
    {label: 'Episodes', value: data.knowledge.total_episodes, color: 'var(--orange)'},
    {label: 'Vocabulary', value: data.encoder.vocabulary_size, color: 'var(--green)'},
    {label: 'Causal Rules', value: data.causal_rules.total_rules, color: 'var(--accent)'},
    {label: 'Abstractions', value: data.abstractions.total_abstractions, color: 'var(--orange)'},
    {label: 'Queries', value: data.engine.query_count, color: 'var(--green)'},
    {label: 'Bigrams', value: data.generator.bigram_vocab, color: 'var(--accent)'},
    {label: 'Texts Trained', value: data.training.texts_trained, color: 'var(--orange)'},
    {label: 'Training Time', value: data.training.training_time_s.toFixed(1) + 's', color: 'var(--green)'},
  ];
  grid.innerHTML = stats.map(s =>
    '<div class="stat-card"><div class="stat-value" style="color:' + s.color + '">' +
    s.value + '</div><div class="stat-label">' + s.label + '</div></div>'
  ).join('');
  // Emotion
  const em = data.emotion;
  document.getElementById('emotion-display').innerHTML =
    '<div style="font-size:32px">😊</div>' +
    '<div><strong>Current:</strong> ' + em.emotion +
    '<br>Valence: ' + em.valence + ' | Arousal: ' + em.arousal +
    '<br>Blend: ' + Object.entries(em.blend || {}).map(([k,v]) => k + ': ' + v).join(', ') + '</div>';
  document.getElementById('raw-stats').textContent = JSON.stringify(data, null, 2);
}

async function refreshHeaderStats() {
  try {
    const r = await fetch('/api/stats');
    const data = await r.json();
    document.getElementById('header-stats').textContent =
      'C:' + data.knowledge.total_concepts +
      ' R:' + data.knowledge.total_relations +
      ' E:' + data.knowledge.total_episodes +
      ' Q:' + data.engine.query_count;
  } catch(e) {}
}

// Logs
async function loadLogs() {
  const r = await fetch('/api/logs');
  const data = await r.json();
  document.getElementById('log-container').innerHTML = data.reverse().map(l =>
    '<div class="log-entry"><span class="ts">' +
    new Date(l.timestamp * 1000).toLocaleTimeString() + '</span> ' +
    '<span class="cat">[' + l.category + ']</span> ' + l.message + '</div>'
  ).join('');
}

// Init
refreshHeaderStats();
setInterval(refreshHeaderStats, 5000);
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSCK AI Dashboard")
    parser.add_argument("--port", type=int, default=5090)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--train", choices=["none", "seed", "full"],
                        default="none",
                        help="Pre-train before starting the dashboard")
    args = parser.parse_args()

    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO)

    if args.train != "none":
        engine = _get_engine()
        pipeline = _get_pipeline()
        if args.train == "seed":
            pipeline.train_from_seed()
        elif args.train == "full":
            pipeline.train_general(hf_limit=200)

    print(f"\n  NSCK AI Dashboard → http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
