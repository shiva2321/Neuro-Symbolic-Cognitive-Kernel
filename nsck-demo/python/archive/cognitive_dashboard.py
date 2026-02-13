"""
NSCK Cognitive Dashboard
========================
Standalone web dashboard for the NSCK cognitive system.

Features:
- Submit text, image, or structured data as input
- Monitor cognitive processing in real-time
- View memory, knowledge, reasoning traces
- Export all system data and logs as JSON/ZIP
"""

import os
import io
import json
import time
import zipfile
import logging
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, render_template_string

# Cognitive system imports
import sys
sys.path.insert(0, os.path.dirname(__file__))

from knowledge_integration import KnowledgeIntegration, CognitiveResponse
from multimodal_processor import MultimodalInput
from context_engine import ContextFrame
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nsck_dashboard")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global cognitive system instance
_system = None
_activity_log: deque = deque(maxlen=500)


def get_system() -> KnowledgeIntegration:
    """Lazy-init the cognitive system."""
    global _system
    if _system is None:
        _system = KnowledgeIntegration()
        _log_activity("system", "Cognitive system initialised.")
    return _system


def _log_activity(category: str, message: str, data: Dict[str, Any] = None):
    """Record an activity log entry."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "category": category,
        "message": message,
    }
    if data:
        entry["data"] = data
    _activity_log.append(entry)
    logger.info(f"[{category}] {message}")


# ---------------------------------------------------------------------------
# HTML dashboard (self-contained single-page app)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSCK Cognitive Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #c9d1d9; --accent: #58a6ff; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }
  .header {
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: center; gap: 20px;
  }
  .header h1 { font-size: 18px; color: var(--accent); }
  .header .status { font-size: 13px; color: var(--green); }
  .main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; min-height: 200px;
  }
  .panel h2 { font-size: 14px; color: var(--accent); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .full-width { grid-column: 1 / -1; }
  textarea { width: 100%; height: 80px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-family: inherit; resize: vertical; }
  input[type="file"] { color: var(--text); }
  button {
    background: var(--accent); color: #000; border: none; padding: 8px 16px;
    border-radius: 4px; cursor: pointer; font-weight: 600; margin: 4px;
  }
  button:hover { opacity: 0.85; }
  button.secondary { background: var(--border); color: var(--text); }
  button.danger { background: var(--red); color: #fff; }
  .response-box {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 12px; margin-top: 8px; white-space: pre-wrap; font-size: 13px;
    max-height: 300px; overflow-y: auto;
  }
  .log-entry { padding: 4px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
  .log-entry .ts { color: var(--yellow); margin-right: 8px; }
  .log-entry .cat { color: var(--purple); margin-right: 8px; font-weight: 600; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }
  .stat-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 10px; text-align: center;
  }
  .stat-card .val { font-size: 24px; font-weight: 700; color: var(--accent); }
  .stat-card .lbl { font-size: 11px; color: #8b949e; text-transform: uppercase; }
  .trace-item { padding: 4px 8px; margin: 2px 0; background: var(--bg); border-radius: 3px; font-size: 12px; }
  .disambiguation { padding: 6px 8px; margin: 3px 0; background: #1c2128; border-left: 3px solid var(--accent); border-radius: 2px; font-size: 12px; }
  .knowledge-entry { padding: 4px 8px; margin: 2px 0; font-size: 12px; }
  .export-bar { display: flex; gap: 8px; margin-top: 12px; }
  .tabs { display: flex; gap: 2px; margin-bottom: 8px; }
  .tab { padding: 6px 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px 4px 0 0; cursor: pointer; font-size: 12px; }
  .tab.active { background: var(--panel); border-bottom-color: var(--panel); color: var(--accent); }
  #loading { display: none; color: var(--yellow); margin: 8px 0; }
</style>
</head>
<body>

<div class="header">
  <h1>🧠 NSCK Cognitive Dashboard</h1>
  <div class="status" id="status">● System Ready</div>
  <div style="flex:1"></div>
  <button class="secondary" onclick="refreshStats()">↻ Refresh</button>
  <button class="secondary" onclick="exportJSON()">📥 Export JSON</button>
  <button class="secondary" onclick="exportZIP()">📦 Export ZIP</button>
</div>

<div class="main">

  <!-- INPUT PANEL -->
  <div class="panel">
    <h2>📝 Input</h2>
    <textarea id="text-input" placeholder="Enter text input here... e.g. 'The red rose is beautiful in the garden'"></textarea>
    <div style="margin: 8px 0; display: flex; gap: 12px; align-items: center;">
      <label>📎 Image: <input type="file" id="image-input" accept="image/*"></label>
      <label>🎤 Audio: <input type="file" id="audio-input" accept="audio/*"></label>
    </div>
    <div style="margin: 8px 0;">
      <textarea id="structured-input" placeholder='Structured data (JSON): e.g. {"location":"garden","type":"flower"}' style="height:40px;font-size:12px;"></textarea>
    </div>
    <div style="margin: 8px 0;">
      <label>Task tag: <input type="text" id="task-tag" value="general" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px 8px;border-radius:4px;width:120px;"></label>
    </div>
    <button onclick="submitInput()">▶ Process Input</button>
    <button class="secondary" onclick="clearAll()">Clear</button>
    <div id="loading">⏳ Processing...</div>
  </div>

  <!-- RESPONSE PANEL -->
  <div class="panel">
    <h2>💬 Response</h2>
    <div class="response-box" id="response-box">Submit an input to see the cognitive response here.</div>
    <div style="margin-top:8px">
      <strong style="font-size:12px;color:var(--yellow);">Confidence:</strong>
      <span id="confidence" style="font-size:13px;">—</span>
    </div>
  </div>

  <!-- REASONING TRACE -->
  <div class="panel">
    <h2>🔍 Reasoning Trace</h2>
    <div id="trace-box" class="response-box" style="max-height:250px;">
      No reasoning trace yet.
    </div>
  </div>

  <!-- DISAMBIGUATIONS -->
  <div class="panel">
    <h2>🎯 Context Disambiguations</h2>
    <div id="disambig-box" class="response-box" style="max-height:250px;">
      No disambiguations yet.
    </div>
  </div>

  <!-- SYSTEM STATS -->
  <div class="panel full-width">
    <h2>📊 System Statistics</h2>
    <div class="stats-grid" id="stats-grid">
      <div class="stat-card"><div class="val" id="stat-queries">0</div><div class="lbl">Queries</div></div>
      <div class="stat-card"><div class="val" id="stat-knowledge">0</div><div class="lbl">Knowledge Entries</div></div>
      <div class="stat-card"><div class="val" id="stat-episodes">0</div><div class="lbl">Episodes</div></div>
      <div class="stat-card"><div class="val" id="stat-facts">0</div><div class="lbl">Facts Learned</div></div>
      <div class="stat-card"><div class="val" id="stat-corrections">0</div><div class="lbl">Corrections</div></div>
      <div class="stat-card"><div class="val" id="stat-concepts">0</div><div class="lbl">Semantic Concepts</div></div>
    </div>
  </div>

  <!-- ACTIVITY LOG -->
  <div class="panel full-width">
    <h2>📋 Activity Log</h2>
    <div id="log-box" class="response-box" style="max-height:300px;">
      Waiting for activity...
    </div>
  </div>

</div>

<script>
const API = '';  // same origin

async function submitInput() {
  const text = document.getElementById('text-input').value.trim();
  const structuredRaw = document.getElementById('structured-input').value.trim();
  const taskTag = document.getElementById('task-tag').value.trim() || 'general';
  const imageFile = document.getElementById('image-input').files[0];
  const audioFile = document.getElementById('audio-input').files[0];

  if (!text && !structuredRaw && !imageFile && !audioFile) {
    alert('Please provide at least one input.');
    return;
  }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('status').textContent = '⏳ Processing...';
  document.getElementById('status').style.color = '#d29922';

  const formData = new FormData();
  if (text) formData.append('text', text);
  if (structuredRaw) formData.append('structured', structuredRaw);
  formData.append('task_tag', taskTag);
  if (imageFile) formData.append('image', imageFile);
  if (audioFile) formData.append('audio', audioFile);

  try {
    const resp = await fetch(API + '/api/process', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('response-box').textContent = '❌ Error: ' + data.error;
    } else {
      // Response
      document.getElementById('response-box').textContent = data.answer || '(no answer)';
      document.getElementById('confidence').textContent =
        (data.confidence * 100).toFixed(1) + '%';

      // Trace
      const traceBox = document.getElementById('trace-box');
      traceBox.innerHTML = '';
      (data.reasoning_trace || []).forEach(t => {
        const d = document.createElement('div');
        d.className = 'trace-item';
        d.textContent = t;
        traceBox.appendChild(d);
      });

      // Disambiguations
      const dBox = document.getElementById('disambig-box');
      dBox.innerHTML = '';
      (data.disambiguations || []).forEach(d => {
        const el = document.createElement('div');
        el.className = 'disambiguation';
        el.innerHTML = `<strong>'${d.concept}'</strong> → <em>${d.meaning}</em> `
          + `<span style="color:#8b949e">(domain: ${d.context_domain}, conf: ${(d.confidence*100).toFixed(0)}%)</span>`
          + `<br><small>${d.explanation}</small>`;
        dBox.appendChild(el);
      });

      // Stats
      if (data.stats) updateStats(data.stats);
    }
  } catch(e) {
    document.getElementById('response-box').textContent = '❌ Network error: ' + e.message;
  }

  document.getElementById('loading').style.display = 'none';
  document.getElementById('status').textContent = '● System Ready';
  document.getElementById('status').style.color = '#3fb950';
  refreshLog();
}

function updateStats(s) {
  document.getElementById('stat-queries').textContent = s.queries_processed || 0;
  document.getElementById('stat-knowledge').textContent = s.knowledge_entries || 0;
  document.getElementById('stat-episodes').textContent = s.episodes_recorded || 0;
  document.getElementById('stat-facts').textContent = s.facts_learned || 0;
  document.getElementById('stat-corrections').textContent = s.corrections_made || 0;
  document.getElementById('stat-concepts').textContent = s.semantic_concepts || 0;
}

async function refreshStats() {
  try {
    const resp = await fetch(API + '/api/stats');
    const data = await resp.json();
    updateStats(data);
  } catch(e) {}
  refreshLog();
}

async function refreshLog() {
  try {
    const resp = await fetch(API + '/api/logs');
    const data = await resp.json();
    const box = document.getElementById('log-box');
    box.innerHTML = '';
    (data.logs || []).reverse().forEach(e => {
      const d = document.createElement('div');
      d.className = 'log-entry';
      d.innerHTML = `<span class="ts">${e.timestamp.slice(11,19)}</span>`
        + `<span class="cat">[${e.category}]</span> ${e.message}`;
      box.appendChild(d);
    });
  } catch(e) {}
}

async function exportJSON() {
  window.location.href = API + '/api/export/json';
}

async function exportZIP() {
  window.location.href = API + '/api/export/zip';
}

function clearAll() {
  document.getElementById('text-input').value = '';
  document.getElementById('structured-input').value = '';
  document.getElementById('image-input').value = '';
  document.getElementById('audio-input').value = '';
  document.getElementById('response-box').textContent = 'Submit an input to see the cognitive response here.';
  document.getElementById('confidence').textContent = '—';
  document.getElementById('trace-box').innerHTML = 'No reasoning trace yet.';
  document.getElementById('disambig-box').innerHTML = 'No disambiguations yet.';
}

// Initial load
refreshStats();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/process", methods=["POST"])
def api_process():
    """Process multimodal input through the cognitive system."""
    try:
        system = get_system()

        text = request.form.get("text", None)
        task_tag = request.form.get("task_tag", "general")
        structured_raw = request.form.get("structured", None)

        # Parse structured data
        structured = None
        if structured_raw:
            try:
                structured = json.loads(structured_raw)
            except json.JSONDecodeError:
                structured = {"raw": structured_raw}

        # Handle image upload
        image = None
        if "image" in request.files:
            img_file = request.files["image"]
            if img_file.filename:
                img_bytes = img_file.read()
                # Decode to numpy array (lightweight — just get shape/stats)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                # Simple reshape heuristic — treat as flat greyscale
                side = int(np.sqrt(len(img_array)))
                if side > 0:
                    image = img_array[:side * side].reshape(side, side)

        # Handle audio upload
        audio = None
        if "audio" in request.files:
            audio_file = request.files["audio"]
            if audio_file.filename:
                audio_bytes = audio_file.read()
                # Treat raw bytes as float32 waveform (simplified)
                audio = np.frombuffer(audio_bytes[:4000], dtype=np.uint8).astype(
                    np.float32
                ) / 255.0

        inp = MultimodalInput(
            text=text or None,
            image=image,
            audio=audio,
            structured=structured,
        )

        _log_activity("input", f"Processing: text={bool(text)}, image={image is not None}, "
                       f"audio={audio is not None}, structured={bool(structured)}, task={task_tag}")

        response = system.process_input(inp, task_tag=task_tag)

        _log_activity("response", f"Generated response (conf={response.confidence:.2f}, "
                       f"episodes_recalled={response.recalled_episodes})")

        # Serialise disambiguations
        disamb_data = []
        for d in response.disambiguations:
            disamb_data.append({
                "concept": d.concept,
                "meaning": d.meaning,
                "context_domain": d.context_domain,
                "confidence": d.confidence,
                "explanation": d.explanation,
                "alternatives": d.alternative_meanings,
            })

        return jsonify({
            "answer": response.answer,
            "confidence": response.confidence,
            "reasoning_trace": response.reasoning_trace,
            "recalled_episodes": response.recalled_episodes,
            "disambiguations": disamb_data,
            "learned_facts": response.learned_facts,
            "emotional_context": response.emotional_context,
            "stats": system.get_statistics(),
        })

    except Exception as e:
        logger.exception("Error processing input")
        _log_activity("error", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Return system statistics."""
    system = get_system()
    return jsonify(system.get_statistics())


@app.route("/api/logs")
def api_logs():
    """Return activity log."""
    return jsonify({"logs": list(_activity_log)})


@app.route("/api/knowledge")
def api_knowledge():
    """Return full knowledge base."""
    system = get_system()
    entries = []
    for k in system.knowledge:
        entries.append({
            "concept": k.concept,
            "relation": k.relation,
            "target": k.target,
            "confidence": k.confidence,
            "source": k.source,
            "timestamp": k.timestamp,
        })
    return jsonify({"knowledge": entries})


@app.route("/api/context/stats")
def api_context_stats():
    """Return context engine statistics."""
    system = get_system()
    return jsonify(system.context.get_statistics())


@app.route("/api/export/json")
def export_json():
    """Export all system data as a single JSON file."""
    system = get_system()

    export_data = {
        "exported_at": datetime.now(timezone.utc).isoformat() + "Z",
        "system_stats": system.get_statistics(),
        "knowledge_base": [
            {
                "concept": k.concept,
                "relation": k.relation,
                "target": k.target,
                "confidence": k.confidence,
                "source": k.source,
                "timestamp": k.timestamp,
            }
            for k in system.knowledge
        ],
        "context_engine_stats": system.context.get_statistics(),
        "context_meanings": {
            concept: dict(domains)
            for concept, domains in system.context.context_meanings.items()
        },
        "activity_log": list(_activity_log),
    }

    buf = io.BytesIO()
    buf.write(json.dumps(export_data, indent=2, default=str).encode("utf-8"))
    buf.seek(0)

    _log_activity("export", "Exported system data as JSON")

    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"nsck_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )


@app.route("/api/export/zip")
def export_zip():
    """Export all system data and logs as a ZIP archive."""
    system = get_system()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # System statistics
        zf.writestr(
            "stats.json",
            json.dumps(system.get_statistics(), indent=2, default=str),
        )

        # Knowledge base
        kb = [
            {
                "concept": k.concept,
                "relation": k.relation,
                "target": k.target,
                "confidence": k.confidence,
                "source": k.source,
                "timestamp": k.timestamp,
            }
            for k in system.knowledge
        ]
        zf.writestr("knowledge_base.json", json.dumps(kb, indent=2, default=str))

        # Context engine
        ctx_data = {
            "stats": system.context.get_statistics(),
            "meanings": {
                c: dict(d) for c, d in system.context.context_meanings.items()
            },
        }
        zf.writestr("context_engine.json", json.dumps(ctx_data, indent=2, default=str))

        # Activity log
        zf.writestr(
            "activity_log.json",
            json.dumps(list(_activity_log), indent=2, default=str),
        )

    buf.seek(0)
    _log_activity("export", "Exported system data as ZIP")

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"nsck_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip",
    )


@app.route("/api/correct", methods=["POST"])
def api_correct():
    """Correct a piece of knowledge."""
    system = get_system()
    data = request.get_json()

    concept = data.get("concept", "")
    relation = data.get("relation", "")
    wrong_target = data.get("wrong_target", "")
    correct_target = data.get("correct_target", "")

    if not all([concept, relation, correct_target]):
        return jsonify({"error": "Missing required fields"}), 400

    system.correct_knowledge(concept, relation, wrong_target, correct_target)
    _log_activity(
        "correction",
        f"Corrected: {concept} {relation} {wrong_target} → {correct_target}",
    )

    return jsonify({"status": "ok", "stats": system.get_statistics()})


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Provide reward feedback for learning."""
    system = get_system()
    data = request.get_json()

    task_tag = data.get("task_tag", "general")
    reward = float(data.get("reward", 0.0))
    concepts = data.get("concepts", [])

    system.learn_from_feedback(task_tag, reward, concepts)
    _log_activity("feedback", f"Feedback: task={task_tag}, reward={reward}")

    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the dashboard server."""
    port = int(os.environ.get("NSCK_DASHBOARD_PORT", 5050))
    print(f"\n{'='*60}")
    print(f"  NSCK Cognitive Dashboard")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
