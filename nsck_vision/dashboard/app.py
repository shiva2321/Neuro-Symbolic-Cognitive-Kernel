"""NSCK Vision Dashboard — Flask web UI for NSCK-UPMA."""
from __future__ import annotations

import os
import sys
import json
import time
import base64
import logging
import argparse
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DIR not in sys.path:
    sys.path.insert(0, _NSCK_DIR)

try:
    from flask import Flask, request, jsonify, Response
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

import numpy as np

logger = logging.getLogger("nsck_vision.dashboard")

# HTML template (embedded as string)
_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSCK Vision Dashboard — NSCK-UPMA</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
header { background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%); padding: 16px 24px;
  border-bottom: 1px solid #2d3748; display: flex; align-items: center; gap: 12px; }
header h1 { font-size: 1.4rem; color: #63b3ed; }
header span { font-size: 0.8rem; color: #718096; }
nav { background: #1a1f2e; border-bottom: 1px solid #2d3748; display: flex; gap: 0; }
nav button { padding: 12px 20px; background: none; border: none; color: #a0aec0; cursor: pointer;
  font-size: 0.9rem; border-bottom: 3px solid transparent; transition: all 0.2s; }
nav button.active, nav button:hover { color: #63b3ed; border-bottom-color: #63b3ed; }
.panel { display: none; padding: 24px; max-width: 1200px; margin: 0 auto; }
.panel.active { display: block; }
.card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.card h2 { color: #63b3ed; font-size: 1rem; margin-bottom: 12px; }
.drop-zone { border: 2px dashed #4a5568; border-radius: 8px; padding: 40px; text-align: center;
  cursor: pointer; transition: border-color 0.2s; }
.drop-zone:hover, .drop-zone.over { border-color: #63b3ed; }
.drop-zone p { color: #718096; }
input[type="text"], input[type="number"], select { background: #252d3d; border: 1px solid #4a5568;
  color: #e2e8f0; padding: 8px 12px; border-radius: 4px; width: 100%; margin: 4px 0 12px; }
button.primary { background: #3182ce; color: white; border: none; padding: 10px 20px;
  border-radius: 4px; cursor: pointer; font-size: 0.9rem; transition: background 0.2s; }
button.primary:hover { background: #2b6cb0; }
.result-label { font-size: 1.5rem; font-weight: 700; color: #68d391; margin: 8px 0; }
.confidence-bar { height: 8px; background: #2d3748; border-radius: 4px; margin: 8px 0; }
.confidence-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3182ce, #63b3ed); }
.accuracy-bar-fill { background: linear-gradient(90deg, #38a169, #68d391); }
.chain-step { padding: 6px 12px; background: #252d3d; border-left: 3px solid #63b3ed;
  margin: 4px 0; border-radius: 0 4px 4px 0; font-size: 0.85rem; }
.analogy-item { display: inline-block; padding: 4px 10px; background: #553c9a; border-radius: 16px;
  font-size: 0.8rem; margin: 3px; }
.prov-item { display: inline-block; padding: 4px 10px; background: #2a4365; border-radius: 16px;
  font-size: 0.8rem; margin: 3px; }
.model-row { display: flex; align-items: center; gap: 12px; padding: 10px;
  border-bottom: 1px solid #2d3748; }
.model-row .model-id { font-weight: 600; color: #63b3ed; flex: 1; }
.domain-tag { padding: 2px 8px; background: #2a4365; border-radius: 12px; font-size: 0.75rem; }
.status { padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
.status.passed { background: #276749; color: #9ae6b4; }
.status.failed { background: #742a2a; color: #fc8181; }
pre { background: #252d3d; padding: 12px; border-radius: 4px; font-size: 0.8rem; overflow-x: auto; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div>🧠</div>
  <h1>NSCK Vision Dashboard</h1>
  <span>NSCK-UPMA Universal Pretrained Model Absorber v1.0.0</span>
</header>
<nav>
  <button class="active" onclick="showPanel('analyze')">📸 Analyze</button>
  <button onclick="showPanel('registry')">📚 Registry</button>
  <button onclick="showPanel('absorb')">⚡ Absorb</button>
  <button onclick="showPanel('benchmarks')">📊 Benchmarks</button>
  <button onclick="showPanel('export')">📤 Export</button>
</nav>

<!-- Analyze Panel -->
<div id="panel-analyze" class="panel active">
  <div class="grid-2">
    <div>
      <div class="card">
        <h2>Image Upload</h2>
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
          <p>Drag &amp; drop image here or click to select</p>
          <p style="font-size:0.75rem;color:#4a5568;margin-top:8px">PNG, JPG, WEBP supported</p>
        </div>
        <input type="file" id="fileInput" style="display:none" accept="image/*" onchange="handleFile(this.files[0])">
        <div id="imagePreview" style="margin-top:12px;"></div>
        <button class="primary" style="margin-top:12px;width:100%;" onclick="analyzeImage()">🔍 Analyze</button>
      </div>
    </div>
    <div>
      <div class="card" id="resultCard" style="display:none">
        <h2>NSCK Analysis Result</h2>
        <div class="result-label" id="resultLabel">—</div>
        <div style="display:flex;gap:16px;margin:8px 0;">
          <div style="flex:1">
            <div style="font-size:0.75rem;color:#718096;">Confidence</div>
            <div class="confidence-bar"><div class="confidence-bar-fill" id="confBar" style="width:0%"></div></div>
            <div id="confVal" style="font-size:0.85rem;"></div>
          </div>
          <div style="flex:1">
            <div style="font-size:0.75rem;color:#718096;">Accuracy Rating</div>
            <div class="confidence-bar"><div class="confidence-bar-fill accuracy-bar-fill" id="accBar" style="width:0%"></div></div>
            <div id="accVal" style="font-size:0.85rem;"></div>
          </div>
        </div>
        <div style="margin-top:12px;">
          <div style="font-size:0.75rem;color:#718096;margin-bottom:4px;">🔗 Causal Chain</div>
          <div id="causalChain"></div>
        </div>
        <div style="margin-top:12px;">
          <div style="font-size:0.75rem;color:#718096;margin-bottom:4px;">🌐 Cross-Domain Analogies</div>
          <div id="analogies"></div>
        </div>
        <div style="margin-top:12px;">
          <div style="font-size:0.75rem;color:#718096;margin-bottom:4px;">📦 Source Models</div>
          <div id="provenance"></div>
        </div>
        <div style="margin-top:12px;font-size:0.75rem;color:#718096;" id="latencyInfo"></div>
      </div>
    </div>
  </div>
</div>

<!-- Registry Panel -->
<div id="panel-registry" class="panel">
  <div class="card">
    <h2>Absorbed Model Registry</h2>
    <button class="primary" onclick="loadRegistry()">🔄 Refresh</button>
    <div id="registryList" style="margin-top:16px;"></div>
  </div>
</div>

<!-- Absorb Panel -->
<div id="panel-absorb" class="panel">
  <div class="card">
    <h2>Absorb New Model</h2>
    <label style="font-size:0.85rem;color:#a0aec0;">Model ID (HuggingFace ID, torchvision name, or custom)</label>
    <input type="text" id="absorbModelId" placeholder="e.g. resnet18, bert-base-uncased">
    <label style="font-size:0.85rem;color:#a0aec0;">Domain Tag</label>
    <input type="text" id="absorbDomain" placeholder="e.g. medical, satellite, general">
    <label style="font-size:0.85rem;color:#a0aec0;">Max Samples</label>
    <input type="number" id="absorbMaxSamples" value="100" min="10" max="10000">
    <button class="primary" onclick="absorbModel()">⚡ Start Absorption</button>
    <div id="absorbStatus" style="margin-top:12px;"></div>
  </div>
</div>

<!-- Benchmarks Panel -->
<div id="panel-benchmarks" class="panel">
  <div class="card">
    <h2>Benchmark Results</h2>
    <button class="primary" onclick="loadBenchmarks()">📊 Load Results</button>
    <div id="benchmarkResults" style="margin-top:16px;"></div>
  </div>
</div>

<!-- Export Panel -->
<div id="panel-export" class="panel">
  <div class="card">
    <h2>Research Export</h2>
    <p style="color:#718096;margin-bottom:16px;">Download benchmark results and system stats for research use.</p>
    <button class="primary" onclick="exportJSON()">💾 Export Stats (JSON)</button>
    <button class="primary" style="margin-left:8px;background:#276749;" onclick="exportCSV()">📄 Export CSV</button>
    <div id="exportPreview" style="margin-top:16px;"></div>
  </div>
</div>

<script>
let _imageB64 = null;
let _panels = ['analyze','registry','absorb','benchmarks','export'];

function showPanel(name) {
  _panels.forEach(p => {
    document.getElementById('panel-'+p).classList.remove('active');
  });
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  event.target.classList.add('active');
}

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    _imageB64 = e.target.result.split(',')[1];
    document.getElementById('imagePreview').innerHTML = 
      '<img src="'+e.target.result+'" style="max-width:100%;max-height:200px;border-radius:4px;">';
  };
  reader.readAsDataURL(file);
}

// Drag & Drop
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('over'); handleFile(e.dataTransfer.files[0]); });

async function analyzeImage() {
  if (!_imageB64) { alert('Please select an image first.'); return; }
  const res = await fetch('/api/vision/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image_b64: _imageB64})
  });
  const data = await res.json();
  if (data.error) { alert('Error: '+data.error); return; }
  
  document.getElementById('resultCard').style.display = 'block';
  document.getElementById('resultLabel').textContent = data.label || 'unknown';
  
  const conf = (data.confidence * 100).toFixed(1);
  document.getElementById('confBar').style.width = conf+'%';
  document.getElementById('confVal').textContent = conf+'%';
  
  const acc = (data.accuracy_rating * 100).toFixed(1);
  document.getElementById('accBar').style.width = acc+'%';
  document.getElementById('accVal').textContent = acc+'%';
  
  const chain = data.causal_chain || [];
  document.getElementById('causalChain').innerHTML = chain.length > 0 
    ? chain.map(s => '<div class="chain-step">'+s+'</div>').join('')
    : '<span style="color:#4a5568;">No causal chain available</span>';
  
  const analogies = data.cross_domain_analogies || [];
  document.getElementById('analogies').innerHTML = analogies.length > 0
    ? analogies.map(a => '<span class="analogy-item">'+a+'</span>').join('')
    : '<span style="color:#4a5568;">No cross-domain analogies</span>';
  
  const prov = data.source_model_provenance || [];
  document.getElementById('provenance').innerHTML = prov.length > 0
    ? prov.map(p => '<span class="prov-item">'+p+'</span>').join('')
    : '<span style="color:#4a5568;">No absorbed models used</span>';
  
  document.getElementById('latencyInfo').textContent = 
    'Latency: '+data.latency_ms.toFixed(1)+'ms | '+data.timestamp;
}

async function loadRegistry() {
  const res = await fetch('/api/vision/registry');
  const data = await res.json();
  const el = document.getElementById('registryList');
  if (!data.length) { el.innerHTML = '<p style="color:#718096;">No models absorbed yet.</p>'; return; }
  el.innerHTML = data.map(m => 
    '<div class="model-row"><span class="model-id">'+m.model_id+'</span>'
    + '<span class="domain-tag">'+m.domain+'</span>'
    + '<span class="status '+(m.metadata && m.metadata.passed ? 'passed':'')+'">'+
      (m.metadata && m.metadata.passed !== undefined ? (m.metadata.passed?'✓ passed':'✗ failed'):'unknown')+'</span>'
    + '<span style="font-size:0.75rem;color:#718096;">'+(m.metadata&&m.metadata.n_concepts||0)+' concepts</span>'
    + '</div>'
  ).join('');
}

async function absorbModel() {
  const model_id = document.getElementById('absorbModelId').value.trim();
  const domain = document.getElementById('absorbDomain').value.trim() || 'general';
  const max_samples = parseInt(document.getElementById('absorbMaxSamples').value) || 100;
  if (!model_id) { alert('Model ID required.'); return; }
  
  document.getElementById('absorbStatus').innerHTML = '<p style="color:#f6ad55;">⏳ Absorbing model...</p>';
  try {
    const res = await fetch('/api/vision/absorb', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model_id, domain, max_samples})
    });
    const data = await res.json();
    if (data.error) {
      document.getElementById('absorbStatus').innerHTML = '<p style="color:#fc8181;">❌ '+data.error+'</p>';
    } else {
      document.getElementById('absorbStatus').innerHTML = 
        '<p style="color:#68d391;">✅ Absorbed '+data.n_concepts_absorbed+' concepts from '+model_id+'</p>'
        + '<pre>'+JSON.stringify(data, null, 2)+'</pre>';
    }
  } catch(e) {
    document.getElementById('absorbStatus').innerHTML = '<p style="color:#fc8181;">❌ '+e.message+'</p>';
  }
}

async function loadBenchmarks() {
  const res = await fetch('/api/vision/benchmarks');
  const data = await res.json();
  document.getElementById('benchmarkResults').innerHTML = '<pre>'+JSON.stringify(data, null, 2)+'</pre>';
}

async function exportJSON() {
  const res = await fetch('/api/vision/stats');
  const data = await res.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'nsck_vision_stats.json'; a.click();
  document.getElementById('exportPreview').innerHTML = '<pre>'+JSON.stringify(data, null, 2)+'</pre>';
}

function exportCSV() {
  fetch('/api/vision/registry').then(r=>r.json()).then(data => {
    const rows = [['model_id','domain','n_concepts','spearman_rho','passed']];
    data.forEach(m => rows.push([
      m.model_id, m.domain,
      m.metadata&&m.metadata.n_concepts||0,
      m.metadata&&m.metadata.spearman_rho||0,
      m.metadata&&m.metadata.passed||false
    ]));
    const csv = rows.map(r=>r.join(',')).join('\n');
    const blob = new Blob([csv], {type:'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'nsck_vision_registry.csv'; a.click();
  });
}
</script>
</body>
</html>"""

app = Flask(__name__) if _FLASK_AVAILABLE else None

if _FLASK_AVAILABLE:
    from nsck_vision.dashboard.api import create_api
    app = create_api(app)

    @app.route("/", methods=["GET"])
    def index():
        return _DASHBOARD_HTML, 200, {"Content-Type": "text/html"}


def run_dashboard(port: int = 5091, debug: bool = False) -> None:
    """Start the Flask dashboard."""
    if not _FLASK_AVAILABLE:
        raise ImportError("Flask is required. Install with: pip install flask")
    logger.info("Starting NSCK Vision Dashboard on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)


def __main__():
    parser = argparse.ArgumentParser(description="NSCK Vision Dashboard")
    parser.add_argument("--port", type=int, default=5091)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_dashboard(port=args.port, debug=args.debug)


if __name__ == "__main__":
    __main__()
