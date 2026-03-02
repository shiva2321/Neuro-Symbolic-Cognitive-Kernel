"""
NSCK V5 Vision Dashboard — Flask web application.

Run with:
    python nsck_vision/dashboard/v5_app.py [--port 5092] [--debug]

Provides:
  - Image batch upload with per-claim accuracy ratings
  - Societal world stats & domain visualization
  - TDA health monitoring panel
  - Benchmark results chart
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add nsck to path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_NSCK = _REPO / "nsck"
for p in [str(_REPO), str(_NSCK)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from flask import Flask, send_from_directory
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False
    print("Flask not installed. Run: pip install flask")

from nsck_vision.dashboard.v5_api import v5_blueprint

logger = logging.getLogger("nsck_vision.v5_app")

# ============================================================
# HTML Dashboard (embedded)
# ============================================================

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSCK V5 — Societal Hypervector Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
header{background:linear-gradient(135deg,#1a1f2e 0%,#252d3d 100%);padding:16px 24px;
  border-bottom:1px solid #2d3748;display:flex;align-items:center;gap:12px}
header h1{font-size:1.4rem;color:#63b3ed;font-weight:700}
header .badge{font-size:0.75rem;background:#2d3748;color:#9ae6b4;padding:2px 8px;border-radius:12px}
nav{background:#1a1f2e;border-bottom:1px solid #2d3748;display:flex;gap:0;overflow-x:auto}
nav button{padding:12px 20px;background:none;border:none;color:#a0aec0;cursor:pointer;
  font-size:0.9rem;border-bottom:3px solid transparent;transition:all 0.2s;white-space:nowrap}
nav button.active,nav button:hover{color:#63b3ed;border-bottom-color:#63b3ed}
.panel{display:none;padding:24px;max-width:1200px;margin:0 auto}
.panel.active{display:block}
.card{background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:20px;margin-bottom:16px}
.card h2{color:#63b3ed;font-size:1rem;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.drop-zone{border:2px dashed #4a5568;border-radius:8px;padding:40px;text-align:center;
  cursor:pointer;transition:border-color 0.2s}
.drop-zone:hover,.drop-zone.over{border-color:#63b3ed;background:rgba(99,179,237,0.05)}
.drop-zone p{color:#718096;margin-top:8px;font-size:0.9rem}
input[type=file]{display:none}
button.primary{background:#3182ce;color:#fff;border:none;padding:10px 24px;
  border-radius:4px;cursor:pointer;font-size:0.9rem;transition:background 0.2s;font-weight:600}
button.primary:hover{background:#2b6cb0}
button.primary:disabled{background:#4a5568;cursor:not-allowed}
.result-img{display:flex;gap:16px;padding:12px;background:#252d3d;border-radius:6px;margin-bottom:8px}
.result-img img{width:80px;height:80px;object-fit:cover;border-radius:4px}
.result-img .details{flex:1}
.result-img .conf{font-size:1.4rem;font-weight:700;color:#68d391}
.claim-row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:0.8rem}
.claim-row .bar-bg{flex:1;background:#2d3748;border-radius:3px;height:6px}
.claim-row .bar-fill{height:6px;border-radius:3px;transition:width 0.4s}
.bar-high{background:#68d391}
.bar-medium{background:#ecc94b}
.bar-low{background:#fc8181}
.stat-pill{display:inline-flex;flex-direction:column;align-items:center;
  background:#252d3d;border-radius:8px;padding:12px 16px;min-width:100px;text-align:center}
.stat-pill .val{font-size:1.6rem;font-weight:700;color:#63b3ed}
.stat-pill .lbl{font-size:0.75rem;color:#718096;margin-top:2px}
.domain-card{background:#252d3d;border:1px solid #3d4a5e;border-radius:8px;padding:16px}
.domain-card h3{color:#9ae6b4;font-size:0.95rem;margin-bottom:8px}
.nbhd-tag{display:inline-block;background:#2d3748;border-radius:4px;padding:2px 8px;
  font-size:0.75rem;color:#a0aec0;margin:2px}
.betti-box{background:#252d3d;border-radius:8px;padding:16px;text-align:center}
.betti-box .val{font-size:2rem;font-weight:700;color:#b794f4}
.betti-box .lbl{color:#718096;font-size:0.8rem;margin-top:4px}
.health-ring{position:relative;width:120px;height:120px;margin:0 auto 12px}
.health-ring svg{transform:rotate(-90deg)}
.health-ring .score-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:1.3rem;font-weight:700;color:#68d391}
.alert-item{background:#2d3748;border-left:3px solid #fc8181;padding:8px 12px;
  border-radius:0 4px 4px 0;margin:4px 0;font-size:0.85rem;color:#fc8181}
.bench-table{width:100%;border-collapse:collapse;font-size:0.85rem}
.bench-table th{color:#718096;text-align:left;padding:8px;border-bottom:1px solid #2d3748}
.bench-table td{padding:8px;border-bottom:1px solid #1a1f2e}
.pass{color:#68d391}.fail{color:#fc8181}
.loading{text-align:center;padding:20px;color:#718096}
</style>
</head>
<body>
<header>
  <h1>⊗ NSCK V5</h1>
  <span class="badge">Societal Hypervector Dashboard</span>
</header>
<nav>
  <button class="active" onclick="showPanel('analysis',this)">🔍 Analysis</button>
  <button onclick="showPanel('world',this)">🌐 Societal World</button>
  <button onclick="showPanel('health',this)">💓 Health</button>
  <button onclick="showPanel('benchmarks',this)">⚡ Benchmarks</button>
</nav>

<!-- ====== ANALYSIS PANEL ====== -->
<div id="panel-analysis" class="panel active">
  <div class="card">
    <h2>📷 Batch Image Analysis</h2>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div style="font-size:2rem">⬆</div>
      <p>Drop images here or click to select (supports multiple)</p>
      <input type="file" id="fileInput" accept="image/*" multiple onchange="handleFiles(this.files)">
    </div>
    <div style="margin-top:12px;display:flex;gap:12px;align-items:center">
      <button class="primary" id="analyzeBtn" onclick="runBatchAnalysis()" disabled>Analyze</button>
      <span id="fileCount" style="color:#718096;font-size:0.9rem">No files selected</span>
    </div>
  </div>
  <div id="analysisResults"></div>
</div>

<!-- ====== WORLD PANEL ====== -->
<div id="panel-world" class="panel">
  <div class="card">
    <h2>🌐 Societal World Overview</h2>
    <div id="worldStats" class="loading">Loading stats...</div>
  </div>
  <div class="card">
    <h2>🏙 Domains</h2>
    <div id="domainCards" class="loading">Loading domains...</div>
  </div>
</div>

<!-- ====== HEALTH PANEL ====== -->
<div id="panel-health" class="panel">
  <div class="grid2">
    <div class="card">
      <h2>💓 Overall Health</h2>
      <div id="healthOverview" class="loading">Loading...</div>
    </div>
    <div class="card">
      <h2>🔺 Betti Numbers (β₀ β₁ β₂)</h2>
      <div id="bettiPanel" class="loading">Loading...</div>
    </div>
  </div>
  <div class="grid2">
    <div class="card">
      <h2>📊 Zipf Compliance</h2>
      <div id="zipfPanel" class="loading">Loading...</div>
    </div>
    <div class="card">
      <h2>🌊 Percolation</h2>
      <div id="percPanel" class="loading">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h2>⚠ Topology Alerts</h2>
    <div id="alertPanel"></div>
  </div>
</div>

<!-- ====== BENCHMARKS PANEL ====== -->
<div id="panel-benchmarks" class="panel">
  <div class="card">
    <h2>⚡ Performance Benchmarks</h2>
    <canvas id="benchChart" height="200"></canvas>
  </div>
  <div class="card">
    <h2>📋 Benchmark Results</h2>
    <div id="benchTable" class="loading">Loading...</div>
  </div>
</div>

<script>
// ── State ───────────────────────────────────────────────────────────────────
let selectedFiles = [];
let benchChart = null;

// ── Navigation ──────────────────────────────────────────────────────────────
function showPanel(name, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'world') loadWorld();
  if (name === 'health') loadHealth();
  if (name === 'benchmarks') loadBenchmarks();
}

// ── File handling ────────────────────────────────────────────────────────────
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
  selectedFiles = Array.from(files);
  document.getElementById('fileCount').textContent =
    selectedFiles.length + ' file(s) selected';
  document.getElementById('analyzeBtn').disabled = selectedFiles.length === 0;
}

// ── Batch Analysis ──────────────────────────────────────────────────────────
async function runBatchAnalysis() {
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  document.getElementById('analysisResults').innerHTML =
    '<div class="loading">Processing ' + selectedFiles.length + ' image(s)...</div>';

  // Build base64 list
  const items = await Promise.all(selectedFiles.map(async (f) => {
    const b64 = await fileToBase64(f);
    return { image: b64.split(',')[1], label: f.name };
  }));

  try {
    const resp = await fetch('/v5/batch_analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(items),
    });
    const data = await resp.json();
    renderAnalysisResults(data);
  } catch (err) {
    document.getElementById('analysisResults').innerHTML =
      '<div class="alert-item">Error: ' + err.message + '</div>';
  }
  btn.disabled = false;
  btn.textContent = 'Analyze';
}

function fileToBase64(file) {
  return new Promise(res => {
    const reader = new FileReader();
    reader.onload = e => res(e.target.result);
    reader.readAsDataURL(file);
  });
}

function renderAnalysisResults(data) {
  let html = '<div class="card"><h2>Results (' + data.n_images + ' image' +
    (data.n_images !== 1 ? 's' : '') + ') — Overall: ' +
    (data.overall_confidence * 100).toFixed(1) + '%</h2>';

  for (const r of (data.results || [])) {
    html += '<div class="result-img">';
    html += '<div class="details">';
    html += '<div style="font-weight:700;margin-bottom:4px">' + r.label + '</div>';
    html += '<div class="conf">' + (r.overall_confidence * 100).toFixed(1) + '%</div>';
    for (const c of (r.claims || [])) {
      const w = Math.round(c.confidence * 100);
      const cls = c.rating === 'high' ? 'bar-high' : c.rating === 'medium' ? 'bar-medium' : 'bar-low';
      html += '<div class="claim-row"><span style="width:220px;flex-shrink:0">' +
        c.claim + '</span><div class="bar-bg"><div class="bar-fill ' + cls +
        '" style="width:' + w + '%"></div></div><span style="width:36px;text-align:right">' +
        w + '%</span></div>';
    }
    html += '</div></div>';
  }
  html += '</div>';
  document.getElementById('analysisResults').innerHTML = html;
}

// ── World ────────────────────────────────────────────────────────────────────
async function loadWorld() {
  const [stats, domains] = await Promise.all([
    fetch('/v5/societal/stats').then(r => r.json()),
    fetch('/v5/societal/domains').then(r => r.json()),
  ]);

  // Stats pills
  const keys = [
    ['n_concepts','Concepts'],['n_neighborhoods','Neighborhoods'],
    ['n_domains','Domains'],['total_bonds','Bonds'],
  ];
  let shtml = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px">';
  for (const [k,l] of keys) {
    shtml += '<div class="stat-pill"><span class="val">' + (stats[k] ?? 0) +
      '</span><span class="lbl">' + l + '</span></div>';
  }
  shtml += '</div>';

  // Stability distribution
  const sc = stats.stability_classes || {};
  shtml += '<div style="margin-top:8px"><strong style="font-size:0.85rem;color:#718096">Stability Distribution</strong>';
  for (const [cls, cnt] of Object.entries(sc)) {
    const total = Object.values(sc).reduce((a, b) => a + b, 0) || 1;
    const pct = Math.round(cnt / total * 100);
    shtml += '<div class="claim-row"><span style="width:100px">' + cls +
      '</span><div class="bar-bg"><div class="bar-fill bar-high" style="width:' + pct +
      '%"></div></div><span>' + cnt + '</span></div>';
  }
  shtml += '</div>';
  if (stats._mock) shtml += '<p style="color:#4a5568;font-size:0.8rem;margin-top:8px">⚠ Mock data (NSCK not initialized)</p>';
  document.getElementById('worldStats').innerHTML = shtml;

  // Domain cards
  let dhtml = '<div class="grid3">';
  for (const d of domains) {
    dhtml += '<div class="domain-card"><h3>' + d.name + '</h3>';
    dhtml += '<div style="font-size:0.8rem;color:#718096;margin-bottom:8px">' +
      d.n_neighborhoods + ' neighborhoods</div>';
    for (const n of (d.neighborhoods || [])) {
      dhtml += '<span class="nbhd-tag">' + (n.id || '').split('_').slice(-1)[0] +
        ' (' + n.n_concepts + ')</span>';
    }
    dhtml += '</div>';
  }
  dhtml += '</div>';
  document.getElementById('domainCards').innerHTML = dhtml;
}

// ── Health ───────────────────────────────────────────────────────────────────
async function loadHealth() {
  const h = await fetch('/v5/societal/health').then(r => r.json());
  const hs = Math.round((h.overall_health || 0.72) * 100);
  const c = hs > 70 ? '#68d391' : hs > 40 ? '#ecc94b' : '#fc8181';
  const r = 46, cx = 60, cy = 60;
  const circ = 2 * Math.PI * r;
  const dash = circ * hs / 100;

  document.getElementById('healthOverview').innerHTML =
    '<div class="health-ring"><svg viewBox="0 0 120 120" width="120" height="120">' +
    '<circle cx="' + cx + '" cy="' + cy + '" r="' + r +
    '" fill="none" stroke="#2d3748" stroke-width="10"/>' +
    '<circle cx="' + cx + '" cy="' + cy + '" r="' + r +
    '" fill="none" stroke="' + c + '" stroke-width="10"' +
    ' stroke-dasharray="' + dash.toFixed(1) + ' ' + (circ - dash).toFixed(1) + '"/>' +
    '</svg><div class="score-text" style="color:' + c + '">' + hs + '%</div></div>' +
    '<p style="text-align:center;color:#718096;font-size:0.85rem">Topological Health Score</p>';

  const tda = h.tda || {};
  document.getElementById('bettiPanel').innerHTML =
    '<div class="grid3">' +
    ['beta0','beta1','beta2'].map((b, i) => {
      const sym = ['β₀','β₁','β₂'][i];
      const meaning = ['Components','Cycles','Voids'][i];
      return '<div class="betti-box"><div class="val">' + (tda[b] ?? 0) +
        '</div><div class="lbl">' + sym + ' ' + meaning + '</div></div>';
    }).join('') + '</div>' +
    '<div style="margin-top:12px;font-size:0.85rem;color:#718096">Persistence entropy: ' +
    (tda.persistence_entropy || 0).toFixed(3) + '</div>';

  const zipf = h.zipf || {};
  document.getElementById('zipfPanel').innerHTML =
    '<div class="stat-pill" style="display:block;text-align:center;padding:12px">' +
    '<div class="val">' + (zipf.is_zipf_like ? '✓' : '✗') + '</div>' +
    '<div class="lbl">Zipf-Like</div></div>' +
    '<div style="margin-top:12px;font-size:0.85rem">' +
    '<div>α = ' + (zipf.alpha || 0).toFixed(3) + ' (ideal: 1.0)</div>' +
    '<div>R² = ' + (zipf.r_squared || 0).toFixed(3) + '</div>' +
    '<div>Health: ' + (zipf.health_score || 0).toFixed(3) + '</div></div>';

  const perc = h.percolation || {};
  const gf = Math.round((perc.giant_fraction || 0) * 100);
  document.getElementById('percPanel').innerHTML =
    '<div class="claim-row"><span>Giant component</span><div class="bar-bg">' +
    '<div class="bar-fill bar-high" style="width:' + gf + '%"></div></div>' +
    '<span>' + gf + '%</span></div>' +
    '<div style="margin-top:8px;font-size:0.85rem;color:#' +
    (perc.is_transitioning ? 'fc8181' : '68d391') + '">' +
    'Status: ' + (perc.transition_type || 'stable') + '</div>';

  const alerts = (tda.alerts || []);
  const ahtml = alerts.length === 0
    ? '<div style="color:#68d391;font-size:0.9rem">✓ No topology alerts</div>'
    : alerts.map(a => '<div class="alert-item">' + a + '</div>').join('');
  document.getElementById('alertPanel').innerHTML = ahtml;
}

// ── Benchmarks ───────────────────────────────────────────────────────────────
async function loadBenchmarks() {
  const data = await fetch('/v5/benchmarks').then(r => r.json());
  const results = data.results || [];

  // Chart
  const labels = results.map(r => r.name);
  const values = results.map(r =>
    r.per_concept_ms || r.per_query_ms || r.per_tick_ms || r.elapsed_ms || r.build_ms || 0
  );
  const colors = results.map(r => (r.passed !== false ? 'rgba(99,179,237,0.7)' : 'rgba(252,129,129,0.7)'));

  const canvas = document.getElementById('benchChart');
  if (benchChart) benchChart.destroy();
  benchChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Time (ms)',
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.7', '1')),
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#a0aec0' } } },
      scales: {
        x: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' } },
        y: { ticks: { color: '#a0aec0' }, grid: { color: '#2d3748' },
             title: { display: true, text: 'ms', color: '#718096' } },
      }
    }
  });

  // Table
  let html = '<table class="bench-table"><thead><tr>' +
    '<th>Benchmark</th><th>Time (ms)</th><th>Status</th></tr></thead><tbody>';
  for (const r of results) {
    const t = (r.per_concept_ms || r.per_query_ms || r.per_tick_ms ||
                r.elapsed_ms || r.build_ms || 0).toFixed(3);
    const st = r.passed !== false ? '<span class="pass">✓ Pass</span>' : '<span class="fail">✗ Fail</span>';
    html += '<tr><td>' + r.name + '</td><td>' + t + '</td><td>' + st + '</td></tr>';
  }
  html += '</tbody></table>';
  if (data._mock) html += '<p style="color:#4a5568;margin-top:8px;font-size:0.8rem">⚠ Demo data</p>';
  document.getElementById('benchTable').innerHTML = html;
}
</script>
</body>
</html>"""


# ============================================================
# Flask application
# ============================================================

def create_app() -> "Flask":  # type: ignore[return]
    if not _FLASK_OK:
        raise ImportError("Flask is required")

    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    # Register V5 blueprint
    if v5_blueprint is not None:
        app.register_blueprint(v5_blueprint)

    # Try to also register V1 API
    try:
        from nsck_vision.dashboard.api import v1_blueprint  # type: ignore[import]
        app.register_blueprint(v1_blueprint)
    except Exception:
        pass

    @app.route("/")
    def index():
        return _DASHBOARD_HTML

    @app.route("/health")
    def health():
        from flask import jsonify as _j
        return _j({"status": "ok", "version": "5.0"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="NSCK V5 Dashboard")
    parser.add_argument("--port", type=int, default=5092, help="Port (default 5092)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if not _FLASK_OK:
        print("ERROR: Flask not installed. Run: pip install flask")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    print(f"Starting NSCK V5 Dashboard on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
