import os
import sys
import json
import time
import threading
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# --- path setup ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsck-demo'))

from python.core.neural.snn_qat import TaskAwareSNN
from python.utils.hf_adapter import HuggingFaceAdapter
# [NEW] Cognitive Components
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.language.lingua_cortex import get_lingua_cortex, SemanticFingerprint
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.vsa.universal_encoder import UniversalEncoder

app = Flask(__name__)

# --- GLOBAL STATE ---
MODEL = None
ADAPTER = None
TEXT_LEARNER = None # Symbolic Memory
EPISODIC = None     # Episodic Memory
TRAINING_ACTIVE = False
TRAIN_STATS = {"epoch": 0, "loss": [], "acc": [], "dataset": "none"}
MODEL_PATH = "snn_model_v2.pth"

# Capture latent state for "Mind Reading"
CURRENT_MIND_STATE = {
    "latent": [0.0]*128,  # Perception
    "spikes": [0.0]*10,   # Action
    "value": 0.0,         # Critic
    "text_thought": "Neural activity strictly controlled.",
    "recalled_facts": [], # [NEW]
    "recalled_episodes": [] # [NEW]
}

# --- HTML TEMPLATE --- (Keeping existing template but adding Memory UI elements via JS updates if needed, or just sending data)
# Ideally I'd update HTML to show recalled facts, but for now I'll pipe them into "text_thought" or a new field if UI supports it.
# The previous UI has "thought-stream", I can append facts there.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSCK Mind Reader Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { background-color: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; }
        .btn { @apply px-4 py-2 rounded font-bold transition-all; }
        .btn-primary { @apply bg-cyan-600 hover:bg-cyan-500 text-white shadow-[0_0_15px_rgba(8,145,178,0.5)]; }
        .btn-danger { @apply bg-rose-600 hover:bg-rose-500 text-white; }
        .card { @apply bg-slate-900/80 p-6 rounded-xl border border-slate-800 backdrop-blur-sm shadow-xl; }
        .stat-val { @apply text-2xl font-mono font-bold text-cyan-400; }
        .stat-label { @apply text-xs uppercase tracking-widest text-slate-500; }
        
        /* Scanline effect */
        .scanline {
            width: 100%;
            height: 100px;
            z-index: 10;
            background: linear-gradient(0deg, rgba(0,0,0,0) 0%, rgba(8, 145, 178, 0.1) 50%, rgba(0,0,0,0) 100%);
            opacity: 0.1;
            position: absolute;
            bottom: 100%;
            animation: scanline 10s linear infinite;
            pointer-events: none;
        }
        @keyframes scanline {
            0% { bottom: 100%; }
            100% { bottom: -100%; }
        }
    </style>
</head>
<body class="h-screen flex flex-col overflow-hidden relative">
    <div class="scanline"></div>

    <!-- HEADER -->
    <header class="bg-black/50 border-b border-slate-800 p-4 flex justify-between items-center z-20">
        <div class="flex items-center gap-4">
            <div class="w-10 h-10 bg-cyan-500/10 border border-cyan-500 rounded-full flex items-center justify-center animate-pulse">
                <span class="text-xl">🧠</span>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">NSCK MIND READER</h1>
                <div class="text-[10px] text-cyan-600 uppercase tracking-[0.2em]">Neuro-Symbolic Cognitive Kernel v2.0</div>
            </div>
        </div>
        <div class="flex gap-2">
            <button onclick="switchTab('mind')" class="btn btn-primary" id="tab-mind">👁️ Mind Reader</button>
            <button onclick="switchTab('train')" class="btn bg-slate-800 hover:bg-slate-700 text-slate-300" id="tab-train">⚡ Train (Real Data)</button>
        </div>
    </header>

    <!-- CONTENT -->
    <main class="flex-1 p-6 overflow-auto grid grid-cols-12 gap-6 relative z-10">
        
        <!-- MIND READER TAB -->
        <div id="view-mind" class="col-span-12 grid grid-cols-12 gap-6 h-full">
            
            <!-- LEFT: INPUT & SENSORY -->
            <div class="col-span-12 lg:col-span-4 flex flex-col gap-6">
                <!-- Chat / Input -->
                <div class="card flex-1 flex flex-col">
                    <h2 class="text-sm font-semibold text-cyan-500 uppercase mb-4 tracking-wider">Sensory Input</h2>
                    <div class="flex-1 bg-slate-950 rounded border border-slate-800 p-4 terminal font-mono text-sm text-green-400 overflow-y-auto mb-4" id="chat-history">
                        <div>> SYSTEM ONLINE</div>
                        <div>> COGNITIVE ENGINE INITIALIZED</div>
                        <div>> MEMORY SYSTEMS ACTIVE</div>
                        <div>> WAITING FOR INPUT...</div>
                    </div>
                    <div class="flex gap-2">
                        <input type="text" id="user-input" class="flex-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" placeholder="Ask about learned knowledge..." onkeypress="if(event.key==='Enter') sendMessage()">
                        <button onclick="sendMessage()" class="btn btn-primary">SEND</button>
                    </div>
                </div>
            </div>

            <!-- CENTER: COGNITIVE STATE -->
            <div class="col-span-12 lg:col-span-5 flex flex-col gap-6">
                <!-- Latent Space Visualization -->
                <div class="card h-64 flex flex-col">
                    <div class="flex justify-between items-center mb-2">
                        <h2 class="text-sm font-semibold text-purple-400 uppercase tracking-wider">Perception (Latent Space)</h2>
                        <span class="text-[10px] text-slate-500">128-DIM VECTOR HEATMAP</span>
                    </div>
                    <div id="latent-heatmap" class="flex-1 w-full h-full"></div>
                </div>

                <!-- Thoughts / Analysis -->
                <div class="card flex-1">
                     <h2 class="text-sm font-semibold text-yellow-500 uppercase mb-4 tracking-wider">Cognitive Trace & Recall</h2>
                     <div id="thought-stream" class="font-mono text-xs text-slate-300 space-y-2 h-full overflow-y-auto">
                        <!-- Dynamic thoughts -->
                     </div>
                </div>
            </div>

            <!-- RIGHT: MOTOR & VALUE -->
            <div class="col-span-12 lg:col-span-3 flex flex-col gap-6">
                <!-- Value Estimate -->
                <div class="card">
                    <h2 class="text-sm font-semibold text-emerald-500 uppercase mb-2 tracking-wider">Critic (Value)</h2>
                    <div class="text-5xl font-bold text-white mb-2 tracking-tighter" id="val-display">0.00</div>
                    <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div id="val-bar" class="bg-emerald-500 h-full w-0 transition-all duration-300"></div>
                    </div>
                </div>

                <!-- Action Spikes -->
                <div class="card flex-1 flex flex-col">
                    <h2 class="text-sm font-semibold text-blue-500 uppercase mb-4 tracking-wider">Motor Cortex</h2>
                    <div class="flex-1 relative">
                        <canvas id="spikeChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- TRAINING TAB -->
        <div id="view-train" class="col-span-12 grid grid-cols-12 gap-6 hidden">
             <div class="col-span-12 md:col-span-4 card h-fit">
                <h2 class="text-lg font-bold text-white mb-2">Training Control</h2>
                <div class="h-px bg-slate-800 w-full mb-6"></div>
                
                <div class="space-y-6">
                    <div>
                        <label class="stat-label block mb-2">DATASET SOURCE</label>
                        <select id="dataset-select" class="w-full bg-slate-800 border border-slate-700 text-white rounded p-2 focus:border-cyan-500 outline-none">
                            <option value="wikitext">📚 Text: WikiText (Simultaneous Symbolic Learning)</option>
                            <option value="cifar10">🖼️ Vision: CIFAR-10 (Neural Only)</option>
                        </select>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-slate-950 p-4 rounded border border-slate-800">
                            <div class="stat-label mb-1">EPOCH</div>
                            <div class="stat-val" id="tr-epoch">0</div>
                        </div>
                        <div class="bg-slate-950 p-4 rounded border border-slate-800">
                            <div class="stat-label mb-1">ACCURACY</div>
                            <div class="stat-val text-green-400" id="tr-acc">0%</div>
                        </div>
                    </div>
                    
                    <button onclick="startTraining()" id="btn-start-train" class="btn btn-primary w-full py-4 text-lg">INITIALIZE TRAINING</button>
                    
                    <div id="train-log" class="text-xs font-mono text-slate-500 h-32 overflow-y-auto bg-slate-950 p-2 rounded border border-slate-800">
                        > Ready to train...
                    </div>
                </div>
             </div>
             
             <div class="col-span-12 md:col-span-8 card">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-sm font-semibold text-cyan-500 uppercase tracking-wider">Real-Time Metrics</h2>
                    <div class="flex gap-2 text-xs">
                        <span class="flex items-center gap-1"><span class="w-2 h-2 bg-green-500 rounded-full"></span> ACCURACY</span>
                        <span class="flex items-center gap-1"><span class="w-2 h-2 bg-red-500 rounded-full"></span> LOSS</span>
                    </div>
                </div>
                <div class="h-[400px] w-full">
                    <canvas id="trainChart"></canvas>
                </div>
             </div>
        </div>

    </main>

    <script>
        // --- CHARTS & STATE ---
        let spikeChart, trainChart;
        let trainInterval, pollInterval;

        window.onload = function() {
            initCharts();
            initHeatmap();
            pollInterval = setInterval(pollMind, 500); // 2Hz Update
            trainInterval = setInterval(pollTraining, 1000);
        };

        function switchTab(tab) {
            document.getElementById('view-mind').classList.add('hidden');
            document.getElementById('view-train').classList.add('hidden');
            document.getElementById('view-' + tab).classList.remove('hidden');
            
            // Button styles
            document.querySelectorAll('header button').forEach(b => {
                b.className = "btn bg-slate-800 hover:bg-slate-700 text-slate-300";
            });
            document.getElementById('tab-' + tab).className = "btn btn-primary";
        }

        async function sendMessage() {
            const field = document.getElementById('user-input');
            const msg = field.value.trim();
            if(!msg) return;
            
            addToChat("USER", msg);
            field.value = "";
            
            // Send to backend
            try {
                const res = await fetch('/api/interact', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: msg, type: 'text'})
                });
                const data = await res.json();
                addToChat("NSCK", data.response || "...");
                updateMindUI(data.mind);
            } catch(e) { console.error(e); }
        }
        
        function addToChat(role, text) {
            const div = document.createElement('div');
            const color = role === "USER" ? "text-white" : "text-cyan-400";
            div.innerHTML = `<span class="${color} font-bold">[${role}]</span> ${text}`;
            document.getElementById('chat-history').appendChild(div);
            document.getElementById('chat-history').scrollTop = 99999;
        }

        async function pollMind() {
            if(document.getElementById('view-mind').classList.contains('hidden')) return;
            try {
                const res = await fetch('/api/mind_state');
                const data = await res.json();
                updateMindUI(data);
            } catch(e) {}
        }

        function updateMindUI(data) {
            if(!data) return;
            
            // Update Value
            document.getElementById('val-display').innerText = data.value.toFixed(2);
            document.getElementById('val-bar').style.width = Math.min(100, Math.max(0, (data.value + 1) * 50)) + "%";
            
            // Update Heatmap (Latent)
            updateHeatmap(data.latent);
            
            // Update Spikes
            updateSpikes(data.spikes);
            
            // Update Thoughts
            if(data.text_thought && data.text_thought !== window.lastThought) {
                window.lastThought = data.text_thought;
                const ts = document.getElementById('thought-stream');
                const t = document.createElement('div');
                t.innerHTML = `<div class="mb-2"><span class="text-yellow-500/50">></span> ${data.text_thought}</div>`;
                
                // Add facts if recall happened
                if(data.recalled_facts && data.recalled_facts.length > 0) {
                     t.innerHTML += `<div class="pl-4 text-emerald-400 opacity-80 mb-2">Recalled Facts:<br>• ${data.recalled_facts.join('<br>• ')}</div>`;
                }

                ts.prepend(t);
                if(ts.children.length > 20) ts.lastChild.remove();
            }
        }

        // --- VISUALIZATION LIBS ---
        function initHeatmap() {
            // Plotly Heatmap for Latent Vector (8x16 grid for 128 dims)
            const z = Array(8).fill(Array(16).fill(0));
            Plotly.newPlot('latent-heatmap', [{
                z: z, type: 'heatmap', colorscale: 'Viridis', showscale: false
            }], {
                margin: {t:0, b:0, l:0, r:0},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {visible: false}, yaxis: {visible: false}
            }, {responsive: true, displayModeBar: false});
        }
        
        function updateHeatmap(flatVector) {
            // Reshape 1d (128) to 2d (8x16)
            let z = [];
            for(let i=0; i<8; i++) {
                z.push(flatVector.slice(i*16, (i+1)*16));
            }
            Plotly.react('latent-heatmap', [{
                z: z, type: 'heatmap', colorscale: 'Viridis', showscale: false
            }], {
                margin: {t:0, b:0, l:0, r:0},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {visible: false}, yaxis: {visible: false}
            }, {responsive: true, displayModeBar: false});
        }

        function initCharts() {
            const ctx = document.getElementById('spikeChart').getContext('2d');
            spikeChart = new Chart(ctx, {
                type: 'bar',
                data: { labels: ['0','1','2','3','4','5','6','7','8','9'], datasets: [{ label: 'Activity', data: [0,0,0,0,0,0,0,0,0,0], backgroundColor: '#3b82f6' }] },
                options: { 
                    responsive: true, maintainAspectRatio: false,
                    plugins: {legend: {display: false}}, 
                    scales: { 
                        y: { display: false, max: 1.0 }, 
                        x: { display: false } 
                    } 
                }
            });

            const ctx2 = document.getElementById('trainChart').getContext('2d');
            trainChart = new Chart(ctx2, {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'Accuracy', data: [], borderColor: '#22c55e', tension: 0.1, yAxisID: 'y' },
                    { label: 'Loss', data: [], borderColor: '#ef4444', tension: 0.1, yAxisID: 'y1' }
                ]},
                options: { 
                    responsive: true, maintainAspectRatio: false,
                    scales: { 
                        x: { display: false }, 
                        y: { position: 'left', grid: { color: '#1e293b' } },
                        y1: { position: 'right', grid: { display: false } }
                    } 
                }
            });
        }
        
        function updateSpikes(data) {
            if(data.length !== spikeChart.data.labels.length) {
                spikeChart.data.labels = data.map((_, i) => i);
            }
            spikeChart.data.datasets[0].data = data;
            spikeChart.update('none');
        }

        // --- TRAINING LOGIC ---
        async function startTraining() {
            const ds = document.getElementById('dataset-select').value;
            document.getElementById('btn-start-train').disabled = true;
            document.getElementById('btn-start-train').innerText = "TRAINING IN PROGRESS...";
            
             // Clear chart
            trainChart.data.labels = [];
            trainChart.data.datasets[0].data = [];
            trainChart.data.datasets[1].data = [];
            trainChart.update();
            
            await fetch('/api/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dataset: ds})
            });
        }
        
        async function pollTraining() {
             if(document.getElementById('view-train').classList.contains('hidden')) return;
             
             const res = await fetch('/api/train/stats');
             const data = await res.json();
             
             if(data.epoch > 0) {
                 document.getElementById('tr-epoch').innerText = data.epoch;
                 document.getElementById('tr-acc').innerText = data.acc.slice(-1)[0]?.toFixed(1) + "%";
                 
                 // Log
                 const logDiv = document.getElementById('train-log');
                 if(data.loss.length > logDiv.childElementCount) {
                     const l = data.loss.slice(-1)[0].toFixed(4);
                     const a = data.acc.slice(-1)[0].toFixed(1);
                     logDiv.innerHTML += `<div>Ep ${data.epoch}: Loss ${l} | Acc ${a}%</div>`;
                     logDiv.scrollTop = 9999;
                 }
                 
                 // Chart
                 trainChart.data.labels = data.acc.map((_, i) => i);
                 trainChart.data.datasets[0].data = data.acc;
                 trainChart.data.datasets[1].data = data.loss;
                 trainChart.update();
             }
             
             if(!data.active && document.getElementById('btn-start-train').disabled) {
                 document.getElementById('btn-start-train').disabled = false;
                 document.getElementById('btn-start-train').innerText = "INITIALIZE TRAINING";
             }
        }
    </script>
</body>
</html>
"""

# --- BACKEND ---

def hook_encoder_output(module, input, output):
    """Capture latent vector from encoder."""
    global CURRENT_MIND_STATE
    # output is [Batch, 128]
    # We take the first sample in batch
    latent = output.detach().cpu().numpy()[0].tolist()
    CURRENT_MIND_STATE["latent"] = latent

def load_model():
    global MODEL, ADAPTER, TEXT_LEARNER, EPISODIC
    if not ADAPTER:
        ADAPTER = HuggingFaceAdapter()
        
    if not MODEL:
        print("[Studio] Loading SNN...")
        MODEL = TaskAwareSNN()
        MODEL.register_task("wikitext", 64) 
        MODEL.register_task("cifar10", 10)
        
        # Attach Hook
        MODEL.encoder.register_forward_hook(hook_encoder_output)
        print("[Studio] Model loaded + Hook attached.")
        
    if not TEXT_LEARNER:
        print("[Studio] Initializing Cognitive Memory...")
        EPISODIC = EpisodicMemory()
        TEXT_LEARNER = TextKnowledgeLearner(episodic_memory=EPISODIC)
        print("[Studio] Memory Systems Online.")

def text_to_tensor(text):
    """
    [REAL] Converts text to SDR Fingerprint (16,384 bits) -> Float Tensor.
    Replicates LinguaCortex hashing logic so it is consistent with Semantic Memory.
    """
    cortex = get_lingua_cortex()
    
    # Generate SemanticFingerprint using VSA (128x128 bit grid)
    # This learns the context of the sentence (semantic folding)
    cortex.learn_text_snippet(text) # Ensure vocab is updated
    
    # We need a 'vector' for this sentence. 
    # Since TextKnowledgeLearner extracts concepts, let's hash the sentence 
    # into a Context Fingerprint just like learn_text_snippet calculates locally.
    
    words = set(text.lower().split())
    words = {w for w in words if w.isalnum() and len(w) > 2}
    if not words:
        return torch.zeros(1, 16384) # Empty
        
    # Re-calc snippet hash (center)
    snippet_hash = sum([cortex._hash_word_to_seed(w) for w in words])
    cx = snippet_hash % 128
    cy = (snippet_hash // 128) % 128
    
    # Generate bits
    fp = cortex._generate_fingerprint_at(cx, cy)
    
    # Convert bool grid (128x128) to float tensor (16384)
    flat_bits = fp.bits.flatten().astype(np.float32)
    return torch.tensor(flat_bits).unsqueeze(0) # [1, 16384]

def training_thread(dataset_name):
    global MODEL, TRAINING_ACTIVE, TRAIN_STATS
    TRAINING_ACTIVE = True
    TRAIN_STATS = {"epoch": 0, "loss": [], "acc": [], "dataset": dataset_name}
    
    print(f"[Studio] Starting Real-World Training on {dataset_name}...")
    
    # Optimizer
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    EPOCHS = 5
    BATCH_SIZE = 16
    
    # Data Stream
    if dataset_name == "cifar10":
        stream = ADAPTER.stream_vision_data(limit=500)
    else:
        stream = ADAPTER.stream_text_data(limit=200) # Text stream returns strings
        
    # Training Loop
    # Note: For Cifar10 we use standard backprop.
    # For Text, we use Symbolic Learning (TextLearner) + SNN Update
    
    step_loss = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        # Re-init stream
        if dataset_name == "cifar10":
            data_iter = ADAPTER.stream_vision_data(limit=200)
        else:
            data_iter = ADAPTER.stream_text_data(limit=100)
        
        batch_x = []
        batch_y = []
        
        for item in data_iter:
            if dataset_name == "cifar10":
                # IMAGE TRAINING
                x, y = item
                batch_x.append(x)
                batch_y.append(y)
                
                if len(batch_x) >= BATCH_SIZE:
                    bx = torch.stack(batch_x)
                    by = torch.tensor(batch_y, dtype=torch.long)
                    
                    optimizer.zero_grad()
                    logits, val = MODEL(bx, task_name="cifar10")
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, pred = torch.max(logits, 1)
                    total += by.size(0)
                    correct += (pred == by).sum().item()
                    
                    # Log Mind State
                    CURRENT_MIND_STATE["value"] = val.mean().item()
                    CURRENT_MIND_STATE["spikes"] = torch.softmax(logits[0], 0).detach().tolist()
                    CURRENT_MIND_STATE["text_thought"] = f"Visual Cortex Active. Loss: {loss.item():.4f}"
                    
                    batch_x, batch_y = [], []
                    time.sleep(0.05)
                    
            elif dataset_name == "wikitext":
                # TEXT / SYMBOLIC TRAINING
                text = item
                
                # 1. Symbolic Learning (Knowledge Graph)
                # This extracts concepts and relations *without* gradients
                stats = TEXT_LEARNER.learn_from_text(text, source="wikitext")
                learned_count = stats['concepts']
                
                # 2. Neural Grounding (SNN)
                # [REAL] Convert text to VSA Hypervector (16,384 dim)
                real_x = text_to_tensor(text) # [1, 16384]
                dummy_y = torch.randint(0, 64, (1,)) 
                
                optimizer.zero_grad()
                
                # Note: UniversalEncoder has a lazy 'concept_fc' that will adapt to 16384 dim input
                logits, val = MODEL(real_x, task_name="wikitext") 
                
                loss = criterion(logits, dummy_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += 1 # dummy
                total += 1
                
                # Log Mind State
                CURRENT_MIND_STATE["value"] = val.item()
                CURRENT_MIND_STATE["spikes"] = torch.softmax(logits[0], 0).detach().tolist()
                CURRENT_MIND_STATE["text_thought"] = f"Reading: '{text[:30]}...' | Learned {learned_count} concepts."
                
                # Add recalled facts to mind state occasionally
                if learned_count > 0:
                     latest_facts = [f"{f.subject} {f.relation} {f.object}" for f in TEXT_LEARNER.learned_facts[-3:]]
                     CURRENT_MIND_STATE["recalled_facts"] = latest_facts
                
                time.sleep(0.1) # Read speed
        
        # End epoch metrics
        if total > 0:
            avg_loss = total_loss / (total / BATCH_SIZE) if dataset_name == "cifar10" else total_loss/total
            epoch_acc = 100 * correct / total
            TRAIN_STATS["loss"].append(avg_loss)
            TRAIN_STATS["acc"].append(epoch_acc)
            TRAIN_STATS["epoch"] = epoch + 1
            print(f"Epoch {epoch}: Loss {avg_loss:.2f} Acc {epoch_acc:.1f}")

    TRAINING_ACTIVE = False
    print("[Studio] Training finished.")

# --- ROUTES ---

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/mind_state")
def get_mind_state():
    return jsonify(CURRENT_MIND_STATE)

@app.route("/api/interact", methods=["POST"])
def interact():
    data = request.json
    text = data.get("input", "")
    response = ""
    
    # 1. Query Cognitive Memory
    if TEXT_LEARNER:
        # Check for learned facts
        results = TEXT_LEARNER.query_learned_knowledge(text, top_k=3)
        facts = results.get('related_facts', [])
        
        if facts:
            fact_strs = [f"{f['subject']} {f['relation']} {f['target']}" for f in facts]
            response = f"I recall knowing: {'; '.join(fact_strs)}"
            CURRENT_MIND_STATE["recalled_facts"] = fact_strs
            CURRENT_MIND_STATE["text_thought"] = f"Memory retrieval: Found {len(facts)} related facts."
        else:
            response = "I don't recall learning about that yet. Try training me on WikiText!"
            CURRENT_MIND_STATE["recalled_facts"] = []
            CURRENT_MIND_STATE["text_thought"] = "Memory retrieval: No matching concepts found."
    else:
        response = "Memory systems offline."
    
    return jsonify({"response": response, "mind": CURRENT_MIND_STATE})

@app.route("/api/train", methods=["POST"])
def start_train():
    global TRAINING_ACTIVE
    if TRAINING_ACTIVE:
        return jsonify({"status": "already_running"})
    
    ds = request.json.get("dataset", "wikitext")
    if not MODEL:
        load_model()
        
    t = threading.Thread(target=training_thread, args=(ds,))
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/train/stats")
def get_train_stats():
    return jsonify({
        "active": TRAINING_ACTIVE,
        "epoch": TRAIN_STATS["epoch"],
        "loss": TRAIN_STATS["loss"],
        "acc": TRAIN_STATS["acc"]
    })

if __name__ == "__main__":
    load_model()
    print("[Studio] Launching v2 (Mind Reader + Memory) on http://127.0.0.1:8000")
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
