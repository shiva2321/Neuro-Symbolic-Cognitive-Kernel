const socket = io();

// --- STATE MAPPING ---
// Map module names to their "Activity" status
const moduleState = {
    "SNN": { selector: "mod-snn" },
    "RULES": { selector: "mod-rules" },
    "GlobalWorkspace": { selector: "mod-global" },
    "CausalReasoner": { selector: "mod-causal" },
    "EmotionSystem": { selector: "mod-emotion" },
    "Planner": { selector: "mod-planner" }
};

// --- DOM ELEMENTS ---
const statusEl = document.getElementById('conn-status');
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const workspaceStream = document.getElementById('workspace-stream');
const logStream = document.getElementById('log-stream');
const chatHistory = document.getElementById('chat-history');
const scoreEl = document.getElementById('metric-score');
const rewardEl = document.getElementById('metric-reward');

// --- CHART SETUP ---
const ctxChart = document.getElementById('scoreChart').getContext('2d');
const perfChart = new Chart(ctxChart, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Reward',
            data: [],
            borderColor: '#00f3ff',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { display: false },
            y: {
                grid: { color: '#1f1f3f' },
                ticks: { color: '#888' }
            }
        },
        plugins: { legend: { display: false } },
        animation: false
    }
});

// --- CONNECTION HANDLERS ---
socket.on('connect', () => {
    statusEl.innerHTML = "ONLINE";
    statusEl.style.color = "#0aff00";
    addLog("SYSTEM", "Connected to NSCK Mission Control.");
});

socket.on('disconnect', () => {
    statusEl.innerHTML = "OFFLINE";
    statusEl.style.color = "#ff0055";
});

// --- GAME STATUS HANDLER ---
socket.on('game_status', (data) => {
    // data = { active: true, game: 'snake' }
    const launchPad = document.querySelector('.launch-pad');
    if (data.active) {
        // Change buttons to reflect active state
        // Find the button for the active game and turn it into STOP
        document.querySelectorAll('.btn-game').forEach(btn => {
            if (btn.innerText.includes(data.game.toUpperCase())) {
                btn.classList.add('btn-active-game');
                btn.innerText = `STOP ${data.game.toUpperCase()}`;
                btn.onclick = () => sendCommand('stop_game');
            } else {
                btn.disabled = true;
                btn.style.opacity = '0.5';
            }
        });
    } else {
        // Reset all buttons
        launchPad.innerHTML = `
            <button class="btn btn-game" onclick="sendCommand('start_game', {game: 'snake'})">SNAKE</button>
            <button class="btn btn-game" onclick="sendCommand('start_game', {game: 'pong'})">PONG</button>
            <button class="btn btn-game" onclick="sendCommand('start_game', {game: 'maze'})">MAZE</button>
            <button class="btn btn-stop" onclick="sendCommand('stop_game')">TERMINATE</button>
        `;
    }
});

// --- VISUAL CORTEX RENDERER ---
socket.on('game_update', (data) => {
    // Hide overlay
    document.getElementById('visual-overlay').style.display = 'none';

    // Render Image
    if (data.image) {
        const img = new Image();
        img.onload = () => {
            ctx.imageSmoothingEnabled = false; // Pixel art style
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = "data:image/png;base64," + data.image;
    }

    // Update Metrics
    if (data.score !== undefined) {
        scoreEl.innerText = data.score;
    }
    if (data.reward !== undefined) {
        rewardEl.innerText = data.reward.toFixed(2);
        updateChart(data.reward);
    }
    if (data.entropy) {
        document.getElementById('entropy-val').innerText = data.entropy.toFixed(2);
    }
    if (data.task) {
        updateActiveModule('mod-snn', true); // Vision always active on update
    }

    // Update Global Frame Count
    if (data.frame !== undefined) {
        document.getElementById('frame-count').innerText = data.frame;
    } else if (data.step !== undefined) {
        document.getElementById('frame-count').innerText = data.step;
    }
});

// --- TELEMETRY & WORKSPACE ---
socket.on('workspace_update', (data) => {
    // data.winner = "SNN", "PLANNER", etc.
    // data.active_modules = ["SNN", "RULES"]

    // Update Module List UI
    for (const [key, val] of Object.entries(moduleState)) {
        updateActiveModule(val.selector, false); // Reset all first? Or fade?
    }

    if (data.active_modules) {
        data.active_modules.forEach(mod => {
            if (moduleState[mod]) {
                updateActiveModule(moduleState[mod].selector, true);
            }
        });
    }

    // Log winner to stream
    if (data.winner && data.winner !== "NONE") {
        addThought(data.winner, `WINNER: ${data.winner}`);
        updateActiveModule("mod-global", true);
    }
});

socket.on('log_entry', (data) => {
    addLog(data.source, data.message);
});

socket.on('chat_response', (data) => {
    addChat("AGENT", data.text);
});

// --- CONTROLS ---
function sendCommand(type, data) {
    socket.emit(type, data);
}

function toggleAdmin(cmd) {
    const btn = document.getElementById(cmd === 'toggle_dream' ? 'btn-dream' : (cmd === 'toggle_teacher' ? 'btn-teacher' : 'btn-sleep'));
    btn.classList.toggle('active');

    socket.emit('admin_command', { cmd: cmd });
}

function fullReset() {
    if (confirm("WARNING: This will delete the brain's memory. Are you sure?")) {
        socket.emit('admin_command', { cmd: 'full_reset' });
        location.reload();
    }
}

function toggleDevice() {
    const isGpu = document.getElementById('device-toggle').checked;
    const dev = isGpu ? 'gpu' : 'cpu';
    document.getElementById('device-stat').innerText = dev.toUpperCase();
    socket.emit('admin_command', { cmd: 'set_device', device: dev });
}

function trainText() {
    const txt = document.getElementById('train-text').value;
    if (txt) {
        socket.emit('train_char', { text: txt });
        document.getElementById('train-text').value = "";
        addLog("USER", `Training started on: "${txt}"`);
    }
}

// --- CHAT ---
function sendChat() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (text) {
        addChat("USER", text);
        socket.emit('submit_chat', { text: text });
        input.value = '';
    }
}

document.getElementById('chat-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChat();
});

// --- UI HELPERS ---
function addLog(source, msg) {
    const div = document.createElement('div');
    div.innerText = `[${new Date().toLocaleTimeString()}] ${source}: ${msg}`;
    div.style.marginBottom = "4px";
    if (source === "ERROR") div.style.color = "#ff0055";
    logStream.appendChild(div);
    logStream.scrollTop = logStream.scrollHeight;
}

function addThought(source, msg) {
    const div = document.createElement('div');
    div.className = "thought-entry";
    div.innerText = msg;
    if (source === "SNN") div.style.color = "#00f3ff";
    if (source === "RULES") div.style.color = "#0aff00";
    if (source === "PLANNER") div.style.color = "#bc13fe";

    workspaceStream.appendChild(div);
    workspaceStream.scrollTop = workspaceStream.scrollHeight;
}

function addChat(sender, text) {
    const div = document.createElement('div');
    div.innerHTML = `<b style="color:${sender === 'USER' ? '#00f3ff' : '#0aff00'}">${sender}:</b> ${text}`;
    div.style.padding = "5px";
    div.style.background = sender === 'USER' ? 'rgba(0,243,255,0.1)' : 'rgba(10,255,0,0.1)';
    div.style.marginBottom = "5px";
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updateChart(reward) {
    if (perfChart.data.datasets[0].data.length > 50) {
        perfChart.data.datasets[0].data.shift();
        perfChart.data.labels.shift();
    }
    perfChart.data.labels.push("");
    perfChart.data.datasets[0].data.push(reward);
    perfChart.update('none'); // 'none' for performance
}

function updateActiveModule(id, isActive) {
    const el = document.getElementById(id);
    if (el) {
        if (isActive) {
            el.classList.remove('mod-inactive');
            el.classList.add('mod-active');
        } else {
            // Optional: Auto-dim after timeout?
            // For now, keep it simple.
        }
    }
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    document.getElementById(`tab-${tabName}`).classList.add('active');
    // Find button
    const btns = document.querySelectorAll('.tab-btn');
    if (tabName === 'chat') btns[0].classList.add('active');
    if (tabName === 'logs') btns[1].classList.add('active');
}
