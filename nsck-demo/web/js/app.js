const socket = io();

// --- STATE ---
let activeTasks = new Set();
let performanceChart;
const taskWindows = {}; // { snake: windowObject }

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initPerformanceChart();

    // Handle Enter key for chat
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChat();
    });
});

function initPerformanceChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'System Competence',
                data: [],
                borderColor: '#00f3ff',
                backgroundColor: 'rgba(0, 243, 255, 0.1)',
                fill: true,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: '#1f1f3f' }, ticks: { color: '#888' }, min: 0, max: 100 },
                x: { display: false }
            },
            plugins: { legend: { display: false } },
            animation: false
        }
    });
}

// --- SOCKET EVENTS ---
socket.on('connect', () => {
    document.getElementById('conn-status').innerText = "ONLINE";
    document.getElementById('conn-status').style.color = "#0aff00";
    addLog("SYSTEM", "Mission Control Online. Global Neural Link Established.");
});

socket.on('disconnect', () => {
    document.getElementById('conn-status').innerText = "OFFLINE";
    document.getElementById('conn-status').style.color = "#ff0055";
});

socket.on('game_status', (data) => {
    if (data.all && !data.active) {
        // Terminate all
        activeTasks.clear();
        document.getElementById('task-grid').innerHTML = '<div id="visual-overlay">All Tasks Terminated.</div>';
        updateTaskCount();

        // Close all child windows
        for (let game in taskWindows) {
            if (taskWindows[game] && !taskWindows[game].closed) {
                taskWindows[game].close();
            }
        }
        return;
    }

    if (data.active) {
        activeTasks.add(data.game);
        createTaskThumbnail(data.game);
        // Open dedicated window
        if (!taskWindows[data.game] || taskWindows[data.game].closed) {
            taskWindows[data.game] = window.open(`/task/${data.game}`, `NSCK_${data.game}`, "width=1000,height=800");
        }
    } else {
        activeTasks.delete(data.game);
        removeTaskThumbnail(data.game);
    }
    updateTaskCount();
});

socket.on('game_update_global', (data) => {
    // Update mini-canvas in grid
    const canvas = document.getElementById(`mini-canvas-${data.game}`);
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = "data:image/png;base64," + data.image;
    }
});

socket.on('telemetry_global', (data) => {
    // 1. Global Metrics (Average of all games if available)
    if (data.avg_reward !== undefined) document.getElementById('metric-reward').innerText = data.avg_reward.toFixed(1);
    if (data.agreement !== undefined) {
        const ag = (data.agreement * 100).toFixed(0) + "%";
        document.getElementById('metric-agree').innerText = ag;
        updatePerformanceChart(data.agreement * 100);
    }

    // 2. Per-Game Stats (Color Coding)
    // Python sends: game, agree_pct, veto_pct, etc.
    if (data.game) {
        const thumb = document.getElementById(`task-thumb-${data.game}`);
        if (thumb) {
            let color = "#1f6f8f"; // Default Blue

            if (data.agree_pct >= 80) color = "#0aff00"; // Green (High Agreement)
            else if (data.agree_pct >= 50) color = "#00f3ff"; // Cyan (Moderate)
            else if (data.agree_pct > 0) color = "#ffaa00"; // Orange (Learning/Struggling)
            else color = "#ff0055"; // Red (Low Agreement/Failure)

            thumb.style.boxShadow = `0 0 15px ${color}`;
            thumb.style.borderColor = color;
        }
    }
});

socket.on('log_entry_global', (data) => {
    addLog(data.source, data.message);
});

socket.on('workspace_update_global', (data) => {
    if (data.active_modules) {
        updateModulePulse(data.active_modules);
    }
});

socket.on('chat_response_global', (data) => {
    appendChat("AGENT", data.text);
});

socket.on('brain_update', (data) => {
    if (data.device) document.getElementById('device-stat').innerText = data.device.toUpperCase();
});

// --- UI HELPERS ---
function startTask(game) {
    socket.emit('start_game', { game: game });
}

function sendCommand(cmd, args) {
    socket.emit(cmd, args);
}

function toggleAdmin(cmd) {
    socket.emit('admin_command', { cmd: cmd });
}

function createTaskThumbnail(game) {
    const grid = document.getElementById('task-grid');
    const overlay = document.getElementById('visual-overlay');
    if (overlay) overlay.remove();

    if (document.getElementById(`task-thumb-${game}`)) return;

    const div = document.createElement('div');
    div.id = `task-thumb-${game}`;
    div.className = "task-thumb";
    div.innerHTML = `
        <span class="thumb-label">${game.toUpperCase()}</span>
        <canvas id="mini-canvas-${game}" width="150" height="150"></canvas>
        <div class="thumb-actions">
            <button class="btn btn-small" onclick="window.open('/task/${game}', 'NSCK_${game}')">FOCUS</button>
            <button class="btn btn-stop-small" onclick="socket.emit('stop_game', {game: '${game}'})">×</button>
        </div>
    `;
    grid.appendChild(div);
}

function removeTaskThumbnail(game) {
    const thumb = document.getElementById(`task-thumb-${game}`);
    if (thumb) thumb.remove();
    if (activeTasks.size === 0) {
        document.getElementById('task-grid').innerHTML = '<div id="visual-overlay">Waiting for neural activity...</div>';
    }
}

function updateTaskCount() {
    document.getElementById('active-tasks-count').innerText = `TASKS: ${activeTasks.size}`;
}

function addLog(source, msg) {
    const stream = document.getElementById('workspace-stream');
    const div = document.createElement('div');
    div.className = `thought-entry ${source.toLowerCase()}`;
    div.innerHTML = `<span class="timestamp">${new Date().toLocaleTimeString()}</span> <span class="source">[${source}]</span> ${msg}`;
    stream.appendChild(div);
    stream.scrollTop = stream.scrollHeight;
}

function sendChat() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    appendChat("USER", text);
    socket.emit('chat_message', { text: text });
    input.value = "";
}

function appendChat(role, msg) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = `chat-bubble ${role.toLowerCase()}`;
    div.innerText = msg;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

function updatePerformanceChart(val) {
    if (performanceChart.data.datasets[0].data.length > 50) {
        performanceChart.data.datasets[0].data.shift();
        performanceChart.data.labels.shift();
    }
    performanceChart.data.labels.push("");
    performanceChart.data.datasets[0].data.push(val);
    performanceChart.update('none');
}

function updateModulePulse(active) {
    const modules = ["snn", "rules", "global", "causal", "emotion", "planner", "episodic", "self_model", "vsa", "curiosity", "value", "dream"];
    modules.forEach(m => {
        const el = document.getElementById(`mod-${m}`);
        if (el) {
            if (active.includes(m.toUpperCase())) {
                el.className = "mod-active pulse";
            } else {
                el.className = "mod-inactive";
            }
        }
    });
}
