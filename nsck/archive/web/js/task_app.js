const socket = io();

// Get target game from URL: /task/snake -> ["", "task", "snake"]
const pathParts = window.location.pathname.split('/');
const gameType = pathParts[pathParts.length - 1] || 'global';

document.getElementById('task-name').innerText = gameType.toUpperCase();

// --- STATE ---
// --- STATE ---
const statusEl = document.getElementById('conn-status');
// Cortex Canvas (Low Res)
const canvasCortex = document.getElementById('canvas-task');
const ctxCortex = canvasCortex.getContext('2d');
// Game Canvas (High Res)
const canvasGame = document.getElementById('canvas-game');
const ctxGame = canvasGame ? canvasGame.getContext('2d') : null;

const logStream = document.getElementById('log-stream');
const reasoningContent = document.getElementById('reasoning-content');

// --- CHART ---
const ctxChart = document.getElementById('taskChart').getContext('2d');
const taskChart = new Chart(ctxChart, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Reward',
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
            y: { grid: { color: '#1f1f3f' }, ticks: { color: '#888' } },
            x: { display: false }
        },
        plugins: { legend: { display: false } },
        animation: false
    }
});

// --- CONNECTION ---
socket.on('connect', () => {
    statusEl.innerText = "ONLINE";
    statusEl.style.color = "#0aff00";
    socket.emit('join_task', { game: gameType });
    addLog("SYSTEM", `Connected to NSCK - Monitoring ${gameType.toUpperCase()}`);
});

socket.on('disconnect', () => {
    statusEl.innerText = "OFFLINE";
    statusEl.style.color = "#ff0055";
});

// --- TELEMETRY ---
socket.on('game_update', (data) => {
    // 1. Render Cortex View (10x10 Reconstructed)
    if (data.image) {
        const img = new Image();
        img.onload = () => {
            ctxCortex.imageSmoothingEnabled = false;
            ctxCortex.drawImage(img, 0, 0, canvasCortex.width, canvasCortex.height);
        };
        img.src = "data:image/png;base64," + data.image;
    }

    // 2. Render Reality View (High Res State)
    if (ctxGame && data.state) { // Check if raw state is available
        // Clear
        ctxGame.fillStyle = "#000";
        ctxGame.fillRect(0, 0, canvasGame.width, canvasGame.height);

        let state = data.state || {};

        // SNAKE RENDER
        if (gameType === 'snake' && state.body) {
            // Scale: 10x10 grid -> 300x300 canvas (30px per cell)
            const S = 30;
            // Food
            if (state.food) {
                ctxGame.fillStyle = "#ff0055"; // Red Apple
                ctxGame.fillRect(state.food[0] * S, state.food[1] * S, S, S);
            }
            // Snake
            ctxGame.fillStyle = "#0aff00"; // Green Snake
            state.body.forEach(seg => {
                ctxGame.fillRect(seg[0] * S, seg[1] * S, S - 2, S - 2); // -2 for gap
            });
        }

        // PONG RENDER
        else if (gameType === 'pong' && state.ball_x !== undefined) {
            const S = 10;
            // Ball
            ctxGame.fillStyle = "#ffffff";
            ctxGame.beginPath();
            ctxGame.arc(state.ball_x * S + 5, state.ball_y * S + 5, 5, 0, Math.PI * 2);
            ctxGame.fill();
            // Paddle 1
            ctxGame.fillStyle = "#0aff00";
            ctxGame.fillRect(0, state.p1_y * S, 10, 60);
            // Paddle 2
            ctxGame.fillStyle = "#ff0055";
            ctxGame.fillRect(290, (state.p2_y !== undefined ? state.p2_y : 10) * S, 10, 60);
        }

        // MAZE RENDER
        else if (gameType === 'maze' && state.player_pos) {
            const S = 30;
            // Walls
            if (state.walls) {
                ctxGame.fillStyle = "#444";
                state.walls.forEach(w => {
                    ctxGame.fillRect(w[0] * S, w[1] * S, S, S);
                });
            }
            // Goal
            if (state.exit_pos) {
                ctxGame.fillStyle = "#ffff00";
                ctxGame.fillRect(state.exit_pos[0] * S, state.exit_pos[1] * S, S, S);
            }
            // Player (Draw last to be on top)
            ctxGame.fillStyle = "#00f3ff";
            ctxGame.beginPath();
            ctxGame.arc(state.player_pos[0] * S + 15, state.player_pos[1] * S + 15, 10, 0, Math.PI * 2);
            ctxGame.fill();
        }
    }

    // 3. Update Teacher Status
    const teacherEl = document.getElementById('teacher-status');
    if (teacherEl) {
        if (data.teacher_active) {
            teacherEl.innerText = "ON";
            teacherEl.style.color = "#0aff00"; // Green
        } else {
            teacherEl.innerText = "OFF";
            teacherEl.style.color = "#ff0055"; // Red
        }
    }

    if (data.reward !== undefined) {
        updateChart(data.reward);
    }

    if (data.entropy !== undefined) document.getElementById('entropy-val').innerText = data.entropy.toFixed(2);
    if (data.avg_conf !== undefined) document.getElementById('conf-val').innerText = (data.avg_conf * 100).toFixed(1) + "%";

    // Update Thoughts/Reasoning Panel
    updateReasoning(data);
});

socket.on('log_entry', (data) => {
    addLog(data.source, data.message);
});

socket.on('game_status', (data) => {
    if (data.game === gameType || data.all) {
        if (!data.active) {
            statusEl.innerText = "DORMANT";
            statusEl.style.color = "#ffaa00";
            clearCanvases();
        } else {
            statusEl.innerText = "ONLINE";
            statusEl.style.color = "#0aff00";
        }
    }
});

socket.on('workspace_update', (data) => {
    if (data.competition) {
        // Find if this game was involved or show specific planning
        const winner = data.winner || "UNKNOWN";
        addLog("BRAIN", `Workspace Winner: ${winner}`);
    }
});

socket.on('chat_response', (data) => {
    if (data.trace) {
        showReasoning(data.trace);
    }
});

// --- HELPERS ---
function clearCanvases() {
    ctxCortex.fillStyle = "#000";
    ctxCortex.fillRect(0, 0, canvasCortex.width, canvasCortex.height);
    if (ctxGame) {
        ctxGame.fillStyle = "#000";
        ctxGame.fillRect(0, 0, canvasGame.width, canvasGame.height);
    }
}

function updateChart(reward) {
    if (taskChart.data.datasets[0].data.length > 100) {
        taskChart.data.datasets[0].data.shift();
        taskChart.data.labels.shift();
    }
    taskChart.data.labels.push("");
    taskChart.data.datasets[0].data.push(reward);
    taskChart.update('none');
}

function addLog(source, msg) {
    const div = document.createElement('div');
    div.innerText = `[${new Date().toLocaleTimeString()}] ${source}: ${msg}`;
    div.style.marginBottom = "4px";
    if (source === "ERROR") div.style.color = "#ff0055";
    if (source === "PLANNER") div.style.color = "#bc13fe";
    logStream.appendChild(div);
    logStream.scrollTop = logStream.scrollHeight;
}

// --- REASONING & THOUGHTS ---
function updateReasoning(data) {
    reasoningContent.innerHTML = ""; // Clear previous frame

    // 1. Decision Source Header
    const srcDiv = document.createElement('div');
    srcDiv.style.padding = "5px";
    srcDiv.style.marginBottom = "10px";
    srcDiv.style.borderRadius = "4px";
    srcDiv.style.fontWeight = "bold";

    let srcColor = "#888";
    if (data.decision_source === "USER") srcColor = "#ff0055"; // Red for Override
    if (data.decision_source === "TEACHER") srcColor = "#0aff00"; // Green for Algo
    if (data.decision_source === "BRAIN") srcColor = "#00f3ff"; // Cyan for AI
    if (data.decision_source === "VETO") srcColor = "#bc13fe"; // Purple for Veto

    srcDiv.style.border = `1px solid ${srcColor}`;
    srcDiv.style.color = srcColor;
    srcDiv.innerHTML = `Running: ${data.decision_source} <br><span style="font-size:0.8em; color:#bbb">${data.decision_reason || ''}</span>`;
    // 2. Continuous "Stream of Consciousness" Log
    // We want a piling up history, not just current frame
    if (data.decision_source) {
        const entry = document.createElement('div');
        entry.className = "thought-entry";
        entry.style.borderLeft = `3px solid ${srcColor}`;
        entry.style.background = "rgba(0,0,0,0.2)";
        entry.style.marginBottom = "5px";
        entry.style.padding = "5px";
        entry.style.fontSize = "0.85em";

        let content = `<span style="color:${srcColor}; font-weight:bold">${data.decision_source}</span>`;
        if (data.decision_reason) content += ` <span style="color:#888">| ${data.decision_reason}</span>`;
        if (data.veto_log && data.veto_log !== "VETO") content += `<br><span style="color:#ddd">RMS: ${data.veto_log}</span>`;

        entry.innerHTML = content;

        // Append
        reasoningContent.appendChild(entry);

        // Limit history (last 50 thoughts)
        if (reasoningContent.childElementCount > 50) {
            reasoningContent.removeChild(reasoningContent.firstChild);
        }

        // Auto-scroll
        reasoningContent.scrollTop = reasoningContent.scrollHeight;
    }
    if (data.veto_log && data.veto_log !== "VETO") {
        const logDiv = document.createElement('div');
        logDiv.innerHTML = `<h4>THOUGHTS</h4><p>${data.veto_log}</p>`;
        reasoningContent.appendChild(logDiv);
    }
}

function terminateTask() {
    socket.emit('stop_game', { game: gameType });
    window.close();
}

function startGame() {
    socket.emit('start_game', { game: gameType });
    addLog("SYSTEM", "Task Start Requested...");
}

function stopGame() {
    socket.emit('stop_game', { game: gameType });
    addLog("SYSTEM", "Task Stop Requested...");
}

function restartTask() {
    // Stop then start
    socket.emit('stop_game', { game: gameType });
    setTimeout(() => {
        socket.emit('start_game', { game: gameType });
        // Clear chart
        taskChart.data.labels = [];
        taskChart.data.datasets[0].data = [];
        taskChart.update();
        addLog("SYSTEM", "Task Restart Initiated...");
    }, 500);
}

// Auto-terminate on window close
window.addEventListener('beforeunload', () => {
    socket.emit('stop_game', { game: gameType });
});

function exportTaskLog() {
    window.open(`/export_logs?task=${gameType}`, '_blank');
}
