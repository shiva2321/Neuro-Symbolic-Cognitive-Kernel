const socket = io();

// --- CHAT ---
function sendChat() {
    const input = document.getElementById('chat-input');
    const teachMode = document.getElementById('teach-mode').checked;
    const text = input.value.trim();
    if (text) {
        // Optimistic update
        addChatEntry("USER", text);
        socket.emit('submit_chat', { text: text, teach: teachMode });
        input.value = '';
    }
}

function handleChatKey(event) {
    if (event.key === 'Enter') {
        sendChat();
    }
}

function addChatEntry(sender, text) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = sender === "USER" ? "chat-msg user-msg" : "chat-msg agent-msg";
    div.innerHTML = `<span class="sender">${sender}:</span> ${text}`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

// --- DOM ELEMENTS ---
const statusEl = document.getElementById('conn-status');
const logStream = document.getElementById('log-stream');
const wsStream = document.getElementById('workspace-stream');
const ctx = document.getElementById('gameCanvas').getContext('2d');
const scoreEl = document.getElementById('score');

// --- CONNECTION ---
socket.on('connect', () => {
    statusEl.innerHTML = "ONLINE";
    statusEl.style.color = "#0aff00";
    log("SYSTEM", "Connected to Brain Interface.");
});

socket.on('disconnect', () => {
    statusEl.innerHTML = "OFFLINE";
    statusEl.style.color = "#ff0055";
    log("SYSTEM", "Connection Lost.");
});

socket.on('status', (data) => {
    log("SERVER", data.msg);
});

// --- COMMANDS ---
function sendCommand(type, data) {
    socket.emit(type, data);
    log("USER", `Sent command: ${type} ${JSON.stringify(data)}`);
}

// --- LOGGING ---
function log(source, msg) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    const time = new Date().toLocaleTimeString();
    div.innerHTML = `<span style="color: #666">[${time}]</span> <span style="color: var(--neon-blue)">${source}</span>: ${msg}`;
    logStream.appendChild(div);
    logStream.scrollTop = logStream.scrollHeight;
}

socket.on('log_entry', (data) => {
    log(data.source, data.message);
});

// --- GAME STATE RENDERING ---
socket.on('game_update', (data) => {
    // Expected format: { grid: [[R,G,B], ...], score: 10 } OR { entities: [...] }
    if (data.image) {
        // Base64 Image
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, 300, 300);
        };
        img.src = "data:image/png;base64," + data.image;
    }
    if (data.score !== undefined) scoreEl.innerText = data.score;
});

// --- BRAIN STATE ---
socket.on('brain_update', (data) => {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.style.color = "#e0e0bd";
    div.innerText = `> ${data.message || JSON.stringify(data)}`;
    wsStream.appendChild(div);
    wsStream.scrollTop = wsStream.scrollHeight;
});

// --- CHART (Placeholder) ---
// Initialize Chart.js here if we had data points
