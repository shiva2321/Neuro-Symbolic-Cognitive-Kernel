const API_BASE = "http://localhost:8080/api/v5";
let visualizer;

async function init() {
    visualizer = new SocietalVisualizer('societal-canvas');
    visualizer.animate();

    // Event Handlers
    document.getElementById('btn-tick').onclick = advanceEpoch;
    window.onNodeSelect = handleNodeSelection;

    // Initial Load
    await refreshData();

    // Auto-refresh every 5 seconds
    setInterval(refreshData, 5000);
}

async function refreshData() {
    try {
        const [status, world] = await Promise.all([
            fetch(`${API_BASE}/status`).then(r => r.json()),
            fetch(`${API_BASE}/world`).then(r => r.json())
        ]);

        updateStats(status);
        visualizer.setData(world);
        addTrace(`Epoch ${status.epoch}: World state synchronized.`);

    } catch (err) {
        console.error("Refresh failed:", err);
        addTrace("Error: Could not connect to API.");
    }
}

function updateStats(status) {
    document.getElementById('stat-epoch').innerText = status.epoch;
    document.getElementById('stat-population').innerText = status.population;
    document.getElementById('stat-domains').innerText = status.domains;
}

async function advanceEpoch() {
    try {
        const res = await fetch(`${API_BASE}/tick`, { method: 'POST' });
        const data = await res.json();
        addTrace(`Manual Advance: Moving to Epoch ${data.epoch}...`);
        await refreshData();
    } catch (err) {
        addTrace("Manual Advance Failed.");
    }
}

async function handleNodeSelection(node) {
    const titleEl = document.getElementById('detail-title');
    const bodyEl = document.getElementById('detail-body');

    titleEl.innerText = `${node.level}: ${node.label}`;
    bodyEl.innerHTML = `<div class="loading-spinner">Fetching details...</div>`;

    try {
        let details;
        if (node.type === 'domain') {
            details = await fetch(`${API_BASE}/domain/${node.id}`).then(r => r.json());
            bodyEl.innerHTML = `
                <div class="detail-item"><span class="label">Hierarchy:</span> <span class="value">${details.level}</span></div>
                <div class="detail-item"><span class="label">Health:</span> <span class="value">${(details.health || 0).toFixed(2)}</span></div>
                <div class="detail-item"><span class="label">Neighborhoods:</span> <span class="value">${details.neighborhoods.length}</span></div>
                <div class="detail-item"><span class="label">Entropy:</span> <span class="value">${details.entropy.toFixed(3)}</span></div>
            `;
        } else {
            details = await fetch(`${API_BASE}/neighborhood/${node.id}`).then(r => r.json());
            bodyEl.innerHTML = `
                <div class="detail-item"><span class="label">Domain:</span> <span class="value">${details.domain}</span></div>
                <div class="detail-item"><span class="label">Population:</span> <span class="value">${details.members.length} citizens</span></div>
                <div class="detail-item"><span class="label">Stability:</span> <span class="value">${details.stability.toFixed(2)}</span></div>
                <div class="detail-item"><span class="label">Topology (B0, B1):</span> <span class="value">${details.betti.b0}, ${details.betti.b1}</span></div>
                <div class="detail-item"><span class="label">Social Energy:</span> <span class="value">${details.energy.toFixed(2)}</span></div>
            `;
        }
    } catch (err) {
        bodyEl.innerText = "Error loading details.";
    }
}

function addTrace(msg) {
    const log = document.getElementById('trace-log');
    const entry = document.createElement('div');
    entry.className = 'trace-entry';
    entry.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    log.prepend(entry);
    if (log.childNodes.length > 50) log.lastChild.remove();
}

document.addEventListener('DOMContentLoaded', init);
