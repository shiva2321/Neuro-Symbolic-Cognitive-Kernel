// NCGN Cognitive Cockpit - Dashboard JavaScript

// Global state
const dashboardState = {
    socket: null,
    charts: {},
    currentPanel: 'overview',
    lossData: [],
    hardwareHistory: [],
    updateInterval: null
};

// Initialize dashboard on load
document.addEventListener('DOMContentLoaded', function() {
    initializeSocket();
    initializeNavigation();
    initializeCharts();
    initializeFileUpload();
    refreshStatus();
    startAutoUpdate();

    addLog('Dashboard initialized');
});

// ============================================================================
// SOCKET.IO CONNECTION
// ============================================================================

function initializeSocket() {
    dashboardState.socket = io();

    dashboardState.socket.on('connect', function() {
        updateConnectionStatus(true);
        addLog('Connected to server');
    });

    dashboardState.socket.on('disconnect', function() {
        updateConnectionStatus(false);
        addLog('Disconnected from server', 'error');
    });

    // Listen for real-time updates
    dashboardState.socket.on('metrics_update', handleMetricsUpdate);
    dashboardState.socket.on('hardware_update', handleHardwareUpdate);
    dashboardState.socket.on('training_update', handleTrainingUpdate);
    dashboardState.socket.on('hebbian_update', handleHebbianUpdate);
    dashboardState.socket.on('training_complete', handleTrainingComplete);
    dashboardState.socket.on('training_error', handleTrainingError);
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-status');
    const text = document.getElementById('status-text');

    if (connected) {
        indicator.style.color = '#10b981';
        text.textContent = 'Connected';
    } else {
        indicator.style.color = '#ef4444';
        text.textContent = 'Disconnected';
    }
}

// ============================================================================
// NAVIGATION
// ============================================================================

function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');

    navButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const panelName = this.getAttribute('data-panel');
            switchPanel(panelName);
        });
    });
}

function switchPanel(panelName) {
    // Update buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-panel') === panelName) {
            btn.classList.add('active');
        }
    });

    // Update panels
    document.querySelectorAll('.dashboard-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    document.getElementById(`panel-${panelName}`).classList.add('active');
    dashboardState.currentPanel = panelName;

    addLog(`Switched to ${panelName} panel`);
}

// ============================================================================
// CHART INITIALIZATION
// ============================================================================

function initializeCharts() {
    // Loss Chart
    const lossCtx = document.getElementById('loss-chart');
    if (lossCtx) {
        dashboardState.charts.loss = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4
                }, {
                    label: 'System 1 Loss',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f1f5f9' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }

    // Memory Timeline Chart
    const memCtx = document.getElementById('memory-timeline');
    if (memCtx) {
        dashboardState.charts.memory = new Chart(memCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU Memory (GB)',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f1f5f9' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' },
                        min: 0,
                        max: 12
                    }
                }
            }
        });
    }

    // Energy Chart
    const energyCtx = document.getElementById('energy-chart');
    if (energyCtx) {
        dashboardState.charts.energy = new Chart(energyCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Energy (mJ)',
                    data: [],
                    backgroundColor: '#10b981',
                    borderColor: '#059669',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f1f5f9' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }
}

// ============================================================================
// STATUS & UPDATES
// ============================================================================

async function refreshStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        // Update status indicators
        document.getElementById('model-loaded').textContent = status.model_loaded ? '✅' : '❌';
        document.getElementById('graph-loaded').textContent = status.graph_loaded ? '✅' : '❌';
        document.getElementById('training-active').textContent = status.training_active ? '▶️' : '⏸️';

        // Update badge
        const badge = document.getElementById('model-status');
        if (status.model_loaded) {
            badge.textContent = 'Ready';
            badge.style.background = '#10b981';
        } else {
            badge.textContent = 'Not Loaded';
            badge.style.background = '#f59e0b';
        }

        updateLastUpdateTime();

    } catch (error) {
        console.error('Error refreshing status:', error);
        addLog('Error refreshing status', 'error');
    }
}

async function loadModel() {
    try {
        addLog('Loading model...');

        const response = await fetch('/api/model/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: 'saved_models/ncgn/' })
        });

        const result = await response.json();

        if (result.status === 'success') {
            addLog('Model loaded successfully', 'success');
            refreshStatus();
        } else {
            addLog(`Error: ${result.error}`, 'error');
        }

    } catch (error) {
        console.error('Error loading model:', error);
        addLog('Failed to load model', 'error');
    }
}

function startAutoUpdate() {
    // Request updates every 2 seconds
    dashboardState.updateInterval = setInterval(() => {
        if (dashboardState.socket) {
            dashboardState.socket.emit('request_update');
        }
    }, 2000);
}

// ============================================================================
// REAL-TIME UPDATE HANDLERS
// ============================================================================

function handleMetricsUpdate(data) {
    // Update metric displays
    if (data.average_performance !== undefined) {
        document.getElementById('metric-ap').textContent = (data.average_performance * 100).toFixed(1) + '%';
    }

    if (data.average_forgetting !== undefined) {
        document.getElementById('metric-af').textContent = (data.average_forgetting * 100).toFixed(1) + '%';
    }

    if (data.train_loss !== undefined) {
        document.getElementById('metric-loss').textContent = data.train_loss.toFixed(4);
    }

    if (data.confidence !== undefined) {
        document.getElementById('metric-confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    }
}

function handleHardwareUpdate(data) {
    const gpu = data.gpu || {};
    const cpu = data.cpu || {};
    const memory = data.memory || {};
    const energy = data.energy || {};

    // Update GPU progress bars
    const gpuMemPercent = (gpu.memory_allocated / gpu.memory_total * 100) || 0;
    document.getElementById('gpu-memory-bar').style.width = gpuMemPercent + '%';
    document.getElementById('gpu-memory-text').textContent =
        `${gpu.memory_allocated?.toFixed(1) || 0} / ${gpu.memory_total?.toFixed(1) || 0} GB`;

    document.getElementById('gpu-util-bar').style.width = (gpu.utilization || 0) + '%';
    document.getElementById('gpu-util-text').textContent = (gpu.utilization || 0) + '%';

    // Update RAM
    const ramPercent = memory.percent || 0;
    document.getElementById('ram-bar').style.width = ramPercent + '%';
    document.getElementById('ram-text').textContent =
        `${memory.used?.toFixed(1) || 0} / ${memory.total?.toFixed(1) || 0} GB`;

    // Update energy
    document.getElementById('energy-text').textContent =
        (energy.total_mj || 0).toFixed(0) + ' mJ';

    // Update hardware panel details
    document.getElementById('hw-gpu-name').textContent = gpu.name || 'N/A';
    document.getElementById('hw-gpu-mem').textContent =
        `${gpu.memory_allocated?.toFixed(1) || 0} GB`;
    document.getElementById('hw-gpu-util').textContent = (gpu.utilization || 0) + '%';
    document.getElementById('hw-gpu-temp').textContent = (gpu.temperature || 0) + '°C';
    document.getElementById('hw-gpu-power').textContent = (gpu.power_usage || 0).toFixed(1) + ' W';

    document.getElementById('hw-cpu-util').textContent = (cpu.utilization || 0).toFixed(1) + '%';
    document.getElementById('hw-ram-used').textContent =
        `${memory.used?.toFixed(1) || 0} GB`;
    document.getElementById('hw-ram-avail').textContent =
        `${memory.available?.toFixed(1) || 0} GB`;

    // Update device name
    document.getElementById('device-name').textContent = gpu.name || 'N/A';

    // Update memory chart
    if (dashboardState.charts.memory) {
        const chart = dashboardState.charts.memory;
        const now = new Date().toLocaleTimeString();

        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(gpu.memory_allocated || 0);

        // Keep only last 50 points
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update('none');
    }
}

function handleTrainingUpdate(data) {
    addLog(`Epoch ${data.epoch}: Loss = ${data.metrics.loss?.toFixed(4) || 'N/A'}`);

    // Update loss chart
    if (dashboardState.charts.loss) {
        const chart = dashboardState.charts.loss;

        chart.data.labels.push(data.epoch);
        chart.data.datasets[0].data.push(data.metrics.loss || 0);
        chart.data.datasets[1].data.push(data.metrics.system1_loss || 0);

        chart.update();
    }

    // Update metrics
    handleMetricsUpdate(data.metrics);
}

function handleHebbianUpdate(data) {
    addLog('Hebbian traces updated');
    // Update visualization if on visualization panel
    if (dashboardState.currentPanel === 'visualization') {
        updateHebbianViz(data);
    }
}

function handleTrainingComplete(data) {
    addLog('Training complete!', 'success');
    alert('Training completed successfully!');
}

function handleTrainingError(data) {
    addLog(`Training error: ${data.error}`, 'error');
    alert(`Training error: ${data.error}`);
}

// ============================================================================
// TRAINING CONTROLS
// ============================================================================

async function startTraining() {
    try {
        const params = {
            learning_rate: parseFloat(document.getElementById('learning-rate').value),
            batch_size: parseInt(document.getElementById('batch-size').value),
            epochs: parseInt(document.getElementById('epochs').value),
            weight_decay: parseFloat(document.getElementById('weight-decay').value)
        };

        addLog('Starting training...');

        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const result = await response.json();

        if (result.status === 'success') {
            addLog('Training started', 'success');
        } else {
            addLog(`Error: ${result.error}`, 'error');
        }

    } catch (error) {
        console.error('Error starting training:', error);
        addLog('Failed to start training', 'error');
    }
}

async function stopTraining() {
    try {
        const response = await fetch('/api/training/stop', {
            method: 'POST'
        });

        const result = await response.json();
        addLog('Training stopped');

    } catch (error) {
        console.error('Error stopping training:', error);
    }
}

// ============================================================================
// FILE UPLOAD
// ============================================================================

function initializeFileUpload() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#2563eb';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#334155';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#334155';

        const files = e.dataTransfer.files;
        handleFileUpload(files);
    });

    // Click upload
    fileInput.addEventListener('change', (e) => {
        handleFileUpload(e.target.files);
    });
}

async function handleFileUpload(files) {
    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            addLog(`Uploading ${file.name}...`);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                addLog(`✓ Uploaded ${file.name} - ${result.chunks_loaded} text chunks extracted`, 'success');
                addFileToList(file.name, result.chunks_loaded);
            } else if (result.status === 'warning') {
                addLog(`⚠ Uploaded ${file.name} but no text extracted - ${result.message}`, 'warning');
                addLog(`  Check if PDF is image-based or needs OCR`, 'warning');
                // Still add to file list but mark as problematic
                addFileToList(file.name + ' (⚠ No text)', 0);
            } else {
                addLog(`✗ Failed to upload ${file.name}: ${result.error || 'Unknown error'}`, 'error');
            }

        } catch (error) {
            console.error('Upload error:', error);
            addLog(`✗ Error uploading ${file.name}: ${error.message}`, 'error');
        }
    }
}

function addFileToList(filename, size) {
    const fileList = document.getElementById('uploaded-files');
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.innerHTML = `
        <span>📄 ${filename}</span>
        <span>${size} items</span>
    `;
    fileList.appendChild(fileItem);
}

// ============================================================================
// INFERENCE / PLAYGROUND
// ============================================================================

async function runInference() {
    try {
        const query = document.getElementById('query-input').value;
        const contextInput = document.getElementById('context-input').value;

        if (!query) {
            alert('Please enter a query');
            return;
        }

        addLog('Running inference...');

        // Parse context
        let context = {};
        if (contextInput) {
            const parts = contextInput.split(':');
            if (parts.length === 2) {
                context[parts[0].trim()] = parts[1].trim();
            }
        }

        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, context })
        });

        const result = await response.json();

        if (result.status === 'success') {
            // Display results
            document.getElementById('response-mode').textContent = result.mode;
            document.getElementById('response-confidence').textContent =
                (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('response-explanation').textContent = result.explanation;

            addLog('Inference complete', 'success');
        } else {
            addLog(`Error: ${result.error}`, 'error');
        }

    } catch (error) {
        console.error('Inference error:', error);
        addLog('Inference failed', 'error');
    }
}

// ============================================================================
// VISUALIZATION
// ============================================================================

async function loadGraphData() {
    try {
        addLog('Loading graph data...');

        const response = await fetch('/api/graph/data');
        const data = await response.json();

        if (data.nodes && data.edges) {
            visualizeGraph(data);
            addLog(`Loaded graph: ${data.nodes.length} nodes, ${data.edges.length} edges`, 'success');
        }

    } catch (error) {
        console.error('Error loading graph:', error);
        addLog('Failed to load graph', 'error');
    }
}

function visualizeGraph(data) {
    // D3.js force-directed graph
    const container = d3.select('#graph-viz');
    container.html(''); // Clear previous

    const width = container.node().getBoundingClientRect().width;
    const height = 600;

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));

    // Draw edges
    const link = svg.append('g')
        .selectAll('line')
        .data(data.edges)
        .enter().append('line')
        .attr('stroke', '#334155')
        .attr('stroke-width', d => Math.sqrt(d.weight) * 2);

    // Draw nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(data.nodes)
        .enter().append('circle')
        .attr('r', d => Math.sqrt(d.frequency) + 3)
        .attr('fill', '#2563eb')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    // Add labels
    const label = svg.append('g')
        .selectAll('text')
        .data(data.nodes)
        .enter().append('text')
        .text(d => d.label)
        .attr('font-size', 10)
        .attr('fill', '#94a3b8')
        .attr('dx', 12)
        .attr('dy', 4);

    // Update positions
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
}

function updateHebbianViz(data) {
    // Placeholder for Hebbian trace visualization
    console.log('Updating Hebbian visualization:', data);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function addLog(message, type = 'info') {
    const logContainer = document.getElementById('activity-log');
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';

    const time = new Date().toLocaleTimeString();
    const icon = type === 'error' ? '❌' :
                 type === 'success' ? '✅' :
                 type === 'warning' ? '⚠️' : 'ℹ️';

    logEntry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message">${icon} ${message}</span>
    `;

    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function clearLog() {
    document.getElementById('activity-log').innerHTML = '';
    addLog('Log cleared');
}

function updateLastUpdateTime() {
    document.getElementById('last-update').textContent = new Date().toLocaleString();
}

function resetZoom(chartId) {
    if (dashboardState.charts[chartId.replace('-chart', '')]) {
        dashboardState.charts[chartId.replace('-chart', '')].resetZoom();
    }
}

// Update graph layout
function updateGraphLayout() {
    loadGraphData();
}

// Update engrams
function updateEngrams() {
    const layer = document.getElementById('engram-layer').value;
    addLog(`Loading engrams for ${layer}...`);
}

