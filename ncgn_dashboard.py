"""
NCGN Cognitive Cockpit - Main Dashboard Application
Real-time monitoring, training, and visualization interface for NCGN.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import torch
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
import threading
import logging
from typing import Dict, Any, List, Optional

# NCGN imports
from ncgn import DualSystemArchitecture, LinguisticGraph, GraphBuilder
from ncgn import SpikingLayer, STDPLearning

# Dashboard utilities
import sys
sys.path.append(str(Path(__file__).parent))
from dashboard_utils.metrics_monitor import MetricsMonitor
from dashboard_utils.hardware_profiler import HardwareProfiler
from dashboard_utils.hebbian_tracker import HebbianTraceMonitor
from dashboard_utils.training_manager import TrainingManager
from dashboard_utils.data_loader import DashboardDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ncgn-cognitive-cockpit-2026'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=100000000,  # 100MB buffer (increased from 50MB)
    ping_timeout=180,  # Increased from 120 to allow more time
    ping_interval=45,  # Increased from 30 to reduce traffic
    engineio_logger=False,
    logger=False,
    max_decode_packets=200  # Increased from 50 to handle more packets per payload
)

# Global state
dashboard_state = {
    'model': None,
    'graph': None,
    'training_active': False,
    'metrics_monitor': None,
    'hardware_profiler': None,
    'hebbian_monitor': None,
    'training_manager': None,
    'config': None,
    'device': 'cpu',  # Default to CPU, will be updated when model is created
    'training_data': None,  # Store uploaded/loaded training data
    'data_filename': None   # Track which file was loaded
}


def load_config(config_path: str = "configs/ncgn_config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_dashboard():
    """Initialize dashboard components"""
    logger.info("Initializing NCGN Cognitive Cockpit...")

    # Load configuration
    dashboard_state['config'] = load_config()

    # Initialize monitors
    dashboard_state['metrics_monitor'] = MetricsMonitor()
    dashboard_state['hardware_profiler'] = HardwareProfiler()
    dashboard_state['hebbian_monitor'] = HebbianTraceMonitor()

    # Initialize training manager
    dashboard_state['training_manager'] = TrainingManager(
        config=dashboard_state['config'],
        metrics_monitor=dashboard_state['metrics_monitor'],
        hardware_profiler=dashboard_state['hardware_profiler']
    )

    logger.info("✓ Dashboard initialized successfully")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current dashboard status"""
    return jsonify({
        'model_loaded': dashboard_state['model'] is not None,
        'graph_loaded': dashboard_state['graph'] is not None,
        'training_active': dashboard_state['training_active'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify(dashboard_state['config'])

    elif request.method == 'POST':
        updates = request.json
        dashboard_state['config'].update(updates)

        # Save to file
        with open('configs/ncgn_config.yaml', 'w') as f:
            yaml.dump(dashboard_state['config'], f)

        return jsonify({'status': 'success', 'config': dashboard_state['config']})


@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Handle dataset upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save file
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    file.save(str(file_path))

    # Process file
    loader = DashboardDataLoader()
    try:
        data = loader.load_file(str(file_path))

        # Store data for training only if we actually got data
        if len(data) > 0:
            dashboard_state['training_data'] = data
            dashboard_state['data_filename'] = file.filename
            logger.info(f"Stored {len(data)} chunks from {file.filename} for training")
        else:
            logger.warning(f"No text chunks extracted from {file.filename}. File may be empty, image-based, or corrupted.")
            return jsonify({
                'status': 'warning',
                'filename': file.filename,
                'size': 0,
                'type': file_path.suffix,
                'chunks_loaded': 0,
                'ready_for_training': False,
                'message': 'No text could be extracted from this file. If it\'s a PDF, it may be image-based (needs OCR) or empty.'
            })

        # Build linguistic graph from the data if we have enough content
        if len(data) > 100:  # Only build graph if substantial data
            try:
                from ncgn.linguistic_graph import GraphBuilder
                logger.info(f"Building linguistic graph from {file.filename}...")

                builder = GraphBuilder(
                    embedding_model="roberta-base",
                    device=dashboard_state.get('device', 'cpu')
                )

                # Build graph from uploaded text
                dashboard_state['graph'] = builder.build_from_corpus(
                    corpus=data[:5000],  # Use first 5000 chunks to avoid memory issues
                    window_size=5,
                    min_frequency=2,
                    max_vocab_size=10000
                )

                logger.info(f"✓ Linguistic graph built: {dashboard_state['graph'].get_statistics()}")
            except Exception as e:
                logger.warning(f"Could not build graph: {e}")
                # Continue anyway, data is still stored

        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'size': len(data),
            'type': file_path.suffix,
            'chunks_loaded': len(data),
            'ready_for_training': True
        })
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load NCGN model"""
    try:
        model_path = request.json.get('path', 'saved_models/ncgn/')

        # Load linguistic graph
        graph_path = Path(model_path) / 'linguistic_graph'
        if graph_path.exists():
            dashboard_state['graph'] = LinguisticGraph.load(str(graph_path))
            logger.info("✓ Linguistic graph loaded")

        # Initialize model
        config = dashboard_state['config']
        dashboard_state['model'] = DualSystemArchitecture(
            node_feat_dim=config['linguistic_graph']['embedding_dim'],
            embed_dim=config['graph_transformer']['embed_dim'],
            num_layers=config['graph_transformer']['num_layers'],
            num_heads=config['graph_transformer']['num_heads']
        )

        logger.info("✓ Model initialized")

        # Attach monitors
        dashboard_state['hebbian_monitor'].attach(dashboard_state['model'])

        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training process"""
    if dashboard_state['training_active']:
        return jsonify({'error': 'Training already active'}), 400

    try:
        params = request.json

        # Initialize model if not already loaded
        if dashboard_state['model'] is None:
            logger.info("Initializing model for training...")
            config = dashboard_state['config']

            # Detect available device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")

            dashboard_state['model'] = DualSystemArchitecture(
                node_feat_dim=config['linguistic_graph']['embedding_dim'],
                embed_dim=config['graph_transformer']['embed_dim'],
                num_layers=config['graph_transformer']['num_layers'],
                num_heads=config['graph_transformer']['num_heads']
            )

            # Move model to correct device
            dashboard_state['model'] = dashboard_state['model'].to(device)
            dashboard_state['device'] = device

            # Attach monitors
            if dashboard_state['hebbian_monitor']:
                dashboard_state['hebbian_monitor'].attach(dashboard_state['model'])

            logger.info(f"✓ Model initialized for training on {device}")

        dashboard_state['training_active'] = True

        # Start training in background thread
        training_thread = threading.Thread(
            target=run_training,
            args=(params,),
            daemon=True
        )
        training_thread.start()

        return jsonify({'status': 'success', 'message': 'Training started'})

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        dashboard_state['training_active'] = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training process"""
    dashboard_state['training_active'] = False
    return jsonify({'status': 'success', 'message': 'Training stopped'})


@app.route('/api/inference', methods=['POST'])
def run_inference():
    """Run inference on query"""
    if dashboard_state['model'] is None:
        return jsonify({'error': 'Model not loaded'}), 400

    try:
        query = request.json.get('query', '')
        context = request.json.get('context', {})

        # Get device from dashboard state
        device = dashboard_state.get('device', 'cpu')

        # Prepare input
        # (Simplified - would need proper tokenization)
        node_features = torch.randn(1, 50, dashboard_state['config']['linguistic_graph']['embedding_dim']).to(device)
        adjacency = torch.eye(50).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = dashboard_state['model'](node_features, adjacency, context=context)

        # Extract attention weights if available
        attention_weights = None
        if 'system1_output' in output and output['system1_output']:
            attention_weights = dashboard_state['hebbian_monitor'].get_latest_attention()

        return jsonify({
            'status': 'success',
            'mode': output['mode'],
            'confidence': float(output['confidence'].mean()),
            'attention': attention_weights,
            'explanation': dashboard_state['model'].explain_decision(output)
        })

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/graph/data')
def get_graph_data():
    """Get graph data for visualization"""
    if dashboard_state['graph'] is None:
        # Return empty graph with message instead of 404
        return jsonify({
            'nodes': [],
            'edges': [],
            'stats': {},
            'message': 'No graph loaded. Upload data or load a model to see graph visualization.'
        }), 200

    try:
        # Extract graph data
        nodes = []
        edges = []

        graph = dashboard_state['graph']

        # Sample nodes (limit to prevent overwhelming frontend)
        max_nodes = 200
        for i, (token, node_id) in enumerate(list(graph.vocab.items())[:max_nodes]):
            nodes.append({
                'id': node_id,
                'label': token,
                'frequency': graph.node_features[node_id].frequency,
                'group': 0  # Could use clustering
            })

        # Sample edges
        for edge in graph.edge_list[:1000]:
            if edge.source < max_nodes and edge.target < max_nodes:
                edges.append({
                    'source': edge.source,
                    'target': edge.target,
                    'weight': float(edge.weight),
                    'pmi': float(edge.pmi_score)
                })

        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'stats': graph.get_statistics()
        })

    except Exception as e:
        logger.error(f"Error getting graph data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/hardware/stats')
def get_hardware_stats():
    """Get hardware statistics"""
    if dashboard_state['hardware_profiler'] is None:
        return jsonify({'error': 'Profiler not initialized'}), 500

    stats = dashboard_state['hardware_profiler'].get_stats()
    return jsonify(stats)


@app.route('/api/metrics/history')
def get_metrics_history():
    """Get training metrics history"""
    if dashboard_state['metrics_monitor'] is None:
        return jsonify({'error': 'Monitor not initialized'}), 500

    history = dashboard_state['metrics_monitor'].get_history()
    return jsonify(history)


# ============================================================================
# SOCKETIO EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")


@socketio.on('request_update')
def handle_update_request():
    """Send live updates to client"""
    if dashboard_state['metrics_monitor']:
        metrics = dashboard_state['metrics_monitor'].get_latest()
        emit('metrics_update', metrics)

    if dashboard_state['hardware_profiler']:
        hw_stats = dashboard_state['hardware_profiler'].get_stats()
        emit('hardware_update', hw_stats)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def run_training(params: Dict[str, Any]):
    """Run training loop with real-time updates"""
    try:
        # Validate dependencies
        if dashboard_state['model'] is None:
            error_msg = "Model not initialized. Please load or initialize a model first."
            logger.error(error_msg)
            socketio.emit('training_error', {'error': error_msg})
            dashboard_state['training_active'] = False
            return

        if dashboard_state['training_manager'] is None:
            error_msg = "Training manager not initialized."
            logger.error(error_msg)
            socketio.emit('training_error', {'error': error_msg})
            dashboard_state['training_active'] = False
            return

        manager = dashboard_state['training_manager']

        # Get training data and graph if available
        training_data = dashboard_state.get('training_data', None)
        linguistic_graph = dashboard_state.get('graph', None)

        if training_data and len(training_data) > 0:
            logger.info(f"Starting training for {params.get('epochs', 10)} epochs on {len(training_data)} text chunks from {dashboard_state.get('data_filename', 'uploaded data')}...")
        else:
            logger.info(f"Starting training for {params.get('epochs', 10)} epochs with synthetic dummy data (no valid file data available)...")

        last_update_time = 0
        update_interval = 10  # Emit training updates every 10 seconds to prevent overflow (was 5)

        # Start training
        for epoch in range(params.get('epochs', 10)):
            if not dashboard_state['training_active']:
                logger.info("Training stopped by user")
                break

            # Training step with real data
            metrics = manager.train_epoch(
                model=dashboard_state['model'],
                epoch=epoch,
                training_data=training_data,
                linguistic_graph=linguistic_graph
            )

            current_time = time.time()

            # Rate limit emissions - only emit if enough time has passed
            if (current_time - last_update_time) >= update_interval:
                # Emit updates
                socketio.emit('training_update', {
                    'epoch': epoch,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                })

                # Hebbian traces (less frequently - every 20 epochs instead of 10)
                if epoch % 20 == 0:
                    traces = dashboard_state['hebbian_monitor'].get_trace_summary()
                    socketio.emit('hebbian_update', traces)

                last_update_time = current_time

                # Longer delay to prevent packet buildup
                socketio.sleep(1)  # Increased from 0.5

        dashboard_state['training_active'] = False
        logger.info("Training completed successfully")
        socketio.emit('training_complete', {'status': 'success'})

    except Exception as e:
        logger.error(f"Training error: {e}")
        dashboard_state['training_active'] = False
        socketio.emit('training_error', {'error': str(e)})


# ============================================================================
# BACKGROUND MONITORING
# ============================================================================

def background_monitor():
    """Background thread for continuous monitoring"""
    last_emit_time = 0
    emit_interval = 10  # Emit every 10 seconds to prevent overflow (was 5)

    while True:
        try:
            current_time = time.time()

            # Rate limit emissions to prevent packet overflow
            if dashboard_state['training_active'] and (current_time - last_emit_time) >= emit_interval:
                # Update hardware stats
                if dashboard_state['hardware_profiler']:
                    hw_stats = dashboard_state['hardware_profiler'].get_stats()
                    socketio.emit('hardware_update', hw_stats)
                    last_emit_time = current_time

            time.sleep(3)  # Check every 3 seconds (was 2)

        except Exception as e:
            logger.error(f"Background monitor error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    # Initialize dashboard
    initialize_dashboard()

    # Start background monitor
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()

    # Print startup info
    print("\n" + "=" * 80)
    print("  NCGN COGNITIVE COCKPIT")
    print("  Real-time Monitoring & Training Dashboard")
    print("=" * 80)
    print("\n✓ Dashboard initialized")
    print(f"✓ Navigate to: http://localhost:5000")
    print(f"✓ Hardware: {dashboard_state['hardware_profiler'].get_gpu_name()}")
    print("\n" + "=" * 80 + "\n")

    # Run Flask app
    # Note: For production deployment, use a production WSGI server like gunicorn
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()

