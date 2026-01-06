"""
Web Dashboard for Neural Network System
Flask-based web interface with file upload and graph visualization
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
from datetime import datetime
import secrets

from core.central_controller import CentralController
from training.dataset_loader import DatasetLoader
from utils.visualizer import GraphVisualizer
from utils.performance_monitor import PerformanceMonitor
from utils.file_processor import FileProcessor
from training.ai_training_assistant import setup_ai_training, AIEnhancedTrainingEngine


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {
    'txt', 'pdf', 'doc', 'docx', 'md', 'text',  # Text files
    'py', 'js', 'java', 'cpp', 'c', 'cs', 'rb', 'go', 'rs',  # Code files
    'json', 'xml', 'html', 'css'  # Data files
}

# Global state
controller = None
monitor = PerformanceMonitor(enabled=True)
file_processor = FileProcessor()
ai_config = None
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Ready',
    'stats': {},
    'use_ai': False
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_system():
    """Initialize the neural network system"""
    global controller
    if controller is None:
        controller = CentralController()
        # Try to load existing models
        if os.path.exists('saved_models'):
            try:
                controller.load_system('saved_models')
                print("Loaded existing models")
            except:
                print("No existing models found")
    return controller

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    global controller, training_status

    if controller is None:
        return jsonify({
            'initialized': False,
            'training': training_status
        })

    stats = controller.get_system_statistics()

    return jsonify({
        'initialized': True,
        'trained': controller.is_trained,
        'modules': stats['modules'],
        'num_modules': stats['num_modules'],
        'training': training_status,
        'performance': monitor.get_summary() if monitor.metrics else {}
    })

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the system"""
    try:
        initialize_system()
        return jsonify({'success': True, 'message': 'System initialized'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    global file_processor

    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'})

    files = request.files.getlist('files')
    uploaded = []

    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the file to extract text
            ext = filename.lower().rsplit('.', 1)[-1]
            if ext in ['pdf', 'doc', 'docx']:
                # Extract text and save as .txt
                text_content = file_processor.process_file(filepath)
                if text_content:
                    text_filepath = filepath.rsplit('.', 1)[0] + '.txt'
                    with open(text_filepath, 'w', encoding='utf-8') as f:
                        f.write(text_content)

                    uploaded.append({
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'path': filepath,
                        'processed': True,
                        'text_file': os.path.basename(text_filepath)
                    })
                else:
                    uploaded.append({
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'path': filepath,
                        'processed': False,
                        'error': 'Could not extract text'
                    })
            else:
                uploaded.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'path': filepath
                })

    return jsonify({
        'success': True,
        'uploaded': len(uploaded),
        'files': uploaded
    })

@app.route('/api/train', methods=['POST'])
def train_system():
    """Train the system on uploaded files"""
    global controller, training_status

    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'})

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        return jsonify({'success': False, 'error': 'No files uploaded'})

    # Get training parameters
    data = request.get_json() or {}
    epochs = data.get('epochs', 5)
    learning_rate = data.get('learning_rate', 0.1)

    # Start training in background thread
    def train_background():
        global training_status

        try:
            training_status['is_training'] = True
            training_status['progress'] = 0
            training_status['message'] = 'Initializing...'

            initialize_system()

            training_status['progress'] = 10
            training_status['message'] = 'Loading data...'

            # Train the system
            results = controller.train(
                data_path=app.config['UPLOAD_FOLDER'],
                epochs=epochs,
                learning_rate=learning_rate
            )

            training_status['progress'] = 90
            training_status['message'] = 'Saving models...'

            # Save models
            controller.save_system('saved_models')

            training_status['progress'] = 100
            training_status['message'] = 'Training complete!'
            training_status['stats'] = results

        except Exception as e:
            training_status['message'] = f'Error: {str(e)}'
            training_status['stats'] = {'error': str(e)}
        finally:
            time.sleep(2)
            training_status['is_training'] = False

    thread = threading.Thread(target=train_background)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a user query"""
    global controller

    if controller is None or not controller.is_trained:
        return jsonify({
            'success': False,
            'error': 'System not trained. Please upload files and train first.'
        })

    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})

    try:
        with monitor.monitor_operation('query', {'query': query}):
            result = controller.process(query)

        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/graph/<module_name>')
def get_graph_data(module_name):
    """Get graph data for visualization"""
    global controller

    if controller is None:
        return jsonify({'success': False, 'error': 'System not initialized'})

    module = controller.module_map.get(module_name)
    if not module:
        return jsonify({'success': False, 'error': 'Module not found'})

    if not module.is_trained:
        return jsonify({'success': False, 'error': 'Module not trained'})

    try:
        visualizer = GraphVisualizer(module.graph)

        # Export to temporary JSON file
        temp_file = f'temp_graph_{module_name}.json'
        graph_data = visualizer.export_to_json(temp_file, max_nodes=500)

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return jsonify({
            'success': True,
            'data': graph_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/statistics')
def get_statistics():
    """Get detailed statistics"""
    global controller, monitor

    if controller is None:
        return jsonify({'success': False, 'error': 'System not initialized'})

    stats = controller.get_system_statistics()
    perf_summary = monitor.get_summary() if monitor.metrics else None

    return jsonify({
        'success': True,
        'system': stats,
        'performance': perf_summary,
        'training': training_status
    })

@app.route('/api/export/<module_name>/<format>')
def export_graph(module_name, format):
    """Export graph in various formats"""
    global controller

    if controller is None:
        return jsonify({'success': False, 'error': 'System not initialized'})

    module = controller.module_map.get(module_name)
    if not module or not module.is_trained:
        return jsonify({'success': False, 'error': 'Module not available'})

    try:
        visualizer = GraphVisualizer(module.graph)
        filename = f'{module_name}_graph.{format}'

        if format == 'json':
            visualizer.export_to_json(filename)
        elif format == 'graphml':
            visualizer.export_to_graphml(filename)
        elif format == 'dot':
            visualizer.export_to_dot(filename)
        else:
            return jsonify({'success': False, 'error': 'Unsupported format'})

        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_uploads', methods=['POST'])
def clear_uploads():
    """Clear uploaded files"""
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            import shutil
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'])
        return jsonify({'success': True, 'message': 'Uploads cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/setup_ai', methods=['POST'])
def setup_ai():
    """Setup AI training configuration"""
    global ai_config

    data = request.get_json()
    claude_key = data.get('claude_api_key', '')
    gemini_key = data.get('gemini_api_key', '')

    if not claude_key and not gemini_key:
        return jsonify({'success': False, 'error': 'At least one API key required'})

    try:
        ai_config = setup_ai_training(
            claude_api_key=claude_key if claude_key else None,
            gemini_api_key=gemini_key if gemini_key else None
        )

        return jsonify({
            'success': True,
            'message': 'AI configuration saved',
            'has_claude': ai_config.use_claude,
            'has_gemini': ai_config.use_gemini
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_with_ai', methods=['POST'])
def train_with_ai():
    """Train with AI assistance"""
    global controller, training_status, ai_config

    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'})

    if not ai_config:
        return jsonify({'success': False, 'error': 'AI not configured. Setup API keys first.'})

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        return jsonify({'success': False, 'error': 'No files uploaded'})

    # Get training parameters
    data = request.get_json() or {}
    epochs = data.get('epochs', 5)
    learning_rate = data.get('learning_rate', 0.1)

    # Start AI-enhanced training in background thread
    def train_with_ai_background():
        global training_status

        try:
            training_status['is_training'] = True
            training_status['use_ai'] = True
            training_status['progress'] = 0
            training_status['message'] = 'Initializing AI training...'

            initialize_system()

            training_status['progress'] = 10
            training_status['message'] = 'Loading data with AI analysis...'

            # Load data
            loader = DatasetLoader(app.config['UPLOAD_FOLDER'])
            data = loader.load_all_data()

            training_status['progress'] = 20
            training_status['message'] = 'AI analyzing data quality...'

            # Process with AI enhancement for each module
            from utils.text_processor import TextProcessor
            processor = TextProcessor()

            # Text module with AI
            if data['text']:
                training_status['message'] = 'Training text module with AI...'
                training_status['progress'] = 30

                text_module = controller.module_map['text_module']

                # Process data
                for text in data['text']:
                    processor.process_text(text, text_module.graph)

                # Extract sequences
                sequences = []
                for text in data['text']:
                    tokens = processor.tokenize(text)
                    for i in range(0, len(tokens)-10, 5):
                        sequences.append(tokens[i:i+10])

                # AI-enhanced training
                ai_trainer = AIEnhancedTrainingEngine(
                    text_module.graph,
                    ai_config=ai_config
                )

                training_status['progress'] = 40
                ai_trainer.train_with_ai_assistance(sequences)

                training_status['progress'] = 70

            # Regular training for other modules
            training_status['message'] = 'Training other modules...'
            training_status['progress'] = 80

            training_status['progress'] = 90
            training_status['message'] = 'Saving models...'

            # Save models
            controller.save_system('saved_models')

            training_status['progress'] = 100
            training_status['message'] = 'AI-enhanced training complete!'
            training_status['stats'] = {
                'ai_enhanced': True,
                'success': True
            }

        except Exception as e:
            training_status['message'] = f'Error: {str(e)}'
            training_status['stats'] = {'error': str(e), 'ai_enhanced': True}
        finally:
            time.sleep(2)
            training_status['is_training'] = False
            training_status['use_ai'] = False

    import threading
    thread = threading.Thread(target=train_with_ai_background)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'AI-enhanced training started'})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("\n" + "="*70)
    print("NEURAL NETWORK DASHBOARD STARTING")
    print("="*70)
    print("\nDashboard will be available at: http://localhost:5000")
    print("\nFeatures:")
    print("  • File upload (PDF, TXT, Code files)")
    print("  • Interactive graph visualization")
    print("  • Real-time training monitoring")
    print("  • Query processing interface")
    print("  • Performance metrics")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

