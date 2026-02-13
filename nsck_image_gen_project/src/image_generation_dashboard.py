"""
NSCK Image Generation Dashboard
================================
Interactive web interface for the NSCK VSA-based image generation system.

NO NEURAL NETWORKS - Pure VSA/hypervector based image generation.

Features:
- Text prompt input for image generation
- Real-time image display
- Model training interface
- Download generated images
- View training statistics
"""

import os
import sys

# Add parent directory paths to access NSCK core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo'))
sys.path.insert(0, os.path.join(parent_dir, 'nsck-demo/python'))

import io
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image as PILImage

from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS

from image_generator import ImageGenerator, GenerationConfig
from python.core.memory.semantic_memory import SemanticMemory


# Global generator instance
_generator: Optional[ImageGenerator] = None
_generation_history = []


def get_generator() -> ImageGenerator:
    """Get or create the image generator."""
    global _generator
    if _generator is None:
        config = GenerationConfig(
            image_size=(32, 32),
            is_color=True,
            smoothing_factor=0.3,
            contrast_boost=1.1
        )
        _generator = ImageGenerator(
            semantic_memory=SemanticMemory(),
            config=config
        )
    return _generator


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    if image.ndim == 3:
        pil_img = PILImage.fromarray(image, mode='RGB')
    else:
        pil_img = PILImage.fromarray(image, mode='L')
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


# Flask app setup
app = Flask(__name__)
CORS(app)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NSCK Image Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .badge {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-left: 10px;
            font-weight: bold;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .panel h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            color: #555;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border 0.3s;
        }
        
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
        }
        
        button:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .image-display {
            text-align: center;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .image-display img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            image-rendering: pixelated;
        }
        
        .placeholder {
            color: #999;
            font-size: 1.2em;
        }
        
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-box {
            background: #f0f4ff;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .history {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .history-item {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .history-item:hover {
            background: #f0f4ff;
            transform: translateX(5px);
        }
        
        .history-prompt {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .history-time {
            font-size: 0.85em;
            color: #999;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-info {
            background: #e3f2fd;
            color: #1976d2;
            border-left: 4px solid #1976d2;
        }
        
        .alert-success {
            background: #e8f5e9;
            color: #388e3c;
            border-left: 4px solid #388e3c;
        }
        
        .alert-warning {
            background: #fff3e0;
            color: #f57c00;
            border-left: 4px solid #f57c00;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .concept-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .concept-tag {
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            h1 { font-size: 1.8em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎨 NSCK Image Generator <span class="badge">VSA-Based</span></h1>
            <p class="subtitle">Generate images from text using Vector Symbolic Architecture (No Neural Networks)</p>
        </header>
        
        <div class="main-grid">
            <div class="panel">
                <h2>Generate Image</h2>
                
                <div class="input-group">
                    <label for="prompt">Enter your prompt:</label>
                    <textarea id="prompt" placeholder="e.g., red cat with blue eyes, flying airplane, green tree"></textarea>
                </div>
                
                <button onclick="generateImage()" id="generateBtn">
                    🎨 Generate Image
                </button>
                
                <div id="status" style="margin-top: 15px;"></div>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value" id="genCount">0</div>
                        <div class="stat-label">Images Generated</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="conceptCount">0</div>
                        <div class="stat-label">Learned Concepts</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>Generated Image</h2>
                <div class="image-display" id="imageDisplay">
                    <div class="placeholder">✨ Your generated image will appear here</div>
                </div>
                <button onclick="downloadImage()" id="downloadBtn" style="margin-top: 15px; display: none;">
                    💾 Download Image
                </button>
            </div>
        </div>
        
        <div class="panel">
            <h2>Training</h2>
            <div class="alert alert-info">
                <strong>ℹ️ Training Info:</strong> The generator learns associations between text concepts and visual features. 
                Train it with example images to improve generation quality.
            </div>
            
            <div class="input-group">
                <label>Training Dataset:</label>
                <select id="dataset" style="width: 100%; padding: 12px; border-radius: 8px; border: 2px solid #e0e0e0;">
                    <option value="synthetic">Synthetic Data (Quick Test)</option>
                    <option value="cifar10">CIFAR-10 (Real Images)</option>
                </select>
            </div>
            
            <div class="input-group">
                <label>Number of Examples:</label>
                <input type="number" id="numExamples" value="100" min="10" max="10000" 
                       style="width: 100%; padding: 12px; border-radius: 8px; border: 2px solid #e0e0e0;">
            </div>
            
            <button onclick="trainModel()" id="trainBtn">
                🚀 Start Training
            </button>
            
            <div id="trainStatus" style="margin-top: 15px;"></div>
        </div>
        
        <div class="panel">
            <h2>Generation History</h2>
            <div class="history" id="history">
                <div class="placeholder">No generations yet. Try creating an image!</div>
            </div>
        </div>
    </div>
    
    <script>
        let currentImage = null;
        let currentPrompt = null;
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                document.getElementById('genCount').textContent = data.generation_count;
                document.getElementById('conceptCount').textContent = data.learned_concepts;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }
        
        async function generateImage() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                showStatus('Please enter a prompt', 'warning');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.innerHTML = '⏳ Generating... <span class="loading"></span>';
            
            showStatus('Generating image...', 'info');
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentImage = data.image_base64;
                    currentPrompt = prompt;
                    
                    document.getElementById('imageDisplay').innerHTML = 
                        `<img src="${data.image_base64}" alt="Generated Image" style="width: 512px; height: 512px;">`;
                    document.getElementById('downloadBtn').style.display = 'block';
                    
                    showStatus('✅ Image generated successfully!', 'success');
                    addToHistory(prompt, data.image_base64);
                    loadStats();
                } else {
                    showStatus('❌ Error: ' + data.error, 'warning');
                }
            } catch (error) {
                showStatus('❌ Error generating image: ' + error, 'warning');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🎨 Generate Image';
            }
        }
        
        async function trainModel() {
            const dataset = document.getElementById('dataset').value;
            const numExamples = parseInt(document.getElementById('numExamples').value);
            
            const btn = document.getElementById('trainBtn');
            btn.disabled = true;
            btn.innerHTML = '⏳ Training... <span class="loading"></span>';
            
            showTrainStatus('Starting training...', 'info');
            
            try {
                const response = await fetch('/api/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        dataset: dataset,
                        num_examples: numExamples
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showTrainStatus(`✅ Training complete! Learned ${data.concepts_learned} concepts in ${data.training_time.toFixed(2)}s`, 'success');
                    loadStats();
                } else {
                    showTrainStatus('❌ Error: ' + data.error, 'warning');
                }
            } catch (error) {
                showTrainStatus('❌ Error training: ' + error, 'warning');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🚀 Start Training';
            }
        }
        
        function downloadImage() {
            if (!currentImage || !currentPrompt) return;
            
            const link = document.createElement('a');
            link.href = currentImage;
            link.download = `nsck_${currentPrompt.replace(/\\s+/g, '_')}.png`;
            link.click();
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
        }
        
        function showTrainStatus(message, type) {
            const status = document.getElementById('trainStatus');
            status.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
        }
        
        function addToHistory(prompt, image) {
            const history = document.getElementById('history');
            if (history.querySelector('.placeholder')) {
                history.innerHTML = '';
            }
            
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-prompt">${prompt}</div>
                <div class="history-time">${new Date().toLocaleString()}</div>
            `;
            item.onclick = () => {
                document.getElementById('imageDisplay').innerHTML = 
                    `<img src="${image}" alt="Generated Image" style="width: 512px; height: 512px;">`;
                currentImage = image;
                currentPrompt = prompt;
                document.getElementById('downloadBtn').style.display = 'block';
            };
            
            history.insertBefore(item, history.firstChild);
        }
        
        // Load stats on page load
        loadStats();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate an image from a text prompt."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        # Generate image
        generator = get_generator()
        image = generator.generate(prompt)
        
        # Convert to base64
        image_base64 = image_to_base64(image)
        
        # Add to history
        _generation_history.append({
            'prompt': prompt,
            'image_base64': image_base64,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'image_base64': image_base64,
            'prompt': prompt
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/train', methods=['POST'])
def api_train():
    """Train the generator on a dataset."""
    try:
        data = request.get_json()
        dataset = data.get('dataset', 'synthetic')
        num_examples = data.get('num_examples', 100)
        
        generator = get_generator()
        
        # Prepare training data
        if dataset == 'synthetic':
            # Use synthetic data
            from train_image_generation import ImageGenerationTrainer
            trainer = ImageGenerationTrainer()
            train_data = trainer._generate_synthetic_data(num_examples)
        else:
            # Try to load from HuggingFace
            from train_image_generation import ImageGenerationTrainer
            trainer = ImageGenerationTrainer()
            train_data = trainer.prepare_training_data(
                dataset_name=dataset,
                max_examples=num_examples
            )
        
        # Train
        import time
        start_time = time.time()
        generator.train_from_examples(train_data, max_examples=num_examples)
        training_time = time.time() - start_time
        
        stats = generator.get_statistics()
        
        return jsonify({
            'success': True,
            'concepts_learned': stats['learned_concepts'],
            'training_time': training_time,
            'examples_processed': len(train_data)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get generator statistics."""
    try:
        generator = get_generator()
        stats = generator.get_statistics()
        
        return jsonify({
            'success': True,
            'generation_count': stats['generation_count'],
            'learned_concepts': stats['learned_concepts'],
            'concept_list': stats['concept_list']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/history', methods=['GET'])
def api_history():
    """Get generation history."""
    try:
        return jsonify({
            'success': True,
            'history': _generation_history
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def main():
    """Run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NSCK Image Generation Dashboard")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5556, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NSCK Image Generation Dashboard")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
