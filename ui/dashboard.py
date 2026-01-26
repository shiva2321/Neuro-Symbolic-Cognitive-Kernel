"""
NCGN v7 Dashboard - Web-based Brain Visualization and Control

Flask + SocketIO server with:
- Interactive D3.js brain graph visualization
- File upload for knowledge learning
- Brain persistence (save/load)
- Detailed activity logging
- Game training with observation (Corridor, Snake)
- Real-time state updates via WebSocket
"""

import json
import threading
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import sys
import os

# Ensure we can import ncgn from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask required: pip install flask flask-socketio")

from ncgn.brain import Brain
from ncgn.config import Config
from ncgn.persistence import BrainPersistence
from ncgn.logger import BrainLogger

# Import games
try:
    from ncgn.games.corridor import CorridorGame
    from ncgn.games.snake import SnakeGame
    from ncgn.games.bridge import GameBrainBridge
    HAS_GAMES = True
except ImportError:
    HAS_GAMES = False
    print("Games not available")


BRAINS_DIR = Path("./saved_brains")
LOGS_DIR = Path("./logs")


class V7Dashboard:
    """
    Web-based dashboard for NCGN v7.
    
    Features:
    - Interactive brain graph visualization
    - File upload and knowledge extraction
    - Save/load brain state
    - Game training with observable decision making
    - Detailed activity logging
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        debug: bool = False,
        brains_dir: str = None,
        use_embeddings: bool = False
    ):
        if not HAS_FLASK:
            raise RuntimeError("Flask required: pip install flask flask-socketio")
        
        self.host = host
        self.port = port
        self.debug = debug
        self.brains_dir = Path(brains_dir) if brains_dir else BRAINS_DIR
        self.brains_dir.mkdir(parents=True, exist_ok=True)
        
        # Brain
        self.brain = Brain(config=Config(), use_embeddings=use_embeddings)
        
        # Logger
        self.logger = BrainLogger(enabled=False)
        
        # Game state
        self.game = None
        self.bridge = None
        self.game_type = None
        self.game_episode = 0
        self.game_total_reward = 0.0
        
        # Flask
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Threading
        self._running = False
        self._training = False
        self._auto_thread: Optional[threading.Thread] = None
        self._train_thread: Optional[threading.Thread] = None
        
        self._setup_routes()
        self._setup_socketio()
    
    def _get_full_state(self) -> Dict:
        """Get complete brain state for visualization."""
        stats = self.brain.get_stats()
        active = self.brain.get_top_concepts(k=20)
        firing = self.brain.engine.get_firing_set()
        
        # Nodes with firing state
        nodes = []
        for label in self.brain.topology.registry.all_labels():
            idx = self.brain.topology.registry.get_index(label)
            nodes.append({
                "label": label,
                "energy": float(self.brain.state.activations[idx]),
                "firing": idx in firing
            })
        
        # Edges
        edges = []
        for source in self.brain.topology.registry.all_labels():
            for target in self.brain.topology.get_neighbors(source):
                weight = self.brain.topology.get_edge_weight(source, target)
                edges.append({"source": source, "target": target, "weight": weight})
        
        return {
            "tick": self.brain.engine.tick_count,
            "stats": stats,
            "active": active,
            "nodes": nodes,
            "edges": edges,
            "total_energy": stats['state']['total_energy']
        }
    
    def _setup_routes(self):
        """Setup Flask API routes."""
        
        @self.app.route('/')
        def index():
            # Serve v2 dashboard
            try:
                return render_template('dashboard_v2.html')
            except Exception:
                return render_template('dashboard_v7.html')
        
        # === Brain State ===
        @self.app.route('/api/brain/state')
        def get_brain_state():
            return jsonify(self._get_full_state())
            
        @self.app.route('/api/brain/clear', methods=['POST'])
        def clear():
            self.brain.clear()
            return jsonify({"success": True})
        
        # === Game Integration ===
        
        @self.app.route('/api/game/start', methods=['POST'])
        def start_game():
            if not HAS_GAMES:
                return jsonify({"success": False, "error": "Games module not available"})
            
            data = request.json or {}
            game_type = data.get('game', 'corridor')
            
            # Init Game
            if game_type == 'corridor':
                game = CorridorGame(length=7)
            elif game_type == 'snake':
                game = SnakeGame(width=8, height=8)
            else:
                return jsonify({"success": False, "error": f"Unknown game: {game_type}"})
            
            # Init Bridge
            self.bridge = GameBrainBridge(self.brain, game, game_type)
            self.bridge.setup_concepts()
            
            # Store refs
            self.game = game
            self.game_type = game_type
            
            return jsonify({
                "success": True,
                "game": game_type,
                "render": self.game.render(),
                "episode": 0
            })
        
        @self.app.route('/api/game/step', methods=['POST'])
        def game_step():
            if not self.bridge:
                return jsonify({"success": False, "error": "No game active"})
            
            # Bridge handles Sense -> Think -> Act -> Learn
            result = self.bridge.step()
            
            self._emit_state_update()
            
            return jsonify({
                "success": True,
                **result
            })
        
        @self.app.route('/api/game/episode', methods=['POST'])
        def game_episode():
            if not self.bridge:
                return jsonify({"success": False, "error": "No game active"})
            
            self.game.reset()
            self.bridge.reset_stats()
            
            max_steps = 100
            history = []
            
            for _ in range(max_steps):
                step_res = self.bridge.step()
                history.append(step_res)
                if step_res['done']:
                    break
            
            self._emit_state_update()
            
            return jsonify({
                "success": True,
                "total_reward": self.bridge.total_reward,
                "steps": self.bridge.steps,
                "render": self.game.render()
            })
            
        # === v2.0 Interaction ===
        # === v2.0 Interaction ===
        @self.app.route('/api/file/upload', methods=['POST'])
        def upload_file():
            data = request.json or {}
            filename = data.get('filename')
            content = data.get('content', '')
            
            if not content:
                return jsonify({"success": False, "error": "No content provided"})
            
            if not self.brain.linguist:
                print("WARNING: No LinguisticProcessor available. Cannot extract knowledge.")
                return jsonify({
                    "success": False, 
                    "error": "No LLM configured. Cannot read file.",
                    "concepts_added": 0,
                    "connections_added": 0
                })

            # Process in chunks (paragraphs) to avoid token limits
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            total_added_nodes = 0
            total_added_edges = 0
            
            print(f"Processing file {filename} with {len(paragraphs)} paragraphs...")
            
            for i, para in enumerate(paragraphs):
                try:
                    print(f"  - Extracting from paragraph {i+1}...")
                    updates = self.brain.linguist.extract_triplets(para)
                    if updates:
                        print(f"    Found {len(updates.new_nodes)} nodes, {len(updates.new_edges)} edges.")
                        # Use internal method to apply updates directly
                        results = self.brain._apply_graph_update(updates)
                        
                        # Count stats 
                        total_added_nodes += len(updates.new_nodes)
                        total_added_edges += len(updates.new_edges)
                    else:
                        print("    No triplets found.")
                        
                except Exception as e:
                    print(f"Error processing chunk: {e}")
            
            # Force a save after learning
            # BrainPersistence.save(self.brain, str(self.brains_dir), "autosave_latest")
            
            self._emit_state_update()
            
            return jsonify({
                 "success": True,
                 "concepts_added": total_added_nodes,
                 "connections_added": total_added_edges
            })

        @self.app.route('/api/brain/process', methods=['POST'])
        def process_input():
            """Phase 3/4 integration point"""
            data = request.json or {}
            user_input = data.get('input', '')
            
            if not user_input:
                return jsonify({"error": "No input provided"})
            
            # Call Brain v2 pipeline
            result = self.brain.process_input(user_input, propagation_steps=10)
            
            if self.logger.enabled:
                self.logger.log_query(user_input, str(result))
                
            self._emit_state_update()
            return jsonify(result)

        @self.app.route('/api/brain/inject', methods=['POST'])
        def inject():
            data = request.json or {}
            label = data.get('label')
            energy = float(data.get('energy', 1.0))
            
            if not label:
                return jsonify({"success": False, "error": "label required"})
            
            if not self.brain.has_concept(label):
                self.brain.add_concept(label)
            
            success = self.brain.inject(label, energy)
            if success:
                self.logger.log_injection(label, energy)
            
            return jsonify({"success": success})
        
        @self.app.route('/api/brain/think', methods=['POST'])
        def think():
            data = request.json or {}
            steps = int(data.get('steps', 10))
            
            active = self.brain.think(steps=steps)
            
            if self.logger.enabled:
                self.logger.log_tick(
                    self.brain.engine.tick_count,
                    self.brain.state.get_total_energy(),
                    len(active),
                    len(self.brain.engine.get_firing_set()),
                    active
                )
            
            self._emit_state_update()
            
            return jsonify({"active": active, "tick": self.brain.engine.tick_count})
        
        @self.app.route('/api/brain/learn', methods=['POST'])
        def learn():
            data = request.json or {}
            reward = float(data.get('reward', 0.5))
            
            updates = self.brain.learn(reward=reward)
            self.logger.log_reward(reward, self.brain.modulator.last_rpe if hasattr(self.brain.modulator, 'last_rpe') else 0)
            
            return jsonify({
                "success": True,
                "updates": updates,
                "learner_stats": self.brain.learner.get_stats()
            })
        
        # === Persistence ===
        
        @self.app.route('/api/brain/save', methods=['POST'])
        def save_brain():
            data = request.json or {}
            name = data.get('name', f"brain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            try:
                path = BrainPersistence.save(self.brain, str(self.brains_dir), name)
                return jsonify({"success": True, "path": path, "name": name})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/api/brain/load', methods=['POST'])
        def load_brain():
            data = request.json or {}
            name = data.get('name')
            if not name:
                return jsonify({"success": False, "error": "name required"})
            try:
                self.brain = BrainPersistence.load(str(self.brains_dir), name, use_embeddings=False)
                return jsonify({"success": True, "stats": self.brain.get_stats()})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/api/brain/list')
        def list_brains():
            return jsonify({"brains": BrainPersistence.list_saved_brains(str(self.brains_dir))})
        
        # === Logging ===
        
        @self.app.route('/api/log/enable', methods=['POST'])
        def enable_logging():
            self.logger.enable()
            return jsonify({"success": True})
        
        @self.app.route('/api/log/disable', methods=['POST'])
        def disable_logging():
            self.logger.disable()
            return jsonify({"success": True})
        
        @self.app.route('/api/log/export', methods=['POST'])
        def export_logs():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = self.logger.export_session(str(LOGS_DIR / filename))
            return jsonify({"success": True, "path": path})
    
    def _emit_state_update(self):
        """Emit current state via WebSocket."""
        state = self._get_full_state()
        self.socketio.emit('brain_state', state)
    
    def _setup_socketio(self):
        """Setup WebSocket handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            self._emit_state_update()
        
        @self.socketio.on('request_state')
        def handle_request_state():
            self._emit_state_update()
        
        @self.socketio.on('start_auto_think')
        def handle_start_auto():
            self._start_auto_think()
        
        @self.socketio.on('stop_auto_think')
        def handle_stop_auto():
            self._stop_auto_think()
    
    def _start_auto_think(self):
        """Start auto-thinking."""
        if self._running:
            return
        self._running = True
        
        def auto_loop():
            while self._running:
                self.brain.think(steps=1)
                self._emit_state_update()
                time.sleep(0.1)
        
        self._auto_thread = threading.Thread(target=auto_loop, daemon=True)
        self._auto_thread.start()
    
    def _stop_auto_think(self):
        """Stop auto-thinking."""
        self._running = False
        if self._auto_thread:
            self._auto_thread.join(timeout=1.0)
            self._auto_thread = None
    
    def run(self):
        """Run the dashboard."""
        print(f"\n🧠 NCGN v2.0 Dashboard")
        print(f"   URL: http://{self.host}:{self.port}")
        print("   Ready.")
        
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            allow_unsafe_werkzeug=True
        )


def main():
    """Run the v7 dashboard."""
    dashboard = V7Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
