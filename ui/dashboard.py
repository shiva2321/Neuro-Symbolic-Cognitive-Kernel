"""
NCGN Unified Dashboard - Web-based Visualization and Control

Flask + SocketIO server providing:
- Live brain graph visualization with D3.js
- Real-time learning during game training
- Game environments (Corridor, Snake)
- Learning metrics (reward, dopamine, weights)
- Interactive controls (train, step, play, dialogue)

Usage:
    python -m ui.dashboard
    # Opens browser to http://localhost:5000
"""

import json
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Warning: Flask/Flask-SocketIO not installed. Run: pip install flask flask-socketio")

from core.runner import NCGNAgent, TrainingConfig
from core.memory import ClusterType
from core.dialogue import DialogueManager, DialogueState
from core.query_engine import QueryEngine
from core.staging import StagingBuffer


@dataclass  
class DashboardConfig:
    """Configuration for the v6.0 dashboard."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    update_interval_ms: int = 100
    ticks_per_second: int = 10


class V6Dashboard:
    """
    Web-based dashboard for NCGN v6.0 visualization.
    
    Displays:
    - Brain graph with node energies and synapse weights
    - Learning metrics (reward history, dopamine, trace activity)
    - Game state (Corridor/Snake)
    - Real-time updates during training
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        if not HAS_FLASK:
            raise RuntimeError("Flask required. Install: pip install flask flask-socketio")
        
        self.config = config or DashboardConfig()
        
        # Create agent
        self.agent = NCGNAgent(TrainingConfig(
            learning_rate=0.15,
            trace_decay=0.9,
            use_episodic=True
        ))
        
        # Set up default game (corridor)
        self.current_game_type = "corridor"
        self.agent.setup_corridor_game(length=7)
        self.agent.create_sensory_motor_network()
        
        # Flask app
        self.app = Flask(__name__, 
                        template_folder='../ui/templates',
                        static_folder='../ui/static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # State
        self._running = False
        self._training = False
        self._auto_thread: Optional[threading.Thread] = None
        
        # Dialogue & Knowledge (Phase 3 Integration)
        self.query_engine = QueryEngine(self.agent.memory)
        self.dialogue_manager = DialogueManager(
            memory=self.agent.memory,
            engine=self.agent.engine,
            controller=self.agent.system2,
            query_engine=self.query_engine
        )
        
        # Set up callbacks
        self.agent.on_step = self._on_step
        self.agent.on_episode_end = self._on_episode_end
        
        self._setup_routes()
        self._setup_socketio()
    
    def _on_step(self, data: dict):
        """Callback when agent takes a step."""
        state = self.agent.get_state_for_dashboard()
        state["step_info"] = data
        state["game_render"] = self.agent.game.render() if self.agent.game else ""
        self.socketio.emit('state_update', state)
    
    def _on_episode_end(self, data: dict):
        """Callback when episode ends."""
        self.socketio.emit('episode_end', data)
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('v6_dashboard.html')
        
        @self.app.route('/api/state')
        def get_state():
            """Get current brain state."""
            return jsonify(self.agent.get_state_for_dashboard())
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get training metrics."""
            return jsonify({
                "episode": len(self.agent.episode_rewards),
                "rewards": self.agent.episode_rewards[-20:],
                "lengths": self.agent.episode_lengths[-20:],
                "avg_reward": (
                    sum(self.agent.episode_rewards[-10:]) / max(1, len(self.agent.episode_rewards[-10:]))
                    if self.agent.episode_rewards else 0
                ),
                "success_rate": (
                    self.agent.game.get_stats()["success_rate"] 
                    if self.agent.game else 0
                ),
                "modulator": self.agent.modulator.get_stats(),
                "learner": self.agent.learner.get_stats(),
                "temperature": self.agent.engine.temperature,
                "total_energy": self.agent.memory.get_total_energy()
            })
        
        @self.app.route('/api/game')
        def get_game():
            """Get game state."""
            if not self.agent.game:
                return jsonify({"active": False})
            
            return jsonify({
                "active": True,
                "render": self.agent.game.render(),
                "position": self.agent.game.position,
                "steps": self.agent.game.steps,
                "stats": self.agent.game.get_stats()
            })
        
        @self.app.route('/api/step', methods=['POST'])
        def step():
            """Execute one step."""
            if not self.agent.game:
                return jsonify({"error": "No game active"})
            
            obs, reward, done, info = self.agent.step(render=False)
            
            return jsonify({
                "reward": reward,
                "done": done,
                "action": info["action"],
                "valence": info["valence"],
                "game": self.agent.game.render(),
                "state": self.agent.get_state_for_dashboard()
            })
        
        @self.app.route('/api/episode', methods=['POST'])
        def run_episode():
            """Run one episode."""
            if not self.agent.game:
                return jsonify({"error": "No game active"})
            
            total_reward = self.agent.run_episode(render=False)
            
            return jsonify({
                "reward": total_reward,
                "episode": len(self.agent.episode_rewards),
                "stats": self.agent.game.get_stats(),
                "state": self.agent.get_state_for_dashboard()
            })
        
        @self.app.route('/api/train', methods=['POST'])
        def train():
            """Train for N episodes."""
            data = request.json or {}
            episodes = int(data.get('episodes', 10))
            
            # Run in background
            def train_background():
                self._training = True
                self.agent.train(episodes=episodes, render=False)
                self._training = False
                self.socketio.emit('training_complete', {
                    "episodes": episodes,
                    "final_reward": self.agent.episode_rewards[-1] if self.agent.episode_rewards else 0
                })
            
            if not self._training:
                thread = threading.Thread(target=train_background, daemon=True)
                thread.start()
                return jsonify({"started": True, "episodes": episodes})
            else:
                return jsonify({"started": False, "reason": "Training already in progress"})
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset():
            """Reset the game."""
            if self.agent.game:
                self.agent.game.reset()
            return jsonify({"success": True})
        
        @self.app.route('/api/switch_game', methods=['POST'])
        def switch_game():
            """Switch between game types (corridor, snake)."""
            data = request.json or {}
            game_type = data.get('game_type', 'corridor')
            
            # Reset agent for new game
            self.agent = NCGNAgent(TrainingConfig(
                learning_rate=0.15,
                trace_decay=0.9,
                use_episodic=True
            ))
            
            if game_type == 'snake':
                self.agent.setup_snake_game(width=8, height=8)
                self.agent.create_snake_network()
                self.current_game_type = 'snake'
            else:
                self.agent.setup_corridor_game(length=7)
                self.agent.create_sensory_motor_network()
                self.current_game_type = 'corridor'
            
            # Re-register callbacks
            self.agent.on_step = self._on_step
            self.agent.on_episode_end = self._on_episode_end
            
            return jsonify({
                "success": True,
                "game_type": self.current_game_type,
                "render": self.agent.game.render() if self.agent.game else "",
                "state": self.agent.get_state_for_dashboard()
            })
        
        @self.app.route('/api/inject', methods=['POST'])
        def inject():
            """Inject energy into a node."""
            data = request.json or {}
            node_id = data.get('node_id')
            energy = float(data.get('energy', 1.0))
            
            if node_id:
                cluster = ClusterType.SENSORY
                if 'motor' in node_id or 'action' in node_id:
                    cluster = ClusterType.MOTOR
                elif 'hidden' in node_id:
                    cluster = ClusterType.HIDDEN
                
                self.agent.engine.inject_energy(node_id, energy, cluster)
                return jsonify({"success": True})
            return jsonify({"success": False, "error": "node_id required"})
        
        @self.app.route('/api/tick', methods=['POST'])
        def tick():
            """Run one physics tick."""
            self.agent.engine.tick()
            return jsonify(self.agent.get_state_for_dashboard())
        
        @self.app.route('/api/nodes')
        def get_nodes():
            """Get all nodes."""
            nodes = []
            for node_id, node in self.agent.memory.nodes.items():
                nodes.append({
                    "id": node_id,
                    "energy": node.energy,
                    "cluster": node.cluster.value if hasattr(node, 'cluster') else "hidden",
                    "threshold": node.threshold,
                    "fired": node_id in self.agent.engine.firing_set
                })
            nodes.sort(key=lambda x: x["energy"], reverse=True)
            return jsonify({"nodes": nodes})
        
        @self.app.route('/api/synapses')
        def get_synapses():
            """Get all synapses with learning info."""
            synapses = []
            for source_id in self.agent.memory.nodes:
                for syn in self.agent.memory.get_outgoing(source_id):
                    synapses.append({
                        "source": source_id,
                        "target": syn.target_id,
                        "weight": syn.weight,
                        "trace": syn.trace,
                        "stability": syn.stability,
                        "confidence": syn.confidence
                    })
            synapses.sort(key=lambda x: x["weight"], reverse=True)
            return jsonify({"synapses": synapses})
        
        @self.app.route('/api/manual_action', methods=['POST'])
        def manual_action():
            """Execute a manual action (for human play mode)."""
            if not self.agent.game:
                return jsonify({"error": "No game active"})
            
            data = request.json or {}
            action = data.get('action', 'action_move_right')
            learn = data.get('learn', True)
            
            # Get sensory before action
            obs = self.agent.game.get_sensory()
            sensory = self.agent.encoder.encode(obs)
            
            # Inject sensory
            for node_id, energy in sensory.items():
                self.agent.engine.inject_energy(node_id, energy, ClusterType.SENSORY)
            
            # Mark traces for learning
            hidden_id = f"hidden_{action}"
            for src, syn in self.agent.memory.get_incoming(hidden_id):
                if self.agent.memory.get_node(src):
                    syn.set_eligible()
                    self.agent.memory.mark_synapse_traced(src, hidden_id)
            
            # Execute action
            new_obs, reward, done = self.agent.game.step(action)
            
            # Apply learning if enabled
            if learn:
                valence = self.agent.valence.estimate(reward, done, survived=not done or reward > 0)
                dopamine = self.agent.modulator.calculate_signal(valence)
                self.agent.learner.apply_reward(dopamine)
            
            # Track episode reward
            if not hasattr(self, '_manual_episode_reward'):
                self._manual_episode_reward = 0.0
            self._manual_episode_reward += reward
            
            result = {
                "reward": reward,
                "done": done,
                "action": action,
                "game": self.agent.game.render(),
                "state": self.agent.get_state_for_dashboard(),
                "total_reward": self._manual_episode_reward
            }
            
            if done:
                self._manual_episode_reward = 0.0
            
            return jsonify(result)
        
        @self.app.route('/api/test', methods=['POST'])
        def test():
            """Run test episodes without learning."""
            data = request.json or {}
            episodes = int(data.get('episodes', 10))
            
            # Save current learning state
            original_use_episodic = self.agent.config.use_episodic
            original_lr = self.agent.learner.learning_rate
            
            # Disable learning
            self.agent.config.use_episodic = False
            self.agent.learner.learning_rate = 0.0
            
            # Run test episodes
            test_rewards = []
            test_steps = []
            successes = 0
            
            for _ in range(episodes):
                self.agent.game.reset()
                total_reward = 0.0
                steps = 0
                
                for step in range(self.agent.config.max_steps_per_episode):
                    _, reward, done, _ = self.agent.step(render=False)
                    total_reward += reward
                    steps += 1
                    if done:
                        break
                
                test_rewards.append(total_reward)
                test_steps.append(steps)
                if total_reward > 0:
                    successes += 1
            
            # Restore learning
            self.agent.config.use_episodic = original_use_episodic
            self.agent.learner.learning_rate = original_lr
            
            return jsonify({
                "episodes": episodes,
                "success_rate": successes / episodes,
                "avg_reward": sum(test_rewards) / len(test_rewards),
                "avg_steps": sum(test_steps) / len(test_steps),
                "rewards": test_rewards
            })
        
        # ============ Dialogue & Knowledge Endpoints ============
        
        @self.app.route('/api/dialogue_state')
        def get_dialogue_state():
            """Get dialogue state."""
            ctx = self.dialogue_manager.context
            return jsonify({
                "available": True,
                "state": ctx.state.value,
                "state_label": self.dialogue_manager.get_state_indicator(),
                "clarifications": ctx.clarifications_asked,
                "exceptions_learned": ctx.exceptions_learned,
                "pending_triple": str(ctx.pending_triple) if ctx.pending_triple else None,
                "has_staging": ctx.staging_buffer is not None
            })

        @self.app.route('/api/staging')
        def get_staging():
            """Get staging buffer content."""
            if not self.dialogue_manager.context.staging_buffer:
                return jsonify({"active": False, "triples": [], "clean": 0, "flagged": 0})
            
            buffer = self.dialogue_manager.context.staging_buffer
            
            clean_list = [{
                "subject": t.subject, "predicate": t.predicate, "object": t.object, "confidence": t.confidence
            } for t in buffer.clean_triples[:10]]
            
            flagged_list = [{
                "subject": t.subject, "predicate": t.predicate, "object": t.object, "reason": info.reason
            } for t, info in buffer.flagged_triples[:5]]
            
            return jsonify({
                "active": True,
                "source": buffer.source_file,
                "total": len(buffer.triples),
                "clean_count": len(buffer.clean_triples),
                "flagged_count": len(buffer.flagged_triples),
                "clean": clean_list,
                "flagged": flagged_list
            })

        @self.app.route('/api/teach', methods=['POST'])
        def teach():
            """Process teaching statement."""
            data = request.json
            statement = data.get('statement', '')
            if not statement: return jsonify({"success": False, "message": "No statement"})
            
            response, new_state = self.dialogue_manager.process_input(statement)
            return jsonify({
                "success": True,
                "response": response,
                "state": new_state.value,
                "state_label": self.dialogue_manager.get_state_indicator()
            })

        @self.app.route('/api/staging/commit', methods=['POST'])
        def commit_staging():
            """Commit staged knowledge."""
            if not self.dialogue_manager.context.staging_buffer:
                return jsonify({"success": False, "message": "No staging buffer"})
            
            buffer = self.dialogue_manager.context.staging_buffer
            result = buffer.merge_to_main(self.agent.memory)
            return jsonify({
                "success": True,
                "nodes_added": result.nodes_added,
                "edges_added": result.edges_added,
                "skipped": result.conflicts_skipped
            })

        @self.app.route('/api/upload_file', methods=['POST'])
        def upload_file():
            """Upload knowledge file."""
            data = request.json
            content = data.get('content', '')
            filename = data.get('filename', 'uploaded.txt')
            if not content: return jsonify({"success": False, "message": "No content"})
            
            response, state = self.dialogue_manager.process_file_content(content, filename)
            return jsonify({"success": True, "response": response, "state": state.value})

        @self.app.route('/api/query_brain', methods=['POST'])
        def query_brain():
            """Query the brain."""
            data = request.json
            question = data.get('question', '')
            if not question: return jsonify({"success": False})
            
            if self.query_engine:
                result = self.query_engine.ask(question)
                return jsonify({
                    "success": True, 
                    "answer": result.answer, 
                    "confidence": result.confidence,
                    "paths": result.paths
                })
            else:
                return jsonify({"success": False, "message": "Query engine unavailable"})

        @self.app.route('/api/query', methods=['POST'])
        def query_system():
            """System 2 diagnostic query."""
            data = request.json
            query = data.get('query', '')
            
            result = {"query": query, "response": "Not implemented", "diagnosis": None}
            
            # Using agent.system2
            if self.agent.system2.active_diagnosis:
                result["diagnosis"] = {
                    "type": self.agent.system2.active_diagnosis.diagnosis_type.value,
                    "explanation": self.agent.system2.active_diagnosis.explanation
                }
            return jsonify(result)
    
    def _setup_socketio(self):
        """Set up WebSocket handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('state_update', self.agent.get_state_for_dashboard())
        
        @self.socketio.on('request_state')
        def handle_request_state():
            emit('state_update', self.agent.get_state_for_dashboard())
        
        @self.socketio.on('start_auto')
        def handle_start_auto():
            self._start_auto_run()
        
        @self.socketio.on('stop_auto')  
        def handle_stop_auto():
            self._stop_auto_run()
    
    def _start_auto_run(self):
        """Start auto-run training."""
        if self._running:
            return
        
        self._running = True
        
        def auto_loop():
            while self._running:
                if self.agent.game:
                    self.agent.run_episode(render=False)
                time.sleep(0.1)
        
        self._auto_thread = threading.Thread(target=auto_loop, daemon=True)
        self._auto_thread.start()
    
    def _stop_auto_run(self):
        """Stop auto-run."""
        self._running = False
        if self._auto_thread:
            self._auto_thread.join(timeout=1.0)
            self._auto_thread = None
    
    def run(self):
        """Run the dashboard server."""
        print(f"\n🧠 NCGN v6.0 Dashboard at http://{self.config.host}:{self.config.port}")
        print("   Press Ctrl+C to stop\n")
        
        self.socketio.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            allow_unsafe_werkzeug=True
        )


def main():
    """Run the v6.0 dashboard."""
    dashboard = V6Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
