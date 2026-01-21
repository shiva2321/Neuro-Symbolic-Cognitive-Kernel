"""
NCGN Brain Dashboard - Web-based Real-time Visualization

Flask + SocketIO server providing:
- Live graph visualization with D3.js
- Real-time state updates via WebSocket
- Interactive controls (inject energy, step, run)
- Metrics panel and history timeline
- Phase 3: Thought Cloud, Surprise Meter, Staging Area, Dialogue State
- User Interaction: File Upload, Query, Teach Knowledge

Usage:
    python -m ui.brain_dashboard
    # Or:
    from ui.brain_dashboard import create_app
    app = create_app(memory, engine, controller)
    app.run(port=5000)
"""

import json
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try to import Flask dependencies
try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Warning: Flask/Flask-SocketIO not installed. Run: pip install flask flask-socketio")

from .graph_visualizer import GraphVisualizer, GraphState


@dataclass
class DashboardConfig:
    """Configuration for the brain dashboard."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    update_interval_ms: int = 100  # WebSocket update interval
    auto_run: bool = False
    auto_run_ticks_per_second: int = 10


class BrainDashboard:
    """
    Web-based dashboard for NCGN visualization and control.
    
    Provides real-time visualization via WebSocket updates
    and interactive controls for the cognitive system.
    """
    
    def __init__(
        self,
        memory,          # GraphMemory
        engine,          # System1Engine
        controller,      # System2Controller
        config: Optional[DashboardConfig] = None
    ):
        if not HAS_FLASK:
            raise RuntimeError("Flask and Flask-SocketIO required. Install with: pip install flask flask-socketio")
        
        self.memory = memory
        self.engine = engine
        self.controller = controller
        self.config = config or DashboardConfig()
        
        self.visualizer = GraphVisualizer()
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Phase 3: DialogueManager integration
        self.dialogue_manager = None
        try:
            from cortex.dialogue import DialogueManager
            self.dialogue_manager = DialogueManager(
                memory=memory,
                engine=engine,
                controller=controller
            )
        except ImportError:
            pass  # Phase 3 not available
        
        self._setup_routes()
        self._setup_socketio()
        
        self._running = False
        self._auto_thread: Optional[threading.Thread] = None
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/state')
        def get_state():
            """Get current graph state as JSON."""
            state = self.visualizer.capture_state(
                self.memory, self.engine,
                self.engine.firing_set
            )
            return jsonify(self.visualizer.get_d3_compatible_json(state))
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get system metrics."""
            return jsonify({
                "tick": self.engine.current_tick,
                "surprise": self.engine.surprise_level,
                "system2_triggered": self.engine.system2_triggered,
                "paused": self.engine.paused,
                "active_nodes": len(self.memory.get_active_nodes()),
                "total_nodes": self.memory.node_count(),
                "total_edges": self.memory.edge_count(),
                "k_winners": self.engine.k_winners
            })
        
        @self.app.route('/api/inject', methods=['POST'])
        def inject_energy():
            """Inject energy into a node."""
            data = request.json
            node_id = data.get('node_id')
            energy = float(data.get('energy', 1.0))
            
            if node_id:
                self.engine.inject_energy(node_id, energy)
                return jsonify({"success": True, "message": f"Injected {energy} into {node_id}"})
            return jsonify({"success": False, "message": "node_id required"})
        
        @self.app.route('/api/step', methods=['POST'])
        def step_tick():
            """Execute one tick."""
            completed = self.engine.tick()
            state = self.visualizer.capture_state(
                self.memory, self.engine,
                self.engine.firing_set
            )
            
            # Emit update via WebSocket
            self.socketio.emit('state_update', 
                             self.visualizer.get_d3_compatible_json(state))
            
            return jsonify({
                "success": True,
                "completed": completed,
                "tick": self.engine.current_tick
            })
        
        @self.app.route('/api/run', methods=['POST'])
        def run_ticks():
            """Run multiple ticks."""
            data = request.json
            num_ticks = int(data.get('ticks', 10))
            completed = self.engine.run(num_ticks)
            
            state = self.visualizer.capture_state(
                self.memory, self.engine,
                self.engine.firing_set
            )
            self.socketio.emit('state_update',
                             self.visualizer.get_d3_compatible_json(state))
            
            return jsonify({
                "success": True,
                "completed": completed,
                "tick": self.engine.current_tick
            })
        
        @self.app.route('/api/pause', methods=['POST'])
        def toggle_pause():
            """Toggle pause state."""
            if self.engine.paused:
                self.engine.resume()
            else:
                self.engine.pause()
            return jsonify({"paused": self.engine.paused})
        
        @self.app.route('/api/add_node', methods=['POST'])
        def add_node():
            """Add a new node."""
            data = request.json
            node_id = data.get('node_id')
            energy = float(data.get('energy', 0.0))
            threshold = float(data.get('threshold', 0.75))
            
            if node_id:
                self.memory.add_node(node_id, energy=energy, threshold=threshold)
                return jsonify({"success": True})
            return jsonify({"success": False, "message": "node_id required"})
        
        @self.app.route('/api/add_synapse', methods=['POST'])
        def add_synapse():
            """Add a new synapse."""
            data = request.json
            source = data.get('source')
            target = data.get('target')
            weight = float(data.get('weight', 0.5))
            confidence = float(data.get('confidence', 0.5))
            
            if source and target:
                self.memory.add_synapse(source, target, weight=weight, confidence=confidence)
                return jsonify({"success": True})
            return jsonify({"success": False, "message": "source and target required"})
        
        @self.app.route('/api/query', methods=['POST'])
        def query_system():
            """Query the system (for System 2 diagnosis)."""
            data = request.json
            query = data.get('query', '')
            
            # Simple query handling - can be expanded
            result = {
                "query": query,
                "response": "Query processing not yet implemented",
                "diagnosis": None
            }
            
            if self.controller.active_diagnosis:
                result["diagnosis"] = {
                    "type": self.controller.active_diagnosis.diagnosis_type.value,
                    "explanation": self.controller.active_diagnosis.explanation
                }
            
            if self.controller.active_intervention:
                result["query_response"] = self.controller.active_intervention.query
            
            return jsonify(result)
        
        # ============ Phase 3 API Endpoints ============
        
        @self.app.route('/api/thought_cloud')
        def get_thought_cloud():
            """Get active nodes as 'thought cloud' for visualization."""
            # Get all nodes and filter by energy
            thoughts = []
            
            for node_id in self.memory.nodes:
                node = self.memory.get_node(node_id)
                if node and node.energy > 0.1:
                    thoughts.append({
                        "id": node_id,
                        "energy": node.energy,
                        "fired": node_id in self.engine.firing_set,
                        "size": 10 + int(node.energy * 30)  # Size based on energy
                    })
            
            # Sort by energy descending
            thoughts.sort(key=lambda x: x["energy"], reverse=True)
            
            return jsonify({
                "thoughts": thoughts[:20],  # Top 20 active thoughts
                "total_active": len(thoughts)
            })
        
        @self.app.route('/api/surprise_history')
        def get_surprise_history():
            """Get surprise level history for the meter."""
            return jsonify({
                "current": self.engine.surprise_level,
                "threshold": 0.5,  # System 2 trigger threshold
                "system2_triggered": self.engine.system2_triggered
            })
        
        @self.app.route('/api/dialogue_state')
        def get_dialogue_state():
            """Get current dialogue state for status indicator."""
            if not self.dialogue_manager:
                return jsonify({
                    "available": False,
                    "state": "unavailable",
                    "message": "DialogueManager not initialized"
                })
            
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
            if not self.dialogue_manager or not self.dialogue_manager.context.staging_buffer:
                return jsonify({
                    "active": False,
                    "triples": [],
                    "clean": 0,
                    "flagged": 0
                })
            
            buffer = self.dialogue_manager.context.staging_buffer
            
            clean_list = []
            for t in buffer.clean_triples[:10]:
                clean_list.append({
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence
                })
            
            flagged_list = []
            for t, info in buffer.flagged_triples[:5]:
                flagged_list.append({
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "reason": info.reason
                })
            
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
            """Process a teaching statement through DialogueManager."""
            if not self.dialogue_manager:
                return jsonify({"success": False, "message": "DialogueManager not available"})
            
            data = request.json
            statement = data.get('statement', '')
            
            if not statement:
                return jsonify({"success": False, "message": "No statement provided"})
            
            response, new_state = self.dialogue_manager.process_input(statement)
            
            return jsonify({
                "success": True,
                "response": response,
                "state": new_state.value,
                "state_label": self.dialogue_manager.get_state_indicator()
            })
        
        @self.app.route('/api/staging/commit', methods=['POST'])
        def commit_staging():
            """Commit staging buffer to main memory."""
            if not self.dialogue_manager or not self.dialogue_manager.context.staging_buffer:
                return jsonify({"success": False, "message": "No staging buffer"})
            
            buffer = self.dialogue_manager.context.staging_buffer
            result = buffer.merge_to_main(self.memory)
            
            return jsonify({
                "success": True,
                "nodes_added": result.nodes_added,
                "edges_added": result.edges_added,
                "skipped": result.conflicts_skipped
            })
        
        @self.app.route('/api/upload_file', methods=['POST'])
        def upload_file():
            """Upload a knowledge file for ingestion."""
            if not self.dialogue_manager:
                return jsonify({"success": False, "message": "DialogueManager not available"})
            
            data = request.json
            content = data.get('content', '')
            filename = data.get('filename', 'uploaded.txt')
            
            if not content:
                return jsonify({"success": False, "message": "No content provided"})
            
            # Process file through dialogue manager
            response, state = self.dialogue_manager.process_file_content(content, filename)
            
            return jsonify({
                "success": True,
                "response": response,
                "state": state.value
            })
        
        @self.app.route('/api/query_brain', methods=['POST'])
        def query_brain():
            """Query the brain for information."""
            data = request.json
            question = data.get('question', '')
            
            if not question:
                return jsonify({"success": False, "message": "No question provided"})
            
            # Use dialogue manager if available
            if self.dialogue_manager:
                response, state = self.dialogue_manager.process_input(question)
                return jsonify({
                    "success": True,
                    "response": response,
                    "state": state.value
                })
            
            # Fallback: basic graph query
            from core.query import QueryEngine
            query_engine = QueryEngine(self.memory, self.controller)
            result = query_engine.query(question)
            return jsonify({
                "success": True,
                "response": result.get("answer", "No answer found"),
                "confidence": result.get("confidence", 0.0)
            })
        
        @self.app.route('/api/all_nodes')
        def get_all_nodes():
            """Get all nodes with their current state."""
            nodes = []
            for node_id in self.memory.nodes:
                node = self.memory.get_node(node_id)
                if node:
                    nodes.append({
                        "id": node_id,
                        "energy": node.energy,
                        "threshold": node.threshold,
                        "fired": node_id in self.engine.firing_set,
                        "active": node.energy > node.threshold * 0.5
                    })
            
            # Sort by energy descending
            nodes.sort(key=lambda x: x["energy"], reverse=True)
            return jsonify({"nodes": nodes, "total": len(nodes)})
        
        @self.app.route('/api/history')
        def get_history():
            """Get dialogue history."""
            if not self.dialogue_manager:
                return jsonify({"history": []})
            
            history = []
            for entry in self.dialogue_manager.context.history[-20:]:
                history.append({
                    "user": entry.get("user", ""),
                    "response": entry.get("response", ""),
                    "state": entry.get("state", "idle")
                })
            return jsonify({"history": history})
    
    def _setup_socketio(self):
        """Set up WebSocket handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Send initial state on connect."""
            state = self.visualizer.capture_state(
                self.memory, self.engine,
                self.engine.firing_set
            )
            emit('state_update', self.visualizer.get_d3_compatible_json(state))
        
        @self.socketio.on('request_state')
        def handle_request_state():
            """Handle state request."""
            state = self.visualizer.capture_state(
                self.memory, self.engine,
                self.engine.firing_set
            )
            emit('state_update', self.visualizer.get_d3_compatible_json(state))
        
        @self.socketio.on('start_auto')
        def handle_start_auto():
            """Start auto-run mode."""
            self._start_auto_run()
        
        @self.socketio.on('stop_auto')
        def handle_stop_auto():
            """Stop auto-run mode."""
            self._stop_auto_run()
    
    def _start_auto_run(self):
        """Start auto-run in background thread."""
        if self._running:
            return
        
        self._running = True
        
        def auto_run_loop():
            interval = 1.0 / self.config.auto_run_ticks_per_second
            while self._running:
                if not self.engine.paused:
                    self.engine.tick()
                    state = self.visualizer.capture_state(
                        self.memory, self.engine,
                        self.engine.firing_set
                    )
                    self.socketio.emit('state_update',
                                      self.visualizer.get_d3_compatible_json(state))
                time.sleep(interval)
        
        self._auto_thread = threading.Thread(target=auto_run_loop, daemon=True)
        self._auto_thread.start()
    
    def _stop_auto_run(self):
        """Stop auto-run."""
        self._running = False
        if self._auto_thread:
            self._auto_thread.join(timeout=1.0)
            self._auto_thread = None
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """Run the dashboard server."""
        host = host or self.config.host
        port = port or self.config.port
        
        print(f"\n🧠 NCGN Brain Dashboard running at http://{host}:{port}")
        print("   Press Ctrl+C to stop\n")
        
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            debug=self.config.debug,
            allow_unsafe_werkzeug=True
        )


def create_app(memory, engine, controller, config: Optional[DashboardConfig] = None):
    """Factory function to create dashboard app."""
    dashboard = BrainDashboard(memory, engine, controller, config)
    return dashboard


# CLI entry point
def main():
    """Run dashboard with a demo system."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.memory import GraphMemory
    from core.system1 import System1Engine
    from core.system2 import System2Controller
    
    # Create demo system
    memory = GraphMemory()
    engine = System1Engine(memory, k_winners=5)
    controller = System2Controller(memory)
    
    # Add some demo nodes
    memory.add_node("dog", threshold=0.5)
    memory.add_node("cat", threshold=0.5)
    memory.add_node("meat", threshold=0.5)
    memory.add_node("fish", threshold=0.5)
    memory.add_node("eat", threshold=0.5)
    
    memory.add_synapse("dog", "meat", type="eats", weight=0.9, confidence=0.9)
    memory.add_synapse("cat", "fish", type="eats", weight=0.9, confidence=0.9)
    memory.add_synapse("dog", "eat", type="can", weight=0.8, confidence=0.8)
    memory.add_synapse("cat", "eat", type="can", weight=0.8, confidence=0.8)
    
    # Set properties
    controller.set_property("meat", "is_edible", True)
    controller.set_property("fish", "is_edible", True)
    
    # Run dashboard
    dashboard = BrainDashboard(memory, engine, controller)
    dashboard.run()


if __name__ == "__main__":
    main()
