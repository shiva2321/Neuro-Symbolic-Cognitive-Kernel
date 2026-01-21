"""
NCGN Dashboard Runner - Launch Brain Visualization Dashboard

Start the web-based brain visualization dashboard.

Usage:
    python run_dashboard.py                 # Default port 5000
    python run_dashboard.py --port 8080     # Custom port
    python run_dashboard.py --demo          # Load demo system
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_demo_system():
    """Create a demo system with pre-trained knowledge."""
    from core.memory import GraphMemory, EventSchema
    from core.system1 import System1Engine
    from core.system2 import System2Controller
    
    memory = GraphMemory()
    engine = System1Engine(memory, k_winners=10, surprise_threshold=0.3)
    controller = System2Controller(memory)
    
    # Add nodes
    animals = ["dog", "cat", "rabbit", "bird", "cow", "horse"]
    foods = ["meat", "fish", "carrot", "seed", "grass", "hay"]
    properties = ["edible", "inedible", "animate", "inanimate"]
    inedibles = ["metal", "plastic", "rock", "glass"]
    
    for animal in animals:
        memory.add_node(animal, threshold=0.5)
        controller.set_property(animal, "is_animate", True)
    
    for food in foods:
        memory.add_node(food, threshold=0.5, novelty_score=0.1)
        controller.set_property(food, "is_edible", True)
    
    for item in inedibles:
        memory.add_node(item, threshold=0.5, novelty_score=0.1)
        controller.set_property(item, "is_edible", False)
    
    for prop in properties:
        memory.add_node(prop, threshold=0.6)
    
    # Associations
    pairs = [
        ("dog", "meat"), ("cat", "fish"), ("rabbit", "carrot"),
        ("bird", "seed"), ("cow", "grass"), ("horse", "hay")
    ]
    
    for animal, food in pairs:
        memory.add_synapse(animal, food, type="eats", weight=0.9, confidence=0.9)
    
    # Category memberships
    for food in foods:
        memory.add_synapse(food, "edible", type="is", weight=0.85, confidence=0.95)
    
    for item in inedibles:
        memory.add_synapse(item, "inedible", type="is", weight=0.9, confidence=0.95)
    
    for animal in animals:
        memory.add_synapse(animal, "animate", type="is", weight=0.9, confidence=0.95)
    
    # Action nodes
    memory.add_node("eat", threshold=0.5)
    for animal in animals:
        memory.add_synapse(animal, "eat", type="can", weight=0.7, confidence=0.8)
    
    # Schema
    schema = EventSchema.from_dict({
        "id": "schema_eat",
        "action": "eat",
        "confidence": 0.95,
        "constraints": {"target": ["is_edible"]}
    })
    controller.add_schema(schema)
    
    return memory, engine, controller


def main():
    parser = argparse.ArgumentParser(description="Run NCGN Brain Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--demo", action="store_true", help="Use demo system")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   🧠 NCGN Brain Dashboard")
    print("=" * 60)
    
    # Check dependencies
    try:
        from flask import Flask
        from flask_socketio import SocketIO
    except ImportError:
        print("\n❌ Flask dependencies not installed.")
        print("   Install with: pip install flask flask-socketio")
        return 1
    
    # Set up system
    print("\n📦 Setting up NCGN system...")
    memory, engine, controller = setup_demo_system()
    print(f"   Loaded {memory.node_count()} nodes, {memory.edge_count()} edges")
    
    # Create and run dashboard
    from ui.brain_dashboard import BrainDashboard, DashboardConfig
    
    config = DashboardConfig(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    dashboard = BrainDashboard(memory, engine, controller, config)
    
    print(f"\n🌐 Starting dashboard at http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
