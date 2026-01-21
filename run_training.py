"""
NCGN Training Runner - Unified Training Entry Point

Train the NCGN system with curriculum-based learning.

Usage:
    python run_training.py                          # Train with default curriculum
    python run_training.py --epochs 100             # Custom epochs
    python run_training.py --curriculum acquired    # Use specific curriculum
    python run_training.py --visualize              # Show progress visualization
    python run_training.py --acquire                # Acquire data from web first
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory import GraphMemory, EventSchema
from core.system1 import System1Engine
from core.system2 import System2Controller
from core.trainer import TrainingSession, Curriculum, create_default_curriculum


def setup_base_system():
    """Create the base NCGN system."""
    memory = GraphMemory()
    engine = System1Engine(memory, k_winners=10, surprise_threshold=0.3)
    controller = System2Controller(memory)
    
    # Add eat schema
    schema = EventSchema.from_dict({
        "id": "schema_eat",
        "action": "eat",
        "confidence": 0.95,
        "roles": {"agent": "animate_object", "target": "edible_object"},
        "constraints": {"target": ["is_edible"]}
    })
    controller.add_schema(schema)
    
    return memory, engine, controller


def load_curriculum(curriculum_name: str) -> Curriculum:
    """Load curriculum from training directory."""
    curricula_dir = os.path.join(os.path.dirname(__file__), "training", "curricula")
    
    if curriculum_name == "default":
        return create_default_curriculum()
    
    return Curriculum.load_from_directory(curricula_dir, curriculum_name)


def acquire_training_data(concepts: list):
    """Acquire training data from external sources."""
    try:
        from core.data_acquisition import DataAcquisitionPipeline
        
        print("\n📥 Acquiring training data...")
        pipeline = DataAcquisitionPipeline()
        
        # Try ConceptNet API
        count = pipeline.acquire_from_web(concepts, sources=["conceptnet"])
        print(f"   Acquired {count} facts from ConceptNet")
        
        # Clean and generate curriculum
        cleaned = pipeline.clean_data()
        print(f"   Cleaned to {cleaned} usable facts")
        
        output_path = pipeline.generate_curriculum("acquired_knowledge.json")
        print(f"   Saved curriculum to {output_path}")
        
        return pipeline.get_stats()
    except Exception as e:
        print(f"   Warning: Data acquisition failed: {e}")
        return None


def print_training_summary(metrics_list):
    """Print a summary of training results."""
    print("\n" + "=" * 60)
    print("   TRAINING SUMMARY")
    print("=" * 60)
    
    total_epochs = sum(m.total_epochs for m in metrics_list)
    total_examples = sum(m.examples_trained for m in metrics_list)
    total_updates = sum(m.synapse_updates for m in metrics_list)
    total_time = sum(m.training_time_seconds for m in metrics_list)
    avg_accuracy = sum(m.accuracy for m in metrics_list) / len(metrics_list) if metrics_list else 0
    
    print(f"\n   Lessons trained:    {len(metrics_list)}")
    print(f"   Total epochs:       {total_epochs}")
    print(f"   Examples trained:   {total_examples}")
    print(f"   Synapse updates:    {total_updates}")
    print(f"   Average accuracy:   {avg_accuracy:.1%}")
    print(f"   Total time:         {total_time:.2f}s")
    
    print("\n   Per-lesson breakdown:")
    for i, m in enumerate(metrics_list):
        status = "✓" if m.accuracy >= 0.95 else "✗"
        print(f"   {status} Lesson {i+1}: {m.accuracy:.1%} accuracy in {m.convergence_epoch} epochs")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train NCGN system")
    parser.add_argument("--curriculum", default="default", help="Curriculum name to use")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per lesson")
    parser.add_argument("--visualize", action="store_true", help="Open visualization dashboard")
    parser.add_argument("--acquire", action="store_true", help="Acquire data from web first")
    parser.add_argument("--concepts", nargs="+", default=["dog", "cat", "food", "animal"],
                       help="Concepts to acquire data for")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   🧠 NCGN Training System")
    print("=" * 60)
    
    # Acquire data if requested
    if args.acquire:
        acquire_training_data(args.concepts)
    
    # Set up system
    print("\n📦 Setting up NCGN system...")
    memory, engine, controller = setup_base_system()
    session = TrainingSession(memory, engine, controller)
    session.setup_knowledge_base()
    
    print(f"   Nodes: {memory.node_count()}, Edges: {memory.edge_count()}")
    
    # Load curriculum
    print(f"\n📚 Loading curriculum: {args.curriculum}")
    curriculum = load_curriculum(args.curriculum)
    print(f"   Lessons: {len(curriculum.lessons)}")
    
    # Progress callback
    def on_progress(epoch, accuracy):
        if not args.quiet:
            bar = "█" * int(accuracy * 20) + "░" * (20 - int(accuracy * 20))
            print(f"\r   [{bar}] {accuracy:.1%}", end="", flush=True)
    
    session.on_progress = on_progress
    
    # Train
    print("\n🎯 Starting training...")
    start_time = time.time()
    
    metrics = session.train_until_mastery(
        curriculum,
        max_total_epochs=args.epochs * len(curriculum.lessons),
        verbose=not args.quiet
    )
    
    elapsed = time.time() - start_time
    
    # Summary
    print_training_summary(metrics)
    
    print(f"✓ Training completed in {elapsed:.1f}s")
    print(f"   Final graph: {memory.node_count()} nodes, {memory.edge_count()} edges")
    
    # Launch dashboard if requested
    if args.visualize:
        print("\n🌐 Launching visualization dashboard...")
        try:
            from ui.brain_dashboard import BrainDashboard
            dashboard = BrainDashboard(memory, engine, controller)
            dashboard.run()
        except ImportError as e:
            print(f"   Warning: Could not start dashboard: {e}")
            print("   Install Flask with: pip install flask flask-socketio")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
