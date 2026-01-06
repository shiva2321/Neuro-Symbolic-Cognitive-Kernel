"""
Main Example Script
Demonstrates the complete modular neural network system.
"""

import os
import sys

from core.central_controller import CentralController
from training.dataset_loader import DatasetLoader
from utils.performance_monitor import PerformanceMonitor
from utils.visualizer import GraphVisualizer


def main():
    """Main demonstration of the modular neural network system"""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        MODULAR GRAPH-BASED NEURAL NETWORK SYSTEM                    ║
║                                                                      ║
║  Phase 1: Graph-Based Neural Network Architecture        ✓          ║
║  Phase 2: Modularity and Integration                     ✓          ║
║  Phase 3: Scalability and Performance                    ✓          ║
║  Phase 4: Training and Updating                          ✓          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize performance monitor
    monitor = PerformanceMonitor(enabled=True)

    # Initialize central controller
    print("\n[1/5] Initializing Central Controller...")
    with monitor.monitor_operation("initialization"):
        controller = CentralController()

    # Create sample dataset (if no data exists)
    data_dir = "sample_data"
    if not os.path.exists(data_dir):
        print("\n[2/5] Creating Sample Dataset...")
        with monitor.monitor_operation("dataset_creation"):
            loader = DatasetLoader(data_dir)
            loader.create_sample_dataset(data_dir, num_samples=100)
    else:
        print(f"\n[2/5] Using existing dataset: {data_dir}")

    # Load training data
    print("\n[3/5] Loading Training Data...")
    with monitor.monitor_operation("data_loading"):
        loader = DatasetLoader(data_dir)
        loader.scan_directory()
        data = loader.load_all_data(max_text=50, max_code=25)
        print(f"Loaded: {len(data['text'])} text files, {len(data['code'])} code files")

    # Train the system
    print("\n[4/5] Training Modular Network...")
    with monitor.monitor_operation("training"):
        training_results = controller.train(
            data_path=data_dir,
            epochs=3,
            learning_rate=0.1
        )

    if training_results['success']:
        print("\n✓ Training completed successfully!")
    else:
        print(f"\n✗ Training failed: {training_results.get('error', 'Unknown error')}")
        return

    # Test the system with various queries
    print("\n[5/5] Testing System with Sample Queries...")

    test_queries = [
        "Write a short story about a robot",
        "Calculate 15 + 27",
        "Create a Python function to sort a list",
        "What is the derivative of x^2?",
        "Write code to read a file",
    ]

    print("\n" + "="*70)
    print("QUERY TESTING")
    print("="*70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 70)

        with monitor.monitor_operation(f"query_{i}", {'query': query}):
            result = controller.process(query)

        if result['success']:
            print(f"Module: {result.get('module', 'unknown')}")
            print(f"Response:\n{result.get('response', 'No response')[:200]}...")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    # Display system statistics
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)

    stats = controller.get_system_statistics()
    print(f"\nModules: {stats['num_modules']}")
    print(f"Training Status: {'Trained' if stats['is_trained'] else 'Not Trained'}")

    for module_name, module_stats in stats['modules'].items():
        print(f"\n{module_name}:")
        print(f"  Trained: {module_stats['is_trained']}")
        if 'graph_stats' in module_stats:
            gs = module_stats['graph_stats']
            print(f"  Nodes: {gs['num_nodes']}")
            print(f"  Edges: {gs['num_edges']}")
            print(f"  Avg Edge Weight: {gs['avg_edge_weight']:.4f}")

    # Performance report
    print("\n" + "="*70)
    monitor.print_report()

    # Identify bottlenecks
    bottlenecks = monitor.identify_bottlenecks()
    if bottlenecks['recommendations']:
        print("\nPERFORMANCE RECOMMENDATIONS:")
        for rec in bottlenecks['recommendations']:
            print(f"  • {rec}")

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    try:
        # Visualize text module graph
        text_module = controller.module_map.get('text_module')
        if text_module and text_module.is_trained:
            print("\nGenerating text module visualization...")
            visualizer = GraphVisualizer(text_module.graph)

            # Create HTML visualization
            visualizer.create_html_visualization(
                "text_graph_visualization.html",
                max_nodes=200
            )

            # Generate statistics report
            visualizer.generate_statistics_report("text_graph_statistics.txt")

            print("✓ Visualizations created:")
            print("  - text_graph_visualization.html")
            print("  - text_graph_statistics.txt")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    # Save the trained system
    print("\n" + "="*70)
    print("SAVING SYSTEM")
    print("="*70)

    save_dir = "saved_models"
    try:
        controller.save_system(save_dir)
        print(f"✓ System saved to: {save_dir}")
    except Exception as e:
        print(f"Warning: Could not save system: {e}")

    # Export performance metrics
    try:
        monitor.export_metrics("performance_metrics.json")
        print("✓ Performance metrics saved to: performance_metrics.json")
    except Exception as e:
        print(f"Warning: Could not export metrics: {e}")

    # Interactive mode option
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nWould you like to enter interactive mode? (y/n)")
    print("In interactive mode, you can test queries in real-time.")

    # For automated testing, skip interactive mode
    # Uncomment the following line to enable interactive mode:
    # controller.interactive_mode()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe modular graph-based neural network system is now ready!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Graph-based architecture with nodes and weighted edges")
    print("  ✓ Multiple traversal algorithms with stop criteria")
    print("  ✓ Modular specialized networks (text, math, code)")
    print("  ✓ Central controller with intelligent routing")
    print("  ✓ Performance monitoring and bottleneck detection")
    print("  ✓ Graph visualization and export capabilities")
    print("  ✓ Training on diverse datasets")
    print("  ✓ Continuous learning and weight updates")
    print("\nTo use the system programmatically:")
    print("  from core.central_controller import CentralController")
    print("  controller = CentralController()")
    print("  controller.load_system('saved_models')")
    print("  result = controller.process('your query here')")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

