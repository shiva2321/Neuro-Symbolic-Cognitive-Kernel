"""
Simple Usage Example
Demonstrates basic usage of the system after training.
"""

from core.central_controller import CentralController


def main():
    print("="*70)
    print("SIMPLE USAGE EXAMPLE")
    print("="*70)

    # Initialize controller
    print("\n1. Initializing controller...")
    controller = CentralController()

    # Load pre-trained models
    print("2. Loading pre-trained models...")
    try:
        controller.load_system("saved_models")
        print("   ✓ Models loaded successfully")
    except Exception as e:
        print(f"   ✗ Could not load models: {e}")
        print("   Please run main.py first to train the system")
        return

    # Test queries
    print("\n3. Testing queries...")
    print("-"*70)

    queries = [
        "Write a paragraph about technology",
        "What is 25 + 17?",
        "Create a function to calculate factorial",
        "Solve the equation x + 5 = 10",
        "Generate a short poem",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] Query: {query}")
        result = controller.process(query)

        if result['success']:
            print(f"    Module: {result.get('module', 'unknown')}")
            print(f"    Response: {result['response'][:100]}...")

            # Show routing info
            if 'routing_info' in result:
                scores = result['routing_info']['all_scores']
                print(f"    Confidence Scores: {scores}")
        else:
            print(f"    Error: {result.get('error', 'Unknown')}")

    # System statistics
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)

    stats = controller.get_system_statistics()
    print(f"\nTotal Modules: {stats['num_modules']}")

    for module_name, module_stats in stats['modules'].items():
        if 'graph_stats' in module_stats:
            gs = module_stats['graph_stats']
            print(f"\n{module_name}:")
            print(f"  - Nodes: {gs['num_nodes']}")
            print(f"  - Edges: {gs['num_edges']}")
            print(f"  - Avg Weight: {gs['avg_edge_weight']:.4f}")

    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()

