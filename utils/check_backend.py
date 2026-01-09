"""
Utility script to check and configure graph backend.

This script helps users understand which backends are available
and test the functionality of each.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.graph_backend import (
    get_available_backends,
    get_default_backend,
    BACKEND_DGL,
    BACKEND_PYG,
    GraphBackend
)


def check_backends():
    """Check which graph backends are available."""
    print("=" * 60)
    print("Graph Backend Status")
    print("=" * 60)

    print(f"\nDGL Available: {'✓' if BACKEND_DGL else '✗'}")
    if BACKEND_DGL:
        import dgl
        print(f"  Version: {dgl.__version__}")

    print(f"\nPyTorch Geometric Available: {'✓' if BACKEND_PYG else '✗'}")
    if BACKEND_PYG:
        import torch_geometric
        print(f"  Version: {torch_geometric.__version__}")

    print(f"\nAvailable backends: {', '.join(get_available_backends())}")
    print(f"Default backend: {get_default_backend()}")

    print("\n" + "=" * 60)


def test_backend(backend_name: str):
    """Test a specific backend."""
    print(f"\nTesting {backend_name.upper()} backend...")
    print("-" * 60)

    try:
        # Create a simple graph
        backend = GraphBackend(backend=backend_name)

        # Create a simple triangle graph
        src = [0, 1, 2, 0]
        dst = [1, 2, 0, 2]
        num_nodes = 3

        graph = backend.create_graph(src, dst, num_nodes)
        print(f"✓ Graph created: {backend.num_nodes()} nodes, {backend.num_edges()} edges")

        # Add node features
        node_feat = torch.randn(num_nodes, 10)
        backend.set_node_features('feat', node_feat)
        retrieved_feat = backend.get_node_features('feat')
        print(f"✓ Node features: shape {retrieved_feat.shape}")

        # Add edge features
        edge_feat = torch.randn(len(src), 5)
        backend.set_edge_features('weight', edge_feat)
        retrieved_edge = backend.get_edge_features('weight')
        print(f"✓ Edge features: shape {retrieved_edge.shape}")

        # Test edges
        edges = backend.edges()
        print(f"✓ Edge list: {len(edges[0])} edges")

        # Test successors
        succ = backend.successors(0)
        print(f"✓ Successors of node 0: {succ.tolist()}")

        # Test device movement
        if torch.cuda.is_available():
            backend.to('cuda')
            print(f"✓ Moved to CUDA")
            backend.to('cpu')
            print(f"✓ Moved back to CPU")

        # Test adjacency matrix
        adj = backend.get_adj_matrix()
        print(f"✓ Adjacency matrix: shape {adj.shape}")

        # Test NetworkX conversion
        nx_graph = backend.to_networkx()
        print(f"✓ NetworkX conversion: {len(nx_graph.nodes())} nodes")

        print(f"\n{backend_name.upper()} backend test: PASSED ✓")
        return True

    except Exception as e:
        print(f"\n{backend_name.upper()} backend test: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linguistic_graph():
    """Test LinguisticGraph with available backends."""
    print("\n" + "=" * 60)
    print("Testing LinguisticGraph with available backends")
    print("=" * 60)

    from ncgn.linguistic_graph import LinguisticGraph
    import numpy as np

    for backend in get_available_backends():
        print(f"\nTesting LinguisticGraph with {backend.upper()} backend...")
        print("-" * 60)

        try:
            # Create graph
            graph = LinguisticGraph(device='cpu', backend=backend)

            # Add some nodes
            for i, word in enumerate(['hello', 'world', 'test']):
                embedding = np.random.randn(768).astype(np.float32)
                graph.add_node(word, embedding)

            # Add edges
            graph.add_edge('hello', 'world', co_occurrence=5)
            graph.add_edge('world', 'test', co_occurrence=3)
            graph.add_edge('hello', 'test', co_occurrence=2)

            # Compute PMI
            graph.compute_pmi_scores()

            # Build graph
            graph.build_dgl_graph()

            # Compute encodings
            graph.compute_structural_encodings(k=2)

            # Get statistics
            stats = graph.get_statistics()
            print(f"✓ Graph statistics: {stats}")

            print(f"\nLinguisticGraph with {backend.upper()}: PASSED ✓")

        except Exception as e:
            print(f"\nLinguisticGraph with {backend.upper()}: FAILED ✗")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("Graph Backend Compatibility Checker")
    print("=" * 60)

    # Check available backends
    check_backends()

    # Test each available backend
    print("\n" + "=" * 60)
    print("Testing Individual Backends")
    print("=" * 60)

    backends = get_available_backends()
    results = {}

    for backend in backends:
        results[backend] = test_backend(backend)

    # Test LinguisticGraph
    test_linguistic_graph()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for backend, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{backend.upper()}: {status}")

    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)

    if BACKEND_PYG:
        print("✓ PyTorch Geometric (PyG) is available and recommended.")
        print("  - Native PyTorch integration")
        print("  - Active development and community")
        print("  - Extensive layer library")
    elif BACKEND_DGL:
        print("✓ DGL is available (legacy support).")
        print("  - Stable and well-tested")
        print("  - Good for existing projects")
    else:
        print("✗ No graph backend available!")
        print("  Install PyTorch Geometric: pip install torch-geometric")
        print("  Or install DGL: pip install dgl")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

