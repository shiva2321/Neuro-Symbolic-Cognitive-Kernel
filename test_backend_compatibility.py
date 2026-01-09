"""
Test Backend Compatibility
Verifies that the graph backend compatibility layer works correctly.
"""

import sys
import os
import numpy as np

# Ensure we can import from project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_backend_imports():
    """Test that backend modules can be imported"""
    print("=" * 70)
    print("Testing Backend Imports")
    print("=" * 70)

    try:
        from utils.graph_backend import (
            get_available_backends,
            get_default_backend,
            BACKEND_DGL,
            BACKEND_PYG,
            GraphBackend
        )
        print("✓ graph_backend module imported")
        print(f"  DGL available: {BACKEND_DGL}")
        print(f"  PyG available: {BACKEND_PYG}")
        print(f"  Available backends: {get_available_backends()}")
        print(f"  Default backend: {get_default_backend()}")
        return True
    except Exception as e:
        print(f"✗ Failed to import backend module: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linguistic_graph_import():
    """Test that LinguisticGraph can be imported"""
    print("\n" + "=" * 70)
    print("Testing LinguisticGraph Import")
    print("=" * 70)

    try:
        from ncgn.linguistic_graph import LinguisticGraph, GraphBuilder
        print("✓ LinguisticGraph imported")
        print("✓ GraphBuilder imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linguistic_graph_creation():
    """Test creating a LinguisticGraph instance"""
    print("\n" + "=" * 70)
    print("Testing LinguisticGraph Creation")
    print("=" * 70)

    try:
        from ncgn.linguistic_graph import LinguisticGraph

        # Create with auto backend selection
        graph = LinguisticGraph(device='cpu')
        backend_name = graph._backend.backend if graph._backend else 'dgl (fallback)'
        print(f"✓ Created LinguisticGraph with backend: {backend_name}")

        # Add some test nodes
        for word in ['test', 'graph', 'network']:
            embedding = np.random.randn(768).astype(np.float32)
            graph.add_node(word, embedding)

        print(f"✓ Added {len(graph.vocab)} nodes")

        # Add test edges
        graph.add_edge('test', 'graph', co_occurrence=5)
        graph.add_edge('graph', 'network', co_occurrence=3)

        print(f"✓ Added {len(graph.edge_list)} edges")

        # Compute PMI
        graph.compute_pmi_scores()
        print("✓ Computed PMI scores")

        # Build graph
        graph.build_dgl_graph()
        print("✓ Built graph structure")

        # Get statistics
        stats = graph.get_statistics()
        print(f"✓ Graph statistics: {stats['unique_tokens']} nodes, {stats['num_edges']} edges")
        print(f"  Backend: {stats.get('backend', 'unknown')}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that old DGL code still works"""
    print("\n" + "=" * 70)
    print("Testing Backward Compatibility")
    print("=" * 70)

    try:
        from ncgn.linguistic_graph import LinguisticGraph

        # Old-style creation (no backend parameter)
        graph = LinguisticGraph(device='cpu')
        print("✓ Old-style LinguisticGraph creation works")

        # Old-style operations
        embedding = np.random.randn(768).astype(np.float32)
        node_id = graph.add_node('test', embedding)
        print(f"✓ Old-style add_node works (node_id={node_id})")

        graph.add_edge('test', 'test', co_occurrence=1)
        print("✓ Old-style add_edge works")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_available_backends():
    """Test with each available backend"""
    print("\n" + "=" * 70)
    print("Testing All Available Backends")
    print("=" * 70)

    try:
        from utils.graph_backend import get_available_backends
        from ncgn.linguistic_graph import LinguisticGraph

        backends = get_available_backends()

        for backend in backends:
            print(f"\nTesting with {backend.upper()}:")
            graph = LinguisticGraph(device='cpu', backend=backend)

            # Add test data
            embedding = np.random.randn(768).astype(np.float32)
            graph.add_node('test', embedding)
            graph.add_edge('test', 'test', co_occurrence=1)
            graph.compute_pmi_scores()
            graph.build_dgl_graph()

            stats = graph.get_statistics()
            print(f"  ✓ {backend.upper()} works: {stats['unique_tokens']} nodes")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Graph Backend Compatibility Test Suite")
    print("=" * 70)

    tests = [
        ("Backend Imports", test_backend_imports),
        ("LinguisticGraph Import", test_linguistic_graph_import),
        ("LinguisticGraph Creation", test_linguistic_graph_creation),
        ("Backward Compatibility", test_backward_compatibility),
        ("All Backends", test_all_available_backends),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! Backend compatibility verified.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

