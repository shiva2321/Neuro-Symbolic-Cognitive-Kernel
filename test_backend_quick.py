"""Quick test to verify backend functionality"""
import sys
import os

# Add to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Testing backend imports...")

try:
    from utils.graph_backend import (
        get_available_backends,
        get_default_backend,
        BACKEND_DGL,
        BACKEND_PYG
    )

    print(f"✓ Backend module imported successfully")
    print(f"  DGL available: {BACKEND_DGL}")
    print(f"  PyG available: {BACKEND_PYG}")
    print(f"  Available backends: {get_available_backends()}")
    print(f"  Default backend: {get_default_backend()}")

    # Test LinguisticGraph import
    from ncgn.linguistic_graph import LinguisticGraph
    print(f"✓ LinguisticGraph imported successfully")

    # Try to create a graph
    graph = LinguisticGraph(device='cpu')
    print(f"✓ LinguisticGraph instance created")
    print(f"  Backend: {graph._backend.backend if graph._backend else 'dgl (fallback)'}")

    print("\n✅ All tests passed!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

