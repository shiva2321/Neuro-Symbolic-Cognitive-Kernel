"""
Complete fix for DGL/PyG backend issue
Handles Python version mismatches and corrupted DGL installations
"""
import sys
import subprocess
import os

print("=" * 70)
print("Graph Backend Fix")
print("=" * 70)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test imports
print("\n" + "=" * 70)
print("Testing Imports")
print("=" * 70)

# Test PyTorch
print("\n1. PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   Location: {torch.__file__}")
    cuda = torch.cuda.is_available()
    print(f"   CUDA: {'Available' if cuda else 'Not available'}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test DGL
print("\n2. DGL...")
dgl_ok = False
try:
    import dgl
    print(f"   ✓ DGL {dgl.__version__}")
    print(f"   Location: {dgl.__file__}")
    dgl_ok = True
except FileNotFoundError as e:
    print(f"   ✗ DGL corrupted (missing C++ library)")
    print(f"   Error: {e}")
except ImportError:
    print(f"   ⚠ DGL not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test PyG
print("\n3. PyTorch Geometric...")
pyg_ok = False
try:
    import torch_geometric
    print(f"   ✓ PyG {torch_geometric.__version__}")
    print(f"   Location: {torch_geometric.__file__}")
    pyg_ok = True
except ImportError:
    print(f"   ⚠ PyG not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test our backend
print("\n4. Graph Backend Compatibility Layer...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.graph_backend import get_available_backends, get_default_backend
    backends = get_available_backends()
    default = get_default_backend()
    print(f"   ✓ Backend layer working")
    print(f"   Available backends: {backends}")
    print(f"   Default backend: {default}")
    backend_ok = True
except Exception as e:
    print(f"   ✗ Error: {e}")
    backend_ok = False
    import traceback
    traceback.print_exc()

# Test LinguisticGraph
print("\n5. LinguisticGraph...")
if backend_ok:
    try:
        from ncgn.linguistic_graph import LinguisticGraph
        print(f"   ✓ LinguisticGraph import successful")

        # Try to create a simple graph
        import numpy as np
        graph = LinguisticGraph(device='cpu')
        embedding = np.random.randn(768).astype(np.float32)
        graph.add_node('test', embedding)
        print(f"   ✓ LinguisticGraph creation works")

        stats = graph.get_statistics()
        print(f"   Backend in use: {stats.get('backend', 'unknown')}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⊗ Skipped (backend layer failed)")

# Summary and fix
print("\n" + "=" * 70)
print("Summary & Fix")
print("=" * 70)

if backend_ok:
    print("\n✅ SUCCESS: Graph backend is working!")
    print(f"\nYou can now run your application.")
    sys.exit(0)

elif not dgl_ok and not pyg_ok:
    print("\n❌ PROBLEM: No graph backend available")
    print("\nSOLUTION: Install PyTorch Geometric")
    print(f"\nRun this command:")
    print(f"  {sys.executable} -m pip install torch-geometric torch-scatter torch-sparse")

    response = input("\nInstall now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nInstalling...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch-geometric", "torch-scatter", "torch-sparse"
            ])
            print("\n✓ Installation complete!")
            print("\nPlease run your application again.")
        except Exception as e:
            print(f"\n✗ Installation failed: {e}")
            print(f"\nPlease install manually:")
            print(f"  {sys.executable} -m pip install torch-geometric torch-scatter torch-sparse")
    sys.exit(1)

elif not dgl_ok and pyg_ok:
    print("\n⚠ PROBLEM: DGL is corrupted but PyG is installed")
    print("\nSOLUTION: The backend should work with PyG")
    print("\nIf you're still seeing errors:")
    print(f"  1. Make sure you're using the correct Python:")
    print(f"     {sys.executable}")
    print(f"  2. Or uninstall corrupted DGL:")
    print(f"     {sys.executable} -m pip uninstall dgl -y")
    sys.exit(1)

else:
    print("\n❌ PROBLEM: Unexpected backend configuration")
    print(f"   DGL OK: {dgl_ok}")
    print(f"   PyG OK: {pyg_ok}")
    print(f"   Backend OK: {backend_ok}")
    print("\nPlease report this issue.")
    sys.exit(1)

