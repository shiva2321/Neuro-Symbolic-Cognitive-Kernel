"""
Quick diagnostic and fix for DGL/PyG issue
"""
import sys
import subprocess

print("=" * 70)
print("Diagnosing Graph Backend Issue")
print("=" * 70)

# Check PyTorch
print("\n1. Checking PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda if torch.cuda.is_available() else 'CPU only'}")
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")
    sys.exit(1)

# Check DGL
print("\n2. Checking DGL...")
try:
    import dgl
    print(f"   ✓ DGL {dgl.__version__}")
    dgl_ok = True
except FileNotFoundError as e:
    print(f"   ✗ DGL installation corrupted: {e}")
    print("   → Missing C++ library (graphbolt)")
    dgl_ok = False
except ImportError as e:
    print(f"   ⚠ DGL not installed: {e}")
    dgl_ok = False
except Exception as e:
    print(f"   ✗ DGL error: {e}")
    dgl_ok = False

# Check PyG
print("\n3. Checking PyTorch Geometric...")
try:
    import torch_geometric
    print(f"   ✓ PyG {torch_geometric.__version__}")
    pyg_ok = True
except ImportError as e:
    print(f"   ⚠ PyG not installed")
    pyg_ok = False
except Exception as e:
    print(f"   ✗ PyG error: {e}")
    pyg_ok = False

# Recommendation
print("\n" + "=" * 70)
print("Recommendation")
print("=" * 70)

if not dgl_ok and not pyg_ok:
    print("\n❌ No working graph backend found!")
    print("\nQuick fix - Install PyTorch Geometric:")

    if cuda_version:
        if '11.8' in str(cuda_version):
            cmd = "pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html"
        elif '12.1' in str(cuda_version):
            cmd = "pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html"
        else:
            cmd = "pip install torch-geometric"
    else:
        cmd = "pip install torch-geometric"

    print(f"\n  {cmd}\n")

    # Offer to install
    response = input("Install PyTorch Geometric now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nInstalling PyTorch Geometric...")
        try:
            subprocess.check_call(cmd.split())
            print("\n✓ PyTorch Geometric installed successfully!")
            print("\nPlease run your application again.")
        except Exception as e:
            print(f"\n✗ Installation failed: {e}")
            print(f"\nPlease run manually:\n  {cmd}")
    else:
        print(f"\nPlease install manually:\n  {cmd}")

elif not dgl_ok and pyg_ok:
    print("\n✓ PyTorch Geometric is working!")
    print("  DGL is corrupted but not needed (PyG is preferred)")
    print("\nOptional: Uninstall corrupted DGL:")
    print("  pip uninstall dgl -y")

elif dgl_ok and not pyg_ok:
    print("\n✓ DGL is working!")
    print("\nOptional: Install PyTorch Geometric for better performance:")
    if cuda_version:
        if '11.8' in str(cuda_version):
            print("  pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html")
        elif '12.1' in str(cuda_version):
            print("  pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html")
        else:
            print("  pip install torch-geometric")
    else:
        print("  pip install torch-geometric")

else:
    print("\n✓ Both backends working!")
    print("  System will use PyTorch Geometric (preferred)")

print("\n" + "=" * 70)

