"""
Automatic PyTorch Geometric installer
Fixes the DGL corruption issue by installing PyG
"""
import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("=" * 70)
    print("PyTorch Geometric Installer")
    print("=" * 70)

    # Check PyTorch
    print("\nChecking PyTorch installation...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA {torch.version.cuda}")
        else:
            print("⚠ CPU only (no CUDA)")
    except ImportError:
        print("✗ PyTorch not found!")
        print("Please install PyTorch first: pip install torch")
        return 1

    # Check if PyG is already installed
    print("\nChecking PyTorch Geometric...")
    try:
        import torch_geometric
        print(f"✓ PyG {torch_geometric.__version__} already installed")
        print("\nTesting import...")
        from utils.graph_backend import get_available_backends
        backends = get_available_backends()
        print(f"✓ Available backends: {backends}")
        print("\n✅ All set! PyTorch Geometric is working.")
        return 0
    except ImportError:
        print("⚠ PyTorch Geometric not installed")
    except Exception as e:
        print(f"⚠ PyG check failed: {e}")

    # Install PyG
    print("\n" + "=" * 70)
    print("Installing PyTorch Geometric")
    print("=" * 70)

    # Use simple pip install (works for most cases)
    cmd = "pip install torch-geometric torch-scatter torch-sparse torch-cluster"

    if not run_command(cmd, "Installing PyTorch Geometric"):
        print("\n❌ Installation failed!")
        print("\nTry manual installation:")
        print(f"  {cmd}")
        return 1

    # Verify installation
    print("\n" + "=" * 70)
    print("Verifying Installation")
    print("=" * 70)

    try:
        import torch_geometric
        print(f"\n✓ PyG {torch_geometric.__version__} installed successfully!")

        # Test backend
        from utils.graph_backend import get_available_backends
        backends = get_available_backends()
        print(f"✓ Available backends: {backends}")

        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print("\nPyTorch Geometric is now installed and working.")
        print("You can now run your application:")
        print("  python ncgn_dashboard.py")
        return 0

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        print("\nPyG may be installed but there might be other issues.")
        print("Try running: python validate_backend_migration.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())

