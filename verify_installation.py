"""
NCGN Quick Start Script
Run this to verify installation and get started with NCGN.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(text):
    """Print section header"""
    print(f"\n>>> {text}")
    print("-" * 80)


def check_python_version():
    """Check Python version"""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ is required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_dependencies():
    """Check if required packages are installed"""
    print_section("Checking Dependencies")

    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'numpy': 'NumPy',
        'networkx': 'NetworkX'
    }

    # Graph backend (at least one required)
    graph_backends = {
        'torch_geometric': 'PyTorch Geometric (Recommended)',
        'dgl': 'Deep Graph Library (Legacy)'
    }

    missing = []

    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)

    # Check graph backends (at least one required)
    print("\nGraph Backends (at least one required):")
    backend_available = False
    for package, name in graph_backends.items():
        try:
            __import__(package)
            print(f"✅ {name}")
            backend_available = True
        except ImportError:
            print(f"⚠️  {name} - NOT INSTALLED")

    if not backend_available:
        print("\n❌ No graph backend available!")
        print("Install at least one:")
        print("  pip install torch-geometric  (Recommended)")
        print("  pip install dgl  (Legacy)")
        missing.append('graph_backend')

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False

    return True


def check_cuda():
    """Check CUDA availability"""
    print_section("Checking CUDA/GPU")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("  For GPU acceleration, install CUDA-enabled PyTorch")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_ncgn():
    """Check if NCGN module is importable"""
    print_section("Checking NCGN Module")

    try:
        import ncgn
        print(f"✅ NCGN module found")
        print(f"  Version: {ncgn.__version__}")

        # Try importing key components
        from ncgn import LinguisticGraph, GraphBuilder
        from ncgn import LIFNeuron, SpikingLayer
        from ncgn import DualSystemArchitecture

        print("✅ All core components importable")
        return True

    except ImportError as e:
        print(f"❌ NCGN module not found: {e}")
        print("\nMake sure you're in the project directory:")
        print(f"  cd {Path(__file__).parent.parent}")
        return False


def print_quick_start_guide():
    """Print quick start guide"""
    print_header("Quick Start Guide")

    print("1. Run the demo:")
    print("   python scripts/ncgn_demo.py")
    print()

    print("2. Run specific phase:")
    print("   python scripts/ncgn_demo.py --phase 1  # Linguistic Graph")
    print("   python scripts/ncgn_demo.py --phase 2  # Spiking Networks")
    print("   python scripts/ncgn_demo.py --phase 3  # Dual System")
    print()

    print("3. Interactive Python:")
    print("   python")
    print("   >>> from ncgn import GraphBuilder")
    print("   >>> builder = GraphBuilder()")
    print("   >>> # Start experimenting!")
    print()

    print("4. Read documentation:")
    print("   - NCGN_README.md - Main documentation")
    print("   - INSTALLATION.md - Installation guide")
    print("   - IMPLEMENTATION_SUMMARY.md - Technical details")
    print()

    print("5. Explore examples:")
    print("   - scripts/ncgn_demo.py - Full demonstration")
    print("   - configs/ncgn_config.yaml - Configuration")


def main():
    """Main verification function"""
    print_header("NCGN Installation Verification")

    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA/GPU", check_cuda),
        ("NCGN Module", check_ncgn)
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error during {name} check: {e}")
            results[name] = False

    # Summary
    print_header("Verification Summary")

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All checks passed! You're ready to use NCGN.")
        print_quick_start_guide()
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help:")
        print("  - Read INSTALLATION.md")
        print("  - Check GitHub Issues")
        print("  - Contact support")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

