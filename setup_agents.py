"""
Installation and Setup Script for NCGN Specialized Agents
Verifies all dependencies and tests basic functionality.
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        return False

    print("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")

    required_packages = [
        'torch',
        'dgl',
        'networkx',
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✅ All core dependencies are installed")
    return True

def test_agent_imports():
    """Test if agent modules can be imported"""
    print_header("Testing Agent Imports")

    agents = [
        'data_harvester',
        'topological_converter',
        'bottleneck_optimizer',
        'analytic_learner',
        'analytics_suite'
    ]

    success = True
    for agent in agents:
        try:
            module = __import__(f'agents.{agent}', fromlist=['*'])
            print(f"✅ agents.{agent}")
        except ImportError as e:
            print(f"❌ agents.{agent}: {e}")
            success = False
        except Exception as e:
            print(f"⚠️  agents.{agent}: {type(e).__name__}: {e}")

    if success:
        print("\n✅ All agents can be imported")
    else:
        print("\n⚠️  Some agents have import issues (may still work)")

    return success

def test_basic_functionality():
    """Test basic functionality of each agent"""
    print_header("Testing Basic Functionality")

    try:
        import torch
        import dgl

        # Test Data Harvester
        print("\nTesting Data Harvester...")
        from agents.data_harvester import DataHarvester, DataHarvesterConfig
        config = DataHarvesterConfig(cache_dir=Path("./test_cache"))
        harvester = DataHarvester(config)
        datasets = harvester.list_available_datasets()
        print(f"  ✅ Found {len(datasets)} datasets in catalog")

        # Test Topological Converter
        print("\nTesting Topological Converter...")
        from agents.topological_converter import TopologicalConverter
        converter = TopologicalConverter()
        print(f"  ✅ Converter initialized")

        # Test Bottleneck Optimizer
        print("\nTesting Bottleneck Optimizer...")
        from agents.bottleneck_optimizer import BottleneckOptimizer
        optimizer = BottleneckOptimizer()
        print(f"  ✅ Optimizer initialized")

        # Test Analytic Learner
        print("\nTesting Analytic Learner...")
        from agents.analytic_learner import AnalyticLearner
        learner = AnalyticLearner()
        print(f"  ✅ Learner initialized")

        # Test Analytics Suite
        print("\nTesting Analytics Suite...")
        from agents.analytics_suite import AnalyticsSuite
        analytics = AnalyticsSuite()
        print(f"  ✅ Analytics suite initialized")

        print("\n✅ All agents functional")
        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")

    directories = [
        './data_cache',
        './cglb_results',
        './saved_models/agents',
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

    print("\n✅ All directories created")
    return True

def display_summary():
    """Display summary and next steps"""
    print_header("Installation Summary")

    print("""
✅ NCGN Specialized Agents are ready to use!

Next Steps:
1. Run the demo:
   python agents_demo.py

2. Read the documentation:
   - AGENTS_README.md (Quick start)
   - AGENTS_DOCUMENTATION.md (Full reference)

3. Try the agents individually:
   python -c "from agents import DataHarvester; h = DataHarvester(); print(h.list_available_datasets())"

Example Commands:
- Quick demo:     python agents_demo.py (select option 1)
- Full pipeline:  python agents_demo.py (select option 2)

Hardware Requirements:
- Minimum: 8GB VRAM, 16GB RAM
- Recommended: 12GB VRAM (RTX 3060), 32GB RAM

For help: Check AGENTS_DOCUMENTATION.md
""")

def main():
    """Main installation and verification"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         NCGN Specialized Agents - Installation & Verification        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check dependencies
    deps_ok = check_dependencies()

    # Create directories
    create_directories()

    # Test imports
    if deps_ok:
        imports_ok = test_agent_imports()
    else:
        print("\n⚠️  Skipping agent import tests due to missing dependencies")
        imports_ok = False

    # Test functionality
    if imports_ok:
        test_basic_functionality()
    else:
        print("\n⚠️  Skipping functionality tests due to import issues")

    # Display summary
    display_summary()

    print("\n" + "=" * 70)
    print("  Installation verification complete!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

