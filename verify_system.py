"""
Simple Test Verification Script
Tests basic functionality without complex dependencies
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all agent modules can be imported"""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)

    try:
        print("Importing agents...")
        sys.path.insert(0, str(Path(__file__).parent))

        from agents import data_harvester
        print("✅ data_harvester imported")

        from agents import topological_converter
        print("✅ topological_converter imported")

        from agents import bottleneck_optimizer
        print("✅ bottleneck_optimizer imported")

        from agents import analytic_learner
        print("✅ analytic_learner imported")

        from agents import analytics_suite
        print("✅ analytics_suite imported")

        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_torch():
    """Test PyTorch and DGL"""
    print("\n" + "="*70)
    print("TEST 2: PyTorch and DGL")
    print("="*70)

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        import dgl
        print(f"✅ DGL {dgl.__version__}")

        # Create simple tensors
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ Created tensor: {x}")

        # Create simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        g = dgl.graph((src, dst))
        print(f"✅ Created graph: {g.num_nodes()} nodes, {g.num_edges()} edges")

        print("\n✅ PyTorch/DGL working!")
        return True
    except Exception as e:
        print(f"\n❌ PyTorch/DGL failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_initialization():
    """Test initializing each agent"""
    print("\n" + "="*70)
    print("TEST 3: Agent Initialization")
    print("="*70)

    try:
        from agents.data_harvester import DataHarvester
        harvester = DataHarvester()
        print(f"✅ DataHarvester: {len(harvester.datasets_catalog)} datasets")

        from agents.topological_converter import TopologicalConverter
        converter = TopologicalConverter()
        print("✅ TopologicalConverter initialized")

        from agents.bottleneck_optimizer import BottleneckOptimizer
        optimizer = BottleneckOptimizer()
        print("✅ BottleneckOptimizer initialized")

        from agents.analytic_learner import AnalyticLearner
        learner = AnalyticLearner()
        print("✅ AnalyticLearner initialized")

        from agents.analytics_suite import AnalyticsSuite
        analytics = AnalyticsSuite()
        print("✅ AnalyticsSuite initialized")

        print("\n✅ All agents initialized successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_operations():
    """Test simple operations with agents"""
    print("\n" + "="*70)
    print("TEST 4: Simple Operations")
    print("="*70)

    try:
        import torch
        import dgl
        from agents.data_harvester import DataHarvester
        from agents.topological_converter import TopologicalConverter

        # Test data harvester
        harvester = DataHarvester()
        datasets = harvester.list_available_datasets()
        print(f"✅ Listed {len(datasets)} datasets")

        # Test converter with simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        converter = TopologicalConverter()
        mcteg = converter.construct_mcteg(graph)
        print(f"✅ Converted graph: {mcteg.num_nodes()} nodes")

        print("\n✅ Simple operations successful!")
        return True
    except Exception as e:
        print(f"\n❌ Operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*80)
    print(" " * 25 + "SIMPLE VERIFICATION TESTS")
    print("="*80)

    tests = [
        ("Module Imports", test_imports),
        ("PyTorch/DGL", test_basic_torch),
        ("Agent Initialization", test_agent_initialization),
        ("Simple Operations", test_simple_operations),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "="*80)
    print(" " * 30 + "SUMMARY")
    print("="*80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:40} {status}")

    print("="*80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check errors above.\n")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

