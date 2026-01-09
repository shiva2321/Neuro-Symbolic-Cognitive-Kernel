"""
Master Test Runner for NCGN System
Runs all tests and generates comprehensive test reports.
"""

import unittest
import sys
import time
from pathlib import Path
import json
from datetime import datetime
import traceback

# Import NCGN Core test modules
from tests.test_linguistic_graph import run_linguistic_graph_tests
from tests.test_graph_embeddings import run_embedding_tests
from tests.test_spiking_neurons import run_spiking_neuron_tests
from tests.test_stdp_learning import run_stdp_tests
from tests.test_graph_transformer import run_graph_transformer_tests
from tests.test_dual_system import run_dual_system_tests
from tests.test_integration import run_integration_tests

# Import Agent test modules
from tests.test_data_harvester import run_data_harvester_tests
from tests.test_topological_converter import run_topological_converter_tests
from tests.test_bottleneck_optimizer import run_bottleneck_optimizer_tests
from tests.test_analytic_learner import run_analytic_learner_tests
from tests.test_analytics_suite import run_analytics_suite_tests


class TestReport:
    """Test report generator"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def add_result(self, component, success, duration):
        """Add test result"""
        self.results[component] = {
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.total_tests += 1

    def generate_report(self):
        """Generate comprehensive test report"""
        report = {
            'test_run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
            'total_tests': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'success_rate': self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
            'results': self.results
        }
        return report

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print(" " * 30 + "TEST SUMMARY")
        print("=" * 80)

        for component, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = f"{result['duration']:.2f}s"
            print(f"{component:40} {status:10} {duration:>10}")

        print("=" * 80)
        print(f"Total Tests:    {self.total_tests}")
        print(f"Passed:         {self.passed_tests} ({self.passed_tests/self.total_tests*100:.1f}%)")
        print(f"Failed:         {self.failed_tests} ({self.failed_tests/self.total_tests*100:.1f}%)")
        print(f"Total Duration: {(self.end_time - self.start_time).total_seconds():.2f}s")
        print("=" * 80)

        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉\n")
        else:
            print(f"\n⚠️  {self.failed_tests} TEST(S) FAILED\n")

    def save_report(self, output_dir="./test_results"):
        """Save test report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        # Save JSON report
        json_path = output_path / f"test_report_{report['test_run_id']}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Test report saved to: {json_path}")

        # Save latest report
        latest_path = output_path / "test_report_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2)


def run_all_tests():
    """Run all system tests"""
    print("\n" + "=" * 80)
    print(" " * 20 + "NCGN COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"\nStarting test run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    report = TestReport()
    report.start_time = datetime.now()

    # Test components - NCGN Core (Phase 1-3)
    test_components = [
        # Phase 1: Linguistic Graph Substrate
        ("Linguistic Graph (Phase 1)", run_linguistic_graph_tests),
        ("Graph Embeddings (Phase 1)", run_embedding_tests),

        # Phase 2: Neuromorphic Core
        ("Spiking Neurons (Phase 2)", run_spiking_neuron_tests),
        ("STDP Learning (Phase 2)", run_stdp_tests),

        # Phase 3: Dual System Architecture
        ("Graph Transformer (Phase 3)", run_graph_transformer_tests),
        ("Dual System (Phase 3)", run_dual_system_tests),

        # Integration Tests
        ("System Integration", run_integration_tests),

        # Specialized Agents
        ("Data Harvester (Agent 1)", run_data_harvester_tests),
        ("Topological Converter (Agent 2)", run_topological_converter_tests),
        ("Bottleneck Optimizer (Agent 3)", run_bottleneck_optimizer_tests),
        ("Analytic Learner (Agent 4)", run_analytic_learner_tests),
        ("Analytics Suite (Agent 5)", run_analytics_suite_tests),
    ]

    for component_name, test_func in test_suites:
        print(f"\n{'='*80}")
        print(f"Testing: {component_name}")
        print(f"{'='*80}")

        start = time.time()
        try:
            success = test_func()
        except Exception as e:
            print(f"\n❌ ERROR in {component_name}: {e}")
            import traceback
            traceback.print_exc()
            success = False

        duration = time.time() - start
        report.add_result(component_name, success, duration)

    report.end_time = datetime.now()

    # Print summary
    report.print_summary()

    # Save report
    report.save_report()

    return report.failed_tests == 0


def run_quick_tests():
    """Run quick smoke tests for basic functionality"""
    print("\n" + "=" * 80)
    print(" " * 25 + "QUICK SMOKE TESTS")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Import all agents
    print("\n[1/5] Testing agent imports...")
    try:
        from agents import DataHarvester, TopologicalConverter, BottleneckOptimizer, AnalyticLearner, AnalyticsSuite
        print("✅ All agents imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Import failed: {e}")
        tests_failed += 1

    # Test 2: Basic graph creation
    print("\n[2/5] Testing graph creation...")
    try:
        import torch
        import dgl
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))
        assert graph.num_nodes() == 3
        assert graph.num_edges() == 3
        print("✅ Graph creation successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        tests_failed += 1

    # Test 3: Data Harvester initialization
    print("\n[3/5] Testing Data Harvester...")
    try:
        from agents import DataHarvester
        harvester = DataHarvester()
        datasets = harvester.list_available_datasets()
        assert len(datasets) > 0
        print(f"✅ Data Harvester initialized ({len(datasets)} datasets)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data Harvester failed: {e}")
        tests_failed += 1

    # Test 4: Topological Converter
    print("\n[4/5] Testing Topological Converter...")
    try:
        from agents import TopologicalConverter
        import torch
        import dgl

        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        converter = TopologicalConverter()
        result = converter.construct_mcteg(graph)
        assert 'laplacian_pe' in result.ndata
        print("✅ Topological Converter working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Topological Converter failed: {e}")
        tests_failed += 1

    # Test 5: Analytic Learner
    print("\n[5/5] Testing Analytic Learner...")
    try:
        from agents import AnalyticLearner
        learner = AnalyticLearner()
        learner.create_rls_estimator('test', input_dim=10, output_dim=3)
        assert 'test' in learner.rls_estimators
        print("✅ Analytic Learner working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Analytic Learner failed: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Quick Tests: {tests_passed}/5 passed")
    if tests_failed == 0:
        print("✅ All quick tests passed!")
    else:
        print(f"⚠️  {tests_failed} test(s) failed")
    print("=" * 80)

    return tests_failed == 0


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description='NCGN Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke tests only')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--component', type=str, help='Test specific component')

    args = parser.parse_args()

    if args.quick:
        success = run_quick_tests()
    elif args.component:
        # Run specific component tests
        component_tests = {
            'harvester': run_data_harvester_tests,
            'converter': run_topological_converter_tests,
            'optimizer': run_bottleneck_optimizer_tests,
            'learner': run_analytic_learner_tests,
            'analytics': run_analytics_suite_tests,
        }

        if args.component.lower() in component_tests:
            success = component_tests[args.component.lower()]()
        else:
            print(f"Unknown component: {args.component}")
            print(f"Available components: {', '.join(component_tests.keys())}")
            success = False
    else:
        # Run full test suite by default
        success = run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

