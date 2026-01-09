"""
Unit Tests for Agent 5: Analytics Suite
Tests evaluation metrics, MIA, and visualization generation.
"""

import unittest
import torch
import dgl
from pathlib import Path
import tempfile
import shutil

from agents.analytics_suite import (
    AnalyticsSuite, CGLBConfig, TaskResult,
    MembershipInferenceAttack, HebbianTraceVisualizer
)


class TestMembershipInferenceAttack(unittest.TestCase):
    """Test suite for Membership Inference Attack"""

    def setUp(self):
        """Set up test fixtures"""
        self.attacker = MembershipInferenceAttack()

    def test_attack_model_training(self):
        """Test attack model training"""
        member_scores = torch.randn(100) + 1.0  # Higher scores for members
        non_member_scores = torch.randn(100)

        self.attacker.train_attack_model(member_scores, non_member_scores)

        self.assertIsNotNone(self.attacker.attack_model)
        print("✓ Attack model training test passed")

    def test_attack_evaluation(self):
        """Test attack evaluation"""
        # Train attack model
        member_scores = torch.randn(100) + 1.0
        non_member_scores = torch.randn(100)
        self.attacker.train_attack_model(member_scores, non_member_scores)

        # Evaluate
        test_member = torch.randn(50) + 1.0
        test_non_member = torch.randn(50)

        results = self.attacker.evaluate_attack(test_member, test_non_member)

        self.assertIn('attack_accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('privacy_leakage', results)
        print("✓ Attack evaluation test passed")


class TestHebbianTraceVisualizer(unittest.TestCase):
    """Test suite for Hebbian Trace Visualizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = HebbianTraceVisualizer(Path(self.temp_dir))

    def tearDown(self):
        """Clean up test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_trace_recording(self):
        """Test trace recording"""
        # Create simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        weight_changes = torch.randn(graph.num_edges())

        self.visualizer.record_trace(graph, weight_changes, "test_task")

        self.assertEqual(len(self.visualizer.trace_history), 1)
        self.assertEqual(self.visualizer.trace_history[0]['task_name'], "test_task")
        print("✓ Trace recording test passed")

    def test_trace_report_generation(self):
        """Test trace report generation"""
        # Record some traces
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        for i in range(3):
            weight_changes = torch.randn(graph.num_edges())
            self.visualizer.record_trace(graph, weight_changes, f"task_{i}")

        # Generate report
        report = self.visualizer.generate_trace_report()

        self.assertIn('num_tasks', report)
        self.assertIn('tasks', report)
        self.assertEqual(report['num_tasks'], 3)
        print("✓ Trace report generation test passed")


class TestAnalyticsSuite(unittest.TestCase):
    """Test suite for Analytics Suite"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CGLBConfig(
            num_tasks=3,
            run_mia=False,  # Disable for faster tests
            visualize_hebbian=False,
            output_dir=Path(self.temp_dir)
        )
        self.analytics = AnalyticsSuite(self.config)

    def tearDown(self):
        """Clean up test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test analytics suite initialization"""
        self.assertIsNotNone(self.analytics)
        self.assertIsNotNone(self.analytics.mia_attacker)
        self.assertIsNotNone(self.analytics.hebbian_visualizer)
        print("✓ Analytics suite initialization test passed")

    def test_task_evaluation(self):
        """Test task evaluation"""
        # Create mock model
        class MockModel:
            def predict(self, x):
                return torch.randn(len(x), 3)

        model = MockModel()

        # Create test data
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))
        features = torch.randn(3, 10)
        labels = torch.randint(0, 3, (3,))

        # Evaluate
        result = self.analytics.evaluate_task(
            model, graph, features, labels,
            task_id=0, task_name="test_task"
        )

        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.task_id, 0)
        self.assertEqual(result.task_name, "test_task")
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)
        print("✓ Task evaluation test passed")

    def test_average_performance_computation(self):
        """Test AP computation"""
        # Add some task results
        for i in range(3):
            result = TaskResult(
                task_id=i,
                task_name=f"task_{i}",
                accuracy=0.7 + i * 0.1,
                loss=0.5,
                training_time=1.0,
                num_samples=100,
                timestamp=float(i)
            )
            self.analytics.task_results.append(result)

        ap = self.analytics.compute_average_performance()

        self.assertGreater(ap, 0.0)
        self.assertLess(ap, 1.0)
        print("✓ Average performance computation test passed")

    def test_average_forgetting_computation(self):
        """Test AF computation"""
        # Simulate task accuracy history with forgetting
        self.analytics.initial_accuracies[0] = 0.9
        self.analytics.initial_accuracies[1] = 0.8

        self.analytics.task_accuracies[0] = [0.9, 0.85, 0.8]  # Forgetting
        self.analytics.task_accuracies[1] = [0.8, 0.78, 0.75]  # Forgetting

        af = self.analytics.compute_average_forgetting()

        self.assertGreater(af, 0.0)  # Should have some forgetting
        print("✓ Average forgetting computation test passed")

    def test_forward_transfer_computation(self):
        """Test forward transfer computation"""
        # Simulate learning multiple tasks
        self.analytics.initial_accuracies[0] = 0.5
        self.analytics.initial_accuracies[1] = 0.6  # Better than baseline
        self.analytics.initial_accuracies[2] = 0.65

        ft = self.analytics.compute_forward_transfer()

        self.assertIsInstance(ft, float)
        print("✓ Forward transfer computation test passed")

    def test_cglb_report_generation(self):
        """Test CGLB report generation"""
        # Add some task results
        for i in range(3):
            result = TaskResult(
                task_id=i,
                task_name=f"task_{i}",
                accuracy=0.7,
                loss=0.5,
                training_time=1.0,
                num_samples=100,
                timestamp=float(i)
            )
            self.analytics.task_results.append(result)
            self.analytics.initial_accuracies[i] = 0.7
            self.analytics.task_accuracies[i] = [0.7]

        # Generate report
        report = self.analytics.generate_cglb_report()

        self.assertGreater(report.average_performance, 0.0)
        self.assertGreaterEqual(report.average_forgetting, 0.0)
        self.assertGreater(report.total_time, 0.0)

        # Check if report file was created
        report_file = self.config.output_dir / "cglb_report.json"
        self.assertTrue(report_file.exists())
        print("✓ CGLB report generation test passed")

    def test_export_results(self):
        """Test results export"""
        # Add some results
        self.analytics.task_accuracies[0] = [0.7, 0.8]
        self.analytics.initial_accuracies[0] = 0.7

        # Export
        results = self.analytics.export_results()

        self.assertIn('task_accuracies', results)
        self.assertIn('task_results', results)
        self.assertIn('metrics', results)
        print("✓ Results export test passed")


class TestAnalyticsSuiteIntegration(unittest.TestCase):
    """Integration tests for Analytics Suite"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CGLBConfig(
            num_tasks=3,
            run_mia=False,
            visualize_hebbian=False,
            output_dir=Path(self.temp_dir)
        )
        self.analytics = AnalyticsSuite(self.config)

    def tearDown(self):
        """Clean up test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_multi_task_evaluation_workflow(self):
        """Test complete multi-task evaluation workflow"""
        # Create mock model
        class MockModel:
            def predict(self, x):
                return torch.randn(len(x), 3)

        model = MockModel()

        # Simulate 3 tasks
        for task_id in range(3):
            src = torch.tensor([0, 1, 2])
            dst = torch.tensor([1, 2, 0])
            graph = dgl.graph((src, dst))
            features = torch.randn(3, 10)
            labels = torch.randint(0, 3, (3,))

            # Evaluate
            result = self.analytics.evaluate_task(
                model, graph, features, labels,
                task_id=task_id, task_name=f"task_{task_id}"
            )

            self.assertIsNotNone(result)

        # Generate final report
        report = self.analytics.generate_cglb_report()

        self.assertEqual(len(self.analytics.task_results), 3)
        self.assertIsNotNone(report)
        print("✓ Multi-task evaluation workflow test passed")


def run_analytics_suite_tests():
    """Run all analytics suite tests"""
    print("\n" + "="*70)
    print("Running Analytics Suite Tests")
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMembershipInferenceAttack))
    suite.addTests(loader.loadTestsFromTestCase(TestHebbianTraceVisualizer))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticsSuite))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticsSuiteIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_analytics_suite_tests()
    exit(0 if success else 1)

