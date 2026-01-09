"""
Unit Tests for Agent 3: Bottleneck Optimizer
Tests curvature computation, bottleneck detection, and graph rewiring.
"""

import unittest
import torch
import dgl
import networkx as nx

from agents.bottleneck_optimizer import (
    BottleneckOptimizer, RewiringConfig,
    RicciCurvatureCalculator, BiharmonicDistanceCalculator
)


class TestRicciCurvatureCalculator(unittest.TestCase):
    """Test suite for Ricci Curvature Calculator"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = RicciCurvatureCalculator()

    def test_curvature_computation_simple(self):
        """Test curvature computation on simple graph"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        # Compute curvature
        curvatures = self.calculator.compute_ollivier_ricci_curvature(graph)

        self.assertEqual(len(curvatures), graph.num_edges())
        self.assertFalse(torch.isnan(curvatures).any())
        self.assertFalse(torch.isinf(curvatures).any())
        print("✓ Simple curvature computation test passed")

    def test_curvature_computation_dense(self):
        """Test curvature computation on dense graph"""
        # Create a dense graph (complete graph K5)
        nodes = 5
        edges = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
        src = torch.tensor([e[0] for e in edges])
        dst = torch.tensor([e[1] for e in edges])
        graph = dgl.graph((src, dst))

        # Compute curvature
        curvatures = self.calculator.compute_ollivier_ricci_curvature(graph)

        # Dense graphs should have positive curvature
        self.assertGreater(curvatures.mean().item(), 0)
        print("✓ Dense graph curvature test passed")

    def test_bottleneck_identification(self):
        """Test bottleneck edge identification"""
        # Create a graph with a known bottleneck
        # Two cliques connected by a single edge
        src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2,  # First clique
                           3, 4,  # Bridge
                           5, 5, 5, 6, 6, 6, 7, 7, 7])  # Second clique
        dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3,
                           5, 6,
                           6, 7, 8, 5, 7, 8, 5, 6, 8])
        graph = dgl.graph((src, dst))

        # Compute curvatures
        curvatures = self.calculator.compute_ollivier_ricci_curvature(graph)

        # Identify bottlenecks
        bottlenecks = self.calculator.identify_bottlenecks(graph, curvatures, threshold=-0.3)

        self.assertGreater(len(bottlenecks), 0)
        print("✓ Bottleneck identification test passed")


class TestBottleneckOptimizer(unittest.TestCase):
    """Test suite for Bottleneck Optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = RewiringConfig(
            use_ricci_curvature=True,
            ricci_threshold=-0.5,
            max_edges_to_add=50,
            max_edges_to_remove=20,
            add_virtual_node=True,
            rewiring_iterations=2
        )
        self.optimizer = BottleneckOptimizer(self.config)

    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.ricci_calculator)
        self.assertIsNotNone(self.optimizer.biharmonic_calculator)
        print("✓ Optimizer initialization test passed")

    def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 1, 2])
        dst = torch.tensor([1, 2, 3, 0, 0, 1])
        graph = dgl.graph((src, dst))

        # Detect bottlenecks
        result = self.optimizer.detect_bottlenecks(graph)

        self.assertIn('bottleneck_edges', result)
        self.assertIn('curvatures', result)
        self.assertIn('high_betweenness_nodes', result)
        print("✓ Bottleneck detection test passed")

    def test_edge_addition(self):
        """Test edge addition around bottlenecks"""
        # Create a graph with bottleneck
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        bottlenecks = [(1, 2)]

        # Add edges
        new_edges = self.optimizer.add_edges_around_bottlenecks(graph, bottlenecks)

        self.assertIsInstance(new_edges, list)
        print("✓ Edge addition test passed")

    def test_edge_removal(self):
        """Test redundant edge removal"""
        # Create a dense graph
        src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3])
        graph = dgl.graph((src, dst))

        # Compute curvatures
        curvatures = self.optimizer.ricci_calculator.compute_ollivier_ricci_curvature(graph)

        # Remove edges
        edges_to_remove = self.optimizer.remove_redundant_edges(graph, curvatures)

        self.assertIsInstance(edges_to_remove, list)
        print("✓ Edge removal test passed")

    def test_virtual_node_addition(self):
        """Test virtual node addition"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 1, 2])
        dst = torch.tensor([1, 2, 3, 0, 0, 1])
        graph = dgl.graph((src, dst))

        initial_nodes = graph.num_nodes()

        # Add virtual node
        new_graph, virtual_id = self.optimizer.add_virtual_node(graph)

        self.assertEqual(new_graph.num_nodes(), initial_nodes + 1)
        self.assertEqual(virtual_id, initial_nodes)
        self.assertGreater(new_graph.num_edges(), graph.num_edges())
        print("✓ Virtual node addition test passed")

    def test_rewiring_application(self):
        """Test rewiring application"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        edges_to_add = [(0, 2)]
        edges_to_remove = []

        # Apply rewiring
        new_graph = self.optimizer.apply_rewiring(graph, edges_to_add, edges_to_remove)

        self.assertGreaterEqual(new_graph.num_edges(), graph.num_edges())
        print("✓ Rewiring application test passed")

    def test_full_optimization(self):
        """Test full optimization pipeline"""
        # Create a graph with bottleneck
        src = torch.tensor([0, 1, 2, 3, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0, 0, 1, 2])
        graph = dgl.graph((src, dst))

        initial_edges = graph.num_edges()

        # Optimize
        result = self.optimizer.optimize(graph, num_iterations=1)

        self.assertIn('optimized_graph', result)
        self.assertIn('optimization_history', result)
        self.assertIn('edges_added_total', result)

        optimized_graph = result['optimized_graph']
        self.assertGreaterEqual(optimized_graph.num_nodes(), graph.num_nodes())
        print("✓ Full optimization test passed")

    def test_statistics(self):
        """Test bottleneck statistics"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        # Get statistics
        stats = self.optimizer.get_bottleneck_statistics(graph)

        self.assertIn('num_bottleneck_edges', stats)
        self.assertIn('num_high_betweenness_nodes', stats)
        print("✓ Statistics test passed")


def run_bottleneck_optimizer_tests():
    """Run all bottleneck optimizer tests"""
    print("\n" + "="*70)
    print("Running Bottleneck Optimizer Tests")
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestRicciCurvatureCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestBottleneckOptimizer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_bottleneck_optimizer_tests()
    exit(0 if success else 1)

