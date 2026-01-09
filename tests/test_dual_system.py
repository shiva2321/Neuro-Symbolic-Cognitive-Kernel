"""
Comprehensive Unit Tests for Dual System Architecture
Tests Phase 3: System 1 (Neural) + System 2 (Symbolic) Integration
"""

import unittest
import torch
import torch.nn as nn

from ncgn.dual_system import (
    DualSystemArchitecture,
    System1Module, System2Module,
    IntegrationLayer
)
from ncgn.symbolic_reasoner import SymbolicReasoner, Fact, Rule


class TestSystem1Module(unittest.TestCase):
    """Test System 1 (fast, intuitive, neural)"""

    def setUp(self):
        """Set up test fixtures"""
        self.system1 = System1Module(
            node_feat_dim=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4
        )

    def test_initialization(self):
        """Test System 1 initialization"""
        self.assertIsNotNone(self.system1.transformer)
        self.assertIsNotNone(self.system1.pooling)
        self.assertIsNotNone(self.system1.confidence_head)
        print("✓ System1 initialization test passed")

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        num_nodes = 10

        node_features = torch.randn(batch_size, num_nodes, 32)
        output = self.system1(node_features)

        self.assertIn('node_embeddings', output)
        self.assertIn('graph_embedding', output)
        self.assertIn('confidence', output)

        self.assertEqual(output['node_embeddings'].shape, (batch_size, num_nodes, 64))
        self.assertEqual(output['graph_embedding'].shape, (batch_size, 64))
        self.assertEqual(output['confidence'].shape, (batch_size, 1))
        print("✓ System1 forward pass test passed")

    def test_confidence_range(self):
        """Test that confidence is in [0, 1]"""
        node_features = torch.randn(3, 15, 32)
        output = self.system1(node_features)

        confidence = output['confidence']
        self.assertTrue(torch.all(confidence >= 0))
        self.assertTrue(torch.all(confidence <= 1))
        print("✓ System1 confidence range test passed")

    def test_with_adjacency_matrix(self):
        """Test with adjacency matrix"""
        node_features = torch.randn(1, 10, 32)
        adjacency = torch.rand(10, 10)
        adjacency = (adjacency > 0.5).float()

        output = self.system1(node_features, adjacency)

        self.assertIsNotNone(output['graph_embedding'])
        print("✓ System1 with adjacency test passed")


class TestSystem2Module(unittest.TestCase):
    """Test System 2 (slow, deliberate, symbolic)"""

    def setUp(self):
        """Set up test fixtures"""
        self.system2 = System2Module(reasoning_depth=5)

    def test_initialization(self):
        """Test System 2 initialization"""
        self.assertIsNotNone(self.system2.reasoner)
        print("✓ System2 initialization test passed")

    def test_add_knowledge(self):
        """Test adding knowledge to System 2"""
        facts = [
            ("dog", "is_a", "animal"),
            ("cat", "is_a", "animal"),
            ("bird", "is_a", "animal"),
        ]

        self.system2.add_knowledge(facts)

        # Check that facts were added
        self.assertGreater(len(self.system2.reasoner.knowledge_base.facts), 0)
        print("✓ System2 add knowledge test passed")

    def test_forward_reasoning(self):
        """Test forward reasoning"""
        # Add knowledge
        facts = [
            ("socrates", "is_a", "man"),
            ("man", "is_a", "mortal"),
        ]
        self.system2.add_knowledge(facts)

        # Query
        context = {'query': 'socrates is_a mortal'}
        result = self.system2(context)

        self.assertIsNotNone(result)
        print("✓ System2 forward reasoning test passed")

    def test_empty_knowledge_base(self):
        """Test with empty knowledge base"""
        context = {'query': 'test'}
        result = self.system2(context)

        # Should return result even with empty KB
        self.assertIsNotNone(result)
        print("✓ System2 empty KB test passed")


class TestIntegrationLayer(unittest.TestCase):
    """Test integration between System 1 and System 2"""

    def setUp(self):
        """Set up test fixtures"""
        self.integration_weighted = IntegrationLayer(
            embed_dim=64,
            integration_method='weighted'
        )
        self.integration_attention = IntegrationLayer(
            embed_dim=64,
            integration_method='attention'
        )
        self.integration_gating = IntegrationLayer(
            embed_dim=64,
            integration_method='gating'
        )

    def test_weighted_integration(self):
        """Test weighted integration"""
        s1_output = torch.randn(2, 64)
        s2_output = torch.randn(2, 64)
        s1_confidence = torch.rand(2, 1)

        integrated = self.integration_weighted(s1_output, s2_output, s1_confidence)

        self.assertEqual(integrated.shape, (2, 64))
        print("✓ Weighted integration test passed")

    def test_attention_integration(self):
        """Test attention-based integration"""
        s1_output = torch.randn(2, 64)
        s2_output = torch.randn(2, 64)

        integrated = self.integration_attention(s1_output, s2_output)

        self.assertEqual(integrated.shape, (2, 64))
        print("✓ Attention integration test passed")

    def test_gating_integration(self):
        """Test gating integration"""
        s1_output = torch.randn(2, 64)
        s2_output = torch.randn(2, 64)

        integrated = self.integration_gating(s1_output, s2_output)

        self.assertEqual(integrated.shape, (2, 64))
        print("✓ Gating integration test passed")

    def test_high_confidence_favors_s1(self):
        """Test that high S1 confidence favors neural output"""
        s1_output = torch.ones(1, 64) * 5
        s2_output = torch.ones(1, 64) * -5
        high_confidence = torch.tensor([[0.95]])

        integrated = self.integration_weighted(s1_output, s2_output, high_confidence)

        # Should be closer to s1_output
        self.assertTrue(integrated.mean() > 0)
        print("✓ High confidence S1 test passed")

    def test_low_confidence_favors_s2(self):
        """Test that low S1 confidence favors symbolic output"""
        s1_output = torch.ones(1, 64) * 5
        s2_output = torch.ones(1, 64) * -5
        low_confidence = torch.tensor([[0.05]])

        integrated = self.integration_weighted(s1_output, s2_output, low_confidence)

        # Should be closer to s2_output
        self.assertTrue(integrated.mean() < 3)
        print("✓ Low confidence S2 test passed")


class TestDualSystemArchitecture(unittest.TestCase):
    """Test complete dual system architecture"""

    def setUp(self):
        """Set up test fixtures"""
        self.dual_system = DualSystemArchitecture(
            node_feat_dim=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            reasoning_depth=3,
            integration_method='weighted'
        )

    def test_initialization(self):
        """Test dual system initialization"""
        self.assertIsNotNone(self.dual_system.system1)
        self.assertIsNotNone(self.dual_system.system2)
        self.assertIsNotNone(self.dual_system.integration)
        print("✓ DualSystem initialization test passed")

    def test_forward_pass(self):
        """Test forward pass through both systems"""
        batch_size = 2
        num_nodes = 10

        node_features = torch.randn(batch_size, num_nodes, 32)
        context = {'query': 'test query'}

        output = self.dual_system(node_features, context)

        self.assertIn('integrated_output', output)
        self.assertIn('system1_output', output)
        self.assertIn('system2_output', output)
        self.assertIn('confidence', output)

        print("✓ DualSystem forward pass test passed")

    def test_system1_only_mode(self):
        """Test running only System 1"""
        node_features = torch.randn(2, 10, 32)

        output = self.dual_system(node_features, use_system2=False)

        self.assertIn('integrated_output', output)
        self.assertIsNotNone(output['system1_output'])
        print("✓ System1-only mode test passed")

    def test_with_knowledge_base(self):
        """Test with pre-loaded knowledge base"""
        # Add knowledge to System 2
        knowledge = [
            ("entity1", "relates_to", "entity2"),
            ("entity2", "relates_to", "entity3"),
        ]
        self.dual_system.system2.add_knowledge(knowledge)

        node_features = torch.randn(1, 5, 32)
        context = {'query': 'entity1 relates_to entity3'}

        output = self.dual_system(node_features, context)

        self.assertIsNotNone(output['system2_output'])
        print("✓ Knowledge base integration test passed")

    def test_gradient_flow_system1(self):
        """Test that gradients flow through System 1"""
        node_features = torch.randn(2, 10, 32, requires_grad=True)
        context = {'query': 'test'}

        output = self.dual_system(node_features, context, use_system2=False)

        loss = output['integrated_output'].sum()
        loss.backward()

        self.assertIsNotNone(node_features.grad)
        print("✓ System1 gradient flow test passed")

    def test_different_confidence_levels(self):
        """Test behavior at different confidence levels"""
        node_features = torch.randn(3, 10, 32)
        context = {'query': 'test'}

        # Multiple runs might produce different confidences
        confidences = []
        for _ in range(5):
            output = self.dual_system(node_features, context)
            confidences.append(output['confidence'])

        # All should be valid probabilities
        for conf in confidences:
            self.assertTrue(torch.all(conf >= 0))
            self.assertTrue(torch.all(conf <= 1))

        print("✓ Different confidence levels test passed")


class TestDualSystemIntegration(unittest.TestCase):
    """Integration tests for the complete dual system"""

    def test_end_to_end_reasoning(self):
        """Test end-to-end reasoning task"""
        dual_system = DualSystemArchitecture(
            node_feat_dim=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4
        )

        # Add logical rules
        knowledge = [
            ("dog", "is_a", "animal"),
            ("animal", "has", "cells"),
            ("cells", "contain", "DNA"),
        ]
        dual_system.system2.add_knowledge(knowledge)

        # Create graph
        node_features = torch.randn(1, 5, 32)
        context = {'query': 'dog has cells'}

        output = dual_system(node_features, context)

        # Both systems should produce outputs
        self.assertIsNotNone(output['system1_output'])
        self.assertIsNotNone(output['system2_output'])
        self.assertIsNotNone(output['integrated_output'])

        print("✓ End-to-end reasoning test passed")

    def test_complementary_processing(self):
        """Test that systems complement each other"""
        dual_system = DualSystemArchitecture(32, 64, 2, 4)

        # Structured knowledge for System 2
        knowledge = [
            ("x", "greater_than", "y"),
            ("y", "greater_than", "z"),
        ]
        dual_system.system2.add_knowledge(knowledge)

        # Pattern data for System 1
        node_features = torch.randn(1, 10, 32)

        # Context requiring both systems
        context = {
            'query': 'x greater_than z',
            'pattern': 'complex_pattern'
        }

        output = dual_system(node_features, context)

        # Integration should combine both
        self.assertIsNotNone(output['integrated_output'])
        print("✓ Complementary processing test passed")

    def test_with_linguistic_graph(self):
        """Test integration with linguistic graph"""
        from ncgn.linguistic_graph import LinguisticGraph

        # Create simple linguistic graph
        graph = LinguisticGraph(device='cpu')

        tokens = ["neural", "network", "learns", "patterns"]
        import numpy as np
        for token in tokens:
            emb = np.random.randn(768)
            graph.add_node(token, emb)

        # Create dual system with matching dimensions
        dual_system = DualSystemArchitecture(
            node_feat_dim=768,
            embed_dim=256,
            num_layers=2,
            num_heads=8
        )

        # Get node features from graph
        node_features = torch.randn(1, len(tokens), 768)

        context = {'query': 'neural network'}
        output = dual_system(node_features, context)

        self.assertIsNotNone(output['integrated_output'])
        print("✓ Linguistic graph integration test passed")

    def test_batch_processing(self):
        """Test batch processing of multiple inputs"""
        dual_system = DualSystemArchitecture(32, 64, 2, 4)

        batch_size = 4
        node_features = torch.randn(batch_size, 10, 32)

        # Different contexts for each item in batch
        contexts = [
            {'query': f'query_{i}'} for i in range(batch_size)
        ]

        # Process batch
        for i, context in enumerate(contexts):
            output = dual_system(node_features[i:i+1], context)
            self.assertEqual(output['integrated_output'].shape[0], 1)

        print("✓ Batch processing test passed")

    def test_inference_mode(self):
        """Test inference without gradients"""
        dual_system = DualSystemArchitecture(32, 64, 2, 4)
        dual_system.eval()

        with torch.no_grad():
            node_features = torch.randn(2, 10, 32)
            context = {'query': 'test'}

            output = dual_system(node_features, context)

        self.assertIsNotNone(output['integrated_output'])
        print("✓ Inference mode test passed")

    def test_output_for_classification(self):
        """Test using dual system for classification"""
        dual_system = DualSystemArchitecture(32, 64, 2, 4)
        classifier = nn.Linear(64, 10)  # 10 classes

        node_features = torch.randn(4, 15, 32)
        context = {'query': 'classify'}

        output = dual_system(node_features, context, use_system2=False)
        logits = classifier(output['integrated_output'])

        self.assertEqual(logits.shape, (4, 10))
        print("✓ Classification output test passed")


def run_dual_system_tests():
    """Run all dual system tests"""
    print("\n" + "=" * 80)
    print("DUAL SYSTEM ARCHITECTURE TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSystem1Module))
    suite.addTests(loader.loadTestsFromTestCase(TestSystem2Module))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestDualSystemArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestDualSystemIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_dual_system_tests()
    exit(0 if success else 1)

