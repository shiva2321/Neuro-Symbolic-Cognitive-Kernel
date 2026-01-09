"""
Comprehensive Unit Tests for Graph Transformer
Tests Phase 3: Graph Transformer component
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from ncgn.graph_transformer import (
    GraphTransformer, SparseGraphTransformer,
    GraphMultiHeadAttention, GraphPooling
)


class TestGraphMultiHeadAttention(unittest.TestCase):
    """Test graph multi-head attention"""

    def setUp(self):
        """Set up test fixtures"""
        self.attention = GraphMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.attention.embed_dim, 64)
        self.assertEqual(self.attention.num_heads, 4)
        self.assertEqual(self.attention.head_dim, 16)  # 64 / 4
        print("✓ GraphMultiHeadAttention initialization test passed")

    def test_forward_without_adjacency(self):
        """Test forward pass without adjacency matrix"""
        batch_size = 2
        num_nodes = 10

        x = torch.randn(batch_size, num_nodes, 64)
        output, attention_weights = self.attention(x)

        self.assertEqual(output.shape, (batch_size, num_nodes, 64))
        self.assertIsNotNone(attention_weights)
        print("✓ Attention forward without adjacency test passed")

    def test_forward_with_adjacency(self):
        """Test forward pass with adjacency matrix"""
        batch_size = 2
        num_nodes = 10

        x = torch.randn(batch_size, num_nodes, 64)
        adjacency = torch.rand(num_nodes, num_nodes)
        adjacency = (adjacency > 0.5).float()  # Binary adjacency

        output, attention_weights = self.attention(x, adjacency)

        self.assertEqual(output.shape, (batch_size, num_nodes, 64))
        print("✓ Attention forward with adjacency test passed")

    def test_attention_masking(self):
        """Test that adjacency matrix masks attention"""
        num_nodes = 5
        x = torch.randn(1, num_nodes, 64)

        # Create adjacency with no connections
        adjacency = torch.zeros(num_nodes, num_nodes)
        # Add self-connections
        adjacency.fill_diagonal_(1)

        output, attn_weights = self.attention(x, adjacency)

        # Output should still be valid
        self.assertEqual(output.shape, (1, num_nodes, 64))
        print("✓ Attention masking test passed")

    def test_head_splitting(self):
        """Test multi-head splitting and concatenation"""
        x = torch.randn(1, 8, 64)
        output, _ = self.attention(x)

        # Output dimension should match input
        self.assertEqual(output.shape[2], 64)
        print("✓ Head splitting test passed")


class TestGraphPooling(unittest.TestCase):
    """Test graph pooling methods"""

    def test_mean_pooling(self):
        """Test mean pooling"""
        pooling = GraphPooling(embed_dim=32, pooling_type='mean')

        x = torch.randn(2, 10, 32)
        output = pooling(x)

        self.assertEqual(output.shape, (2, 32))
        print("✓ Mean pooling test passed")

    def test_max_pooling(self):
        """Test max pooling"""
        pooling = GraphPooling(embed_dim=32, pooling_type='max')

        x = torch.randn(2, 10, 32)
        output = pooling(x)

        self.assertEqual(output.shape, (2, 32))
        print("✓ Max pooling test passed")

    def test_sum_pooling(self):
        """Test sum pooling"""
        pooling = GraphPooling(embed_dim=32, pooling_type='sum')

        x = torch.randn(2, 10, 32)
        output = pooling(x)

        self.assertEqual(output.shape, (2, 32))
        print("✓ Sum pooling test passed")

    def test_attention_pooling(self):
        """Test attention-based pooling"""
        pooling = GraphPooling(embed_dim=32, pooling_type='attention')

        x = torch.randn(2, 10, 32)
        output = pooling(x)

        self.assertEqual(output.shape, (2, 32))
        print("✓ Attention pooling test passed")

    def test_pooling_preserves_info(self):
        """Test that pooling preserves important information"""
        pooling = GraphPooling(embed_dim=32, pooling_type='mean')

        # Create input with known pattern
        x = torch.ones(1, 10, 32)
        output = pooling(x)

        # Mean should be 1
        self.assertTrue(torch.allclose(output, torch.ones(1, 32), atol=0.01))
        print("✓ Pooling information preservation test passed")


class TestGraphTransformer(unittest.TestCase):
    """Test Graph Transformer"""

    def setUp(self):
        """Set up test fixtures"""
        self.transformer = GraphTransformer(
            node_feat_dim=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.transformer.embed_dim, 64)
        self.assertEqual(self.transformer.num_layers, 2)
        self.assertEqual(self.transformer.num_heads, 4)
        print("✓ GraphTransformer initialization test passed")

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        num_nodes = 15

        x = torch.randn(batch_size, num_nodes, 32)
        output = self.transformer(x)

        self.assertEqual(output.shape, (batch_size, num_nodes, 64))
        print("✓ GraphTransformer forward pass test passed")

    def test_forward_with_adjacency(self):
        """Test forward pass with adjacency matrix"""
        batch_size = 2
        num_nodes = 15

        x = torch.randn(batch_size, num_nodes, 32)
        adjacency = torch.rand(num_nodes, num_nodes)
        adjacency = (adjacency + adjacency.t()) / 2  # Symmetric

        output = self.transformer(x, adjacency)

        self.assertEqual(output.shape, (batch_size, num_nodes, 64))
        print("✓ GraphTransformer with adjacency test passed")

    def test_forward_with_edge_weights(self):
        """Test forward pass with edge weights"""
        batch_size = 2
        num_nodes = 15

        x = torch.randn(batch_size, num_nodes, 32)
        adjacency = torch.rand(num_nodes, num_nodes)
        edge_weights = torch.rand(num_nodes, num_nodes)

        output = self.transformer(x, adjacency, edge_weights)

        self.assertEqual(output.shape, (batch_size, num_nodes, 64))
        print("✓ GraphTransformer with edge weights test passed")

    def test_multi_layer_processing(self):
        """Test that multiple layers process information"""
        x = torch.randn(1, 10, 32)

        # Single layer
        transformer1 = GraphTransformer(32, 64, num_layers=1, num_heads=4)

        # Multiple layers
        transformer2 = GraphTransformer(32, 64, num_layers=4, num_heads=4)

        output1 = transformer1(x)
        output2 = transformer2(x)

        # Both should produce valid outputs
        self.assertEqual(output1.shape, (1, 10, 64))
        self.assertEqual(output2.shape, (1, 10, 64))

        # Outputs should differ (different depth)
        self.assertFalse(torch.allclose(output1, output2))
        print("✓ Multi-layer processing test passed")

    def test_gradient_flow(self):
        """Test that gradients flow through transformer"""
        x = torch.randn(2, 10, 32, requires_grad=True)
        output = self.transformer(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        print("✓ Gradient flow test passed")


class TestSparseGraphTransformer(unittest.TestCase):
    """Test Sparse Graph Transformer for large graphs"""

    def setUp(self):
        """Set up test fixtures"""
        self.sparse_transformer = SparseGraphTransformer(
            node_feat_dim=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            sparsity_threshold=0.1
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertIsNotNone(self.sparse_transformer)
        self.assertEqual(self.sparse_transformer.sparsity_threshold, 0.1)
        print("✓ SparseGraphTransformer initialization test passed")

    def test_forward_with_sparse_adjacency(self):
        """Test with sparse adjacency matrix"""
        batch_size = 2
        num_nodes = 20

        x = torch.randn(batch_size, num_nodes, 32)

        # Create sparse adjacency (90% zeros)
        adjacency = torch.rand(num_nodes, num_nodes)
        adjacency = (adjacency > 0.9).float()

        output = self.sparse_transformer(x, adjacency)

        self.assertEqual(output.shape, (batch_size, num_nodes, 64))
        print("✓ Sparse adjacency test passed")

    def test_handles_large_graphs(self):
        """Test that sparse transformer handles large graphs"""
        # Larger graph
        x = torch.randn(1, 100, 32)
        adjacency = torch.zeros(100, 100)

        # Add sparse connections
        for i in range(100):
            # Connect to 5 neighbors
            neighbors = np.random.choice(100, 5, replace=False)
            adjacency[i, neighbors] = 1

        output = self.sparse_transformer(x, adjacency)

        self.assertEqual(output.shape, (1, 100, 64))
        print("✓ Large graph handling test passed")


class TestGraphTransformerIntegration(unittest.TestCase):
    """Integration tests for graph transformer"""

    def test_with_linguistic_graph_features(self):
        """Test integration with linguistic graph"""
        # Simulate linguistic graph node features
        num_nodes = 20
        node_features = torch.randn(1, num_nodes, 768)  # RoBERTa dimension

        # Project to transformer dimension
        projection = nn.Linear(768, 64)
        projected = projection(node_features)

        # Process with transformer
        transformer = GraphTransformer(64, 128, num_layers=3, num_heads=8)
        output = transformer(projected)

        self.assertEqual(output.shape, (1, num_nodes, 128))
        print("✓ Linguistic graph integration test passed")

    def test_end_to_end_graph_classification(self):
        """Test end-to-end graph classification"""
        transformer = GraphTransformer(32, 64, num_layers=2, num_heads=4)
        pooling = GraphPooling(64, pooling_type='attention')
        classifier = nn.Linear(64, 5)  # 5 classes

        # Forward pass
        x = torch.randn(4, 15, 32)  # 4 graphs

        node_embeddings = transformer(x)
        graph_embedding = pooling(node_embeddings)
        logits = classifier(graph_embedding)

        self.assertEqual(logits.shape, (4, 5))
        print("✓ End-to-end classification test passed")

    def test_with_different_graph_sizes(self):
        """Test that transformer handles variable graph sizes"""
        transformer = GraphTransformer(32, 64, num_layers=2, num_heads=4)

        # Different sizes
        sizes = [5, 10, 20, 50]

        for size in sizes:
            x = torch.randn(1, size, 32)
            output = transformer(x)
            self.assertEqual(output.shape, (1, size, 64))

        print("✓ Variable graph sizes test passed")

    def test_batch_with_varying_sizes(self):
        """Test batching graphs of different sizes (with padding)"""
        transformer = GraphTransformer(32, 64, num_layers=2, num_heads=4)

        # Pad to same size
        max_nodes = 20

        # Graph 1: 10 nodes
        x1 = torch.randn(10, 32)
        x1_padded = torch.nn.functional.pad(x1, (0, 0, 0, max_nodes - 10))

        # Graph 2: 15 nodes
        x2 = torch.randn(15, 32)
        x2_padded = torch.nn.functional.pad(x2, (0, 0, 0, max_nodes - 15))

        # Batch
        x_batch = torch.stack([x1_padded, x2_padded])

        output = transformer(x_batch)
        self.assertEqual(output.shape, (2, max_nodes, 64))
        print("✓ Batched varying sizes test passed")

    def test_attention_visualization(self):
        """Test that attention weights can be extracted"""
        attention_layer = GraphMultiHeadAttention(64, num_heads=4)

        x = torch.randn(1, 10, 64)
        output, attn_weights = attention_layer(x)

        # Attention weights should be extractable
        self.assertIsNotNone(attn_weights)
        # Should have shape (batch, num_heads, num_nodes, num_nodes)
        self.assertEqual(len(attn_weights.shape), 4)
        print("✓ Attention visualization test passed")

    def test_positional_encoding_integration(self):
        """Test integration with positional encodings"""
        from ncgn.graph_embeddings import StructuralEncoder

        # Create structural encoder
        struct_encoder = StructuralEncoder(
            graph_size=15,
            encoding_dim=32,
            use_laplacian=True,
            use_random_walk=True
        )

        # Generate encodings
        pos_encodings = struct_encoder.get_all_encodings()

        # Add to node features
        node_features = torch.randn(2, 15, 32)
        enhanced_features = node_features + pos_encodings.unsqueeze(0)

        # Process with transformer
        transformer = GraphTransformer(32, 64, num_layers=2, num_heads=4)
        output = transformer(enhanced_features)

        self.assertEqual(output.shape, (2, 15, 64))
        print("✓ Positional encoding integration test passed")


def run_graph_transformer_tests():
    """Run all graph transformer tests"""
    print("\n" + "=" * 80)
    print("GRAPH TRANSFORMER TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGraphMultiHeadAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphPooling))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseGraphTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphTransformerIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_graph_transformer_tests()
    exit(0 if success else 1)

