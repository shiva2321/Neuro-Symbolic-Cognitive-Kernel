"""
Unit Tests for Agent 2: Topological Converter
Tests MC-TEG construction, positional encodings, and Graph2Seq tokenization.
"""

import unittest
import torch
import dgl
import numpy as np

from agents.topological_converter import (
    TopologicalConverter, MCTEGConfig, Graph2SeqConfig,
    PositionalEncoder, GraphWordTokenizer
)


class TestPositionalEncoder(unittest.TestCase):
    """Test suite for Positional Encoder"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MCTEGConfig(
            use_laplacian_pe=True,
            use_rwpe=True,
            num_laplacian_eigenvectors=4,
            rwpe_walk_length=10
        )
        self.encoder = PositionalEncoder(self.config)

    def test_laplacian_pe_computation(self):
        """Test Laplacian positional encoding computation"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 1, 2])
        dst = torch.tensor([1, 2, 3, 0, 0, 1])
        graph = dgl.graph((src, dst))

        # Compute Laplacian PE
        pe = self.encoder.compute_laplacian_pe(graph)

        self.assertEqual(pe.shape[0], graph.num_nodes())
        self.assertLessEqual(pe.shape[1], self.config.num_laplacian_eigenvectors)
        self.assertFalse(torch.isnan(pe).any())
        print("✓ Laplacian PE computation test passed")

    def test_rwpe_computation(self):
        """Test Random Walk PE computation"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        # Compute RWPE
        rwpe = self.encoder.compute_rwpe(graph)

        self.assertEqual(rwpe.shape[0], graph.num_nodes())
        self.assertEqual(rwpe.shape[1], self.config.rwpe_walk_length)
        self.assertFalse(torch.isnan(rwpe).any())
        print("✓ RWPE computation test passed")


class TestGraphWordTokenizer(unittest.TestCase):
    """Test suite for Graph Word Tokenizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Graph2SeqConfig(
            walk_length=10,
            num_walks_per_node=2,
            graph_word_dim=32,
            vocab_size=100
        )
        self.tokenizer = GraphWordTokenizer(self.config)

    def test_random_walk_generation(self):
        """Test random walk generation"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 4, 0, 0, 1, 2])
        graph = dgl.graph((src, dst))

        # Generate a single walk
        walk = self.tokenizer.random_walk(graph, start_node=0, walk_length=5)

        self.assertGreater(len(walk), 0)
        self.assertLessEqual(len(walk), 5)
        self.assertEqual(walk[0], 0)  # Should start at start_node
        print("✓ Random walk generation test passed")

    def test_walks_generation(self):
        """Test generation of multiple walks"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 1, 2])
        dst = torch.tensor([1, 2, 3, 0, 0, 1])
        graph = dgl.graph((src, dst))

        # Generate walks
        walks = self.tokenizer.generate_walks(graph)

        expected_walks = graph.num_nodes() * self.config.num_walks_per_node
        self.assertEqual(len(walks), expected_walks)
        print("✓ Multiple walks generation test passed")

    def test_walks_to_sequences(self):
        """Test conversion of walks to token sequences"""
        walks = [[0, 1, 2], [1, 2, 3], [2, 3, 0]]

        sequences, vocab = self.tokenizer.walks_to_sequences(walks)

        self.assertGreater(len(sequences), 0)
        self.assertGreater(len(vocab), 0)
        self.assertIsNotNone(self.tokenizer.node_to_token_id)
        print("✓ Walks to sequences conversion test passed")


class TestTopologicalConverter(unittest.TestCase):
    """Test suite for Topological Converter"""

    def setUp(self):
        """Set up test fixtures"""
        self.mcteg_config = MCTEGConfig(
            use_laplacian_pe=True,
            use_rwpe=True,
            num_laplacian_eigenvectors=4
        )
        self.graph2seq_config = Graph2SeqConfig(
            walk_length=10,
            num_walks_per_node=2,
            vocab_size=50
        )
        self.converter = TopologicalConverter(self.mcteg_config, self.graph2seq_config)

    def test_initialization(self):
        """Test converter initialization"""
        self.assertIsNotNone(self.converter)
        self.assertIsNotNone(self.converter.pe_encoder)
        self.assertIsNotNone(self.converter.tokenizer)
        print("✓ Converter initialization test passed")

    def test_mcteg_construction_basic(self):
        """Test basic MC-TEG construction"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        # Construct MC-TEG
        mcteg = self.converter.construct_mcteg(graph)

        self.assertEqual(mcteg.num_nodes(), graph.num_nodes())
        self.assertEqual(mcteg.num_edges(), graph.num_edges())
        self.assertIn('laplacian_pe', mcteg.ndata)
        self.assertIn('rwpe', mcteg.ndata)
        print("✓ Basic MC-TEG construction test passed")

    def test_mcteg_with_descriptions(self):
        """Test MC-TEG construction with node descriptions"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        node_descriptions = {
            0: "First node",
            1: "Second node",
            2: "Third node"
        }

        # Construct MC-TEG
        mcteg = self.converter.construct_mcteg(graph, node_descriptions=node_descriptions)

        self.assertIn('description_feat', mcteg.ndata)
        print("✓ MC-TEG with descriptions test passed")

    def test_mcteg_with_edge_types(self):
        """Test MC-TEG construction with edge types"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        edge_types = {
            (0, 1): "contains",
            (1, 2): "follows",
            (2, 0): "precedes"
        }

        # Construct MC-TEG
        mcteg = self.converter.construct_mcteg(graph, edge_types=edge_types)

        self.assertIn('edge_type', mcteg.edata)
        print("✓ MC-TEG with edge types test passed")

    def test_graph_to_sequence(self):
        """Test graph to sequence conversion"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 1, 2])
        dst = torch.tensor([1, 2, 3, 0, 0, 1])
        graph = dgl.graph((src, dst))

        # Convert to sequences
        result = self.converter.graph_to_sequence(graph, train_embeddings=False)

        self.assertIn('token_sequences', result)
        self.assertIn('vocab_mapping', result)
        self.assertGreater(result['num_unique_tokens'], 0)
        self.assertGreater(result['num_sequences'], 0)
        print("✓ Graph to sequence conversion test passed")

    def test_full_conversion(self):
        """Test full conversion pipeline"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0, 0, 1, 2])
        graph = dgl.graph((src, dst))

        # Full conversion
        result = self.converter.convert(
            graph,
            node_descriptions={i: f"node_{i}" for i in range(4)},
            generate_sequences=True
        )

        self.assertIn('mcteg', result)
        self.assertIn('token_sequences', result)
        self.assertEqual(result['num_nodes'], graph.num_nodes())
        print("✓ Full conversion pipeline test passed")

    def test_positional_coordinates(self):
        """Test positional coordinates extraction"""
        # Create a simple graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))

        # Add positional encodings
        mcteg = self.converter.construct_mcteg(graph)

        # Get coordinates
        coords = self.converter.get_node_positional_coords(mcteg)

        self.assertEqual(coords.shape[0], graph.num_nodes())
        self.assertGreater(coords.shape[1], 0)
        print("✓ Positional coordinates test passed")


def run_topological_converter_tests():
    """Run all topological converter tests"""
    print("\n" + "="*70)
    print("Running Topological Converter Tests")
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPositionalEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphWordTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTopologicalConverter))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_topological_converter_tests()
    exit(0 if success else 1)

