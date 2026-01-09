"""
Comprehensive Unit Tests for Linguistic Graph Module
Tests Phase 1: Linguistic Graph Substrate
"""

import unittest
import torch
import dgl
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ncgn.linguistic_graph import (
    LinguisticGraph, GraphBuilder, NodeFeatures, EdgeFeatures
)


class TestNodeFeatures(unittest.TestCase):
    """Test NodeFeatures dataclass"""

    def test_node_features_creation(self):
        """Test creating node features"""
        embedding = np.random.randn(768)
        node = NodeFeatures(
            node_id=0,
            token="test",
            embedding=embedding,
            frequency=5
        )

        self.assertEqual(node.node_id, 0)
        self.assertEqual(node.token, "test")
        self.assertEqual(node.frequency, 5)
        self.assertEqual(node.embedding.shape, (768,))
        print("✓ NodeFeatures creation test passed")

    def test_node_features_with_metadata(self):
        """Test node features with metadata"""
        node = NodeFeatures(
            node_id=1,
            token="neural",
            embedding=np.zeros(768),
            metadata={'pos': 'noun', 'domain': 'tech'}
        )

        self.assertIn('pos', node.metadata)
        self.assertEqual(node.metadata['pos'], 'noun')
        print("✓ NodeFeatures metadata test passed")


class TestEdgeFeatures(unittest.TestCase):
    """Test EdgeFeatures dataclass"""

    def test_edge_features_creation(self):
        """Test creating edge features"""
        edge = EdgeFeatures(
            source=0,
            target=1,
            pmi_score=2.5,
            co_occurrence=10,
            weight=0.8
        )

        self.assertEqual(edge.source, 0)
        self.assertEqual(edge.target, 1)
        self.assertEqual(edge.pmi_score, 2.5)
        self.assertEqual(edge.co_occurrence, 10)
        self.assertEqual(edge.weight, 0.8)
        print("✓ EdgeFeatures creation test passed")


class TestLinguisticGraph(unittest.TestCase):
    """Test LinguisticGraph class"""

    def setUp(self):
        """Set up test fixtures"""
        self.graph = LinguisticGraph(device='cpu')

    def test_initialization(self):
        """Test graph initialization"""
        self.assertIsNotNone(self.graph)
        self.assertEqual(self.graph.unique_tokens, 0)
        self.assertEqual(self.graph.total_tokens, 0)
        self.assertEqual(len(self.graph.vocab), 0)
        print("✓ Graph initialization test passed")

    def test_add_node(self):
        """Test adding nodes to graph"""
        embedding = np.random.randn(768)
        node_id = self.graph.add_node("test", embedding)

        self.assertEqual(node_id, 0)
        self.assertEqual(self.graph.unique_tokens, 1)
        self.assertIn("test", self.graph.vocab)
        self.assertEqual(self.graph.vocab["test"], 0)
        print("✓ Add node test passed")

    def test_add_duplicate_node(self):
        """Test adding duplicate nodes (should increment frequency)"""
        embedding = np.random.randn(768)
        node_id1 = self.graph.add_node("test", embedding)
        node_id2 = self.graph.add_node("test", embedding)

        self.assertEqual(node_id1, node_id2)
        self.assertEqual(self.graph.unique_tokens, 1)
        self.assertEqual(self.graph.node_features[node_id1].frequency, 2)
        print("✓ Duplicate node test passed")

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes"""
        tokens = ["neural", "network", "learning"]
        for token in tokens:
            embedding = np.random.randn(768)
            self.graph.add_node(token, embedding)

        self.assertEqual(self.graph.unique_tokens, 3)
        self.assertEqual(len(self.graph.vocab), 3)
        for token in tokens:
            self.assertIn(token, self.graph.vocab)
        print("✓ Multiple nodes test passed")

    def test_add_edge(self):
        """Test adding edges"""
        # Add nodes first
        emb1 = np.random.randn(768)
        emb2 = np.random.randn(768)
        self.graph.add_node("source", emb1)
        self.graph.add_node("target", emb2)

        # Add edge
        self.graph.add_edge("source", "target", co_occurrence=5)

        self.assertEqual(len(self.graph.edge_list), 1)
        edge = self.graph.edge_list[0]
        self.assertEqual(edge.source, 0)
        self.assertEqual(edge.target, 1)
        self.assertEqual(edge.co_occurrence, 5)
        print("✓ Add edge test passed")

    def test_add_edge_nonexistent_nodes(self):
        """Test adding edge between nonexistent nodes"""
        # Should not raise error, but should log warning
        self.graph.add_edge("fake1", "fake2")
        self.assertEqual(len(self.graph.edge_list), 0)
        print("✓ Nonexistent edge test passed")

    def test_reverse_vocab(self):
        """Test reverse vocabulary lookup"""
        embedding = np.random.randn(768)
        node_id = self.graph.add_node("test", embedding)

        self.assertEqual(self.graph.reverse_vocab[node_id], "test")
        print("✓ Reverse vocab test passed")


class TestGraphBuilder(unittest.TestCase):
    """Test GraphBuilder class"""

    def setUp(self):
        """Set up test fixtures"""
        # Use CPU and mock embedding for speed
        self.builder = GraphBuilder(
            embedding_model="roberta-base",
            device='cpu'
        )

    def test_initialization(self):
        """Test builder initialization"""
        self.assertIsNotNone(self.builder)
        self.assertIsNotNone(self.builder.embedder)
        print("✓ Builder initialization test passed")

    def test_tokenize(self):
        """Test text tokenization"""
        text = "Neural networks learn from data"
        tokens = self.builder._tokenize(text)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn("neural", [t.lower() for t in tokens])
        print("✓ Tokenization test passed")

    def test_build_from_corpus_small(self):
        """Test building graph from small corpus"""
        corpus = [
            "Neural networks learn",
            "Networks process data",
            "Learning from examples"
        ]

        graph = self.builder.build_from_corpus(
            corpus=corpus,
            window_size=3,
            min_frequency=1
        )

        self.assertIsNotNone(graph)
        self.assertGreater(graph.unique_tokens, 0)
        self.assertIsInstance(graph, LinguisticGraph)
        print("✓ Build from corpus test passed")

    def test_window_size_effect(self):
        """Test that window size affects edge creation"""
        corpus = ["word1 word2 word3 word4 word5"]

        # Small window
        graph1 = self.builder.build_from_corpus(corpus, window_size=2)
        edges1 = len(graph1.edge_list)

        # Larger window
        graph2 = self.builder.build_from_corpus(corpus, window_size=4)
        edges2 = len(graph2.edge_list)

        # Larger window should create more edges
        self.assertGreaterEqual(edges2, edges1)
        print("✓ Window size effect test passed")

    def test_min_frequency_filtering(self):
        """Test frequency-based filtering"""
        corpus = [
            "common common common rare",
            "common common another"
        ]

        # No filtering
        graph1 = self.builder.build_from_corpus(corpus, min_frequency=1)

        # Filter rare words
        graph2 = self.builder.build_from_corpus(corpus, min_frequency=2)

        self.assertLess(graph2.unique_tokens, graph1.unique_tokens)
        print("✓ Frequency filtering test passed")


class TestGraphOperations(unittest.TestCase):
    """Test graph operations and algorithms"""

    def setUp(self):
        """Set up test graph"""
        self.graph = LinguisticGraph(device='cpu')

        # Create simple graph
        tokens = ["a", "b", "c", "d"]
        for token in tokens:
            emb = np.random.randn(768)
            self.graph.add_node(token, emb)

        # Add edges
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")]
        for src, tgt in edges:
            self.graph.add_edge(src, tgt, co_occurrence=1)

    def test_get_statistics(self):
        """Test graph statistics"""
        stats = self.graph.get_statistics()

        self.assertIn('num_nodes', stats)
        self.assertIn('num_edges', stats)
        self.assertEqual(stats['num_nodes'], 4)
        self.assertEqual(stats['num_edges'], 4)
        print("✓ Statistics test passed")

    def test_compute_pmi(self):
        """Test PMI computation"""
        self.graph.compute_pmi()

        # Check that PMI scores were computed
        for edge in self.graph.edge_list:
            self.assertIsNotNone(edge.pmi_score)
        print("✓ PMI computation test passed")

    def test_build_dgl_graph(self):
        """Test DGL graph construction"""
        self.graph.compute_pmi()
        dgl_graph = self.graph.build_dgl_graph()

        self.assertIsInstance(dgl_graph, dgl.DGLGraph)
        self.assertEqual(dgl_graph.num_nodes(), 4)
        self.assertGreater(dgl_graph.num_edges(), 0)

        # Check node features exist
        self.assertIn('feat', dgl_graph.ndata)
        print("✓ DGL graph construction test passed")


class TestGraphSaveLoad(unittest.TestCase):
    """Test graph serialization"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.graph = LinguisticGraph(device='cpu')

        # Build simple graph
        tokens = ["test1", "test2", "test3"]
        for token in tokens:
            emb = np.random.randn(768)
            self.graph.add_node(token, emb)

        self.graph.add_edge("test1", "test2")
        self.graph.add_edge("test2", "test3")

    def tearDown(self):
        """Clean up"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_save_graph(self):
        """Test saving graph"""
        save_path = Path(self.temp_dir) / "test_graph"
        self.graph.save(save_path)

        self.assertTrue(save_path.exists())
        print("✓ Save graph test passed")

    def test_load_graph(self):
        """Test loading graph"""
        save_path = Path(self.temp_dir) / "test_graph"
        self.graph.save(save_path)

        # Load graph
        loaded_graph = LinguisticGraph.load(save_path)

        self.assertEqual(loaded_graph.unique_tokens, self.graph.unique_tokens)
        self.assertEqual(len(loaded_graph.vocab), len(self.graph.vocab))
        self.assertEqual(len(loaded_graph.edge_list), len(self.graph.edge_list))
        print("✓ Load graph test passed")

    def test_save_load_preserves_data(self):
        """Test that save/load preserves all data"""
        save_path = Path(self.temp_dir) / "test_graph"
        self.graph.save(save_path)
        loaded_graph = LinguisticGraph.load(save_path)

        # Check vocab
        self.assertEqual(loaded_graph.vocab, self.graph.vocab)

        # Check node features
        for node_id in self.graph.node_features:
            orig = self.graph.node_features[node_id]
            loaded = loaded_graph.node_features[node_id]
            self.assertEqual(orig.token, loaded.token)
            self.assertEqual(orig.frequency, loaded.frequency)

        print("✓ Data preservation test passed")


def run_linguistic_graph_tests():
    """Run all linguistic graph tests"""
    print("\n" + "=" * 80)
    print("LINGUISTIC GRAPH TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNodeFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestLinguisticGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphSaveLoad))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_linguistic_graph_tests()
    exit(0 if success else 1)

