"""
Comprehensive Unit Tests for Graph Embeddings and Structural Encoders
Tests Phase 1: Embedding components
"""

import unittest
import torch
import numpy as np
from pathlib import Path

from ncgn.graph_embeddings import (
    GraphEmbedding, StructuralEncoder
)


class TestGraphEmbedding(unittest.TestCase):
    """Test pre-trained embedding wrapper"""

    def setUp(self):
        """Set up test fixtures"""
        # Use small model for testing
        self.embedder = GraphEmbedding(
            model_name="roberta-base",
            device='cpu'
        )

    def test_initialization(self):
        """Test embedder initialization"""
        self.assertIsNotNone(self.embedder)
        self.assertGreater(self.embedder.embedding_dim, 0)
        print("✓ GraphEmbedding initialization test passed")

    def test_embed_single_token(self):
        """Test embedding single token"""
        token = "neural"
        embedding = self.embedder.embed_single_token(token)

        self.assertEqual(embedding.shape, (self.embedder.embedding_dim,))
        self.assertFalse(np.any(np.isnan(embedding)))
        print("✓ Single token embedding test passed")

    def test_embed_multiple_tokens(self):
        """Test embedding multiple tokens"""
        tokens = ["neural", "network", "learning", "artificial"]
        embeddings = self.embedder.embed_tokens(tokens, batch_size=2)

        self.assertEqual(embeddings.shape, (len(tokens), self.embedder.embedding_dim))
        self.assertFalse(np.any(np.isnan(embeddings)))
        print("✓ Multiple token embedding test passed")

    def test_embedding_consistency(self):
        """Test that same token produces same embedding"""
        token = "test"

        emb1 = self.embedder.embed_single_token(token)
        emb2 = self.embedder.embed_single_token(token)

        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)
        print("✓ Embedding consistency test passed")

    def test_different_tokens_different_embeddings(self):
        """Test that different tokens produce different embeddings"""
        emb1 = self.embedder.embed_single_token("cat")
        emb2 = self.embedder.embed_single_token("dog")

        # Should be different
        diff = np.abs(emb1 - emb2).sum()
        self.assertGreater(diff, 0.1)
        print("✓ Different tokens test passed")

    def test_semantic_similarity(self):
        """Test that semantically similar words have similar embeddings"""
        emb1 = self.embedder.embed_single_token("happy")
        emb2 = self.embedder.embed_single_token("joyful")
        emb3 = self.embedder.embed_single_token("computer")

        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        # happy-joyful should be more similar than happy-computer
        self.assertGreater(sim_12, sim_13)
        print("✓ Semantic similarity test passed")

    def test_batch_processing(self):
        """Test batch processing efficiency"""
        tokens = [f"word_{i}" for i in range(20)]

        # Should handle batches
        embeddings = self.embedder.embed_tokens(tokens, batch_size=5)

        self.assertEqual(embeddings.shape, (20, self.embedder.embedding_dim))
        print("✓ Batch processing test passed")


class TestStructuralEncoder(unittest.TestCase):
    """Test structural positional encoders"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = StructuralEncoder(
            graph_size=10,
            encoding_dim=32,
            use_laplacian=True,
            use_random_walk=True
        )

    def test_initialization(self):
        """Test encoder initialization"""
        self.assertEqual(self.encoder.graph_size, 10)
        self.assertEqual(self.encoder.encoding_dim, 32)
        print("✓ StructuralEncoder initialization test passed")

    def test_laplacian_encoding(self):
        """Test Laplacian positional encoding"""
        # Create simple adjacency matrix
        adjacency = torch.zeros(10, 10)
        for i in range(9):
            adjacency[i, i+1] = 1
            adjacency[i+1, i] = 1

        lap_encoding = self.encoder.compute_laplacian_pe(adjacency, num_eigenvectors=8)

        self.assertEqual(lap_encoding.shape, (10, 8))
        self.assertFalse(torch.any(torch.isnan(lap_encoding)))
        print("✓ Laplacian encoding test passed")

    def test_random_walk_encoding(self):
        """Test random walk positional encoding"""
        adjacency = torch.rand(10, 10)
        adjacency = (adjacency > 0.7).float()
        adjacency = adjacency + adjacency.t()  # Symmetric

        rw_encoding = self.encoder.compute_random_walk_pe(
            adjacency,
            walk_length=20,
            embedding_dim=16
        )

        self.assertEqual(rw_encoding.shape, (10, 16))
        print("✓ Random walk encoding test passed")

    def test_distance_encoding(self):
        """Test distance-based encoding"""
        adjacency = torch.zeros(5, 5)
        # Chain: 0-1-2-3-4
        for i in range(4):
            adjacency[i, i+1] = 1
            adjacency[i+1, i] = 1

        dist_encoding = self.encoder.compute_distance_encoding(adjacency)

        self.assertEqual(dist_encoding.shape[0], 5)
        print("✓ Distance encoding test passed")

    def test_get_all_encodings(self):
        """Test getting combined encodings"""
        all_encodings = self.encoder.get_all_encodings()

        self.assertEqual(all_encodings.shape, (10, 32))
        print("✓ Combined encodings test passed")

    def test_encoding_uniqueness(self):
        """Test that different positions get different encodings"""
        encodings = self.encoder.get_all_encodings()

        # Each node should have unique encoding
        for i in range(10):
            for j in range(i+1, 10):
                diff = torch.abs(encodings[i] - encodings[j]).sum()
                self.assertGreater(diff.item(), 0.01)

        print("✓ Encoding uniqueness test passed")


class TestEmbeddingIntegration(unittest.TestCase):
    """Test integration between embedding components"""

    def test_embeddings_with_positional_encoding(self):
        """Test combining semantic and positional encodings"""
        # Semantic embeddings
        embedder = GraphEmbedding("roberta-base", device='cpu')
        tokens = ["neural", "network", "graph"]
        semantic_emb = embedder.embed_tokens(tokens)

        # Positional encodings
        struct_encoder = StructuralEncoder(
            graph_size=3,
            encoding_dim=embedder.embedding_dim,
            use_laplacian=False,
            use_random_walk=False
        )

        # Simple position encoding
        positional_emb = torch.randn(3, embedder.embedding_dim)

        # Combine
        combined = torch.from_numpy(semantic_emb) + positional_emb

        self.assertEqual(combined.shape, (3, embedder.embedding_dim))
        print("✓ Embedding combination test passed")

    def test_embedding_pipeline(self):
        """Test complete embedding pipeline"""
        # Step 1: Tokenize and embed
        embedder = GraphEmbedding("roberta-base", device='cpu')
        tokens = ["deep", "learning", "neural", "networks"]
        embeddings = embedder.embed_tokens(tokens)

        # Step 2: Add structural information
        struct_encoder = StructuralEncoder(
            graph_size=len(tokens),
            encoding_dim=embeddings.shape[1]
        )

        # Create adjacency (sequential connections)
        adjacency = torch.zeros(len(tokens), len(tokens))
        for i in range(len(tokens)-1):
            adjacency[i, i+1] = 1
            adjacency[i+1, i] = 1

        # Get positional encodings
        pos_enc = struct_encoder.compute_laplacian_pe(adjacency, num_eigenvectors=8)

        # Combine (project if needed)
        if pos_enc.shape[1] != embeddings.shape[1]:
            projection = torch.nn.Linear(pos_enc.shape[1], embeddings.shape[1])
            pos_enc = projection(pos_enc)

        combined = torch.from_numpy(embeddings).float() + pos_enc

        self.assertEqual(combined.shape[0], len(tokens))
        print("✓ Complete embedding pipeline test passed")


class TestEmbeddingPerformance(unittest.TestCase):
    """Test embedding performance and efficiency"""

    def test_large_batch_embedding(self):
        """Test embedding large batch of tokens"""
        embedder = GraphEmbedding("roberta-base", device='cpu')

        # Large batch
        tokens = [f"token_{i}" for i in range(100)]

        import time
        start = time.time()
        embeddings = embedder.embed_tokens(tokens, batch_size=16)
        duration = time.time() - start

        self.assertEqual(embeddings.shape, (100, embedder.embedding_dim))
        print(f"✓ Large batch embedding test passed ({duration:.2f}s)")

    def test_encoding_computation_speed(self):
        """Test structural encoding speed"""
        encoder = StructuralEncoder(
            graph_size=50,
            encoding_dim=64,
            use_laplacian=True,
            use_random_walk=False
        )

        adjacency = torch.rand(50, 50)
        adjacency = (adjacency > 0.9).float()

        import time
        start = time.time()
        lap_enc = encoder.compute_laplacian_pe(adjacency, num_eigenvectors=16)
        duration = time.time() - start

        self.assertEqual(lap_enc.shape, (50, 16))
        print(f"✓ Encoding speed test passed ({duration:.3f}s)")


def run_embedding_tests():
    """Run all embedding tests"""
    print("\n" + "=" * 80)
    print("GRAPH EMBEDDINGS TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGraphEmbedding))
    suite.addTests(loader.loadTestsFromTestCase(TestStructuralEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_embedding_tests()
    exit(0 if success else 1)

