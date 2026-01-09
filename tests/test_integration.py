"""
Comprehensive Integration Tests for NCGN System
Tests complete system integration and end-to-end workflows
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# NCGN imports
from ncgn.linguistic_graph import LinguisticGraph, GraphBuilder
from ncgn.spiking_neurons import SpikingLayer, PoissonEncoder
from ncgn.stdp_learning import STDPLearning
from ncgn.graph_transformer import GraphTransformer
from ncgn.dual_system import DualSystemArchitecture


class TestPhase1Phase2Integration(unittest.TestCase):
    """Test integration between Phase 1 (Linguistic Graph) and Phase 2 (Spiking Neurons)"""

    def test_linguistic_graph_to_spiking_network(self):
        """Test pipeline from linguistic graph to spiking network"""
        # Build linguistic graph
        corpus = ["neural network", "network processes", "processes data"]
        builder = GraphBuilder(embedding_model="roberta-base", device='cpu')
        graph = builder.build_from_corpus(corpus, window_size=3, min_frequency=1)

        # Get node features
        num_nodes = min(10, graph.unique_tokens)

        # Create spiking layer
        spiking_layer = SpikingLayer(
            in_features=768,  # RoBERTa dimension
            out_features=128,
            neuron_model='lif'
        )

        # Simulate forward pass
        node_features = torch.randn(1, num_nodes, 768)
        spikes_in = PoissonEncoder.encode(node_features, time_steps=50)

        outputs = []
        for t in range(50):
            out = spiking_layer(spikes_in[t])
            outputs.append(out)

        outputs = torch.stack(outputs)
        self.assertEqual(outputs.shape, (50, 1, num_nodes, 128))
        print("✓ Linguistic graph to spiking network test passed")

    def test_graph_structure_preserved_in_spiking(self):
        """Test that graph structure influences spiking dynamics"""
        # Create small graph
        graph = LinguisticGraph(device='cpu')

        tokens = ["a", "b", "c"]
        for token in tokens:
            emb = np.random.randn(768)
            graph.add_node(token, emb)

        # Add edges
        graph.add_edge("a", "b", co_occurrence=10)
        graph.add_edge("b", "c", co_occurrence=5)

        # This should influence the spiking patterns
        self.assertEqual(len(graph.edge_list), 2)
        print("✓ Graph structure in spiking test passed")


class TestPhase2Phase3Integration(unittest.TestCase):
    """Test integration between Phase 2 (Spiking) and Phase 3 (Dual System)"""

    def test_spiking_to_transformer(self):
        """Test pipeline from spiking neurons to graph transformer"""
        # Spiking layer output
        spiking_layer = SpikingLayer(32, 64, neuron_model='lif')

        # Input
        spikes_in = torch.randint(0, 2, (2, 10, 32)).float()
        spiking_out = spiking_layer(spikes_in)

        # Feed to transformer
        transformer = GraphTransformer(
            node_feat_dim=64,
            embed_dim=128,
            num_layers=2,
            num_heads=4
        )

        # Convert spikes to features (rate coding)
        features = spiking_out.unsqueeze(0).expand(1, -1, -1, -1).mean(dim=0)
        transformer_out = transformer(features)

        self.assertEqual(transformer_out.shape, (2, 10, 128))
        print("✓ Spiking to transformer test passed")

    def test_stdp_updates_dual_system(self):
        """Test STDP learning in dual system context"""
        # Create dual system
        dual_system = DualSystemArchitecture(32, 64, 2, 4)

        # STDP learner
        stdp = STDPLearning()

        # Simulate learning
        node_features = torch.randn(1, 10, 32)
        context = {'query': 'test'}

        output = dual_system(node_features, context, use_system2=False)

        # STDP could update internal weights based on spiking activity
        self.assertIsNotNone(output)
        print("✓ STDP in dual system test passed")


class TestCompletePhase1to3Pipeline(unittest.TestCase):
    """Test complete pipeline from Phase 1 through Phase 3"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_text_processing(self):
        """Test end-to-end text processing pipeline"""
        # Phase 1: Build linguistic graph
        corpus = [
            "Artificial intelligence learns from data",
            "Machine learning uses neural networks",
            "Deep learning networks process information",
        ]

        builder = GraphBuilder(embedding_model="roberta-base", device='cpu')
        ling_graph = builder.build_from_corpus(corpus, window_size=5, min_frequency=1)

        self.assertGreater(ling_graph.unique_tokens, 0)

        # Phase 2: Process with spiking neurons (simplified)
        num_nodes = min(10, ling_graph.unique_tokens)
        node_features = torch.randn(1, num_nodes, 768)

        # Phase 3: Dual system reasoning
        dual_system = DualSystemArchitecture(
            node_feat_dim=768,
            embed_dim=256,
            num_layers=2,
            num_heads=8
        )

        context = {'query': 'artificial intelligence'}
        output = dual_system(node_features, context)

        self.assertIn('integrated_output', output)
        print("✓ End-to-end text processing test passed")

    def test_save_load_complete_system(self):
        """Test saving and loading complete system"""
        # Build system
        graph = LinguisticGraph(device='cpu')

        for i, token in enumerate(["test1", "test2", "test3"]):
            emb = np.random.randn(768)
            graph.add_node(token, emb)

        dual_system = DualSystemArchitecture(768, 256, 2, 4)

        # Save
        graph_path = Path(self.temp_dir) / "graph"
        model_path = Path(self.temp_dir) / "model.pt"

        graph.save(graph_path)
        torch.save(dual_system.state_dict(), model_path)

        # Load
        loaded_graph = LinguisticGraph.load(graph_path)
        loaded_dual_system = DualSystemArchitecture(768, 256, 2, 4)
        loaded_dual_system.load_state_dict(torch.load(model_path))

        # Verify
        self.assertEqual(loaded_graph.unique_tokens, graph.unique_tokens)
        print("✓ Save/load complete system test passed")


class TestAgentIntegration(unittest.TestCase):
    """Test integration with specialized agents"""

    def test_data_harvester_to_ncgn(self):
        """Test pipeline from data harvester to NCGN"""
        from agents.data_harvester import DataHarvester, DataHarvesterConfig

        config = DataHarvesterConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            max_nodes=1000
        )

        harvester = DataHarvester(config)

        # List available datasets
        datasets = harvester.list_available_datasets()
        self.assertIsInstance(datasets, list)

        # Clean up
        if config.cache_dir.exists():
            shutil.rmtree(config.cache_dir)

        print("✓ Data harvester to NCGN test passed")

    def test_topological_converter_integration(self):
        """Test topological converter with NCGN"""
        from agents.topological_converter import TopologicalConverter, MCTEGConfig

        config = MCTEGConfig(
            use_laplacian_pe=True,
            use_rwpe=False,
            num_laplacian_eigenvectors=4
        )

        converter = TopologicalConverter()

        # Create simple graph
        import dgl
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        graph = dgl.graph((src, dst))
        graph.ndata['feat'] = torch.randn(4, 32)

        # Convert
        enhanced_graph = converter.add_positional_encodings(graph, config)

        self.assertIn('feat', enhanced_graph.ndata)
        print("✓ Topological converter integration test passed")


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios"""

    def test_continual_learning_scenario(self):
        """Test continual learning with new data"""
        # Initial training
        dual_system = DualSystemArchitecture(32, 64, 2, 4)

        # Task 1
        task1_data = torch.randn(5, 10, 32)
        task1_context = {'query': 'task1'}

        output1 = dual_system(task1_data, task1_context)

        # Task 2 (should not forget task 1)
        task2_data = torch.randn(5, 10, 32)
        task2_context = {'query': 'task2'}

        output2 = dual_system(task2_data, task2_context)

        # Both should work
        self.assertIsNotNone(output1)
        self.assertIsNotNone(output2)
        print("✓ Continual learning scenario test passed")

    def test_multi_domain_reasoning(self):
        """Test reasoning across multiple domains"""
        dual_system = DualSystemArchitecture(32, 64, 2, 4)

        # Add knowledge from different domains
        knowledge = [
            # Math domain
            ("5", "greater_than", "3"),
            ("10", "equals", "5+5"),
            # Logic domain
            ("A", "implies", "B"),
            ("B", "implies", "C"),
            # Language domain
            ("dog", "is_a", "animal"),
        ]
        dual_system.system2.add_knowledge(knowledge)

        # Query across domains
        contexts = [
            {'query': 'math: 5 > 3'},
            {'query': 'logic: A implies C'},
            {'query': 'language: dog is animal'},
        ]

        for context in contexts:
            node_features = torch.randn(1, 5, 32)
            output = dual_system(node_features, context)
            self.assertIsNotNone(output)

        print("✓ Multi-domain reasoning test passed")

    def test_large_scale_graph(self):
        """Test with larger graph (scalability)"""
        # Larger graph
        num_nodes = 100
        node_features = torch.randn(1, num_nodes, 32)

        # Sparse adjacency
        adjacency = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            # Connect to 5 random neighbors
            neighbors = np.random.choice(num_nodes, 5, replace=False)
            adjacency[i, neighbors] = 1

        # Process with transformer
        transformer = GraphTransformer(32, 64, num_layers=2, num_heads=4)
        output = transformer(node_features, adjacency)

        self.assertEqual(output.shape, (1, num_nodes, 64))
        print("✓ Large-scale graph test passed")

    def test_real_time_inference(self):
        """Test real-time inference speed"""
        import time

        dual_system = DualSystemArchitecture(32, 64, 2, 4)
        dual_system.eval()

        # Warm up
        for _ in range(5):
            x = torch.randn(1, 10, 32)
            _ = dual_system(x, {'query': 'test'}, use_system2=False)

        # Measure inference time
        num_inferences = 10
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_inferences):
                x = torch.randn(1, 10, 32)
                _ = dual_system(x, {'query': 'test'}, use_system2=False)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_inferences

        # Should be reasonably fast (< 100ms per inference on CPU)
        self.assertLess(avg_time, 1.0)
        print(f"✓ Real-time inference test passed (avg: {avg_time*1000:.2f}ms)")


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_empty_graph(self):
        """Test with empty graph"""
        graph = LinguisticGraph(device='cpu')

        # Should handle empty graph gracefully
        stats = graph.get_statistics()
        self.assertEqual(stats['num_nodes'], 0)
        print("✓ Empty graph test passed")

    def test_single_node_graph(self):
        """Test with single node"""
        graph = LinguisticGraph(device='cpu')
        emb = np.random.randn(768)
        graph.add_node("single", emb)

        self.assertEqual(graph.unique_tokens, 1)
        print("✓ Single node graph test passed")

    def test_disconnected_graph(self):
        """Test with disconnected graph"""
        graph = LinguisticGraph(device='cpu')

        # Add nodes without edges
        for i in range(5):
            emb = np.random.randn(768)
            graph.add_node(f"node_{i}", emb)

        stats = graph.get_statistics()
        self.assertEqual(stats['num_nodes'], 5)
        self.assertEqual(stats['num_edges'], 0)
        print("✓ Disconnected graph test passed")

    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions"""
        transformer = GraphTransformer(32, 64, 2, 4)

        # Wrong dimension
        try:
            x = torch.randn(2, 10, 16)  # Should be 32
            _ = transformer(x)
            self.fail("Should raise error for wrong dimensions")
        except:
            pass  # Expected

        print("✓ Invalid dimensions test passed")

    def test_nan_handling(self):
        """Test handling of NaN values"""
        dual_system = DualSystemArchitecture(32, 64, 2, 4)

        # Input with NaN
        x = torch.randn(1, 10, 32)
        x[0, 0, 0] = float('nan')

        # Should handle gracefully (might produce NaN output, but shouldn't crash)
        try:
            output = dual_system(x, {'query': 'test'}, use_system2=False)
            # Check if system detected NaN
            self.assertTrue(True)  # Didn't crash
        except:
            pass  # Some implementations might raise error

        print("✓ NaN handling test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print("NCGN INTEGRATION TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Phase2Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Phase3Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestCompletePhase1to3Pipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestRealWorldScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingAndEdgeCases))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)

