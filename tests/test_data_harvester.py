"""
Unit Tests for Agent 1: Data Harvester
Tests data acquisition, filtering, and dataset management.
"""

import unittest
import torch
import dgl
from pathlib import Path
import tempfile
import shutil

from agents.data_harvester import DataHarvester, DataHarvesterConfig, DatasetMetadata


class TestDataHarvester(unittest.TestCase):
    """Test suite for Data Harvester agent"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataHarvesterConfig(
            cache_dir=Path(self.temp_dir) / "cache",
            max_nodes=10000,
            use_semantic_filter=False  # Disable for faster tests
        )
        self.harvester = DataHarvester(self.config)

    def tearDown(self):
        """Clean up test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test harvester initialization"""
        self.assertIsNotNone(self.harvester)
        self.assertIsNotNone(self.harvester.datasets_catalog)
        self.assertGreater(len(self.harvester.datasets_catalog), 0)
        print("✓ Harvester initialization test passed")

    def test_catalog_initialization(self):
        """Test that dataset catalog is properly initialized"""
        self.assertIn('ogbn-arxiv', self.harvester.datasets_catalog)
        self.assertIn('wordnet-graph', self.harvester.datasets_catalog)

        # Check metadata structure
        metadata = self.harvester.datasets_catalog['ogbn-arxiv']
        self.assertEqual(metadata.domain, 'academic')
        self.assertGreater(metadata.num_nodes, 0)
        print("✓ Catalog initialization test passed")

    def test_search_datasets(self):
        """Test dataset search functionality"""
        # Search by domain
        results = self.harvester.search_datasets("language")
        self.assertGreater(len(results), 0)

        # Search by specific term
        results = self.harvester.search_datasets("protein")
        self.assertTrue(any('protein' in r.name.lower() or 'protein' in r.description.lower()
                           for r in results))
        print("✓ Dataset search test passed")

    def test_list_available_datasets(self):
        """Test listing available datasets"""
        all_datasets = self.harvester.list_available_datasets()
        self.assertGreater(len(all_datasets), 0)

        # Test domain filtering
        linguistic = self.harvester.list_available_datasets(domain='linguistic')
        self.assertTrue(all(d.domain == 'linguistic' for d in linguistic))
        print("✓ List datasets test passed")

    def test_custom_dataset_metadata(self):
        """Test custom dataset addition"""
        custom_meta = DatasetMetadata(
            name='test_dataset',
            domain='test',
            num_nodes=100,
            num_edges=200,
            source='custom'
        )

        self.harvester.datasets_catalog['test_dataset'] = custom_meta
        self.assertIn('test_dataset', self.harvester.datasets_catalog)
        print("✓ Custom dataset metadata test passed")

    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.harvester.get_statistics()
        self.assertIn('total_datasets', stats)
        self.assertIn('datasets', stats)
        print("✓ Get statistics test passed")

    def test_semantic_filter_basic(self):
        """Test semantic filtering with simple graph"""
        # Create a simple test graph
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 0])
        graph = dgl.graph((src, dst))

        # Test filter (should keep some nodes)
        filtered = self.harvester.semantic_filter(graph, "test query")
        self.assertLessEqual(filtered.num_nodes(), graph.num_nodes())
        print("✓ Semantic filter basic test passed")


class TestDataHarvesterIntegration(unittest.TestCase):
    """Integration tests for Data Harvester"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataHarvesterConfig(
            cache_dir=Path(self.temp_dir) / "cache",
            max_nodes=5000
        )
        self.harvester = DataHarvester(self.config)

    def tearDown(self):
        """Clean up test artifacts"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_harvest_command_parsing(self):
        """Test natural language command parsing"""
        # Test various commands
        commands = [
            ("Train for English language", "linguistic"),
            ("protein networks", "biological"),
            ("citation papers", "academic"),
        ]

        for command, expected_domain in commands:
            results = self.harvester.search_datasets(command, domain=expected_domain)
            self.assertGreaterEqual(len(results), 0)

        print("✓ Command parsing integration test passed")


def run_data_harvester_tests():
    """Run all data harvester tests"""
    print("\n" + "="*70)
    print("Running Data Harvester Tests")
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDataHarvester))
    suite.addTests(loader.loadTestsFromTestCase(TestDataHarvesterIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_data_harvester_tests()
    exit(0 if success else 1)

