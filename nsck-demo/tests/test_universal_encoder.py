import torch
import unittest
import sys
import os

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from universal_encoder import UniversalEncoder

class TestUniversalEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = UniversalEncoder()

    def test_visual_input(self):
        """Test with 4D visual tensor (B, C, H, W)"""
        # Batch=1, Channels=4, Height=10, Width=10 (Standard Snake)
        x = torch.randn(1, 4, 10, 10)
        out = self.encoder(x)
        print(f"Visual (10x10) Output: {out.shape}")
        self.assertEqual(out.shape, (1, 256))

    def test_variable_grid_size(self):
        """Test with abnormal grid size (20x20)"""
        x = torch.randn(1, 4, 20, 20)
        out = self.encoder(x)
        print(f"Visual (20x20) Output: {out.shape}")
        self.assertEqual(out.shape, (1, 256))

    def test_temporal_input(self):
        """Test with 3D temporal tensor (B, C, T)"""
        # Batch=1, Channel=1, Time=100 (Audio Waveform)
        x = torch.randn(1, 1, 100)
        out = self.encoder(x)
        print(f"Temporal (1x100) Output: {out.shape}")
        self.assertEqual(out.shape, (1, 256))

    def test_vector_input(self):
        """Test with 2D vector tensor (B, Dim)"""
        # Batch=1, Dim=50 (Text Embedding)
        x = torch.randn(1, 50)
        out = self.encoder(x)
        print(f"Vector (50) Output: {out.shape}")
        self.assertEqual(out.shape, (1, 256))

if __name__ == '__main__':
    unittest.main()
