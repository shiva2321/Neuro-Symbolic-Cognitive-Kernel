import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Add project root to path

from python.core.reasoning.cognitive_engine import create_cognitive_engine, CognitiveState
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.perception.symbol_grounding import GLOBAL_PRIMITIVES_MAP

class TestDreaming(unittest.TestCase):
    def setUp(self):
        self.engine = create_cognitive_engine()
        # Mock Episodic Memory to return dummy episodes
        self.engine.episodic_memory = MagicMock()
        
        # Create a dummy episode
        dummy_hv = hypervec_rs.HyperVector(123)
        dummy_state = {"head": (1,1), "food": (1,0)} # Food is ABOVE
        
        mock_episode = MagicMock()
        mock_episode.situation_hv = dummy_hv
        mock_episode.state = dummy_state
        mock_episode.task_tag = "snake"
        
        self.engine.episodic_memory.sample.return_value = [mock_episode]
        
    def test_dream_generation(self):
        """Verify that dream() produces synthetic experiences with imagined outcomes."""
        
        # We need to mock imagine_rollout to return a specific reward for ACTION_UP
        # In this dummy state, Food is at (1,0) and Head is at (1,1) -> UP is good.
        
        def mock_imagine(hv, actions, task):
            if actions == ["ACTION_UP"]:
                return 1.0 # Success
            return -0.1 # Neutral/Meh
            
        self.engine.imagine_rollout = MagicMock(side_effect=mock_imagine)
        
        dreams = self.engine.dream(num_samples=1)
        
        self.assertEqual(len(dreams), 1)
        dream = dreams[0]
        
        self.assertTrue(dream["is_dream"])
        self.assertEqual(dream["action"], "UP") # Best action should be UP
        self.assertEqual(dream["reward"], 1.0)
        self.assertEqual(dream["task_tag"], "snake")
        
        print(f"\n[DREAM TEST] Generated Dream: {dream['action']} with reward {dream['reward']}")

if __name__ == "__main__":
    unittest.main()
