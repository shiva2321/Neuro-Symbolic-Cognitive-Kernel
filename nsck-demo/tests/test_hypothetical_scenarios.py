import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_engine import create_cognitive_engine
import hypervec_shim as hypervec_rs

class TestHypotheticalScenarios(unittest.TestCase):
    def setUp(self):
        self.engine = create_cognitive_engine()
        self.engine.episodic_memory = MagicMock()
        self.engine.world_model = MagicMock()
        
        # Mock anchor
        mock_anchor = MagicMock()
        mock_anchor.situation_hv = hypervec_rs.HyperVector(123)
        mock_anchor.state = {"head": (5,5)}
        mock_anchor.image = np.zeros((10, 10))
        self.engine.episodic_memory.sample.return_value = [mock_anchor]

    def test_scenario_discovery(self):
        """Verify that hypothetical scenarios (e.g. death) are discovered and packaged."""
        
        # Mock a path that leads to death (-1.0)
        # Step: source_bits, action_hv, reward, target_bits
        mock_path = [
            {
                "source_bits": np.zeros(10240),
                "action_hv": hypervec_rs.HyperVector(30), # ACTION_LEFT 
                "reward": -1.0, 
                "target_bits": np.zeros(10240)
            }
        ]
        self.engine.world_model.sample_hypothetical_trajectories.return_value = [mock_path]
        
        # We need to mock _get_action_name to return something valid
        self.engine._get_action_name = MagicMock(return_value="LEFT")
        
        lessons = self.engine.generate_hypothetical_lessons(num_anchors=1)
        
        self.assertEqual(len(lessons), 1)
        lesson = lessons[0]
        self.assertTrue(lesson["is_hypothetical"])
        self.assertEqual(lesson["action"], "LEFT")
        self.assertEqual(lesson["reward"], -1.0)
        self.assertEqual(lesson["task_tag"], "snake")
        
        print(f"\n[HYPO TEST] Discovered Scenario: Action {lesson['action']} leads to reward {lesson['reward']}")

if __name__ == "__main__":
    unittest.main()
