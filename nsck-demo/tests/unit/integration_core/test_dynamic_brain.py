import torch
import unittest
import sys
import os

from python.core.neural.snn_qat import TaskAwareSNN

class TestDynamicBrain(unittest.TestCase):
    def setUp(self):
        self.brain = TaskAwareSNN()

    def test_dynamic_growth(self):
        """Test if brain grows new heads"""
        print("\n--- Testing Dynamic Growth ---")
        self.brain.register_task("alien_invaders", 6)
        
        self.assertTrue("alien_invaders" in self.brain.heads)
        # Access ModuleDict properly
        self.assertTrue(hasattr(self.brain.heads["alien_invaders"], "actor"))
        
        # Check shapes
        # Actor: 128 -> 6 (corrected from 256)
        actor_head = self.brain.heads["alien_invaders"].actor
        self.assertEqual(actor_head.out_features, 6)
        # Critic: 128 -> 1
        critic_head = self.brain.heads["alien_invaders"].critic
        self.assertEqual(critic_head.out_features, 1)
        print("Success: Brain grew new lobes for Alien Invaders (6 actions).")

    def test_multimodal_forward(self):
        """Test forward pass with different modalities"""
        print("\n--- Testing Multi-Modal Thought ---")
        self.brain.register_task("snake", 4)
        
        # 1. Visual Input (Snake)
        x_vis = torch.randn(1, 4, 10, 10)
        logits, value = self.brain(x_vis, "snake")
        
        print(f"Visual Action: {logits.shape}, Value: {value.shape}")
        self.assertEqual(logits.shape, (1, 4)) # Batch=1, Actions=4
        self.assertEqual(value.shape, (1, 1))  # Batch=1, Value=1

        # 2. Text Input (Chat) - Auto register
        x_text = torch.randn(1, 128) # Embedding
        # Should auto-register "chat" with default 4 actions if not exists (checked in code)
        self.brain.register_task("chat", 100)
        logit_txt, val_txt = self.brain(x_text, "chat")
        
        print(f"Text Action: {logit_txt.shape}, Value: {val_txt.shape}")
        self.assertEqual(logit_txt.shape, (1, 100))

    def test_pruning(self):
        """Test if brain can forget"""
        print("\n--- Testing Pruning (Forgetting) ---")
        self.brain.register_task("temp_task", 10)
        self.assertTrue("temp_task" in self.brain.heads)
        
        self.brain.forget_task("temp_task")
        self.assertFalse("temp_task" in self.brain.heads)
        print("Success: Brain successfully forgot useless task.")

if __name__ == '__main__':
    unittest.main()
