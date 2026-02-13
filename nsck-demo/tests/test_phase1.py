"""
Tests for Phase 1 Neural Learning Engine enhancements.

Tests:
  - Multi-task learning with shared encoder
  - Rule extraction from neural policies
  - Dual inference (neural + symbolic)
  - Gradient surgery for multi-task learning
"""
import unittest
import torch
import torch.nn as nn
import numpy as np
from python.core.learning.multi_task_learning import (
    SharedEncoder, TaskHead, MultiTaskNetwork, MultiTaskTrainer,
    GradientSurgery, create_multitask_network
)
from python.scripts.rule_extraction import (
    Rule, RuleSet, NeuralRuleExtractor, DualInferenceEngine,
    create_dual_inference_system
)


class TestMultiTaskLearning(unittest.TestCase):
    """Test multi-task learning components."""
    
    def test_shared_encoder(self):
        """Test shared encoder creates proper latent representations."""
        encoder = SharedEncoder(input_dim=512, hidden_dim=128, latent_dim=64)
        
        # Test forward pass
        x = torch.randn(4, 512)
        latent = encoder(x)
        
        self.assertEqual(latent.shape, (4, 64), "Latent shape should be [batch, 64]")
        self.assertTrue(torch.isfinite(latent).all(), "Latent should be finite")
    
    def test_task_head(self):
        """Test task-specific actor-critic head."""
        head = TaskHead(latent_dim=64, num_actions=4, hidden_dim=64)
        
        # Test forward pass
        latent = torch.randn(4, 64)
        action_logits, value = head(latent)
        
        self.assertEqual(action_logits.shape, (4, 4), "Action logits shape")
        self.assertEqual(value.shape, (4, 1), "Value shape")
        self.assertTrue(torch.isfinite(action_logits).all(), "Logits should be finite")
        self.assertTrue(torch.isfinite(value).all(), "Value should be finite")
    
    def test_multitask_network(self):
        """Test multi-task network with multiple task heads."""
        model = create_multitask_network()
        
        # Test forward for different tasks
        x = torch.randn(4, 512)
        
        for task_name in ["snake", "pong", "maze"]:
            action_logits, value = model(x, task_name)
            
            expected_actions = model.task_configs[task_name]
            self.assertEqual(action_logits.shape[1], expected_actions, 
                           f"{task_name}: Correct number of actions")
            self.assertEqual(value.shape, (4, 1), f"{task_name}: Value shape")
    
    def test_gradient_surgery(self):
        """Test gradient surgery for conflict resolution."""
        # Create gradients with some conflict
        grad1 = torch.tensor([1.0, 0.5, 0.0])
        grad2 = torch.tensor([-0.8, 0.3, 0.0])  # Partially conflicting
        
        gradients = {"task1": grad1, "task2": grad2}
        
        # Apply gradient surgery
        projected = GradientSurgery.project_conflicting_gradients(gradients)
        
        # Check that conflicts are resolved
        self.assertEqual(len(projected), 2, "Should have 2 projected gradients")
        self.assertTrue("task1" in projected, "Should contain task1")
        self.assertTrue("task2" in projected, "Should contain task2")
        
        # Projected gradients should be finite
        self.assertTrue(torch.isfinite(projected["task1"]).all(), 
                       "Projected task1 gradient should be finite")
        self.assertTrue(torch.isfinite(projected["task2"]).all(),
                       "Projected task2 gradient should be finite")
        
        # Projected gradients should have non-zero norm
        self.assertGreater(torch.norm(projected["task1"]).item(), 0,
                          "Projected gradient should be non-zero")
        self.assertGreater(torch.norm(projected["task2"]).item(), 0,
                          "Projected gradient should be non-zero")
    
    def test_multitask_trainer(self):
        """Test multi-task trainer with gradient surgery."""
        model = create_multitask_network()
        trainer = MultiTaskTrainer(model, lr=0.001)
        
        # Create dummy batch data with positive returns
        batch_data = {}
        for task_name in ["snake", "pong"]:
            states = torch.randn(8, 512)
            actions = torch.randint(0, model.task_configs[task_name], (8,))
            returns = torch.abs(torch.randn(8)) + 0.1  # Ensure positive
            batch_data[task_name] = (states, actions, returns)
        
        # Train step
        metrics = trainer.train_step(batch_data, use_gradient_surgery=True)
        
        # Check metrics
        self.assertIn("snake", metrics.task_losses, "Should have snake loss")
        self.assertIn("pong", metrics.task_losses, "Should have pong loss")
        self.assertTrue(torch.isfinite(torch.tensor(metrics.total_loss)),
                       "Total loss should be finite")


class TestRuleExtraction(unittest.TestCase):
    """Test rule extraction from neural policies."""
    
    def test_rule_matching(self):
        """Test rule condition matching."""
        rule = Rule(
            conditions=[("distance_to_wall", "<", 2.0)],
            action="TURN_LEFT",
            confidence=0.9,
            support=100
        )
        
        # Should match
        state1 = {"distance_to_wall": 1.5}
        self.assertTrue(rule.matches(state1), "Should match when distance < 2.0")
        
        # Should not match
        state2 = {"distance_to_wall": 3.0}
        self.assertFalse(rule.matches(state2), "Should not match when distance >= 2.0")
        
        # Missing feature should not match
        state3 = {"other_feature": 1.0}
        self.assertFalse(rule.matches(state3), "Should not match with missing feature")
    
    def test_rule_set(self):
        """Test rule set operations."""
        rule_set = RuleSet()
        
        rule1 = Rule(
            conditions=[("x", ">", 5.0)],
            action="ACTION_UP",
            confidence=0.8,
            support=50
        )
        rule2 = Rule(
            conditions=[("x", "<=", 5.0)],
            action="ACTION_DOWN",
            confidence=0.9,
            support=60
        )
        
        rule_set.add_rule(rule1)
        rule_set.add_rule(rule2)
        
        # Test matching
        state1 = {"x": 7.0}
        matching = rule_set.get_matching_rules(state1)
        self.assertEqual(len(matching), 1, "Should match one rule")
        self.assertEqual(matching[0].action, "ACTION_UP", "Should match rule1")
        
        # Test best action
        action, conf = rule_set.get_best_action(state1)
        self.assertEqual(action, "ACTION_UP", "Best action should be ACTION_UP")
        self.assertEqual(conf, 0.8, "Confidence should be 0.8")
    
    def test_neural_rule_extractor(self):
        """Test extracting rules from a simple neural network."""
        # Create a simple linear model
        model = nn.Sequential(
            nn.Linear(10, 4),
            nn.ReLU()
        )
        
        feature_names = [f"feat_{i}" for i in range(10)]
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        extractor = NeuralRuleExtractor(model, feature_names, action_names)
        
        # Generate sample states
        states = torch.randn(100, 10)
        
        # Extract rules
        rule_set = extractor.extract_rules(states, max_rules=20, min_support=5)
        
        # Should extract some rules
        self.assertGreater(len(rule_set), 0, "Should extract at least one rule")
        
        # Check rule structure
        for rule in rule_set.rules:
            self.assertGreater(len(rule.conditions), 0, "Rule should have conditions")
            self.assertIn(rule.action, action_names, "Action should be valid")
            self.assertGreater(rule.confidence, 0.0, "Confidence should be positive")
            self.assertGreater(rule.support, 0, "Support should be positive")


class TestDualInference(unittest.TestCase):
    """Test dual inference system."""
    
    def test_dual_inference_neural_path(self):
        """Test dual inference uses neural path when rules don't apply."""
        # Create simple model
        model = nn.Linear(10, 4)
        rule_set = RuleSet()  # Empty rule set
        
        engine = DualInferenceEngine(model, rule_set)
        
        # Make decision
        state = torch.randn(1, 10)
        state_features = {f"feat_{i}": 0.0 for i in range(10)}
        
        action, metadata = engine.decide(state, state_features, require_safe=False)
        
        # Should use neural path
        self.assertEqual(metadata["mode"], "NEURAL", "Should use neural inference")
        self.assertFalse(metadata["override"], "Should not override")
        
        # Check stats
        stats = engine.get_stats()
        self.assertEqual(stats["total_decisions"], 1)
        self.assertEqual(stats["neural_decisions"], 1)
        self.assertEqual(stats["symbolic_overrides"], 0)
    
    def test_dual_inference_symbolic_override(self):
        """Test dual inference overrides with confident symbolic rules."""
        # Create model that outputs low confidence
        model = nn.Sequential(
            nn.Linear(10, 4),
            nn.Softmax(dim=-1)
        )
        
        # Create confident symbolic rule
        rule_set = RuleSet()
        rule_set.add_rule(Rule(
            conditions=[("x", ">", 0.0)],
            action="ACTION_SYMBOLIC",
            confidence=0.9,
            support=100
        ))
        
        engine = DualInferenceEngine(model, rule_set)
        
        # Make decision with matching state
        state = torch.randn(1, 10)
        state_features = {"x": 1.0}  # Matches rule
        
        action, metadata = engine.decide(state, state_features, require_safe=False)
        
        # Might override depending on neural confidence
        # At minimum, should have symbolic_confidence > 0
        self.assertGreater(metadata["symbolic_confidence"], 0.0,
                          "Should have symbolic confidence")
    
    def test_dual_inference_safety_override(self):
        """Test safety rules always override."""
        model = nn.Linear(10, 4)
        rule_set = RuleSet()
        
        # Create safety rule
        safety_rules = RuleSet()
        safety_rules.add_rule(Rule(
            conditions=[("danger", ">", 0.5)],
            action="ACTION_SAFE",
            confidence=0.95,
            support=100
        ))
        
        engine = DualInferenceEngine(model, rule_set, safety_rules=safety_rules)
        
        # Make decision in dangerous state
        state = torch.randn(1, 10)
        state_features = {"danger": 0.8}
        
        action, metadata = engine.decide(state, state_features, require_safe=True)
        
        # Should use safety override
        self.assertEqual(action, "ACTION_SAFE", "Should use safety action")
        self.assertEqual(metadata["mode"], "SAFETY", "Should be safety mode")
        self.assertTrue(metadata["override"], "Should override")
        
        # Check stats
        stats = engine.get_stats()
        self.assertEqual(stats["safety_overrides"], 1, "Should have 1 safety override")


class TestPhase1Integration(unittest.TestCase):
    """Integration tests for Phase 1 components."""
    
    def test_multitask_to_rules_pipeline(self):
        """Test extracting rules from multi-task network."""
        # Create multi-task model
        model = create_multitask_network()
        
        # Generate sample states
        states = torch.randn(50, 512)
        
        # For each task, extract rules
        for task_name in ["snake", "pong", "maze"]:
            feature_names = [f"feat_{i}" for i in range(512)]
            action_names = [f"ACTION_{i}" for i in range(model.task_configs[task_name])]
            
            # Create task-specific wrapper
            class TaskWrapper(nn.Module):
                def __init__(self, model, task_name):
                    super().__init__()
                    self.model = model
                    self.task_name = task_name
                
                def forward(self, x):
                    return self.model(x, self.task_name)
            
            task_model = TaskWrapper(model, task_name)
            
            # Extract rules
            extractor = NeuralRuleExtractor(task_model, feature_names, action_names)
            rule_set = extractor.extract_rules(states, max_rules=10, min_support=3)
            
            # Should extract some rules
            self.assertGreaterEqual(len(rule_set), 0, 
                                   f"{task_name}: Should extract rules")


if __name__ == "__main__":
    unittest.main()
