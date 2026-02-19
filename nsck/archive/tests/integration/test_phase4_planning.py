"""
Tests for Phase 4: World Models & Planning
"""
import unittest
import torch
import numpy as np

from python.core.neural.world_model import WorldModel, WorldModelConfig, DynamicsPredictor

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
except ImportError:
    from python.core.vsa.hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


class TestWorldModel(unittest.TestCase):
    """Test world model dynamics learning."""
    
    def test_world_model_creation(self):
        """Test creating world model."""
        wm = WorldModel(hv_dim=10240)
        self.assertIsNotNone(wm.predictor)
        self.assertEqual(wm.config.hv_dim, 10240)
    
    def test_world_model_training(self):
        """Test training world model on transitions."""
        wm = WorldModel(hv_dim=10240)
        
        # Train on some transitions (need >100 to be ready)
        for i in range(120):
            state = hypervec_rs.HyperVector(i * 10)
            action = hypervec_rs.HyperVector(i * 10 + 1)
            next_state = hypervec_rs.HyperVector(i * 10 + 2)
            reward = 1.0 if i % 10 == 0 else 0.0
            
            wm.update(state, action, next_state, reward)
        
        self.assertEqual(wm.stats["train_steps"], 120)
        self.assertTrue(wm.is_ready("test"))
    
    def test_world_model_imagination(self):
        """Test forward prediction (imagination)."""
        wm = WorldModel(hv_dim=10240)
        
        # Train briefly
        for i in range(20):
            state = hypervec_rs.HyperVector(i * 10)
            action = hypervec_rs.HyperVector(i * 10 + 1)
            next_state = hypervec_rs.HyperVector(i * 10 + 2)
            reward = 0.5
            wm.update(state, action, next_state, reward)
        
        # Test imagination
        test_state = hypervec_rs.HyperVector(999)
        test_action = hypervec_rs.HyperVector(888)
        
        predicted_next, predicted_reward = wm.imagine(test_state, test_action)
        
        self.assertIsNotNone(predicted_next)
        self.assertIsInstance(predicted_reward, float)
    
    def test_trajectory_rollout(self):
        """Test multi-step trajectory simulation."""
        wm = WorldModel(hv_dim=10240)
        
        # Train
        for i in range(30):
            state = hypervec_rs.HyperVector(i * 10)
            action = hypervec_rs.HyperVector(i * 10 + 1)
            next_state = hypervec_rs.HyperVector(i * 10 + 2)
            reward = 0.1
            wm.update(state, action, next_state, reward)
        
        # Test rollout
        initial_state = hypervec_rs.HyperVector(123)
        actions = [hypervec_rs.HyperVector(i) for i in range(4)]
        
        trajectories = wm.sample_hypothetical_trajectories(
            initial_state, actions, horizon=3, num_paths=5
        )
        
        self.assertEqual(len(trajectories), 5)
        for traj in trajectories:
            self.assertLessEqual(len(traj), 3)
            for step in traj:
                self.assertIn('reward', step)
                self.assertIn('source_bits', step)


class TestModelPredictiveControl(unittest.TestCase):
    """Test MPC planning."""
    
    def test_mpc_planning(self):
        """Test MPC can select actions."""
        from python.training.demos.demo_reasoning import ModelPredictiveController
        
        wm = WorldModel(hv_dim=10240)
        
        # Train world model
        for i in range(50):
            state = hypervec_rs.HyperVector(i * 10)
            action = hypervec_rs.HyperVector(i * 10 + 1)
            next_state = hypervec_rs.HyperVector(i * 10 + 2)
            reward = 1.0 if i % 5 == 0 else 0.0
            wm.update(state, action, next_state, reward)
        
        # Create MPC controller
        mpc = ModelPredictiveController(wm, horizon=3, num_samples=20)
        
        # Test planning
        current_state = hypervec_rs.HyperVector(777)
        actions = [hypervec_rs.HyperVector(i) for i in range(4)]
        
        best_action, value = mpc.plan(current_state, actions)
        
        self.assertIsNotNone(best_action)
        self.assertIsInstance(value, float)


class TestMonteCarloTreeSearch(unittest.TestCase):
    """Test MCTS planning."""
    
    def test_mcts_search(self):
        """Test MCTS can perform tree search."""
        from python.training.demos.demo_reasoning import MonteCarloTreeSearch, MCTSNode
        
        wm = WorldModel(hv_dim=10240)
        
        # Train world model
        for i in range(50):
            state = hypervec_rs.HyperVector(i * 10)
            action = hypervec_rs.HyperVector(i * 10 + 1)
            next_state = hypervec_rs.HyperVector(i * 10 + 2)
            reward = 1.0 if i % 7 == 0 else 0.0
            wm.update(state, action, next_state, reward)
        
        # Create MCTS
        mcts = MonteCarloTreeSearch(wm, n_simulations=20)
        
        # Test search
        initial_state = hypervec_rs.HyperVector(555)
        actions = [hypervec_rs.HyperVector(i) for i in range(4)]
        
        best_action = mcts.search(initial_state, actions)
        
        self.assertIsNotNone(best_action)
    
    def test_mcts_node(self):
        """Test MCTS node operations."""
        from python.training.demos.demo_reasoning import MCTSNode
        
        node = MCTSNode(state="test_state")
        node.untried_actions = [0, 1, 2]
        
        self.assertFalse(node.is_fully_expanded())
        
        child = node.expand(0, "next_state")
        
        self.assertEqual(len(node.children), 1)
        self.assertNotIn(0, node.untried_actions)
        self.assertEqual(child.parent, node)


class TestHierarchicalPlanning(unittest.TestCase):
    """Test hierarchical planning with Options."""
    
    def test_option_creation(self):
        """Test creating options."""
        from python.training.demos.demo_reasoning import Option
        
        option = Option(
            name="test_option",
            policy_fn=lambda s: "action",
            termination_fn=lambda s: s == "goal",
            initiation_fn=lambda s: s != "goal"
        )
        
        self.assertEqual(option.name, "test_option")
        self.assertTrue(option.can_initiate("start"))
        self.assertFalse(option.can_initiate("goal"))
        self.assertTrue(option.should_terminate("goal"))
    
    def test_hierarchical_planner(self):
        """Test hierarchical planning."""
        from python.training.demos.demo_reasoning import Option, HierarchicalPlanner
        
        options = [
            Option("opt1", lambda s: "a1", lambda s: False),
            Option("opt2", lambda s: "a2", lambda s: False)
        ]
        
        planner = HierarchicalPlanner(options)
        
        plan = planner.plan_with_options(
            state="start",
            goal_check_fn=lambda s: s == "goal",
            max_steps=5
        )
        
        self.assertIsInstance(plan, list)
        self.assertLessEqual(len(plan), 5)


class TestPhase4Integration(unittest.TestCase):
    """Integration tests for Phase 4."""
    
    def test_world_model_to_planning_pipeline(self):
        """Test world model → planning pipeline."""
        # Create and train world model (need >100 steps to be ready)
        wm = WorldModel(hv_dim=10240)
        
        for i in range(120):
            state = hypervec_rs.HyperVector(i * 10)
            action = hypervec_rs.HyperVector(i * 10 + 1)
            next_state = hypervec_rs.HyperVector(i * 10 + 2)
            reward = 1.0 if i % 10 == 0 else 0.0
            wm.update(state, action, next_state, reward)
        
        self.assertTrue(wm.is_ready("test"))
        
        # Test imagination
        state = hypervec_rs.HyperVector(999)
        action = hypervec_rs.HyperVector(888)
        next_pred, reward_pred = wm.imagine(state, action)
        
        self.assertIsNotNone(next_pred)
        self.assertIsInstance(reward_pred, float)


if __name__ == "__main__":
    unittest.main()
