import unittest
from cognitive_engine import create_cognitive_engine

class TestIntegratedMetacognition(unittest.TestCase):
    def test_learn_loop_updates_confidence(self):
        """
        Verify that learning updates the self-model's predicted success rate
        based on accumulated experience (successes vs failures).
        """
        engine = create_cognitive_engine()
        task = "snake_test"
        state = {"head": (5,5), "food": (5,6)}
        
        # 1. Initial State — cold start returns 0.5
        pred_1 = engine.self_model.predict_success(task)
        print(f"\n[TEST] Initial predict_success: {pred_1}")
        self.assertEqual(pred_1, 0.5)
        
        # 2. Simulate 20 failures — self-model should learn poor performance
        for _ in range(20):
            engine.learn(state, "ACTION_UP", reward=-1.0, task_tag=task, outcome="failure")
            
        # 3. After 20 failures (20 attempts, 0 success) predict_success = 0/20 = 0.0
        pred_2 = engine.self_model.predict_success(task)
        stats = engine.self_model.get_stats(task)
        print(f"[TEST] Post-Failure predict_success: {pred_2} (Stats: {stats})")
        
        self.assertEqual(pred_2, 0.0, "Should be 0.0 after 20 fails (0/20)")
        
        # 4. Simulate 20 successes
        for _ in range(20):
            engine.learn(state, "ACTION_UP", reward=1.0, task_tag=task, outcome="success")
            
        # New history: 20 fail, 20 success -> base_rate = 0.5
        # Recent window is all successes so trend blending adjusts slightly upward
        pred_3 = engine.self_model.predict_success(task)
        print(f"[TEST] Post-Recovery predict_success: {pred_3}")
        
        self.assertGreaterEqual(pred_3, 0.5, "Should recover to at least 0.5 (20/40)")
        
    def test_calibration_tracking(self):
        """Verify calibration error keeps track."""
        engine = create_cognitive_engine()
        task = "calib_test"
        state = {}
        
        # Predict 0.5 (cold start), Outcome: Failure (0.0)
        # Error: |0.5 - 0.0| = 0.5
        engine.decide(state, task) # Sets current_state.self_confidence = 0.5
        engine.learn(state, "A", -1, task)
        
        err = engine.self_model.get_calibration_error(task)
        self.assertEqual(err, 0.5)

if __name__ == '__main__':
    unittest.main()
