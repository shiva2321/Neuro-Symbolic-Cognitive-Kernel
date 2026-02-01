import unittest
from cognitive_engine import create_cognitive_engine

class TestIntegratedMetacognition(unittest.TestCase):
    def test_learn_loop_updates_confidence(self):
        """
        Verify that learning updates the self-model, which in turn
        updates the confidence of future decisions.
        """
        engine = create_cognitive_engine()
        task = "snake_test"
        state = {"head": (5,5), "food": (5,6)}
        
        # 1. Initial State (Confidence should be 0.5 due to cold start)
        decision_1 = engine.decide(state, task)
        conf_1 = decision_1.self_confidence
        print(f"\n[TEST] Initial Confidence: {conf_1}")
        self.assertEqual(conf_1, 0.5)
        
        # 2. Simulate 20 failures
        # This should tank the confidence
        for _ in range(20):
            engine.learn(state, "ACTION_UP", reward=-1.0, task_tag=task, outcome="failure")
            
        # 3. Check new confidence
        decision_2 = engine.decide(state, task)
        conf_2 = decision_2.self_confidence
        stats = engine.self_model.get_stats(task)
        print(f"[TEST] Post-Failure Confidence: {conf_2} (Stats: {stats})")
        
        
        self.assertEqual(conf_2, 0.0, "Should be 0.0 after 20 fails")
        # With Fusion: SNN(0.5)*0.4 + Self(0.0)*0.6 = 0.2
        self.assertAlmostEqual(decision_2.confidence, 0.2, places=2, msg="Fused confidence should be 0.2")
        
        # 4. Simulate 20 successes
        for _ in range(20):
            engine.learn(state, "ACTION_UP", reward=1.0, task_tag=task, outcome="success")
            
        # New history: 20 fail, 20 success -> 0.5
        decision_3 = engine.decide(state, task)
        conf_3 = decision_3.self_confidence
        print(f"[TEST] Post-Recovery Confidence: {conf_3}")
        
        self.assertEqual(conf_3, 0.5, "Should recover to 0.5 (20/40)")
        
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
