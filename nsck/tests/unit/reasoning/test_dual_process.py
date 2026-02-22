import unittest

class TestDualProcess(unittest.TestCase):
    def test_system1_for_familiar_state(self):
        """High-confidence familiar states should use fast System 1 when dual_process enabled."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        config = NSCKConfig(enable_dual_process=True, system1_confidence_threshold=0.0)
        engine = CognitiveEngine(config)
        state = {"head": (5, 5), "food": (5, 4), "score": 100}
        result = engine.decide(state, task_tag="default")
        # With threshold=0.0, System 1 should always win
        self.assertIsNotNone(result.chosen_action)
        self.assertEqual(result.system_used, "system_1")

    def test_system2_for_novel_state(self):
        """Novel states with high threshold should trigger System 2."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        config = NSCKConfig(enable_dual_process=True, system1_confidence_threshold=0.99)
        engine = CognitiveEngine(config)
        state = {"novel_key_xyz": "novel_value_abc"}
        result = engine.decide(state, task_tag="default")
        # With threshold=0.99, System 1 should never win → System 2 engaged
        self.assertIsNotNone(result.chosen_action)
        self.assertEqual(result.system_used, "system_2")

    def test_engine_decides_without_crash(self):
        """Verify that decide() returns a CognitiveState without crashing."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.reasoning.cognitive_engine import CognitiveState
        from python.core.integration.config import NSCKConfig
        engine = CognitiveEngine(NSCKConfig())
        for i in range(3):
            result = engine.decide({"x": i}, task_tag="default")
            self.assertIsInstance(result, CognitiveState)
            self.assertIsNotNone(result.chosen_action)

if __name__ == "__main__":
    unittest.main()
