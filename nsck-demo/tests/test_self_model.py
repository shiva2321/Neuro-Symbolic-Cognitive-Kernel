import unittest
from self_model import SelfModel

class TestSelfModel(unittest.TestCase):
    def test_cold_start(self):
        """Test behavior with no data."""
        model = SelfModel()
        prob = model.predict_success("snake")
        self.assertEqual(prob, 0.5, "Should be uncertain (0.5) initially")
        
    def test_learning_success_rate(self):
        """Test that success rate updates correctly."""
        model = SelfModel()
        
        # 1. Fail 5 times
        for _ in range(5):
            model.update("snake", predicted_confidence=0.5, actual_success=False)
            
        prob = model.predict_success("snake")
        self.assertEqual(prob, 0.5, "Still < 10 samples, should stay 0.5")
        
        # 2. Succeed 5 times (Total 10: 5 fail, 5 success)
        for _ in range(5):
            model.update("snake", predicted_confidence=0.5, actual_success=True)
            
        prob = model.predict_success("snake")
        self.assertEqual(prob, 0.5, "5/10 successes = 0.5 probability")
        
        # 3. Succeed 10 more times (Total 20: 5 fail, 15 success)
        for _ in range(10):
            model.update("snake", predicted_confidence=0.5, actual_success=True)
            
        prob = model.predict_success("snake")
        self.assertEqual(prob, 15/20, "Should be 0.75")
        
    def test_calibration_error(self):
        """Test calibration error calculation."""
        model = SelfModel()
        
        # Scenario: Overconfident Agent
        # Says 1.0 (sure), but fails (0.0)
        model.update("snake", predicted_confidence=1.0, actual_success=False)
        
        err = model.get_calibration_error("snake")
        self.assertEqual(err, 1.0, "Absolute error should be |1.0 - 0.0| = 1.0")
        
        # Scenario: Perfectly Calibrated
        # Says 1.0, Succeeds
        model.update("snake", predicted_confidence=1.0, actual_success=True)
        
        # Err = (|1-0| + |1-1|) / 2 = 0.5
        err = model.get_calibration_error("snake")
        self.assertEqual(err, 0.5)

if __name__ == '__main__':
    unittest.main()
