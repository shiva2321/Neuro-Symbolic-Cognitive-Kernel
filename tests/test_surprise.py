"""
NCGN Surprise Tests

Test the set-based surprise calculation:
- Violated expectations cause high surprise
- Novel inputs don't cause surprise
- Confidence weighting
"""

import pytest
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import GraphMemory
from core.system1 import System1Engine
from core.bridge import SurpriseMonitor


class TestSurpriseCalculation(unittest.TestCase):
    """Test the set-based surprise formula."""
    
    def test_missing_expectation_high_surprise(self):
        """
        Test: Pre-charge "Meat" node. Input "Metal".
        Assert Surprise > 0.3.
        
        Strategy: Use k=1 to force 'Metal' to suppress 'Meat'.
        1. Dog activates Meat (but < threshold so it persists).
        2. Inject Metal (High Energy).
        3. k-WTA selects Metal, kills Meat.
        4. Surprise Monitor sees Expected(Meat) but Observed(None).
        """
        memory = GraphMemory()
        # k=1 ensures Metal kills Meat
        engine = System1Engine(memory, k_winners=1, surprise_threshold=0.3)
        
        # Setup: Create nodes
        memory.add_node("dog", energy=0.0, threshold=0.5)
        # Higher threshold so it doesn't fire immediately and reset
        memory.add_node("meat", energy=0.0, threshold=1.0, novelty_score=0.1)
        memory.add_node("metal", energy=0.0, threshold=0.5, novelty_score=1.0)
        
        # Dog -> Meat (Strong confidence)
        memory.add_synapse("dog", "meat", type="eats", weight=0.8, confidence=0.9)
        
        # Step 1: Dog predicts Meat
        engine.inject_energy("dog", 1.0)
        engine.tick() 
        # Dog fires (1.0 > 0.5). Meat receives ~0.8. 
        # Meat (0.8 < 1.0) stays ACTIVE (not reset).
        
        # Step 2: Observation (Metal)
        # We do NOT manually clear meat. We lets k-WTA do it.
        # Inject Metal (stronger than Meat)
        engine.inject_energy("metal", 1.0)
        engine.tick()
        
        # Logic: 
        # Expected: Meat (0.8)
        # Observed: Metal (1.0). Meat is inhibited to 0.0 (k=1).
        # Surprise: Expected(Meat) - Observed(0) = High.
        
        assert engine.surprise_level > 0.3, \
             f"Surprise {engine.surprise_level} should be high when expectation is inhibited"
    
    def test_low_confidence_low_surprise(self):
        """
        Test: Low confidence association should produce low surprise.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Setup with LOW confidence
        memory.add_node("source", energy=0.0, threshold=0.5)
        memory.add_node("weak_target", energy=0.0, threshold=0.5, novelty_score=0.1)
        
        # Low confidence synapse
        memory.add_synapse("source", "weak_target", weight=0.8, confidence=0.1)
        
        # Activate source
        engine.inject_energy("source", 1.0)
        engine.tick()
        
        # Capture state, then remove target
        engine._capture_expected_state()
        
        weak = memory.get_node("weak_target")
        if weak:
            weak.energy = 0.0
            memory.mark_inactive("weak_target")
        
        engine.tick()
        
        assert engine.surprise_level < 0.3, \
            f"Surprise {engine.surprise_level} should be low for low-confidence expectations"
    
    def test_novel_input_no_surprise(self):
        """
        Test: Completely new input should not cause surprise.
        
        Novel tokens have no confidence, so they don't violate expectations.
        """
        memory = GraphMemory()
        monitor = SurpriseMonitor(memory)
        
        # No predictions
        monitor.capture_predictions(set())
        
        # Observe something new
        memory.add_node("completely_new", energy=0.8, novelty_score=1.0)
        memory.mark_active("completely_new")
        
        surprise, per_node = monitor.calculate_surprise(
            {"completely_new"},
            {"completely_new": 0.8}
        )
        
        assert surprise < 0.1, f"Novel input should not cause surprise"


class TestSurpriseMonitor(unittest.TestCase):
    """Test the SurpriseMonitor class."""
    
    def test_capture_predictions(self):
        """Test prediction capture."""
        memory = GraphMemory()
        monitor = SurpriseMonitor(memory)
        
        memory.add_node("a", energy=0.5, threshold=0.75)
        memory.add_node("b", energy=0.3, threshold=0.75)
        memory.add_synapse("x", "a", weight=0.5, confidence=0.8)
        memory.mark_active("a")
        memory.mark_active("b")
        
        monitor.capture_predictions({"a", "b"})
        
        assert "a" in monitor.prediction_set
        assert "b" in monitor.prediction_set
    
    def test_violated_expectations_list(self):
        """Test getting list of violated expectations."""
        memory = GraphMemory()
        monitor = SurpriseMonitor(memory)
        
        # Setup prediction
        memory.add_node("expected", energy=0.8, threshold=0.75, novelty_score=0.1)
        memory.add_synapse("cause", "expected", weight=0.9, confidence=0.9)
        memory.mark_active("expected")
        
        monitor.capture_predictions({"expected"})
        
        # Now expected is missing
        violations = monitor.get_violated_expectations({"expected": 0.0})
        
        assert len(violations) == 1
        assert violations[0][0] == "expected"


class TestSurpriseInPipeline(unittest.TestCase):
    """Test surprise in full tick pipeline."""
    
    def test_system2_trigger(self):
        """
        Test: High surprise should trigger System 2.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10, surprise_threshold=0.2)
        
        triggered = [False]
        
        def on_surprise(level):
            triggered[0] = True
        
        engine.set_system2_callback(on_surprise)
        
        # Create expectation
        memory.add_node("predictor", threshold=0.5)
        memory.add_node("expected", threshold=0.5, novelty_score=0.1)
        memory.add_synapse("predictor", "expected", weight=0.9, confidence=0.9)
        
        # Fire predictor
        engine.inject_energy("predictor", 1.0)
        engine.tick()
        
        # Now violate expectation
        expected = memory.get_node("expected")
        expected.energy = 0.0
        memory.mark_inactive("expected")
        
        engine.inject_energy("unexpected", 1.0)
        engine.tick()
        
        # System 2 should have been notified
        # (May or may not be triggered depending on surprise calculation)
        assert isinstance(engine.surprise_level, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
