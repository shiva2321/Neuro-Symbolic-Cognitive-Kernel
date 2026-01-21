"""
NCGN K-WTA Tests

Test the Winner-Take-All attention mechanism:
- Min-Heap selection algorithm
- Correct number of winners
- Proper suppression of losers
"""

import pytest
import unittest
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import GraphMemory
from core.system1 import System1Engine


class TestKWTAAlgorithm(unittest.TestCase):
    """Test the k-Winners-Take-All inhibition."""
    
    def test_exactly_k_winners(self):
        """
        Test: Activate 100 nodes with random energies.
        Set K=5. Verify exactly 5 nodes remain active.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=5)
        
        # Create 100 nodes with random energies (below threshold to not fire)
        random.seed(42)
        for i in range(100):
            energy = random.uniform(0.1, 0.7)  # Below threshold
            memory.add_node(f"node_{i}", energy=energy, threshold=0.75)
            memory.mark_active(f"node_{i}")
        
        # Run the k-WTA phase directly
        engine._phase_6_kwta_inhibition()
        
        # Count active nodes
        active = memory.get_active_nodes()
        assert len(active) == 5, f"Expected 5 winners, got {len(active)}"
    
    def test_winners_are_highest_energy(self):
        """
        Test: Winners should be the K highest energy nodes.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=3)
        
        # Create nodes with known energies
        energies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for i, e in enumerate(energies):
            memory.add_node(f"node_{i}", energy=e, threshold=0.75)
            memory.mark_active(f"node_{i}")
        
        engine._phase_6_kwta_inhibition()
        
        active = memory.get_active_nodes()
        
        # Winners should be node_4, node_5, node_6 (highest energies)
        expected_winners = {"node_4", "node_5", "node_6"}
        assert active == expected_winners, \
            f"Expected {expected_winners}, got {active}"
    
    def test_no_inhibition_when_under_k(self):
        """
        Test: No inhibition when active nodes <= K.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Only 5 nodes (less than K=10)
        for i in range(5):
            memory.add_node(f"node_{i}", energy=0.5, threshold=0.75)
            memory.mark_active(f"node_{i}")
        
        engine._phase_6_kwta_inhibition()
        
        active = memory.get_active_nodes()
        assert len(active) == 5, "All 5 nodes should remain active"
    
    def test_losers_suppressed_to_zero(self):
        """
        Test: Losing nodes have energy set to exactly 0.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=2)
        
        for i in range(5):
            memory.add_node(f"node_{i}", energy=0.1 * (i + 1), threshold=0.75)
            memory.mark_active(f"node_{i}")
        
        engine._phase_6_kwta_inhibition()
        
        # Check losers
        for i in range(3):  # node_0, node_1, node_2 are losers
            node = memory.get_node(f"node_{i}")
            assert node.energy == 0.0, f"Loser node_{i} should have 0 energy"
    
    def test_kwta_efficiency_large_scale(self):
        """
        Test: k-WTA should handle large numbers efficiently.
        
        This is O(M log K) where M = active nodes.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Create 10000 nodes
        random.seed(123)
        for i in range(10000):
            energy = random.uniform(0.01, 0.7)
            memory.add_node(f"node_{i}", energy=energy, threshold=0.75)
            memory.mark_active(f"node_{i}")
        
        # This should complete quickly (< 1 second)
        import time
        start = time.time()
        engine._phase_6_kwta_inhibition()
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"k-WTA took {elapsed:.2f}s, should be < 1s"
        
        active = memory.get_active_nodes()
        assert len(active) == 10, f"Expected 10 winners, got {len(active)}"


class TestKWTAIntegration(unittest.TestCase):
    """Test k-WTA within full tick pipeline."""
    
    def test_kwta_in_tick_pipeline(self):
        """
        Test: k-WTA runs correctly within full tick.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=3, decay_alpha=0.99)
        
        # Inject energy into many nodes
        for i in range(20):
            engine.inject_energy(f"node_{i}", 0.3 + 0.02 * i)
        
        engine.tick()
        
        active = memory.get_active_nodes()
        assert len(active) <= 3, f"Should have at most 3 active, got {len(active)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
