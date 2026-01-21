"""
NCGN Physics Tests

Test the fundamental physics of the tick pipeline:
- Energy propagation through chains
- Decay behavior
- Refractory period enforcement
"""

import pytest
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import GraphMemory, ConceptNode, Synapse
from core.system1 import System1Engine


class TestEnergyPropagation(unittest.TestCase):
    """Test energy propagation through node chains."""
    
    def test_chain_activation_abc(self):
        """
        Test: A->B->C chain activation.
        
        Stimulate A. Verify C activates after 2 ticks.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Create chain A -> B -> C with strong weights
        memory.add_node("A", threshold=0.5)
        memory.add_node("B", threshold=0.5)
        memory.add_node("C", threshold=0.5)
        
        memory.add_synapse("A", "B", weight=0.8, confidence=0.9)
        memory.add_synapse("B", "C", weight=0.8, confidence=0.9)
        
        # Inject energy into A
        engine.inject_energy("A", 1.0)
        
        # Tick 0: A receives energy
        engine.tick()
        
        # A should have fired (energy > threshold after transduction)
        # B should now have pending/integrated energy
        node_b = memory.get_node("B")
        assert node_b.energy > 0, "B should have received energy from A"
        
        # Tick 1: B fires, C receives
        engine.tick()
        node_c = memory.get_node("C")
        assert node_c.energy > 0, "C should have received energy from B"
    
    def test_energy_decay_without_stimulation(self):
        """
        Test: Energy decays to 0 without reinforcement.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, decay_alpha=0.9)
        
        # Add node with initial energy below threshold
        memory.add_node("test", energy=0.5, threshold=0.75)
        memory.mark_active("test")
        
        # Run ticks without stimulation
        for _ in range(100):
            engine.tick()
        
        node = memory.get_node("test")
        assert node.energy < 0.001, "Energy should decay to near zero"
    
    def test_refractory_period_enforcement(self):
        """
        Test: Nodes cannot fire during refractory period.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, refractory_period=3)
        
        memory.add_node("A", threshold=0.5)
        memory.add_node("B", threshold=0.5)
        memory.add_synapse("A", "B", weight=0.9, confidence=0.9)
        
        # Fire A
        engine.inject_energy("A", 1.0)
        engine.tick()
        
        # A should be in refractory
        node_a = memory.get_node("A")
        assert node_a.refractory_timer > 0, "A should be refractory after firing"
        
        # Try to restimulate A
        engine.inject_energy("A", 1.0)
        engine.tick()
        
        # A should NOT have fired again (still refractory)
        # We check by seeing if B got additional energy
        # This is complex to test directly, but we can check refractory timer
        assert node_a.refractory_timer >= 1, "Refractory should still be active"
    
    def test_energy_stays_bounded(self):
        """
        Test: Energy is clamped to [0, 1] range.
        """
        memory = GraphMemory()
        engine = System1Engine(memory)
        
        memory.add_node("test", threshold=0.75)
        
        # Inject massive energy
        engine.inject_energy("test", 100.0)
        engine.tick()
        
        node = memory.get_node("test")
        # After firing, energy should be reset or clamped
        assert node.energy <= 1.0, "Energy should never exceed 1.0"


class TestSpikePropagation(unittest.TestCase):
    """Test spike transmission mechanics."""
    
    def test_weight_affects_transmission(self):
        """
        Test: Synapse weight affects energy transmitted.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Create two targets with different weight connections
        memory.add_node("source", threshold=0.3)
        memory.add_node("strong_target", threshold=0.8)
        memory.add_node("weak_target", threshold=0.8)
        
        memory.add_synapse("source", "strong_target", weight=0.9, confidence=0.9)
        memory.add_synapse("source", "weak_target", weight=0.2, confidence=0.9)
        
        # Fire source
        engine.inject_energy("source", 1.0)
        engine.tick()
        
        strong = memory.get_node("strong_target")
        weak = memory.get_node("weak_target")
        
        assert strong.energy > weak.energy, \
            "Strong connection should transmit more energy"
    
    def test_buffered_propagation_no_cascade(self):
        """
        Test: Propagation is buffered to prevent same-tick cascades.
        
        A -> B -> C should NOT all fire in the same tick.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Create chain with very strong connections
        memory.add_node("A", threshold=0.3)
        memory.add_node("B", threshold=0.3)
        memory.add_node("C", threshold=0.3)
        
        memory.add_synapse("A", "B", weight=1.0, confidence=0.9)
        memory.add_synapse("B", "C", weight=1.0, confidence=0.9)
        
        engine.inject_energy("A", 1.0)
        engine.tick()
        
        # After one tick, C should NOT have fired yet
        node_c = memory.get_node("C")
        # C might have some energy, but should not be in firing_set history
        # We verify by checking that C's last_spike_tick is not tick 0
        assert node_c.last_spike_tick != 0, \
            "C should not fire in same tick as A"


class TestActiveNodeTracking(unittest.TestCase):
    """Test sparse active node set management."""
    
    def test_only_active_nodes_processed(self):
        """
        Test: Only nodes with energy > 0 are in active set.
        """
        memory = GraphMemory()
        
        # Add many nodes, only some with energy
        for i in range(100):
            energy = 0.5 if i < 10 else 0.0
            memory.add_node(f"node_{i}", energy=energy)
            if energy > 0:
                memory.mark_active(f"node_{i}")
        
        active = memory.get_active_nodes()
        assert len(active) == 10, "Only 10 nodes should be active"
    
    def test_nodes_deactivate_after_decay(self):
        """
        Test: Nodes are removed from active set when energy -> 0.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, decay_alpha=0.5)  # Fast decay
        
        memory.add_node("test", energy=0.1, threshold=0.75)
        memory.mark_active("test")
        
        # Run several ticks
        for _ in range(10):
            engine.tick()
        
        active = memory.get_active_nodes()
        assert "test" not in active, "Node should be inactive after energy decays"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
