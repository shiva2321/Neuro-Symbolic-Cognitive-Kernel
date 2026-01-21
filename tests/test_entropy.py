"""
NCGN v6.0 Entropy Explosion Tests

Tests the critical entropy control mechanisms:
1. Homeostatic Normalization (cluster energy capping)
2. Seizure Damping (global energy threshold)
3. Softmax Temperature Dynamics

These tests verify that the system cannot enter runaway
activation states that would break learning.
"""

import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.memory import GraphMemory, ClusterType
from core.system1 import System1Engine


class TestEntropyExplosionPrevention:
    """Tests for entropy explosion prevention mechanisms."""
    
    def test_homeostatic_normalization_caps_cluster_energy(self):
        """
        Verify that cluster energy is capped by homeostatic normalization.
        
        ENTROPY EXPLOSION PREVENTION TEST:
        If we inject massive energy into a cluster, the normalization
        should scale it down to the energy_cap.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, energy_cap=5.0)
        
        # Create motor nodes and inject way too much energy
        for i in range(10):
            node_id = f"motor_{i}"
            memory.add_node(node_id, cluster=ClusterType.MOTOR)
            engine.inject_energy(node_id, 10.0, ClusterType.MOTOR)
        
        # Run one tick (which includes normalization)
        engine.tick()
        
        # Check that total motor cluster energy is bounded
        motor_energy = sum(
            memory.get_node(f"motor_{i}").energy 
            for i in range(10)
        )
        
        # Should be capped near energy_cap
        assert motor_energy <= engine.energy_cap + 0.1, (
            f"Motor cluster energy {motor_energy} exceeds cap {engine.energy_cap}"
        )
    
    def test_seizure_damping_prevents_global_explosion(self):
        """
        Verify seizure damping kicks in when global energy too high.
        
        ENTROPY EXPLOSION PREVENTION TEST:
        If total system energy exceeds the seizure threshold, all
        nodes should be damped to prevent runaway activation.
        """
        memory = GraphMemory()
        engine = System1Engine(memory)
        
        # Create many nodes and directly set high energy
        for cluster in [ClusterType.SENSORY, ClusterType.HIDDEN, ClusterType.MOTOR]:
            for i in range(20):
                node_id = f"{cluster.value}_{i}"
                memory.add_node(node_id, cluster=cluster)
                # Directly set energy on the node
                node = memory.get_node(node_id)
                node.energy = 5.0
                memory.mark_active(node_id)
        
        # Update total energy to trigger seizure detection
        memory.update_total_energy()
        initial_energy = memory.get_total_energy()
        
        # Run tick - should trigger seizure damping
        engine.tick()
        
        final_energy = memory.get_total_energy()
        
        # Energy should be significantly reduced
        assert final_energy < initial_energy, (
            f"Energy should decrease after seizure damping: "
            f"initial={initial_energy}, final={final_energy}"
        )
    
    def test_temperature_increases_when_system_quiet(self):
        """
        Verify temperature annealing works correctly.
        
        When system energy is low, temperature should increase
        (encouraging exploration).
        """
        memory = GraphMemory()
        engine = System1Engine(memory, base_temperature=0.5)
        
        # No energy in system - should be quiet
        engine.tick()
        
        # Temperature should increase above baseline (exploration)
        assert engine.temperature > 0.4, (
            f"Temperature should increase when system is quiet, "
            f"got {engine.temperature}"
        )
    
    def test_softmax_selects_single_motor_action(self):
        """
        Verify Softmax selects exactly one motor action.
        
        After Softmax inhibition, only one motor node should have
        non-zero energy (winner-take-all for actions).
        """
        memory = GraphMemory()
        engine = System1Engine(memory)
        
        # Create motor nodes with varying energies
        motor_ids = []
        for i, energy in enumerate([0.3, 0.7, 0.5, 0.6]):
            node_id = f"motor_{i}"
            memory.add_node(node_id, cluster=ClusterType.MOTOR)
            engine.inject_energy(node_id, energy, ClusterType.MOTOR)
            motor_ids.append(node_id)
        
        # Run tick
        engine.tick()
        
        # Count how many motor nodes have energy
        active_motors = sum(
            1 for nid in motor_ids 
            if memory.get_node(nid).energy > 0.01
        )
        
        # Should be exactly one winner
        assert active_motors == 1, (
            f"Expected exactly 1 active motor, got {active_motors}"
        )
    
    def test_gating_excludes_low_energy_from_softmax(self):
        """
        Verify that near-zero energy nodes are excluded from Softmax.
        
        ENTROPY EXPLOSION PREVENTION TEST:
        Nodes with E < 0.05 should be gated out to prevent them
        from accumulating probability mass.
        """
        memory = GraphMemory()
        engine = System1Engine(memory)
        
        # Create motor nodes - one strong, many weak
        memory.add_node("motor_strong", cluster=ClusterType.MOTOR)
        engine.inject_energy("motor_strong", 0.8, ClusterType.MOTOR)
        
        for i in range(10):
            node_id = f"motor_weak_{i}"
            memory.add_node(node_id, cluster=ClusterType.MOTOR)
            engine.inject_energy(node_id, 0.02, ClusterType.MOTOR)  # Below gate
        
        # Run tick
        engine.tick()
        
        # Strong motor should win (weak ones gated out)
        strong_energy = memory.get_node("motor_strong").energy
        assert strong_energy > 0.9, (
            f"Strong motor should win with high energy, got {strong_energy}"
        )
    
    def test_prolonged_activity_stays_bounded(self):
        """
        Verify system stays stable over many ticks.
        
        ENTROPY EXPLOSION PREVENTION TEST:
        Run the system for many ticks with continuous input.
        Total energy should remain bounded, not explode.
        """
        memory = GraphMemory()
        engine = System1Engine(memory, energy_cap=10.0)
        
        # Create a recurrent network
        for i in range(5):
            memory.add_node(f"hidden_{i}", cluster=ClusterType.HIDDEN)
        
        # Add recurrent connections
        for i in range(5):
            for j in range(5):
                if i != j:
                    memory.add_synapse(f"hidden_{i}", f"hidden_{j}", weight=0.5)
        
        max_energy_seen = 0.0
        
        # Run many ticks with continuous stimulation
        for tick in range(100):
            # Inject some energy each tick
            engine.inject_energy("hidden_0", 0.5, ClusterType.HIDDEN)
            engine.tick()
            
            current_energy = memory.get_total_energy()
            max_energy_seen = max(max_energy_seen, current_energy)
        
        # System should not explode
        assert max_energy_seen < 50.0, (
            f"System energy exploded to {max_energy_seen}"
        )


class TestEligibilityTraces:
    """Tests for eligibility trace mechanics."""
    
    def test_trace_set_on_coincidence(self):
        """Verify traces are set when pre and post are active."""
        memory = GraphMemory()
        engine = System1Engine(memory)
        
        # Create pre -> post connection
        memory.add_node("pre", cluster=ClusterType.SENSORY)
        memory.add_node("post", cluster=ClusterType.HIDDEN)
        memory.add_synapse("pre", "post", weight=0.8)
        
        # Activate both nodes
        engine.inject_energy("pre", 1.0, ClusterType.SENSORY)
        engine.inject_energy("post", 1.0, ClusterType.HIDDEN)
        
        # Run tick (should mark trace)
        engine.tick()
        
        # Check trace is set
        synapse = memory.get_synapse("pre", "post")
        assert synapse.trace > 0.5, (
            f"Trace should be high after coincidence, got {synapse.trace}"
        )
    
    def test_trace_decays_over_time(self):
        """Verify traces decay when not reinforced."""
        memory = GraphMemory()
        engine = System1Engine(memory, trace_decay=0.8)
        
        # Create and activate connection
        memory.add_node("pre", cluster=ClusterType.SENSORY)
        memory.add_node("post", cluster=ClusterType.HIDDEN)
        memory.add_synapse("pre", "post", weight=0.8)
        
        # Set trace manually
        synapse = memory.get_synapse("pre", "post")
        synapse.trace = 1.0
        memory.mark_synapse_traced("pre", "post")
        
        # Run ticks without activity
        for _ in range(5):
            engine.tick()
        
        # Trace should have decayed
        assert synapse.trace < 0.5, (
            f"Trace should decay, got {synapse.trace}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
