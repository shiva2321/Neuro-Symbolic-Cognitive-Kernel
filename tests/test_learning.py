"""
NCGN v6.0 Learning Tests

Tests the 3-Factor Hebbian learning mechanisms:
1. Dopamine Modulator (RPE, washout prevention)
2. Three-Factor Learner (stability, catastrophic forgetting)
3. Episodic Buffer (Monte Carlo returns)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.memory import GraphMemory, ClusterType
from core.learning import DopamineModulator, ThreeFactorLearner, EpisodicBuffer


class TestDopamineModulator:
    """Tests for the dopamine/RPE calculation system."""
    
    def test_positive_rpe_on_better_than_expected(self):
        """Verify positive RPE when reward exceeds baseline."""
        modulator = DopamineModulator(baseline_alpha=0.1)
        
        # Set baseline low
        for _ in range(10):
            modulator.calculate_signal(0.1)
        
        # Now get a big reward
        rpe = modulator.calculate_signal(1.0)
        
        assert rpe > 0, f"RPE should be positive for unexpected reward, got {rpe}"
    
    def test_negative_rpe_on_worse_than_expected(self):
        """Verify negative RPE when reward below baseline."""
        modulator = DopamineModulator(baseline_alpha=0.1)
        
        # Set baseline high
        for _ in range(10):
            modulator.calculate_signal(1.0)
        
        # Now get nothing
        rpe = modulator.calculate_signal(0.0)
        
        assert rpe < 0, f"RPE should be negative for missing reward, got {rpe}"
    
    def test_washout_prevention_constant_reward(self):
        """
        DOPAMINE WASHOUT PREVENTION TEST:
        
        If rewards are constant, RPE should approach zero over time.
        This prevents runaway weight growth in stable environments.
        """
        modulator = DopamineModulator(baseline_alpha=0.1)
        
        # Send constant reward for many steps
        rpe_values = []
        for _ in range(50):
            rpe = modulator.calculate_signal(1.0)
            rpe_values.append(abs(rpe))
        
        # Early RPE should be higher than late RPE
        early_avg = sum(rpe_values[:10]) / 10
        late_avg = sum(rpe_values[-10:]) / 10
        
        assert late_avg < early_avg, (
            f"RPE should decrease with constant reward: "
            f"early={early_avg:.3f}, late={late_avg:.3f}"
        )
        
        # Late RPE should be near zero
        assert late_avg < 0.2, (
            f"Late RPE should approach zero, got {late_avg:.3f}"
        )


class TestThreeFactorLearner:
    """Tests for the 3-Factor Hebbian learning system."""
    
    def test_positive_dopamine_increases_weights(self):
        """Verify weights increase with positive RPE and trace."""
        memory = GraphMemory()
        learner = ThreeFactorLearner(memory, learning_rate=0.5)
        
        # Create synapse with trace
        memory.add_node("pre", cluster=ClusterType.SENSORY)
        memory.add_node("post", cluster=ClusterType.HIDDEN)
        memory.add_synapse("pre", "post", weight=0.5)
        
        synapse = memory.get_synapse("pre", "post")
        synapse.trace = 1.0
        memory.mark_synapse_traced("pre", "post")
        
        initial_weight = synapse.weight
        
        # Apply positive dopamine
        learner.apply_reward(0.5)
        
        assert synapse.weight > initial_weight, (
            f"Weight should increase: initial={initial_weight}, "
            f"final={synapse.weight}"
        )
    
    def test_negative_dopamine_decreases_weights(self):
        """Verify weights decrease with negative RPE and trace."""
        memory = GraphMemory()
        learner = ThreeFactorLearner(memory, learning_rate=0.5)
        
        memory.add_node("pre", cluster=ClusterType.SENSORY)
        memory.add_node("post", cluster=ClusterType.HIDDEN)
        memory.add_synapse("pre", "post", weight=0.5)
        
        synapse = memory.get_synapse("pre", "post")
        synapse.trace = 1.0
        memory.mark_synapse_traced("pre", "post")
        
        initial_weight = synapse.weight
        
        # Apply negative dopamine
        learner.apply_reward(-0.5)
        
        assert synapse.weight < initial_weight, (
            f"Weight should decrease: initial={initial_weight}, "
            f"final={synapse.weight}"
        )
    
    def test_stability_protects_weights(self):
        """
        CATASTROPHIC FORGETTING PREVENTION TEST:
        
        Synapses with high stability should resist weight changes.
        """
        memory = GraphMemory()
        learner = ThreeFactorLearner(memory, learning_rate=0.5)
        
        memory.add_node("pre", cluster=ClusterType.SENSORY)
        memory.add_node("post", cluster=ClusterType.HIDDEN)
        memory.add_synapse("pre", "post", weight=0.5)
        
        synapse = memory.get_synapse("pre", "post")
        synapse.trace = 1.0
        synapse.stability = 0.9  # 90% consolidated
        memory.mark_synapse_traced("pre", "post")
        
        initial_weight = synapse.weight
        
        # Apply large dopamine
        learner.apply_reward(1.0)
        
        change = abs(synapse.weight - initial_weight)
        
        # Change should be small due to stability
        assert change < 0.1, (
            f"Weight change should be small with high stability, "
            f"got change={change}"
        )
    
    def test_no_trace_no_learning(self):
        """Verify no learning occurs without eligibility trace."""
        memory = GraphMemory()
        learner = ThreeFactorLearner(memory, learning_rate=0.5)
        
        memory.add_node("pre", cluster=ClusterType.SENSORY)
        memory.add_node("post", cluster=ClusterType.HIDDEN)
        memory.add_synapse("pre", "post", weight=0.5)
        
        synapse = memory.get_synapse("pre", "post")
        synapse.trace = 0.0  # No trace
        
        initial_weight = synapse.weight
        
        # Apply dopamine
        learner.apply_reward(1.0)
        
        assert synapse.weight == initial_weight, (
            f"Weight should not change without trace"
        )


class TestEpisodicBuffer:
    """Tests for episodic reward distribution."""
    
    def test_compute_returns_discounted(self):
        """Verify returns are correctly discounted."""
        buffer = EpisodicBuffer()
        
        # Record rewards: [0, 0, 0, 1] (reward at end)
        buffer.episode_traces.append([])
        buffer.episode_rewards.append(0.0)
        buffer.episode_traces.append([])
        buffer.episode_rewards.append(0.0)
        buffer.episode_traces.append([])
        buffer.episode_rewards.append(0.0)
        buffer.episode_traces.append([])
        buffer.episode_rewards.append(1.0)
        
        returns = buffer.compute_returns(gamma=0.9)
        
        # Returns should be discounted backwards
        # G_3 = 1.0
        # G_2 = 0 + 0.9 * 1.0 = 0.9
        # G_1 = 0 + 0.9 * 0.9 = 0.81
        # G_0 = 0 + 0.9 * 0.81 = 0.729
        
        assert len(returns) == 4
        assert abs(returns[3] - 1.0) < 0.01
        assert abs(returns[2] - 0.9) < 0.01
        assert abs(returns[1] - 0.81) < 0.01
        assert abs(returns[0] - 0.729) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
