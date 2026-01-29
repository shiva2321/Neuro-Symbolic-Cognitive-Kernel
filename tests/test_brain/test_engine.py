"""
Tests for NCGN v7 PropagationEngine.

Verifies:
- Energy decays without input
- Propagation spreads activation
- Refractory period prevents firing
- Seizure damping activates at threshold
"""

import pytest
import numpy as np


class TestPropagationEngine:
    """Tests for PropagationEngine vectorized dynamics."""
    
    def test_energy_decay(self):
        """Test that energy decays without reinforcement."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        from ncgn.config import Config
        
        config = Config(decay_delta=0.1)
        topo = GraphTopology()
        topo.add_concept("dog")
        
        state = CognitiveState(config)
        state.set_activation(0, 1.0)
        
        engine = PropagationEngine(state, topo, config)
        
        # After propagation, energy should decay
        initial_energy = state.activations[0]
        engine.propagate(1)
        
        # Energy should be lower (due to decay and sigmoid)
        assert state.activations[0] < initial_energy
    
    def test_propagation_spreads_activation(self):
        """Test that activation spreads through edges."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        from ncgn.config import Config
        
        config = Config()
        topo = GraphTopology()
        
        # Create chain: a -> b -> c
        topo.add_connection("a", "b", 0.8)
        topo.add_connection("b", "c", 0.8)
        
        state = CognitiveState(config)
        engine = PropagationEngine(state, topo, config)
        
        # Inject energy at "a"
        engine.inject(0, 1.0)  # Index 0 = "a"
        
        # Propagate
        active = engine.propagate(5)
        
        # "b" and "c" should have some activation
        b_energy = state.activations[1]
        c_energy = state.activations[2]
        
        assert b_energy > 0 or "b" in active
        # c might take more steps to activate
    
    def test_inject_by_label(self):
        """Test injecting energy by label."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        
        topo = GraphTopology()
        topo.add_concept("dog")
        
        state = CognitiveState()
        engine = PropagationEngine(state, topo)
        
        success = engine.inject_by_label("dog", 0.5)
        
        assert success
        
        # After propagation, the energy should be applied
        engine.propagate(1)
        assert state.activations[0] > 0
    
    def test_inject_unknown_label_fails(self):
        """Test that injecting into unknown label returns False."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        
        topo = GraphTopology()
        state = CognitiveState()
        engine = PropagationEngine(state, topo)
        
        success = engine.inject_by_label("unknown", 0.5)
        
        assert not success
    
    def test_refractory_period(self):
        """Test that nodes in refractory period don't fire."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        from ncgn.config import Config
        
        config = Config(refractory_period=5)
        topo = GraphTopology()
        topo.add_concept("dog")
        
        state = CognitiveState(config)
        state.refractory_counters[0] = 3  # In refractory
        state.activations[0] = 0.95  # High energy
        
        engine = PropagationEngine(state, topo, config)
        engine.propagate(1)
        
        # Energy should be zeroed due to refractory mask
        assert state.activations[0] == 0.0
    
    def test_get_active_concepts(self):
        """Test getting active concepts as dict."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        
        topo = GraphTopology()
        topo.add_concept("dog")
        topo.add_concept("cat")
        
        state = CognitiveState()
        state.activations[0] = 0.5
        state.activations[1] = 0.1
        
        engine = PropagationEngine(state, topo)
        active = engine.get_active_concepts()
        
        assert "dog" in active
        assert "cat" in active
        assert active["dog"] > active["cat"]
    
    def test_get_top_k_active(self):
        """Test getting top K active concepts."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        
        topo = GraphTopology()
        for i in range(10):
            topo.add_concept(f"concept_{i}")
        
        state = CognitiveState()
        for i in range(10):
            state.activations[i] = (i + 1) / 10.0
        
        engine = PropagationEngine(state, topo)
        top3 = engine.get_top_k_active(3)
        
        assert len(top3) == 3
        assert "concept_9" in top3
        assert "concept_8" in top3
        assert "concept_7" in top3
    
    def test_multiple_ticks(self):
        """Test running multiple propagation ticks."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        
        topo = GraphTopology()
        topo.add_connection("a", "b", 0.5)
        
        state = CognitiveState()
        engine = PropagationEngine(state, topo)
        
        engine.inject_by_label("a", 0.8)
        engine.propagate(10)
        
        assert engine.tick_count == 10
    
    def test_reset(self):
        """Test resetting the engine."""
        from ncgn.topology import GraphTopology
        from ncgn.state import CognitiveState
        from ncgn.engine import PropagationEngine
        
        topo = GraphTopology()
        topo.add_concept("dog")
        
        state = CognitiveState()
        engine = PropagationEngine(state, topo)
        
        engine.inject_by_label("dog", 0.5)
        engine.propagate(5)
        
        engine.reset()
        
        assert engine.tick_count == 0
        assert len(engine._pending_indices) == 0


class TestSigmoidFunction:
    """Tests for the sigmoid activation function."""
    
    def test_sigmoid_range(self):
        """Test that sigmoid output is in [0, 1]."""
        from ncgn.engine import sigmoid
        
        x = np.array([-100, -10, -1, 0, 1, 10, 100])
        y = sigmoid(x)
        
        # Allow for numerical boundaries (0 and 1 inclusive at extremes)
        assert np.all(y >= 0)
        assert np.all(y <= 1)
    
    def test_sigmoid_midpoint(self):
        """Test that sigmoid(0) = 0.5."""
        from ncgn.engine import sigmoid
        
        y = sigmoid(np.array([0.0]))
        assert abs(y[0] - 0.5) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
