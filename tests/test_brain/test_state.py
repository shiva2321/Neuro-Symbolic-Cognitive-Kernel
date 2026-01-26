"""
Tests for NCGN v7 CognitiveState.

Verifies:
- Array resizing works correctly
- zero_index clears all arrays
- CSR matrix matches topology
- Capacity management
"""

import pytest
import numpy as np


class TestCognitiveState:
    """Tests for CognitiveState SoA arrays."""
    
    def test_initial_state(self):
        """Test initial state of arrays."""
        from src.brain.core.state import CognitiveState
        from src.brain.core.config import Config
        
        config = Config(initial_capacity=100)
        state = CognitiveState(config)
        
        assert state.capacity == 100
        assert len(state.activations) == 100
        assert len(state.thresholds) == 100
        assert len(state.refractory_counters) == 100
        assert np.all(state.activations == 0.0)
    
    def test_ensure_capacity_no_resize(self):
        """Test that ensure_capacity doesn't resize unnecessarily."""
        from src.brain.core.state import CognitiveState
        from src.brain.core.config import Config
        
        config = Config(initial_capacity=100)
        state = CognitiveState(config)
        
        resized = state.ensure_capacity(50)
        
        assert not resized
        assert state.capacity == 100
    
    def test_ensure_capacity_resize(self):
        """Test that ensure_capacity doubles when needed."""
        from src.brain.core.state import CognitiveState
        from src.brain.core.config import Config
        
        config = Config(initial_capacity=100)
        state = CognitiveState(config)
        
        # Set some values before resize
        state.activations[50] = 0.5
        state.thresholds[50] = 0.8
        
        resized = state.ensure_capacity(150)
        
        assert resized
        assert state.capacity == 200  # Doubled
        assert len(state.activations) == 200
        
        # Check values preserved
        assert state.activations[50] == 0.5
        assert state.thresholds[50] == 0.8
    
    def test_zero_index(self):
        """Test that zero_index clears all arrays at index."""
        from src.brain.core.state import CognitiveState
        from src.brain.core.config import Config
        
        config = Config(initial_capacity=100)
        state = CognitiveState(config)
        
        # Set values at index 10
        state.activations[10] = 0.9
        state.thresholds[10] = 0.5
        state.refractory_counters[10] = 5
        state.novelty_scores[10] = 0.2
        
        # Zero the index
        state.zero_index(10)
        
        # Check all values cleared
        assert state.activations[10] == 0.0
        assert state.thresholds[10] == config.default_threshold
        assert state.refractory_counters[10] == 0
        assert state.novelty_scores[10] == 1.0
    
    def test_set_activation(self):
        """Test setting activation with clamping."""
        from src.brain.core.state import CognitiveState
        
        state = CognitiveState()
        
        state.set_activation(5, 0.7)
        assert state.activations[5] == 0.7
        
        # Test clamping
        state.set_activation(5, 1.5)
        assert state.activations[5] == 1.0
        
        state.set_activation(5, -0.5)
        assert state.activations[5] == 0.0
    
    def test_add_activation(self):
        """Test adding to activation."""
        from src.brain.core.state import CognitiveState
        
        state = CognitiveState()
        
        state.set_activation(5, 0.4)
        state.add_activation(5, 0.3)
        
        assert abs(state.activations[5] - 0.7) < 0.001
    
    def test_get_active_indices(self):
        """Test getting active node indices."""
        from src.brain.core.state import CognitiveState
        
        state = CognitiveState()
        
        state.activations[0] = 0.1
        state.activations[5] = 0.3
        state.activations[10] = 0.8
        state.activations[15] = 0.02  # Below threshold
        
        active = state.get_active_indices(threshold=0.05)
        
        assert set(active) == {0, 5, 10}
    
    def test_get_firing_indices(self):
        """Test getting nodes ready to fire."""
        from src.brain.core.state import CognitiveState
        
        state = CognitiveState()
        
        state.activations[0] = 0.9  # Above threshold
        state.activations[5] = 0.9  # Above threshold but refractory
        state.activations[10] = 0.5  # Below threshold
        
        state.refractory_counters[5] = 3  # In refractory
        
        firing = state.get_firing_indices()
        
        assert 0 in firing
        assert 5 not in firing  # Refractory
        assert 10 not in firing  # Below threshold
    
    def test_synchronize_matrix(self):
        """Test CSR matrix synchronization."""
        from src.brain.core.state import CognitiveState
        from src.brain.core.topology import GraphTopology
        
        topo = GraphTopology()
        topo.add_connection("a", "b", 0.3)
        topo.add_connection("a", "c", 0.5)
        
        state = CognitiveState()
        state.synchronize_matrix(topo)
        
        assert state.adjacency is not None
        assert state.adjacency.nnz == 2
    
    def test_matrix_not_rebuilt_when_clean(self):
        """Test that matrix isn't rebuilt when topology is clean."""
        from src.brain.core.state import CognitiveState
        from src.brain.core.topology import GraphTopology
        
        topo = GraphTopology()
        topo.add_connection("a", "b", 0.3)
        
        state = CognitiveState()
        state.synchronize_matrix(topo)
        
        # Get reference to matrix
        matrix1 = state.adjacency
        
        # Sync again without changes
        state.synchronize_matrix(topo)
        matrix2 = state.adjacency
        
        # Should be same object (not rebuilt)
        assert matrix1 is matrix2
    
    def test_clear_all(self):
        """Test clearing all state."""
        from src.brain.core.state import CognitiveState
        
        state = CognitiveState()
        
        state.activations[0:10] = 0.5
        state.refractory_counters[0:5] = 3
        
        state.clear_all()
        
        assert np.all(state.activations == 0.0)
        assert np.all(state.refractory_counters == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
