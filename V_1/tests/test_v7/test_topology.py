"""
Tests for NCGN v7 GraphTopology and IndexRegistry.

Verifies:
- Add/remove nodes maintain index sync
- Deleted indices get reused correctly
- Thread safety of registry
- Edge weights preserved
"""

import pytest
import threading
import time


class TestIndexRegistry:
    """Tests for IndexRegistry bidirectional mapping."""
    
    def test_register_and_lookup(self):
        """Test basic registration and lookup."""
        from ncgn.topology import IndexRegistry
        
        registry = IndexRegistry()
        registry.register("dog", 0)
        registry.register("cat", 1)
        registry.register("bird", 2)
        
        assert registry.get_index("dog") == 0
        assert registry.get_index("cat") == 1
        assert registry.get_index("bird") == 2
        
        assert registry.get_label(0) == "dog"
        assert registry.get_label(1) == "cat"
        assert registry.get_label(2) == "bird"
    
    def test_unregister_by_index(self):
        """Test removal by index."""
        from ncgn.topology import IndexRegistry
        
        registry = IndexRegistry()
        registry.register("dog", 0)
        registry.register("cat", 1)
        
        removed = registry.unregister(0)
        
        assert removed == "dog"
        assert registry.get_index("dog") is None
        assert registry.get_label(0) is None
        assert registry.get_index("cat") == 1
    
    def test_unregister_by_label(self):
        """Test removal by label."""
        from ncgn.topology import IndexRegistry
        
        registry = IndexRegistry()
        registry.register("dog", 0)
        registry.register("cat", 1)
        
        removed_idx = registry.unregister_by_label("dog")
        
        assert removed_idx == 0
        assert registry.get_index("dog") is None
    
    def test_duplicate_registration_fails(self):
        """Test that duplicate labels are rejected."""
        from ncgn.topology import IndexRegistry
        
        registry = IndexRegistry()
        registry.register("dog", 0)
        
        with pytest.raises(ValueError):
            registry.register("dog", 1)
    
    def test_thread_safety(self):
        """Test concurrent access from multiple threads."""
        from ncgn.topology import IndexRegistry
        
        registry = IndexRegistry()
        errors = []
        
        def register_labels(start_idx):
            try:
                for i in range(100):
                    label = f"concept_{start_idx}_{i}"
                    idx = start_idx * 1000 + i
                    registry.register(label, idx)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=register_labels, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(registry) == 500


class TestGraphTopology:
    """Tests for GraphTopology with Rustworkx backend."""
    
    def test_add_concept(self):
        """Test adding concepts."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        idx0 = topo.add_concept("dog")
        idx1 = topo.add_concept("cat")
        idx2 = topo.add_concept("bird")
        
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2
        assert topo.num_nodes == 3
    
    def test_add_existing_returns_same_index(self):
        """Test that adding existing label returns same index."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        idx0 = topo.add_concept("dog")
        idx1 = topo.add_concept("dog")
        
        assert idx0 == idx1
        assert topo.num_nodes == 1
    
    def test_remove_concept(self):
        """Test removing concepts."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        topo.add_concept("dog")
        topo.add_concept("cat")
        
        removed_idx = topo.remove_concept("dog")
        
        assert removed_idx == 0
        assert not topo.registry.has_label("dog")
        assert topo.registry.has_label("cat")
    
    def test_add_connection(self):
        """Test adding connections."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        src_idx, tgt_idx = topo.add_connection("dog", "cat", 0.8)
        
        assert src_idx == 0
        assert tgt_idx == 1
        assert topo.num_edges == 1
        assert topo.get_edge_weight("dog", "cat") == 0.8
    
    def test_connection_creates_nodes(self):
        """Test that add_connection creates nodes if needed."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        topo.add_connection("x", "y", 0.5)
        
        assert topo.registry.has_label("x")
        assert topo.registry.has_label("y")
    
    def test_dirty_flag(self):
        """Test that dirty flag is set on modifications."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        assert not topo.is_dirty
        
        topo.add_concept("dog")
        assert topo.is_dirty
        
        topo.mark_clean()
        assert not topo.is_dirty
        
        topo.add_connection("dog", "cat", 0.5)
        assert topo.is_dirty
    
    def test_get_adjacency_data(self):
        """Test extracting edge data for matrix construction."""
        from ncgn.topology import GraphTopology
        import numpy as np
        
        topo = GraphTopology()
        
        topo.add_connection("a", "b", 0.3)
        topo.add_connection("a", "c", 0.5)
        topo.add_connection("b", "c", 0.7)
        
        sources, targets, weights = topo.get_adjacency_data()
        
        assert len(sources) == 3
        assert len(targets) == 3
        assert len(weights) == 3
        
        # Check weight values
        assert 0.3 in weights
        assert 0.5 in weights
        assert 0.7 in weights
    
    def test_index_reuse_after_deletion(self):
        """Test that Rustworkx reuses deleted indices."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        idx0 = topo.add_concept("dog")
        idx1 = topo.add_concept("cat")
        
        topo.remove_concept("dog")
        
        # Add new concept - should reuse index 0
        idx2 = topo.add_concept("bird")
        
        # Rustworkx reuses the index
        assert idx2 == 0  # Reused!
        assert topo.registry.get_label(0) == "bird"
    
    def test_get_neighbors(self):
        """Test getting neighbors."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        topo.add_connection("a", "b", 0.5)
        topo.add_connection("a", "c", 0.5)
        topo.add_connection("a", "d", 0.5)
        
        neighbors = topo.get_neighbors("a")
        
        assert set(neighbors) == {"b", "c", "d"}
    
    def test_clear(self):
        """Test clearing the graph."""
        from ncgn.topology import GraphTopology
        
        topo = GraphTopology()
        
        topo.add_connection("a", "b", 0.5)
        topo.add_connection("c", "d", 0.5)
        
        topo.clear()
        
        assert topo.num_nodes == 0
        assert topo.num_edges == 0
        assert len(topo.registry) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
