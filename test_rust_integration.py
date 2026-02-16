"""
Integration tests for concurrent NSCK cognitive architecture

These tests validate the Python-Rust interop and concurrent operations
"""

import pytest
import sys
import os

# Add the rust_vsa path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo', 'rust_vsa', 'target', 'release'))

try:
    import hypervec_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Rust extension not built")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestRustExtension:
    """Test basic Rust extension functionality"""
    
    def test_hypervector_creation(self):
        """Test HyperVector can be created from Python"""
        hv = hypervec_rs.HyperVector()
        assert hv is not None
        
    def test_hypervector_deterministic(self):
        """Test deterministic creation with seed"""
        hv1 = hypervec_rs.HyperVector(seed=42)
        hv2 = hypervec_rs.HyperVector(seed=42)
        
        # Same seed should produce identical vectors
        similarity = hv1.similarity(hv2)
        assert similarity == 1.0
        
    def test_xor_operation(self):
        """Test XOR binding"""
        a = hypervec_rs.HyperVector(seed=1)
        b = hypervec_rs.HyperVector(seed=2)
        
        c = a.xor(b)
        assert c is not None
        
        # XOR is reversible
        recovered = c.xor(b)
        similarity = a.similarity(recovered)
        assert similarity > 0.99
        
    def test_bundle_operation(self):
        """Test bundle operation"""
        a = hypervec_rs.HyperVector(seed=1)
        b = hypervec_rs.HyperVector(seed=2)
        
        bundled = a.bundle(b)
        
        # Bundled should be similar to both
        sim_a = bundled.similarity(a)
        sim_b = bundled.similarity(b)
        
        assert 0.4 < sim_a < 1.0
        assert 0.4 < sim_b < 1.0
        
    def test_permute_operation(self):
        """Test permutation"""
        hv = hypervec_rs.HyperVector(seed=42)
        
        # Permute and inverse should recover original
        permuted = hv.permute(100)
        restored = permuted.permute_inverse(100)
        
        similarity = hv.similarity(restored)
        assert similarity == 1.0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestHyperVectorRegistry:
    """Test concurrent HyperVectorRegistry"""
    
    def test_registry_creation(self):
        """Test registry can be created"""
        registry = hypervec_rs.HyperVectorRegistry()
        assert registry.size() == 0
        
    def test_register_and_get(self):
        """Test registering and retrieving vectors"""
        registry = hypervec_rs.HyperVectorRegistry()
        hv = hypervec_rs.HyperVector(seed=42)
        
        registry.register("test_vector", hv)
        assert registry.size() == 1
        
        retrieved = registry.get("test_vector")
        assert retrieved is not None
        
        # Should be identical
        similarity = hv.similarity(retrieved)
        assert similarity == 1.0
        
    def test_nearest_neighbors(self):
        """Test nearest neighbor search"""
        registry = hypervec_rs.HyperVectorRegistry()
        
        # Register 100 vectors
        for i in range(100):
            hv = hypervec_rs.HyperVector(seed=i)
            registry.register(f"vec_{i}", hv)
            
        # Search for nearest neighbors
        query = hypervec_rs.HyperVector(seed=50)
        results = registry.nearest_neighbors(query, k=10)
        
        assert len(results) == 10
        
        # Results should be sorted by similarity
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]
            
        # Top result should be vec_50 (identical)
        assert results[0][0] == "vec_50"
        assert results[0][1] == 1.0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSemanticMemory:
    """Test concurrent semantic memory"""
    
    def test_semantic_memory_creation(self):
        """Test semantic memory can be created"""
        memory = hypervec_rs.SemanticMemoryConcurrent()
        assert memory.concept_count() == 0
        
    def test_add_concepts_and_relations(self):
        """Test adding concepts and relations"""
        memory = hypervec_rs.SemanticMemoryConcurrent()
        
        # Add concepts
        hv_a = hypervec_rs.HyperVector(seed=1)
        hv_b = hypervec_rs.HyperVector(seed=2)
        
        memory.add_concept("concept_A", hv_a)
        memory.add_concept("concept_B", hv_b)
        
        assert memory.concept_count() == 2
        
        # Add relation
        memory.add_relation("concept_A", "concept_B")
        assert memory.relation_count() == 1
        
    def test_parallel_semantic_search(self):
        """Test parallel semantic search"""
        memory = hypervec_rs.SemanticMemoryConcurrent()
        
        # Add 50 concepts
        for i in range(50):
            hv = hypervec_rs.HyperVector(seed=i)
            memory.add_concept(f"concept_{i}", hv)
            
        # Search
        query = hypervec_rs.HyperVector(seed=25)
        results = memory.parallel_semantic_search(query, k=10)
        
        assert len(results) == 10
        assert results[0][0] == "concept_25"
        assert results[0][1] == 1.0
        
    def test_spreading_activation(self):
        """Test spreading activation"""
        memory = hypervec_rs.SemanticMemoryConcurrent()
        
        # Create a simple graph: A -> B -> C
        for i in range(3):
            hv = hypervec_rs.HyperVector(seed=i)
            memory.add_concept(f"concept_{i}", hv)
            
        memory.add_relation("concept_0", "concept_1")
        memory.add_relation("concept_1", "concept_2")
        
        # Spread from concept_0
        activation = memory.parallel_spread_activation(
            ["concept_0"],
            steps=2,
            decay=0.7,
            min_activation=0.01
        )
        
        # Check that activation reaches concept_1 and concept_2
        assert "concept_0" in activation
        assert "concept_1" in activation
        assert "concept_2" in activation
        
        # Activation should decrease with distance
        assert activation["concept_0"] > activation["concept_1"]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestEpisodicMemory:
    """Test concurrent episodic memory"""
    
    def test_episodic_memory_creation(self):
        """Test episodic memory can be created"""
        memory = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=1000)
        assert memory.size() == 0
        
    def test_add_episode(self):
        """Test adding episodes"""
        memory = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=10)
        
        # Create episode
        hv = hypervec_rs.HyperVector(seed=1)
        episode = hypervec_rs.Episode(
            timestamp=1234567890.0,
            task_tag="test_task",
            situation_hv=hv,
            action="test_action",
            outcome="success",
            reward=1.0,
            impact_score=0.8
        )
        
        evicted = memory.add_episode(episode)
        assert evicted is None  # No eviction yet
        assert memory.size() == 1
        
    def test_parallel_knn_search(self):
        """Test parallel k-NN search"""
        memory = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=100)
        
        # Add 50 episodes
        for i in range(50):
            hv = hypervec_rs.HyperVector(seed=i)
            episode = hypervec_rs.Episode(
                timestamp=float(i),
                task_tag="test_task",
                situation_hv=hv,
                action=f"action_{i}",
                outcome="success",
                reward=0.5,
                impact_score=0.3
            )
            memory.add_episode(episode)
            
        # Search
        query = hypervec_rs.HyperVector(seed=25)
        results = memory.parallel_knn_search(query, k=10)
        
        assert len(results) == 10
        
        # Results should be sorted by similarity
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]
            
    def test_task_filtering(self):
        """Test task-based filtering"""
        memory = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=100)
        
        # Add episodes with different tasks
        for i in range(25):
            hv = hypervec_rs.HyperVector(seed=i)
            episode = hypervec_rs.Episode(
                timestamp=float(i),
                task_tag="task_A",
                situation_hv=hv,
                action=f"action_{i}",
                outcome="success",
                reward=0.5,
                impact_score=0.3
            )
            memory.add_episode(episode)
            
        for i in range(25, 50):
            hv = hypervec_rs.HyperVector(seed=i)
            episode = hypervec_rs.Episode(
                timestamp=float(i),
                task_tag="task_B",
                situation_hv=hv,
                action=f"action_{i}",
                outcome="success",
                reward=0.5,
                impact_score=0.3
            )
            memory.add_episode(episode)
            
        # Search only task_A
        query = hypervec_rs.HyperVector(seed=12)
        results = memory.parallel_knn_search(query, k=10, task_filter="task_A")
        
        assert len(results) == 10
        
        # All results should be from task_A
        for _, _, episode in results:
            assert episode.task_tag == "task_A"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestParallelOperations:
    """Test parallel VSA operations"""
    
    def test_parallel_similarity_search(self):
        """Test parallel similarity search function"""
        query = hypervec_rs.HyperVector(seed=50)
        candidates = [hypervec_rs.HyperVector(seed=i) for i in range(100)]
        
        results = hypervec_rs.parallel_similarity_search(query, candidates, k=10)
        
        assert len(results) == 10
        
        # Results should be sorted
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]
            
        # Top result should be index 50
        assert results[0][0] == 50
        assert results[0][1] == 1.0
        
    def test_parallel_bundle(self):
        """Test parallel bundle operation"""
        vectors = [hypervec_rs.HyperVector(seed=i) for i in range(10)]
        
        bundled = hypervec_rs.parallel_bundle(vectors)
        assert bundled is not None
        
        # Bundled should be similar to all inputs
        for hv in vectors:
            similarity = bundled.similarity(hv)
            assert similarity > 0.3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
