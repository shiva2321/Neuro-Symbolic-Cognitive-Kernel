"""
Comprehensive stress tests for NSCK concurrent cognitive architecture

Tests thread safety, performance, deadlock detection, and edge cases
"""

import pytest
import threading
import time
import random
import sys
import os

# Add rust_vsa to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nsck-demo', 'rust_vsa', 'target', 'release'))

try:
    import hypervec_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Rust extension not built")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestStressConcurrency:
    """Stress tests for concurrent operations"""
    
    def test_1000_concurrent_queries(self):
        """Test 1000 concurrent semantic search queries"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        
        # Add 1000 concepts
        for i in range(1000):
            hv = hypervec_rs.HyperVector(seed=i)
            semantic.add_concept(f"concept_{i}", hv)
        
        # Run 1000 concurrent queries
        results = []
        errors = []
        
        def query_thread(thread_id):
            try:
                query = hypervec_rs.HyperVector(seed=thread_id % 100)
                result = semantic.parallel_semantic_search(query, k=10)
                results.append((thread_id, len(result)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = []
        start = time.time()
        for i in range(1000):
            t = threading.Thread(target=query_thread, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)  # 30s timeout per thread
        
        elapsed = time.time() - start
        
        assert len(errors) == 0, f"Errors occurred: {errors[:5]}"
        assert len(results) == 1000, f"Only {len(results)}/1000 queries completed"
        
        throughput = 1000 / elapsed
        print(f"\n1000 queries completed in {elapsed:.2f}s ({throughput:.1f} QPS)")
        assert throughput > 50, "Throughput too low"
    
    def test_concurrent_spreading_activation(self):
        """Test concurrent spreading activation from multiple threads"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        
        # Create connected graph
        for i in range(100):
            hv = hypervec_rs.HyperVector(seed=i)
            semantic.add_concept(f"concept_{i}", hv)
        
        for i in range(99):
            semantic.add_relation(f"concept_{i}", f"concept_{i+1}")
        
        results = []
        errors = []
        
        def spread_thread(thread_id):
            try:
                start_concept = f"concept_{thread_id % 50}"
                activation = semantic.parallel_spread_activation(
                    [start_concept], steps=3, decay=0.7, min_activation=0.01
                )
                results.append((thread_id, len(activation)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = []
        for i in range(50):
            t = threading.Thread(target=spread_thread, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=60)
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50, f"Only {len(results)}/50 spreads completed"
        print(f"\n50 concurrent spreading activations completed successfully")
    
    def test_concurrent_episode_insertion(self):
        """Test concurrent episode insertions"""
        episodic = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=10000)
        
        errors = []
        inserted = []
        
        def insert_thread(thread_id):
            try:
                for i in range(100):
                    hv = hypervec_rs.HyperVector(seed=thread_id * 1000 + i)
                    episode = hypervec_rs.Episode(
                        timestamp=float(thread_id * 1000 + i),
                        task_tag=f"task_{thread_id}",
                        situation_hv=hv,
                        action=f"action_{i}",
                        outcome="success",
                        reward=0.5
                    )
                    evicted = episodic.add_episode(episode)
                    inserted.append((thread_id, i))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = []
        for i in range(50):
            t = threading.Thread(target=insert_thread, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors occurred: {errors[:5]}"
        assert len(inserted) == 5000, f"Only {len(inserted)}/5000 episodes inserted"
        
        # Verify final count
        final_count = episodic.size()
        print(f"\n5000 episodes inserted concurrently, final count: {final_count}")
        assert final_count <= 10000, "Max capacity exceeded"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestDeadlockDetection:
    """Tests for deadlock detection and timeouts"""
    
    def test_no_deadlock_mixed_operations(self):
        """Test mixed read/write operations don't deadlock"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        
        # Pre-populate
        for i in range(100):
            hv = hypervec_rs.HyperVector(seed=i)
            semantic.add_concept(f"concept_{i}", hv)
        
        results = []
        
        def mixed_operations(thread_id):
            for _ in range(50):
                # Mix of reads and writes
                if thread_id % 3 == 0:
                    # Read
                    query = hypervec_rs.HyperVector(seed=thread_id)
                    semantic.parallel_semantic_search(query, k=5)
                elif thread_id % 3 == 1:
                    # Write concept
                    hv = hypervec_rs.HyperVector(seed=thread_id + 1000)
                    semantic.add_concept(f"new_concept_{thread_id}", hv)
                else:
                    # Write relation
                    semantic.add_relation(f"concept_{thread_id % 50}", f"concept_{(thread_id + 1) % 50}")
            results.append(thread_id)
        
        threads = []
        start = time.time()
        for i in range(30):
            t = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(t)
            t.start()
        
        # All threads should complete within 30 seconds (no deadlock)
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Thread deadlocked or hung"
        
        elapsed = time.time() - start
        assert len(results) == 30, f"Only {len(results)}/30 threads completed"
        print(f"\n30 threads with mixed operations completed in {elapsed:.2f}s (no deadlock)")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestMemoryExhaustion:
    """Tests for memory exhaustion handling"""
    
    def test_large_episode_batch(self):
        """Test handling of very large episode batches"""
        episodic = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=1000)
        
        # Insert more than capacity (should evict oldest)
        for i in range(2000):
            hv = hypervec_rs.HyperVector(seed=i)
            episode = hypervec_rs.Episode(
                timestamp=float(i),
                task_tag="stress_test",
                situation_hv=hv,
                action=f"action_{i}",
                outcome="success",
                reward=0.5
            )
            evicted = episodic.add_episode(episode)
            
            if i >= 1000:
                assert evicted is not None, f"Should evict at episode {i}"
        
        final_count = episodic.size()
        assert final_count <= 1000, f"Capacity violated: {final_count} > 1000"
        print(f"\n2000 episodes inserted, FIFO eviction maintained capacity at {final_count}")
    
    def test_large_spreading_activation(self):
        """Test spreading activation on large graphs"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        
        # Create large densely connected graph
        n_concepts = 1000
        for i in range(n_concepts):
            hv = hypervec_rs.HyperVector(seed=i)
            semantic.add_concept(f"concept_{i}", hv)
        
        # Add many relations (each concept connected to 5 others)
        for i in range(n_concepts):
            for j in range(5):
                target = (i + j + 1) % n_concepts
                semantic.add_relation(f"concept_{i}", f"concept_{target}")
        
        # Spread from single concept
        start = time.time()
        activation = semantic.parallel_spread_activation(
            ["concept_0"], steps=5, decay=0.7, min_activation=0.001
        )
        elapsed = time.time() - start
        
        print(f"\nSpread on 1000-node graph (5 steps): {elapsed:.2f}s, {len(activation)} concepts activated")
        assert len(activation) > 0, "No activation spread"
        assert elapsed < 10.0, f"Spreading too slow: {elapsed:.2f}s"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_semantic_search(self):
        """Test search on empty memory"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        query = hypervec_rs.HyperVector(seed=42)
        
        results = semantic.parallel_semantic_search(query, k=10)
        assert len(results) == 0, "Should return empty list"
    
    def test_spreading_nonexistent_concept(self):
        """Test spreading from nonexistent concept"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        
        # Add some concepts but spread from nonexistent one
        for i in range(10):
            hv = hypervec_rs.HyperVector(seed=i)
            semantic.add_concept(f"concept_{i}", hv)
        
        activation = semantic.parallel_spread_activation(
            ["nonexistent_concept"], steps=3, decay=0.7
        )
        
        # Should return empty or minimal activation
        assert len(activation) <= 1, "Should not spread from nonexistent concept"
    
    def test_zero_k_search(self):
        """Test search with k=0"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        
        for i in range(10):
            hv = hypervec_rs.HyperVector(seed=i)
            semantic.add_concept(f"concept_{i}", hv)
        
        query = hypervec_rs.HyperVector(seed=5)
        results = semantic.parallel_semantic_search(query, k=0)
        
        assert len(results) == 0, "k=0 should return empty"
    
    def test_concurrent_registry_operations(self):
        """Test concurrent operations on HyperVectorRegistry"""
        registry = hypervec_rs.HyperVectorRegistry()
        
        errors = []
        
        def register_vectors(thread_id):
            try:
                for i in range(100):
                    name = f"vec_{thread_id}_{i}"
                    hv = hypervec_rs.HyperVector(seed=thread_id * 1000 + i)
                    registry.register(name, hv)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = []
        for i in range(20):
            t = threading.Thread(target=register_vectors, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert registry.size() == 2000, f"Expected 2000, got {registry.size()}"
        print(f"\n2000 vectors registered concurrently from 20 threads")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestWorkerPool:
    """Tests for cognitive worker pool"""
    
    def test_worker_pool_creation(self):
        """Test worker pool can be created and shut down"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        episodic = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=1000)
        
        # Create with 4 workers
        pool = hypervec_rs.CognitiveWorkerPool(semantic, episodic, num_workers=4)
        
        assert pool.worker_count() == 4
        
        # Shutdown
        pool.shutdown()
        print("\nWorker pool created and shut down successfully")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestPersistence:
    """Tests for persistent storage"""
    
    def test_batch_persistence(self, tmp_path):
        """Test batch writes to persistent storage"""
        db_path = tmp_path / "test.db"
        storage = hypervec_rs.PersistentStorage(str(db_path), batch_size=50)
        
        # Buffer 100 episodes (should trigger 2 flushes)
        for i in range(100):
            hv = hypervec_rs.HyperVector(seed=i)
            episode = hypervec_rs.Episode(
                timestamp=float(i),
                task_tag="test_task",
                situation_hv=hv,
                action=f"action_{i}",
                outcome="success",
                reward=0.5
            )
            should_flush = storage.buffer_episode(episode)
            
            if should_flush:
                flushed = storage.flush()
                assert flushed > 0, "Flush should write episodes"
        
        # Final flush
        remaining = storage.flush()
        
        # Verify all stored
        total = storage.episode_count()
        assert total == 100, f"Expected 100 episodes, got {total}"
        print(f"\n100 episodes persisted via batch writes")


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestAsyncRuntime:
    """Tests for async runtime"""
    
    def test_async_runtime_creation(self):
        """Test async runtime can be created"""
        semantic = hypervec_rs.SemanticMemoryConcurrent()
        episodic = hypervec_rs.EpisodicMemoryConcurrent(max_hot_size=1000)
        
        runtime = hypervec_rs.AsyncCognitiveRuntime(semantic, episodic)
        
        stats = runtime.get_stats()
        assert stats['worker_threads'] == 4
        assert stats['runtime_type'] == 'tokio-multi-thread'
        print("\nAsync runtime created successfully with 4 worker threads")


def run_all_stress_tests():
    """Run all stress tests and print summary"""
    pytest.main([__file__, "-v", "--tb=short", "-x"])


if __name__ == "__main__":
    run_all_stress_tests()
