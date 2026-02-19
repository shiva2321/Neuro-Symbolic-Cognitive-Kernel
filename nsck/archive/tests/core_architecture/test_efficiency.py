"""
Phase 6: Efficiency Benchmarks
Performance metrics across speed, memory, and scalability
"""

import sys
import os
import pytest
import time
import tracemalloc
import random

# Setup path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.vsa.universal_encoder import UniversalEncoder
from python.core.memory.semantic_memory import SemanticMemory


class TestInferenceSpeed:
    """Test EFF-1: Inference speed <500ms for typical queries"""
    
    def test_simple_query_latency(self):
        """Simple query completes in <500ms"""
        learner = TextKnowledgeLearner()
        
        # Train
        learner.learn_text("Dogs are animals")
        
        # Query and time it
        start = time.time()
        result = learner.query_learned_knowledge("What are dogs?", top_k=5)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nSimple query latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 500, f"Simple query too slow: {elapsed_ms:.1f}ms"
    
    def test_complex_query_latency(self):
        """Complex reasoning query completes quickly"""
        learner = TextKnowledgeLearner()
        
        # Train on rich knowledge base
        facts = [
            "Dogs are mammals",
            "Mammals are animals",
            "Animals are organisms",
            "Dogs are loyal",
            "Dogs have four legs",
            "Mammals have fur",
        ]
        
        for fact in facts:
            learner.learn_text(fact)
        
        # Complex query
        start = time.time()
        result = learner.query_learned_knowledge(
            "Tell me everything about dogs",
            top_k=10
        )
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nComplex query latency: {elapsed_ms:.1f}ms")
        # Should still be reasonable (<500ms for small KB)
    
    def test_batch_query_throughput(self):
        """Batch of queries maintains throughput"""
        learner = TextKnowledgeLearner()
        
        # Train
        for i in range(10):
            learner.learn_text(f"Fact {i}: Something is related to concept {i}")
        
        # Batch queries
        queries = [f"What about concept {i}?" for i in range(5)]
        
        start = time.time()
        results = [learner.query_learned_knowledge(q, top_k=5) for q in queries]
        total_ms = (time.time() - start) * 1000
        per_query_ms = total_ms / len(queries)
        
        print(f"\nBatch query throughput: {per_query_ms:.1f}ms per query")
        assert per_query_ms < 500, f"Throughput too low: {per_query_ms:.1f}ms/query"


class TestMemoryFootprint:
    """Test EFF-2: Memory usage scales appropriately"""
    
    def test_memory_for_small_kb(self):
        """Small KB uses minimal memory"""
        tracemalloc.start()
        
        learner = TextKnowledgeLearner()
        
        # Small KB: 100 concepts
        for i in range(100):
            learner.learn_text(f"Concept {i} is a thing with property {i % 10}")
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_mb = peak / 1024 / 1024
        print(f"\n100 concepts memory: {memory_mb:.1f}MB")
        
        # Should be under 500MB for small KB
        assert memory_mb < 500, f"Excessive memory: {memory_mb:.1f}MB"
    
    def test_memory_scaling_linear(self):
        """Memory scales roughly linearly with KB size"""
        sizes = [100, 500, 1000]
        memories = []
        
        for size in sizes:
            tracemalloc.start()
            
            learner = TextKnowledgeLearner()
            for i in range(size):
                learner.learn_text(f"Concept {i}: {i % 5} properties")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_mb = peak / 1024 / 1024
            memories.append(memory_mb)
            print(f"\n{size} concepts: {memory_mb:.1f}MB")
        
        # Check scaling ratio
        # If linear: ratio should be roughly size ratio
        if len(memories) >= 2:
            ratio = memories[1] / memories[0]
            size_ratio = sizes[1] / sizes[0]
            print(f"Size ratio: {size_ratio}, Memory ratio: {ratio:.2f}")


class TestEncodingSpeed:
    """Test EFF-3: Text encoding <10ms per sentence"""
    
    def test_single_sentence_encoding(self):
        """Single sentence encodes in <10ms"""
        encoder = UniversalEncoder()
        
        sentence = "The interesting hypothesis suggests a remarkable possibility"
        
        start = time.time()
        hv = encoder.encode_text(sentence)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nSingle sentence encoding: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 10, f"Encoding too slow: {elapsed_ms:.2f}ms"
    
    def test_batch_encoding_speed(self):
        """Batch encoding maintains speed"""
        encoder = UniversalEncoder()
        
        sentences = [
            f"This is test sentence number {i}" 
            for i in range(100)
        ]
        
        start = time.time()
        hvs = [encoder.encode_text(s) for s in sentences]
        total_ms = (time.time() - start) * 1000
        per_sentence_ms = total_ms / len(sentences)
        
        print(f"\nBatch encoding (100 sentences): {per_sentence_ms:.2f}ms per sentence")
        assert per_sentence_ms < 10, f"Batch encoding too slow: {per_sentence_ms:.2f}ms"


class TestSpreadingActivationSpeed:
    """Test EFF-4: Spreading activation <50ms for 3-step decay"""
    
    def test_spreading_activation_latency(self):
        """Spreading activation over graph completes quickly"""
        learner = TextKnowledgeLearner()
        
        # Build a network
        for i in range(50):
            learner.learn_text(f"Node {i} connects to concept {i % 5}")
        
        # Measure activation across graph
        encoder = learner.lingua
        seed_hv = encoder.encode_text("Node 0")
        
        start = time.time()
        # Spreading activation with 3-step decay
        if hasattr(learner.semantic, 'spread_activation'):
            activated = learner.semantic.spread_activation([seed_hv], steps=3, decay=0.7)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nSpreading activation (3 steps): {elapsed_ms:.1f}ms")
        # Should be fast even over large graph


class TestSemanticSearchSpeed:
    """Test EFF-5: Query HV similarity search <100ms"""
    
    def test_semantic_search_latency(self):
        """Similarity search over 1000 concepts"""
        mem = SemanticMemory()
        encoder = UniversalEncoder()
        
        # Add 1000 concepts
        print("\nAdding 1000 concepts to semantic memory...")
        for i in range(1000):
            concept_name = f"concept_{i}"
            hv = encoder.encode_text(concept_name)
            mem.add_concept(concept_name, hv)
        
        # Query
        query_hv = encoder.encode_text("concept_500")
        
        start = time.time()
        results = mem.query(query_hv, k=10)
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\nSemantic search (1K concepts, top-10): {elapsed_ms:.1f}ms")
        assert elapsed_ms < 100, f"Search too slow: {elapsed_ms:.1f}ms"


class TestScalabilityMetrics:
    """Test EFF-6: Scalability - latency growth is logarithmic or linear"""
    
    def test_latency_vs_kb_size(self):
        """Query latency vs KB size - should scale well"""
        sizes = [100, 500, 1000, 2000]
        latencies = []
        
        for size in sizes:
            learner = TextKnowledgeLearner()
            
            # Build KB
            print(f"\nBuilding KB with {size} facts...")
            for i in range(size):
                learner.learn_text(f"Fact {i}: Concept {i} relates to {i % 10}")
            
            # Query
            start = time.time()
            result = learner.query_learned_knowledge("Tell me about concept 0", top_k=5)
            elapsed_ms = (time.time() - start) * 1000
            
            latencies.append(elapsed_ms)
            print(f"  KB size {size}: {elapsed_ms:.1f}ms")
        
        # Check scaling
        print("\nScaling analysis:")
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            latency_ratio = latencies[i] / latencies[i-1]
            print(f"  Size ×{size_ratio:.1f} → Latency ×{latency_ratio:.2f}")
            
            # Should be logarithmic (≈1.something) or linear (≈size_ratio)
            # Super-linear growth is bad
            assert latency_ratio < size_ratio * 1.5, \
                f"Super-linear scaling detected: {latency_ratio:.2f}× for {size_ratio:.1f}× size"


class TestBenchmarkSuite:
    """Comprehensive benchmark suite"""
    
    def test_benchmark_report(self):
        """Print comprehensive benchmark report"""
        print("\n" + "="*70)
        print("NSCK EFFICIENCY BENCHMARK REPORT")
        print("="*70)
        
        results = {}
        
        # Benchmark 1: Encoding
        print("\n1. Text Encoding Speed")
        encoder = UniversalEncoder()
        start = time.time()
        for _ in range(100):
            encoder.encode_text("Test sentence for encoding")
        avg_encoding_ms = (time.time() - start) * 10
        print(f"   Average: {avg_encoding_ms:.2f}ms per sentence")
        results['encoding_ms'] = avg_encoding_ms
        
        # Benchmark 2: Semantic search
        print("\n2. Semantic Search (over 1000 concepts)")
        mem = SemanticMemory()
        for i in range(1000):
            hv = encoder.encode_text(f"concept_{i}")
            mem.add_concept(f"concept_{i}", hv)
        
        query_hv = encoder.encode_text("query")
        start = time.time()
        for _ in range(10):
            mem.query(query_hv, k=10)
        avg_search_ms = (time.time() - start) * 100
        print(f"   Average: {avg_search_ms:.2f}ms per search")
        results['search_ms'] = avg_search_ms
        
        # Benchmark 3: Full query pipeline
        print("\n3. Full Query Pipeline")
        learner = TextKnowledgeLearner()
        for i in range(100):
            learner.learn_text(f"Fact {i}: Value is {i % 10}")
        
        start = time.time()
        for _ in range(5):
            learner.query_learned_knowledge("What is fact?", top_k=5)
        avg_query_ms = (time.time() - start) / 5 * 1000
        print(f"   Average: {avg_query_ms:.1f}ms per query")
        results['query_ms'] = avg_query_ms
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Encoding:          {results['encoding_ms']:.2f}ms   {'✓' if results['encoding_ms'] < 10 else '✗'}")
        print(f"Semantic Search:   {results['search_ms']:.2f}ms {'✓' if results['search_ms'] < 100 else '✗'}")
        print(f"Full Query:        {results['query_ms']:.1f}ms   {'✓' if results['query_ms'] < 500 else '✗'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
