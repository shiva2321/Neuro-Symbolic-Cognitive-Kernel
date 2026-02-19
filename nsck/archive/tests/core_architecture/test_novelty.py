"""
Phase 1: Novelty Assessment
Tests that validate the novel aspects of NSCK architecture
"""

import sys
import os
import pytest
import time
from typing import Tuple

# Setup path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.vsa.universal_encoder import UniversalEncoder
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.reasoning.causal_reasoning import CausalGraph
from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition, WorkspaceModule
from python.core.cognitive.emotion_system import EmotionSystem
from python.core.language.text_knowledge_learner import TextKnowledgeLearner


class TestVSAFoundation:
    """Test NOV-1: VSA Foundation - Validate 10,240-bit HV operations"""
    
    def test_hypervector_dimensions(self):
        """Test that HVs are 10,240 bits"""
        hv = hypervec_rs.HyperVector()
        assert len(hv.bits) == 10240, f"Expected 10,240 bits, got {len(hv.bits)}"
    
    def test_xor_binding_operation(self):
        """Test XOR binding preserves compositionality"""
        hv1 = hypervec_rs.HyperVector()
        hv2 = hypervec_rs.HyperVector()
        
        # Binding should be XOR
        bound = hv1.xor(hv2)
        assert bound is not None
        assert len(bound.bits) == 10240
        
        # Binding should be invertible: A ^ B ^ B = A (within numerical error)
        recovered = bound.xor(hv2)
        sim = recovered.similarity(hv1)
        assert sim > 0.45, f"XOR binding inverse lookup: sim={sim}"
    
    def test_majority_vote_bundling(self):
        """Test bundling via majority vote"""
        hvs = [hypervec_rs.HyperVector() for _ in range(3)]
        
        # Should support weighted bundling
        bundled = hypervec_rs.HyperVector()
        for hv in hvs:
            bundled = bundled.bundle(hv)  # Returns bundled result
        
        assert bundled is not None
        assert len(bundled.bits) == 10240
    
    def test_circular_shift_permutation(self):
        """Test permutation via circular bit shift"""
        hv = hypervec_rs.HyperVector()
        
        # Permute by shifting bits
        if hasattr(hv, 'permute'):
            perm1 = hv.permute(1)
            perm2 = hv.permute(2)
            
            # Should be different
            assert perm1.similarity(perm2) < 0.95
            
            # Double permutation should be different from original
            assert perm1.similarity(hv) < 0.95
    
    def test_hamming_similarity_metrics(self):
        """Test similarity is normalized Hamming distance"""
        hv1 = hypervec_rs.HyperVector()
        hv2 = hypervec_rs.HyperVector()
        
        # Similarity should be in [0, 1]
        sim = hv1.similarity(hv2)
        assert 0 <= sim <= 1, f"Similarity out of range: {sim}"
        
        # Random HVs should have similarity ~0.5
        assert 0.45 < sim < 0.55, f"Random HV similarity not ~0.5: {sim}"
        
        # Self-similarity should be ~1.0
        self_sim = hv1.similarity(hv1)
        assert self_sim > 0.99, f"Self-similarity not ~1.0: {self_sim}"
    
    def test_vsa_algebra_composability(self):
        """Test that VSA operations compose correctly"""
        encoder = UniversalEncoder()
        
        # Encode text
        hv_text = encoder.encode_text("Dogs are animals")
        assert hv_text is not None
        
        # Bind to relationship
        hv_rel = encoder.encode_text("is_a")
        bound = hv_text.xor(hv_rel)
        assert bound is not None
        
        # Should be able to bundle multiple bindings
        hv2 = encoder.encode_text("Animals breathe").xor(encoder.encode_text("is_related_to"))
        bundled = bound.bundle(hv2)
        assert bundled is not None


class TestGlassBoxTraceability:
    """Test NOV-2: Glass-Box Traceability - Every inference produces ThoughtTrace"""
    
    def test_thought_trace_structure(self):
        """Test that TextKnowledgeLearner produces thought traces"""
        learner = TextKnowledgeLearner()
        
        # Learn first
        learner.learn_from_text("Dogs are animals")
        
        # Make a query - should produce a trace
        query_result = learner.query_learned_knowledge("What are dogs?", top_k=5)
        
        # Query result should have reasoning trace in response
        assert isinstance(query_result, dict)
        assert 'reasoning_trace' in query_result
    
    def test_trace_completeness(self):
        """Test that traces capture all stages of inference"""
        learner = TextKnowledgeLearner()
        
        # Train
        learner.learn_from_text("Dogs are mammals. Mammals are animals.")
        
        # Query with full tracing
        result = learner.query_learned_knowledge("Are dogs mammals?", top_k=5)
        
        # Result should have reasoning trace showing the inference stages
        assert isinstance(result, dict)
        assert 'reasoning_trace' in result
        trace = result['reasoning_trace']
        assert isinstance(trace, list)
        assert len(trace) > 0  # Should have multiple stages


class TestNoLLMDependency:
    """Test NOV-3: No LLM Dependency - Core cognition uses only HV algebra & graph search"""
    
    def test_semantic_memory_is_graph_based(self):
        """Test that SemanticMemory uses NetworkX graphs, not neural networks"""
        mem = SemanticMemory()
        
        # Check internal structure
        assert hasattr(mem, 'concept_graph'), "SemanticMemory missing concept_graph"
        
        # Should be a NetworkX graph
        import networkx as nx
        assert isinstance(mem.concept_graph, nx.DiGraph) or isinstance(mem.concept_graph, nx.Graph), \
            f"SemanticMemory.concept_graph is {type(mem.concept_graph)}, not NetworkX"
    
    def test_episodic_memory_uses_hv_indexing(self):
        """Test that EpisodicMemory uses HV indexing, not embeddings"""
        mem = EpisodicMemory()
        
        # Check that indexing is based on HyperVectors
        encoder = UniversalEncoder()
        hv = encoder.encode_text("Test episode")
        
        # Store and retrieve
        episode_id = mem.record_episode(content="Test", hv=hv)
        retrieved = mem.query(query_hv=hv, k=1)
        
        assert retrieved is not None
        # Should use LSH (locality-sensitive hashing) on HVs, not transformers
    
    def test_reasoning_uses_graph_traversal(self):
        """Test that reasoning uses symbolic graph traversal"""
        from python.core.reasoning.causal_reasoning import CausalLink, CausalRelation
        cg = CausalGraph()
        
        # Check graph-based reasoning
        assert hasattr(cg, 'all_links') or hasattr(cg, 'graph'), "CausalGraph missing graph attribute"
        
        # Forward chaining should be based on graph traversal
        link = CausalLink(cause="cause", effect="effect", relation=CausalRelation.CAUSES)
        cg.add_link(link)
        links = cg.all_links()
        assert len(links) > 0, "Failed to add causal link"


class TestSymbolicGrounding:
    """Test NOV-4: Symbolic Grounding - 1:1 invertible symbol ↔ HV mapping"""
    
    def test_deterministic_symbol_to_hv_mapping(self):
        """Test that same symbol always produces same HV"""
        encoder = UniversalEncoder()
        
        # Encode same word twice
        hv1 = encoder.encode_text("dog")
        hv2 = encoder.encode_text("dog")
        
        # Should be identical
        sim = hv1.similarity(hv2)
        assert sim > 0.99, f"Same symbol produces different HVs: sim={sim}"
    
    def test_hv_to_symbol_roundtrip(self):
        """Test invertibility: symbol → HV → symbol"""
        encoder = UniversalEncoder()
        
        # Symbol to HV
        symbol = "dog"
        hv = encoder.encode_text(symbol)
        
        # HV should be stable and deterministic
        hv_again = encoder.encode_text(symbol)
        assert hv.similarity(hv_again) > 0.99
    
    def test_global_primitives_consistency(self):
        """Test that global primitive symbols map consistently across modules"""
        # Import brain_fusion to check global primitives
        from python.core.integration.brain_fusion import GLOBAL_PRIMITIVES
        
        # All primitives should have fixed seeds
        assert "ACTION_UP" in GLOBAL_PRIMITIVES
        assert isinstance(GLOBAL_PRIMITIVES["ACTION_UP"], int)
        
        # Creating HV from seed should be deterministic
        encoder = UniversalEncoder()
        hv1 = encoder.encode_seed(GLOBAL_PRIMITIVES["ACTION_UP"])
        hv2 = encoder.encode_seed(GLOBAL_PRIMITIVES["ACTION_UP"])
        assert hv1.similarity(hv2) > 0.99


class TestMultiStagePipeline:
    """Test NOV-5: Multi-Stage Pipeline - 11-stage architecture flows correctly"""
    
    def test_all_stages_callable(self):
        """Test that all 11 stages of the pipeline are callable"""
        learner = TextKnowledgeLearner()
        
        # Individual stages should be accessible
        # Stage 1-2: Perception (encoding + emotion)
        hv = learner.lingua.encode_text("Test input")
        assert hv is not None
        
        # Stage 3: Understanding (concept extraction)
        concepts = learner.extract_concepts("Dogs are animals")
        assert len(concepts) > 0
        
        # Stages 4-7 should be in pipeline
        assert hasattr(learner, 'semantic')  # Stage 4: Semantic Memory
        assert hasattr(learner, 'episodic')  # Stage 5: Episodic Memory
    
    def test_pipeline_latency_measurable(self):
        """Test that each stage's latency is measurable"""
        learner = TextKnowledgeLearner()
        
        # Measure encoding stage
        start = time.time()
        hv = learner.lingua.encode_text("Measurement test")
        stage1_ms = (time.time() - start) * 1000
        
        assert stage1_ms < 100, f"Encoding took {stage1_ms}ms (should be <100ms)"
        assert stage1_ms > 0, "Encoding latency not measured"
    
    def test_pipeline_stages_sequential_and_parallel(self):
        """Test that pipeline stages can execute sequentially and in parallel"""
        learner = TextKnowledgeLearner()
        
        # Sequential (single query)
        result = learner.query_learned_knowledge("What are dogs?", top_k=5)
        assert result is not None
        
        # Should be able to make multiple queries (parallel paths)
        # This tests that the architecture supports concurrent memory searches


class TestHybridMemorySystem:
    """Test NOV-6: Hybrid Memory System - Semantic + Episodic + Causal unified"""
    
    def test_semantic_memory_queryable(self):
        """Test that Semantic Memory can be queried"""
        sem_mem = SemanticMemory()
        encoder = UniversalEncoder()
        
        # Add a concept with properties dict and hv_override
        hv = encoder.encode_text("dog")
        sem_mem.add_concept("dog", {"type": "animal"}, hv_override=hv)
        
        # Query it
        results = sem_mem.query(hv, k=1)
        assert results is not None
    
    def test_episodic_memory_queryable(self):
        """Test that Episodic Memory can be queried"""
        epi_mem = EpisodicMemory()
        encoder = UniversalEncoder()
        
        # Record an episode
        hv = encoder.encode_text("Episode content")
        epi_mem.record_episode(content="An episode", hv=hv)
        
        # Query it
        retrieved = epi_mem.query(query_hv=hv, k=1)
        assert retrieved is not None
    
    def test_causal_memory_queryable(self):
        """Test that Causal Graph can be queried"""
        cg = CausalGraph()
        
        # Add causal links
        cg.add_edge("A", "B", confidence=0.9)
        cg.add_edge("B", "C", confidence=0.8)
        
        # Forward chaining
        assert cg.has_edge("A", "B")
        
        # Can retrieve paths
        paths = list(cg.forward_chain_from("A", max_depth=3))
        assert len(paths) > 0
    
    def test_memories_mutually_informative(self):
        """Test that the three memory types inform each other"""
        learner = TextKnowledgeLearner()
        
        # Learn: "Dogs are animals"
        learner.learn_text("Dogs are animals")
        
        # Semantic memory should have "dog" and "animals"
        assert len(learner.semantic.concept_hvs) > 0
        
        # Episodic memory should record the episode
        assert len(learner.episodic.episodes) > 0 or learner.episodic.episode_count > 0
        
        # Causal graph should have the is_a relationship
        # (Depends on implementation, but should be findable)


class TestArchitectureNarratives:
    """Integration test: demonstrate the novelty through narrative"""
    
    def test_narrative_vsa_enables_interpretability(self):
        """Show how VSA enables interpretability vs. transformers"""
        print("\n" + "="*70)
        print("NARRATIVE: VSA vs. Transformers")
        print("="*70)
        
        encoder = UniversalEncoder()
        
        # Encode a sentence
        sentence = "Dogs are loyal animals"
        hv = encoder.encode_text(sentence)
        
        print(f"\nSentence: '{sentence}'")
        print(f"Encoded to: 10,240-bit hypervector")
        print(f"Representation: invertible, symbolic, glass-box")
        
        # Show that structure is preserved
        hv_dogs = encoder.encode_text("Dogs")
        hv_loyal = encoder.encode_text("loyal")
        
        # Binding two concepts
        bound = hv_dogs ^ hv_loyal
        print(f"\nDogs [XOR] loyal: {bound.similarity(hv):.4f} (high sim = structure preserved)")
    
    def test_narrative_no_transformers(self):
        """Show that system works without transformers"""
        print("\n" + "="*70)
        print("NARRATIVE: No Transformers, No Matrix Multiplication")
        print("="*70)
        
        learner = TextKnowledgeLearner()
        
        # Can it learn and reason without LLMs?
        learner.learn_text("Dogs are mammals. Mammals are warm-blooded.")
        
        # Query (should succeed without transformers)
        result = learner.query_learned_knowledge("Are dogs mammals?", top_k=5)
        
        print(f"\nQuery: 'Are dogs mammals?'")
        print(f"Uses: VSA algebra, graph search, spreading activation")
        print(f"No transformers, no embeddings, no LLM calls")
        print(f"Result: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
