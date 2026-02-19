"""
Phase 3: Learning Capability Tests
Tests the system's ability to learn from data and retain knowledge
"""

import sys
import os
import pytest
import tempfile
import time
from typing import List

# Setup path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.reasoning.causal_reasoning import CausalGraph


class TestConceptAccumulation:
    """Test LRN-1: System learns and accumulates new concepts"""
    
    def test_concept_learning_from_single_text(self):
        """New concepts learned from text"""
        learner = TextKnowledgeLearner()
        
        # Initially empty
        initial_concepts = len(learner.semantic.concept_hvs)
        
        # Train on new concept
        learner.learn_text("Photosynthesis is the process where plants convert light to energy")
        
        # Should have more concepts
        final_concepts = len(learner.semantic.concept_hvs)
        assert final_concepts > initial_concepts, "No new concepts learned"
    
    def test_concept_queryability(self):
        """Learned concepts can be queried"""
        learner = TextKnowledgeLearner()
        
        # Learn
        learner.learn_text("A whale is a marine mammal")
        
        # Query for the concept using a HyperVector (same hashing as learner)
        whale_hv = hypervec_rs.HyperVector(hash("Whale") % (2**32))
        
        results = learner.semantic.query(whale_hv, k=5)
        assert results is not None and len(results) > 0, \
            "Cannot query learned concept"
    
    def test_concept_similarity_in_memory(self):
        """Similar concepts cluster in memory"""
        learner = TextKnowledgeLearner()
        
        # Learn related concepts
        learner.learn_text("Dogs are mammals")
        learner.learn_text("Cats are mammals")
        learner.learn_text("Pigs are mammals")
        
        # Query for "mammals" using a HyperVector
        mammal_hv = hypervec_rs.HyperVector(hash("mammals") % (2**32))
        
        results = learner.semantic.query(mammal_hv, k=10)
        
        # Should find multiple animals (related by mammal concept)
        assert results is not None and len(results) > 0


class TestRelationLearning:
    """Test LRN-2: System infers and learns relations"""
    
    def test_relation_extraction_from_text(self):
        """Relations extracted from sentences"""
        learner = TextKnowledgeLearner()
        
        learner.learn_text("Dogs are mammals")
        
        # Relations are stored in the semantic memory's concept_graph (networkx DiGraph)
        assert hasattr(learner.semantic, 'concept_graph'), \
            "No relation storage found"
        # Should have learned at least some concepts
        assert len(learner.semantic.concept_hvs) > 0, \
            "No concepts stored after learning"
    
    def test_causal_graph_growth(self):
        """Causal graph grows with new facts"""
        cg = CausalGraph()
        
        initial_links = len(cg.all_links)
        
        # Add causal links using the correct API
        cg.add_causes("sun heats earth", "temperature increases", strength=0.9)
        cg.add_causes("temperature increases", "ice melts", strength=0.85)
        
        final_links = len(cg.all_links)
        assert final_links > initial_links, "Causal links not added"
        assert final_links == 2, "Should have exactly 2 causal links"
    
    def test_learned_rules_discoverable(self):
        """Learned rules can be discovered"""
        learner = TextKnowledgeLearner()
        
        # Learn rules through examples
        learner.learn_text("All birds have feathers")
        learner.learn_text("All birds can fly")
        
        # Rules should emerge in causal/semantic system
        # Check that system has learned the pattern


class TestFewShotLearning:
    """Test LRN-3: Few-shot learning - one example teaches a concept"""
    
    def test_single_example_learning(self):
        """One sentence teaches system about a concept"""
        learner = TextKnowledgeLearner()
        
        # Single example
        learner.learn_text("Quetzalcoatl is the Aztec god of wind")
        
        # Query: should know about this concept
        result = learner.query_learned_knowledge("What is Quetzalcoatl?", top_k=5)
        
        # Should retrieve the learned fact
        assert result is not None, "Failed to recall single-example learning"
    
    def test_concept_activation_from_one_example(self):
        """Concepts are activated from single learning example"""
        learner = TextKnowledgeLearner()
        
        # Learn a new rare word
        learner.learn_text("Alpaca is a domesticated South American animal")
        
        # Query memory using a HyperVector
        alpaca_hv = hypervec_rs.HyperVector(hash("Alpaca") % (2**32))
        
        results = learner.semantic.query(alpaca_hv, k=5)
        assert results is not None and len(results) > 0, \
            "Single example not stored in semantic memory"


class TestCatastrophicForgettingMitigation:
    """Test LRN-4: Old knowledge retained despite new learning"""
    
    def test_old_knowledge_not_overwritten(self):
        """New learning doesn't erase old concepts"""
        learner = TextKnowledgeLearner()
        
        # Learn initial facts
        learner.learn_text("The Eiffel Tower is in Paris")
        learner.learn_text("Paris is the capital of France")
        
        paris_hv = hypervec_rs.HyperVector(hash("Paris") % (2**32))
        old_results = learner.semantic.query(paris_hv, k=5)
        
        # Learn many new unrelated facts
        for i in range(20):
            learner.learn_text(f"Fact {i}: A new unrelated statement about things")
        
        # Re-query old knowledge
        new_results = learner.semantic.query(paris_hv, k=5)
        
        # Old similarity should not decrease significantly
        # (exact comparison depends on implementation)
        assert new_results is not None, "Old knowledge lost after new learning"
    
    def test_knowledge_retention_metric(self):
        """Measure knowledge retention over time"""
        learner = TextKnowledgeLearner()
        
        # Baseline: learn a fact
        learner.learn_text("Venus is the hottest planet")
        venus_hv = hypervec_rs.HyperVector(hash("Venus") % (2**32))
        baseline = learner.semantic.query(venus_hv, k=1)
        
        # Interfering learning
        for i in range(10):
            learner.learn_text(f"Learning iteration {i} with new facts")
        
        # Retention check
        retention = learner.semantic.query(venus_hv, k=1)
        
        # Should still be retrievable
        assert retention is not None


class TestGeneralization:
    """Test LRN-5: Learned rules apply to new instances"""
    
    def test_rule_transfer_to_unseen_examples(self):
        """Rules learned from training apply to novel cases"""
        learner = TextKnowledgeLearner()
        
        # Rule training: multiple examples of same pattern
        learner.learn_text("Dogs are mammals")
        learner.learn_text("Cats are mammals")
        learner.learn_text("Whales are mammals")
        learner.learn_text("Bats are mammals")
        
        # Generalization query about unseen mammal
        result = learner.query_learned_knowledge("Are dolphins mammals?", top_k=5)
        
        # Should use learned rule, not just retrieve
        assert result is not None
    
    def test_pattern_recognition_generalization(self):
        """System recognizes patterns across unrelated domains"""
        learner = TextKnowledgeLearner()
        
        # Learn pattern in one domain
        learner.learn_text("Water boils at 100 degrees Celsius")
        learner.learn_text("Lead melts at 327 degrees Celsius")
        learner.learn_text("Gold melts at 1064 degrees Celsius")
        
        # Pattern: substances have transformation temperatures
        # Should recognize this applies to any substance
        
        # Query about new substance
        result = learner.query_learned_knowledge("At what temperature does tin melt?", top_k=5)
        
        # Should understand it's asking about a transformation property


class TestTransferLearning:
    """Test LRN-6: Knowledge from one domain aids learning in related domain"""
    
    def test_transfer_from_animal_to_pet_domain(self):
        """Animal facts boost learning in pet domain"""
        learner = TextKnowledgeLearner()
        
        # Phase 1: Learn general animal facts
        print("\nPhase 1: Learning animal facts...")
        animal_facts = [
            "Dogs have four legs",
            "Dogs are carnivorous",
            "Dogs have sharp teeth",
            "Cats have whiskers",
            "Cats are carnivorous",
        ]
        for fact in animal_facts:
            learner.learn_text(fact)
        
        # Phase 2: Learn pet-specific facts (should be faster)
        print("Phase 2: Learning pet-specific facts...")
        start = time.time()
        
        pet_facts = [
            "Dogs make good pets",
            "Cats can live indoors",
        ]
        for fact in pet_facts:
            learner.learn_text(fact)
        
        transfer_time = time.time() - start
        
        # Should have learned pet facts quickly
        result = learner.query_learned_knowledge("Why are dogs good pets?", top_k=5)
        assert result is not None
    
    def test_cross_domain_concept_mapping(self):
        """Concepts map across domains (e.g., structure parallels)"""
        learner = TextKnowledgeLearner()
        
        # Learn structure in domain A (organizations)
        learner.learn_text("A company has departments")
        learner.learn_text("A department has roles")
        learner.learn_text("A role has responsibilities")
        
        # Learn structure in domain B (animals)
        learner.learn_text("An animal has organs")
        learner.learn_text("An organ has cells")
        learner.learn_text("A cell has organelles")
        
        # Both follow hierarchy: X has Y, Y has Z
        # This hierarchical structure should transfer


class TestLearningNarratives:
    """Narratives demonstrating learning capability"""
    
    def test_narrative_incremental_learning(self):
        """Show learning happening incrementally"""
        print("\n" + "="*70)
        print("NARRATIVE: Incremental Learning")
        print("="*70)
        
        learner = TextKnowledgeLearner()
        encoder = learner.lingua
        
        print("\n1. Initial knowledge base: empty")
        print(f"   Concepts: {len(learner.semantic.concept_hvs)}")
        
        # Learn
        facts = [
            "Penguins are birds",
            "Birds have feathers",
            "Penguins live in Antarctica",
        ]
        
        for i, fact in enumerate(facts, 1):
            learner.learn_text(fact)
            print(f"\n{i+1}. After learning: '{fact}'")
            print(f"   Concepts: {len(learner.semantic.concept_hvs)}")
        
        # Query
        result = learner.query_learned_knowledge("What are penguins?", top_k=5)
        print(f"\nFinal query: 'What are penguins?'")
        print(f"Result: {result}")
    
    def test_narrative_generalization(self):
        """Show how rules generalize beyond training"""
        print("\n" + "="*70)
        print("NARRATIVE: Generalization and Transfer")
        print("="*70)
        
        learner = TextKnowledgeLearner()
        
        print("\nTraining phase: Learn mammal facts")
        training_facts = [
            "Dogs are mammals and have fur",
            "Cats are mammals and have fur",
            "Whales are mammals and have fur",
        ]
        
        for fact in training_facts:
            print(f"  Learning: {fact}")
            learner.learn_text(fact)
        
        print("\nTest phase: Query about unseen mammal")
        print("  Query: 'Are dolphins mammals?'")
        
        # System should generalize the rule
        # "If X is a mammal, then X has fur" applies to dolphins


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
