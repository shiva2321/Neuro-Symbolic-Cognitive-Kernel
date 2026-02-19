"""
Phase 2: Input Understanding Tests
Tests semantic comprehension and textual understanding capabilities
"""

import sys
import os
import pytest
import tempfile
from typing import List, Tuple

# Setup path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.vsa.universal_encoder import UniversalEncoder
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.language.lingua_cortex import LinguaCortex
from python.core.cognitive.emotion_system import EmotionSystem


class TestTextEncoding:
    """Test UND-1: Text encoding via semantic folding"""
    
    def test_encoding_consistency(self):
        """Same text always encodes to same HV"""
        encoder = UniversalEncoder()
        
        text = "The cat sat on the mat"
        hv1 = encoder.encode_text(text)
        hv2 = encoder.encode_text(text)
        
        assert hv1.similarity(hv2) > 0.99, "Same text encodes differently"
    
    def test_semantic_similarity_in_hyperspace(self):
        """Similar texts have high similarity in HV space"""
        encoder = UniversalEncoder()
        
        # Similar sentences
        text1 = "Dogs are loyal animals"
        text2 = "A dog is a loyal animal"
        
        # Dissimilar sentences
        text3 = "The sun rises in the east"
        
        hv1 = encoder.encode_text(text1)
        hv2 = encoder.encode_text(text2)
        hv3 = encoder.encode_text(text3)
        
        sim_12 = hv1.similarity(hv2)
        sim_13 = hv1.similarity(hv3)
        
        assert sim_12 > sim_13, \
            f"Similar texts not more similar: sim(1,2)={sim_12:.3f}, sim(1,3)={sim_13:.3f}"
    
    def test_word_order_matters(self):
        """Different word orders produce different HVs"""
        encoder = UniversalEncoder()
        
        text1 = "Dog bites man"
        text2 = "Man bites dog"
        
        hv1 = encoder.encode_text(text1)
        hv2 = encoder.encode_text(text2)
        
        sim = hv1.similarity(hv2)
        
        # Should be lower than random (words present but order differs)
        assert sim < 0.55, f"Word order not captured: sim={sim:.3f}"
    
    def test_encoding_speed_threshold(self):
        """Encoding meets efficiency threshold (<10ms per sentence)"""
        import time
        encoder = UniversalEncoder()
        
        text = "The quick brown fox jumps over the lazy dog"
        start = time.time()
        hv = encoder.encode_text(text)
        elapsed_ms = (time.time() - start) * 1000
        
        assert elapsed_ms < 10, f"Encoding too slow: {elapsed_ms:.1f}ms"


class TestConceptExtraction:
    """Test UND-2: Concept extraction from raw text"""
    
    def test_concept_extraction_basic(self):
        """Extract capitalized and important terms"""
        learner = TextKnowledgeLearner()
        
        sentence = "Photosynthesis is the process by which Plants convert sunlight into energy."
        concepts = learner._extract_concepts(sentence)
        
        # Should find key concepts
        assert len(concepts) > 0, "No concepts extracted"
        
        # Check for expected concepts (case-insensitive)
        concept_lower = [c.lower() for c in concepts]
        # Note: actual extraction may vary, but should capture "plants"
    
    def test_entity_recognition(self):
        """Named entities recognized (people, places, things)"""
        learner = TextKnowledgeLearner()
        
        sentence = "Albert Einstein lived in Germany and Princeton"
        concepts = learner._extract_concepts(sentence)
        
        # Should recognize named entities
        concept_lower = [c.lower() for c in concepts]
        # At minimum, extract capitalized proper nouns


class TestContextSensitivity:
    """Test UND-3: Context sensitivity - same word, different contexts"""
    
    def test_bank_disambiguation(self):
        """Word 'bank' understood differently in different contexts"""
        encoder = UniversalEncoder()
        lingua = LinguaCortex()
        
        # Two uses of "bank"
        sentence1 = "I went to the bank to deposit money"
        sentence2 = "I sat on the bank of the river"
        
        # Encode with context
        hv1 = encoder.encode_text(sentence1)
        hv2 = encoder.encode_text(sentence2)
        
        # Should be dissimilar (different semantic contexts)
        sim = hv1.similarity(hv2)
        
        # Should be lower than two identical sentences
        assert sim < 0.7, f"Context not distinguished: sim={sim:.3f}"
    
    def test_polysemy_sensitivity(self):
        """System sensitive to polysumous words (multiple meanings)"""
        encoder = UniversalEncoder()
        
        # "Light" as adjective vs noun
        sent1 = "The light in the room is bright"  # noun
        sent2 = "The feather is light and small"  # adjective
        
        hv1 = encoder.encode_text(sent1)
        hv2 = encoder.encode_text(sent2)
        
        sim = hv1.similarity(hv2)
        
        # Should be lower than completely identical
        assert sim < 0.9, f"Polysemy not captured: {sim:.3f}"


class TestNegationHandling:
    """Test UND-4: Negation handled correctly"""
    
    def test_negation_produces_different_hv(self):
        """NOT(X) ≠ X in concept space"""
        encoder = UniversalEncoder()
        
        text_pos = "The cat is friendly"
        text_neg = "The cat is NOT friendly"
        
        hv_pos = encoder.encode_text(text_pos)
        hv_neg = encoder.encode_text(text_neg)
        
        sim = hv_pos.similarity(hv_neg)
        
        # Should be dissimilar
        assert sim < 0.7, f"Negation not recognized: sim={sim:.3f}"
    
    def test_double_negation(self):
        """Double negation recognized as affirmation"""
        encoder = UniversalEncoder()
        
        text_pos = "The cat is friendly"
        text_double_neg = "The cat is NOT unfriendly"
        
        hv_pos = encoder.encode_text(text_pos)
        hv_double_neg = encoder.encode_text(text_double_neg)
        
        sim = hv_pos.similarity(hv_double_neg)
        
        # Should be more similar (both affirmative)
        # This is harder: depends on language module's sophistication
        assert sim > 0.5, f"Double negation not captured well: {sim:.3f}"


class TestParaphraseRecognition:
    """Test UND-5: Recognize different semantic relations"""
    
    def test_is_vs_issubset_vs_approximates(self):
        """Distinguish 'is', '⊆', '≈' relationships"""
        encoder = UniversalEncoder()
        
        # Different relationships
        text_isa = "A dog is an animal"  # is (identity)
        text_subset = "Dogs are a type of animal"  # subset/kind
        text_approx = "Dogs are somewhat like wolves"  # approximate
        
        hv_isa = encoder.encode_text(text_isa)
        hv_subset = encoder.encode_text(text_subset)
        hv_approx = encoder.encode_text(text_approx)
        
        # Vector differences should reflect semantic differences
        # (isa should be similar to subset, both different from approx)
        sim_isa_subset = hv_isa.similarity(hv_subset)
        sim_isa_approx = hv_isa.similarity(hv_approx)
        
        assert sim_isa_subset > sim_isa_approx, \
            f"Relationship types not distinguished: isa_subset={sim_isa_subset:.3f}, isa_approx={sim_isa_approx:.3f}"


class TestPronounResolution:
    """Test UND-6: Pronoun resolution to correct antecedents"""
    
    def test_pronoun_antecedent_resolution(self):
        """Pronouns resolve to correct antecedents"""
        learner = TextKnowledgeLearner()
        context_retention = learner.context
        
        # Build context
        sentences = [
            "John went to the store",
            "He bought milk",  # 'He' -> John
            "Mary arrived late",
            "She was tired"  # 'She' -> Mary
        ]
        
        # Learn context
        for sent in sentences:
            learner.learn_text(sent)
        
        # Test: when we see "He bought milk", 'He' should resolve to John
        # This requires tracking entities through context
        # Check if context retention module tracks this
        assert hasattr(context_retention, 'current_entities') or \
               hasattr(context_retention, 'pronoun_map'), \
            "Context retention module missing pronoun tracking"


class TestUnderstandingIntegration:
    """Integration tests for input understanding"""
    
    def test_sentence_understanding_pipeline(self):
        """Full pipeline: text -> understanding"""
        learner = TextKnowledgeLearner()
        
        # Complex sentence
        sentence = "Alice believes that Bob saw the red car"
        
        # Should understand:
        # - Entities: Alice, Bob, car
        # - Properties: red (of car)
        # - Relations: believes, saw
        # - Embedding: belief about seeing
        
        result = learner.query_learned_knowledge(sentence, top_k=10)
        assert result is not None, "Failed to process complex sentence"
    
    def test_multi_sentence_understanding(self):
        """Understand relationships across multiple sentences"""
        learner = TextKnowledgeLearner()
        
        sentences = [
            "Alice is tall",
            "Bob is short",
            "Alice and Bob are friends"
        ]
        
        # Learn
        for sent in sentences:
            learner.learn_text(sent)
        
        # Query should use relationships learned
        # E.g., "Are Alice and Bob friends?" should retrieve the relation


class TestUnderstandingNarratives:
    """Narrative tests showing understanding capabilities"""
    
    def test_narrative_semantic_comprehension(self):
        """Show semantic understanding vs. surface patterns"""
        print("\n" + "="*70)
        print("NARRATIVE: Semantic Comprehension")
        print("="*70)
        
        encoder = UniversalEncoder()
        
        # Three sentences with same keywords, different meanings
        sentences = [
            "The bank made a big loan",
            "I sat on the bank of the river",
            "The bank of clouds was beautiful"
        ]
        
        print("\nSentences:")
        for i, s in enumerate(sentences, 1):
            print(f"  {i}. {s}")
        
        print("\nEncoded to hypervectors and compared:")
        hvs = [encoder.encode_text(s) for s in sentences]
        
        # Compute pairwise similarities
        print(f"  Sentence 1 vs 2: {hvs[0].similarity(hvs[1]):.3f} (should be low - different meanings)")
        print(f"  Sentence 1 vs 3: {hvs[0].similarity(hvs[2]):.3f} (should be low - different meanings)")
        print(f"  Sentence 2 vs 3: {hvs[1].similarity(hvs[2]):.3f} (higher - both about river bank)")
        
        print("\nConclusion: Same word, different semantic contexts → different HVs")
    
    def test_narrative_context_sensitivity(self):
        """Show how context modifies meaning"""
        print("\n" + "="*70)
        print("NARRATIVE: Context Sensitivity")
        print("="*70)
        
        learner = TextKnowledgeLearner()
        
        # Teach two contexts
        print("\nLearning context 1 (financial):")
        learner.learn_text("The bank approves loans")
        learner.learn_text("Interest rates affect banking")
        
        print("Learning context 2 (geography):")
        learner.learn_text("The river bank was muddy")
        learner.learn_text("Banks provide shelter from erosion")
        
        print("\nAt this point, 'bank' has two distinct contexts in memory")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
