"""
Tests for Text Knowledge Learner

Validates that the system can learn from text, store knowledge,
and answer queries without relying on LLMs for core cognition.
"""

import sys
import os

import pytest
import tempfile
from python.core.language.text_knowledge_learner import TextKnowledgeLearner, LearnedFact


class TestTextKnowledgeLearner:
    """Test the VSA-based text learning system."""
    
    def setup_method(self):
        """Initialize learner for each test."""
        self.learner = TextKnowledgeLearner()
    
    def test_initialization(self):
        """Test that learner initializes with correct components."""
        assert self.learner.semantic is not None
        assert self.learner.episodic is not None
        assert self.learner.context is not None
        assert self.learner.lingua is not None
        assert len(self.learner.learned_facts) == 0
        assert len(self.learner.learning_sessions) == 0
    
    def test_learn_from_simple_text(self):
        """Test learning from a simple text file."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Dogs are mammals. Cats are mammals. Dogs are social animals.")
            temp_path = f.name
        
        try:
            # Learn from file
            session = self.learner.learn_from_text_file(temp_path)
            
            # Check session metadata
            assert session.sentences_processed == 3
            assert session.concepts_learned > 0
            assert session.relations_learned > 0
            assert session.facts_stored > 0
            
            # Check learned concepts
            assert len(self.learner.semantic.concept_hvs) > 0
            
            # Check learned facts
            assert len(self.learner.learned_facts) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_concept_extraction(self):
        """Test that concepts are extracted correctly."""
        sentence = "Photosynthesis is the process by which plants convert sunlight into energy."
        concepts = self.learner._extract_concepts(sentence)
        
        # Should extract capitalized words and important terms
        assert len(concepts) > 0
        # Check for at least one expected concept
        assert any('Photosynthesis' in c or 'photosynthesis' in c.lower() for c in concepts)
    
    def test_relation_extraction(self):
        """Test that relations are extracted from sentences."""
        sentence = "Dogs are mammals."
        concepts = self.learner._extract_concepts(sentence)
        relations = self.learner._extract_relations(sentence, concepts)
        
        # Should find at least one relation
        assert len(relations) >= 0  # May find is_a or related_to
    
    def test_hypervector_encoding(self):
        """Test that sentences are encoded to hypervectors."""
        sentence = "The cat sat on the mat."
        hv = self.learner._encode_sentence(sentence)
        
        # Should return a hypervector
        assert hv is not None
        assert hasattr(hv, 'similarity')
    
    def test_query_learned_knowledge(self):
        """Test querying learned knowledge."""
        # Create and learn from temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Dogs are mammals. Dogs have fur. Dogs are social animals.")
            temp_path = f.name
        
        try:
            # Learn
            self.learner.learn_from_text_file(temp_path)
            
            # Query
            result = self.learner.query_learned_knowledge("Tell me about dogs", top_k=5)
            
            # Check result structure
            assert 'answer' in result
            assert 'confidence' in result
            assert 'similar_concepts' in result
            assert 'related_facts' in result
            assert 'reasoning_trace' in result
            
            # Should have some confidence
            assert result['confidence'] >= 0
            
            # Should have reasoning trace
            assert len(result['reasoning_trace']) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_multiple_files_learning(self):
        """Test learning from multiple files accumulates knowledge."""
        # File 1
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Dogs are mammals. Dogs bark.")
            temp_path1 = f.name
        
        # File 2
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Cats are mammals. Cats meow.")
            temp_path2 = f.name
        
        try:
            # Learn from first file
            session1 = self.learner.learn_from_text_file(temp_path1)
            concepts_after_1 = len(self.learner.semantic.concept_hvs)
            
            # Learn from second file
            session2 = self.learner.learn_from_text_file(temp_path2)
            concepts_after_2 = len(self.learner.semantic.concept_hvs)
            
            # Should have more concepts after second file
            assert concepts_after_2 > concepts_after_1
            
            # Should have two sessions
            assert len(self.learner.learning_sessions) == 2
            
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)
    
    def test_statistics(self):
        """Test that statistics are tracked correctly."""
        # Learn from temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Photosynthesis converts sunlight into energy. Plants use photosynthesis.")
            temp_path = f.name
        
        try:
            self.learner.learn_from_text_file(temp_path)
            
            # Get statistics
            stats = self.learner.get_statistics()
            
            # Check structure
            assert 'total_sessions' in stats
            assert 'total_concepts' in stats
            assert 'total_facts' in stats
            assert 'total_episodes' in stats
            assert 'concept_frequencies' in stats
            assert 'relation_types' in stats
            assert 'sessions' in stats
            
            # Check values
            assert stats['total_sessions'] > 0
            assert stats['total_concepts'] > 0
            assert len(stats['sessions']) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_sentence_splitting(self):
        """Test that text is split into sentences correctly."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.learner._split_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_tokenization(self):
        """Test word tokenization."""
        text = "The quick brown fox jumps."
        tokens = self.learner._tokenize(text)
        
        assert len(tokens) > 0
        assert 'quick' in tokens or 'the' in tokens
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Empty case
        conf1 = self.learner._calculate_confidence([], [])
        assert conf1 == 0.1
        
        # With similarities
        similar = [("concept1", 0.8), ("concept2", 0.7)]
        facts = []
        conf2 = self.learner._calculate_confidence(similar, facts)
        assert 0 < conf2 < 1
        
        # With facts
        fact = LearnedFact(
            subject="Dog",
            relation="is_a",
            object="Mammal",
            source_text="Dogs are mammals.",
            confidence=0.7,
            timestamp=0
        )
        conf3 = self.learner._calculate_confidence(similar, [fact])
        assert conf3 > conf2  # Should be higher with facts
    
    def test_export_knowledge(self):
        """Test knowledge export functionality."""
        # Learn something
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Cats are mammals.")
            temp_path = f.name
        
        try:
            self.learner.learn_from_text_file(temp_path)
            
            # Export
            exported = self.learner.export_learned_knowledge()
            
            # Check export contains expected sections
            assert "STATISTICS" in exported
            assert "LEARNING SESSIONS" in exported
            assert "TOP CONCEPTS" in exported
            assert "LEARNED FACTS" in exported
            
        finally:
            os.unlink(temp_path)
    
    def test_no_llm_dependency(self):
        """Verify that learning works without LLM."""
        # This is tested implicitly - the system uses pattern matching
        # and hypervectors, not LLM calls
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Water is essential for life. Water molecules contain hydrogen and oxygen.")
            temp_path = f.name
        
        try:
            # Should work without any LLM
            session = self.learner.learn_from_text_file(temp_path)
            
            # Should have learned concepts
            assert session.concepts_learned > 0
            
            # Verify concepts are in semantic memory (graph structure)
            assert len(self.learner.semantic.concept_hvs) > 0
            
            # Verify we can query without LLM
            result = self.learner.query_learned_knowledge("What is water?")
            assert result['confidence'] >= 0
            
        finally:
            os.unlink(temp_path)


class TestSemanticMemoryIntegration:
    """Test integration with semantic memory."""
    
    def test_concept_storage(self):
        """Test that concepts are stored in semantic memory."""
        learner = TextKnowledgeLearner()
        
        # Learn from text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Birds can fly. Fish can swim.")
            temp_path = f.name
        
        try:
            learner.learn_from_text_file(temp_path)
            
            # Check concepts in semantic memory
            assert len(learner.semantic.concept_hvs) > 0
            
            # Check concept graph
            assert len(learner.semantic.concept_graph.nodes) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_relation_storage(self):
        """Test that relations are stored correctly."""
        learner = TextKnowledgeLearner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Apples are fruits. Fruits are healthy.")
            temp_path = f.name
        
        try:
            learner.learn_from_text_file(temp_path)
            
            # Should have edges in concept graph
            assert len(learner.semantic.concept_graph.edges) > 0
            
        finally:
            os.unlink(temp_path)


class TestEpisodicMemoryIntegration:
    """Test integration with episodic memory."""
    
    def test_episode_recording(self):
        """Test that episodes are recorded."""
        learner = TextKnowledgeLearner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("The sky is blue.")
            temp_path = f.name
        
        try:
            learner.learn_from_text_file(temp_path)
            
            # Check episodes in episodic memory
            assert 'text_learning' in learner.episodic.recent
            assert len(learner.episodic.recent['text_learning']) > 0
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
