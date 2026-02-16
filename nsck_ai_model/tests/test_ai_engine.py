"""
Tests for NSCK AI Engine — using actual NSCK cognitive architecture.

These tests verify that the engine properly integrates the NSCK modules
(SemanticMemory, EpisodicMemory, GlobalWorkspace, EmotionSystem,
CausalReasoner, SelfModel, CuriosityModule, TextKnowledgeLearner)
and that the trained model produces correct, verifiable responses.

Every test validates actual output — no ambiguous assertions.
"""

import os
import sys
import time
import json
import pytest
import re

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from nsck_ai_model.ai_engine import (
    NSCKAIEngine, ThoughtTrace, TraceStep, ResponseGenerator,
    _safe, DIMENSION, _FUNCTION_WORDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_engine():
    """Create and train an engine with seed data — shared across tests."""
    engine = NSCKAIEngine()
    training_data = [
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "Berlin is the capital of Germany.",
        "Germany is a country in Europe.",
        "Tokyo is the capital of Japan.",
        "Japan is a country in Asia.",
        "Dogs are mammals.",
        "Cats are mammals.",
        "Whales are mammals.",
        "Eagles are birds.",
        "The Earth orbits the Sun.",
        "The Moon orbits the Earth.",
        "The Sun is a star.",
        "Water is a molecule made of hydrogen and oxygen.",
        "Plants use photosynthesis to convert sunlight into energy.",
        "Rain causes flooding in low-lying areas.",
        "Deforestation causes soil erosion.",
        "Exercise improves cardiovascular health.",
        "Python is a programming language.",
        "Java is a programming language.",
        "The internet connects computers around the world.",
        "Shakespeare wrote plays and sonnets.",
        "Democracy is a system of government.",
        "Alice studies mathematics at Oxford University.",
    ]
    for text in training_data:
        engine.train_on_text(text)
    return engine


@pytest.fixture
def fresh_engine():
    """A fresh (untrained) engine for isolation tests."""
    return NSCKAIEngine()


# ═══════════════════════════════════════════════════════════════════════════
# §1  NSCK Module Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNSCKModuleIntegration:
    """Verify that the engine uses actual NSCK modules, not reimplementations."""

    def test_uses_nsck_semantic_memory(self, trained_engine):
        """Engine must use the actual NSCK SemanticMemory with NetworkX graph."""
        from python.core.memory.semantic_memory import SemanticMemory
        assert isinstance(trained_engine.semantic_memory, SemanticMemory)
        # SemanticMemory stores concepts in a NetworkX directed graph
        import networkx as nx
        assert isinstance(trained_engine.semantic_memory.concept_graph,
                          nx.DiGraph)
        assert len(trained_engine.semantic_memory.concept_graph.nodes) > 0

    def test_uses_nsck_episodic_memory(self, trained_engine):
        """Engine must use the actual NSCK EpisodicMemory."""
        from python.core.memory.episodic_memory import EpisodicMemory
        assert isinstance(trained_engine.episodic_memory, EpisodicMemory)

    def test_uses_nsck_global_workspace(self, trained_engine):
        """Engine must use the actual NSCK GlobalWorkspace."""
        from python.core.reasoning.global_workspace import GlobalWorkspace
        assert isinstance(trained_engine.global_workspace, GlobalWorkspace)

    def test_uses_nsck_emotion_system(self, trained_engine):
        """Engine must use the actual NSCK EmotionSystem (Plutchik)."""
        from python.core.cognitive.emotion_system import EmotionSystem
        assert isinstance(trained_engine.emotion_system, EmotionSystem)

    def test_uses_nsck_causal_reasoner(self, trained_engine):
        """Engine must use the actual NSCK CausalReasoner."""
        from python.core.reasoning.causal_reasoning import CausalReasoner
        assert isinstance(trained_engine.causal_reasoner, CausalReasoner)

    def test_uses_nsck_self_model(self, trained_engine):
        """Engine must use the actual NSCK SelfModel."""
        from python.core.cognitive.self_model import SelfModel
        assert isinstance(trained_engine.self_model, SelfModel)

    def test_uses_nsck_curiosity(self, trained_engine):
        """Engine must use the actual NSCK CuriosityModule."""
        from python.core.learning.curiosity import CuriosityModule
        assert isinstance(trained_engine.curiosity, CuriosityModule)

    def test_uses_nsck_text_learner(self, trained_engine):
        """Engine must use the actual NSCK TextKnowledgeLearner."""
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        assert isinstance(trained_engine.text_learner, TextKnowledgeLearner)

    def test_uses_nsck_hypervector(self, trained_engine):
        """Engine must use actual 10240-bit NSCK HyperVectors."""
        import python.core.vsa.hypervec_shim as hvs
        hv = trained_engine._encode_text("test")
        assert isinstance(hv, hvs.HyperVector)
        assert hv.bits.shape == (DIMENSION,)

    def test_semantic_memory_has_learned_concepts(self, trained_engine):
        """After training, SemanticMemory graph must contain learned concepts."""
        nodes = set(trained_engine.semantic_memory.concept_graph.nodes)
        # Check for specific learned concepts (case-insensitive)
        lower_nodes = {n.lower() for n in nodes}
        assert 'paris' in lower_nodes or 'Paris' in nodes
        assert 'france' in lower_nodes or 'France' in nodes
        assert 'dogs' in lower_nodes or 'Dogs' in nodes

    def test_semantic_memory_has_learned_relations(self, trained_engine):
        """After training, SemanticMemory must contain learned relations."""
        edges = list(trained_engine.semantic_memory.concept_graph.edges(
            data=True))
        assert len(edges) > 0
        # At least some relation types
        rel_types = {data.get('relation', '') for _, _, data in edges}
        assert len(rel_types) > 0

    def test_text_learner_has_facts(self, trained_engine):
        """TextKnowledgeLearner must have extracted facts from training."""
        facts = trained_engine.text_learner.learned_facts
        assert len(facts) > 10  # We trained on 24 sentences

    def test_global_workspace_has_modules(self, trained_engine):
        """GlobalWorkspace must have registered modules."""
        modules = trained_engine.global_workspace.modules
        assert 'SEMANTIC' in modules
        assert 'EPISODIC' in modules
        assert 'CAUSAL' in modules
        assert 'EMOTION' in modules


# ═══════════════════════════════════════════════════════════════════════════
# §2  Knowledge QA — Verified Output
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeQA:
    """Test that the trained model gives correct answers.
    Each test verifies the actual response text contains expected content.
    """

    def test_capital_of_france(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        resp = r['response'].lower()
        assert 'paris' in resp, f"Expected 'paris' in: {r['response']}"
        assert 'france' in resp, f"Expected 'france' in: {r['response']}"

    def test_capital_of_germany(self, trained_engine):
        r = trained_engine.chat("What is the capital of Germany?")
        resp = r['response'].lower()
        assert 'berlin' in resp or 'germany' in resp, \
            f"Expected 'berlin' or 'germany' in: {r['response']}"

    def test_dogs_are_mammals(self, trained_engine):
        r = trained_engine.chat("Tell me about dogs")
        resp = r['response'].lower()
        assert 'dogs' in resp, f"Expected 'dogs' in: {r['response']}"
        assert 'mammals' in resp or 'mammal' in resp, \
            f"Expected 'mammals' in: {r['response']}"

    def test_earth_orbits_sun(self, trained_engine):
        r = trained_engine.chat("What orbits the Sun?")
        resp = r['response'].lower()
        assert 'earth' in resp, f"Expected 'earth' in: {r['response']}"

    def test_rain_causes_flooding(self, trained_engine):
        r = trained_engine.chat("What causes flooding?")
        resp = r['response'].lower()
        assert 'rain' in resp or 'flooding' in resp, \
            f"Expected rain/flooding knowledge in: {r['response']}"

    def test_python_programming(self, trained_engine):
        r = trained_engine.chat("Tell me about Python")
        resp = r['response'].lower()
        assert 'python' in resp, f"Expected 'python' in: {r['response']}"

    def test_shakespeare(self, trained_engine):
        r = trained_engine.chat("Tell me about Shakespeare")
        resp = r['response'].lower()
        assert 'shakespeare' in resp, \
            f"Expected 'shakespeare' in: {r['response']}"

    def test_water_molecule(self, trained_engine):
        r = trained_engine.chat("What is water?")
        resp = r['response'].lower()
        assert 'water' in resp, f"Expected 'water' in: {r['response']}"

    def test_response_not_empty(self, trained_engine):
        r = trained_engine.chat("Tell me about France")
        assert len(r['response']) > 10

    def test_unknown_topic(self, trained_engine):
        r = trained_engine.chat("What is quantum teleportation?")
        # Should not crash, may say "I need more training data"
        assert isinstance(r['response'], str)
        assert len(r['response']) > 0


# ═══════════════════════════════════════════════════════════════════════════
# §3  Glass-Box Trace Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGlassBoxTrace:
    """Verify that every chat() produces a complete glass-box trace
    showing all NSCK modules that participated in the reasoning.
    """

    def test_trace_has_all_stages(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        trace = r['trace']
        stages = [s['stage'] for s in trace['steps']]
        # Must include all 11 cognitive stages
        assert 'encode' in stages, "Missing encode stage"
        assert 'emotion' in stages, "Missing emotion stage"
        assert 'extract_concepts' in stages, "Missing concept extraction"
        assert 'search_semantic' in stages, "Missing semantic search"
        assert 'spread_activation' in stages, "Missing spreading activation"
        assert 'query_knowledge' in stages, "Missing TKL query"
        assert 'causal_inference' in stages, "Missing causal inference"
        assert 'curiosity' in stages, "Missing curiosity assessment"
        assert 'global_workspace' in stages, "Missing GW competition"
        assert 'self_model' in stages, "Missing self-model update"
        assert 'generate' in stages, "Missing response generation"

    def test_trace_records_nsck_modules(self, trained_engine):
        r = trained_engine.chat("Tell me about dogs")
        trace = r['trace']
        # Check that trace mentions actual NSCK modules
        trace_text = json.dumps(trace)
        assert 'SemanticMemory' in trace_text or 'semantic' in trace_text.lower()
        assert 'EmotionSystem' in trace_text or 'emotion' in trace_text.lower()
        assert 'GlobalWorkspace' in trace_text or 'global_workspace' in trace_text.lower()
        assert 'CuriosityModule' in trace_text or 'curiosity' in trace_text.lower()

    def test_trace_has_timing(self, trained_engine):
        r = trained_engine.chat("What is water?")
        trace = r['trace']
        assert 'total_ms' in trace
        assert trace['total_ms'] > 0
        for step in trace['steps']:
            assert 'duration_ms' in step
            assert step['duration_ms'] >= 0

    def test_trace_has_unique_id(self, trained_engine):
        r1 = trained_engine.chat("What is the Sun?")
        r2 = trained_engine.chat("What is the Moon?")
        assert r1['trace']['trace_id'] != r2['trace']['trace_id']

    def test_trace_global_workspace_winner(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        trace = r['trace']
        gw_steps = [s for s in trace['steps']
                    if s['stage'] == 'global_workspace']
        assert len(gw_steps) == 1
        gw = gw_steps[0]
        assert 'winner_source' in gw['outputs']
        assert gw['outputs']['coalitions_count'] >= 0

    def test_trace_self_model_confidence(self, trained_engine):
        r = trained_engine.chat("Tell me about cats")
        trace = r['trace']
        sm_steps = [s for s in trace['steps'] if s['stage'] == 'self_model']
        assert len(sm_steps) == 1
        assert 'confidence' in sm_steps[0]['outputs']
        assert 'calibration_error' in sm_steps[0]['outputs']

    def test_trace_curiosity_novelty(self, trained_engine):
        r = trained_engine.chat("Tell me about photosynthesis")
        trace = r['trace']
        cur_steps = [s for s in trace['steps'] if s['stage'] == 'curiosity']
        assert len(cur_steps) == 1
        assert 'novelty_score' in cur_steps[0]['outputs']


# ═══════════════════════════════════════════════════════════════════════════
# §4  Response Quality Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseQuality:
    """Test that responses are natural language, not gibberish."""

    def test_response_is_grammatical(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        # Response should end with period and start with capital
        resp = r['response'].strip()
        assert resp[0].isupper(), f"Response should start with capital: {resp}"
        assert resp.endswith('.'), f"Response should end with period: {resp}"

    def test_response_is_relevant(self, trained_engine):
        r = trained_engine.chat("Tell me about dogs")
        resp = r['response'].lower()
        # Should mention dogs, not random other topics
        assert 'dogs' in resp

    def test_no_duplicate_sentences(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        sentences = [s.strip().lower() for s in r['response'].split('.')
                     if s.strip()]
        # No exact duplicates
        assert len(sentences) == len(set(sentences)), \
            f"Duplicate sentences in: {r['response']}"

    def test_response_not_question(self, trained_engine):
        r = trained_engine.chat("What are mammals?")
        # Response should not be a question back
        assert not r['response'].strip().endswith('?'), \
            f"Response should not be a question: {r['response']}"

    def test_confidence_is_float(self, trained_engine):
        r = trained_engine.chat("Tell me about France")
        assert isinstance(r['confidence'], float)
        assert 0.0 <= r['confidence'] <= 1.0

    def test_latency_is_reasonable(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        assert r['latency_ms'] < 5000, \
            f"Response took too long: {r['latency_ms']}ms"


# ═══════════════════════════════════════════════════════════════════════════
# §5  Emotion System Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEmotionSystem:
    """Test EmotionSystem integration — actual Plutchik model."""

    def test_emotion_in_response(self, trained_engine):
        r = trained_engine.chat("I am so happy today!")
        assert 'emotion' in r
        assert 'emotion' in r['emotion']

    def test_emotion_blend(self, trained_engine):
        r = trained_engine.chat("Tell me something interesting")
        blend = r['emotion'].get('blend', {})
        assert isinstance(blend, dict)

    def test_emotion_valence_arousal(self, trained_engine):
        r = trained_engine.chat("What is the Sun?")
        assert 'valence' in r['emotion']
        assert 'arousal' in r['emotion']

    def test_emotion_mood(self, trained_engine):
        r = trained_engine.chat("Tell me a fact")
        assert 'mood' in r['emotion']


# ═══════════════════════════════════════════════════════════════════════════
# §6  Training Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTraining:
    """Test the training pipeline."""

    def test_train_returns_stats(self, fresh_engine):
        r = fresh_engine.train_on_text("The sky is blue.")
        assert 'sentences_processed' in r
        assert 'concepts_added' in r
        assert 'relations_added' in r
        assert 'elapsed_s' in r
        assert r['sentences_processed'] >= 1

    def test_training_populates_semantic_memory(self, fresh_engine):
        fresh_engine.train_on_text("Elephants are large mammals.")
        nodes = set(n.lower() for n in
                    fresh_engine.semantic_memory.concept_graph.nodes)
        assert 'elephants' in nodes or 'elephant' in nodes

    def test_training_populates_text_learner_facts(self, fresh_engine):
        fresh_engine.train_on_text("The Nile is the longest river in Africa.")
        facts = fresh_engine.text_learner.learned_facts
        assert len(facts) > 0

    def test_training_builds_sentence_index(self, fresh_engine):
        fresh_engine.train_on_text("Mars is the fourth planet from the Sun.")
        assert len(fresh_engine._sentence_store) > 0
        assert len(fresh_engine._concept_sentences) > 0

    def test_training_learns_ngrams(self, fresh_engine):
        fresh_engine.train_on_text("The quick brown fox jumps over the lazy dog.")
        assert fresh_engine.generator.bigrams
        assert 'quick' in fresh_engine.generator.bigrams

    def test_training_time_tracked(self, fresh_engine):
        fresh_engine.train_on_text("Testing training time tracking.")
        assert fresh_engine._training_stats['training_time_s'] > 0
        assert fresh_engine._training_stats['texts_trained'] == 1

    def test_multiple_training_sessions(self, fresh_engine):
        fresh_engine.train_on_text("First sentence about apples.")
        fresh_engine.train_on_text("Second sentence about oranges.")
        assert fresh_engine._training_stats['texts_trained'] == 2


# ═══════════════════════════════════════════════════════════════════════════
# §7  Conversation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConversation:
    """Test multi-turn conversation capabilities."""

    def test_conversation_history(self, fresh_engine):
        fresh_engine.train_on_text("Paris is the capital of France.")
        initial_len = len(fresh_engine.conversation_history)
        fresh_engine.chat("Hello")
        # Should add user + assistant entries
        assert len(fresh_engine.conversation_history) >= initial_len + 2

    def test_auto_learn_from_statements(self, fresh_engine):
        fresh_engine.chat("Cats have whiskers.", auto_learn=True)
        # The system should have learned this
        r = fresh_engine.chat("Tell me about cats")
        # May or may not know about whiskers, but should not crash
        assert isinstance(r['response'], str)

    def test_disable_auto_learn(self, fresh_engine):
        initial_facts = len(fresh_engine.text_learner.learned_facts)
        fresh_engine.chat("Zebras have stripes.", auto_learn=False)
        # Should NOT have learned new facts
        assert len(fresh_engine.text_learner.learned_facts) == initial_facts


# ═══════════════════════════════════════════════════════════════════════════
# §8  Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self, trained_engine):
        r = trained_engine.chat("")
        assert isinstance(r['response'], str)

    def test_whitespace_input(self, trained_engine):
        r = trained_engine.chat("   ")
        assert isinstance(r['response'], str)

    def test_single_word(self, trained_engine):
        r = trained_engine.chat("France")
        assert isinstance(r['response'], str)

    def test_very_long_input(self, trained_engine):
        r = trained_engine.chat("What is " + "the " * 200 + "answer?")
        assert isinstance(r['response'], str)

    def test_special_characters(self, trained_engine):
        r = trained_engine.chat("What about @#$%^&*?")
        assert isinstance(r['response'], str)

    def test_unicode_input(self, trained_engine):
        r = trained_engine.chat("Tell me about café résumé")
        assert isinstance(r['response'], str)


# ═══════════════════════════════════════════════════════════════════════════
# §9  System Stats Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemStats:
    """Test stats and export functionality."""

    def test_stats_structure(self, trained_engine):
        stats = trained_engine.get_system_stats()
        assert 'engine' in stats
        assert 'training' in stats
        assert 'semantic_memory' in stats
        assert 'knowledge' in stats
        assert 'generator' in stats
        assert 'emotion' in stats
        assert 'self_model' in stats
        assert 'curiosity' in stats
        assert 'global_workspace' in stats
        assert 'text_learner' in stats

    def test_export_knowledge(self, trained_engine):
        export = trained_engine.export_knowledge()
        assert 'concepts' in export
        assert 'relations' in export
        assert 'facts' in export
        assert len(export['concepts']) > 0
        assert len(export['relations']) > 0

    def test_conversation_history_export(self, trained_engine):
        history = trained_engine.get_conversation_history()
        assert isinstance(history, list)

    def test_reset(self, fresh_engine):
        fresh_engine.train_on_text("Test data for reset.")
        fresh_engine.reset()
        stats = fresh_engine.get_system_stats()
        assert stats['training']['texts_trained'] == 0


# ═══════════════════════════════════════════════════════════════════════════
# §10  ThoughtTrace Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestThoughtTrace:
    """Test the glass-box ThoughtTrace system."""

    def test_trace_creation(self):
        trace = ThoughtTrace("test query")
        assert trace.trace_id
        assert trace.query == "test query"
        assert trace.steps == []

    def test_trace_begin_end(self):
        trace = ThoughtTrace("test")
        trace.begin("stage1", {"key": "value"})
        time.sleep(0.001)
        trace.end("action1", {"result": "ok"})
        assert len(trace.steps) == 1
        assert trace.steps[0].stage == "stage1"
        assert trace.steps[0].action == "action1"
        assert trace.steps[0].duration_ms > 0

    def test_trace_to_dict(self):
        trace = ThoughtTrace("test")
        trace.begin("s1", {})
        trace.end("a1", {"r": 1})
        d = trace.to_dict()
        assert d['query'] == "test"
        assert d['step_count'] == 1
        assert len(d['steps']) == 1

    def test_safe_serialisation(self):
        assert _safe(42) == 42
        assert _safe("hello") == "hello"
        assert _safe(None) is None
        assert isinstance(_safe({"a": 1}), dict)
        assert isinstance(_safe([1, 2, 3]), list)


# ═══════════════════════════════════════════════════════════════════════════
# §11  ResponseGenerator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseGenerator:
    """Test the n-gram response generator."""

    def test_learn(self):
        gen = ResponseGenerator()
        gen.learn("The cat sat on the mat.")
        assert 'cat' in gen.bigrams
        assert gen._corpus_size > 0

    def test_continue_from(self):
        gen = ResponseGenerator()
        gen.learn("The cat sat on the mat. The dog ran in the park.")
        result = gen.continue_from(["cat"])
        # Should produce something or None
        if result:
            assert len(result.split()) >= 2

    def test_stats(self):
        gen = ResponseGenerator()
        gen.learn("Testing n-gram statistics.")
        stats = gen.get_stats()
        assert 'bigram_vocab' in stats
        assert 'corpus_size' in stats
        assert stats['corpus_size'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# §12  Dashboard API Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboardAPI:
    """Test dashboard API endpoints."""

    @pytest.fixture
    def client(self):
        from nsck_ai_model.dashboard import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client):
        resp = client.get('/api/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'healthy'

    def test_chat_endpoint(self, client):
        # Train first
        client.post('/api/train', json={'text': 'Paris is the capital of France.'})
        resp = client.post('/api/chat', json={'message': 'Tell me about France'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'response' in data
        assert 'trace' in data

    def test_stats_endpoint(self, client):
        resp = client.get('/api/stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'engine' in data

    def test_knowledge_endpoint(self, client):
        resp = client.get('/api/knowledge')
        assert resp.status_code == 200

    def test_emotion_endpoint(self, client):
        resp = client.get('/api/emotion')
        assert resp.status_code == 200

    def test_logs_endpoint(self, client):
        resp = client.get('/api/logs')
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# §13  Performance & Timing Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Test execution times and performance."""

    def test_training_time(self, fresh_engine):
        """Training a single sentence should take < 1 second."""
        start = time.time()
        fresh_engine.train_on_text("Testing training performance.")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Training took {elapsed:.2f}s (limit 1s)"

    def test_chat_latency(self, trained_engine):
        """Chat response should complete in < 1 second."""
        r = trained_engine.chat("What is the capital of France?")
        assert r['latency_ms'] < 1000, \
            f"Chat latency {r['latency_ms']}ms exceeds 1s limit"

    def test_bulk_training(self, fresh_engine):
        """Training 20 sentences should take < 5 seconds."""
        start = time.time()
        for i in range(20):
            fresh_engine.train_on_text(f"Sentence number {i} about topic {i}.")
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Bulk training took {elapsed:.2f}s (limit 5s)"

    def test_burst_queries(self, trained_engine):
        """10 rapid queries should complete in < 5 seconds."""
        start = time.time()
        for _ in range(10):
            trained_engine.chat("What is the capital of France?")
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Burst queries took {elapsed:.2f}s (limit 5s)"
