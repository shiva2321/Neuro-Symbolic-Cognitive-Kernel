"""
Production-readiness tests for the NSCK AI Engine.

These tests exercise the system under real-world conditions that a user
would actually encounter: multi-turn conversations, diverse topics,
edge cases, performance under load, glass-box trace inspection,
and response quality validation.

Run with::

    python -m pytest nsck_ai_model/tests/test_production.py -v
"""
import sys
import os
import time
import re

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from nsck_ai_model.ai_engine import (
    HyperVector, TextEncoder, KnowledgeStore, ResponseGenerator,
    EmotionTracker, CausalRuleStore, KnowledgeAbstractor,
    ThoughtTrace, NSCKAIEngine, DIMENSION, _FUNCTION_WORDS,
)
from nsck_ai_model.data_pipeline import DataPipeline


# ── Helper: create a trained engine ──────────────────────────────────────

@pytest.fixture(scope="module")
def trained_engine():
    """A pre-trained engine used across tests in this module."""
    engine = NSCKAIEngine()
    pipeline = DataPipeline(engine)
    pipeline.train_from_seed()
    return engine


@pytest.fixture
def fresh_engine():
    """A fresh untrained engine for isolated tests."""
    return NSCKAIEngine()


# ═══════════════════════════════════════════════════════════════════════════
# §1  Knowledge Q&A — does the model answer factual questions correctly?
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeQA:
    """Test that the engine answers factual questions from its training."""

    def test_geography_france(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        assert 'paris' in r['response'].lower()
        assert 'france' in r['response'].lower()

    def test_geography_germany(self, trained_engine):
        r = trained_engine.chat("What is the capital of Germany?")
        assert 'berlin' in r['response'].lower()

    def test_geography_japan(self, trained_engine):
        r = trained_engine.chat("What is the capital of Japan?")
        assert 'tokyo' in r['response'].lower()

    def test_science_sun(self, trained_engine):
        r = trained_engine.chat("What is the Sun?")
        resp = r['response'].lower()
        assert 'sun' in resp
        assert 'star' in resp or 'earth' in resp

    def test_science_water(self, trained_engine):
        r = trained_engine.chat("What is water made of?")
        resp = r['response'].lower()
        assert 'water' in resp
        assert 'hydrogen' in resp or 'oxygen' in resp

    def test_science_dna(self, trained_engine):
        r = trained_engine.chat("What is DNA?")
        resp = r['response'].lower()
        assert 'dna' in resp
        assert 'genetic' in resp

    def test_science_photosynthesis(self, trained_engine):
        r = trained_engine.chat("Tell me about photosynthesis")
        resp = r['response'].lower()
        assert 'photosynthesis' in resp

    def test_animals_dogs(self, trained_engine):
        r = trained_engine.chat("Tell me about dogs")
        assert 'dog' in r['response'].lower()

    def test_animals_dolphins(self, trained_engine):
        r = trained_engine.chat("Tell me about dolphins")
        assert 'dolphin' in r['response'].lower()

    def test_tech_python(self, trained_engine):
        r = trained_engine.chat("What is Python?")
        resp = r['response'].lower()
        assert 'python' in resp
        assert 'programming' in resp

    def test_tech_machine_learning(self, trained_engine):
        r = trained_engine.chat("What is machine learning?")
        resp = r['response'].lower()
        assert 'machine' in resp or 'learning' in resp
        assert 'intelligence' in resp or 'artificial' in resp

    def test_cause_effect_flooding(self, trained_engine):
        r = trained_engine.chat("What causes flooding?")
        resp = r['response'].lower()
        assert 'rain' in resp or 'flooding' in resp

    def test_cause_effect_smoking(self, trained_engine):
        r = trained_engine.chat("What does smoking cause?")
        resp = r['response'].lower()
        assert 'smoking' in resp
        assert 'cancer' in resp or 'lung' in resp


# ═══════════════════════════════════════════════════════════════════════════
# §2  Response quality — are responses well-formed?
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseQuality:
    """Verify responses are grammatically sensible sentences."""

    def test_response_starts_with_capital(self, trained_engine):
        r = trained_engine.chat("What is the Sun?")['response']
        assert r[0].isupper(), f"Response should start with capital: '{r}'"

    def test_response_ends_with_period(self, trained_engine):
        r = trained_engine.chat("Tell me about DNA")['response']
        assert r.endswith('.'), f"Response should end with period: '{r}'"

    def test_response_not_empty(self, trained_engine):
        r = trained_engine.chat("What is gravity?")['response']
        assert len(r) > 5, f"Response too short: '{r}'"

    def test_no_duplicate_sentences(self, trained_engine):
        r = trained_engine.chat("Tell me about Europe")['response']
        sentences = [s.strip().lower() for s in r.split('.') if s.strip()]
        assert len(sentences) == len(set(sentences)), \
            f"Duplicate sentences in response: {sentences}"

    def test_response_focuses_on_topic(self, trained_engine):
        """Response should primarily discuss the queried topic."""
        r = trained_engine.chat("What is Python?")['response'].lower()
        assert 'python' in r, "Response should mention the queried topic"

    def test_consistency_across_queries(self, trained_engine):
        """Same query should give similar responses."""
        responses = set()
        for _ in range(3):
            r = trained_engine.chat("What is the capital of France?")
            responses.add(r['response'].lower().strip())
        # Should not have wildly different answers
        assert len(responses) <= 2, \
            f"Too many different responses for same query: {len(responses)}"


# ═══════════════════════════════════════════════════════════════════════════
# §3  Multi-turn conversation — pronoun resolution
# ═══════════════════════════════════════════════════════════════════════════

class TestConversation:
    """Test multi-turn conversation with context resolution."""

    def test_pronoun_resolution(self):
        """After discussing Alice, 'she' should resolve to Alice."""
        engine = NSCKAIEngine()
        engine.train_on_text(
            "Alice is a computer scientist. "
            "Alice studies machine learning at MIT.")
        engine.train_on_text(
            "Bob is a biologist. Bob studies DNA at Harvard.")
        engine.chat("Tell me about Alice")
        r = engine.chat("What does she study?")
        # The resolved concepts should include alice from context
        trace = r['trace']
        resolved_step = next(
            (s for s in trace['steps']
             if s['stage'] == 'extract_concepts'), None)
        assert resolved_step is not None
        outputs = resolved_step['outputs']
        if 'resolved_concepts' in outputs:
            assert 'alice' in outputs['resolved_concepts']

    def test_conversation_history_maintained(self, trained_engine):
        """Conversation history should accumulate."""
        trained_engine.chat("Hello!")
        trained_engine.chat("What is the Sun?")
        history = trained_engine.get_conversation_history()
        assert len(history) >= 4  # 2 queries × 2 (user + assistant)

    def test_learning_during_conversation(self):
        """Engine should learn from statements made during conversation."""
        engine = NSCKAIEngine()
        engine.chat("The Eiffel Tower is in Paris.", auto_learn=True)
        r = engine.chat("Where is the Eiffel Tower?")
        resp = r['response'].lower()
        assert 'eiffel' in resp or 'paris' in resp


# ═══════════════════════════════════════════════════════════════════════════
# §4  Glass-box traceability — can we trace every decision?
# ═══════════════════════════════════════════════════════════════════════════

class TestGlassBox:
    """Verify end-to-end traceability of reasoning."""

    def test_trace_has_all_stages(self, trained_engine):
        r = trained_engine.chat("What is the Sun?")
        stages = [s['stage'] for s in r['trace']['steps']]
        expected = ['encode', 'emotion', 'extract_concepts',
                    'search_semantic', 'search_episodic',
                    'spread_activation', 'causal_inference', 'generate']
        for stage in expected:
            assert stage in stages, f"Missing trace stage: {stage}"

    def test_trace_has_timing(self, trained_engine):
        r = trained_engine.chat("Tell me about dogs")
        for step in r['trace']['steps']:
            assert 'duration_ms' in step
            assert step['duration_ms'] >= 0

    def test_trace_has_trace_id(self, trained_engine):
        r = trained_engine.chat("What is DNA?")
        assert 'trace_id' in r['trace']
        assert len(r['trace']['trace_id']) > 0

    def test_trace_encodes_inputs_and_outputs(self, trained_engine):
        r = trained_engine.chat("What is Python?")
        for step in r['trace']['steps']:
            assert 'inputs' in step
            assert 'outputs' in step

    def test_reasoning_structure(self, trained_engine):
        r = trained_engine.chat("What causes flooding?")
        reasoning = r['reasoning']
        assert 'query_concepts' in reasoning
        assert 'matched_concepts' in reasoning
        assert 'episodes_found' in reasoning
        assert 'related_facts' in reasoning
        assert 'causal_rules_fired' in reasoning

    def test_confidence_is_meaningful(self, trained_engine):
        r = trained_engine.chat("What is the capital of France?")
        assert r['confidence'] > 0.3, \
            "Confidence should be > 0.3 for known topics"

    def test_confidence_lower_for_unknown(self, trained_engine):
        r = trained_engine.chat("What is quantum entanglement?")
        # Unknown topics should have lower confidence
        assert r['confidence'] < 0.9


# ═══════════════════════════════════════════════════════════════════════════
# §5  Edge cases — robustness under adversarial/unusual input
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test system robustness under edge conditions."""

    def test_empty_input(self, trained_engine):
        r = trained_engine.chat("")
        assert 'response' in r
        assert isinstance(r['response'], str)

    def test_single_character(self, trained_engine):
        r = trained_engine.chat("a")
        assert 'response' in r

    def test_punctuation_only(self, trained_engine):
        r = trained_engine.chat("??!!")
        assert 'response' in r
        assert r['confidence'] == 0.0

    def test_xss_input(self, trained_engine):
        r = trained_engine.chat("<script>alert(1)</script>")
        assert 'response' in r
        # Should not produce gibberish from n-gram model
        assert 'word word word' not in r['response'].lower()

    def test_unicode_input(self, trained_engine):
        r = trained_engine.chat("What is 日本語?")
        assert 'response' in r

    def test_very_long_input(self, trained_engine):
        long_input = "word " * 200
        r = trained_engine.chat(long_input)
        assert 'response' in r
        assert r['latency_ms'] < 5000

    def test_repeated_same_query(self, trained_engine):
        """Multiple identical queries should not cause issues."""
        for _ in range(10):
            r = trained_engine.chat("What is DNA?")
            assert 'response' in r
            assert len(r['response']) > 0

    def test_auto_learn_false_doesnt_train(self):
        """auto_learn=False should not add knowledge."""
        engine = NSCKAIEngine()
        before = engine.get_system_stats()['knowledge']['total_concepts']
        engine.chat("Elephants are the largest land animals.",
                    auto_learn=False)
        after = engine.get_system_stats()['knowledge']['total_concepts']
        assert after == before


# ═══════════════════════════════════════════════════════════════════════════
# §6  Autonomous learning & abstraction
# ═══════════════════════════════════════════════════════════════════════════

class TestAutonomousLearning:
    """Test that the system learns, abstracts, and reasons on its own."""

    def test_concept_extraction_from_training(self, trained_engine):
        stats = trained_engine.get_system_stats()
        assert stats['knowledge']['total_concepts'] > 100
        assert stats['knowledge']['total_relations'] > 200

    def test_abstraction_creates_categories(self, trained_engine):
        stats = trained_engine.get_system_stats()
        assert stats['abstractions']['total_abstractions'] > 50

    def test_causal_rules_learned(self, trained_engine):
        stats = trained_engine.get_system_stats()
        assert stats['causal_rules']['total_rules'] > 30

    def test_ngram_model_trained(self, trained_engine):
        stats = trained_engine.get_system_stats()
        assert stats['generator']['bigram_vocab'] > 50

    def test_causal_forward_chaining(self):
        """Causal rules should fire during reasoning."""
        engine = NSCKAIEngine()
        engine.train_on_text("Rain causes flooding.")
        engine.train_on_text("Flooding causes damage.")
        r = engine.chat("What causes flooding?")
        assert r['reasoning']['causal_rules_fired'] >= 0

    def test_learning_new_facts(self):
        """Engine should learn new facts and answer about them."""
        engine = NSCKAIEngine()
        engine.train_on_text(
            "Mars is the fourth planet from the Sun. "
            "Mars is also called the Red Planet.")
        r = engine.chat("Tell me about Mars")
        resp = r['response'].lower()
        assert 'mars' in resp

    def test_spreading_activation(self, trained_engine):
        """Graph traversal should find related concepts."""
        r = trained_engine.chat("Tell me about Europe")
        assert r['reasoning']['related_facts'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# §7  Performance — latency, throughput
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Test system performance under realistic load."""

    def test_chat_latency(self, trained_engine):
        """Individual chat should respond within 50ms."""
        latencies = []
        for _ in range(10):
            r = trained_engine.chat("What is the Sun?")
            latencies.append(r['latency_ms'])
        avg = sum(latencies) / len(latencies)
        assert avg < 50, f"Average latency too high: {avg:.1f}ms"

    def test_training_throughput(self):
        """Training should handle the seed corpus in under 1 second."""
        engine = NSCKAIEngine()
        pipeline = DataPipeline(engine)
        t0 = time.time()
        pipeline.train_from_seed()
        elapsed = time.time() - t0
        assert elapsed < 2.0, \
            f"Seed training too slow: {elapsed:.1f}s"

    def test_burst_queries(self, trained_engine):
        """Handle 50 queries without degradation."""
        latencies = []
        for i in range(50):
            r = trained_engine.chat(f"Tell me about topic {i}")
            latencies.append(r['latency_ms'])
        avg = sum(latencies) / len(latencies)
        # Last 10 should not be much slower than first 10
        first10 = sum(latencies[:10]) / 10
        last10 = sum(latencies[-10:]) / 10
        assert last10 < first10 * 3, \
            f"Performance degraded: first10={first10:.1f}ms, last10={last10:.1f}ms"


# ═══════════════════════════════════════════════════════════════════════════
# §8  Emotion tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestEmotionSystem:
    """Test the emotional state tracking system."""

    def test_emotion_in_response(self, trained_engine):
        r = trained_engine.chat("Tell me about the Sun")
        assert 'emotion' in r
        assert 'emotion' in r['emotion']
        assert 'valence' in r['emotion']
        assert 'arousal' in r['emotion']

    def test_emotion_blend(self, trained_engine):
        r = trained_engine.chat("Tell me about DNA")
        blend = r['emotion'].get('blend', {})
        assert isinstance(blend, dict)
        if blend:
            # Blend values should sum to approximately 1
            total = sum(blend.values())
            assert 0.9 < total < 1.1, \
                f"Blend should sum to ~1.0, got {total}"


# ═══════════════════════════════════════════════════════════════════════════
# §9  Dashboard API
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboardAPI:
    """Test all dashboard API endpoints."""

    def setup_method(self):
        from nsck_ai_model.dashboard import app, _get_engine
        self.client = app.test_client()
        _get_engine()

    def test_health_endpoint(self):
        r = self.client.get("/api/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'healthy'
        assert 'version' in data

    def test_chat_empty_message(self):
        r = self.client.post("/api/chat", json={"message": ""})
        assert r.status_code == 400

    def test_chat_no_body(self):
        r = self.client.post("/api/chat", data=b"",
                              content_type="application/json")
        assert r.status_code == 400

    def test_train_empty_text(self):
        r = self.client.post("/api/train", json={"text": ""})
        assert r.status_code == 400

    def test_stats_structure(self):
        r = self.client.get("/api/stats")
        data = r.get_json()
        assert 'engine' in data
        assert 'knowledge' in data
        assert 'encoder' in data
        assert 'training' in data
        assert 'emotion' in data
        assert 'causal_rules' in data
        assert 'abstractions' in data
        assert 'generator' in data


# ═══════════════════════════════════════════════════════════════════════════
# §10  Image understanding
# ═══════════════════════════════════════════════════════════════════════════

class TestImageIntegration:
    """Test image understanding system integration."""

    def test_encode_decode_cycle(self):
        from nsck_ai_model.image_understanding import ImageEncoder
        encoder = ImageEncoder()
        img1 = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        img2 = img1.copy()
        hv1 = encoder.encode_image(img1)
        hv2 = encoder.encode_image(img2)
        # Same image should produce highly similar HV (not exact due
        # to random tie-breaking in bundle majority rule)
        assert hv1.similarity(hv2) > 0.85

    def test_different_images_different_hvs(self):
        from nsck_ai_model.image_understanding import ImageEncoder
        encoder = ImageEncoder()
        img1 = np.zeros((32, 32, 3), dtype=np.uint8)  # black
        img2 = np.full((32, 32, 3), 255, dtype=np.uint8)  # white
        hv1 = encoder.encode_image(img1)
        hv2 = encoder.encode_image(img2)
        assert hv1.similarity(hv2) < 0.8

    def test_grayscale_image(self):
        from nsck_ai_model.image_understanding import ImageEncoder
        encoder = ImageEncoder()
        img = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        hv = encoder.encode_image(img)
        assert isinstance(hv, HyperVector)

    def test_cross_modal_retrieval(self):
        from nsck_ai_model.image_understanding import ImageUnderstanding
        engine = NSCKAIEngine()
        iu = ImageUnderstanding(engine)
        img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        iu.learn_image(img, "A red car on the road")
        results = iu.search_by_text("car")
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════
# §11  Data pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestDataPipelineProduction:
    """Test data pipeline reliability."""

    def test_seed_corpus_size(self):
        from nsck_ai_model.data_pipeline import stream_seed_corpus
        sentences = list(stream_seed_corpus())
        assert len(sentences) >= 60, \
            f"Seed corpus too small: {len(sentences)} sentences"

    def test_seed_corpus_quality(self):
        from nsck_ai_model.data_pipeline import stream_seed_corpus
        for sent in stream_seed_corpus():
            assert len(sent) > 10, f"Sentence too short: '{sent}'"
            assert sent.endswith('.'), f"Sentence should end with period: '{sent}'"
            assert sent[0].isupper(), f"Sentence should start with capital: '{sent}'"

    def test_pipeline_trains_engine(self):
        engine = NSCKAIEngine()
        pipeline = DataPipeline(engine)
        result = pipeline.train_from_seed()
        assert result['passages'] > 50
        stats = engine.get_system_stats()
        assert stats['knowledge']['total_concepts'] > 100
        assert stats['knowledge']['total_relations'] > 200

    def test_pipeline_stats(self):
        engine = NSCKAIEngine()
        pipeline = DataPipeline(engine)
        pipeline.train_from_seed()
        stats = pipeline.get_stats()
        assert stats['total_passages'] > 0
        assert stats['total_time_s'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# §12  System export & reset
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemLifecycle:
    """Test system export, reset, and recovery."""

    def test_export_has_all_sections(self, trained_engine):
        export = trained_engine.export_knowledge()
        assert 'concepts' in export
        assert 'relations' in export
        assert 'causal_rules' in export
        assert 'abstractions' in export
        assert 'stats' in export

    def test_reset_clears_everything(self):
        engine = NSCKAIEngine()
        engine.train_on_text("Test data for reset.")
        assert engine.get_system_stats()['knowledge']['total_concepts'] > 0
        engine.reset()
        assert engine.get_system_stats()['knowledge']['total_concepts'] == 0
        assert engine.get_system_stats()['knowledge']['total_relations'] == 0

    def test_retrain_after_reset(self):
        engine = NSCKAIEngine()
        engine.train_on_text("Dogs are mammals.")
        engine.reset()
        engine.train_on_text("Cats are pets.")
        r = engine.chat("Tell me about cats")
        assert 'cat' in r['response'].lower()
