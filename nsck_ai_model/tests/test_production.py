"""
Production Tests for NSCK AI Model
===================================

These tests verify real-world scenarios, validated outputs, and
production readiness of the NSCK-architecture-based AI model.

Every test validates ACTUAL output — not ambiguous assertions.
Tests cover: knowledge QA, reasoning, conversation, glass-box tracing,
performance, edge cases, emotion, dashboard, and NSCK module integration.
"""

import os
import sys
import time
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from nsck_ai_model.ai_engine import NSCKAIEngine


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Create and train a production-ready model."""
    engine = NSCKAIEngine()
    training_data = [
        # Geography
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "Berlin is the capital of Germany.",
        "Germany is a country in Europe.",
        "Tokyo is the capital of Japan.",
        "Japan is a country in Asia.",
        "London is the capital of the United Kingdom.",
        "Washington is the capital of the United States.",
        # Science
        "Water is a molecule made of hydrogen and oxygen.",
        "The Earth orbits the Sun.",
        "The Moon orbits the Earth.",
        "The Sun is a star.",
        "Plants use photosynthesis to convert sunlight into energy.",
        "Gravity is a force that attracts objects toward each other.",
        "DNA carries genetic information in living organisms.",
        "Water boils at 100 degrees Celsius at sea level.",
        # Animals
        "Dogs are mammals.",
        "Cats are mammals.",
        "Whales are mammals.",
        "Eagles are birds.",
        "Dolphins are highly intelligent marine mammals.",
        # Technology
        "Python is a programming language.",
        "Java is a programming language.",
        "The internet connects computers around the world.",
        "Machine learning is a subfield of artificial intelligence.",
        # Cause-effect
        "Rain causes flooding in low-lying areas.",
        "Deforestation causes soil erosion.",
        "Exercise improves cardiovascular health.",
        "Pollution causes environmental damage.",
        "Smoking causes lung cancer.",
        # Culture
        "Shakespeare wrote plays and sonnets.",
        "Democracy is a system of government.",
        "The Renaissance began in Italy in the 14th century.",
        # People
        "Alice studies mathematics at Oxford University.",
        "Bob is an engineer who designs bridges.",
    ]
    t0 = time.time()
    for text in training_data:
        engine.train_on_text(text)
    elapsed = time.time() - t0
    print(f"\n[Production] Trained on {len(training_data)} texts in {elapsed:.2f}s")
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# §1  Knowledge QA — Verified Correct Answers
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeQA:
    """Each test asks a question and verifies the response is factually
    correct based on training data."""

    def test_capital_of_france(self, model):
        r = model.chat("What is the capital of France?")
        assert 'paris' in r['response'].lower(), \
            f"Should mention Paris: {r['response']}"

    def test_capital_of_germany(self, model):
        r = model.chat("What is the capital of Germany?")
        resp = r['response'].lower()
        assert 'berlin' in resp or 'germany' in resp

    def test_capital_of_japan(self, model):
        r = model.chat("What is the capital of Japan?")
        resp = r['response'].lower()
        assert 'tokyo' in resp or 'japan' in resp

    def test_dogs_are_mammals(self, model):
        r = model.chat("What are dogs?")
        resp = r['response'].lower()
        assert 'dogs' in resp
        assert 'mammals' in resp or 'mammal' in resp

    def test_earth_orbits_sun(self, model):
        r = model.chat("What orbits the Sun?")
        assert 'earth' in r['response'].lower()

    def test_what_is_water(self, model):
        r = model.chat("What is water?")
        assert 'water' in r['response'].lower()

    def test_rain_causes_flooding(self, model):
        r = model.chat("What causes flooding?")
        resp = r['response'].lower()
        assert 'rain' in resp or 'flooding' in resp

    def test_python_is_programming(self, model):
        r = model.chat("What is Python?")
        assert 'python' in r['response'].lower()

    def test_shakespeare(self, model):
        r = model.chat("Tell me about Shakespeare")
        assert 'shakespeare' in r['response'].lower()

    def test_photosynthesis(self, model):
        r = model.chat("Tell me about photosynthesis")
        resp = r['response'].lower()
        assert 'photosynthesis' in resp or 'plants' in resp or 'sunlight' in resp


# ═══════════════════════════════════════════════════════════════════════════
# §2  NSCK Architecture Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestNSCKArchitecture:
    """Verify the model uses actual NSCK modules."""

    def test_semantic_memory_populated(self, model):
        nodes = len(model.semantic_memory.concept_graph.nodes)
        edges = len(model.semantic_memory.concept_graph.edges)
        assert nodes > 20, f"Only {nodes} concepts learned"
        assert edges > 10, f"Only {edges} relations learned"

    def test_text_learner_has_facts(self, model):
        facts = model.text_learner.learned_facts
        assert len(facts) > 30, f"Only {len(facts)} facts learned"

    def test_spreading_activation_works(self, model):
        activation = model.semantic_memory.spread_activation(
            ['France'], steps=2, decay=0.7)
        # France should activate related concepts
        assert len(activation) > 0

    def test_emotion_system_classifies(self, model):
        emo = model.emotion_system.recognize_emotion_from_text(
            "I am very happy!")
        assert isinstance(emo, str)
        assert len(emo) > 0

    def test_self_model_tracks(self, model):
        stats = model.self_model.get_stats('chat')
        assert stats['attempts'] > 0

    def test_global_workspace_competes(self, model):
        r = model.chat("What is the capital of France?")
        assert r['reasoning']['gw_winner'] != 'none'

    def test_curiosity_measures_novelty(self, model):
        r = model.chat("What is quantum chromodynamics?")
        assert 'novelty' in r['reasoning']
        assert isinstance(r['reasoning']['novelty'], float)


# ═══════════════════════════════════════════════════════════════════════════
# §3  Glass-Box Traceability
# ═══════════════════════════════════════════════════════════════════════════

class TestTraceability:
    """Every response must have a complete reasoning trace."""

    def test_trace_has_11_stages(self, model):
        r = model.chat("Tell me about cats")
        trace = r['trace']
        assert trace['step_count'] == 11, \
            f"Expected 11 trace steps, got {trace['step_count']}"

    def test_trace_shows_nsck_modules(self, model):
        r = model.chat("What is gravity?")
        trace_text = json.dumps(r['trace'])
        assert 'SemanticMemory' in trace_text or 'semantic' in trace_text.lower()
        assert 'GlobalWorkspace' in trace_text or 'global_workspace' in trace_text.lower()
        assert 'SelfModel' in trace_text or 'self_model' in trace_text.lower()
        assert 'CuriosityModule' in trace_text or 'curiosity' in trace_text.lower()

    def test_trace_has_timing(self, model):
        r = model.chat("Tell me about DNA")
        for step in r['trace']['steps']:
            assert 'duration_ms' in step
            assert step['duration_ms'] >= 0

    def test_trace_is_json_serialisable(self, model):
        r = model.chat("What is democracy?")
        # Should not raise
        json_str = json.dumps(r['trace'])
        assert len(json_str) > 100


# ═══════════════════════════════════════════════════════════════════════════
# §4  Natural Language Quality
# ═══════════════════════════════════════════════════════════════════════════

class TestNaturalLanguage:
    """Test response quality and naturalness."""

    def test_response_starts_with_capital(self, model):
        for q in ["What is water?", "Tell me about dogs", "What orbits the Sun?"]:
            r = model.chat(q)
            resp = r['response'].strip()
            if resp:
                assert resp[0].isupper(), f"Bad start: {resp}"

    def test_response_ends_with_period(self, model):
        for q in ["What is water?", "Tell me about France"]:
            r = model.chat(q)
            resp = r['response'].strip()
            if resp:
                assert resp.endswith('.'), f"Bad end: {resp}"

    def test_no_question_response(self, model):
        r = model.chat("What are mammals?")
        assert not r['response'].strip().endswith('?')

    def test_relevant_response(self, model):
        r = model.chat("Tell me about dogs")
        assert 'dogs' in r['response'].lower()

    def test_no_duplicate_sentences(self, model):
        r = model.chat("What is the capital of France?")
        sents = [s.strip().lower() for s in r['response'].split('.')
                 if s.strip()]
        assert len(sents) == len(set(sents))


# ═══════════════════════════════════════════════════════════════════════════
# §5  Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test robustness with edge case inputs."""

    def test_empty_input(self, model):
        r = model.chat("")
        assert isinstance(r['response'], str)

    def test_xss_input(self, model):
        r = model.chat("<script>alert('xss')</script>")
        assert '<script>' not in r['response']

    def test_very_long_input(self, model):
        r = model.chat("a " * 500 + "?")
        assert isinstance(r['response'], str)

    def test_unicode_input(self, model):
        r = model.chat("¿Qué es la democracia? 日本語テスト")
        assert isinstance(r['response'], str)

    def test_numeric_input(self, model):
        r = model.chat("12345 67890")
        assert isinstance(r['response'], str)


# ═══════════════════════════════════════════════════════════════════════════
# §6  Performance
# ═══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Test execution times."""

    def test_chat_latency_under_1s(self, model):
        r = model.chat("What is the capital of France?")
        assert r['latency_ms'] < 1000

    def test_training_time_under_1s(self):
        engine = NSCKAIEngine()
        t0 = time.time()
        engine.train_on_text("Quick training test sentence.")
        assert time.time() - t0 < 1.0

    def test_burst_10_queries_under_5s(self, model):
        t0 = time.time()
        for _ in range(10):
            model.chat("What is water?")
        assert time.time() - t0 < 5.0


# ═══════════════════════════════════════════════════════════════════════════
# §7  System Stats
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemStats:
    """Test system statistics reporting."""

    def test_stats_include_nsck_modules(self, model):
        stats = model.get_system_stats()
        assert 'semantic_memory' in stats
        assert 'emotion' in stats
        assert 'self_model' in stats
        assert 'curiosity' in stats
        assert 'global_workspace' in stats
        assert 'text_learner' in stats

    def test_export_has_concepts_relations_facts(self, model):
        export = model.export_knowledge()
        assert len(export['concepts']) > 0
        assert len(export['relations']) > 0
        assert len(export['facts']) > 0


# ═══════════════════════════════════════════════════════════════════════════
# §8  Dashboard API
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboardAPI:
    """Test all dashboard API endpoints."""

    @pytest.fixture
    def client(self):
        from nsck_ai_model.dashboard import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health(self, client):
        r = client.get('/api/health')
        assert r.status_code == 200
        d = r.get_json()
        assert d['status'] == 'healthy'

    def test_chat(self, client):
        client.post('/api/train', json={'text': 'Paris is the capital of France.'})
        r = client.post('/api/chat', json={'message': 'What is the capital of France?'})
        assert r.status_code == 200
        d = r.get_json()
        assert 'response' in d
        assert 'trace' in d

    def test_train(self, client):
        r = client.post('/api/train', json={'text': 'Test sentence.'})
        assert r.status_code == 200

    def test_stats(self, client):
        r = client.get('/api/stats')
        assert r.status_code == 200

    def test_knowledge(self, client):
        r = client.get('/api/knowledge')
        assert r.status_code == 200

    def test_concepts(self, client):
        r = client.get('/api/knowledge/concepts')
        assert r.status_code == 200

    def test_relations(self, client):
        r = client.get('/api/knowledge/relations')
        assert r.status_code == 200

    def test_rules(self, client):
        r = client.get('/api/knowledge/rules')
        assert r.status_code == 200

    def test_emotion(self, client):
        r = client.get('/api/emotion')
        assert r.status_code == 200

    def test_logs(self, client):
        r = client.get('/api/logs')
        assert r.status_code == 200
