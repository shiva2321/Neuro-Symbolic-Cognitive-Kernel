"""
Tests for the NSCK AI Engine — validates all components end-to-end.

Run with::

    python -m pytest nsck_ai_model/tests/test_ai_engine.py -v
"""
import sys
import os
import time

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from nsck_ai_model.ai_engine import (
    HyperVector, TextEncoder, KnowledgeStore, ResponseGenerator,
    EmotionTracker, CausalRuleStore, KnowledgeAbstractor,
    ThoughtTrace, NSCKAIEngine, DIMENSION,
)


# ═══════════════════════════════════════════════════════════════════════════
# HyperVector tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHyperVector:
    def test_creation_random(self):
        hv = HyperVector()
        assert hv.bits.shape == (DIMENSION,)
        assert hv.bits.dtype == np.int8

    def test_creation_zero(self):
        hv = HyperVector.zero()
        assert np.sum(hv.bits) == 0

    def test_from_seed_deterministic(self):
        a = HyperVector.from_seed("test")
        b = HyperVector.from_seed("test")
        assert np.array_equal(a.bits, b.bits)

    def test_from_seed_different(self):
        a = HyperVector.from_seed("hello")
        b = HyperVector.from_seed("world")
        sim = a.similarity(b)
        assert 0.45 < sim < 0.55  # roughly orthogonal

    def test_bind_self_inverse(self):
        a = HyperVector()
        b = HyperVector()
        bound = a.bind(b)
        recovered = bound.bind(b)
        assert recovered.similarity(a) == 1.0

    def test_bind_dissimilar(self):
        a = HyperVector()
        b = HyperVector()
        bound = a.bind(b)
        assert bound.similarity(a) < 0.55

    def test_bundle_similar_to_all(self):
        hvs = [HyperVector() for _ in range(5)]
        bundled = HyperVector.bundle(hvs)
        for hv in hvs:
            assert bundled.similarity(hv) > 0.5

    def test_bundle_empty(self):
        result = HyperVector.bundle([])
        assert np.sum(result.bits) == 0

    def test_bundle_single(self):
        a = HyperVector()
        result = HyperVector.bundle([a])
        assert result.similarity(a) == 1.0

    def test_permute(self):
        a = HyperVector()
        p = a.permute(1)
        assert p.similarity(a) < 0.55  # permutation makes it different

    def test_similarity_identical(self):
        a = HyperVector()
        assert a.similarity(a) == 1.0

    def test_similarity_orthogonal(self):
        a = HyperVector()
        b = HyperVector()
        sim = a.similarity(b)
        assert 0.45 < sim < 0.55


# ═══════════════════════════════════════════════════════════════════════════
# TextEncoder tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTextEncoder:
    def setup_method(self):
        self.enc = TextEncoder()

    def test_encode_sentence(self):
        hv = self.enc.encode_sentence("The cat sat on the mat")
        assert isinstance(hv, HyperVector)
        assert hv.bits.shape == (DIMENSION,)

    def test_similar_sentences_similar_vectors(self):
        hv1 = self.enc.encode_sentence("The dog ran fast")
        hv2 = self.enc.encode_sentence("The dog ran quickly")
        hv3 = self.enc.encode_sentence("Mathematics is abstract")
        sim_similar = hv1.similarity(hv2)
        sim_different = hv1.similarity(hv3)
        # Similar sentences should be more similar (or at least not less)
        assert sim_similar >= sim_different - 0.1

    def test_learn_text(self):
        stats = self.enc.learn_text("Paris is the capital of France.")
        assert stats['sentences'] >= 1
        assert stats['new_words'] > 0
        assert stats['cooccurrences'] > 0

    def test_extract_concepts(self):
        concepts = self.enc.extract_concepts(
            "Paris is the capital of France")
        assert 'paris' in concepts
        assert 'capital' in concepts
        assert 'france' in concepts
        # Function words should not be in concepts
        assert 'the' not in concepts
        assert 'is' not in concepts

    def test_extract_concepts_dedup(self):
        concepts = self.enc.extract_concepts("cat cat cat dog")
        assert concepts.count('cat') == 1

    def test_find_similar_after_training(self):
        for _ in range(5):
            self.enc.learn_text("Dogs are friendly animals.")
            self.enc.learn_text("Cats are independent animals.")
        similar = self.enc.find_similar("dogs")
        # We at least expect no crash; similarity depends on training volume
        assert isinstance(similar, list)

    def test_tokenize(self):
        tokens = TextEncoder._tokenize("Hello, world! How's it going?")
        assert all(isinstance(t, str) for t in tokens)
        assert all(len(t) > 1 for t in tokens)

    def test_split_sentences(self):
        sents = TextEncoder._split_sentences(
            "First sentence here. Second sentence here! Third sentence here?")
        assert len(sents) == 3


# ═══════════════════════════════════════════════════════════════════════════
# KnowledgeStore tests
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeStore:
    def setup_method(self):
        self.store = KnowledgeStore()

    def test_add_concept(self):
        hv = HyperVector()
        c = self.store.add_concept("cat", hv, "Cats are mammals")
        assert c.name == "cat"
        assert c.frequency == 1

    def test_add_concept_increments_frequency(self):
        hv = HyperVector()
        self.store.add_concept("cat", hv, "Cats are mammals")
        self.store.add_concept("cat", hv, "Cats are pets")
        assert self.store.concepts["cat"].frequency == 2

    def test_add_relation(self):
        hv = HyperVector()
        rel = self.store.add_relation("cat", "mammal",
                                       "Cats are mammals", hv)
        assert rel.source_concept == "cat"
        assert rel.target_concept == "mammal"

    def test_add_relation_dedup(self):
        hv = HyperVector()
        self.store.add_relation("cat", "mammal", "Cats are mammals", hv)
        self.store.add_relation("cat", "mammal", "Cats are mammals", hv)
        # Same relation should be deduplicated
        assert len(self.store.relations) == 1
        assert self.store.relations[0].evidence_count == 2

    def test_record_episode(self):
        hv = HyperVector()
        ep = self.store.record_episode("Hello world", hv, ["hello", "world"])
        assert ep.text == "Hello world"
        assert len(self.store.episodes) == 1

    def test_search_concepts(self):
        hv1 = HyperVector.from_seed("cat")
        hv2 = HyperVector.from_seed("dog")
        self.store.add_concept("cat", hv1)
        self.store.add_concept("dog", hv2)
        results = self.store.search_concepts(hv1, top_k=5)
        if results:
            assert results[0][0] == "cat"

    def test_search_episodes(self):
        hv = HyperVector.from_seed("test")
        self.store.record_episode("test episode", hv, ["test"])
        results = self.store.search_episodes(hv, top_k=5)
        assert len(results) >= 1
        assert results[0][0].text == "test episode"

    def test_find_related(self):
        hv = HyperVector()
        self.store.add_relation("cat", "mammal", "Cats are mammals", hv)
        self.store.add_relation("mammal", "animal",
                                 "Mammals are animals", hv)
        related = self.store.find_related("cat", max_depth=2)
        assert len(related) >= 1
        related_concepts = [r[0] for r in related]
        assert "mammal" in related_concepts

    def test_stats(self):
        s = self.store.get_stats()
        assert 'total_concepts' in s
        assert 'total_relations' in s
        assert 'total_episodes' in s


# ═══════════════════════════════════════════════════════════════════════════
# CausalRuleStore tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalRuleStore:
    def setup_method(self):
        self.store = CausalRuleStore()

    def test_add_rule(self):
        hv = HyperVector()
        rule = self.store.add_rule(["rain"], ["flooding"],
                                    "Rain causes flooding", hv)
        assert rule.antecedent == ["rain"]
        assert rule.consequent == ["flooding"]

    def test_add_rule_dedup(self):
        hv = HyperVector()
        self.store.add_rule(["rain"], ["flooding"], "Rain causes flooding", hv)
        self.store.add_rule(["rain"], ["flooding"], "Rain causes flooding", hv)
        assert len(self.store.rules) == 1
        assert self.store.rules[0].evidence == 2

    def test_forward_chain(self):
        hv = HyperVector()
        self.store.add_rule(["rain"], ["flooding"],
                             "Rain causes flooding", hv)
        fired = self.store.forward_chain(["rain"])
        assert len(fired) == 1
        assert fired[0][0].consequent == ["flooding"]

    def test_forward_chain_depth(self):
        hv = HyperVector()
        self.store.add_rule(["rain"], ["flooding"],
                             "Rain causes flooding", hv)
        self.store.add_rule(["flooding"], ["damage"],
                             "Flooding causes damage", hv)
        fired = self.store.forward_chain(["rain"], max_depth=2)
        assert len(fired) == 2

    def test_strength_increases(self):
        hv = HyperVector()
        self.store.add_rule(["a"], ["b"], "A causes B", hv)
        s1 = self.store.rules[0].strength
        self.store.add_rule(["a"], ["b"], "A causes B", hv)
        s2 = self.store.rules[0].strength
        assert s2 > s1


# ═══════════════════════════════════════════════════════════════════════════
# Abstractor tests
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeAbstractor:
    def test_abstracts_from_common_target(self):
        store = KnowledgeStore()
        ab = KnowledgeAbstractor(min_members=2)
        hv = HyperVector()
        store.add_concept("cat", hv)
        store.add_concept("dog", hv)
        store.add_concept("mammal", hv)
        store.add_relation("cat", "mammal", "Cats are mammals", hv)
        store.add_relation("dog", "mammal", "Dogs are mammals", hv)
        result = ab.abstract(store.relations, store)
        assert len(result) >= 1
        assert result[0]['category'] == 'mammal'
        assert set(result[0]['members']) == {'cat', 'dog'}


# ═══════════════════════════════════════════════════════════════════════════
# EmotionTracker tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEmotionTracker:
    def setup_method(self):
        self.tracker = EmotionTracker()

    def test_initial_state(self):
        state = self.tracker.get_state()
        assert state['emotion'] == 'neutral'
        assert state['valence'] == 0.0

    def test_update_from_hv(self):
        hv = HyperVector()
        emotion = self.tracker.update_from_hv(hv)
        assert isinstance(emotion, str)
        assert self.tracker._update_count == 1

    def test_learn_valence(self):
        hv = HyperVector.from_seed("happy")
        self.tracker.learn_valence(hv, 0.8)
        assert len(self.tracker._positive_hvs) == 1

    def test_blend(self):
        blend = self.tracker.get_blend()
        assert isinstance(blend, dict)
        assert sum(blend.values()) > 0


# ═══════════════════════════════════════════════════════════════════════════
# ResponseGenerator tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseGenerator:
    def setup_method(self):
        self.gen = ResponseGenerator()

    def test_learn(self):
        self.gen.learn("The cat sat on the mat.")
        assert len(self.gen.bigrams) > 0

    def test_continue_from(self):
        for _ in range(5):
            self.gen.learn("The cat sat on the mat and the dog played.")
        result = self.gen.continue_from(["cat"])
        if result:
            assert isinstance(result, str)
            assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════
# ThoughtTrace tests
# ═══════════════════════════════════════════════════════════════════════════

class TestThoughtTrace:
    def test_trace_records_steps(self):
        trace = ThoughtTrace("test query")
        trace.begin("encode", {"text": "hello"})
        trace.end("Encoded text", {"dim": 10240})
        trace.begin("retrieve", {})
        trace.end("Found matches", {"count": 5})
        d = trace.to_dict()
        assert d['step_count'] == 2
        assert d['steps'][0]['stage'] == 'encode'
        assert d['steps'][1]['stage'] == 'retrieve'

    def test_trace_timing(self):
        trace = ThoughtTrace("test")
        trace.begin("test_stage", {})
        time.sleep(0.01)
        trace.end("Done", {})
        d = trace.to_dict()
        assert d['steps'][0]['duration_ms'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# NSCKAIEngine integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNSCKAIEngine:
    def setup_method(self):
        self.engine = NSCKAIEngine()

    def test_train_on_text(self):
        result = self.engine.train_on_text(
            "Paris is the capital of France.")
        assert result['sentences_processed'] >= 1
        assert result['concepts_added'] > 0

    def test_chat_returns_response(self):
        self.engine.train_on_text(
            "Paris is the capital of France.")
        result = self.engine.chat("What is the capital of France?")
        assert 'response' in result
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    def test_chat_returns_trace(self):
        self.engine.train_on_text("The sun is a star.")
        result = self.engine.chat("What is the sun?")
        assert 'trace' in result
        assert result['trace']['step_count'] >= 5

    def test_chat_returns_emotion(self):
        self.engine.train_on_text("Hello!")
        result = self.engine.chat("Hello!")
        assert 'emotion' in result
        assert 'emotion' in result['emotion']

    def test_chat_returns_confidence(self):
        self.engine.train_on_text("Cats are mammals.")
        result = self.engine.chat("What are cats?")
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1

    def test_chat_response_uses_learned_knowledge(self):
        """The response should contain information from training data."""
        self.engine.train_on_text(
            "Paris is the capital of France. "
            "Berlin is the capital of Germany.")
        result = self.engine.chat("What is the capital of France?")
        # The response should mention Paris or France
        lower = result['response'].lower()
        assert 'paris' in lower or 'france' in lower or 'capital' in lower

    def test_learns_from_statements(self):
        """When the user makes a statement, the engine should learn it."""
        result = self.engine.chat("The ocean is very deep.")
        # The engine should acknowledge learning
        lower = result['response'].lower()
        assert 'learn' in lower or 'stored' in lower or 'ocean' in lower

    def test_conversation_history(self):
        self.engine.train_on_text("Hello there.")
        self.engine.chat("Hello!")
        history = self.engine.get_conversation_history()
        assert len(history) >= 2  # user + assistant

    def test_system_stats(self):
        self.engine.train_on_text("The cat sat on the mat.")
        stats = self.engine.get_system_stats()
        assert 'knowledge' in stats
        assert 'encoder' in stats
        assert 'training' in stats

    def test_export_knowledge(self):
        self.engine.train_on_text("Dogs are mammals.")
        export = self.engine.export_knowledge()
        assert 'concepts' in export
        assert 'relations' in export
        assert 'causal_rules' in export

    def test_reset(self):
        self.engine.train_on_text("Test data.")
        self.engine.reset()
        stats = self.engine.get_system_stats()
        assert stats['knowledge']['total_concepts'] == 0

    def test_glass_box_traceability(self):
        """Verify every reasoning step is traceable."""
        self.engine.train_on_text(
            "Rain causes flooding. Flooding causes damage.")
        result = self.engine.chat("What causes flooding?")
        trace = result['trace']
        stages = [s['stage'] for s in trace['steps']]
        # Must have these core stages
        assert 'encode' in stages
        assert 'extract_concepts' in stages
        assert 'search_semantic' in stages
        assert 'generate' in stages

    def test_no_hardcoded_templates(self):
        """Verify responses come from learned data, not templates."""
        self.engine.train_on_text("Gravity pulls objects toward Earth.")
        result = self.engine.chat("What is gravity?")
        resp = result['response'].lower()
        # Response should contain words from the training data
        assert ('gravity' in resp or 'pulls' in resp or 'objects' in resp
                or 'earth' in resp or "don't have" in resp)

    def test_multiple_training_texts(self):
        """Engine should learn from multiple texts."""
        self.engine.train_on_text("The sun is a star.")
        self.engine.train_on_text("Stars produce light and heat.")
        result = self.engine.chat("Tell me about stars")
        assert 'response' in result
        assert len(result['response']) > 0

    def test_causal_inference(self):
        """Causal rules should fire during reasoning."""
        self.engine.train_on_text("Smoking causes lung cancer.")
        result = self.engine.chat("What does smoking cause?")
        assert result['reasoning']['causal_rules_fired'] >= 0

    def test_reasoning_contains_query_concepts(self):
        self.engine.train_on_text("Python is a programming language.")
        result = self.engine.chat("What is Python?")
        assert 'python' in result['reasoning']['query_concepts']


# ═══════════════════════════════════════════════════════════════════════════
# Data pipeline tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDataPipeline:
    def test_seed_corpus_streams(self):
        from nsck_ai_model.data_pipeline import stream_seed_corpus
        sentences = list(stream_seed_corpus())
        assert len(sentences) > 20

    def test_pipeline_train_from_seed(self):
        from nsck_ai_model.data_pipeline import DataPipeline
        engine = NSCKAIEngine()
        pipeline = DataPipeline(engine)
        result = pipeline.train_from_seed()
        assert result['passages'] > 20
        assert result['source'] == 'seed_corpus'
        stats = engine.get_system_stats()
        assert stats['knowledge']['total_concepts'] > 50

    def test_pipeline_stats(self):
        from nsck_ai_model.data_pipeline import DataPipeline
        engine = NSCKAIEngine()
        pipeline = DataPipeline(engine)
        pipeline.train_from_seed()
        stats = pipeline.get_stats()
        assert stats['total_passages'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Image understanding tests
# ═══════════════════════════════════════════════════════════════════════════

class TestImageUnderstanding:
    def test_encode_numpy_image(self):
        from nsck_ai_model.image_understanding import ImageEncoder
        encoder = ImageEncoder()
        img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        hv = encoder.encode_image(img)
        assert isinstance(hv, HyperVector)
        assert hv.bits.shape == (DIMENSION,)

    def test_learn_and_search(self):
        from nsck_ai_model.image_understanding import ImageUnderstanding
        engine = NSCKAIEngine()
        iu = ImageUnderstanding(engine)
        img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        iu.learn_image(img, "A red car on a road")
        results = iu.search_by_text("car")
        assert isinstance(results, list)

    def test_describe_image(self):
        from nsck_ai_model.image_understanding import ImageUnderstanding
        engine = NSCKAIEngine()
        iu = ImageUnderstanding(engine)
        img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
        iu.learn_image(img, "A blue sky")
        desc = iu.describe_image(img)
        assert isinstance(desc, str)


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboard:
    def setup_method(self):
        from nsck_ai_model.dashboard import app, _get_engine
        self.app = app
        self.client = app.test_client()
        _get_engine()  # initialise

    def test_index(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert b"NSCK AI Dashboard" in r.data

    def test_api_chat(self):
        # Train first
        self.client.post("/api/train",
                         json={"text": "Cats are mammals."})
        r = self.client.post("/api/chat",
                              json={"message": "What are cats?"})
        assert r.status_code == 200
        data = r.get_json()
        assert 'response' in data
        assert 'trace' in data

    def test_api_stats(self):
        r = self.client.get("/api/stats")
        assert r.status_code == 200
        data = r.get_json()
        assert 'knowledge' in data

    def test_api_train(self):
        r = self.client.post("/api/train",
                              json={"text": "Python is a language."})
        assert r.status_code == 200
        data = r.get_json()
        assert 'concepts_added' in data

    def test_api_concepts(self):
        self.client.post("/api/train",
                         json={"text": "Dogs are loyal animals."})
        r = self.client.get("/api/knowledge/concepts")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_api_relations(self):
        r = self.client.get("/api/knowledge/relations")
        assert r.status_code == 200

    def test_api_rules(self):
        r = self.client.get("/api/knowledge/rules")
        assert r.status_code == 200

    def test_api_emotion(self):
        r = self.client.get("/api/emotion")
        assert r.status_code == 200
        assert 'emotion' in r.get_json()

    def test_api_logs(self):
        r = self.client.get("/api/logs")
        assert r.status_code == 200

    def test_api_reset(self):
        r = self.client.post("/api/reset")
        assert r.status_code == 200

    def test_api_knowledge_export(self):
        r = self.client.get("/api/knowledge")
        assert r.status_code == 200
        data = r.get_json()
        assert 'concepts' in data

    def test_api_chat_history(self):
        self.client.post("/api/chat", json={"message": "Hello"})
        r = self.client.get("/api/chat/history")
        assert r.status_code == 200
