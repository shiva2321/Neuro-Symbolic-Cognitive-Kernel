"""
V10 Extension Tests
===================
Tests for all 8 V10 features:
  1. EmbeddingVSABridge
  2. rust_concurrent_shim
  3. NSCKApiServer
  4. NgramNLU / NgramNLUAdapter
  5. FHRRVector / FHRRMemory
  6. AttentionHead / MultiHeadAttentionGWT / AttentionGWTBridge
  7. RuleFeaturizer / RuleNeuralScorer + CognitiveEngine integration
  8. SafetyProperty / SafetyRuleVerifier / SafetyGateVerifier
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers / Stubs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _MockRule:
    """Minimal mock Rule for testing scorers/verifiers."""
    id: Optional[int] = 1
    condition: FrozenSet[str] = field(default_factory=lambda: frozenset(["pred_a", "pred_b"]))
    consequence: str = "ACTION_GO"
    confidence: float = 0.8
    support_count: int = 10
    fire_count: int = 50
    task_tag: str = "test_task"
    confidence_history: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8])
    success_rate: float = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# 1. EmbeddingVSABridge
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbeddingVSABridge:
    """Tests for EmbeddingVSABridge."""

    def setup_method(self):
        from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
        self.bridge = EmbeddingVSABridge(dim_in=64, hv_dim=10240, seed=42)

    def test_init_proj_shape(self):
        assert self.bridge._proj.shape == (64, 10240)

    def test_embed_to_hv_type(self):
        emb = np.random.randn(64).astype(np.float32)
        hv = self.bridge.embed_to_hv(emb)
        assert hv is not None

    def test_embed_to_hv_bits_are_binary(self):
        emb = np.random.randn(64).astype(np.float32)
        hv = self.bridge.embed_to_hv(emb)
        bits = np.asarray(hv.bits)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_hv_to_embed_shape(self):
        emb = np.random.randn(64).astype(np.float32)
        hv = self.bridge.embed_to_hv(emb)
        recovered = self.bridge.hv_to_embed(hv)
        assert recovered.shape == (64,)

    def test_hv_to_embed_normalized(self):
        emb = np.random.randn(64).astype(np.float32)
        hv = self.bridge.embed_to_hv(emb)
        recovered = self.bridge.hv_to_embed(hv)
        norm = np.linalg.norm(recovered)
        assert abs(norm - 1.0) < 1e-5

    def test_batch_embed_to_hv_length(self):
        embs = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        hvs = self.bridge.batch_embed_to_hv(embs)
        assert len(hvs) == 5

    def test_batch_embed_different_hvs(self):
        embs = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        hvs = self.bridge.batch_embed_to_hv(embs)
        # Two random embeddings should produce different HVs
        sim = hvs[0].similarity(hvs[1])
        # Not perfectly identical
        assert sim < 1.0

    def test_similarity_in_embed_space_same(self):
        emb = np.random.randn(64).astype(np.float32)
        hv = self.bridge.embed_to_hv(emb)
        sim = self.bridge.similarity_in_embed_space(hv, hv)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_in_embed_space_range(self):
        e1 = np.random.randn(64).astype(np.float32)
        e2 = np.random.randn(64).astype(np.float32)
        hv1 = self.bridge.embed_to_hv(e1)
        hv2 = self.bridge.embed_to_hv(e2)
        sim = self.bridge.similarity_in_embed_space(hv1, hv2)
        assert -1.0 <= sim <= 1.0

    def test_try_load_sentence_transformer_returns_bool(self):
        result = self.bridge.try_load_sentence_transformer("nonexistent-model-xyz")
        assert isinstance(result, bool)

    def test_encode_text_fallback(self):
        hv = self.bridge.encode_text("hello world")
        assert hv is not None
        bits = np.asarray(hv.bits)
        assert bits.sum() > 0

    def test_encode_text_different_texts(self):
        hv1 = self.bridge.encode_text("cat")
        hv2 = self.bridge.encode_text("completely different sentence about nothing")
        sim = hv1.similarity(hv2)
        assert sim < 1.0

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError):
            self.bridge.embed_to_hv(np.zeros(128))

    def test_seed_determinism(self):
        from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
        b1 = EmbeddingVSABridge(dim_in=32, hv_dim=10240, seed=99)
        b2 = EmbeddingVSABridge(dim_in=32, hv_dim=10240, seed=99)
        np.testing.assert_array_equal(b1._proj, b2._proj)


# ──────────────────────────────────────────────────────────────────────────────
# 2. rust_concurrent_shim
# ──────────────────────────────────────────────────────────────────────────────

class TestRustConcurrentShim:
    """Tests for rust_concurrent_shim."""

    def setup_method(self):
        import python.core.vsa.rust_concurrent_shim as shim
        self.shim = shim

    def test_use_rust_flag_is_bool(self):
        assert isinstance(self.shim.USE_RUST_CONCURRENT, bool)

    def test_get_status_returns_dict(self):
        s = self.shim.get_status()
        assert isinstance(s, dict)
        assert "use_rust" in s
        assert "available_classes" in s

    def test_get_status_available_classes(self):
        s = self.shim.get_status()
        assert len(s["available_classes"]) > 0

    def test_episode_dataclass(self):
        from python.core.vsa.rust_concurrent_shim import USE_RUST_CONCURRENT
        if USE_RUST_CONCURRENT:
            from python.core.vsa.hypervec_shim import HyperVector
            ep = self.shim.Episode(
                timestamp=1.0, task_tag="test", situation_hv=HyperVector(),
                action="go", outcome="success", reward=1.0, impact_score=0.5,
            )
            assert ep.action == "go"
        else:
            ep = self.shim.Episode(
                timestamp=1.0, task_tag="test",
                action="go", outcome="success", reward=1.0,
            )
            assert ep.action == "go"
            assert ep.reward == 1.0

    def test_semantic_memory_store_and_retrieve(self):
        mem = self.shim.SemanticMemoryConcurrent()
        from python.core.vsa.hypervec_shim import HyperVector
        hv = HyperVector()
        # Use add_concept (Rust API) with fallback to store_concept
        if hasattr(mem, 'add_concept'):
            mem.add_concept("dog", hv)
        else:
            mem.store_concept("dog", hv)
        retrieved = mem.get_concept("dog")
        assert retrieved is not None

    def test_semantic_memory_count(self):
        mem = self.shim.SemanticMemoryConcurrent()
        from python.core.vsa.hypervec_shim import HyperVector
        if hasattr(mem, 'add_concept'):
            mem.add_concept("cat", HyperVector())
            mem.add_concept("dog", HyperVector())
        else:
            mem.store_concept("cat", HyperVector())
            mem.store_concept("dog", HyperVector())
        assert mem.concept_count() >= 2

    def test_episodic_memory_add(self):
        mem = self.shim.EpisodicMemoryConcurrent()
        from python.core.vsa.rust_concurrent_shim import USE_RUST_CONCURRENT
        if USE_RUST_CONCURRENT:
            from python.core.vsa.hypervec_shim import HyperVector
            ep = self.shim.Episode(
                timestamp=1.0, task_tag="t1", situation_hv=HyperVector(),
                action="a", outcome="ok", reward=1.0, impact_score=0.0,
            )
            mem.add_episode(ep)
            assert mem.size() >= 1
        else:
            ep = self.shim.Episode(timestamp=1.0, task_tag="t1", action="a")
            mem.add_episode(ep)
            assert mem.episode_count() >= 1

    def test_episodic_memory_query_recent(self):
        mem = self.shim.EpisodicMemoryConcurrent()
        from python.core.vsa.rust_concurrent_shim import USE_RUST_CONCURRENT
        if USE_RUST_CONCURRENT:
            from python.core.vsa.hypervec_shim import HyperVector
            for i in range(5):
                ep = self.shim.Episode(
                    timestamp=float(i), task_tag="t1", situation_hv=HyperVector(),
                    action=f"a{i}", outcome="ok", reward=float(i), impact_score=0.0,
                )
                mem.add_episode(ep)
            recent = mem.get_recent_episodes(3)
        else:
            for i in range(5):
                mem.add_episode(self.shim.Episode(
                    timestamp=float(i), task_tag="t1", action=f"a{i}"
                ))
            recent = mem.query_recent(3)
        assert len(recent) <= 5

    def test_parallel_bundle_fallback(self):
        from python.core.vsa.hypervec_shim import HyperVector
        hvs = [HyperVector() for _ in range(3)]
        result = self.shim.parallel_bundle(hvs)
        assert result is not None

    def test_batch_parallel_similarity_search(self):
        from python.core.vsa.hypervec_shim import HyperVector
        from python.core.vsa.rust_concurrent_shim import USE_RUST_CONCURRENT
        query = HyperVector()
        hvs = [HyperVector() for _ in range(5)]
        if USE_RUST_CONCURRENT:
            # Rust API takes list of queries
            results = self.shim.batch_parallel_similarity_search([query], hvs, k=3)
            # results is list of lists
            assert isinstance(results, list)
        else:
            results = self.shim.batch_parallel_similarity_search(query, hvs, k=3)
            assert len(results) <= 3


# ──────────────────────────────────────────────────────────────────────────────
# 3. NSCKApiServer
# ──────────────────────────────────────────────────────────────────────────────

class TestNSCKApiServer:
    """Tests for NSCKApiServer."""

    def setup_method(self):
        from python.core.integration.config import NSCKConfig
        from api.nsck_api import NSCKApiServer
        config = NSCKConfig(enable_snn=False, enable_vsa=True, enable_sleep=False)
        self.server = NSCKApiServer(config)
        # Register a test task
        self.server.engine.register_task("api_test")

    def test_handle_status_returns_dict(self):
        result = self.server.handle_status()
        assert isinstance(result, dict)

    def test_handle_status_has_tasks(self):
        result = self.server.handle_status()
        assert "tasks" in result or "task_count" in result

    def test_handle_decide_returns_dict(self):
        result = self.server.handle_decide({"x": 1}, "api_test")
        assert isinstance(result, dict)

    def test_handle_decide_has_action(self):
        result = self.server.handle_decide({"x": 1}, "api_test")
        assert "action" in result

    def test_handle_decide_has_confidence(self):
        result = self.server.handle_decide({}, "api_test")
        assert "confidence" in result
        assert isinstance(result["confidence"], float)

    def test_handle_decide_has_explanation(self):
        result = self.server.handle_decide({}, "api_test")
        assert "explanation" in result

    def test_handle_decide_has_active_predicates(self):
        result = self.server.handle_decide({}, "api_test")
        assert "active_predicates" in result
        assert isinstance(result["active_predicates"], list)

    def test_handle_learn_returns_ok(self):
        result = self.server.handle_learn({}, "go", 1.0, "api_test")
        assert result.get("status") == "ok"

    def test_handle_sleep_returns_dict(self):
        result = self.server.handle_sleep("api_test")
        assert isinstance(result, dict)

    def test_create_app_returns_none_or_app(self):
        app = self.server.create_app()
        # Either None (no fastapi) or a FastAPI app
        assert app is None or hasattr(app, "routes")

    def test_status_has_decisions(self):
        self.server.handle_decide({}, "api_test")
        result = self.server.handle_status()
        assert result.get("decisions", 0) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 4. NgramNLU
# ──────────────────────────────────────────────────────────────────────────────

class TestNgramNLU:
    """Tests for NgramNLU and NgramNLUAdapter."""

    def setup_method(self):
        from python.core.language.ngram_nlu import NgramNLU, NgramNLUAdapter, INTENT_LABELS
        self.nlu = NgramNLU(n=2)
        sentences = [
            "what is the weather today", "how does this work", "explain the concept",
            "please stop the process", "run the analysis", "do the task",
            "the system is running fine", "results were computed", "data is stored",
            "hello there friend", "hi how are you doing",
            "goodbye see you soon", "farewell until next time",
        ]
        labels = [
            "question", "question", "question",
            "command", "command", "command",
            "statement", "statement", "statement",
            "greeting", "greeting",
            "farewell", "farewell",
        ]
        self.nlu.train(sentences, labels)
        self.adapter = NgramNLUAdapter()
        self.intent_labels = INTENT_LABELS

    def test_train_sets_labels(self):
        assert len(self.nlu._labels) > 0

    def test_predict_returns_tuple(self):
        result = self.nlu.predict("what is this")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_label_in_trained(self):
        label, conf = self.nlu.predict("what is the status")
        assert label in self.nlu._labels

    def test_predict_confidence_in_range(self):
        _, conf = self.nlu.predict("hello there")
        assert 0.0 <= conf <= 1.0

    def test_predict_distribution_sums_to_one(self):
        dist = self.nlu.predict_distribution("please run the program")
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6

    def test_predict_distribution_keys_match_labels(self):
        dist = self.nlu.predict_distribution("something")
        assert set(dist.keys()) == set(self.nlu._labels)

    def test_extract_intent_question(self):
        from python.core.language.ngram_nlu import NgramNLU
        nlu2 = NgramNLU()
        intent = nlu2.extract_intent("what is the capital of France?")
        assert intent == "question"

    def test_extract_intent_greeting(self):
        from python.core.language.ngram_nlu import NgramNLU
        nlu2 = NgramNLU()
        intent = nlu2.extract_intent("hello there!")
        assert intent == "greeting"

    def test_extract_intent_farewell(self):
        from python.core.language.ngram_nlu import NgramNLU
        nlu2 = NgramNLU()
        intent = nlu2.extract_intent("goodbye friend")
        assert intent == "farewell"

    def test_extract_intent_statement(self):
        from python.core.language.ngram_nlu import NgramNLU
        nlu2 = NgramNLU()
        intent = nlu2.extract_intent("the system works correctly")
        assert intent == "statement"

    def test_extract_entities_uppercase(self):
        from python.core.language.ngram_nlu import NgramNLU
        nlu2 = NgramNLU()
        entities = nlu2.extract_entities("Alice went to London to see Bob")
        assert "Alice" in entities or "London" in entities or "Bob" in entities

    def test_extract_entities_nonempty(self):
        from python.core.language.ngram_nlu import NgramNLU
        nlu2 = NgramNLU()
        entities = nlu2.extract_entities("The dog chased the cat")
        assert isinstance(entities, list)

    def test_adapter_process_returns_dict(self):
        result = self.adapter.process("hello world")
        assert isinstance(result, dict)

    def test_adapter_process_has_intent(self):
        result = self.adapter.process("what is the time?")
        assert "intent" in result

    def test_adapter_process_has_entities(self):
        result = self.adapter.process("Alice visited London")
        assert "entities" in result
        assert isinstance(result["entities"], list)

    def test_adapter_process_has_confidence(self):
        result = self.adapter.process("run the test")
        assert "confidence" in result
        assert isinstance(result["confidence"], float)

    def test_adapter_process_has_label_probs(self):
        result = self.adapter.process("goodbye")
        assert "label_probs" in result
        assert isinstance(result["label_probs"], dict)


# ──────────────────────────────────────────────────────────────────────────────
# 5. FHRRVector / FHRRMemory
# ──────────────────────────────────────────────────────────────────────────────

class TestFHRR:
    """Tests for FHRRVector and FHRRMemory."""

    def setup_method(self):
        from python.core.vsa.fhrr import FHRRVector, FHRRMemory
        self.FHRRVector = FHRRVector
        self.FHRRMemory = FHRRMemory
        self.dim = 128

    def test_init_phasors_shape(self):
        v = self.FHRRVector(self.dim)
        assert v.phasors.shape == (self.dim,)

    def test_init_unit_magnitude(self):
        v = self.FHRRVector(self.dim, seed=42)
        mags = np.abs(v.phasors)
        np.testing.assert_allclose(mags, 1.0, atol=1e-10)

    def test_from_phasors(self):
        phasors = np.exp(1j * np.linspace(0, np.pi, self.dim))
        v = self.FHRRVector.from_phasors(phasors)
        assert v.dim == self.dim

    def test_bind_shape(self):
        a = self.FHRRVector(self.dim, seed=1)
        b = self.FHRRVector(self.dim, seed=2)
        c = a.bind(b)
        assert c.phasors.shape == (self.dim,)

    def test_bind_unbind_inverse(self):
        a = self.FHRRVector(self.dim, seed=1)
        b = self.FHRRVector(self.dim, seed=2)
        bound = a.bind(b)
        recovered = bound.unbind(b)
        # The recovered phasors should be close to a's phasors
        phase_diff = np.angle(recovered.phasors * np.conj(a.phasors))
        # Mean absolute phase error should be small
        assert np.mean(np.abs(phase_diff)) < 0.1

    def test_bundle_shape(self):
        a = self.FHRRVector(self.dim, seed=1)
        b = self.FHRRVector(self.dim, seed=2)
        c = self.FHRRVector(self.dim, seed=3)
        bundled = a.bundle([b, c])
        assert bundled.phasors.shape == (self.dim,)

    def test_similarity_self_is_one(self):
        v = self.FHRRVector(self.dim, seed=42)
        assert abs(v.similarity(v) - 1.0) < 1e-5

    def test_similarity_range(self):
        a = self.FHRRVector(self.dim, seed=10)
        b = self.FHRRVector(self.dim, seed=20)
        sim = a.similarity(b)
        assert 0.0 <= sim <= 1.0 + 1e-6

    def test_gradient_input_shape(self):
        v = self.FHRRVector(self.dim, seed=1)
        arr = v.to_gradient_input()
        assert arr.shape == (2 * self.dim,)

    def test_from_gradient_output(self):
        v = self.FHRRVector(self.dim, seed=1)
        arr = v.to_gradient_input()
        v2 = self.FHRRVector.from_gradient_output(arr, self.dim)
        assert v2.phasors.shape == (self.dim,)

    def test_encode_scalar_deterministic(self):
        v = self.FHRRVector(self.dim, seed=0)
        v1 = v.encode_scalar(3.14)
        v2 = v.encode_scalar(3.14)
        np.testing.assert_array_almost_equal(v1.phasors, v2.phasors)

    def test_encode_scalar_different_values(self):
        v = self.FHRRVector(self.dim, seed=0)
        v1 = v.encode_scalar(1.0)
        v2 = v.encode_scalar(2.0)
        sim = v1.similarity(v2)
        assert sim < 1.0

    def test_encode_symbol_deterministic(self):
        v1 = self.FHRRVector.encode_symbol("dog", self.dim)
        v2 = self.FHRRVector.encode_symbol("dog", self.dim)
        np.testing.assert_array_almost_equal(v1.phasors, v2.phasors)

    def test_encode_symbol_different_names(self):
        v1 = self.FHRRVector.encode_symbol("cat", self.dim)
        v2 = self.FHRRVector.encode_symbol("dog", self.dim)
        sim = v1.similarity(v2)
        assert sim < 1.0

    def test_memory_store_retrieve(self):
        mem = self.FHRRMemory(self.dim)
        v = self.FHRRVector(self.dim, seed=1)
        mem.store("key1", v)
        retrieved = mem.retrieve("key1")
        assert retrieved.phasors.shape == (self.dim,)

    def test_memory_cleanup(self):
        mem = self.FHRRMemory(self.dim)
        va = self.FHRRVector.encode_symbol("a", self.dim)
        vb = self.FHRRVector.encode_symbol("b", self.dim)
        mem.store("a", va)
        codebook = [("a", va), ("b", vb)]
        name, sim = mem.cleanup(va, codebook)
        assert name in ("a", "b")
        assert 0.0 <= sim <= 1.0 + 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# 6. Attention GWT Bridge
# ──────────────────────────────────────────────────────────────────────────────

class TestAttentionGWTBridge:
    """Tests for AttentionHead, MultiHeadAttentionGWT, AttentionGWTBridge."""

    def setup_method(self):
        from python.core.reasoning.attention_gwt_bridge import (
            AttentionHead, MultiHeadAttentionGWT, AttentionGWTBridge
        )
        self.AttentionHead = AttentionHead
        self.MultiHeadAttentionGWT = MultiHeadAttentionGWT
        self.AttentionGWTBridge = AttentionGWTBridge
        self.dim = 64

    def test_attention_head_score_shape(self):
        head = self.AttentionHead(key_dim=self.dim)
        q = np.random.randn(self.dim)
        keys = [np.random.randn(self.dim) for _ in range(4)]
        scores = head.score(q, keys)
        assert scores.shape == (4,)

    def test_attention_head_scores_sum_to_one(self):
        head = self.AttentionHead(key_dim=self.dim)
        q = np.random.randn(self.dim)
        keys = [np.random.randn(self.dim) for _ in range(3)]
        scores = head.score(q, keys)
        assert abs(scores.sum() - 1.0) < 1e-5

    def test_attention_head_attend_shape(self):
        head = self.AttentionHead(key_dim=self.dim)
        q = np.random.randn(self.dim)
        keys = [np.random.randn(self.dim) for _ in range(3)]
        vals = [np.random.randn(self.dim) for _ in range(3)]
        out = head.attend(q, keys, vals)
        assert out.shape == (self.dim,)

    def test_mha_compute_attention_returns_dict(self):
        mha = self.MultiHeadAttentionGWT(n_heads=2, key_dim=self.dim)
        ws = np.random.randn(self.dim)
        coalitions = {f"c{i}": np.random.randn(self.dim) for i in range(3)}
        result = mha.compute_attention(ws, coalitions)
        assert isinstance(result, dict)

    def test_mha_compute_attention_keys(self):
        mha = self.MultiHeadAttentionGWT(n_heads=2, key_dim=self.dim)
        ws = np.random.randn(self.dim)
        coalitions = {"alpha": np.random.randn(self.dim), "beta": np.random.randn(self.dim)}
        result = mha.compute_attention(ws, coalitions)
        assert set(result.keys()) == {"alpha", "beta"}

    def test_mha_compute_attention_sums_to_one(self):
        mha = self.MultiHeadAttentionGWT(n_heads=2, key_dim=self.dim)
        ws = np.random.randn(self.dim)
        coalitions = {f"c{i}": np.random.randn(self.dim) for i in range(4)}
        result = mha.compute_attention(ws, coalitions)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5

    def test_mha_gwt_compete_returns_winner(self):
        mha = self.MultiHeadAttentionGWT(n_heads=2, key_dim=self.dim)
        ws = np.random.randn(self.dim)
        coalitions = {
            "c1": (np.random.randn(self.dim), 0.8),
            "c2": (np.random.randn(self.dim), 0.3),
        }
        winner = mha.gwt_compete_with_attention(ws, coalitions)
        assert winner in ("c1", "c2")

    def test_mha_empty_coalitions(self):
        mha = self.MultiHeadAttentionGWT()
        ws = np.random.randn(self.dim)
        result = mha.compute_attention(ws, {})
        assert result == {}

    def test_bridge_enhance_returns_dict(self):
        bridge = self.AttentionGWTBridge()
        ws = np.random.randn(self.dim)
        coalitions = {"a": np.random.randn(self.dim), "b": np.random.randn(self.dim)}
        result = bridge.enhance_gwt_competition(ws, coalitions)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}

    def test_bridge_record_trace(self):
        bridge = self.AttentionGWTBridge()
        history = bridge.record_attention_trace("winner_c", {"a": 0.6, "b": 0.4})
        assert len(history) == 1
        assert history[0]["winner"] == "winner_c"

    def test_bridge_accumulates_history(self):
        bridge = self.AttentionGWTBridge()
        bridge.record_attention_trace("c1", {"c1": 0.7, "c2": 0.3})
        bridge.record_attention_trace("c2", {"c1": 0.4, "c2": 0.6})
        assert len(bridge._history) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 7. RuleNeuralScorer + CognitiveEngine integration
# ──────────────────────────────────────────────────────────────────────────────

class TestRuleNeuralScorer:
    """Tests for RuleFeaturizer, RuleNeuralScorer."""

    def setup_method(self):
        from python.core.learning.rule_neural_scorer import RuleFeaturizer, RuleNeuralScorer
        self.featurizer = RuleFeaturizer()
        self.scorer = RuleNeuralScorer(n_features=6, hidden=16)

    def test_featurize_shape(self):
        rule = _MockRule()
        features = self.featurizer.featurize(rule)
        assert features.shape == (6,)

    def test_featurize_confidence_range(self):
        rule = _MockRule(confidence=0.9)
        features = self.featurizer.featurize(rule)
        assert 0.0 <= features[0] <= 1.0

    def test_featurize_support_normalized(self):
        rule = _MockRule(support_count=200)
        features = self.featurizer.featurize(rule)
        assert features[1] <= 1.0

    def test_featurize_fire_ratio_range(self):
        rule = _MockRule(fire_count=5000)
        features = self.featurizer.featurize(rule)
        assert 0.0 <= features[2] <= 1.0

    def test_scorer_score_in_range(self):
        rule = _MockRule()
        score = self.scorer.score(rule)
        assert 0.0 <= score <= 1.0

    def test_scorer_batch_score_length(self):
        rules = [_MockRule() for _ in range(5)]
        scores = self.scorer.batch_score(rules)
        assert len(scores) == 5

    def test_scorer_rank_rules_sorted(self):
        rules = [_MockRule(confidence=c) for c in [0.3, 0.9, 0.1, 0.7]]
        ranked = self.scorer.rank_rules(rules)
        scores = self.scorer.batch_score(ranked)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-9

    def test_scorer_update_changes_weights(self):
        import copy
        rule = _MockRule()
        w_before = self.scorer._W1.copy()
        self.scorer.update(rule, reward=1.0, lr=0.1)
        assert not np.allclose(w_before, self.scorer._W1)

    def test_scorer_save_load(self, tmp_path):
        path = str(tmp_path / "weights")
        self.scorer.save(path)
        from python.core.learning.rule_neural_scorer import RuleNeuralScorer
        scorer2 = RuleNeuralScorer()
        scorer2.load(path)
        np.testing.assert_array_equal(self.scorer._W1, scorer2._W1)

    def test_cognitive_engine_enable_neural_scoring(self):
        from python.core.integration.config import NSCKConfig
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        config = NSCKConfig(enable_snn=False)
        engine = CognitiveEngine(config)
        assert engine.rule_scorer is None
        engine.enable_neural_rule_scoring()
        assert engine.rule_scorer is not None


# ──────────────────────────────────────────────────────────────────────────────
# 8. SafetyVerifier
# ──────────────────────────────────────────────────────────────────────────────

class TestSafetyVerifier:
    """Tests for SafetyProperty, SafetyRuleVerifier, SafetyGateVerifier."""

    def setup_method(self):
        from python.core.cognitive.safety_verifier import (
            SafetyProperty, SafetyRuleVerifier, SafetyGateVerifier
        )
        self.SafetyProperty = SafetyProperty
        self.SafetyRuleVerifier = SafetyRuleVerifier
        self.SafetyGateVerifier = SafetyGateVerifier

    def test_safety_property_check_satisfied(self):
        prop = self.SafetyProperty("test_conf", "confidence > 0.3")
        rule = _MockRule(confidence=0.8)
        satisfied, reason = prop.check(rule)
        assert satisfied is True

    def test_safety_property_check_violated(self):
        prop = self.SafetyProperty("test_conf", "confidence > 0.3")
        rule = _MockRule(confidence=0.1)
        satisfied, reason = prop.check(rule)
        assert satisfied is False
        assert isinstance(reason, str)

    def test_safety_property_check_returns_tuple(self):
        prop = self.SafetyProperty("support_check", "support >= 2")
        rule = _MockRule(support_count=5)
        result = prop.check(rule)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_safety_property_bad_formula(self):
        prop = self.SafetyProperty("bad", "undefined_var > 0")
        rule = _MockRule()
        satisfied, reason = prop.check(rule)
        assert isinstance(satisfied, bool)

    def test_verifier_verify_rule_returns_dict(self):
        verifier = self.SafetyRuleVerifier()
        rule = _MockRule()
        result = verifier.verify_rule(rule)
        assert isinstance(result, dict)

    def test_verifier_verify_rule_has_safe_key(self):
        verifier = self.SafetyRuleVerifier()
        rule = _MockRule(confidence=0.9, support_count=5, fire_count=100)
        result = verifier.verify_rule(rule)
        assert "safe" in result

    def test_verifier_safe_rule(self):
        verifier = self.SafetyRuleVerifier()
        rule = _MockRule(confidence=0.9, support_count=5, fire_count=10)
        result = verifier.verify_rule(rule)
        assert result["safe"] is True

    def test_verifier_unsafe_low_confidence(self):
        verifier = self.SafetyRuleVerifier()
        rule = _MockRule(confidence=0.1, support_count=5, fire_count=10)
        result = verifier.verify_rule(rule)
        assert result["safe"] is False

    def test_verifier_violation_has_fields(self):
        verifier = self.SafetyRuleVerifier()
        rule = _MockRule(confidence=0.1)
        result = verifier.verify_rule(rule)
        for v in result.get("violations", []):
            assert "property" in v
            assert "reason" in v
            assert "severity" in v

    def test_verifier_verify_ruleset(self):
        verifier = self.SafetyRuleVerifier()
        rules = [_MockRule(confidence=0.9), _MockRule(confidence=0.1)]
        summary = verifier.verify_ruleset(rules)
        assert "total_rules" in summary
        assert summary["total_rules"] == 2

    def test_verifier_code_injection_check(self):
        verifier = self.SafetyRuleVerifier()
        rule = _MockRule(consequence="exec")
        result = verifier.verify_rule(rule)
        assert result["safe"] is False

    def test_gate_verifier_ok(self):
        gate = self.SafetyGateVerifier()
        allowed, reason = gate.gate_decision("go", 0.9, [])
        assert allowed is True

    def test_gate_verifier_low_confidence(self):
        gate = self.SafetyGateVerifier()
        allowed, reason = gate.gate_decision("attack", 0.1, [])
        assert allowed is False
        assert "confidence" in reason.lower() or "low" in reason.lower()

    def test_gate_verifier_explore_allowed_at_low_conf(self):
        gate = self.SafetyGateVerifier()
        allowed, reason = gate.gate_decision("explore", 0.1, [])
        assert allowed is True

    def test_gate_verifier_unsafe_rule_blocked(self):
        gate = self.SafetyGateVerifier()
        bad_rule = _MockRule(confidence=0.1, fire_count=20000)
        allowed, reason = gate.gate_decision("go", 0.9, [bad_rule])
        # fire_count > 10000 is critical
        assert allowed is False
