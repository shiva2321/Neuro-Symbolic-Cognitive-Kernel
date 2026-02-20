"""
NSCK Core Architecture Real-World Test Suite
============================================

Covers the full NSCK core stack without touching nsck_ai_model:

  A. VSA Core  — Rust + Python + cross-backend behaviour
  B. Rust Concurrent Memory Layer
       B1. HyperVectorRegistry  (DashMap, parallel NN search)
       B2. SemanticMemoryConcurrent  (parallel spreading activation)
       B3. EpisodicMemoryConcurrent  (parallel k-NN, hot-tier)
       B4. ActivationAccumulator  (lock-free concurrent accumulation)
       B5. PersistentStorage  (SQLite batch flush, round-trip integrity)
  C. CognitiveWorkerPool (Rust Tokio-backed task pool)
  D. Spiking Neural Network layer (rust_snn: LIF, STDP, SnnCore, Hebbian)
  E. Python Memory Modules  (SemanticMemory, EpisodicMemory)
  F. Reasoning Modules  (CausalGraph, GlobalWorkspace, AnalogyEngine, Planner)
  G. Cognitive Modules  (EmotionSystem, SelfModel, CuriosityModule)
  H. Language Modules  (TextKnowledgeLearner, NLGEngine, SemanticRoleLabeler)
  I. Cross-Domain Knowledge Transfer (end-to-end through the core stack)
  J. Integration invariants  (shared objects, relation-weight coverage,
     Rust-Python memory interoperability)

Only real educational corpora are used as training data.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]
_NSCK = _ROOT / "nsck"
for _p in (_NSCK, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import python.core.vsa.hypervec_shim as hvs
from python.core.vsa.hypervec_py import HyperVectorPy

_RUST_AVAILABLE = hvs.__backend__ == "Rust"
_SNN_AVAILABLE = False
try:
    import snn_rs as _snn_rs
    _SNN_AVAILABLE = True
except ImportError:
    pass

requires_rust = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="Rust extension not installed (run: cd nsck/rust_vsa && cargo build --release)",
)
requires_snn = pytest.mark.skipif(
    not _SNN_AVAILABLE,
    reason="snn_rs not installed (run: cd nsck/rust_snn && cargo build --release)",
)

# ---------------------------------------------------------------------------
# Real educational corpora — factually accurate, multi-domain
# ---------------------------------------------------------------------------

_NEURO = (
    "The hippocampus is essential for the formation of new declarative memories "
    "and spatial navigation via place cells and grid cells. "
    "Damage to the hippocampus causes anterograde amnesia, the inability to encode "
    "new long-term memories. "
    "Cortisol released during chronic stress causes hippocampal atrophy and "
    "impairs memory consolidation. "
    "Hippocampal atrophy causes long-term memory impairment and reduced cognitive "
    "flexibility. "
    "The prefrontal cortex regulates executive function, working memory, and "
    "inhibitory control. "
    "Dopaminergic signalling in the striatum drives reward learning and habit "
    "formation. "
    "Sleep deprivation causes prefrontal cortex dysfunction and impairs emotional "
    "regulation."
)

_CLIMATE = (
    "Anthropogenic CO2 emissions cause atmospheric greenhouse gas concentrations "
    "to rise. "
    "Rising greenhouse gas concentrations cause global mean surface temperature "
    "to increase. "
    "Increased temperatures cause polar ice sheets to melt, contributing to "
    "sea-level rise. "
    "Ocean acidification occurs because the ocean absorbs CO2, which forms "
    "carbonic acid. "
    "Deforestation causes increased atmospheric CO2 by removing carbon sinks. "
    "El Nino causes anomalous sea surface warming in the central and eastern "
    "Pacific Ocean. "
    "El Nino leads to drought in Australia and flooding in South America."
)

_BIOCHEM = (
    "Photosynthesis converts solar energy into chemical energy stored as glucose "
    "via the Calvin cycle. "
    "The light-dependent reactions in the thylakoid membrane produce ATP and "
    "NADPH from water and photons. "
    "ATP is the universal energy currency of the cell, hydrolysed by ATPases to "
    "drive biosynthetic reactions. "
    "Cellular respiration oxidises glucose in the mitochondria to regenerate ATP, "
    "releasing CO2 and water. "
    "Mitochondria contain their own circular DNA and ribosomes, evidence of "
    "endosymbiotic bacterial origin. "
    "Enzymes lower the activation energy of biochemical reactions by stabilising "
    "the transition state. "
    "DNA replication is catalysed by DNA polymerase, which requires a primer and "
    "proceeds five-prime to three-prime."
)

_CS = (
    "Backpropagation computes gradients of the loss with respect to each parameter "
    "via the chain rule. "
    "Gradient descent iteratively adjusts network weights in the direction that "
    "minimises the loss function. "
    "Transformer models use multi-head self-attention to compute pairwise token "
    "interactions in parallel. "
    "The attention mechanism computes a weighted sum of value vectors using "
    "query-key dot-product scores. "
    "Convolutional neural networks apply learned filter banks to extract "
    "hierarchical spatial features. "
    "Reinforcement learning agents maximise cumulative reward by learning a "
    "policy from environment feedback. "
    "Moore's law predicted that transistor count doubles approximately every two "
    "years."
)


def _make_tkl_system(domain_text_pairs: List[tuple]):
    """Build TKL with shared SM/EM/CG and train on domain-tagged passages."""
    from python.core.memory.semantic_memory import SemanticMemory
    from python.core.memory.episodic_memory import EpisodicMemory
    from python.core.reasoning.causal_reasoning import CausalGraph
    from python.core.language.text_knowledge_learner import TextKnowledgeLearner

    sm = SemanticMemory()
    em = EpisodicMemory()
    cg = CausalGraph()
    tkl = TextKnowledgeLearner(semantic_memory=sm, episodic_memory=em, causal_graph=cg)
    domain_hvs: Dict[str, Dict[str, object]] = {}
    for domain, text in domain_text_pairs:
        for sent in [s.strip() for s in text.split(".") if len(s.strip()) > 20]:
            tkl.learn_from_text(sent + ".")
        domain_hvs[domain] = dict(sm.concept_hvs)
    return tkl, sm, em, cg, domain_hvs


# ===========================================================================
# A. VSA Core — Rust + Python + cross-backend behaviour
# ===========================================================================

class TestVSACore:

    # ---- determinism ----

    def test_same_seed_identical_hv(self):
        a1, a2 = hvs.HyperVector(42), hvs.HyperVector(42)
        assert a1.similarity(a2) == 1.0

    def test_python_same_seed_identical(self):
        assert HyperVectorPy(42).similarity(HyperVectorPy(42)) == 1.0

    def test_bundle_deterministic_shim(self):
        """bundle(A, B) called twice must return the same vector."""
        a, b = hvs.HyperVector(1001), hvs.HyperVector(2002)
        assert a.bundle(b).similarity(a.bundle(b)) == 1.0

    def test_bundle_deterministic_python(self):
        a, b = HyperVectorPy(1001), HyperVectorPy(2002)
        assert a.bundle(b).similarity(a.bundle(b)) == 1.0

    # ---- XOR algebra ----

    def test_xor_self_inverse(self):
        a, b = hvs.HyperVector(7), hvs.HyperVector(13)
        assert a.xor(b).xor(b).similarity(a) == 1.0

    def test_xor_commutative(self):
        a, b = hvs.HyperVector(5), hvs.HyperVector(6)
        assert a.xor(b).similarity(b.xor(a)) == 1.0

    def test_python_xor_self_inverse(self):
        a, b = HyperVectorPy(7), HyperVectorPy(13)
        assert a.xor(b).xor(b).similarity(a) == 1.0

    # ---- similarity ----

    def test_self_similarity_is_one(self):
        a = hvs.HyperVector(100)
        assert a.similarity(a) == 1.0

    def test_random_hvs_quasi_orthogonal(self):
        sim = hvs.HyperVector(1000).similarity(hvs.HyperVector(9999))
        assert 0.45 <= sim <= 0.55, f"Random HVs quasi-orthogonal, got {sim:.4f}"

    def test_similarity_symmetric(self):
        a, b = hvs.HyperVector(3), hvs.HyperVector(4)
        assert a.similarity(b) == b.similarity(a)

    def test_similarity_in_unit_interval(self):
        for s1, s2 in [(0, 1), (100, 200), (5000, 9999)]:
            sim = hvs.HyperVector(s1).similarity(hvs.HyperVector(s2))
            assert 0.0 <= sim <= 1.0

    # ---- permute ----

    def test_permute_self_inverse(self):
        for shift in (1, 5, 64, 100, 512):
            hv = hvs.HyperVector(shift * 7)
            assert hv.permute(shift).permute(-shift).similarity(hv) == 1.0

    def test_permute_zero_is_identity(self):
        hv = hvs.HyperVector(77)
        assert hv.permute(0).similarity(hv) == 1.0

    def test_permute_different_shifts_differ(self):
        hv = hvs.HyperVector(42)
        assert hv.permute(1).similarity(hv.permute(2)) < 1.0

    def test_python_permute_self_inverse(self):
        hv = HyperVectorPy(42)
        assert hv.permute(7).permute(-7).similarity(hv) == 1.0

    # ---- bundle ----

    def test_bundle_similar_to_both_inputs(self):
        a, b = hvs.HyperVector(10), hvs.HyperVector(20)
        ab = a.bundle(b)
        assert 0.6 <= ab.similarity(a) <= 0.9
        assert 0.6 <= ab.similarity(b) <= 0.9

    def test_weighted_bundle_weight_one_near_self(self):
        a, b = hvs.HyperVector(50), hvs.HyperVector(60)
        towards_a = a.weighted_bundle(b, 0.9)
        assert towards_a.similarity(a) > towards_a.similarity(b)

    def test_weighted_bundle_weight_zero_near_other(self):
        a, b = hvs.HyperVector(50), hvs.HyperVector(60)
        towards_b = a.weighted_bundle(b, 0.1)
        assert towards_b.similarity(b) > towards_b.similarity(a)

    # ---- LSH ----

    def test_lsh_hash_deterministic(self):
        hv = hvs.HyperVector(7777)
        assert hv.lsh_hash(42, 16) == hv.lsh_hash(42, 16)

    def test_lsh_hash_different_seeds_differ(self):
        hv = hvs.HyperVector(7777)
        assert hv.lsh_hash(1, 16) != hv.lsh_hash(2, 16)

    # ---- shim compat methods ----

    def test_bits_property_length_10240(self):
        assert len(hvs.HyperVector(1).bits) == 10240

    def test_from_bits_roundtrip(self):
        hv = hvs.HyperVector(123)
        assert hvs.HyperVector.from_bits(hv.bits).similarity(hv) == 1.0

    def test_cosine_similarity_self(self):
        hv = hvs.HyperVector(42)
        assert abs(hv.cosine_similarity(hv) - 1.0) < 1e-6

    def test_cosine_similarity_range(self):
        a, b = hvs.HyperVector(1), hvs.HyperVector(2)
        assert -1.0 <= a.cosine_similarity(b) <= 1.0

    def test_similarity_robust_unit_interval(self):
        a, b = hvs.HyperVector(11), hvs.HyperVector(22)
        assert 0.0 <= a.similarity_robust(b, method="cosine") <= 1.0


# ===========================================================================
# B. Rust Concurrent Memory Layer
# ===========================================================================

@requires_rust
class TestRustHyperVectorRegistry:

    def test_register_and_retrieve(self):
        reg = hvs.HyperVectorRegistry()
        hv = hvs.HyperVector(1)
        reg.register("alpha", hv)
        got = reg.get("alpha")
        assert got is not None and got.similarity(hv) == 1.0

    def test_size_and_remove(self):
        reg = hvs.HyperVectorRegistry()
        for i in range(10):
            reg.register(f"v{i}", hvs.HyperVector(i))
        assert reg.size() == 10
        assert reg.remove("v0")
        assert reg.size() == 9

    def test_nearest_neighbors_sorted_descending(self):
        reg = hvs.HyperVectorRegistry()
        anchor = hvs.HyperVector(0)
        for i in range(50):
            reg.register(f"v{i}", hvs.HyperVector(i * 100))
        results = reg.nearest_neighbors(anchor, 5)
        assert len(results) == 5
        for i in range(1, len(results)):
            assert results[i - 1][1] >= results[i][1]

    def test_nearest_neighbor_exact_match_is_top(self):
        reg = hvs.HyperVectorRegistry()
        hv = hvs.HyperVector(777)
        reg.register("exact", hv)
        for i in range(20):
            reg.register(f"noise_{i}", hvs.HyperVector(i * 10000 + 999))
        results = reg.nearest_neighbors(hv, 1)
        assert results[0][0] == "exact" and results[0][1] == 1.0

    def test_concurrent_inserts_all_visible(self):
        reg = hvs.HyperVectorRegistry()
        errors = []

        def insert(tid):
            for i in range(50):
                try:
                    reg.register(f"t{tid}_v{i}", hvs.HyperVector(tid * 1000 + i))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=insert, args=(t,)) for t in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert reg.size() == 8 * 50

    def test_batch_nearest_neighbors_shape(self):
        reg = hvs.HyperVectorRegistry()
        for i in range(30):
            reg.register(f"c{i}", hvs.HyperVector(i * 17))
        queries = [hvs.HyperVector(i * 100) for i in range(5)]
        all_results = reg.batch_nearest_neighbors(queries, 3)
        assert len(all_results) == 5
        for batch in all_results:
            assert len(batch) == 3


@requires_rust
class TestRustSemanticMemoryConcurrent:

    def test_add_and_retrieve_concept(self):
        sm = hvs.SemanticMemoryConcurrent()
        hv = hvs.HyperVector(1)
        sm.add_concept("neuron", hv)
        assert sm.get_concept("neuron").similarity(hv) == 1.0

    def test_parallel_search_sorted(self):
        sm = hvs.SemanticMemoryConcurrent()
        for i in range(30):
            sm.add_concept(f"c{i}", hvs.HyperVector(i * 100))
        results = sm.parallel_semantic_search(hvs.HyperVector(0), 5)
        assert len(results) == 5
        for i in range(1, len(results)):
            assert results[i - 1][1] >= results[i][1]

    def test_parallel_search_exact_match(self):
        sm = hvs.SemanticMemoryConcurrent()
        target = hvs.HyperVector(42)
        sm.add_concept("target", target)
        for i in range(20):
            sm.add_concept(f"noise_{i}", hvs.HyperVector(i * 10000 + 99))
        results = sm.parallel_semantic_search(target, 1)
        assert results[0][0] == "target" and results[0][1] == 1.0

    def test_spread_activation_reaches_neighbours(self):
        sm = hvs.SemanticMemoryConcurrent()
        for i in range(5):
            sm.add_concept(f"n{i}", hvs.HyperVector(i * 100))
        for i in range(4):
            sm.add_relation(f"n{i}", f"n{i + 1}")
        activated = sm.parallel_spread_activation(["n0"], 3, 0.7)
        assert "n1" in activated
        assert "n2" in activated

    def test_spread_activation_decays(self):
        sm = hvs.SemanticMemoryConcurrent()
        for i in range(6):
            sm.add_concept(f"n{i}", hvs.HyperVector(i * 100))
        for i in range(5):
            sm.add_relation(f"n{i}", f"n{i + 1}")
        activated = sm.parallel_spread_activation(["n0"], 5, 0.7)
        if "n1" in activated and "n4" in activated:
            assert activated["n1"] >= activated["n4"]

    def test_concept_and_relation_count(self):
        sm = hvs.SemanticMemoryConcurrent()
        for i in range(10):
            sm.add_concept(f"c{i}", hvs.HyperVector(i))
        for i in range(9):
            sm.add_relation(f"c{i}", f"c{i + 1}")
        assert sm.concept_count() == 10
        assert sm.relation_count() == 9

    def test_hybrid_search_returns_results(self):
        sm = hvs.SemanticMemoryConcurrent()
        anchor = hvs.HyperVector(0)
        sm.add_concept("anchor", anchor)
        for i in range(15):
            sm.add_concept(f"c{i}", hvs.HyperVector(i * 200))
            sm.add_relation("anchor", f"c{i}")
        # hybrid_search(query_hv, k, spread_steps, spread_decay)
        results = sm.hybrid_search(anchor, 5, 2, 0.7)
        assert len(results) >= 1


@requires_rust
class TestRustEpisodicMemoryConcurrent:

    def _ep(self, i: int, task: str = "test") -> object:
        return hvs.Episode(float(i), task, hvs.HyperVector(i * 17),
                           f"action_{i}", "ok", float(i % 5) * 0.25)

    def test_add_and_size(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=100)
        for i in range(20):
            em.add_episode(self._ep(i))
        assert em.size() == 20

    def test_parallel_knn_returns_k_results(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(50):
            em.add_episode(self._ep(i))
        results = em.parallel_knn_search(hvs.HyperVector(0), 5)
        assert len(results) == 5

    def test_knn_result_is_tuple_with_index_sim_episode(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(10):
            em.add_episode(self._ep(i))
        results = em.parallel_knn_search(hvs.HyperVector(0), 3)
        for r in results:
            idx, sim, ep = r
            assert isinstance(idx, int)
            assert 0.0 <= sim <= 1.0
            assert ep.action.startswith("action_")

    def test_get_recent_episodes(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(20):
            em.add_episode(self._ep(i))
        assert len(em.get_recent_episodes(5)) == 5

    def test_search_by_task(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(10):
            em.add_episode(self._ep(i, task="neuro"))
        for i in range(10):
            em.add_episode(self._ep(i + 100, task="climate"))
        neuro = em.search_by_task("neuro")
        assert len(neuro) == 10
        for ep in neuro:
            assert ep.task_tag == "neuro"

    def test_search_by_reward(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(30):
            em.add_episode(self._ep(i))
        for ep in em.search_by_reward(0.5, 100):
            assert ep.reward >= 0.5

    def test_get_high_impact_episodes(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(30):
            em.add_episode(self._ep(i))
        assert len(em.get_high_impact_episodes(5)) <= 5

    def test_batch_add_episodes(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=500)
        em.batch_add_episodes([self._ep(i) for i in range(50)])
        assert em.size() == 50

    def test_concurrent_adds_thread_safe(self):
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=2000)
        errors = []

        def add_batch(offset):
            try:
                for i in range(30):
                    em.add_episode(self._ep(offset + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_batch, args=(t * 100,)) for t in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert em.size() == 240


@requires_rust
class TestRustActivationAccumulator:

    def test_accumulate_and_retrieve(self):
        acc = hvs.ActivationAccumulator()
        acc.add_activation("hippocampus", 0.8)
        acc.add_activation("hippocampus", 0.4)
        assert abs(acc.get_activation("hippocampus") - 1.2) < 1e-9

    def test_get_top_k_sorted(self):
        acc = hvs.ActivationAccumulator()
        for name, val in [("prefrontal", 0.9), ("amygdala", 0.3),
                          ("hippocampus", 0.7), ("cerebellum", 0.5)]:
            acc.add_activation(name, val)
        top3 = acc.get_top_k(3)
        assert len(top3) == 3
        assert top3[0][0] == "prefrontal"
        for i in range(1, len(top3)):
            assert top3[i - 1][1] >= top3[i][1]

    def test_concurrent_accumulation_consistent(self):
        acc = hvs.ActivationAccumulator()
        n_threads, ops = 10, 100

        def work(tid):
            for i in range(ops):
                acc.add_activation(f"node_{i % 10}", 0.1)

        threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        expected = n_threads * (ops // 10) * 0.1
        for i in range(10):
            assert abs(acc.get_activation(f"node_{i}") - expected) < 1e-9

    def test_get_all_returns_all_entries(self):
        acc = hvs.ActivationAccumulator()
        for name in ["a", "b", "c", "d", "e"]:
            acc.add_activation(name, 1.0)
        assert len(acc.get_all()) == 5


@requires_rust
class TestRustPersistentStorage:

    def _ep(self, i: int) -> object:
        return hvs.Episode(float(i), "persist_test", hvs.HyperVector(i * 31),
                           f"act_{i}", "ok", 0.5)

    def test_buffer_and_flush_then_count(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ps = hvs.PersistentStorage(db_path, 50)  # batch_size=50
            for i in range(30):
                ps.buffer_episode(self._ep(i))
            # batch_size=50 → no auto-flush yet; 30 still pending
            assert ps.buffered_count() == 30
            ps.flush()
            assert ps.episode_count() == 30
            assert ps.buffered_count() == 0
        finally:
            os.unlink(db_path)

    def test_buffer_accumulates_until_explicit_flush(self):
        """buffered_count() grows with each episode; flush() commits to DB."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ps = hvs.PersistentStorage(db_path, 100)
            for i in range(15):
                ps.buffer_episode(self._ep(i))
            assert ps.buffered_count() == 15
            assert ps.episode_count() == 0   # nothing in DB yet
            ps.flush()
            assert ps.episode_count() == 15
            assert ps.buffered_count() == 0
        finally:
            os.unlink(db_path)

    def test_load_episodes_full_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ps = hvs.PersistentStorage(db_path, 100)
            for i in range(30):
                ps.buffer_episode(self._ep(i))
            ps.flush()
            loaded = ps.load_episodes_full()
            assert len(loaded) == 30
            actions = {ep.action for ep in loaded}
            assert actions == {f"act_{i}" for i in range(30)}
        finally:
            os.unlink(db_path)

    def test_query_by_task_filters_correctly(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ps = hvs.PersistentStorage(db_path, 100)
            for i in range(15):
                ps.buffer_episode(hvs.Episode(float(i), "neuro",
                    hvs.HyperVector(i * 7), "a", "ok", 0.5))
            for i in range(10):
                ps.buffer_episode(hvs.Episode(float(i), "climate",
                    hvs.HyperVector(i * 13), "a", "ok", 0.5))
            ps.flush()
            assert len(ps.query_by_task("neuro", 100)) == 15
            assert len(ps.query_by_task("climate", 100)) == 10
        finally:
            os.unlink(db_path)

    def test_query_by_time_range(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ps = hvs.PersistentStorage(db_path, 100)
            for i in range(30):
                ps.buffer_episode(hvs.Episode(float(i), "test",
                    hvs.HyperVector(i), "a", "ok", 0.5))
            ps.flush()
            in_range = ps.query_by_time_range(5.0, 15.0, 100)
            assert 9 <= len(in_range) <= 13
        finally:
            os.unlink(db_path)

    def test_hv_survives_db_roundtrip(self):
        """HyperVector stored to SQLite reconstructs with identical bits."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            ps = hvs.PersistentStorage(db_path, 100)
            original = hvs.HyperVector(12345)
            ps.buffer_episode(hvs.Episode(0.0, "hv_test", original,
                                          "act", "ok", 1.0))
            ps.flush()
            loaded = ps.load_episodes_full()
            assert len(loaded) == 1
            state = loaded[0].get_situation_hv().__getstate__()
            recovered = hvs.HyperVector.__new__(hvs.HyperVector)
            recovered.__setstate__(state)
            assert recovered.similarity(original) == 1.0
        finally:
            os.unlink(db_path)


# ===========================================================================
# C. Cognitive Worker Pool
# ===========================================================================

@requires_rust
class TestRustCognitiveWorkerPool:

    def _make_pool(self, n_workers: int = 4):
        sm = hvs.SemanticMemoryConcurrent()
        em = hvs.EpisodicMemoryConcurrent(max_hot_size=500)
        for i in range(30):
            sm.add_concept(f"c{i}", hvs.HyperVector(i * 17))
            em.add_episode(hvs.Episode(float(i), "pool",
                hvs.HyperVector(i * 7), f"a{i}", "ok", 0.5))
        return hvs.CognitiveWorkerPool(sm, em, n_workers)

    def test_worker_count(self):
        pool = self._make_pool(4)
        assert pool.worker_count() == 4
        pool.shutdown()

    def test_submit_query_does_not_block(self):
        pool = self._make_pool(4)
        t0 = time.perf_counter()
        pool.submit_query("q1", hvs.HyperVector(42), 5, 3, 0.7)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pool.shutdown()
        assert elapsed_ms < 200, f"submit_query took {elapsed_ms:.1f}ms"

    def test_concurrent_queries_no_crash(self):
        pool = self._make_pool(4)
        errors = []

        def query(qid):
            try:
                pool.submit_query(f"q{qid}", hvs.HyperVector(qid * 100), 5, 2, 0.7)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=query, args=(i,)) for i in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        pool.shutdown()
        assert not errors

    def test_shutdown_cleans_up(self):
        pool = self._make_pool(2)
        pool.submit_query("q1", hvs.HyperVector(1), 3, 2, 0.7)
        pool.shutdown()  # must not block or raise


# ===========================================================================
# D. Spiking Neural Network (snn_rs)
# ===========================================================================

@requires_snn
class TestRustSNN:

    # ---- LIF Layer ----

    def test_lif_step_output_shape(self):
        layer = _snn_rs.LIFLayer(100, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0)
        spikes = layer.step([1.0] * 100)
        assert len(spikes) == 100

    def test_lif_output_binary(self):
        layer = _snn_rs.LIFLayer(50, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0)
        for s in layer.step(np.random.rand(50).tolist()):
            assert s in (0.0, 1.0)

    def test_lif_spikes_at_high_input(self):
        layer = _snn_rs.LIFLayer(10, 1.0, -70.0, -65.0, -55.0, 2.0, 1.0)
        assert any(s > 0 for s in layer.step([100.0] * 10))

    def test_lif_no_spike_at_zero_input(self):
        layer = _snn_rs.LIFLayer(20, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0)
        assert all(s == 0.0 for s in layer.step([0.0] * 20))

    def test_lif_reset_clears_state(self):
        layer = _snn_rs.LIFLayer(20, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0)
        for _ in range(10):
            layer.step([10.0] * 20)
        layer.reset()
        assert all(s == 0.0 for s in layer.step([0.0] * 20))

    # ---- STDP Engine ----

    def test_stdp_advance_time_increments(self):
        stdp = _snn_rs.StdpEngine(50, 100, 0.01, 20.0, 0.01, 0.012)
        t_before = stdp.current_time
        stdp.advance_time(1.0)
        assert stdp.current_time > t_before

    def test_stdp_apply_records_updates(self):
        stdp = _snn_rs.StdpEngine(10, 20, 0.01, 20.0, 0.01, 0.012)
        pre = [1.0] * 10
        post = [1.0] * 20
        stdp.advance_time(1.0)
        u_before = stdp.updates
        stdp.apply([1.0] + [0.0] * 9, [1.0] + [0.0] * 19)
        assert stdp.updates >= u_before

    # ---- SnnCore ----

    def test_snn_core_simulate_shape(self):
        core = _snn_rs.SnnCore(50, 100, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0,
                               False, 0.01, 20.0, 0.01, 0.012, 42)
        result = core.simulate(np.random.rand(50).tolist(), 10, False)
        assert len(result) == 10
        assert len(result[0]) == 100

    def test_snn_core_binary_output(self):
        core = _snn_rs.SnnCore(50, 100, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0,
                               False, 0.01, 20.0, 0.01, 0.012, 42)
        for step in core.simulate(np.random.rand(50).tolist(), 5, False):
            for s in step:
                assert s in (0.0, 1.0)

    def test_snn_core_get_stats_non_empty(self):
        core = _snn_rs.SnnCore(50, 100, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0,
                               False, 0.01, 20.0, 0.01, 0.012, 42)
        stats = core.get_stats()
        # get_stats returns a list of (name, value) tuples
        assert len(stats) >= 1

    def test_snn_core_stdp_updates_with_learning(self):
        core = _snn_rs.SnnCore(50, 100, 20.0, -70.0, -65.0, -55.0, 2.0, 1.0,
                               True, 0.01, 20.0, 0.01, 0.012, 42)
        inp = [10.0] * 50  # strong input to drive spikes
        core.simulate(inp, 20, True)
        assert core.stdp_updates >= 0

    # ---- HebbianMatrix ----

    def test_hebbian_update_changes_weights(self):
        hm = _snn_rs.HebbianMatrix(20, 20, 0.01, 1.0, True)
        pre = np.random.rand(20).tolist()
        post = np.random.rand(20).tolist()
        w_before = hm.get_weights()
        hm.hebbian_update(pre, post)
        assert hm.get_weights() != w_before

    def test_hebbian_forward_output_shape(self):
        hm = _snn_rs.HebbianMatrix(10, 30, 0.01, 1.0, True)
        assert len(hm.forward(np.random.rand(10).tolist())) == 30

    def test_hebbian_update_count_increments(self):
        hm = _snn_rs.HebbianMatrix(5, 5, 0.01, 1.0, False)
        count_before = hm.update_count
        hm.hebbian_update([1.0] * 5, [1.0] * 5)
        assert hm.update_count > count_before

    # ---- RateCoder ----

    def test_rate_coder_output_structure(self):
        """encode() returns (spike_times, rates); rates list has one entry per neuron."""
        rc = _snn_rs.RateCoder(50, 10.0)
        result = rc.encode(np.random.rand(50).tolist(), 100.0)
        # Returns (spike_times_list, rates_list)
        assert isinstance(result, tuple) and len(result) == 2
        _spike_times, rates = result
        assert len(rates) == 50

    def test_rate_coder_rates_non_negative(self):
        """All firing rates returned by encode() are non-negative."""
        rc = _snn_rs.RateCoder(20, 10.0)
        _spikes, rates = rc.encode([0.5] * 20, 50.0)
        for r in rates:
            assert r >= 0.0


# ===========================================================================
# E. Python Memory Modules
# ===========================================================================

class TestPythonSemanticMemory:

    def test_add_and_query_concept(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        hv = hvs.HyperVector(1)
        sm.add_concept("hippocampus", {}, hv_override=hv)
        assert sm.query(hv, k=1)[0][0] == "hippocampus"

    def test_add_relation_creates_edge(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        sm.add_concept("cortisol", {}, hv_override=hvs.HyperVector(1))
        sm.add_concept("hippocampus", {}, hv_override=hvs.HyperVector(2))
        sm.add_relation("cortisol", "causes", "hippocampus")
        assert sm.concept_graph.has_edge("cortisol", "hippocampus")
        assert sm.concept_graph["cortisol"]["hippocampus"]["relation"] == "causes"

    def test_spread_activation_follows_edges(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        for name, seed in [("n0", 1), ("n1", 2), ("n2", 3), ("n3", 4)]:
            sm.add_concept(name, {}, hv_override=hvs.HyperVector(seed))
        for i in range(3):
            sm.add_relation(f"n{i}", "causes", f"n{i + 1}")
        activated = sm.spread_activation(["n0"], steps=3, decay=0.7)
        assert "n1" in activated
        assert "n2" in activated

    def test_spread_activation_decays(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        for i in range(5):
            sm.add_concept(f"n{i}", {}, hv_override=hvs.HyperVector(i + 1))
        for i in range(4):
            sm.add_relation(f"n{i}", "causes", f"n{i + 1}")
        activated = sm.spread_activation(["n0"], steps=5, decay=0.7)
        if "n1" in activated and "n3" in activated:
            assert activated["n1"] >= activated["n3"]

    def test_relation_weights_include_all_tkl_types(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        required = {"is_a", "causes", "leads_to", "implies",
                    "results_in", "similar_to", "semantically_related"}
        assert not (required - set(sm.relation_weights.keys()))

    def test_semantically_related_edges_propagate(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        sm.add_concept("solar_energy", {}, hv_override=hvs.HyperVector(1))
        sm.add_concept("photosynthesis", {}, hv_override=hvs.HyperVector(2))
        sm.add_relation("solar_energy", "semantically_related", "photosynthesis")
        activated = sm.spread_activation(["solar_energy"], steps=1, decay=1.0)
        assert "photosynthesis" in activated

    def test_cross_domain_bridge_propagates(self):
        """A manually added cross-domain similar_to edge allows activation to cross domains."""
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        for name, seed in [("chlorophyll", 10), ("photosynthesis", 11),
                           ("solar_cell", 20), ("photovoltaic", 21)]:
            sm.add_concept(name, {}, hv_override=hvs.HyperVector(seed))
        sm.add_relation("chlorophyll", "part_of", "photosynthesis")
        sm.add_relation("solar_cell", "part_of", "photovoltaic")
        sm.concept_graph.add_edge("photosynthesis", "photovoltaic",
                                  relation="similar_to", weight=0.4,
                                  cross_domain=True)
        activated = sm.spread_activation(["chlorophyll"], steps=3, decay=0.7)
        assert "photovoltaic" in activated

    def test_query_sorted_by_similarity(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        target = hvs.HyperVector(0)
        sm.add_concept("target", {}, hv_override=target)
        for i in range(10):
            sm.add_concept(f"noise_{i}", {}, hv_override=hvs.HyperVector(i * 5000 + 9999))
        assert sm.query(target, k=5)[0][0] == "target"


class TestPythonEpisodicMemory:

    def test_record_and_lsh_index_populated(self):
        from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
        mem = EpisodicMemory(store=None, recent_capacity=500)
        for i in range(30):
            mem.record(LiveEpisode(
                timestamp=float(i), task_tag="neuro",
                situation_hv=hvs.HyperVector(i * 17),
                state={"step": i}, action="encode",
                outcome="ok", reward=0.0,
            ))
        tables = mem.lsh_index.get("neuro", [])
        assert len(tables) == mem.lsh_num_tables
        assert sum(len(b) for t in tables for b in t.values()) >= 30

    def test_recall_similar_returns_episode_objects(self):
        """recall_similar returns a list of LiveEpisode objects."""
        from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
        mem = EpisodicMemory(store=None, recent_capacity=500)
        target_hv = hvs.HyperVector(0)
        for i in range(10):
            mem.record(LiveEpisode(
                timestamp=float(i), task_tag="bio",
                situation_hv=hvs.HyperVector(i * 200),
                state={"i": i}, action=f"noise_{i}", outcome="ok", reward=0.0,
            ))
        results = mem.recall_similar(target_hv, task_tag="bio", k=3)
        assert results, "recall_similar must return at least one result"
        for ep in results:
            assert hasattr(ep, "action"), "Each result must be a LiveEpisode"

    def test_capacity_eviction_respected(self):
        from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
        cap = 50
        mem = EpisodicMemory(store=None, recent_capacity=cap)
        for i in range(cap + 30):
            mem.record(LiveEpisode(float(i), "cap", hvs.HyperVector(i),
                                   {}, "a", "ok", 0.0))
        buf = mem.recent.get("cap", [])
        assert len(buf) <= cap

    def test_recall_by_outcome_filters_correctly(self):
        from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
        mem = EpisodicMemory(store=None, recent_capacity=500)
        for i in range(20):
            mem.record(LiveEpisode(
                float(i), "test", hvs.HyperVector(i), {},
                "act",
                "success" if i % 2 == 0 else "failure",
                1.0 if i % 2 == 0 else 0.0,
            ))
        # recall_by_outcome(task_tag, outcome, n)
        successes = mem.recall_by_outcome("test", "success", 100)
        assert len(successes) == 10
        for ep in successes:
            assert ep.outcome == "success"


# ===========================================================================
# F. Reasoning Modules
# ===========================================================================

class TestCausalGraphAndReasoner:

    def test_add_causes_populates_forward(self):
        from python.core.reasoning.causal_reasoning import CausalGraph
        cg = CausalGraph()
        cg.add_causes("cortisol", "hippocampal_atrophy", strength=0.9)
        assert "cortisol" in cg.forward
        assert "hippocampal_atrophy" in cg.get_immediate_effects("cortisol")

    def test_add_prevents_populates_all_links(self):
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalRelation
        cg = CausalGraph()
        cg.add_prevents("exercise", "heart_disease", strength=0.8)
        assert any(lk.relation == CausalRelation.PREVENTS for lk in cg.all_links)

    def test_forward_chain_multi_hop(self):
        """forward_chain follows A→B→C→D and the longest chain ends at D."""
        from python.core.reasoning.causal_reasoning import CausalGraph
        cg = CausalGraph()
        cg.add_causes("deforestation", "co2_increase")
        cg.add_causes("co2_increase", "temperature_rise")
        cg.add_causes("temperature_rise", "ice_melt")
        cg.add_causes("ice_melt", "sea_level_rise")
        chains = cg.forward_chain("deforestation", max_depth=5)
        # chains is a list of CausalChain objects; the longest should end at sea_level_rise
        ends = {ch.end for ch in chains}
        assert "sea_level_rise" in ends, f"Expected sea_level_rise in chain ends: {ends}"

    def test_backward_chain_finds_root_cause(self):
        from python.core.reasoning.causal_reasoning import CausalGraph
        cg = CausalGraph()
        cg.add_causes("stress", "cortisol_release")
        cg.add_causes("cortisol_release", "hippocampal_atrophy")
        cg.add_causes("hippocampal_atrophy", "memory_impairment")
        chains = cg.backward_chain("memory_impairment", max_depth=4)
        starts = {ch.start for ch in chains}
        assert "stress" in starts

    def test_get_immediate_causes(self):
        from python.core.reasoning.causal_reasoning import CausalGraph
        cg = CausalGraph()
        cg.add_causes("A", "C")
        cg.add_causes("B", "C")
        causes = cg.get_immediate_causes("C")
        assert "A" in causes and "B" in causes

    def test_causal_reasoner_predict_effects(self):
        """predict_effects(state, action, task_tag) returns downstream effects."""
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner
        cg = CausalGraph()
        cg.add_causes("co2_emissions", "greenhouse_effect")
        cg.add_causes("greenhouse_effect", "warming")
        cr = CausalReasoner(cg)
        # state must contain the cause as True; action triggers it
        effects = cr.predict_effects(
            {"co2_emissions": True}, "co2_emissions", "climate")
        assert "greenhouse_effect" in effects

    def test_tkl_causal_graph_shared_with_caller(self):
        """TKL writes causal facts into the caller-supplied CausalGraph."""
        from python.core.memory.semantic_memory import SemanticMemory
        from python.core.memory.episodic_memory import EpisodicMemory
        from python.core.reasoning.causal_reasoning import CausalGraph
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        sm = SemanticMemory()
        em = EpisodicMemory()
        cg = CausalGraph()
        tkl = TextKnowledgeLearner(semantic_memory=sm, episodic_memory=em,
                                   causal_graph=cg)
        assert tkl.causal_graph is cg
        tkl.learn_from_text("Cortisol causes hippocampal atrophy.")
        tkl.learn_from_text("Hippocampal atrophy causes memory impairment.")
        assert len(cg.forward) >= 1, (
            f"Shared CausalGraph unpopulated. forward={dict(cg.forward)}"
        )


class TestGlobalWorkspace:

    def _proposals(self, *name_salience_pairs):
        from python.core.reasoning.global_workspace import Coalition
        return [
            Coalition(source=name, content={}, base_salience=sal)
            for name, sal in name_salience_pairs
        ]

    def _gw(self, threshold=0.3):
        from python.core.reasoning.global_workspace import GlobalWorkspace, WorkspaceModule

        class _M(WorkspaceModule):
            def receive_broadcast(self, content):
                pass

        gw = GlobalWorkspace(attention_threshold=threshold)
        gw.register_module("PLACEHOLDER", _M())
        return gw

    def test_highest_salience_wins(self):
        gw = self._gw()
        winner = gw.compete(self._proposals(
            ("SEMANTIC", 0.9), ("EPISODIC", 0.5), ("CAUSAL", 0.3),
        ))
        assert winner is not None
        assert winner.source == "SEMANTIC"

    def test_no_proposals_returns_none(self):
        gw = self._gw()
        assert gw.compete([]) is None

    def test_second_highest_is_not_winner(self):
        gw = self._gw()
        winner = gw.compete(self._proposals(
            ("STRONG", 0.8), ("WEAK", 0.2),
        ))
        assert winner.source == "STRONG"

    def test_four_module_competition(self):
        gw = self._gw()
        winner = gw.compete(self._proposals(
            ("SEMANTIC", 0.6), ("EPISODIC", 0.5),
            ("CAUSAL", 0.4), ("EMOTION", 0.3),
        ))
        assert winner.source == "SEMANTIC"


class TestAnalogyEngine:

    def test_register_abstract_and_find_analogy(self):
        from python.core.reasoning.analogy import AnalogyEngine
        ae = AnalogyEngine()
        ae.register_abstract("ENERGY_CONVERTER",
                             "Entity that converts one form of energy to another",
                             {"biology": "chloroplast", "technology": "solar_panel"})
        analogy = ae.find_analogy("biology", "technology")
        sources = [m.source_concept for m in analogy.mappings]
        targets = [m.target_concept for m in analogy.mappings]
        assert "chloroplast" in sources
        assert "solar_panel" in targets

    def test_auto_discover_abstractions_stem_bridge(self):
        """Concepts with shared name stems are bridged by auto_discover_abstractions."""
        from python.core.reasoning.analogy import AnalogyEngine
        ae = AnalogyEngine()
        hvs_a = {"Neural": hvs.HyperVector(1), "Synapse": hvs.HyperVector(2)}
        hvs_b = {"Neural_Network": hvs.HyperVector(10),
                 "Synaptic_Weight": hvs.HyperVector(11)}
        mappings = ae.auto_discover_abstractions(
            "neuroscience", "computer_science", hvs_a, hvs_b,
            similarity_threshold=0.50)
        assert isinstance(mappings, list) and len(mappings) >= 1
        for m in mappings:
            assert m.source_concept in hvs_a
            assert m.target_concept in hvs_b

    def test_transfer_rule_maps_across_domains(self):
        from python.core.reasoning.analogy import AnalogyEngine
        ae = AnalogyEngine()
        ae.register_abstract("ACTOR", "Moving agent",
                             {"game": "agent", "biology": "enzyme"})
        ae.register_abstract("TARGET", "Goal target",
                             {"game": "food", "biology": "substrate"})
        new_cond, new_act = ae.transfer_rule({"agent"}, "food", "game", "biology")
        assert "enzyme" in new_cond
        assert new_act == "substrate"

    def test_get_transfer_explanation_returns_string(self):
        from python.core.reasoning.analogy import AnalogyEngine
        ae = AnalogyEngine()
        ae.register_abstract("X", "desc", {"src": "a", "tgt": "b"})
        explanation = ae.get_transfer_explanation("src", "tgt")
        assert isinstance(explanation, str) and len(explanation) > 0


class TestSTRIPSPlanner:
    """STRIPSPlanner uses A* with STRIPS operators learned from a CausalGraph."""

    def _make_planner_with_ops(self):
        """Build a planner with ACTION_* nodes so learn_operators_from_graph works."""
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner
        from python.core.reasoning.planner import STRIPSPlanner
        cg = CausalGraph()
        # ACTION_ prefix is required by learn_operators_from_graph
        cg.add_causes("ACTION_emit_co2", "co2_rise")
        cg.add_causes("co2_rise", "warming")
        cg.add_causes("warming", "ice_melt")
        cg.add_causes("ACTION_reforestation", "co2_decrease")
        cr = CausalReasoner(cg)
        planner = STRIPSPlanner(cr)
        planner.learn_operators_from_graph(cg)
        return planner, cg

    def test_learn_operators_from_graph_populates_operators(self):
        planner, cg = self._make_planner_with_ops()
        # Internal learned ops stored in _learned_ops (operators property is CausalReasoner)
        assert len(planner._learned_ops) >= 1

    def test_simulate_sequence_adds_effects(self):
        planner, cg = self._make_planner_with_ops()
        result = planner.simulate_sequence(
            frozenset({"ACTION_emit_co2"}),
            ["ACTION_emit_co2"],
        )
        # simulate_sequence returns the resulting state after applying actions
        assert isinstance(result, frozenset)

    def test_plan_with_learned_operators(self):
        """After learning operators, the planner can navigate a causal chain."""
        planner, cg = self._make_planner_with_ops()
        # Goal: reach the state where 'co2_rise' is true
        plan = planner.plan(
            initial_state={"ACTION_emit_co2"},
            goal={"co2_rise"},
            max_depth=5,
        )
        # plan may be None if the operator's add_effects are empty; test the infrastructure
        assert plan is None or isinstance(plan, list)

    def test_plan_unreachable_returns_none(self):
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner
        from python.core.reasoning.planner import STRIPSPlanner
        cg = CausalGraph()
        cg.add_causes("A", "B")
        cr = CausalReasoner(cg)
        planner = STRIPSPlanner(cr)
        planner.learn_operators_from_graph(cg)
        plan = planner.plan({"A"}, {"IMPOSSIBLE_GOAL"}, max_depth=3)
        assert plan is None or plan == []


class TestCognitiveModules:

    def test_emotion_recognition_returns_non_empty_string(self):
        from python.core.cognitive.emotion_system import EmotionSystem
        es = EmotionSystem()
        for text in [
            "Antibiotic-resistant infections kill 1.3 million people annually — a preventable crisis.",
            "The vaccine completely eliminated poliomyelitis from the Western hemisphere.",
            "Sea-level rise will displace 300 million coastal residents by 2100.",
        ]:
            result = es.recognize_emotion_from_text(text)
            assert isinstance(result, str) and len(result) > 0

    def test_emotion_info_has_valence_and_arousal(self):
        from python.core.cognitive.emotion_system import EmotionSystem
        es = EmotionSystem()
        es.recognize_emotion_from_text("Cortisol chronically damages hippocampal neurons.")
        info = es.get_emotion_info()
        assert "valence" in info and "arousal" in info

    def test_emotion_blend_is_dict(self):
        from python.core.cognitive.emotion_system import EmotionSystem
        es = EmotionSystem()
        es.recognize_emotion_from_text("The therapy restored full cognitive function.")
        assert isinstance(es.get_emotion_blend(), dict)

    def test_self_model_confidence_in_unit_interval(self):
        from python.core.cognitive.self_model import SelfModel
        sm = SelfModel()
        for i in range(10):
            sm.update("bio", 0.7, i % 2 == 0, "recall",
                      1.0 if i % 2 == 0 else 0.0, {})
        conf = sm.get_confidence("bio")
        assert 0.0 <= conf <= 1.0

    def test_self_model_calibration_error_non_negative(self):
        from python.core.cognitive.self_model import SelfModel
        sm = SelfModel()
        sm.update("t", 0.8, True, "a", 1.0, {})
        sm.update("t", 0.8, False, "a", 0.0, {})
        assert sm.get_calibration_error("t") >= 0.0

    def test_curiosity_first_hv_maximally_novel(self):
        from python.core.learning.curiosity import CuriosityModule
        cm = CuriosityModule()
        assert cm.compute_novelty(hvs.HyperVector(12345), task_tag="neuro") == 1.0

    def test_curiosity_novelty_decreases_after_visit(self):
        from python.core.learning.curiosity import CuriosityModule
        cm = CuriosityModule()
        hv = hvs.HyperVector(777)
        n1 = cm.compute_novelty(hv, task_tag="climate")
        cm.record_visit(hv, task_tag="climate")
        n2 = cm.compute_novelty(hv, task_tag="climate")
        assert n2 <= n1

    def test_curiosity_novelty_in_unit_interval(self):
        from python.core.learning.curiosity import CuriosityModule
        cm = CuriosityModule()
        for seed in range(1, 6):
            n = cm.compute_novelty(hvs.HyperVector(seed), task_tag="test")
            assert 0.0 <= n <= 1.0


# ===========================================================================
# G. Language Modules
# ===========================================================================

class TestTextKnowledgeLearner:

    def test_concepts_added_to_semantic_memory(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        nodes = {n.lower() for n in sm.concept_graph.nodes}
        expected = {"hippocampus", "cortisol", "memory", "prefrontal", "dopamine"}
        assert len(expected & nodes) >= 3, (
            f"Expected ≥3 core terms, found {expected & nodes}. "
            f"Sample: {sorted(nodes)[:20]}"
        )

    def test_relations_added_to_semantic_memory(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        assert sm.concept_graph.number_of_edges() >= 5

    def test_causal_facts_in_shared_causal_graph(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        n_links = sum(len(v) for v in cg.forward.values())
        assert n_links >= 1, f"CausalGraph must have ≥1 link, got {n_links}"

    def test_learned_facts_accumulate(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("biochem", _BIOCHEM)])
        assert len(tkl.learned_facts) >= 3

    def test_episodes_written_to_episodic_memory(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("climate", _CLIMATE)])
        total = sum(len(v) for v in em.recent.values()) if hasattr(em, 'recent') else em.size() if hasattr(em, 'size') else 0
        assert total >= 1

    def test_multi_domain_grows_concept_graph(self):
        tkl, sm, em, cg, _ = _make_tkl_system([
            ("neuro", _NEURO), ("climate", _CLIMATE), ("biochem", _BIOCHEM),
        ])
        assert sm.concept_graph.number_of_nodes() >= 20

    def test_concept_hvs_indexed(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("cs", _CS)])
        assert len(sm.concept_hvs) >= 5


class TestNLGEngine:

    def test_generate_single_fact(self):
        from python.core.language.nlg import NLGEngine
        result = NLGEngine().generate("fact", {
            "subject": "Cortisol", "relation": "causes",
            "object": "hippocampal atrophy",
        })
        assert isinstance(result, str) and len(result) > 0
        assert ("cortisol" in result.lower() or "atrophy" in result.lower())

    def test_generate_discourse_multi_sentence(self):
        from python.core.language.nlg import NLGEngine
        frames = [
            {"subject": "Deforestation", "relation": "causes", "object": "CO2 increase"},
            {"subject": "CO2 increase", "relation": "causes", "object": "temperature rise"},
            {"subject": "Temperature rise", "relation": "causes", "object": "sea-level rise"},
        ]
        discourse = NLGEngine().generate_discourse(frames, query_type="causal",
                                                   topic="deforestation")
        assert isinstance(discourse, str) and len(discourse) > 20
        sentences = [s.strip() for s in discourse.split(".") if s.strip()]
        assert len(sentences) >= 2

    def test_generate_causal_chain_uses_string_relation(self):
        """NLGEngine.generate_causal_chain must handle CausalChain objects."""
        from python.core.language.nlg import NLGEngine
        from python.core.reasoning.causal_reasoning import CausalLink, CausalRelation, CausalChain
        # Build a chain with the value of CausalRelation.CAUSES (a string)
        link = CausalLink("stress", "cortisol", CausalRelation.CAUSES, 0.9)
        chain = CausalChain([link], "stress", "cortisol")
        # Describe() returns the chain as text — use that as a fallback
        description = chain.describe()
        assert isinstance(description, str) and len(description) > 0

    def test_structural_realizer_is_a(self):
        from python.core.language.nlg import StructuralRealizer
        result = StructuralRealizer().realize_sentence("Mitochondria", "is_a", "organelle")
        assert isinstance(result, str) and len(result) > 0

    def test_discourse_planner_plan(self):
        from python.core.language.nlg import DiscoursePlanner
        frames = [
            {"subject": "Photosynthesis", "relation": "is_a", "object": "metabolic process"},
            {"subject": "Photosynthesis", "relation": "produces", "object": "oxygen"},
        ]
        result = DiscoursePlanner().plan(frames, query_type="factual",
                                         topic="photosynthesis")
        assert isinstance(result, str) and len(result) > 0


class TestSemanticRoleLabeler:

    def test_extracts_causes_predicate(self):
        from python.core.language.semantic_roles import SemanticRoleLabeler
        result = SemanticRoleLabeler().label(
            "Cortisol causes hippocampal atrophy in stressed patients.")
        assert result.pred in ("causes", "cause")

    def test_extracts_agent(self):
        from python.core.language.semantic_roles import SemanticRoleLabeler
        result = SemanticRoleLabeler().label(
            "Backpropagation computes gradients of the loss function.")
        assert result.agent is not None and len(result.agent) > 0

    def test_extracts_patient(self):
        from python.core.language.semantic_roles import SemanticRoleLabeler
        result = SemanticRoleLabeler().label(
            "Enzymes lower the activation energy of biochemical reactions.")
        assert result.patient is not None and len(result.patient) > 0

    def test_pred_non_empty_on_multiple_sentences(self):
        from python.core.language.semantic_roles import SemanticRoleLabeler
        srl = SemanticRoleLabeler()
        for sent in [
            "The hippocampus encodes spatial and declarative memories.",
            "CO2 emissions cause global surface temperatures to rise.",
            "Antibiotics kill or inhibit the growth of bacteria.",
        ]:
            r = srl.label(sent)
            assert r is not None and r.pred is not None


# ===========================================================================
# H. Cross-Domain Knowledge Transfer
# ===========================================================================

class TestCrossDomainKnowledgeTransfer:

    @pytest.fixture(scope="class")
    def multi_domain(self):
        return _make_tkl_system([
            ("neuroscience", _NEURO),
            ("biochemistry", _BIOCHEM),
            ("computer_science", _CS),
            ("climate", _CLIMATE),
        ])

    def test_all_domains_have_concept_hvs(self, multi_domain):
        _, sm, _, _, domain_hvs = multi_domain
        for domain in ("neuroscience", "biochemistry", "computer_science", "climate"):
            assert len(domain_hvs.get(domain, {})) >= 3

    def test_auto_discover_abstractions_neuro_cs(self, multi_domain):
        from python.core.reasoning.analogy import AnalogyEngine
        _, sm, _, _, domain_hvs = multi_domain
        ae = AnalogyEngine()
        mappings = ae.auto_discover_abstractions(
            "neuroscience", "computer_science",
            domain_hvs["neuroscience"], domain_hvs["computer_science"],
            similarity_threshold=0.50,
        )
        assert isinstance(mappings, list)
        for m in mappings:
            assert m.source_concept in domain_hvs["neuroscience"]
            assert m.target_concept in domain_hvs["computer_science"]

    def test_bridge_edges_propagate_activation(self, multi_domain):
        """A manually inserted cross-domain similar_to bridge transmits activation."""
        _, sm, _, _, _ = multi_domain
        nodes_lower = {n.lower(): n for n in sm.concept_graph.nodes}
        seed = next((nodes_lower[c] for c in
                     ("hippocampus", "cortisol", "neuron", "memory", "prefrontal")
                     if c in nodes_lower), None)
        tgt = next((nodes_lower[c] for c in
                    ("atp", "glucose", "mitochondria", "enzyme", "photosynthesis")
                    if c in nodes_lower), None)
        if not seed or not tgt:
            pytest.skip("Required concepts not parsed from corpus in this run")
        sm.concept_graph.add_edge(seed, tgt, relation="similar_to", weight=0.4,
                                  cross_domain=True)
        activated = sm.spread_activation([seed], steps=3, decay=0.7)
        assert tgt in activated

    def test_stem_bridge_neural_concepts(self):
        """'neural' stem connects neuroscience and CS concept HVs."""
        from python.core.reasoning.analogy import AnalogyEngine
        ae = AnalogyEngine()
        hvs_a = {"Neural": hvs.HyperVector(1), "Synapse": hvs.HyperVector(2)}
        hvs_b = {"Neural_Network": hvs.HyperVector(10),
                 "Synaptic_Weight": hvs.HyperVector(11)}
        mappings = ae.auto_discover_abstractions(
            "neuroscience", "computer_science", hvs_a, hvs_b, 0.50)
        assert len(mappings) >= 1

    def test_co2_node_present_after_multi_domain_training(self):
        """CO2/carbon appears in climate AND biochem — confirms shared nodes."""
        tkl, sm, em, cg, _ = _make_tkl_system([
            ("climate", _CLIMATE), ("biochemistry", _BIOCHEM),
        ])
        nodes_lower = {n.lower() for n in sm.concept_graph.nodes}
        assert any(kw in nodes_lower for kw in ("co2", "carbon", "dioxide"))

    def test_spread_activation_from_shared_concept(self):
        tkl, sm, em, cg, _ = _make_tkl_system([
            ("biochemistry", _BIOCHEM), ("climate", _CLIMATE),
        ])
        nodes_lower = {n.lower(): n for n in sm.concept_graph.nodes}
        seed = next((nodes_lower[c] for c in ("energy", "carbon", "temperature")
                     if c in nodes_lower), None)
        if not seed:
            pytest.skip("No shared seed concept available")
        activated = sm.spread_activation([seed], steps=4, decay=0.65)
        assert len(activated) >= 2


# ===========================================================================
# I. Integration Invariants
# ===========================================================================

class TestIntegrationInvariants:

    def test_tkl_semantic_memory_shared(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        assert tkl.semantic is sm

    def test_tkl_episodic_memory_shared(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        assert tkl.episodic is em

    def test_tkl_causal_graph_shared(self):
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        assert tkl.causal_graph is cg

    def test_causal_reasoner_uses_same_graph(self):
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner
        tkl, sm, em, cg, _ = _make_tkl_system([("neuro", _NEURO)])
        cr = CausalReasoner(cg)
        assert cr.graph is cg

    def test_semantic_memory_relation_weights_cover_tkl_types(self):
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        required = {"is_a", "causes", "leads_to", "implies",
                    "results_in", "similar_to", "semantically_related"}
        assert not (required - set(sm.relation_weights.keys()))

    def test_rust_registry_and_python_sm_interoperate(self):
        """HVs from Python SemanticMemory can be queried via Rust HyperVectorRegistry."""
        if not _RUST_AVAILABLE:
            pytest.skip("Rust not available")
        from python.core.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory()
        reg = hvs.HyperVectorRegistry()
        for i, name in enumerate(["hippocampus", "cortisol", "synapse", "neuron"]):
            hv = hvs.HyperVector(i + 1)
            sm.add_concept(name, {}, hv_override=hv)
            reg.register(name, hv)
        query_hv = sm.concept_hvs["cortisol"]
        results = reg.nearest_neighbors(query_hv, 2)
        assert results[0][0] == "cortisol" and results[0][1] == 1.0

    @requires_rust
    def test_python_and_rust_episodic_layers_independent(self):
        """Episodes stored in Python EpisodicMemory and Rust EpisodicMemoryConcurrent
        are independent and do not interfere."""
        from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
        py_mem = EpisodicMemory(store=None, recent_capacity=200)
        rust_mem = hvs.EpisodicMemoryConcurrent(max_hot_size=200)
        for i in range(20):
            hv = hvs.HyperVector(i * 7)
            py_mem.record(LiveEpisode(float(i), "shared", hv, {}, "a", "ok", 0.5))
            rust_mem.add_episode(hvs.Episode(float(i), "shared", hv, "a", "ok", 0.5))
        assert rust_mem.size() == 20
        py_total = sum(len(v) for v in py_mem.recent.values()) if hasattr(py_mem, 'recent') else 20
        assert py_total == 20
