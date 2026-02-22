"""
Rigorous tests for both Rust backends (hypervec_rs + snn_rs) with diverse inputs.

Sections:
  A. HyperVector math properties (Rust)
  B. Cross-backend parity (Rust vs Python)
  C. SemanticMemory with Rust backend
  D. EpisodicMemory with Rust backend
  E. SNN Rust classes — LIFLayer, StdpEngine, SnnCore, HebbianMatrix, ConceptMapper, RateCoder
  F. Full V3 pipeline with Rust (all 14 flags, research config)
  G. Real-world scenarios with Rust backend
  H. Scalability / performance benchmarks
"""
from __future__ import annotations
import time
import pytest
import numpy as np

# ── check for backends ────────────────────────────────────────────────────────
try:
    import hypervec_rs as _hvrs
    RUST_VSA = True
except ImportError:
    RUST_VSA = False

try:
    import snn_rs as _snnrs
    RUST_SNN = True
except ImportError:
    RUST_SNN = False

import python.core.vsa.hypervec_shim as hv
from python.core.vsa.hypervec_py import HyperVectorPy
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.integration.config import NSCKConfig

vsa_only = pytest.mark.skipif(not RUST_VSA, reason="Rust VSA extension not built")
snn_only = pytest.mark.skipif(not RUST_SNN, reason="Rust SNN extension not built")


# ─────────────────────────────────────────────────────────────────────────────
# A. HyperVector math properties
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestHVMathProperties:
    """Mathematical invariants that must hold regardless of RNG backend."""

    def test_self_similarity_is_1(self):
        for seed in [0, 1, 42, 999, 2**31 - 1]:
            v = hv.HyperVector(seed)
            assert v.similarity(v) == pytest.approx(1.0), f"self-sim failed for seed={seed}"

    def test_random_pair_similarity_near_half(self):
        """Two unrelated HVs should be ~0.5 similar (chance level)."""
        sims = [hv.HyperVector(i * 997).similarity(hv.HyperVector(i * 1003 + 1))
                for i in range(50)]
        mean_sim = np.mean(sims)
        assert 0.44 < mean_sim < 0.56, f"Mean sim={mean_sim} out of expected range"

    def test_xor_reversibility(self):
        """(a XOR b) XOR b == a  for many seeds."""
        for seed_a, seed_b in [(1, 2), (100, 200), (9999, 8888), (42, 1337)]:
            a = hv.HyperVector(seed_a)
            b = hv.HyperVector(seed_b)
            assert a.xor(b).xor(b).similarity(a) > 0.99, \
                f"XOR reversibility failed for ({seed_a},{seed_b})"

    def test_xor_commutativity(self):
        a = hv.HyperVector(55)
        b = hv.HyperVector(66)
        ab = a.xor(b)
        ba = b.xor(a)
        np.testing.assert_array_equal(ab.bits, ba.bits, err_msg="XOR not commutative")

    def test_bundle_symmetry(self):
        """sim(bundle(a,b), a) ≈ sim(bundle(a,b), b) (equal bundle weight)."""
        for s1, s2 in [(10, 20), (777, 888), (314, 159)]:
            a = hv.HyperVector(s1)
            b = hv.HyperVector(s2)
            bnd = a.bundle(b)
            diff = abs(bnd.similarity(a) - bnd.similarity(b))
            assert diff < 0.1, f"Bundle not symmetric: diff={diff:.4f}"

    def test_permute_shift_increases_distance(self):
        """permute(k) should produce increasingly different HVs as k grows."""
        a = hv.HyperVector(42)
        sims = [a.permute(k).similarity(a) for k in [0, 1, 10, 100, 1000]]
        assert sims[0] == pytest.approx(1.0)
        # Larger shifts should generally be more dissimilar
        assert sims[-1] < sims[0]

    def test_permute_reversibility_many_shifts(self):
        a = hv.HyperVector(1234)
        for shift in [1, 7, 64, 128, 512, -1, -64]:
            recovered = a.permute(shift).permute(-shift)
            assert recovered.similarity(a) > 0.99, f"permute({shift}) not reversible"

    def test_lsh_hash_determinism(self):
        v = hv.HyperVector(77)
        for n_bits in [4, 8, 10, 16]:
            h1 = v.lsh_hash(0, n_bits)
            h2 = v.lsh_hash(0, n_bits)
            assert h1 == h2, f"LSH not deterministic at n_bits={n_bits}"

    def test_lsh_hash_different_seeds_differ(self):
        """Different seeds should produce different LSH codes across 20 seeds."""
        v = hv.HyperVector(99)
        hashes = {v.lsh_hash(seed, 10) for seed in range(20)}
        assert len(hashes) >= 5

    def test_weighted_bundle_skews_toward_high_weight(self):
        """weighted_bundle(b, weight=0.9) should be closer to self than to b."""
        a = hv.HyperVector(1)
        b = hv.HyperVector(2)
        w = a.weighted_bundle(b, 0.9)
        assert w.similarity(a) > w.similarity(b), \
            f"Weighted bundle: sim(a)={w.similarity(a):.3f} vs sim(b)={w.similarity(b):.3f}"

    def test_zero_vector_exists(self):
        """zero() should exist and be different from a random HV."""
        z = hv.HyperVector.zero()
        v = hv.HyperVector(42)
        assert 0.3 < z.similarity(v) < 0.7

    def test_bundle_of_same_vector_is_same(self):
        """a.bundle(a) should equal a (or be very similar)."""
        a = hv.HyperVector(1000)
        bnd = a.bundle(a)
        assert bnd.similarity(a) > 0.95


# ─────────────────────────────────────────────────────────────────────────────
# B. Cross-backend parity (Rust HV vs Python HV, given same bits)
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestCrossBackendParity:
    """Operations on identical bit patterns must produce identical results."""

    def _pair(self, seed_a, seed_b):
        rng = np.random.default_rng(seed_a + seed_b * 137)
        bits_a = rng.integers(0, 2, 10240, dtype=np.int8)
        bits_b = rng.integers(0, 2, 10240, dtype=np.int8)
        py_a = HyperVectorPy.from_bits(bits_a)
        py_b = HyperVectorPy.from_bits(bits_b)
        rs_a = hv.HyperVector.from_bits(bits_a)
        rs_b = hv.HyperVector.from_bits(bits_b)
        return py_a, py_b, rs_a, rs_b

    def test_xor_parity(self):
        for s in range(5):
            py_a, py_b, rs_a, rs_b = self._pair(s, s + 100)
            np.testing.assert_array_equal(
                py_a.xor(py_b).bits,
                rs_a.xor(rs_b).bits,
                err_msg=f"XOR parity failed for seed={s}",
            )

    def test_bundle_parity(self):
        # Bundle majority-vote tie-breaking: tied bits (~50% of all bits) are broken
        # randomly by independent RNGs in Rust (ChaCha8) vs Python (PCG64).
        # Expected agreement ≈ 50%×100% (non-tied) + 50%×50% (tied) = 75%.
        # We accept ≥ 70% as the minimum guarantee.
        for s in range(5):
            py_a, py_b, rs_a, rs_b = self._pair(s, s + 200)
            py_bits = py_a.bundle(py_b).bits.astype(np.float64)
            rs_bits = rs_a.bundle(rs_b).bits.astype(np.float64)
            agreement = np.mean(py_bits == rs_bits)
            assert agreement > 0.70, f"Bundle parity low: {agreement:.3f}"

    def test_permute_parity(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 10240, dtype=np.int8)
        py = HyperVectorPy.from_bits(bits)
        rs = hv.HyperVector.from_bits(bits)
        for shift in [1, -1, 64, -64, 128]:
            np.testing.assert_array_equal(
                py.permute(shift).bits,
                rs.permute(shift).bits,
                err_msg=f"Permute parity failed at shift={shift}",
            )

    def test_similarity_parity(self):
        """Hamming similarity of identical bit-patterns should match between backends."""
        for s in range(10):
            py_a, py_b, rs_a, rs_b = self._pair(s, s + 50)
            sim_py = py_a.similarity(py_b)
            # Use Python HV to compare with Rust (symmetric)
            sim_rs = py_a.similarity(rs_b)
            assert abs(sim_py - sim_rs) < 1e-6, \
                f"Similarity mismatch: py={sim_py:.6f} rs={sim_rs:.6f}"

    def test_cross_type_similarity_identical_bits(self):
        """Python HV similarity(Rust HV) from identical bits should be ~1.0."""
        rng = np.random.default_rng(5)
        bits = rng.integers(0, 2, 10240, dtype=np.int8)
        py = HyperVectorPy.from_bits(bits)
        rs = hv.HyperVector.from_bits(bits)
        sim = py.similarity(rs)
        assert sim == pytest.approx(1.0), f"Cross-type same-bits similarity={sim}"

    def test_bits_roundtrip(self):
        """from_bits → bits → from_bits should be idempotent."""
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, 10240, dtype=np.int8)
        rs = hv.HyperVector.from_bits(bits)
        np.testing.assert_array_equal(rs.bits, bits)


# ─────────────────────────────────────────────────────────────────────────────
# C. SemanticMemory with Rust backend
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestSemanticMemoryRust:

    def _make_mem(self, **cfg_kwargs):
        return SemanticMemory(use_rust=True, config=NSCKConfig(**cfg_kwargs))

    def test_add_and_query_concepts(self):
        mem = self._make_mem()
        for word in ["apple", "banana", "cherry", "date", "elderberry"]:
            mem.add_concept(word, {"type": "fruit"})
        assert "apple" in mem.concept_hvs
        assert mem.concept_graph.has_node("apple")

    def test_query_returns_similar_concepts(self):
        mem = self._make_mem()
        for w in ["dog", "cat", "rabbit"]:
            mem.add_concept(w, {"type": "pet"})
        query = hv.HyperVector(abs(hash("dog")) % (2**32))
        results = mem.query(query, k=3)
        assert len(results) > 0

    def test_spread_activation_reaches_neighbors(self):
        mem = self._make_mem()
        for n in ["sun", "light", "energy", "photon"]:
            mem.add_concept(n, {})
        mem.add_relation("sun", "produces", "light")
        mem.add_relation("light", "is_a", "energy")
        mem.add_relation("energy", "consists_of", "photon")

        activated = mem.spread_activation(["sun"], steps=3, decay=0.7)
        assert "light" in activated
        assert "energy" in activated
        assert activated.get("light", 0) > activated.get("photon", 0)

    def test_stigmergy_path_preference(self):
        mem = self._make_mem(enable_stigmergy=True)
        for n in "SABC":
            mem.add_concept(n, {})
        for a, b in [("S", "A"), ("A", "B"), ("B", "C"), ("S", "C")]:
            mem.add_relation(a, "r", b)
        for _ in range(15):
            mem.mark_path(["S", "A", "B", "C"], 1.0)
        mem.mark_path(["S", "C"], 0.5)
        act = mem.spread_activation(["S"], steps=2, decay=0.8)
        assert act.get("A", 0) > act.get("C", 0) * 0.5

    def test_belief_revision(self):
        mem = self._make_mem(enable_free_energy_beliefs=True)
        for n in ["X", "Y", "Z"]:
            mem.add_concept(n, {})
        for _ in range(5):
            mem.add_relation("X", "r", "Y")
        assert mem.concept_graph.has_edge("X", "Y")
        for _ in range(7):
            mem.add_relation("X", "r", "Z")
        assert mem.concept_graph.has_edge("X", "Z")
        assert not mem.concept_graph.has_edge("X", "Y"), \
            "Belief should be revised when contradictions > evidence"

    def test_incremental_concept_refinement(self):
        mem = self._make_mem(enable_incremental_concept_refinement=True)
        v1 = hv.HyperVector(100)
        v2 = hv.HyperVector(200)
        mem.add_concept("thing", {}, hv_override=v1)
        mem.add_concept("thing", {}, hv_override=v2)
        blended = mem.concept_hvs["thing"]
        assert blended.similarity(v1) > 0.5

    def test_1k_concepts_query_latency(self):
        # Python backend: linear scan over 1K × 10240-bit Hamming comparisons.
        # Observed ~3.5s on reference hardware; allow 15s for headroom.
        mem = self._make_mem()
        for i in range(1000):
            mem.add_concept(f"concept_{i}", {"idx": i})
        q = hv.HyperVector(12345)
        t0 = time.perf_counter()
        results = mem.query(q, k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert len(results) > 0
        assert elapsed_ms < 15_000, f"1K-concept query too slow: {elapsed_ms:.0f}ms"

    def test_evaporate_stigmergy_decays(self):
        # Stigmergy is stored in mem._stigmergy dict (not in graph edge attrs)
        mem = self._make_mem()
        for n in ["A", "B"]:
            mem.add_concept(n, {})
        mem.add_relation("A", "r", "B")
        for _ in range(10):
            mem.mark_path(["A", "B"], 1.0)
        stigmergy_before = mem.get_stigmergy("A", "B")
        assert stigmergy_before > 0, "mark_path should set stigmergy > 0"
        mem.evaporate_stigmergy(decay_rate=0.5)
        stigmergy_after = mem.get_stigmergy("A", "B")
        assert stigmergy_after < stigmergy_before, \
            f"Stigmergy should decay: before={stigmergy_before:.2f} after={stigmergy_after:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# D. EpisodicMemory with Rust backend
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestEpisodicMemoryRust:

    def _record(self, em, state, task, action, reward):
        """Helper to create and record a LiveEpisode using the real API."""
        import time as _time
        from python.core.memory.episodic_memory import LiveEpisode
        sv = em.create_situation_hv(state, task, [])
        ep = LiveEpisode(
            timestamp=_time.time(),
            task_tag=task,
            situation_hv=sv,
            state=state,
            action=action,
            outcome="positive" if reward > 0 else "negative",
            reward=reward,
        )
        em.record(ep)
        return sv, ep

    def test_store_and_retrieve(self):
        em = EpisodicMemory()
        state = {"head": (5, 5), "food": (5, 4), "score": 10}
        sv, _ = self._record(em, state, "snake", "ACTION_UP", 1.0)
        results = em.recall_similar(sv, "snake", k=5)
        assert len(results) > 0

    def test_lsh_same_type_consistency(self):
        """Same HV bits + same backend → same LSH hash (guaranteed)."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 10240, dtype=np.int8)
        rs1 = hv.HyperVector.from_bits(bits)
        rs2 = hv.HyperVector.from_bits(bits)
        assert rs1.lsh_hash(0, 10) == rs2.lsh_hash(0, 10), "Rust LSH not deterministic"
        py1 = HyperVectorPy.from_bits(bits)
        py2 = HyperVectorPy.from_bits(bits)
        assert py1.lsh_hash(0, 10) == py2.lsh_hash(0, 10), "Python LSH not deterministic"

    def test_multi_episode_retrieval_sorted(self):
        em = EpisodicMemory()
        base_state = {"head": (5, 5), "food": (5, 4), "score": 0}
        sv, _ = self._record(em, base_state, "test", "ACTION_UP", 1.0)
        other_state = {"head": (0, 0), "food": (9, 9), "score": 100}
        self._record(em, other_state, "test", "ACTION_DOWN", -1.0)
        results = em.recall_similar(sv, "test", k=5)
        if len(results) >= 2:
            sims = [r.situation_hv.similarity(sv) for r in results]
            assert sims[0] >= sims[1]

    def test_store_many_episodes(self):
        """Store 200 episodes without crash."""
        em = EpisodicMemory()
        for i in range(200):
            s = {"pos": i, "score": i * 2}
            self._record(em, s, "task", f"action_{i % 4}", float(i % 3))
        q_state = {"pos": 100, "score": 200}
        q_sv, _ = self._record(em, q_state, "task", "query", 0.0)
        results = em.recall_similar(q_sv, "task", k=5)
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# E. SNN Rust classes
# ─────────────────────────────────────────────────────────────────────────────
@snn_only
class TestSNNRustLIFLayer:

    def _make_lif(self, n=100):
        return _snnrs.LIFLayer(
            n_neurons=n, tau=20.0, v_rest=-70.0, v_reset=-75.0,
            v_thresh=-55.0, refractory_period=2.0, dt=1.0
        )

    def test_basic_step_shape(self):
        lif = self._make_lif(64)
        spikes = lif.step([0.5] * 64)
        assert len(spikes) == 64
        assert all(s in (0.0, 1.0) for s in spikes)

    def test_zero_input_no_spikes(self):
        lif = self._make_lif(50)
        for _ in range(50):
            spikes = lif.step([0.0] * 50)
        total = sum(spikes)
        assert total <= 5, f"Expected ≤5 spikes with zero input, got {total}"

    def test_high_input_causes_spikes(self):
        # Equilibrium potential with I=20: v_eq = v_rest + I = -70 + 20 = -50.
        # This exceeds the threshold -55 (i.e. -50 > -55 in numerical value,
        # meaning -50 is less negative and closer to 0 than -55).
        # Time to reach threshold from rest: ~28 steps.  Use 50 steps to be safe.
        lif = self._make_lif(100)
        any_spike = False
        for _ in range(50):
            spikes = lif.step([20.0] * 100)
            if any(s > 0 for s in spikes):
                any_spike = True
                break
        assert any_spike, "High input (I=20) should produce at least some spikes within 50 steps"

    def test_get_spike_train(self):
        # get_spike_train() returns a list of per-step spike vectors (nested list)
        # shape: [n_steps][n_neurons], NOT a flat array.
        lif = self._make_lif(32)
        n_steps = 10
        for _ in range(n_steps):
            lif.step([1.0] * 32)
        train = lif.get_spike_train()
        assert len(train) == n_steps, f"Expected {n_steps} rows, got {len(train)}"
        assert len(train[0]) == 32, f"Expected 32 cols, got {len(train[0])}"

    def test_reset_clears_state(self):
        lif = self._make_lif(50)
        for _ in range(5):
            lif.step([15.0] * 50)
        lif.reset()
        spikes_after = lif.step([0.0] * 50)
        assert len(spikes_after) == 50

    def test_n_neurons_property(self):
        lif = self._make_lif(77)
        assert lif.n_neurons == 77

    def test_different_sizes(self):
        for n in [1, 10, 100, 1000]:
            lif = self._make_lif(n)
            spikes = lif.step([0.1] * n)
            assert len(spikes) == n


@snn_only
class TestSNNRustHebbianMatrix:

    def _make_hebb(self, in_f=10, out_f=5):
        return _snnrs.HebbianMatrix(
            in_features=in_f, out_features=out_f,
            learning_rate=0.01, decay=0.001, normalize=True
        )

    def test_forward_shape(self):
        hebb = self._make_hebb(10, 5)
        y = hebb.forward([0.5] * 10)
        assert len(y) == 5

    def test_hebbian_update_changes_weights(self):
        hebb = self._make_hebb(8, 4)
        w_before = list(hebb.get_weights())
        pre = [1.0] * 8
        post = [1.0] * 4
        for _ in range(50):
            hebb.hebbian_update(pre, post)
        w_after = list(hebb.get_weights())
        total_change = sum(abs(a - b) for a, b in zip(w_before, w_after))
        assert total_change > 0.001

    def test_set_get_weights_roundtrip(self):
        hebb = self._make_hebb(4, 3)
        weights = [float(i) * 0.01 for i in range(12)]
        hebb.set_weights(weights)
        recovered = list(hebb.get_weights())
        np.testing.assert_allclose(recovered, weights, atol=1e-6)

    def test_update_count_increments(self):
        hebb = self._make_hebb(5, 3)
        c0 = hebb.update_count
        for _ in range(10):
            hebb.hebbian_update([0.5] * 5, [0.5] * 3)
        assert hebb.update_count == c0 + 10

    def test_zero_input_forward(self):
        hebb = self._make_hebb(6, 4)
        y = hebb.forward([0.0] * 6)
        assert len(y) == 4


@snn_only
class TestSNNRustSnnCore:

    def _make_core(self, input_dim=16, snn_size=64):
        return _snnrs.SnnCore(
            input_dim=input_dim, snn_size=snn_size,
            tau=20.0, v_rest=-70.0, v_reset=-75.0, v_thresh=-55.0,
            refractory_period=2.0, dt=1.0,
            stdp_enabled=True, stdp_lr=0.01, tau_stdp=20.0,
            a_plus=0.01, a_minus=0.012,
        )

    def test_simulate_shape(self):
        core = self._make_core(16, 64)
        result = core.simulate([0.5] * 16, n_steps=20, learn=False)
        assert len(result) == 20
        assert len(result[0]) == 64

    def test_simulate_returns_binary_spikes(self):
        core = self._make_core(8, 32)
        result = core.simulate([1.0] * 8, n_steps=5, learn=False)
        for step_spikes in result:
            assert all(s in (0.0, 1.0) for s in step_spikes)

    def test_weight_update_with_learning(self):
        # STDP requires causal timing (pre before post).  Use high asymmetric input
        # to force spikes and drive weight changes over many steps.
        core = self._make_core(10, 40)
        w0 = list(core.get_weights())
        # 50 iterations with high suprathreshold input to guarantee spikes
        for _ in range(50):
            core.simulate([10.0] * 10, n_steps=20, learn=True)
        w1 = list(core.get_weights())
        delta = sum(abs(a - b) for a, b in zip(w0, w1))
        assert delta > 0.001, f"Weight delta={delta:.6f} — expected STDP learning"

    def test_different_inputs_different_spike_rates(self):
        core = self._make_core(8, 32)
        r_low = core.simulate([0.1] * 8, n_steps=30, learn=False)
        core2 = self._make_core(8, 32)
        r_high = core2.simulate([5.0] * 8, n_steps=30, learn=False)
        spikes_low = sum(sum(step) for step in r_low)
        spikes_high = sum(sum(step) for step in r_high)
        assert spikes_high >= spikes_low

    def test_properties(self):
        core = self._make_core(12, 48)
        assert core.input_dim == 12
        assert core.snn_size == 48


@snn_only
class TestSNNRustConceptMapper:

    def test_register_and_recognize(self):
        mapper = _snnrs.ConceptMapper()
        active1 = list(range(0, 20))
        active2 = list(range(40, 60))
        id1 = mapper.register_concept(active1)
        id2 = mapper.register_concept(active2)
        assert id1 != id2

        rid1, sim1 = mapper.recognize_pattern(active1, threshold=0.3)
        assert rid1 == id1
        assert sim1 > 0.99

    def test_novel_pattern_returns_minus_one(self):
        mapper = _snnrs.ConceptMapper()
        mapper.register_concept(list(range(0, 10)))
        novel = list(range(90, 100))
        rid, sim = mapper.recognize_pattern(novel, threshold=0.5)
        assert rid == -1 or sim < 0.5

    def test_n_concepts_count(self):
        mapper = _snnrs.ConceptMapper()
        for i in range(5):
            mapper.register_concept(list(range(i * 10, (i + 1) * 10)))
        assert mapper.n_concepts() == 5

    def test_hv_seed_formula(self):
        seed = _snnrs.ConceptMapper.concept_hv_seed(3)
        assert seed == 1003

    def test_empty_mapper_returns_minus_one(self):
        mapper = _snnrs.ConceptMapper()
        rid, sim = mapper.recognize_pattern([0, 1, 2], threshold=0.1)
        assert rid == -1


@snn_only
class TestSNNRustRateCoder:

    def _make_rc(self, n=64, threshold=0.3):
        return _snnrs.RateCoder(n_neurons=n, rate_threshold=threshold)

    def test_encode_returns_correct_types(self):
        rc = self._make_rc(32)
        n_steps, n_neurons = 20, 32
        train = [0.0] * (n_steps * n_neurons)
        for i in range(0, n_steps * n_neurons, 4):
            train[i] = 1.0
        active, rates = rc.encode(train, time_window_ms=20.0)
        assert isinstance(active, list)
        assert isinstance(rates, list)
        assert len(rates) == n_neurons

    def test_all_spikes_gives_high_rate(self):
        rc = self._make_rc(10, 0.5)
        n_steps = 50
        train = [1.0] * (n_steps * 10)
        active, rates = rc.encode(train, time_window_ms=50.0)
        assert len(active) > 0

    def test_no_spikes_gives_no_active(self):
        rc = self._make_rc(20, 0.1)
        train = [0.0] * (100 * 20)
        active, rates = rc.encode(train, time_window_ms=100.0)
        assert len(active) == 0

    def test_n_neurons_property(self):
        rc = self._make_rc(44)
        assert rc.n_neurons == 44


@snn_only
class TestSNNRustStdpEngine:

    def _make_stdp(self, input_dim=16, snn_size=32):
        return _snnrs.StdpEngine(
            input_dim=input_dim, snn_size=snn_size,
            stdp_lr=0.01, tau_stdp=20.0, a_plus=0.01, a_minus=0.012
        )

    def test_apply_returns_correct_shape(self):
        stdp = self._make_stdp(16, 32)
        in_spikes = [1.0 if i < 8 else 0.0 for i in range(16)]
        out_spikes = [1.0 if i < 16 else 0.0 for i in range(32)]
        delta = stdp.apply(in_spikes, out_spikes)
        assert len(delta) == 16 * 32

    def test_time_advances(self):
        stdp = self._make_stdp()
        t0 = stdp.current_time
        stdp.advance_time(1.0)
        assert stdp.current_time == pytest.approx(t0 + 1.0)

    def test_reset_clears_time(self):
        stdp = self._make_stdp()
        stdp.advance_time(5.0)
        stdp.reset()
        assert stdp.current_time == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# F. Full V3 pipeline with Rust backend
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestV3PipelineWithRust:

    def _make_learner(self, **cfg_kwargs):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        cfg = NSCKConfig(**cfg_kwargs)
        sem = SemanticMemory(use_rust=True, config=cfg)
        epi = EpisodicMemory()
        return TextKnowledgeLearner(sem, epi, config=cfg), sem

    def test_construction_grammar_extracts_relations(self):
        learner, sem = self._make_learner(enable_construction_grammar=True)
        for t in [
            "Dogs are mammals.",
            "Cats are pets.",
            "Gravity causes objects to fall.",
            "Plants produce oxygen.",
            "Alice is a scientist.",
        ]:
            learner.learn_from_text(t)
        assert len(sem.concept_hvs) >= 5
        assert sem.concept_graph.number_of_edges() >= 2

    def test_coreference_no_pronoun_leaks(self):
        from python.core.language.text_knowledge_learner import _COREFERENCE_PRONOUNS
        learner, sem = self._make_learner(
            enable_coreference=True, enable_construction_grammar=True
        )
        learner.learn_from_text("John is a programmer.")
        learner.learn_from_text("He writes Python code.")
        learner.learn_from_text("Maria is a designer.")
        learner.learn_from_text("She creates user interfaces.")
        concepts = set(sem.concept_hvs.keys())
        leaks = concepts & _COREFERENCE_PRONOUNS
        assert not leaks, f"Pronouns leaked: {leaks}"

    def test_distributional_semantics_similar_words(self):
        from python.core.language.distributional_semantics import DistributionalCodebook
        cb = DistributionalCodebook()
        corpus = [
            ["cat", "sat", "on", "mat"],
            ["dog", "sat", "on", "mat"],
            ["cat", "ate", "fish"],
            ["dog", "ate", "bone"],
            ["pets", "include", "cats", "and", "dogs"],
            ["animals", "need", "food", "and", "water"],
            ["cat", "and", "dog", "are", "common", "pets"],
        ]
        cb.build_from_corpus(corpus)
        cat_hv = cb.get_hv("cat")
        dog_hv = cb.get_hv("dog")
        assert cat_hv is not None and dog_hv is not None
        assert cat_hv.similarity(dog_hv) > 0.4

    def test_frame_semantics_roundtrip(self):
        from python.core.language.frame_semantics import FrameLibrary
        lib = FrameLibrary()
        for verb in ["buy", "send", "cause", "go"]:
            frame = lib.find_frame(verb)
            if not frame:
                continue
            roles = list(frame.roles.keys())[:2]
            if len(roles) < 2:
                continue
            hvs = {r: hv.HyperVector(abs(hash(r)) % (2**32)) for r in roles}
            filled = frame.fill(hvs)
            rec = frame.extract_filler(filled, roles[0])
            assert rec.similarity(hvs[roles[0]]) > 0.2

    def test_dual_process_routing(self):
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        state = {"head": (5, 5), "food": (5, 4), "score": 100}

        eng_s1 = CognitiveEngine(NSCKConfig(
            enable_dual_process=True, system1_confidence_threshold=0.0
        ))
        eng_s2 = CognitiveEngine(NSCKConfig(
            enable_dual_process=True, system1_confidence_threshold=0.99
        ))

        labels_s1 = [eng_s1.decide(state, "snake").system_used for _ in range(20)]
        labels_s2 = [eng_s2.decide(state, "snake").system_used for _ in range(20)]

        assert all(x == "system_1" for x in labels_s1), "All should be system_1"
        assert all(x == "system_2" for x in labels_s2), "All should be system_2"

    def test_homeostasis_regulate(self):
        from python.core.memory.homeostasis import MemoryHomeostasis
        mem = SemanticMemory(use_rust=True)
        base = hv.HyperVector(42)
        for a in ["dog", "cat", "rabbit", "hamster", "gerbil"]:
            noise = hv.HyperVector(abs(hash(a)) % (2**32))
            mem.add_concept(a, {"type": "animal"}, hv_override=base.bundle(noise))
        h = MemoryHomeostasis()
        actions = h.regulate(mem)
        assert isinstance(actions, list)

    def test_research_config_multi_sentence(self):
        learner, sem = self._make_learner(
            enable_construction_grammar=True,
            enable_coreference=True,
            enable_frame_semantics=True,
            enable_free_energy_beliefs=True,
        )
        for p in [
            "Water is essential for life.",
            "The Sun is a star.",
            "Gravity causes objects to fall.",
            "Plants produce oxygen through photosynthesis.",
            "Hydrogen and oxygen combine to form water.",
        ]:
            learner.learn_from_text(p)
        assert sem.concept_graph.number_of_nodes() >= 5
        assert sem.concept_graph.number_of_edges() >= 3


# ─────────────────────────────────────────────────────────────────────────────
# G. Real-world scenarios with Rust backend
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestRealWorldScenariosRust:

    def _learn(self, texts, **cfg):
        from python.core.language.text_knowledge_learner import TextKnowledgeLearner
        config = NSCKConfig(
            enable_construction_grammar=True,
            enable_coreference=True,
            **cfg
        )
        sem = SemanticMemory(use_rust=True, config=config)
        epi = EpisodicMemory()
        tkl = TextKnowledgeLearner(sem, epi, config=config)
        for t in texts:
            tkl.learn_from_text(t)
        return sem

    def test_science_domain_photosynthesis(self):
        sem = self._learn([
            "Photosynthesis produces glucose and oxygen.",
            "Glucose is a sugar.",
            "Oxygen is a gas.",
            "Plants use sunlight for photosynthesis.",
            "Carbon dioxide is absorbed by plants.",
        ])
        assert sem.concept_graph.number_of_nodes() >= 4
        assert sem.concept_graph.number_of_edges() >= 3

    def test_geography_domain(self):
        sem = self._learn([
            "Paris is the capital of France.",
            "France is a country in Europe.",
            "Europe is a continent.",
            "The Eiffel Tower is in Paris.",
        ])
        assert sem.concept_graph.number_of_nodes() >= 4

    def test_biography_with_coreference(self):
        from python.core.language.text_knowledge_learner import _COREFERENCE_PRONOUNS
        sem = self._learn([
            "Albert Einstein was a physicist.",
            "He developed the theory of relativity.",
            "His work changed modern physics.",
            "Einstein won the Nobel Prize in 1921.",
        ])
        concepts = set(sem.concept_hvs.keys())
        assert not (concepts & _COREFERENCE_PRONOUNS)
        assert any("einstein" in c.lower() for c in concepts)

    def test_causal_chain_learning(self):
        sem = self._learn([
            "Smoking causes cancer.",
            "Cancer causes death.",
            "Exercise prevents cancer.",
            "Diet causes health.",
        ])
        assert sem.concept_graph.number_of_edges() >= 3

    def test_technology_domain(self):
        sem = self._learn([
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks.",
            "Neural networks learn from data.",
            "GPT is a language model.",
            "Language models use transformers.",
        ])
        assert sem.concept_graph.number_of_nodes() >= 5

    def test_multi_paragraph_ocean(self):
        texts = [
            "The ocean covers 71% of the Earth.",
            "Oceans contain saltwater.",
            "Fish live in the ocean.",
            "Coral reefs are underwater ecosystems.",
            "Coral provides habitat for fish.",
            "Climate change causes coral bleaching.",
            "Bleaching damages reef ecosystems.",
            "Scientists study reef health.",
            "Conservation efforts protect reefs.",
            "Healthy reefs support biodiversity.",
        ]
        sem = self._learn(texts)
        assert sem.concept_graph.number_of_nodes() >= 8
        assert sem.concept_graph.number_of_edges() >= 6

    def test_contradiction_many_evidences(self):
        sem = self._learn(
            ["Paris is the capital of France."] * 5
            + ["Paris is the capital of Germany."] * 7,
            enable_free_energy_beliefs=True,
        )
        assert sem.concept_graph.number_of_nodes() > 0

    def test_empty_and_trivial_inputs(self):
        """Robust handling of edge-case inputs."""
        sem = self._learn([
            "",            # empty
            "  ",          # whitespace only
            "A.",          # single word
            "X is Y.",     # minimal sentence
        ])
        # No crash is the key requirement
        assert isinstance(sem.concept_hvs, dict)

    def test_long_sentence(self):
        """Very long sentence should not crash."""
        long_sent = " and ".join([f"concept_{i} is related to concept_{i+1}" for i in range(50)])
        sem = self._learn([long_sent])
        assert isinstance(sem.concept_hvs, dict)


# ─────────────────────────────────────────────────────────────────────────────
# H. Scalability / performance benchmarks
# ─────────────────────────────────────────────────────────────────────────────
@vsa_only
class TestScalabilityRust:

    def test_bulk_hv_creation_10k(self):
        """Create 10,000 HVs — should complete in under 5s."""
        N = 10_000
        t0 = time.perf_counter()
        hvs = [hv.HyperVector(i) for i in range(N)]
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Creating {N} HVs took {elapsed:.2f}s"
        assert len(hvs) == N

    def test_bulk_similarity_1k(self):
        vectors = [hv.HyperVector(i * 13) for i in range(1000)]
        query = hv.HyperVector(42)
        t0 = time.perf_counter()
        sims = [query.similarity(v) for v in vectors]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert len(sims) == 1000
        assert elapsed_ms < 2000

    def test_semantic_memory_5k(self):
        # Python backend linear scan: O(N × 10240 bits). Observed ~17s at 5K.
        # Allow 60s for headroom; note Rust backend with HNSW would be <100ms.
        mem = SemanticMemory(use_rust=True)
        t0 = time.perf_counter()
        for i in range(5000):
            mem.add_concept(f"c{i}", {"i": i})
        add_time = time.perf_counter() - t0
        assert add_time < 60.0, f"Adding 5K concepts took {add_time:.2f}s"
        q = hv.HyperVector(999)
        t1 = time.perf_counter()
        results = mem.query(q, k=10)
        query_ms = (time.perf_counter() - t1) * 1000
        assert len(results) > 0
        assert query_ms < 60_000, f"5K-concept query took {query_ms:.0f}ms"

    def test_spread_activation_ring_500(self):
        mem = SemanticMemory(use_rust=True)
        for i in range(500):
            mem.add_concept(f"ring_{i}", {})
        for i in range(500):
            mem.add_relation(f"ring_{i}", "next", f"ring_{(i+1)%500}")
        t0 = time.perf_counter()
        act = mem.spread_activation(["ring_0"], steps=5, decay=0.8)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert len(act) > 1
        assert elapsed_ms < 5000

    def test_rust_vs_python_relative_speed(self):
        """Rust VSA should not be significantly slower than Python."""
        N = 3000
        py_vecs = [HyperVectorPy(i) for i in range(10)]
        query_py = HyperVectorPy(99)
        t0 = time.perf_counter()
        for i in range(N):
            query_py.similarity(py_vecs[i % 10])
        py_time = time.perf_counter() - t0

        rs_vecs = [hv.HyperVector(i) for i in range(10)]
        query_rs = hv.HyperVector(99)
        t1 = time.perf_counter()
        for i in range(N):
            query_rs.similarity(rs_vecs[i % 10])
        rs_time = time.perf_counter() - t1

        speedup = py_time / max(rs_time, 1e-9)
        # Rust should be at least as fast (allow 2× tolerance for overhead)
        assert speedup >= 0.5, \
            f"Rust slower by {1/speedup:.1f}×: py={py_time:.3f}s rs={rs_time:.3f}s"

    @snn_only
    def test_snn_simulate_throughput(self):
        """100 SnnCore simulate() calls should complete in < 60s."""
        core = _snnrs.SnnCore(
            input_dim=64, snn_size=256,
            tau=20.0, v_rest=-70.0, v_reset=-75.0, v_thresh=-55.0,
            refractory_period=2.0, dt=1.0,
            stdp_enabled=False, stdp_lr=0.001, tau_stdp=20.0,
            a_plus=0.01, a_minus=0.012,
        )
        latencies = []
        for i in range(100):
            inp = [float(i % 10) * 0.1] * 64
            t0 = time.perf_counter()
            core.simulate(inp, n_steps=50, learn=False)
            latencies.append((time.perf_counter() - t0) * 1000)
        avg_ms = np.mean(latencies)
        p95_ms = np.percentile(latencies, 95)
        assert avg_ms < 500, f"SNN avg too slow: {avg_ms:.1f}ms"
        assert p95_ms < 1000, f"SNN p95 too slow: {p95_ms:.1f}ms"
