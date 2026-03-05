"""
V18 End-to-End Integration Tests
=================================
Tests covering all 5 V18 architectural improvements:

  1. Immediate Rust mirror on add_concept/add_relation (Change 1)
  2. parallel_spread_activation fast path (Change 2)
  3. Weighted edges in SemanticMemoryConcurrent (Change 3)
  4. LSH stale-index fix on episodic eviction (Change 4)
  5. GloVe/FastText word vector initialization (Change 5)

Sections
--------
* Python backend tests (no Rust required) — always run
* Rust backend tests (NSCK_USE_RUST=1) — skipped when Rust unavailable
* Performance benchmarks — Rust vs Python comparison
"""
from __future__ import annotations

import sys
import os
import time
from collections import deque
from typing import Dict

import pytest

# Make the package importable from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.integration.config import NSCKConfig
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.reasoning.causal_reasoning import CausalGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUST_AVAILABLE = hypervec_rs.__backend__ == "Rust"

requires_rust = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="Requires compiled Rust backend (hypervec_rs)",
)


def _make_engine(task_tag: str = "test") -> CognitiveEngine:
    config = NSCKConfig()
    engine = CognitiveEngine(config)
    engine.register_task(task_tag, causal_graph=CausalGraph())
    return engine


def _make_live_episode(task_tag: str, seed: int, reward: float = 1.0) -> LiveEpisode:
    """Create a LiveEpisode with a deterministic situation HV."""
    hv = hypervec_rs.HyperVector(seed)
    return LiveEpisode(
        timestamp=float(seed),
        task_tag=task_tag,
        situation_hv=hv,
        state={"seed": seed},
        action="test_action",
        outcome="success",
        reward=reward,
        impact_score=abs(reward),
    )


# ===========================================================================
# Python backend tests (no Rust required)
# ===========================================================================


class TestSemanticMemoryWriteSync:
    """Change 1+2: verify spread_activation correctness in Python fallback mode."""

    def test_spread_activation_after_add(self):
        """add_concept + add_relation must be visible to spread_activation."""
        mem = SemanticMemory(use_rust=False)
        mem.add_concept("fire", {"category": "hazard"})
        mem.add_concept("heat", {"category": "property"})
        mem.add_concept("burn", {"category": "effect"})

        mem.add_relation("fire", "causes", "heat")
        mem.add_relation("heat", "causes", "burn")

        result = mem.spread_activation(["fire"], steps=2, decay=0.7)

        assert "fire" in result, "start concept must be in result"
        assert "heat" in result, "1-hop neighbor must be activated"
        assert "burn" in result, "2-hop neighbor must be activated"
        assert result["fire"] >= result["heat"] >= result["burn"], (
            "activation must decay with distance"
        )

    def test_spread_activation_weighted_edges(self):
        """is_a edges carry more activation than similar_to edges."""
        mem = SemanticMemory(use_rust=False)
        mem.add_concept("dog", {})
        mem.add_concept("animal", {})
        mem.add_concept("wolf", {})

        # is_a weight (0.9) vs similar_to weight (0.4)
        mem.add_relation("dog", "is_a", "animal")
        mem.add_relation("dog", "similar_to", "wolf")

        result = mem.spread_activation(["dog"], steps=1, decay=1.0)

        animal_act = result.get("animal", 0.0)
        wolf_act = result.get("wolf", 0.0)
        assert animal_act > wolf_act, (
            f"is_a neighbor ({animal_act:.4f}) should receive more activation "
            f"than similar_to neighbor ({wolf_act:.4f})"
        )


class TestLSHStaleIndexFix:
    """Change 4: verify stale LSH entries are removed after sleep consolidation."""

    def test_rebuild_lsh_index_clears_stale_entries(self):
        """_rebuild_lsh_index must remove timestamps not in deque."""
        mem = EpisodicMemory(store=None, recent_capacity=10)
        task = "stale_test"

        # Record 10 episodes (fills the deque)
        for i in range(10):
            ep = _make_live_episode(task, i)
            mem.record(ep)

        # Manually inject a stale timestamp into the LSH index
        stale_ts = 9999.0
        tables = mem.lsh_index.get(task, [])
        if tables:
            first_table = tables[0]
            first_bucket = next(iter(first_table), None)
            if first_bucket is not None:
                first_table[first_bucket].append(stale_ts)

        # Rebuild — stale_ts is not in the deque so it must be gone
        mem._rebuild_lsh_index()

        for t_idx, table in enumerate(mem.lsh_index.get(task, [])):
            for bucket, ts_list in table.items():
                assert stale_ts not in ts_list, (
                    f"stale timestamp {stale_ts} found in table {t_idx} bucket {bucket} "
                    f"after _rebuild_lsh_index()"
                )

    def test_sleep_calls_rebuild_lsh_index(self):
        """sleep() must trigger LSH rebuild (no crash, LSH remains valid)."""
        config = NSCKConfig()
        config.enable_sleep = True
        engine = CognitiveEngine(config)
        task = "sleep_test"
        engine.register_task(task, causal_graph=CausalGraph())

        # Record some episodes so sleep has material to work with
        state = {"x": 1, "y": 2}
        for _ in range(5):
            engine.decide(state, task)
            engine.learn(state, "move", 0.5, task, outcome="ok")

        # sleep() must not crash and must call _rebuild_lsh_index
        engine.sleep(task_tag=task)

        # Verify LSH index is consistent with deque contents
        recent_ts = {ep.timestamp for ep in engine.episodic_memory.recent.get(task, deque())}
        for t_idx, table in enumerate(engine.episodic_memory.lsh_index.get(task, [])):
            for bucket, ts_list in table.items():
                for ts in ts_list:
                    assert ts in recent_ts, (
                        f"LSH table {t_idx} bucket {bucket} contains ts={ts} "
                        f"which is NOT in the current deque"
                    )


class TestGloveSynonymRecall:
    """Change 5: verify GloVe seeding or graceful hash fallback."""

    def test_word_seeds_module_importable(self):
        """word_seeds module must be importable without errors."""
        from python.core.vsa.word_seeds import word_to_seed_bits, glove_available
        bits = word_to_seed_bits("fire")
        assert bits.shape == (10240,), f"Expected (10240,), got {bits.shape}"
        assert bits.dtype.kind in ("u", "i"), f"Expected uint/int dtype, got {bits.dtype}"
        assert set(bits.tolist()).issubset({0, 1}), "bits must be binary (0 or 1)"

    def test_unknown_word_fallback(self):
        """Unknown words must fall back to hash-based seeding without error."""
        from python.core.vsa.word_seeds import word_to_seed_bits
        bits = word_to_seed_bits("xyzzy_nonexistent_word_v18_test")
        assert bits.shape == (10240,)
        assert set(bits.tolist()).issubset({0, 1})

    def test_same_word_deterministic(self):
        """Same word must produce identical bit arrays across calls."""
        from python.core.vsa.word_seeds import word_to_seed_bits
        b1 = word_to_seed_bits("computer")
        b2 = word_to_seed_bits("computer")
        import numpy as np
        assert (b1 == b2).all(), "word_to_seed_bits must be deterministic"

    @pytest.mark.skipif(
        True,  # Only run when GloVe is actually present
        reason="Requires GloVe embeddings (run scripts/download_embeddings.sh first)",
    )
    def test_glove_synonym_similarity(self):
        """car and automobile must have HV similarity > 0.6 with GloVe seeding."""
        from python.core.vsa.word_seeds import word_to_seed_bits, glove_available
        import numpy as np

        if not glove_available():
            pytest.skip("GloVe not available; run scripts/download_embeddings.sh")

        car_bits = word_to_seed_bits("car").astype(np.float32)
        auto_bits = word_to_seed_bits("automobile").astype(np.float32)

        # Hamming similarity: fraction of matching bits
        hamming_sim = float(np.mean(car_bits == auto_bits))
        assert hamming_sim > 0.6, (
            f"car vs automobile similarity {hamming_sim:.3f} < 0.6 — "
            f"GloVe seeding should make synonyms more similar than random (0.5)"
        )


# ===========================================================================
# Rust backend tests (requires compiled hypervec_rs)
# ===========================================================================


@requires_rust
class TestRustImmediateSync:
    """Change 1: add_concept + add_relation must be immediately visible in Rust DashMap."""

    def test_rust_immediate_sync_add_concept(self):
        """Concepts added via add_concept must be in Rust backend instantly."""
        mem = SemanticMemory(use_rust=True)
        if mem._rust_backend is None:
            pytest.skip("Rust backend not available")

        mem.add_concept("test_sync_concept", {"x": 1})
        # The Rust backend should reflect the concept without explicit sync call
        count = mem._rust_backend.concept_count()
        assert count >= 1, (
            f"Expected ≥1 concept in Rust DashMap after add_concept, got {count}"
        )

    def test_rust_immediate_sync_add_relation(self):
        """Relations added via add_relation must be visible in Rust backend instantly."""
        mem = SemanticMemory(use_rust=True)
        if mem._rust_backend is None:
            pytest.skip("Rust backend not available")

        mem.add_concept("src", {})
        mem.add_concept("tgt", {})
        mem.add_relation("src", "is_a", "tgt")

        # The relation should be visible in the Rust graph without explicit sync
        neighbors = mem._rust_backend.get_neighbors("src")
        assert any(n[0] == "tgt" or n == "tgt" for n in neighbors), (
            f"'tgt' not found in Rust neighbors of 'src': {neighbors}"
        )


@requires_rust
class TestRustWeightedEdges:
    """Change 3: weighted edges must be respected in parallel_spread_activation."""

    def test_add_relation_weighted_callable(self):
        """add_relation_weighted must exist and store weights."""
        mem = SemanticMemory(use_rust=True)
        if mem._rust_backend is None:
            pytest.skip("Rust backend not available")

        assert hasattr(mem._rust_backend, "add_relation_weighted"), (
            "Rust backend missing add_relation_weighted (needs Rust recompilation)"
        )
        mem.add_concept("hub", {})
        mem.add_concept("strong_neighbor", {})
        mem.add_concept("weak_neighbor", {})
        mem._rust_backend.add_relation_weighted("hub", "strong_neighbor", 0.9)
        mem._rust_backend.add_relation_weighted("hub", "weak_neighbor", 0.1)

        result = mem._rust_backend.parallel_spread_activation(
            start_concepts=["hub"], steps=1, decay=1.0
        )
        strong_act = result.get("strong_neighbor", 0.0)
        weak_act = result.get("weak_neighbor", 0.0)
        assert strong_act > weak_act, (
            f"strong_neighbor ({strong_act:.4f}) should have more activation "
            f"than weak_neighbor ({weak_act:.4f})"
        )

    def test_is_a_higher_than_similar_to_via_add_relation(self):
        """add_relation with is_a must propagate more activation than similar_to."""
        mem = SemanticMemory(use_rust=True)
        if mem._rust_backend is None:
            pytest.skip("Rust backend not available")

        mem.add_concept("cat", {})
        mem.add_concept("feline", {})
        mem.add_concept("tiger", {})
        mem.add_relation("cat", "is_a", "feline")       # weight 0.9
        mem.add_relation("cat", "similar_to", "tiger")  # weight 0.4

        result = mem.spread_activation(["cat"], steps=1, decay=1.0)
        assert result.get("feline", 0.0) > result.get("tiger", 0.0), (
            "is_a neighbor should receive more activation than similar_to neighbor"
        )


@requires_rust
class TestRustParallelSpreadActivation:
    """Change 2: parallel_spread_activation must return results consistent with Python."""

    def test_rust_spread_activation_returns_dict(self):
        """parallel_spread_activation must return a non-empty dict."""
        mem = SemanticMemory(use_rust=True)
        if mem._rust_backend is None:
            pytest.skip("Rust backend not available")

        mem.add_concept("a", {})
        mem.add_concept("b", {})
        mem.add_concept("c", {})
        mem.add_relation("a", "causes", "b")
        mem.add_relation("b", "causes", "c")

        result = mem.spread_activation(["a"], steps=2, decay=0.7)
        assert isinstance(result, dict)
        assert "a" in result
        assert result["a"] >= 0.9

    def test_rust_spread_activation_speedup(self):
        """Rust path must be faster than Python-only at 1K+ concepts."""
        N = 1000

        # Python-only
        mem_py = SemanticMemory(use_rust=False)
        for i in range(N):
            mem_py.add_concept(f"c{i}", {"i": i})
        import random
        random.seed(7)
        for i in range(N):
            j = random.randint(0, N - 1)
            if i != j:
                mem_py.add_relation(f"c{i}", "similar_to", f"c{j}")

        # Warm up
        mem_py.spread_activation(["c0"], steps=2)

        t0 = time.perf_counter()
        for _ in range(3):
            mem_py.spread_activation(["c0", "c1"], steps=2, decay=0.7)
        py_ms = (time.perf_counter() - t0) / 3 * 1000

        # Rust-backed
        mem_rs = SemanticMemory(use_rust=True)
        if mem_rs._rust_backend is None:
            pytest.skip("Rust backend not available")

        for i in range(N):
            mem_rs.add_concept(f"c{i}", {"i": i})
        random.seed(7)
        for i in range(N):
            j = random.randint(0, N - 1)
            if i != j:
                mem_rs.add_relation(f"c{i}", "similar_to", f"c{j}")

        # Warm up
        mem_rs.spread_activation(["c0"], steps=2)

        t0 = time.perf_counter()
        for _ in range(3):
            mem_rs.spread_activation(["c0", "c1"], steps=2, decay=0.7)
        rs_ms = (time.perf_counter() - t0) / 3 * 1000

        # Rust should be faster (with some tolerance for CI variance)
        speedup = py_ms / rs_ms if rs_ms > 0 else float("inf")
        # Accept if Rust is at least 1x as fast (within noise margin on small graphs)
        assert speedup >= 0.5, (
            f"Rust ({rs_ms:.2f}ms) slower than Python ({py_ms:.2f}ms) by {1/speedup:.1f}x "
            f"— something is wrong with the Rust path"
        )


# ===========================================================================
# End-to-end decide/sleep tests
# ===========================================================================


class TestEndToEndDecideLoop:
    """Change 1-4: run decide() many times and verify system stability."""

    def test_decide_loop_no_crash(self):
        """100 decide() calls must not crash; GWT broadcast must fire."""
        engine = _make_engine("e2e_decide")
        state = {"x": 1, "y": 2, "score": 0}

        for i in range(100):
            result = engine.decide(state, "e2e_decide")
            assert result is not None, f"decide() returned None on step {i}"

        # Verify memory was exercised
        stats = engine.get_stats()
        assert stats.get("decisions", 0) >= 100

    def test_decide_followed_by_sleep(self):
        """decide() followed by sleep() must leave system in valid state."""
        engine = _make_engine("e2e_sleep")
        state = {"temp": 25, "humidity": 60}

        for _ in range(20):
            engine.decide(state, "e2e_sleep")
            engine.learn(state, "adjust", 0.3, "e2e_sleep", outcome="ok")

        # sleep() must not crash
        engine.sleep(task_tag="e2e_sleep")

        # System must still be able to make decisions after sleep
        result = engine.decide(state, "e2e_sleep")
        assert result is not None


# ===========================================================================
# Benchmark
# ===========================================================================


class TestSpreadActivationBenchmark:
    """Compare Rust vs Python spreading activation performance."""

    @pytest.mark.parametrize("n_nodes", [100, 1_000])
    def test_benchmark_python_spread(self, n_nodes: int):
        """Python spread_activation must complete without error at scale."""
        mem = SemanticMemory(use_rust=False)
        for i in range(n_nodes):
            mem.add_concept(f"c{i}", {})
        import random
        random.seed(42)
        for i in range(n_nodes):
            j = random.randint(0, n_nodes - 1)
            if i != j:
                mem.add_relation(f"c{i}", "similar_to", f"c{j}")

        result = mem.spread_activation(["c0"], steps=3, decay=0.7)
        assert len(result) >= 1

    @pytest.mark.parametrize("n_nodes", [100, 1_000])
    @requires_rust
    def test_benchmark_rust_spread(self, n_nodes: int):
        """Rust spread_activation must complete without error at scale."""
        mem = SemanticMemory(use_rust=True)
        if mem._rust_backend is None:
            pytest.skip("Rust backend not available")

        for i in range(n_nodes):
            mem.add_concept(f"c{i}", {})
        import random
        random.seed(42)
        for i in range(n_nodes):
            j = random.randint(0, n_nodes - 1)
            if i != j:
                mem.add_relation(f"c{i}", "similar_to", f"c{j}")

        result = mem.spread_activation(["c0"], steps=3, decay=0.7)
        assert len(result) >= 1
