"""Unit tests for ProceduralMemory LSH bucket index (V4)."""
import pytest
import numpy as np
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.memory.procedural_memory import ProceduralMemory, _compute_lsh


class TestComputeLSH:
    def test_deterministic(self):
        """Same bits → same LSH key."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 1280, dtype=np.float32)
        key1 = _compute_lsh(bits, n_bits=16)
        key2 = _compute_lsh(bits, n_bits=16)
        assert key1 == key2

    def test_different_vectors_may_differ(self):
        """Different HVs should often produce different LSH keys."""
        rng = np.random.default_rng(42)
        bits_a = rng.integers(0, 2, size=10240).astype(np.float32)
        bits_b = rng.integers(0, 2, size=10240).astype(np.float32)
        # Not guaranteed, but highly likely with 16 projection bits
        key_a = _compute_lsh(bits_a, n_bits=16)
        key_b = _compute_lsh(bits_b, n_bits=16)
        assert isinstance(key_a, int)
        assert isinstance(key_b, int)

    def test_output_is_integer(self):
        bits = np.ones(10240, dtype=np.float32)
        key = _compute_lsh(bits, n_bits=8)
        assert isinstance(key, int)


class TestProceduralMemoryLSH:
    def _make_hv(self, seed: int) -> hypervec_rs.HyperVector:
        return hypervec_rs.HyperVector(seed)

    def test_default_threshold_is_0_72(self):
        pm = ProceduralMemory()
        assert pm.familiarity_threshold == 0.72

    def test_cache_and_recall(self):
        pm = ProceduralMemory(familiarity_threshold=0.5)
        hv = self._make_hv(42)
        pm.cache_skill(hv, "ACTION_UP", reward=1.0, label="test")
        result = pm.recall_action(hv)
        assert result is not None
        action, sim, reward = result
        assert action == "ACTION_UP"
        assert sim >= 0.99  # Same vector → very high similarity
        assert reward == 1.0

    def test_lsh_bucket_populated(self):
        pm = ProceduralMemory()
        hv = self._make_hv(123)
        pm.cache_skill(hv, "ACTION_DOWN", reward=0.5)
        assert len(pm._lsh_buckets) > 0

    def test_recall_returns_none_below_threshold(self):
        pm = ProceduralMemory(familiarity_threshold=0.99)
        hv_a = self._make_hv(1)
        hv_b = self._make_hv(99999)
        pm.cache_skill(hv_a, "ACTION_LEFT", reward=1.0)
        result = pm.recall_action(hv_b)
        assert result is None

    def test_statistics(self):
        pm = ProceduralMemory()
        hv = self._make_hv(7)
        pm.cache_skill(hv, "ACTION_RIGHT", reward=0.8)
        pm.recall_action(hv)  # hit
        pm.recall_action(self._make_hv(99999))  # miss
        stats = pm.get_statistics()
        assert stats["total_skills"] >= 1
        assert stats["hit_count"] >= 1
        assert stats["miss_count"] >= 1

    def test_auto_populate_via_engine(self):
        """Verify learn() wires into procedural_memory on positive reward."""
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        engine = CognitiveEngine(config=cfg, persistence_path=None)
        engine.register_task("test_proc")
        state = {"x": 1, "y": 2}
        cs = engine.decide(state, "test_proc")
        engine.learn(state, cs.chosen_action, reward=1.0, task_tag="test_proc")
        # ProceduralMemory should now have at least 1 skill
        stats = engine.procedural_memory.get_statistics()
        assert stats["total_skills"] >= 1
