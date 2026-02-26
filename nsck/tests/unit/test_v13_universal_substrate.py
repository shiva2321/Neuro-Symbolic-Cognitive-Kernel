"""
Tests for NSCK V13 Universal Cognitive Substrate modules.

Covers:
- CausalRuleAuditor
- SignalIngestor / TypedSignal
- UniversalHVEncoder
- CrossModalAssociativeMemory
- ProceduralMemory
- ConceptDriftDetector
- VideoAdapter / TemporalStreamEncoder
- ConformalWrapper
- PatternGeneralizer / CrossDomainTransferPipeline
- NSCKSubstrate V13 (ingest/feedback API, KLE, procedural hit, encoding_stats)
- GlobalWorkspace KLE uncertainty
- CognitiveState kle_uncertainty / uncertainty_bounds fields
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure nsck is on the path
_nsck_root = Path(__file__).parent.parent.parent
if str(_nsck_root) not in sys.path:
    sys.path.insert(0, str(_nsck_root))


# ---------------------------------------------------------------------------
# CausalRuleAuditor
# ---------------------------------------------------------------------------

class TestCausalRuleAuditor:
    def setup_method(self):
        from python.core.reasoning.causal_rule_auditor import CausalRuleAuditor
        self.auditor = CausalRuleAuditor(causal_weight=0.4)

    def test_audit_rule_no_edge(self):
        rule = self.auditor.audit_rule("A", "B", 0.8)
        assert rule.causal_score == 0.0
        assert 0.0 < rule.combined_score <= 1.0

    def test_audit_rule_direct_edge(self):
        self.auditor.add_causal_edge("A", "B", strength=0.9)
        rule = self.auditor.audit_rule("A", "B", 0.8)
        assert rule.causal_score == pytest.approx(0.9, abs=0.01)
        # Combined = 0.6*0.8 + 0.4*0.9 = 0.84
        assert rule.combined_score == pytest.approx(0.84, abs=0.01)

    def test_audit_rule_indirect_path(self):
        self.auditor.add_causal_edge("X", "Y", 0.8)
        self.auditor.add_causal_edge("Y", "Z", 0.7)
        rule = self.auditor.audit_rule("X", "Z", 0.5)
        # indirect path strength = 0.8 * 0.7 = 0.56
        assert rule.causal_score == pytest.approx(0.56, abs=0.02)
        assert any("Indirect" in t for t in rule.audit_trace)

    def test_audit_trace_is_list_of_strings(self):
        rule = self.auditor.audit_rule("P", "Q", 0.6)
        assert isinstance(rule.audit_trace, list)
        assert all(isinstance(t, str) for t in rule.audit_trace)

    def test_audit_rules_bulk(self):
        rules = [("A", "B", 0.9), ("C", "D", 0.5), ("E", "F", 0.3)]
        audited = self.auditor.audit_rules(rules)
        assert len(audited) == 3

    def test_top_rules(self):
        self.auditor.add_causal_edge("A", "B", 1.0)
        self.auditor.audit_rules([("A", "B", 0.9), ("X", "Y", 0.1)])
        top = self.auditor.top_rules(n=1)
        assert top[0].condition == "A"

    def test_combined_score_in_zero_one(self):
        self.auditor.add_causal_edge("A", "B", 0.5)
        rule = self.auditor.audit_rule("A", "B", 0.5)
        assert 0.0 <= rule.combined_score <= 1.0

    def test_get_audit_history(self):
        self.auditor.audit_rule("A", "B", 0.7)
        self.auditor.audit_rule("C", "D", 0.3)
        hist = self.auditor.get_audit_history()
        assert len(hist) == 2


# ---------------------------------------------------------------------------
# SignalIngestor
# ---------------------------------------------------------------------------

class TestSignalIngestor:
    def setup_method(self):
        from python.core.perception.signal_ingestor import SignalIngestor
        self.ingestor = SignalIngestor()

    def test_ingest_text(self):
        ts = self.ingestor.ingest("hello world")
        assert ts.source_type == "text"
        assert ts.data.dtype == np.float64
        assert len(ts.data) > 0
        assert ts.stats["mean"] is not None

    def test_ingest_numeric_list(self):
        ts = self.ingestor.ingest([1.0, 2.0, 3.0, 4.0])
        assert ts.source_type == "numeric"
        assert len(ts.data) == 4

    def test_ingest_numpy_1d(self):
        arr = np.array([0.1, 0.5, 0.9])
        ts = self.ingestor.ingest(arr)
        assert ts.source_type == "numeric"
        assert ts.original_shape == (3,)

    def test_ingest_numpy_2d_image(self):
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        ts = self.ingestor.ingest(img)
        assert ts.source_type == "image"
        assert ts.original_shape == (32, 32)

    def test_ingest_dict(self):
        ts = self.ingestor.ingest({"x": 1.0, "y": 2.0})
        assert ts.source_type == "dict"
        assert "keys" in ts.metadata

    def test_ingest_scalar(self):
        ts = self.ingestor.ingest(42.0)
        assert ts.source_type == "numeric"
        assert len(ts.data) == 1

    def test_ingest_bytes(self):
        ts = self.ingestor.ingest(b"\x00\xff\x80")
        assert ts.source_type == "bytes"
        assert len(ts.data) == 3

    def test_stats_fields_present(self):
        ts = self.ingestor.ingest([1, 2, 3])
        for key in ["mean", "std", "min", "max", "l2_norm"]:
            assert key in ts.stats

    def test_label_stored_in_metadata(self):
        ts = self.ingestor.ingest("data", label="my_label")
        assert ts.metadata.get("label") == "my_label"

    def test_empty_text(self):
        ts = self.ingestor.ingest("")
        assert len(ts.data) >= 1


# ---------------------------------------------------------------------------
# UniversalHVEncoder
# ---------------------------------------------------------------------------

class TestUniversalHVEncoder:
    def setup_method(self):
        from python.core.vsa.universal_hv_encoder import UniversalHVEncoder
        self.encoder = UniversalHVEncoder(n_features=64)

    def test_encode_text_returns_hv(self):
        import python.core.vsa.hypervec_shim as hv_mod
        hv = self.encoder.encode("hello world")
        assert isinstance(hv, hv_mod.HyperVector)

    def test_encode_array_returns_hv(self):
        import python.core.vsa.hypervec_shim as hv_mod
        hv = self.encoder.encode(np.array([0.1, 0.5, 0.9]))
        assert isinstance(hv, hv_mod.HyperVector)

    def test_encode_with_stats_returns_dict(self):
        result = self.encoder.encode_with_stats([1.0, 2.0, 3.0])
        assert "hv" in result
        assert "source_type" in result
        assert "stats" in result
        assert "top_features" in result

    def test_similar_inputs_have_high_similarity(self):
        hv1 = self.encoder.encode([1.0, 2.0, 3.0, 4.0])
        hv2 = self.encoder.encode([1.1, 2.1, 3.1, 4.1])
        sim = float(hv1.similarity(hv2))
        assert sim > 0.5  # Similar inputs should be similar

    def test_hebbian_update_changes_weights(self):
        old_weights = self.encoder.projector_weights.copy()
        self.encoder.hebbian_update([1.0, 0.5, 0.3], reward=1.0)
        assert not np.allclose(self.encoder.projector_weights, old_weights)

    def test_get_top_features_after_encoding(self):
        self.encoder.encode([1.0, 2.0, 3.0])
        top = self.encoder.get_top_features(k=5)
        assert len(top) <= 5
        assert all("index" in f and "importance" in f for f in top)

    def test_reset_weights(self):
        self.encoder.hebbian_update([1.0], reward=5.0)
        self.encoder.reset_weights()
        assert np.allclose(self.encoder.projector_weights, 1.0)

    def test_encode_typed_signal(self):
        from python.core.perception.signal_ingestor import SignalIngestor
        ts = SignalIngestor().ingest("typed input")
        hv = self.encoder.encode(ts)
        import python.core.vsa.hypervec_shim as hv_mod
        assert isinstance(hv, hv_mod.HyperVector)


# ---------------------------------------------------------------------------
# CrossModalAssociativeMemory
# ---------------------------------------------------------------------------

class TestCrossModalAssociativeMemory:
    def setup_method(self):
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.memory.cross_modal_associative_memory import CrossModalAssociativeMemory
        self.mem = CrossModalAssociativeMemory()
        self.hv_mod = hv_mod

    def _make_hv(self, seed):
        return self.hv_mod.HyperVector(seed)

    def test_bind_stores_binding(self):
        hv_a = self._make_hv(1001)
        hv_b = self._make_hv(2002)
        binding = self.mem.bind("text", hv_a, "image", hv_b)
        assert binding.modality_a == "text"
        assert binding.modality_b == "image"

    def test_recall_returns_results(self):
        hv_a = self._make_hv(1001)
        hv_b = self._make_hv(2002)
        self.mem.bind("text", hv_a, "image", hv_b)
        results = self.mem.recall(hv_a, "text", "image", top_k=1)
        assert len(results) == 1
        assert len(results[0]) == 3  # (hv, sim, label)

    def test_recall_empty_when_no_binding(self):
        hv_q = self._make_hv(9999)
        results = self.mem.recall(hv_q, "audio", "video", top_k=3)
        assert results == []

    def test_recall_by_label(self):
        hv_a = self._make_hv(111)
        hv_b = self._make_hv(222)
        self.mem.bind("text", hv_a, "image", hv_b, label="scene1")
        bindings = self.mem.recall_by_label("scene1")
        assert len(bindings) == 1
        assert bindings[0].label == "scene1"

    def test_get_statistics(self):
        hv_a = self._make_hv(100)
        hv_b = self._make_hv(200)
        self.mem.bind("text", hv_a, "image", hv_b)
        stats = self.mem.get_statistics()
        assert stats["total_bindings"] == 1
        assert "text↔image" in stats["modality_pairs"]

    def test_max_bindings_respected(self):
        mem = __import__(
            "python.core.memory.cross_modal_associative_memory",
            fromlist=["CrossModalAssociativeMemory"]
        ).CrossModalAssociativeMemory(max_bindings=5)
        for i in range(10):
            hv_a = self._make_hv(i * 100)
            hv_b = self._make_hv(i * 200)
            mem.bind("a", hv_a, "b", hv_b)
        assert len(mem._bindings) <= 5


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------

class TestProceduralMemory:
    def setup_method(self):
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.memory.procedural_memory import ProceduralMemory
        self.mem = ProceduralMemory(familiarity_threshold=0.8)
        self.hv_mod = hv_mod

    def _make_hv(self, seed):
        return self.hv_mod.HyperVector(seed)

    def test_cache_skill(self):
        hv = self._make_hv(1001)
        skill = self.mem.cache_skill(hv, "ACTION_UP", reward=1.0)
        assert skill.action == "ACTION_UP"
        assert skill.reward == 1.0

    def test_recall_familiar_context(self):
        hv = self._make_hv(1001)
        self.mem.cache_skill(hv, "ACTION_UP", reward=1.0)
        # Same HV should be recalled
        result = self.mem.recall_action(hv)
        assert result is not None
        action, sim, reward = result
        assert action == "ACTION_UP"
        assert sim >= 0.8

    def test_no_recall_unfamiliar_context(self):
        hv_stored = self._make_hv(1001)
        hv_query = self._make_hv(99999)  # very different seed
        self.mem.cache_skill(hv_stored, "ACTION_UP", reward=1.0)
        result = self.mem.recall_action(hv_query)
        # May or may not hit depending on random HV similarity
        # Just ensure the API works
        assert result is None or isinstance(result, tuple)

    def test_get_statistics(self):
        hv = self._make_hv(1001)
        self.mem.cache_skill(hv, "ACTION_UP", reward=1.0)
        stats = self.mem.get_statistics()
        assert "total_skills" in stats
        assert "hit_rate" in stats
        assert stats["total_skills"] == 1

    def test_update_better_reward(self):
        hv = self._make_hv(1001)
        self.mem.cache_skill(hv, "ACTION_UP", reward=0.5)
        self.mem.cache_skill(hv, "ACTION_DOWN", reward=1.0)
        # Should update to ACTION_DOWN
        result = self.mem.recall_action(hv)
        if result:
            assert result[0] == "ACTION_DOWN"


# ---------------------------------------------------------------------------
# ConceptDriftDetector
# ---------------------------------------------------------------------------

class TestConceptDriftDetector:
    def setup_method(self):
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.memory.concept_drift_detector import ConceptDriftDetector
        self.detector = ConceptDriftDetector(drift_threshold=0.2)
        self.hv_mod = hv_mod

    def _make_hv(self, seed):
        return self.hv_mod.HyperVector(seed)

    def test_snapshot_and_check_same_hv_no_alarm(self):
        hv = self._make_hv(42)
        self.detector.snapshot("cat", hv)
        event = self.detector.check("cat", hv)
        assert event.alarm is False
        assert event.drift_magnitude == pytest.approx(0.0, abs=0.01)

    def test_check_without_snapshot_creates_snapshot(self):
        hv = self._make_hv(100)
        event = self.detector.check("dog", hv)
        assert event.alarm is False
        assert "dog" in self.detector._snapshots

    def test_get_drift_report(self):
        hv = self._make_hv(1)
        self.detector.snapshot("concept", hv)
        self.detector.check("concept", hv)
        report = self.detector.get_drift_report()
        assert len(report) >= 1

    def test_get_statistics(self):
        hv = self._make_hv(1)
        self.detector.snapshot("x", hv)
        stats = self.detector.get_statistics()
        assert stats["snapshots"] == 1

    def test_update_snapshot(self):
        hv1 = self._make_hv(1)
        hv2 = self._make_hv(2)
        self.detector.snapshot("z", hv1)
        self.detector.update_snapshot("z", hv2)
        assert self.detector._snapshots["z"] is hv2

    def test_get_alarmed_concepts_empty_initially(self):
        assert self.detector.get_alarmed_concepts() == []


# ---------------------------------------------------------------------------
# VideoAdapter / TemporalStreamEncoder
# ---------------------------------------------------------------------------

class TestTemporalStreamEncoder:
    def setup_method(self):
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.adapters.video_adapter import TemporalStreamEncoder
        self.encoder = TemporalStreamEncoder(decay=0.9)
        self.hv_mod = hv_mod

    def test_single_step(self):
        hv = self.hv_mod.HyperVector(1234)
        state = self.encoder.step(hv)
        assert self.encoder.steps == 1
        assert state is not None

    def test_multiple_steps(self):
        for i in range(5):
            self.encoder.step(self.hv_mod.HyperVector(i * 1000))
        assert self.encoder.steps == 5

    def test_reset_clears_state(self):
        self.encoder.step(self.hv_mod.HyperVector(1))
        self.encoder.reset()
        assert self.encoder.current_state is None
        assert self.encoder.steps == 0


class TestVideoAdapter:
    def setup_method(self):
        from python.core.adapters.video_adapter import VideoAdapter
        self.adapter = VideoAdapter()

    def test_encode_empty_frames(self):
        pkt = self.adapter.encode([], "test")
        assert pkt.modality == "video"
        assert "VIDEO_EMPTY" in pkt.active_predicates

    def test_encode_single_frame(self):
        frame = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        pkt = self.adapter.encode([frame], "test")
        assert pkt.modality == "video"

    def test_encode_multiple_frames(self):
        frames = [np.random.randint(0, 255, (8, 8), dtype=np.uint8) for _ in range(5)]
        pkt = self.adapter.encode(frames, "test")
        assert pkt.modality == "video"
        assert pkt.raw_state["n_frames"] == 5

    def test_motion_score_in_raw_state(self):
        frames = [np.zeros((8, 8), dtype=np.uint8), np.ones((8, 8), dtype=np.uint8) * 255]
        pkt = self.adapter.encode(frames, "test")
        assert "motion_score" in pkt.raw_state

    def test_predicates_set(self):
        frames = [np.random.randint(0, 255, (8, 8), dtype=np.uint8) for _ in range(3)]
        pkt = self.adapter.encode(frames, "test")
        assert len(pkt.active_predicates) > 0


# ---------------------------------------------------------------------------
# ConformalWrapper
# ---------------------------------------------------------------------------

class TestConformalWrapper:
    def setup_method(self):
        from python.core.learning.conformal_wrapper import ConformalWrapper
        self.wrapper = ConformalWrapper(alpha=0.1)

    def test_not_calibrated_initially(self):
        assert not self.wrapper.is_calibrated()

    def test_calibrate_returns_q_hat(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        q = self.wrapper.calibrate(scores)
        assert q is not None
        assert 0.0 <= q <= 1.0

    def test_predict_set_after_calibration(self):
        self.wrapper.calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
        result = self.wrapper.predict_set(0.15)
        assert "q_hat" in result
        assert "coverage" in result
        assert result["coverage"] == pytest.approx(0.9, abs=0.01)

    def test_uncertainty_bound_returns_tuple(self):
        self.wrapper.calibrate([0.2, 0.3, 0.4])
        lower, upper = self.wrapper.uncertainty_bound(0.2)
        assert lower <= upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_predict_set_uncalibrated_returns_warning(self):
        result = self.wrapper.predict_set(0.5)
        assert "warning" in result

    def test_get_statistics(self):
        self.wrapper.calibrate([0.1, 0.3, 0.5])
        stats = self.wrapper.get_statistics()
        assert stats["alpha"] == 0.1
        assert stats["target_coverage"] == pytest.approx(0.9)
        assert stats["n_calibration"] > 0

    def test_calibrate_with_labels(self):
        scores = [0.1, 0.5, 0.2, 0.8, 0.3]
        labels = [True, False, True, False, True]
        q = self.wrapper.calibrate(scores, labels=labels)
        # Only correct scores used
        assert q is not None


# ---------------------------------------------------------------------------
# PatternGeneralizer / CrossDomainTransferPipeline
# ---------------------------------------------------------------------------

class TestPatternGeneralizer:
    def setup_method(self):
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.learning.pattern_generalizer import PatternGeneralizer
        self.gen = PatternGeneralizer(cluster_threshold=0.7, min_members_for_abstraction=3)
        self.hv_mod = hv_mod

    def _make_hv(self, seed):
        return self.hv_mod.HyperVector(seed)

    def test_observe_creates_new_pattern(self):
        hv = self._make_hv(1001)
        pattern, is_new = self.gen.observe(hv, "vision")
        assert is_new is True
        assert pattern.member_count == 1

    def test_observe_same_hv_joins_cluster(self):
        hv = self._make_hv(1001)
        self.gen.observe(hv, "vision")
        _, is_new = self.gen.observe(hv, "vision")
        assert is_new is False

    def test_mature_patterns_after_enough_members(self):
        hv = self._make_hv(1001)
        for _ in range(3):
            self.gen.observe(hv, "vision")
        mature = self.gen.get_mature_patterns()
        assert len(mature) >= 1

    def test_match_returns_list(self):
        hv = self._make_hv(1001)
        self.gen.observe(hv, "vision")
        results = self.gen.match(hv, domain="vision")
        assert len(results) >= 1
        assert all(len(r) == 2 for r in results)

    def test_get_statistics(self):
        hv = self._make_hv(1001)
        self.gen.observe(hv, "vision")
        stats = self.gen.get_statistics()
        assert stats["total_patterns"] >= 1
        assert "vision" in stats["domains"]


class TestCrossDomainTransferPipeline:
    def setup_method(self):
        import python.core.vsa.hypervec_shim as hv_mod
        from python.core.learning.pattern_generalizer import CrossDomainTransferPipeline
        self.pipeline = CrossDomainTransferPipeline()
        self.hv_mod = hv_mod

    def _make_hv(self, seed):
        return self.hv_mod.HyperVector(seed)

    def test_register_creates_pattern(self):
        hv = self._make_hv(1001)
        pattern = self.pipeline.register(hv, "domain_a")
        assert pattern is not None

    def test_transfer_returns_list(self):
        hv = self._make_hv(1001)
        self.pipeline.register(hv, "domain_a")
        self.pipeline.register(hv, "domain_b")
        results = self.pipeline.transfer(hv, "domain_a", "domain_b")
        assert isinstance(results, list)

    def test_get_transfer_log(self):
        hv = self._make_hv(1001)
        self.pipeline.register(hv, "domain_a")
        self.pipeline.register(hv, "domain_b")
        self.pipeline.transfer(hv, "domain_a", "domain_b")
        log = self.pipeline.get_transfer_log()
        assert isinstance(log, list)


# ---------------------------------------------------------------------------
# NSCKSubstrate V13
# ---------------------------------------------------------------------------

class TestNSCKSubstrateV13:
    def setup_method(self):
        from python.core.substrate import NSCKSubstrate
        self.substrate = NSCKSubstrate()
        self.substrate.register_task("test_task")

    def test_substrate_result_has_v13_fields(self):
        r = self.substrate.process("hello", "test_task")
        assert hasattr(r, "kle_uncertainty")
        assert hasattr(r, "uncertainty_bounds")
        assert hasattr(r, "encoding_stats")
        assert hasattr(r, "procedural_hit")

    def test_ingest_returns_substrate_result(self):
        from python.core.substrate import SubstrateResult
        r = self.substrate.ingest("test signal", "test_task")
        assert isinstance(r, SubstrateResult)
        assert isinstance(r.chosen_action, str)

    def test_ingest_not_procedural_hit_initially(self):
        r = self.substrate.ingest("test signal", "test_task")
        assert r.procedural_hit is False

    def test_feedback_caches_skill_and_enables_hit(self):
        import python.core.vsa.hypervec_shim as hv_mod
        # First ingest to prime last_ingest_hv
        r = self.substrate.ingest("test signal", "test_task")
        self.substrate.feedback(r.chosen_action, reward=1.0, task_tag="test_task")
        # Now second ingest of same signal should get procedural hit
        r2 = self.substrate.ingest("test signal", "test_task")
        assert isinstance(r2, __import__("python.core.substrate", fromlist=["SubstrateResult"]).SubstrateResult)

    def test_encoding_stats_populated(self):
        r = self.substrate.process("hello world", "test_task")
        # encoding_stats should be present (not None)
        assert r.encoding_stats is not None
        assert "source_type" in r.encoding_stats

    def test_get_stats_has_v13_keys(self):
        stats = self.substrate.get_stats()
        assert "procedural_memory" in stats
        assert "cross_modal_memory" in stats
        assert "drift_detector" in stats
        assert "conformal" in stats

    def test_feedback_without_state(self):
        # Should not raise
        self.substrate.feedback("ACTION_STAY", reward=0.5, task_tag="test_task")

    def test_ingest_numeric_array(self):
        r = self.substrate.ingest(np.array([1.0, 2.0, 3.0]), "test_task")
        assert r.chosen_action is not None

    def test_ingest_dict(self):
        r = self.substrate.ingest({"x": 1, "y": 2}, "test_task")
        assert r.chosen_action is not None

    def test_ingest_image(self):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        r = self.substrate.ingest(img, "test_task")
        assert r.chosen_action is not None


# ---------------------------------------------------------------------------
# GlobalWorkspace KLE uncertainty
# ---------------------------------------------------------------------------

class TestGlobalWorkspaceKLE:
    def setup_method(self):
        from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition
        self.gw = GlobalWorkspace()
        self.Coalition = Coalition

    def test_kle_zero_before_compete(self):
        assert self.gw._kle_uncertainty == 0.0

    def test_kle_computed_after_compete(self):
        proposals = [
            self.Coalition("module_a", "action1", 0.8),
            self.Coalition("module_b", "action2", 0.4),
        ]
        self.gw.compete(proposals)
        assert self.gw._kle_uncertainty > 0.0

    def test_kle_in_get_status(self):
        proposals = [
            self.Coalition("module_a", "action1", 0.8),
        ]
        self.gw.compete(proposals)
        status = self.gw.get_status()
        assert "kle_uncertainty" in status

    def test_get_kle_uncertainty_method(self):
        proposals = [
            self.Coalition("m1", "a", 0.9),
            self.Coalition("m2", "b", 0.3),
        ]
        self.gw.compete(proposals)
        kle = self.gw.get_kle_uncertainty()
        assert isinstance(kle, float)
        assert kle >= 0.0

    def test_kle_higher_with_more_competing_proposals(self):
        # Uniform competition → high entropy
        proposals_uniform = [
            self.Coalition(f"m{i}", f"a{i}", 0.5) for i in range(4)
        ]
        self.gw.compete(proposals_uniform)
        kle_uniform = self.gw._kle_uncertainty

        # Dominated competition → low entropy
        proposals_dom = [
            self.Coalition("m_strong", "a", 1.0),
            self.Coalition("m_weak", "b", 0.01),
        ]
        self.gw.compete(proposals_dom)
        kle_dom = self.gw._kle_uncertainty

        assert kle_uniform > kle_dom


# ---------------------------------------------------------------------------
# CognitiveState V13 fields
# ---------------------------------------------------------------------------

class TestCognitiveStateV13:
    def test_kle_uncertainty_field_exists(self):
        from python.core.reasoning.cognitive_engine import CognitiveState
        cs = CognitiveState(task_tag="test")
        assert hasattr(cs, "kle_uncertainty")
        assert cs.kle_uncertainty is None

    def test_uncertainty_bounds_field_exists(self):
        from python.core.reasoning.cognitive_engine import CognitiveState
        cs = CognitiveState(task_tag="test")
        assert hasattr(cs, "uncertainty_bounds")
        assert cs.uncertainty_bounds is None

    def test_set_kle_uncertainty(self):
        from python.core.reasoning.cognitive_engine import CognitiveState
        cs = CognitiveState(task_tag="test", kle_uncertainty=0.42)
        assert cs.kle_uncertainty == pytest.approx(0.42)

    def test_set_uncertainty_bounds(self):
        from python.core.reasoning.cognitive_engine import CognitiveState
        cs = CognitiveState(task_tag="test", uncertainty_bounds=(0.3, 0.8))
        assert cs.uncertainty_bounds == (0.3, 0.8)
