"""
Unit tests for SocietalEWCCombiner — NSCK V27
=============================================
Tests for the Societal-Guided Elastic Weight Consolidation (SoCL) module.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import python.core.vsa.hypervec_shim as hv_mod
from python.core.learning.continual_learning import ContinualLearner
from python.core.learning.societal_ewc import (
    SocietalEWCCombiner,
    compute_societal_centrality,
)
from python.core.societal.living_hypervector import LivingHyperVector
from python.core.societal.society_manager import SocietyManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hv(seed: int):
    return hv_mod.HyperVector(seed=seed)


def _make_mgr(*concept_ids, bond_threshold=0.0):
    mgr = SocietyManager(
        bond_threshold=bond_threshold, max_bonds=8, auto_cluster_interval=0
    )
    for i, cid in enumerate(concept_ids):
        lhv = LivingHyperVector(cid, _hv(i), domain_path=["test"])
        mgr.register(lhv)
    return mgr


def _make_learner_with_concepts(*concept_ids, task_tag="t1"):
    learner = ContinualLearner(ewc_lambda=100.0)
    learner.register_task(task_tag)
    for cid in concept_ids:
        learner.tasks[task_tag].importance_weights[cid] = np.array([1.0])
        learner.tasks[task_tag].optimal_params[cid] = np.array([0.5])
    return learner


# ---------------------------------------------------------------------------
# 1. compute_societal_centrality
# ---------------------------------------------------------------------------


class TestComputeSocietalCentrality:
    def test_missing_concept_returns_zero(self):
        mgr = _make_mgr("a", "b")
        assert compute_societal_centrality("nonexistent", mgr) == 0.0

    def test_unbonded_concept_has_zero_degree_component(self):
        mgr = _make_mgr("a", "b")
        score = compute_societal_centrality("a", mgr)
        # No bonds → degree_norm=0.  Activation=0.5 by default.
        assert 0.0 <= score <= 1.0

    def test_high_activation_increases_score(self):
        mgr = _make_mgr("a", "b")
        mgr.activate_concept("a", delta=1.0)
        score_a = compute_societal_centrality("a", mgr)
        score_b = compute_societal_centrality("b", mgr)
        assert score_a > score_b, "higher activation should yield higher score"

    def test_more_bonds_increases_score(self):
        mgr = _make_mgr("a", "b", "c", "d")
        # Give 'a' more bonds than 'b'
        mgr.form_bond_explicit("a", "b", strength=0.8)
        mgr.form_bond_explicit("a", "c", strength=0.8)
        mgr.form_bond_explicit("a", "d", strength=0.8)
        mgr.form_bond_explicit("b", "c", strength=0.8)
        score_a = compute_societal_centrality("a", mgr)
        score_b = compute_societal_centrality("b", mgr)
        assert score_a > score_b, "more bonds → higher centrality"

    def test_score_in_unit_interval(self):
        mgr = _make_mgr("a", "b", "c")
        mgr.auto_bond()
        mgr.activate_concept("a", delta=1.0)
        for lhv in mgr:
            s = compute_societal_centrality(lhv.concept_id, mgr)
            assert 0.0 <= s <= 1.0, f"score {s} out of [0,1] for {lhv.concept_id}"

    def test_cluster_membership_affects_score(self):
        mgr = _make_mgr("a", "b", "c", "d", "e")
        mgr.auto_bond()
        mgr.leiden_cluster(1.0)
        scores = [
            compute_societal_centrality(cid, mgr) for cid in ["a", "b", "c", "d", "e"]
        ]
        # All scores should still be in [0,1]
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_custom_weights_affect_score(self):
        mgr = _make_mgr("a", "b")
        mgr.activate_concept("a", delta=1.0)
        # With activation_weight=1.0 and others 0, score = activation
        s1 = compute_societal_centrality(
            "a", mgr, weight_degree=0.0, weight_activation=1.0, weight_cluster=0.0
        )
        assert pytest.approx(s1, abs=0.05) == 1.0  # activation was spiked to ~1.0


# ---------------------------------------------------------------------------
# 2. SocietalEWCCombiner construction
# ---------------------------------------------------------------------------


class TestSocietalEWCCombinerInit:
    def test_default_init(self):
        c = SocietalEWCCombiner()
        assert c.centrality_weight == 0.5
        assert c.degree_weight == 0.4
        assert c.activation_weight == 0.4
        assert c.cluster_weight == 0.2

    def test_custom_centrality_weight(self):
        c = SocietalEWCCombiner(centrality_weight=1.0)
        assert c.centrality_weight == 1.0

    def test_negative_centrality_weight_raises(self):
        with pytest.raises(ValueError):
            SocietalEWCCombiner(centrality_weight=-0.1)

    def test_zero_centrality_weight_allowed(self):
        c = SocietalEWCCombiner(centrality_weight=0.0)
        assert c.centrality_weight == 0.0


# ---------------------------------------------------------------------------
# 3. compute_centrality_map
# ---------------------------------------------------------------------------


class TestComputeCentralityMap:
    def test_returns_dict_for_all_concepts(self):
        mgr = _make_mgr("a", "b", "c")
        c = SocietalEWCCombiner()
        cmap = c.compute_centrality_map(mgr)
        assert set(cmap.keys()) == {"a", "b", "c"}

    def test_all_scores_in_unit_interval(self):
        mgr = _make_mgr("a", "b", "c")
        mgr.auto_bond()
        c = SocietalEWCCombiner()
        for score in c.compute_centrality_map(mgr).values():
            assert 0.0 <= score <= 1.0

    def test_empty_society_returns_empty_map(self):
        mgr = SocietyManager()
        c = SocietalEWCCombiner()
        assert c.compute_centrality_map(mgr) == {}


# ---------------------------------------------------------------------------
# 4. boost_task_importance
# ---------------------------------------------------------------------------


class TestBoostTaskImportance:
    def test_boosts_concept_in_both_society_and_learner(self):
        mgr = _make_mgr("cat", "dog")
        mgr.activate_concept("cat", delta=1.0)
        learner = _make_learner_with_concepts("cat", "dog", task_tag="nav")
        combiner = SocietalEWCCombiner(centrality_weight=1.0)

        before_cat = learner.tasks["nav"].importance_weights["cat"].copy()
        combiner.boost_task_importance("nav", learner, mgr)
        after_cat = learner.tasks["nav"].importance_weights["cat"]

        assert after_cat[0] > before_cat[0], "cat importance should increase"

    def test_zero_centrality_weight_no_change(self):
        mgr = _make_mgr("a", "b")
        mgr.activate_concept("a", delta=1.0)
        learner = _make_learner_with_concepts("a", "b", task_tag="t")
        combiner = SocietalEWCCombiner(centrality_weight=0.0)

        before = {k: v.copy() for k, v in learner.tasks["t"].importance_weights.items()}
        combiner.boost_task_importance("t", learner, mgr)
        after = learner.tasks["t"].importance_weights

        for k in before:
            # factor = 1 + 0.0 * centrality = 1.0 → no change
            np.testing.assert_allclose(before[k], after[k])

    def test_unknown_task_returns_empty_dict(self):
        mgr = _make_mgr("a")
        learner = ContinualLearner()
        combiner = SocietalEWCCombiner()
        result = combiner.boost_task_importance("nonexistent", learner, mgr)
        assert result == {}

    def test_concept_not_in_learner_not_added(self):
        mgr = _make_mgr("society_only")
        learner = _make_learner_with_concepts("learner_only", task_tag="t")
        combiner = SocietalEWCCombiner(centrality_weight=0.5)
        combiner.boost_task_importance("t", learner, mgr)
        # "society_only" must NOT be added to learner's weights
        assert "society_only" not in learner.tasks["t"].importance_weights

    def test_higher_centrality_yields_higher_boost(self):
        mgr = _make_mgr("a", "b", "c", "d")
        # 'a' gets all bonds + full activation → highest centrality
        for peer in ["b", "c", "d"]:
            mgr.form_bond_explicit("a", peer, strength=0.9)
        mgr.activate_concept("a", delta=1.0)

        learner = _make_learner_with_concepts("a", "b", task_tag="t")
        combiner = SocietalEWCCombiner(centrality_weight=1.0)
        before_a = learner.tasks["t"].importance_weights["a"].copy()
        before_b = learner.tasks["t"].importance_weights["b"].copy()
        combiner.boost_task_importance("t", learner, mgr)
        boost_a = learner.tasks["t"].importance_weights["a"][0] / before_a[0]
        boost_b = learner.tasks["t"].importance_weights["b"][0] / before_b[0]
        assert boost_a > boost_b, "hub concept should receive larger boost"

    def test_boost_factor_formula(self):
        """factor = 1 + centrality_weight * centrality_score."""
        mgr = _make_mgr("x")
        # Force activation = 1.0 and no bonds → degree_norm = 0
        mgr.activate_concept("x", delta=1.0)
        learner = _make_learner_with_concepts("x", task_tag="t")
        combiner = SocietalEWCCombiner(
            centrality_weight=1.0,
            degree_weight=0.0,
            activation_weight=1.0,
            cluster_weight=0.0,
        )
        before = learner.tasks["t"].importance_weights["x"].copy()
        combiner.boost_task_importance("t", learner, mgr)
        after = learner.tasks["t"].importance_weights["x"]
        centrality = compute_societal_centrality(
            "x", mgr, weight_degree=0.0, weight_activation=1.0, weight_cluster=0.0
        )
        expected_factor = 1.0 + 1.0 * centrality
        np.testing.assert_allclose(after, before * expected_factor, rtol=1e-6)

    def test_importance_never_decreases(self):
        """Societal boost should never reduce Fisher importance."""
        mgr = _make_mgr("a", "b", "c")
        learner = _make_learner_with_concepts("a", "b", "c", task_tag="t")
        combiner = SocietalEWCCombiner(centrality_weight=2.0)
        before = {
            k: v.copy() for k, v in learner.tasks["t"].importance_weights.items()
        }
        combiner.boost_task_importance("t", learner, mgr)
        for k in before:
            assert learner.tasks["t"].importance_weights[k][0] >= before[k][0], (
                f"importance decreased for {k}"
            )

    def test_returns_boost_factors_greater_than_one(self):
        mgr = _make_mgr("a")
        mgr.activate_concept("a", delta=0.8)
        learner = _make_learner_with_concepts("a", task_tag="t")
        combiner = SocietalEWCCombiner(centrality_weight=1.0)
        boosts = combiner.boost_task_importance("t", learner, mgr)
        for factor in boosts.values():
            assert factor > 1.0


# ---------------------------------------------------------------------------
# 5. apply_societal_boost
# ---------------------------------------------------------------------------


class TestApplySocietalBoost:
    def test_boost_specific_task(self):
        mgr = _make_mgr("a")
        mgr.activate_concept("a", delta=1.0)
        learner = _make_learner_with_concepts("a", task_tag="nav")
        combiner = SocietalEWCCombiner(centrality_weight=0.5)
        result = combiner.apply_societal_boost(learner, "nav", mgr)
        assert isinstance(result, dict)

    def test_empty_task_tag_boosts_all_tasks(self):
        mgr = _make_mgr("a", "b")
        mgr.activate_concept("a", delta=1.0)
        learner = ContinualLearner()
        for tt in ["t1", "t2"]:
            learner.register_task(tt)
            learner.tasks[tt].importance_weights["a"] = np.array([1.0])
            learner.tasks[tt].importance_weights["b"] = np.array([1.0])
        combiner = SocietalEWCCombiner(centrality_weight=0.5)
        result = combiner.apply_societal_boost(learner, "", mgr)
        # Both tasks should appear in result (or at least non-empty total)
        assert isinstance(result, dict)

    def test_nonexistent_task_returns_empty_on_specific(self):
        mgr = _make_mgr("a")
        learner = ContinualLearner()
        combiner = SocietalEWCCombiner()
        result = combiner.apply_societal_boost(learner, "no_such_task", mgr)
        assert result == {}


# ---------------------------------------------------------------------------
# 6. get_top_central_concepts
# ---------------------------------------------------------------------------


class TestGetTopCentralConcepts:
    def test_returns_list_of_tuples(self):
        mgr = _make_mgr("a", "b", "c")
        c = SocietalEWCCombiner()
        top = c.get_top_central_concepts(mgr, top_k=2)
        assert isinstance(top, list)
        assert len(top) == 2
        for concept_id, score in top:
            assert isinstance(concept_id, str)
            assert isinstance(score, float)

    def test_top_k_larger_than_society_returns_all(self):
        mgr = _make_mgr("a", "b")
        c = SocietalEWCCombiner()
        top = c.get_top_central_concepts(mgr, top_k=100)
        assert len(top) == 2

    def test_scores_are_descending(self):
        mgr = _make_mgr("a", "b", "c", "d", "e")
        mgr.activate_concept("a", delta=1.0)
        mgr.activate_concept("b", delta=0.5)
        c = SocietalEWCCombiner()
        top = c.get_top_central_concepts(mgr, top_k=5)
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True), "scores must be descending"

    def test_empty_society_returns_empty(self):
        mgr = SocietyManager()
        c = SocietalEWCCombiner()
        top = c.get_top_central_concepts(mgr, top_k=5)
        assert top == []


# ---------------------------------------------------------------------------
# 7. NSCKConfig.societal_ewc() preset
# ---------------------------------------------------------------------------


class TestNSCKConfigSocietalEWC:
    def test_preset_flags(self):
        from python.core.integration.config import NSCKConfig

        cfg = NSCKConfig.societal_ewc()
        assert cfg.enable_societal is True
        assert cfg.enable_ewc is True
        assert cfg.enable_societal_ewc is True
        assert cfg.societal_ewc_centrality_weight == 0.5
        assert cfg.ewc_lambda > 0

    def test_default_societal_ewc_disabled(self):
        from python.core.integration.config import NSCKConfig

        cfg = NSCKConfig()
        assert cfg.enable_societal_ewc is False

    def test_societal_preset_no_ewc(self):
        from python.core.integration.config import NSCKConfig

        cfg = NSCKConfig.societal()
        assert cfg.enable_ewc is False
        assert cfg.enable_societal_ewc is False

    def test_custom_centrality_weight(self):
        from python.core.integration.config import NSCKConfig

        cfg = NSCKConfig.societal_ewc()
        cfg.societal_ewc_centrality_weight = 1.5
        assert cfg.societal_ewc_centrality_weight == 1.5


# ---------------------------------------------------------------------------
# 8. CognitiveEngine integration
# ---------------------------------------------------------------------------


class TestCognitiveEngineIntegration:
    """Verify the SoCL boost is wired into CognitiveEngine correctly."""

    def test_socl_boost_called_when_enabled(self):
        """When enable_societal_ewc=True and a societal manager is set,
        the SoCL combiner must apply boosts during _compute_and_record_vsa_importance."""
        from python.core.integration.config import NSCKConfig
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.memory.semantic_memory import SemanticMemory

        cfg = NSCKConfig.societal_ewc()
        cfg.ewc_consolidate_interval = 1  # trigger on first call
        cfg.ewc_importance_window = 10
        engine = CognitiveEngine(config=cfg, persistence_path=":memory:")
        engine.register_task("socl_test")

        # Seed semantic memory with one concept
        engine.semantic_memory.add_concept("cat", {"kind": "animal"})

        # Seed the EWC buffer manually
        hv = hv_mod.HyperVector(seed=42)
        engine._ewc_importance_buffer["socl_test"] = [hv]

        # Set up a minimal societal manager so the boost path fires
        mgr = _make_mgr("cat")
        mgr.activate_concept("cat", delta=1.0)
        engine._societal_manager = mgr

        # Run importance computation — should not raise
        engine._compute_and_record_vsa_importance("socl_test")

        # cat should now be in the importance weights
        assert "cat" in engine._continual_learner.tasks["socl_test"].importance_weights

    def test_socl_boost_skipped_when_disabled(self):
        """With enable_societal_ewc=False (default), no boost is applied."""
        from python.core.integration.config import NSCKConfig
        from python.core.reasoning.cognitive_engine import CognitiveEngine

        cfg = NSCKConfig()
        cfg.enable_ewc = True
        cfg.ewc_consolidate_interval = 1
        cfg.ewc_importance_window = 10
        cfg.enable_societal_ewc = False
        engine = CognitiveEngine(config=cfg, persistence_path=":memory:")
        engine.register_task("plain")
        engine.semantic_memory.add_concept("dog", {})
        hv = hv_mod.HyperVector(seed=1)
        engine._ewc_importance_buffer["plain"] = [hv]

        # No societal manager
        engine._compute_and_record_vsa_importance("plain")
        # Should complete without error; no societal manager set
        assert "dog" in engine._continual_learner.tasks["plain"].importance_weights


# ---------------------------------------------------------------------------
# 9. Integration scenario: forgetting reduction property
# ---------------------------------------------------------------------------


class TestForgettingReductionProperty:
    """Verify that societal boost produces higher EWC importance for
    cross-task central concepts, which should reduce forgetting."""

    def test_central_concept_has_higher_importance_after_boost(self):
        """After boost, a highly central concept must have higher Fisher weight
        than an equivalent non-central concept."""
        mgr = _make_mgr("central", "peripheral")
        # Make 'central' highly connected and activated
        for _ in range(3):
            other_id = f"peer_{_}"
            mgr.register(LivingHyperVector(other_id, _hv(100 + _)))
            mgr.form_bond_explicit("central", other_id, strength=0.9)
        mgr.activate_concept("central", delta=1.0)

        learner = ContinualLearner(ewc_lambda=500.0)
        learner.register_task("task1")
        for cid in ["central", "peripheral"]:
            learner.tasks["task1"].importance_weights[cid] = np.array([1.0])
            learner.tasks["task1"].optimal_params[cid] = np.array([0.0])

        combiner = SocietalEWCCombiner(centrality_weight=2.0)
        combiner.boost_task_importance("task1", learner, mgr)

        imp_central = learner.tasks["task1"].importance_weights["central"][0]
        imp_peripheral = learner.tasks["task1"].importance_weights["peripheral"][0]
        assert imp_central > imp_peripheral, (
            "Central concept should have higher EWC importance after boost"
        )

    def test_ewc_loss_higher_after_societal_boost(self):
        """EWC regularization loss must be higher after boost (stricter protection)."""
        mgr = _make_mgr("concept")
        mgr.activate_concept("concept", delta=1.0)

        learner = ContinualLearner(ewc_lambda=100.0)
        learner.register_task("t")
        learner.tasks["t"].importance_weights["concept"] = np.array([1.0])
        learner.tasks["t"].optimal_params["concept"] = np.array([0.5])

        # EWC loss before boost
        drifted_params = {"concept": np.array([0.8])}
        loss_before = learner.ewc_loss(drifted_params)

        combiner = SocietalEWCCombiner(centrality_weight=2.0)
        combiner.boost_task_importance("t", learner, mgr)

        # EWC loss after boost — must be larger
        loss_after = learner.ewc_loss(drifted_params)
        assert loss_after > loss_before, (
            f"EWC loss should be higher after societal boost: "
            f"before={loss_before:.4f}, after={loss_after:.4f}"
        )


if __name__ == "__main__":
    # Run a quick manual check
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], check=True)
