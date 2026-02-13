"""
Smoke tests for the rigorous transfer experiment infrastructure.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from benchmark import (
    SymbolicHasher, InvertedCatcherEnv, EasyCatcherEnv, HardCatcherEnv,
    RandomAgent, ENVS, TeacherAgent, LearnedPolicy, StudentAgent,
)
from transfer_experiments import (
    run_training, run_testing, run_random_baseline,
    exp1_transfer_matrix, exp2_significance, exp3_scaling,
    exp4_ablation, exp5_negative_transfer, exp6_curriculum, exp7_continual,
    exp4_ablation, exp5_negative_transfer, exp6_curriculum, exp7_continual,
    exp8_language_transfer, exp9_cognitive_transfer,
    welch_ttest,
)


class TestSymbolicHasher:
    """Tests for configurable predicate hashing."""

    def test_default_hash_backward_compatible(self):
        state = {"object_x": 0.3, "object_vel": 0.01}
        h = SymbolicHasher.default_hash("balancer", state)
        assert h.startswith("physics:")

    def test_custom_bins(self):
        hasher_coarse = SymbolicHasher(bins=2)
        hasher_fine = SymbolicHasher(bins=20)
        state = {"object_x": 0.33, "object_vel": 0.01}
        h_coarse = hasher_coarse.hash("balancer", state)
        h_fine = hasher_fine.hash("balancer", state)
        # Fine should produce more distinct values
        assert h_coarse != h_fine

    def test_predicate_removal(self):
        full = SymbolicHasher()
        no_vel = SymbolicHasher(predicates={"position", "danger"})
        state = {"object_x": 0.3, "object_vel": 0.01}
        h_full = full.hash("balancer", state)
        h_no_vel = no_vel.hash("balancer", state)
        assert len(h_full.split(":")) > len(h_no_vel.split(":"))

    def test_empty_predicates(self):
        hasher = SymbolicHasher(predicates=set())
        state = {"object_x": 0.3, "object_vel": 0.01}
        h = hasher.hash("balancer", state)
        assert h == "physics"  # only prefix, no predicate parts

    def test_invert_flips_signal(self):
        normal = SymbolicHasher()
        inverted = SymbolicHasher(invert=True)
        state = {"object_x": 0.5, "object_vel": 0.01}
        h_normal = normal.hash("balancer", state)
        h_inverted = inverted.hash("balancer", state)
        assert h_normal != h_inverted

    def test_cross_game_same_hash(self):
        hasher = SymbolicHasher()
        state = {"object_x": 0.3, "object_vel": 0.01}
        h_bal = hasher.hash("balancer", state)
        h_cat = hasher.hash("catcher", state)
        assert h_bal == h_cat


class TestGameVariants:
    """Tests for InvertedCatcher, EasyCatcher, HardCatcher."""

    def test_inverted_catcher_flips(self):
        env = InvertedCatcherEnv()
        state = env.reset()
        assert state["type"] == "inverted_catcher"

    def test_easy_catcher_runs(self):
        env = EasyCatcherEnv()
        state = env.reset()
        assert state["type"] == "catcher"

    def test_hard_catcher_runs(self):
        env = HardCatcherEnv()
        state = env.reset()
        assert state["type"] == "catcher"

    def test_variants_in_envs(self):
        assert "inverted_catcher" in ENVS
        assert "easy_catcher" in ENVS
        assert "hard_catcher" in ENVS

    def test_random_agent(self):
        agent = RandomAgent()
        a1 = agent.choose_action("snake", {})
        assert a1 in ["UP", "DOWN", "LEFT", "RIGHT"]
        a2 = agent.choose_action("balancer", {})
        assert a2 in ["TILT_LEFT", "TILT_RIGHT", "HOLD"]


class TestCoreRunner:
    """Tests for run_training/run_testing/run_random_baseline."""

    def test_run_training_returns_policy(self):
        policy, avg = run_training("balancer", 5)
        assert policy.size() > 0
        assert avg >= 0

    def test_run_testing_returns_scores(self):
        policy, _ = run_training("balancer", 5)
        scores = run_testing("catcher", policy, 5)
        assert len(scores) == 5

    def test_run_random_baseline(self):
        scores = run_random_baseline("catcher", 5)
        assert len(scores) == 5


class TestLanguageTeacher:
    """Tests for natural language parsing."""

    def test_parse_move_left(self):
        from benchmark import NaturalLanguageTeacher
        teacher = NaturalLanguageTeacher()
        teacher.teach("If object is to the left then move left")
        # should produce rule: obj_x < -0.05 -> TILT_LEFT
        state = {"object_x": -0.2}
        action = teacher.choose_action("balancer", state)
        assert action == "TILT_LEFT"

    def test_parse_avoid_danger(self):
        from benchmark import NaturalLanguageTeacher
        teacher = NaturalLanguageTeacher()
        teacher.teach("Avoid danger on the left")
        # "danger" implies object_type="danger".
        # Balancer: Avoid (Dump) Left (-x) -> TILT_LEFT
        state = {"object_x": -0.8, "object_type": "danger"}
        action = teacher.choose_action("balancer", state)
        assert action == "TILT_LEFT"

    def test_synonyms(self):
        from benchmark import NaturalLanguageTeacher
        teacher = NaturalLanguageTeacher()
        teacher.teach("If ball is left go left")
        state = {"ball_pos": -0.2}
        action = teacher.choose_action("balancer", state)
        assert action == "TILT_LEFT" 



class TestExperiments:
    """Smoke tests for each experiment (very small scale)."""

    def test_exp1_matrix_small(self):
        # Only test 2 pairs for speed
        results = exp1_transfer_matrix(train_eps=5, test_eps=5)
        assert len(results) > 0
        for key, val in results.items():
            assert "transfer_avg" in val
            assert "random_avg" in val

    def test_exp2_significance_small(self):
        results = exp2_significance(
            pairs=[("balancer", "catcher")], runs=2, train_eps=5, test_eps=5)
        assert "balancer->catcher" in results
        assert "p_value" in results["balancer->catcher"]

    def test_exp3_scaling_small(self):
        results = exp3_scaling(steps=[5, 10], test_eps=5)
        assert 5 in results
        assert 10 in results

    def test_exp4_ablation_small(self):
        results = exp4_ablation(train_eps=5, test_eps=5)
        assert "predicate_ablation" in results
        assert "bin_resolution" in results
        assert "full" in results["predicate_ablation"]

    def test_exp5_negative_small(self):
        results = exp5_negative_transfer(train_eps=5, test_eps=5)
        assert "adversarial" in results
        assert "null_transfer" in results
        assert "incompatible" in results

    def test_exp6_curriculum_small(self):
        results = exp6_curriculum(test_eps=5)
        assert "grid_curriculum" in results
        assert "physics_curriculum" in results

    def test_welch_ttest(self):
        a = [1, 2, 3, 4, 5]
        b = [0, 1, 2, 3, 4]
        t, p, d = welch_ttest(a, b)
        assert isinstance(t, float)
        assert 0 <= p <= 1

    def test_exp7_continual_small(self):
        results = exp7_continual(train_eps=5, test_eps=5)
        assert "score_a_initial" in results
        assert "score_b" in results
        assert "score_a_final" in results
        assert "retention_pct" in results

    def test_exp8_language_small(self):
        results = exp8_language_transfer(test_eps=5)
        assert "balancer_zero_shot" in results
        assert "catcher_transfer" in results

    def test_exp9_cognitive_small(self):
        results = exp9_cognitive_transfer(test_eps=5)
        assert "catcher_score" in results
        assert "balancer_score" in results

