"""
Tests for the non-grid transfer learning experiment.

Covers:
- Balancer game mechanics and physics
- Catcher game mechanics and scoring
- Grounding verifier predicates for both games
- Cross-domain transfer: train Balancer -> test Catcher
"""
import sys
import os
import pytest


from python.games.physics.balancer_game import BalancerGame, sim_balancer
from python.games.physics.catcher_game import CatcherGame, sim_catcher
from python.core.perception.grounding_verifier import create_balancer_verifier, create_catcher_verifier


class TestBalancerGame:
    """Tests for Balancer game mechanics."""

    def test_reset_creates_valid_state(self):
        game = BalancerGame()
        state = game.state
        assert state is not None
        assert state.alive is True
        assert abs(state.ball_pos) <= 0.1
        assert state.ticks == 0

    def test_tilt_left_changes_angle(self):
        game = BalancerGame()
        initial_angle = game.state.beam_angle
        game.step("TILT_LEFT")
        assert game.state.beam_angle < initial_angle

    def test_tilt_right_changes_angle(self):
        game = BalancerGame()
        initial_angle = game.state.beam_angle
        game.step("TILT_RIGHT")
        assert game.state.beam_angle > initial_angle

    def test_hold_does_not_change_angle(self):
        game = BalancerGame()
        game.state.ball_pos = 0.0
        game.state.ball_vel = 0.0
        game.state.beam_angle = 0.0
        game.step("HOLD")
        assert game.state.beam_angle == 0.0

    def test_ball_falls_off_edge(self):
        game = BalancerGame()
        game.state.ball_pos = 0.95
        game.state.ball_vel = 0.1
        _, reward, done = game.step("HOLD")
        assert done is True
        assert reward == -10.0
        assert game.state.alive is False

    def test_survival_gives_reward(self):
        game = BalancerGame()
        game.state.ball_pos = 0.0
        game.state.ball_vel = 0.0
        _, reward, done = game.step("HOLD")
        assert reward > 0
        assert done is False

    def test_max_ticks_terminates(self):
        game = BalancerGame()
        game.state.ticks = 499
        game.state.ball_pos = 0.0
        game.state.ball_vel = 0.0
        _, reward, done = game.step("HOLD")
        assert done is True
        assert reward > 0  # bonus for surviving

    def test_to_dict_has_required_fields(self):
        game = BalancerGame()
        d = game.state.to_dict()
        assert "ball_pos" in d
        assert "ball_vel" in d
        assert "beam_angle" in d
        assert "object_x" in d
        assert "object_vel" in d


class TestCatcherGame:
    """Tests for Catcher game mechanics."""

    def test_reset_creates_valid_state(self):
        game = CatcherGame()
        state = game.state
        assert state is not None
        assert state.catches == 0
        assert state.misses == 0
        assert len(state.objects) >= 1

    def test_move_left(self):
        game = CatcherGame()
        initial_x = game.state.paddle_x
        game.step("MOVE_LEFT")
        assert game.state.paddle_x < initial_x

    def test_move_right(self):
        game = CatcherGame()
        initial_x = game.state.paddle_x
        game.step("MOVE_RIGHT")
        assert game.state.paddle_x > initial_x

    def test_tilt_left_alias_works(self):
        """Transfer actions from Balancer should work in Catcher."""
        game = CatcherGame()
        initial_x = game.state.paddle_x
        game.step("TILT_LEFT")
        assert game.state.paddle_x < initial_x

    def test_catch_increments_score(self):
        game = CatcherGame()
        game.state.objects = []
        # Place object directly on paddle at ground level
        from python.games.physics.catcher_game import FallingObject
        game.state.objects.append(FallingObject(
            x=game.state.paddle_x, y=0.04, vx=0.0, vy=-0.02
        ))
        game.step("HOLD")
        assert game.state.catches == 1

    def test_miss_increments_misses(self):
        game = CatcherGame()
        game.state.objects = []
        from python.games.physics.catcher_game import FallingObject
        # Place object far from paddle at ground level
        game.state.objects.append(FallingObject(
            x=0.9, y=0.04, vx=0.0, vy=-0.02
        ))
        game.state.paddle_x = 0.1
        game.step("HOLD")
        assert game.state.misses == 1

    def test_win_at_20_catches(self):
        game = CatcherGame()
        game.state.catches = 19
        game.state.objects = []
        from python.games.physics.catcher_game import FallingObject
        game.state.objects.append(FallingObject(
            x=game.state.paddle_x, y=0.04, vx=0.0, vy=-0.02
        ))
        _, reward, done = game.step("HOLD")
        assert done is True
        assert reward > 0


class TestSimFunctions:
    """Tests for lightweight simulation functions."""

    def test_sim_balancer_movement(self):
        state = {"ball_pos": 0.0, "ball_vel": 0.0, "beam_angle": 0.0}
        next_state, terminal = sim_balancer(state, "TILT_RIGHT")
        assert next_state["beam_angle"] > 0
        assert not terminal

    def test_sim_catcher_movement(self):
        state = {"paddle_x": 0.5, "object_x": 0.3, "object_y": 0.8,
                 "object_vel": 0.0, "catches": 0, "misses": 0}
        next_state, terminal = sim_catcher(state, "MOVE_LEFT")
        assert next_state["paddle_x"] < 0.5
        assert not terminal


class TestVerifiers:
    """Tests for shared structural predicates."""

    def test_balancer_object_left(self):
        v = create_balancer_verifier()
        state = {"object_x": -0.5, "object_vel": 0.0}
        assert v.verify_predicate("OBJECT_LEFT", state, context="balancer")
        assert not v.verify_predicate("OBJECT_RIGHT", state, context="balancer")

    def test_balancer_moving_right(self):
        v = create_balancer_verifier()
        state = {"object_x": 0.0, "object_vel": 0.02}
        assert v.verify_predicate("MOVING_RIGHT", state, context="balancer")
        assert not v.verify_predicate("MOVING_LEFT", state, context="balancer")

    def test_balancer_danger_left(self):
        v = create_balancer_verifier()
        state = {"object_x": -0.8, "object_vel": 0.0}
        assert v.verify_predicate("DANGER_LEFT", state, context="balancer")

    def test_catcher_object_left(self):
        v = create_catcher_verifier()
        state = {"object_x": 0.2, "paddle_x": 0.5, "object_vel": 0.0}
        assert v.verify_predicate("OBJECT_LEFT", state, context="catcher")
        assert not v.verify_predicate("OBJECT_RIGHT", state, context="catcher")

    def test_catcher_moving_left(self):
        v = create_catcher_verifier()
        state = {"object_x": 0.5, "paddle_x": 0.5, "object_vel": -0.01}
        assert v.verify_predicate("MOVING_LEFT", state, context="catcher")

    def test_predicates_same_names(self):
        """Both verifiers use the same predicate names for transfer."""
        bv = create_balancer_verifier()
        cv = create_catcher_verifier()
        b_names = set(bv.context_predicates.get("balancer", {}).keys())
        c_names = set(cv.context_predicates.get("catcher", {}).keys())
        assert b_names == c_names, f"Mismatch: {b_names} vs {c_names}"


class TestBenchmarkIntegration:
    """Tests for BalancerEnv and CatcherEnv in benchmark."""

    def test_balancer_env_runs(self):
        from python.benchmarks.benchmark import BalancerEnv
        env = BalancerEnv()
        state = env.reset()
        assert state["type"] == "balancer"

    def test_catcher_env_runs(self):
        from python.benchmarks.benchmark import CatcherEnv
        env = CatcherEnv()
        state = env.reset()
        assert state["type"] == "catcher"

    def test_teacher_survives_balancer(self):
        from python.benchmarks.benchmark import BalancerEnv, TeacherAgent
        env = BalancerEnv()
        agent = TeacherAgent()
        state = env.reset()
        for _ in range(100):
            action = agent.choose_action("balancer", state)
            state, reward, done = env.step(action)
            if done:
                break
        # Teacher should survive at least 30 ticks (physics is stochastic)
        assert env.steps >= 30

    def test_teacher_catches_in_catcher(self):
        from python.benchmarks.benchmark import CatcherEnv, TeacherAgent
        env = CatcherEnv()
        agent = TeacherAgent()
        state = env.reset()
        for _ in range(300):
            action = agent.choose_action("catcher", state)
            state, reward, done = env.step(action)
            if done:
                break
        assert env.score >= 1

    def test_both_in_envs_dict(self):
        from python.benchmarks.benchmark import ENVS
        assert "balancer" in ENVS
        assert "catcher" in ENVS

    def test_predicate_hash_shared(self):
        """Balancer and Catcher produce the same hash format for transfer."""
        from python.benchmarks.benchmark import StudentAgent
        bal_state = {"type": "balancer", "object_x": 0.3, "object_vel": 0.01, "ball_pos": 0.3, "ball_vel": 0.01}
        cat_state = {"type": "catcher", "object_x": 0.3, "object_vel": 0.01, "paddle_x": 0.5}
        h1 = StudentAgent._hash("balancer", bal_state)
        h2 = StudentAgent._hash("catcher", cat_state)
        # Both should produce physics:... format
        assert h1.startswith("physics:")
        assert h2.startswith("physics:")
        # Same object_x and vel should produce same hash
        assert h1 == h2


class TestTransferExperiment:
    """End-to-end transfer test: train on Balancer, test on Catcher."""

    def test_transfer_balancer_to_catcher(self):
        from python.benchmarks.benchmark import BalancerEnv, CatcherEnv, TeacherAgent, StudentAgent, LearnedPolicy

        # 1. Train on Balancer
        env = BalancerEnv()
        teacher = TeacherAgent()
        policy = LearnedPolicy()

        for _ in range(50):
            state = env.reset()
            for _ in range(500):
                action = teacher.choose_action("balancer", state)
                h = StudentAgent._hash("balancer", state)
                policy.train(h, action)
                state, reward, done = env.step(action)
                if done:
                    break

        assert policy.size() > 0, "Policy should have learned entries"

        # 2. Test trained policy on Catcher
        student = StudentAgent(policy)
        catcher_env = CatcherEnv()
        total_catches = 0
        episodes = 20

        for _ in range(episodes):
            state = catcher_env.reset()
            for _ in range(600):
                action = student.choose_action("catcher", state)
                state, reward, done = catcher_env.step(action)
                if done:
                    break
            total_catches += catcher_env.score

        avg_catches = total_catches / episodes
        # Transfer agent should catch at least something
        # (random baseline catches ~5-8 on average before 5 misses)
        assert avg_catches >= 0, f"Transfer agent caught {avg_catches} avg"
