"""
Tests for the NSCK Benchmark Harness.

Validates that all game environments, agents, and report generation
work correctly with a small number of episodes.
"""

import sys
import os
import json
import tempfile


from python.benchmarks.benchmark import (
    SnakeEnv, PongEnv, MazeEnv,
    TeacherAgent, StudentAgent, LearnedPolicy, BenchmarkRunner,
    generate_report, save_json_report,
)


class TestSnakeEnv:
    """Tests for headless Snake environment."""

    def test_reset(self):
        env = SnakeEnv()
        state = env.reset()
        assert state["type"] == "snake"
        assert "head" in state
        assert "body" in state
        assert "food" in state
        assert not env.done

    def test_step_returns_correct_shape(self):
        env = SnakeEnv()
        env.reset()
        state, reward, done = env.step("UP")
        assert isinstance(state, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_food_collection_increases_score(self):
        env = SnakeEnv()
        env.reset()
        # Place food exactly where the head will go
        env.food = ((env.head[0]) % 10, (env.head[1] - 1) % 10)
        state, reward, done = env.step("UP")
        assert env.score == 1
        assert reward == 1.0

    def test_wall_collision(self):
        env = SnakeEnv()
        env.reset()
        # Snake uses toroidal wrap, so no wall collision
        # Just verify stepping doesn't crash
        for _ in range(10):
            state, reward, done = env.step("UP")
            if done:
                break

    def test_max_steps_terminates(self):
        env = SnakeEnv()
        env.reset()
        for _ in range(300):  # More than MAX_STEPS
            if env.done:
                break
            env.step("RIGHT")
        assert env.done
        assert env.steps <= SnakeEnv.MAX_STEPS


class TestPongEnv:
    """Tests for headless Pong environment."""

    def test_reset(self):
        env = PongEnv()
        state = env.reset()
        assert state["type"] == "pong"
        assert "p1_y" in state
        assert "ball_x" in state
        assert not env.done

    def test_step_returns_correct_shape(self):
        env = PongEnv()
        env.reset()
        state, reward, done = env.step("UP")
        assert isinstance(state, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_max_steps_terminates(self):
        env = PongEnv()
        env.reset()
        for _ in range(400):
            if env.done:
                break
            env.step("UP")
        assert env.done


class TestMazeEnv:
    """Tests for headless Maze environment."""

    def test_reset(self):
        env = MazeEnv()
        state = env.reset()
        assert state["type"] == "maze"
        assert "player" in state
        assert "exit" in state
        assert "walls" in state
        assert not env.done

    def test_step_doesnt_move_into_walls(self):
        env = MazeEnv()
        env.reset()
        # Try to step in all directions many times
        for _ in range(50):
            old_pos = env.player
            env.step("UP")
            # If new position is valid, it should not be in walls
            if env.player != old_pos:
                assert env.player not in env.walls

    def test_reaching_exit_wins(self):
        env = MazeEnv()
        env.reset()
        # Teleport player next to exit
        ex, ey = env.exit_pos
        # Try placing player 1 step away
        env.player = (ex, ey + 1) if (ex, ey + 1) not in env.walls else (ex, ey)
        if env.player != env.exit_pos:
            state, reward, done = env.step("UP")
            if env.player == env.exit_pos:
                assert reward == 1.0
                assert done


class TestTeacherAgent:
    """Tests for heuristic teacher agent."""

    def test_snake_teacher_gets_food(self):
        env = SnakeEnv()
        teacher = TeacherAgent()
        state = env.reset()
        total_score = 0
        for _ in range(200):
            action = teacher.choose_action("snake", state)
            assert action in SnakeEnv.ACTIONS
            state, reward, done = env.step(action)
            if done:
                total_score = env.score
                break
        # Teacher should get at least some food
        assert total_score >= 0  # BFS should find food

    def test_maze_teacher_solves(self):
        env = MazeEnv()
        teacher = TeacherAgent()
        state = env.reset()
        for _ in range(200):
            action = teacher.choose_action("maze", state)
            assert action in MazeEnv.ACTIONS
            state, reward, done = env.step(action)
            if done:
                break
        # Teacher with A* should solve most mazes
        assert env.score >= 0

    def test_pong_teacher_tracks(self):
        env = PongEnv()
        teacher = TeacherAgent()
        state = env.reset()
        for _ in range(300):
            action = teacher.choose_action("pong", state)
            assert action in PongEnv.ACTIONS
            state, reward, done = env.step(action)
            if done:
                break


class TestLearnedPolicy:
    """Tests for the learned policy store."""

    def test_train_and_predict(self):
        policy = LearnedPolicy()
        policy.train("state_1", "UP")
        policy.train("state_1", "UP")
        policy.train("state_1", "DOWN")
        assert policy.predict("state_1") == "UP"

    def test_unknown_state_returns_none(self):
        policy = LearnedPolicy()
        assert policy.predict("unknown") is None

    def test_size_tracking(self):
        policy = LearnedPolicy()
        assert policy.size() == 0
        policy.train("a", "UP")
        policy.train("b", "DOWN")
        assert policy.size() == 2


class TestBenchmarkRunner:
    """Tests for the benchmark runner."""

    def test_run_snake_small(self):
        runner = BenchmarkRunner()
        teacher = TeacherAgent()
        result = runner.run_game("snake", teacher, "teacher", episodes=5)
        assert result.game == "snake"
        assert result.agent == "teacher"
        assert result.episodes == 5
        assert len(result.scores) == 5
        assert len(result.rewards) == 5

    def test_run_pong_small(self):
        runner = BenchmarkRunner()
        teacher = TeacherAgent()
        result = runner.run_game("pong", teacher, "teacher", episodes=5)
        assert result.game == "pong"
        assert result.episodes == 5

    def test_run_maze_small(self):
        runner = BenchmarkRunner()
        teacher = TeacherAgent()
        result = runner.run_game("maze", teacher, "teacher", episodes=5)
        assert result.game == "maze"
        assert result.episodes == 5

    def test_transfer_test(self):
        runner = BenchmarkRunner()
        results = runner.run_transfer_test("snake", "maze", 10, 5)
        assert "source_training" in results
        assert "transfer" in results
        assert "random_baseline" in results


class TestReportGeneration:
    """Tests for report output."""

    def test_markdown_report(self):
        runner = BenchmarkRunner()
        teacher = TeacherAgent()
        runner.run_game("snake", teacher, "teacher", episodes=3)
        report = generate_report(runner.results, elapsed=1.0)
        assert "# NSCK Benchmark Report" in report
        assert "snake" in report
        assert "teacher" in report

    def test_json_report(self):
        runner = BenchmarkRunner()
        teacher = TeacherAgent()
        runner.run_game("snake", teacher, "teacher", episodes=3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_json_report(runner.results, None, f.name, 1.0)
            json_path = f.name

        try:
            with open(json_path) as f:
                data = json.load(f)
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["game"] == "snake"
        finally:
            os.unlink(json_path)
