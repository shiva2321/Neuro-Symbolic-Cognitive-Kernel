"""
Tests for the Collector game environment.

Covers:
- Game mechanics (reset, step, sequential collection, obstacles)
- Grounding verifier predicates
- Benchmark integration (CollectorEnv + TeacherAgent)
"""
import sys
import os
import pytest

# Ensure nsck-demo/python is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from collector_game import CollectorGame, CollectorState, sim_collector
from grounding_verifier import create_collector_verifier


class TestCollectorGame:
    """Tests for CollectorGame mechanics."""

    def test_reset_creates_valid_state(self):
        game = CollectorGame(width=10, height=10, num_items=4)
        state = game.state
        assert state is not None
        assert state.collected == 0
        assert state.total_items == 4
        assert len(state.items) == 4
        assert state.done is False
        assert 0 <= state.player_pos[0] < 10
        assert 0 <= state.player_pos[1] < 10

    def test_items_not_on_obstacles(self):
        game = CollectorGame(width=10, height=10, num_items=4)
        for item in game.state.items:
            assert item not in game.state.obstacles

    def test_player_not_on_obstacle(self):
        game = CollectorGame(width=10, height=10, num_items=4)
        assert game.state.player_pos not in game.state.obstacles

    def test_step_moves_player(self):
        game = CollectorGame(width=10, height=10, num_items=4, obstacle_density=0.0)
        game.state.player_pos = (5, 5)
        game.state.obstacles = set()
        state, reward, done = game.step("UP")
        assert state.player_pos == (5, 4)

    def test_step_blocked_by_obstacle(self):
        game = CollectorGame(width=10, height=10, num_items=4, obstacle_density=0.0)
        game.state.player_pos = (5, 5)
        game.state.obstacles = {(5, 4)}  # block UP
        state, reward, done = game.step("UP")
        assert state.player_pos == (5, 5)  # didn't move

    def test_step_blocked_by_bounds(self):
        game = CollectorGame(width=10, height=10, num_items=4, obstacle_density=0.0)
        game.state.player_pos = (0, 0)
        game.state.obstacles = set()
        state, reward, done = game.step("LEFT")
        assert state.player_pos == (0, 0)  # didn't move

    def test_sequential_collection(self):
        game = CollectorGame(width=10, height=10, num_items=3, obstacle_density=0.0)
        game.state.player_pos = (3, 3)
        game.state.items = [(4, 3), (5, 3), (6, 3)]
        game.state.obstacles = set()
        game.state.collected = 0

        # Move right to item 1
        state, reward, done = game.step("RIGHT")
        assert state.collected == 1
        assert reward == 1.0
        assert not done

        # Move right to item 2
        state, reward, done = game.step("RIGHT")
        assert state.collected == 2
        assert not done

        # Move right to item 3 (last)
        state, reward, done = game.step("RIGHT")
        assert state.collected == 3
        assert reward == 5.0
        assert done

    def test_must_collect_in_order(self):
        game = CollectorGame(width=10, height=10, num_items=3, obstacle_density=0.0)
        game.state.player_pos = (3, 3)
        game.state.items = [(4, 3), (5, 3), (6, 3)]
        game.state.obstacles = set()
        game.state.collected = 0

        # Try to reach item 2 without collecting item 1 first
        game.state.player_pos = (5, 3)
        state, reward, done = game.step("UP")
        state, reward, done = game.step("DOWN")
        # Should not have collected anything since item 1 wasn't collected
        assert state.collected == 0

    def test_max_steps_terminates(self):
        game = CollectorGame(width=10, height=10, num_items=4, obstacle_density=0.0)
        game.state.obstacles = set()
        game.state.steps = 299
        game.state.player_pos = (5, 5)
        state, reward, done = game.step("UP")
        assert done is True

    def test_to_dict_has_compatibility_fields(self):
        game = CollectorGame(width=10, height=10, num_items=4)
        d = game.state.to_dict()
        # Snake compatibility
        assert "head" in d
        assert "food" in d
        assert "body" in d
        # Maze compatibility
        assert "target" in d
        assert "walls" in d
        # Collector-specific
        assert "items" in d
        assert "collected" in d
        assert "obstacles" in d

    def test_render_ascii(self):
        game = CollectorGame(width=10, height=10, num_items=3, obstacle_density=0.0)
        ascii_str = game.render_ascii()
        assert "P" in ascii_str     # Player
        assert "Score:" in ascii_str


class TestSimCollector:
    """Tests for the lightweight sim_collector function."""

    def test_basic_movement(self):
        state = {
            "player_pos": (5, 5),
            "head": (5, 5),
            "items": [(7, 7)],
            "collected": 0,
            "obstacles": [],
            "width": 10,
            "height": 10,
        }
        next_state, terminal = sim_collector(state, "DOWN")
        assert next_state["player_pos"] == (5, 6)
        assert not terminal

    def test_item_collection(self):
        state = {
            "player_pos": (4, 5),
            "head": (4, 5),
            "items": [(5, 5)],
            "collected": 0,
            "obstacles": [],
            "width": 10,
            "height": 10,
        }
        next_state, terminal = sim_collector(state, "RIGHT")
        assert next_state["collected"] == 1
        assert terminal  # Only 1 item, so we're done

    def test_obstacle_blocks(self):
        state = {
            "player_pos": (5, 5),
            "head": (5, 5),
            "items": [(7, 7)],
            "collected": 0,
            "obstacles": [(5, 4)],
            "width": 10,
            "height": 10,
        }
        next_state, terminal = sim_collector(state, "UP")
        assert next_state["player_pos"] == (5, 5)  # Blocked


class TestCollectorVerifier:
    """Tests for create_collector_verifier predicates."""

    def test_rel_above(self):
        verifier = create_collector_verifier()
        state = {"head": (5, 5), "target": (5, 3), "obstacles": [], "width": 10, "height": 10}
        assert verifier.verify_predicate("REL_ABOVE", state, context="collector")
        assert not verifier.verify_predicate("REL_BELOW", state, context="collector")

    def test_rel_right(self):
        verifier = create_collector_verifier()
        state = {"head": (3, 5), "target": (7, 5), "obstacles": [], "width": 10, "height": 10}
        assert verifier.verify_predicate("REL_RIGHT", state, context="collector")
        assert not verifier.verify_predicate("REL_LEFT", state, context="collector")

    def test_obstacle_detection(self):
        verifier = create_collector_verifier()
        state = {"head": (5, 5), "target": (7, 7), "obstacles": [(5, 4)], "width": 10, "height": 10}
        assert verifier.verify_predicate("OBSTACLE_UP", state, context="collector")
        assert not verifier.verify_predicate("OBSTACLE_DOWN", state, context="collector")

    def test_boundary_as_obstacle(self):
        verifier = create_collector_verifier()
        state = {"head": (0, 0), "target": (5, 5), "obstacles": [], "width": 10, "height": 10}
        assert verifier.verify_predicate("OBSTACLE_UP", state, context="collector")
        assert verifier.verify_predicate("OBSTACLE_LEFT", state, context="collector")


class TestBenchmarkIntegration:
    """Tests for CollectorEnv and teacher agent in benchmark."""

    def test_collector_env_runs(self):
        from benchmark import CollectorEnv
        env = CollectorEnv()
        state = env.reset()
        assert state["type"] == "collector"
        assert not env.done

    def test_teacher_solves_collector(self):
        from benchmark import CollectorEnv, TeacherAgent
        env = CollectorEnv()
        agent = TeacherAgent()
        state = env.reset()
        for _ in range(env.MAX_STEPS):
            action = agent.choose_action("collector", state)
            state, reward, done = env.step(action)
            if done:
                break
        # Teacher should collect at least some items
        assert env.score >= 1

    def test_collector_in_envs_dict(self):
        from benchmark import ENVS
        assert "collector" in ENVS
