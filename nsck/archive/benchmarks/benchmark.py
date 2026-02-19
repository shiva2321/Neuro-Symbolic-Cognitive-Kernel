"""
NSCK Benchmark Harness
======================
Headless benchmark runner for systematically measuring agent performance.

Runs configurable episodes of Snake, Pong, Maze, and Collector games, measures key
metrics, and generates JSON + Markdown reports.

Usage:
    python benchmark.py                         # Default: 100 episodes per game
    python benchmark.py --episodes 500          # 500 episodes per game
    python benchmark.py --games snake,maze      # Only Snake and Maze
    python benchmark.py --transfer              # Include transfer learning test
    python benchmark.py --output results.md     # Custom output path
"""

import os
import sys
import json
import time
import random
import argparse
import math
from datetime import datetime, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

# Ensure nsck-demo/python is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


import numpy as np

from python.games.pong.simulation import sim_snake, sim_pong
from python.games.maze.maze_game import MazeGame
from python.games.collector.collector_game import CollectorGame
from python.games.physics.balancer_game import BalancerGame, CognitiveBalancerGame
from python.games.physics.catcher_game import CatcherGame, CognitiveCatcherGame


# =============================================================================
# Game Environments — Headless wrappers
# =============================================================================

class Environment:
    """Base class for environment wrappers."""
    def reset(self) -> Dict: raise NotImplementedError
    def step(self, action: str) -> Tuple[Dict, float, bool]: raise NotImplementedError
    @property
    def score(self) -> float: return 0.0


class SnakeEnv:
    """Headless Snake environment."""

    GRID = 10
    MAX_STEPS = 200
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self):
        self.head = (5, 5)
        self.body = [(5, 5)]
        self.food = self._random_food()
        self.score = 0
        self.steps = 0
        self.done = False
        return self._state()

    def _random_food(self):
        while True:
            pos = (random.randint(0, self.GRID - 1), random.randint(0, self.GRID - 1))
            if pos not in self.body:
                return pos

    def _state(self) -> Dict:
        return {
            "type": "snake",
            "head": self.head,
            "body": list(self.body),
            "food": self.food,
        }

    def step(self, action: str) -> Tuple[Dict, float, bool]:
        state_in = {"head": self.head, "body": list(self.body), "food": self.food}
        next_state, collision = sim_snake(state_in, action)

        self.head = next_state["head"]
        reward = 0.0

        if self.head == self.food:
            self.score += 1
            self.food = self._random_food()
            self.body.append(self.head)
            reward = 1.0
        else:
            if len(self.body) > 1:
                self.body.pop(0)
            self.body.append(self.head)

        if collision:
            self.done = True
            reward = -1.0

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.done = True

        return self._state(), reward, self.done


class PongEnv:
    """Headless Pong environment."""

    MAX_STEPS = 300
    ACTIONS = ["UP", "DOWN"]

    def reset(self):
        self.p1_y = 12
        self.ball_x = 15
        self.ball_y = 15
        self.ball_dx = -1
        self.ball_dy = random.choice([-1, 1])
        self.score = 0
        self.steps = 0
        self.done = False
        return self._state()

    def _state(self) -> Dict:
        return {
            "type": "pong",
            "p1_y": self.p1_y,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_dx": self.ball_dx,
            "ball_dy": self.ball_dy,
        }

    def step(self, action: str) -> Tuple[Dict, float, bool]:
        state_in = {
            "p1_y": self.p1_y,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_dx": self.ball_dx,
            "ball_dy": self.ball_dy,
        }
        next_state, miss = sim_pong(state_in, action)
        self.p1_y = next_state["p1_y"]
        self.ball_x = next_state["ball_x"]
        self.ball_y = next_state["ball_y"]
        self.ball_dx = next_state["ball_dx"]
        self.ball_dy = next_state["ball_dy"]

        reward = 0.0
        if self.ball_y <= 0 or self.ball_y >= 29:
            self.ball_dy = -self.ball_dy
        if self.ball_x >= 28:
            self.ball_dx = -self.ball_dx

        if miss:
            reward = -1.0
            self.ball_x = 15
            self.ball_y = 15
            self.ball_dx = -1
            self.ball_dy = random.choice([-1, 1])
        else:
            if self.ball_dx < 0 and self.ball_x <= 2:
                self.score += 1
                reward = 1.0
                self.ball_dx = -self.ball_dx

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.done = True

        return self._state(), reward, self.done


class MazeEnv:
    """Headless Maze environment."""

    MAX_STEPS = 200
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Using ACTION_ prefix mapping for consistency with cognitive engine
    ACTION_MAP = {
        "UP": "ACTION_UP",
        "DOWN": "ACTION_DOWN",
        "LEFT": "ACTION_LEFT",
        "RIGHT": "ACTION_RIGHT",
    }

    def reset(self):
        self.game = MazeGame(width=10, height=10)
        state_dict = self.game.state.to_dict()
        self.walls = set()
        for w in state_dict["walls"]:
            self.walls.add(tuple(w))
        self.player = tuple(state_dict["player_pos"])
        self.exit_pos = tuple(state_dict["exit_pos"])
        self.score = 0
        self.steps = 0
        self.done = False
        return self._state()

    def _state(self) -> Dict:
        return {
            "type": "maze",
            "player": self.player,
            "exit": self.exit_pos,
            "walls": self.walls,
            "size": 10,
        }

    def step(self, action: str) -> Tuple[Dict, float, bool]:
        px, py = self.player
        nx, ny = px, py

        if action == "UP":
            ny -= 1
        elif action == "DOWN":
            ny += 1
        elif action == "LEFT":
            nx -= 1
        elif action == "RIGHT":
            nx += 1

        reward = 0.0
        if (nx, ny) not in self.walls and 0 <= nx < 10 and 0 <= ny < 10:
            self.player = (nx, ny)

        if self.player == self.exit_pos:
            self.score += 1
            self.done = True
            reward = 1.0

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.done = True

        return self._state(), reward, self.done


class CollectorEnv:
    """Headless Collector environment."""

    MAX_STEPS = 300
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self):
        self.game = CollectorGame(width=10, height=10, num_items=4)
        sd = self.game.state.to_dict()
        self.player = tuple(sd["player_pos"])
        self.items = [tuple(i) for i in sd["items"]]
        self.collected = 0
        self.obstacles = set(tuple(o) for o in sd["obstacles"])
        self.score = 0
        self.steps = 0
        self.done = False
        return self._state()

    def _state(self) -> Dict:
        target = self.items[self.collected] if self.collected < len(self.items) else None
        return {
            "type": "collector",
            "player": self.player,
            "head": self.player,
            "items": self.items,
            "collected": self.collected,
            "target": target,
            "food": target,
            "obstacles": self.obstacles,
            "walls": self.obstacles,
            "body": [],
            "size": 10,
        }

    def step(self, action: str) -> Tuple[Dict, float, bool]:
        px, py = self.player
        nx, ny = px, py

        if action == "UP":
            ny -= 1
        elif action == "DOWN":
            ny += 1
        elif action == "LEFT":
            nx -= 1
        elif action == "RIGHT":
            nx += 1

        reward = -0.01
        if (nx, ny) not in self.obstacles and 0 <= nx < 10 and 0 <= ny < 10:
            self.player = (nx, ny)

        # Check sequential collection
        if self.collected < len(self.items) and self.player == self.items[self.collected]:
            self.collected += 1
            self.score += 1
            reward = 1.0
            if self.collected >= len(self.items):
                self.done = True
                reward = 5.0

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.done = True

        return self._state(), reward, self.done


class BalancerEnv(Environment):
    def __init__(self): 
        self.game = BalancerGame()
        self.done = False
        self.steps = 0
    def reset(self): 
        self.done = False
        self.steps = 0
        return self.game.reset().to_dict()
    def step(self, action):
        s, r, d = self.game.step(action)
        self.done = d
        self.steps += 1
        return s.to_dict(), r, d
    @property
    def score(self): return self.game.state.score

class CognitiveBalancerEnv(Environment):
    def __init__(self): 
        self.game = CognitiveBalancerGame()
        self.done = False
        self.steps = 0
    def reset(self): 
        self.done = False
        self.steps = 0
        return self.game.reset().to_dict()
    def step(self, action):
        s, r, d = self.game.step(action)
        self.done = d
        self.steps += 1
        return s.to_dict(), r, d
    @property
    def score(self): return self.game.state.score

class CatcherEnv(Environment):
    def __init__(self): 
        self.game = CatcherGame()
        self.done = False
        self.steps = 0
    def reset(self): 
        self.done = False
        self.steps = 0
        return self.game.reset().to_dict()
    def step(self, action):
        s, r, d = self.game.step(action)
        self.done = d
        self.steps += 1
        return s.to_dict(), r, d
    @property
    def score(self): return self.game.state.score

class CognitiveCatcherEnv(Environment):
    def __init__(self): 
        self.game = CognitiveCatcherGame()
        self.done = False
        self.steps = 0
    def reset(self): 
        self.done = False
        self.steps = 0
        return self.game.reset().to_dict()
    def step(self, action): 
        s, r, d = self.game.step(action)
        self.done = d
        self.steps += 1
        return s.to_dict(), r, d
    @property
    def score(self): return self.game.state.score

class InvertedCatcherEnv(CatcherEnv):
    """Adversarial variant: Inputs are inverted!"""
    def reset(self):
        self.steps = 0
        self.done = False
        s = self.game.reset().to_dict()
        s["object_x"] = 1.0 - s["object_x"]
        s["object_vel"] = -s.get("object_vel", 0.0)
        s["type"] = "inverted_catcher"
        return s
    def step(self, action):
        s, r, d = self.game.step(action)
        self.done = d
        self.steps += 1
        s = s.to_dict()
        s["object_x"] = 1.0 - s["object_x"]
        s["object_vel"] = -s.get("object_vel", 0.0)
        s["type"] = "inverted_catcher"
        return s, r, d

class EasyCatcherEnv(CatcherEnv):
    """Curriculum variant: Slower objects."""
    def __init__(self):
        super().__init__()
        self.game.FALL_SPEED = -0.01

class HardCatcherEnv(CatcherEnv):
    """Curriculum variant: Faster objects."""
    def __init__(self):
        super().__init__()
        self.game.FALL_SPEED = -0.03

ENVS = {
    "snake": SnakeEnv,
    "maze": MazeEnv,
    "collector": CollectorEnv,
    "pong": PongEnv,
    "balancer": BalancerEnv,
    "catcher": CatcherEnv,
    "inverted_catcher": InvertedCatcherEnv,
    "easy_catcher": EasyCatcherEnv,
    "hard_catcher": HardCatcherEnv,
    "cognitive_catcher": CognitiveCatcherEnv,
    "cognitive_balancer": CognitiveBalancerEnv,
}


# =============================================================================
# Agents — Teacher (heuristic solvers) and Student (learned policy)
# =============================================================================

def solve_snake_bfs(head, food, body, width=10, height=10):
    """BFS solver for Snake."""
    queue = [(head, [])]
    visited = {head}
    body_set = set(body)

    while queue:
        current, path = queue.pop(0)
        if current == food:
            if not path:
                return None
            next_step = path[0]
            dx, dy = next_step[0] - head[0], next_step[1] - head[1]
            if dx == 1: return "RIGHT"
            if dx == -1: return "LEFT"
            if dy == 1: return "DOWN"
            if dy == -1: return "UP"

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (current[0] + dx) % width, (current[1] + dy) % height
            if (nx, ny) not in visited and (nx, ny) not in body_set:
                visited.add((nx, ny))
                if not path:
                    queue.append(((nx, ny), [(nx, ny)]))
                else:
                    queue.append(((nx, ny), path))
    return None


def solve_maze_astar(start, goal, walls, width=10, height=10):
    """A* solver for Maze."""
    import heapq

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            if not path:
                return "UP"
            next_step = path[0]
            dx, dy = next_step[0] - start[0], next_step[1] - start[1]
            if dx == 1: return "RIGHT"
            if dx == -1: return "LEFT"
            if dy == 1: return "DOWN"
            if dy == -1: return "UP"
            return "UP"

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                if neighbor in walls:
                    continue
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None


def solve_collector_astar(player, target, obstacles, width=10, height=10):
    """A* solver for Collector — navigate to the next target item."""
    if target is None:
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
    return solve_maze_astar(player, target, obstacles, width, height)


def solve_pong(state):
    """Perfect-tracking solver for Pong."""
    ball_y = state["ball_y"]
    p1_y = state["p1_y"]
    if ball_y < p1_y:
        return "UP"
    elif ball_y > p1_y + 6:
        return "DOWN"
    return random.choice(["UP", "DOWN"])


class TeacherAgent:
    """Heuristic teacher that uses solvers."""

    def choose_action(self, game_type: str, state: Dict) -> str:
        if game_type == "snake":
            action = solve_snake_bfs(state["head"], state["food"], state["body"])
            if not action:
                hx, hy = state["head"]
                fx, fy = state["food"]
                dx, dy = fx - hx, fy - hy
                if abs(dx) > abs(dy):
                    action = "RIGHT" if dx > 0 else "LEFT"
                else:
                    action = "DOWN" if dy > 0 else "UP"
            return action
        elif game_type == "pong":
            return solve_pong(state)
        elif game_type == "maze":
            action = solve_maze_astar(state["player"], state["exit"], state["walls"])
            return action or "UP"
        elif game_type == "collector":
            target = state.get("target") or state.get("food")
            action = solve_collector_astar(state.get("player") or state.get("head"), target, state.get("obstacles") or state.get("walls", set()))
            return action or "UP"
        elif game_type == "balancer":
            # Center-seeking: tilt toward ball
            ball_pos = state.get("ball_pos", state.get("object_x", 0.0))
            ball_vel = state.get("ball_vel", state.get("object_vel", 0.0))
            # Predict where ball will be
            predicted = ball_pos + ball_vel * 5
            if predicted < -0.05:
                return "TILT_LEFT"
            elif predicted > 0.05:
                return "TILT_RIGHT"
            return "HOLD"
        elif game_type == "catcher":
            # Track the nearest falling object
            obj_x = state.get("object_x", 0.5)
            paddle_x = state.get("paddle_x", 0.5)
            if obj_x < paddle_x - 0.03:
                return "MOVE_LEFT"
            elif obj_x > paddle_x + 0.03:
                return "MOVE_RIGHT"
            return "HOLD"
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])


class LearnedPolicy:
    """Simple state-hash -> action lookup table (same as dashboard uses)."""

    def __init__(self):
        self.policy = defaultdict(lambda: defaultdict(int))

    def train(self, state_hash: str, action: str):
        self.policy[state_hash][action] += 1

    def predict(self, state_hash: str) -> Optional[str]:
        if state_hash not in self.policy:
            return None
        counts = self.policy[state_hash]
        return max(counts, key=counts.get)

    def size(self) -> int:
        return len(self.policy)


class StudentAgent:
    """Agent that uses learned policy, with random fallback."""

    def __init__(self, policy: LearnedPolicy):
        self.policy = policy

    def choose_action(self, game_type: str, state: Dict) -> str:
        h = self._hash(game_type, state)
        learned = self.policy.predict(h)
        if learned:
            return learned
        if game_type == "pong":
            return random.choice(["UP", "DOWN"])
        if game_type in ("balancer", "catcher"):
            return random.choice(["TILT_LEFT", "TILT_RIGHT", "HOLD"])
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    @staticmethod
    def _hash(game_type, state):
        return SymbolicHasher.default_hash(game_type, state)


class SymbolicHasher:
    """
    Configurable predicate-based state hasher for transfer experiments.

    Controls which predicates are included and how finely states are discretized.
    Both Balancer and Catcher produce the same hash format to enable cross-game transfer.
    """

    PREDICATE_POSITION = "position"
    PREDICATE_VELOCITY = "velocity"
    PREDICATE_DANGER = "danger"
    ALL_PREDICATES = {PREDICATE_POSITION, PREDICATE_VELOCITY, PREDICATE_DANGER}

    def __init__(self, predicates=None, bins=5, invert=False):
        """
        Args:
            predicates: Set of predicate names to include. None = all.
            bins: Number of position discretization bins (2=coarse, 20=fine).
            invert: If True, invert position signal (for adversarial tests).
        """
        self.predicates = self.ALL_PREDICATES.copy() if predicates is None else predicates
        self.bins = bins
        self.invert = invert

    def hash(self, game_type, state):
        """Hash a state using configured predicates."""
        # Grid games use game-specific hashing
        if game_type == "snake":
            return f"snake:{state['head']}:{state['food']}"
        elif game_type == "maze":
            return f"maze:{state['player']}"
        elif game_type == "collector":
            return f"collector:{state.get('player') or state.get('head')}:{state.get('target')}"
        elif game_type == "pong":
            return f"pong:{state['ball_x']}:{state['ball_y']}:{state['p1_y']}"
        elif game_type in ("balancer", "catcher", "inverted_catcher",
                           "easy_catcher", "hard_catcher"):
            return self._physics_hash(state)
        return "unknown"

    def _physics_hash(self, state):
        """Predicate-based hash for physics games."""
        obj_x = state.get("object_x", state.get("ball_pos", 0.0))
        obj_vel = state.get("object_vel", state.get("ball_vel", 0.0))

        if self.invert:
            obj_x = -obj_x
            obj_vel = -obj_vel

        parts = ["physics"]

        if self.PREDICATE_POSITION in self.predicates:
            pos_bin = round(obj_x * self.bins) / self.bins
            parts.append(f"{pos_bin}")

        if self.PREDICATE_VELOCITY in self.predicates:
            vel_bin = "L" if obj_vel < -0.005 else "R" if obj_vel > 0.005 else "0"
            parts.append(vel_bin)

        if self.PREDICATE_DANGER in self.predicates:
            danger = "DL" if obj_x < -0.7 else "DR" if obj_x > 0.7 else "-"
            parts.append(danger)

        return ":".join(parts)

    @staticmethod
    def default_hash(game_type, state):
        """Original hardcoded hash for backward compatibility."""
        if game_type == "snake":
            return f"snake:{state['head']}:{state['food']}"
        elif game_type == "maze":
            return f"maze:{state['player']}"
        elif game_type == "collector":
            return f"collector:{state.get('player') or state.get('head')}:{state.get('target')}"
        elif game_type == "pong":
            return f"pong:{state['ball_x']}:{state['ball_y']}:{state['p1_y']}"
        elif game_type in ("balancer", "catcher", "inverted_catcher",
                           "easy_catcher", "hard_catcher"):
            obj_x = state.get("object_x", state.get("ball_pos", 0.0))
            obj_vel = state.get("object_vel", state.get("ball_vel", 0.0))
            pos_bin = round(obj_x * 5) / 5
            vel_bin = "L" if obj_vel < -0.005 else "R" if obj_vel > 0.005 else "0"
            danger = "DL" if obj_x < -0.7 else "DR" if obj_x > 0.7 else "-"
            return f"physics:{pos_bin}:{vel_bin}:{danger}"
        return "unknown"


class RandomAgent:
    """Agent that picks random actions."""

    def choose_action(self, game_type: str, state: Dict) -> str:
        if game_type in ("balancer", "catcher", "cognitive_balancer", "cognitive_catcher",
                         "inverted_catcher", "easy_catcher", "hard_catcher"):
             return random.choice(["TILT_LEFT", "TILT_RIGHT", "HOLD"])
        if game_type == "pong":
            return random.choice(["UP", "DOWN"])
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])


# =============================================================================
# Natural Language Instruction Teacher
# =============================================================================

class NaturalLanguageTeacher:
    """
    Parses natural language instructions into symbolic rules.
    Enables zero-shot transfer via text.
    
    Supported grammar:
    "If [condition] then [action]"
    "Avoid [condition]"
    
    Keywords:
    - Object/Ball Left/Right
    - Danger Left/Right/Center
    - Move/Tilt Left/Right/Hold
    """
    
    def __init__(self):
        self.rules = []
        
    def teach(self, instruction: str):
        """Parse text instruction into a rule tuple (condition_fn, action)."""
        instruction = instruction.lower()
        
        # Action parsing
        action = "HOLD"
        is_avoid = "avoid" in instruction
        
        if "move left" in instruction or "tilt left" in instruction or "go left" in instruction:
            action = "TILT_LEFT"
        elif "move right" in instruction or "tilt right" in instruction or "go right" in instruction:
            action = "TILT_RIGHT"
        elif is_avoid:
            action = "AVOID" # Abstract action
        elif "catch" in instruction or "balance" in instruction:
            action = "APPROACH" # Abstract action (default)
            
        # Condition parsing
        cond = None
        
        # "If object is left..."
        if "object" in instruction or "ball" in instruction or "danger" in instruction or is_avoid:
            # Color/Type check
            target_type = None
            if "red" in instruction or "bad" in instruction: target_type = "red"
            if "green" in instruction or "good" in instruction: target_type = "green"
            if "danger" in instruction: target_type = "danger"

            # Position check
            if "left" in instruction:
                # "If [red] object is to the left"
                cond = lambda s: self._check_obj(s, side="left", type_=target_type)
            elif "right" in instruction:
                # "If [red] object is to the right"
                cond = lambda s: self._check_obj(s, side="right", type_=target_type)
            elif target_type:
                # "If object is red" (regardless of position)
                cond = lambda s: self._check_obj(s, side=None, type_=target_type)

        if cond:
            self.rules.append((cond, action))
            
    def _check_obj(self, state, side=None, type_=None) -> bool:
        # Check type
        if type_:
            obj_type = state.get("object_type", "green") # Default to green/neutral
            if obj_type != type_:
                return False
        
        # Check position
        if side:
            dx = self._get_obj_x(state)
            if side == "left": return dx < -0.05
            if side == "right": return dx > 0.05
            
        return True # Condition met

    def choose_action(self, game_type: str, state: Dict) -> str:
        """Evaluate rules against state."""
        # Check all rules
        for condition, action in self.rules:
            if condition(state):
                return self._resolve_action(game_type, action, state)
                
        return "HOLD" # Default
        
    def _resolve_action(self, game_type: str, abstract_action: str, state: Dict) -> str:
        """Map abstract actions (AVOID, APPROACH) to concrete game actions."""
        # 1. Concrete actions pass through
        if abstract_action in ["TILT_LEFT", "TILT_RIGHT", "HOLD"]:
            if game_type in ["catcher", "cognitive_catcher"] or "catcher" in game_type:
                return abstract_action.replace("TILT", "MOVE")
            return abstract_action

        # 2. Abstract AVOID (Red/Bad)
        if abstract_action == "AVOID":
            dx = self._get_obj_x(state)
            if "catcher" in game_type:
                # Catcher: DODGE (Move AWAY from object)
                # If obj is Left -> Move Right
                if dx < 0: return "MOVE_RIGHT"
                if dx > 0: return "MOVE_LEFT"
            elif "balancer" in game_type:
                # Balancer: DUMP (Tilt TOWARDS object to roll it off)
                # If obj is Left -> Tilt Left (Down) -> Rolls Left (Off)
                if dx < 0: return "TILT_LEFT"
                if dx > 0: return "TILT_RIGHT"

        # 3. Abstract APPROACH (Green/Good)
        if abstract_action == "APPROACH":
            dx = self._get_obj_x(state)
            if "catcher" in game_type:
                # Catcher: CATCH (Move TOWARDS object)
                if dx < -0.05: return "MOVE_LEFT"
                if dx > 0.05: return "MOVE_RIGHT"
            elif "balancer" in game_type:
                # Balancer: BALANCE (Tilt OPPOSITE to object to center it)
                # If obj is Left -> Tilt Right (Up) -> Rolls Right (Center)
                if dx < -0.05: return "TILT_RIGHT"
                if dx > 0.05: return "TILT_LEFT"
                
        return "HOLD"
        
    def _get_obj_x(self, state):
        # Normalize object position to -1..1 range relative to center/paddle
        if "ball_pos" in state:
            return state["ball_pos"]
        if "object_x" in state:
            # For catcher, relative to paddle
            # object_x is 0..1, paddle_x is 0..1
            # We want (obj - paddle)
            return state["object_x"] - state.get("paddle_x", 0.5)
        return 0.0


# =============================================================================
# Metrics collection
# =============================================================================

@dataclass
class EpisodeResult:
    game: str
    episode: int
    score: int
    steps: int
    total_reward: float
    reached_goal: bool
    agent_type: str  # "teacher" or "student" or "random"


@dataclass
class BenchmarkResult:
    game: str
    agent: str
    episodes: int
    avg_score: float
    median_score: float
    max_score: int
    min_score: int
    avg_steps: float
    avg_reward: float
    success_rate: float  # % of episodes reaching goal
    first_success_episode: Optional[int]
    scores: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "game": self.game,
            "agent": self.agent,
            "episodes": self.episodes,
            "avg_score": round(self.avg_score, 3),
            "median_score": round(self.median_score, 1),
            "max_score": self.max_score,
            "min_score": self.min_score,
            "avg_steps": round(self.avg_steps, 1),
            "avg_reward": round(self.avg_reward, 3),
            "success_rate": round(self.success_rate, 3),
            "first_success_episode": self.first_success_episode,
        }


# =============================================================================
# Benchmark runner
# =============================================================================

class BenchmarkRunner:
    """Runs headless benchmark episodes and collects metrics."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def run_game(
        self,
        game_type: str,
        agent,
        agent_name: str,
        episodes: int,
        policy: Optional[LearnedPolicy] = None,
    ) -> BenchmarkResult:
        """Run N episodes of a game and return aggregate metrics."""

        env_cls = ENVS[game_type]
        env = env_cls()

        all_scores = []
        all_steps = []
        all_rewards = []
        successes = 0
        first_success = None

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            reached_goal = False

            while not env.done:
                action = agent.choose_action(game_type, state)

                # Optionally train the policy (for teacher-then-student scenarios)
                if policy is not None:
                    h = StudentAgent._hash(game_type, state)
                    policy.train(h, action)

                state, reward, done = env.step(action)
                total_reward += reward

            score = env.score
            steps = env.steps
            reached_goal = score > 0

            all_scores.append(score)
            all_steps.append(steps)
            all_rewards.append(total_reward)

            if reached_goal:
                successes += 1
                if first_success is None:
                    first_success = ep

            if self.verbose and (ep + 1) % max(1, episodes // 10) == 0:
                print(
                    f"  [{game_type}/{agent_name}] ep {ep+1}/{episodes} "
                    f"| score={score} steps={steps} reward={total_reward:.1f}"
                )

        sorted_scores = sorted(all_scores)
        median = sorted_scores[len(sorted_scores) // 2]

        result = BenchmarkResult(
            game=game_type,
            agent=agent_name,
            episodes=episodes,
            avg_score=sum(all_scores) / len(all_scores),
            median_score=median,
            max_score=max(all_scores),
            min_score=min(all_scores),
            avg_steps=sum(all_steps) / len(all_steps),
            avg_reward=sum(all_rewards) / len(all_rewards),
            success_rate=successes / episodes,
            first_success_episode=first_success,
            scores=all_scores,
            rewards=all_rewards,
        )
        self.results.append(result)
        return result

    def run_transfer_test(
        self,
        source_game: str,
        target_game: str,
        train_episodes: int,
        test_episodes: int,
    ) -> Dict[str, BenchmarkResult]:
        """
        Transfer learning test:
        1. Train teacher policy on source_game
        2. Test trained policy on target_game (student)
        3. Compare vs random baseline on target_game
        """
        print(f"\n{'='*60}")
        print(f"TRANSFER TEST: {source_game} -> {target_game}")
        print(f"{'='*60}")

        # Phase 1: Train on source
        teacher = TeacherAgent()
        shared_policy = LearnedPolicy()

        print(f"\n[1/3] Training on {source_game} ({train_episodes} episodes)...")
        source_result = self.run_game(
            source_game, teacher, "teacher", train_episodes, policy=shared_policy
        )
        print(f"  -> Policy size: {shared_policy.size()} state-action pairs")
        print(f"  -> Avg score: {source_result.avg_score:.2f}")

        # Phase 2: Test on target with trained policy
        student = StudentAgent(shared_policy)
        print(f"\n[2/3] Testing trained policy on {target_game} ({test_episodes} episodes)...")
        transfer_result = self.run_game(
            target_game, student, f"transfer({source_game}->{target_game})", test_episodes
        )

        # Phase 3: Random baseline on target
        class RandomAgent:
            def choose_action(self, game_type, state):
                if game_type == "pong":
                    return random.choice(["UP", "DOWN"])
                return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        print(f"\n[3/3] Random baseline on {target_game} ({test_episodes} episodes)...")
        random_result = self.run_game(
            target_game, RandomAgent(), "random", test_episodes
        )

        return {
            "source_training": source_result,
            "transfer": transfer_result,
            "random_baseline": random_result,
        }


# =============================================================================
# Report generation
# =============================================================================

def generate_report(
    results: List[BenchmarkResult],
    transfer_results: Optional[Dict] = None,
    elapsed: float = 0.0,
) -> str:
    """Generate Markdown report from benchmark results."""

    lines = []
    lines.append("# NSCK Benchmark Report")
    lines.append(f"")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Duration:** {elapsed:.1f}s")
    lines.append("")

    # Summary table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Game | Agent | Episodes | Avg Score | Max | Success % | Avg Steps | Avg Reward |")
    lines.append("|------|-------|----------|-----------|-----|-----------|-----------|------------|")

    for r in results:
        lines.append(
            f"| {r.game} | {r.agent} | {r.episodes} "
            f"| {r.avg_score:.2f} | {r.max_score} "
            f"| {r.success_rate*100:.1f}% "
            f"| {r.avg_steps:.0f} | {r.avg_reward:.2f} |"
        )
    lines.append("")

    # Per-game details
    lines.append("## Per-Game Details")
    lines.append("")

    games_seen = set()
    for r in results:
        if r.game in games_seen:
            continue
        games_seen.add(r.game)

        # Collect all results for this game
        game_results = [x for x in results if x.game == r.game]

        lines.append(f"### {r.game.title()}")
        lines.append("")

        for gr in game_results:
            lines.append(f"**{gr.agent}** ({gr.episodes} episodes)")
            lines.append(f"- Score: avg={gr.avg_score:.2f}, median={gr.median_score:.1f}, "
                         f"max={gr.max_score}, min={gr.min_score}")
            lines.append(f"- Steps: avg={gr.avg_steps:.0f}")
            lines.append(f"- Reward: avg={gr.avg_reward:.3f}")
            lines.append(f"- Success rate: {gr.success_rate*100:.1f}%")
            if gr.first_success_episode is not None:
                lines.append(f"- First success: episode {gr.first_success_episode}")

            # Score distribution
            if gr.scores:
                q25 = sorted(gr.scores)[len(gr.scores) // 4]
                q75 = sorted(gr.scores)[3 * len(gr.scores) // 4]
                lines.append(f"- Score quartiles: Q1={q25}, Q3={q75}")
            lines.append("")

    # Transfer test results
    if transfer_results:
        lines.append("## Transfer Learning Test")
        lines.append("")

        src = transfer_results["source_training"]
        xfr = transfer_results["transfer"]
        rnd = transfer_results["random_baseline"]

        lines.append(f"**Source:** {src.game} ({src.episodes} training episodes)")
        lines.append(f"**Target:** {xfr.game} ({xfr.episodes} test episodes)")
        lines.append("")

        lines.append("| Metric | Transfer | Random Baseline | Δ |")
        lines.append("|--------|----------|-----------------|---|")

        delta_score = xfr.avg_score - rnd.avg_score
        delta_success = (xfr.success_rate - rnd.success_rate) * 100
        delta_reward = xfr.avg_reward - rnd.avg_reward

        lines.append(f"| Avg Score | {xfr.avg_score:.2f} | {rnd.avg_score:.2f} "
                     f"| {'+' if delta_score >= 0 else ''}{delta_score:.2f} |")
        lines.append(f"| Success % | {xfr.success_rate*100:.1f}% | {rnd.success_rate*100:.1f}% "
                     f"| {'+' if delta_success >= 0 else ''}{delta_success:.1f}% |")
        lines.append(f"| Avg Reward | {xfr.avg_reward:.3f} | {rnd.avg_reward:.3f} "
                     f"| {'+' if delta_reward >= 0 else ''}{delta_reward:.3f} |")
        lines.append("")

        if delta_score > 0:
            lines.append(f"> ✅ Transfer shows **{delta_score:.2f}** score improvement over random baseline")
        elif delta_score == 0:
            lines.append("> ⚠️ Transfer shows **no improvement** over random baseline")
        else:
            lines.append(f"> ❌ Transfer shows **negative transfer** ({delta_score:.2f} worse than random)")
        lines.append("")

    return "\n".join(lines)


def save_json_report(
    results: List[BenchmarkResult],
    transfer_results: Optional[Dict],
    filepath: str,
    elapsed: float,
):
    """Save results as JSON."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(elapsed, 2),
        "results": [r.to_dict() for r in results],
    }
    if transfer_results:
        data["transfer_test"] = {
            k: v.to_dict() for k, v in transfer_results.items()
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON report saved to: {filepath}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NSCK Benchmark Harness — measure agent performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                          # Default: 100 episodes
  python benchmark.py --episodes 500           # 500 episodes per game
  python benchmark.py --games snake,maze       # Only selected games
  python benchmark.py --transfer               # Include transfer test
  python benchmark.py --transfer-source snake --transfer-target maze
        """,
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes per game (default: 100)"
    )
    parser.add_argument(
        "--games", type=str, default="snake,pong,maze,collector",
        help="Comma-separated list of games to benchmark (default: snake,pong,maze,collector)"
    )
    parser.add_argument(
        "--transfer", action="store_true",
        help="Run transfer learning test (train on source, test on target)"
    )
    parser.add_argument(
        "--transfer-source", type=str, default="snake",
        help="Source game for transfer test (default: snake)"
    )
    parser.add_argument(
        "--transfer-target", type=str, default="maze",
        help="Target game for transfer test (default: maze)"
    )
    parser.add_argument(
        "--transfer-train-episodes", type=int, default=None,
        help="Training episodes for transfer source (default: same as --episodes)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for Markdown report (default: benchmark_report.md)"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Output path for JSON report (default: benchmark_report.json)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress during benchmark"
    )

    args = parser.parse_args()

    games = [g.strip().lower() for g in args.games.split(",")]
    for g in games:
        if g not in ENVS:
            print(f"Unknown game: {g}. Available: {', '.join(ENVS.keys())}")
            sys.exit(1)

    # Compute output paths
    output_dir = os.path.dirname(__file__)
    md_path = args.output or os.path.join(output_dir, "benchmark_report.md")
    json_path = args.json or os.path.join(output_dir, "benchmark_report.json")

    print("=" * 60)
    print("  NSCK BENCHMARK HARNESS")
    print("=" * 60)
    print(f"  Games:    {', '.join(games)}")
    print(f"  Episodes: {args.episodes} per game")
    print(f"  Transfer: {'Yes' if args.transfer else 'No'}")
    print(f"  Output:   {md_path}")
    print("=" * 60)

    runner = BenchmarkRunner(verbose=args.verbose)
    teacher = TeacherAgent()
    start_time = time.time()

    # Run teacher benchmark for each game
    for game in games:
        print(f"\n>>> Benchmarking {game.upper()} with Teacher agent ({args.episodes} episodes)...")
        result = runner.run_game(game, teacher, "teacher", args.episodes)
        print(f"  -> avg_score={result.avg_score:.2f}  success={result.success_rate*100:.1f}%  "
              f"avg_steps={result.avg_steps:.0f}")

    # Run random baseline for comparison
    class RandomAgent:
        def choose_action(self, game_type, state):
            if game_type == "pong":
                return random.choice(["UP", "DOWN"])
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    random_agent = RandomAgent()
    for game in games:
        print(f"\n>>> Benchmarking {game.upper()} with Random agent ({args.episodes} episodes)...")
        result = runner.run_game(game, random_agent, "random", args.episodes)
        print(f"  -> avg_score={result.avg_score:.2f}  success={result.success_rate*100:.1f}%  "
              f"avg_steps={result.avg_steps:.0f}")

    # Transfer test
    transfer_results = None
    if args.transfer:
        train_ep = args.transfer_train_episodes or args.episodes
        transfer_results = runner.run_transfer_test(
            args.transfer_source,
            args.transfer_target,
            train_ep,
            args.episodes,
        )

    elapsed = time.time() - start_time

    # Generate reports
    report = generate_report(runner.results, transfer_results, elapsed)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[*] Markdown report: {md_path}")

    save_json_report(runner.results, transfer_results, json_path, elapsed)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE -- {elapsed:.1f}s")
    print(f"{'='*60}")


# =============================================================================
# Transfer Learning Utilities
# =============================================================================

def run_training(game_type: str, episodes: int, hasher: Optional[SymbolicHasher] = None, policy_override: Optional[LearnedPolicy] = None) -> LearnedPolicy:
    """Train a policy by cloning the TeacherAgent."""
    if hasher is None:
        hasher = SymbolicHasher()
    
    policy = policy_override or LearnedPolicy()
    teacher = TeacherAgent()
    runner = BenchmarkRunner(verbose=False)
    
    class RecordingAgent:
        def choose_action(self, g, s):
            a = teacher.choose_action(g, s)
            # Hash state and record action
            h = hasher.hash(g, s)
            policy.train(h, a)
            return a
            
    runner.run_game(game_type, RecordingAgent(), "recorder", episodes)
    return policy


def run_testing(game_type: str, episodes: int, policy: LearnedPolicy, hasher: Optional[SymbolicHasher] = None) -> List[int]:
    """Test a policy and return scores."""
    if hasher is None:
        hasher = SymbolicHasher()
        
    class CustomStudent(StudentAgent):
        def __init__(self, p, h):
            super().__init__(p)
            self.hasher = h
        
        # Override to use instance hasher
        def choose_action(self, game_type, state):
            h = self.hasher.hash(game_type, state)
            learned = self.policy.predict(h)
            if learned: return learned
            # Fallback
            if game_type in ("balancer", "catcher", "cognitive_balancer", "cognitive_catcher"):
                return random.choice(["TILT_LEFT", "TILT_RIGHT", "HOLD"]) 
            if game_type == "pong": return random.choice(["UP", "DOWN"])
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    runner = BenchmarkRunner(verbose=False)
    res = runner.run_game(game_type, CustomStudent(policy, hasher), "student", episodes)
    return res.scores


def run_random_baseline(game_type: str, episodes: int) -> List[int]:
    """Run random agent for baseline."""
    class RobustRandom:
        def choose_action(self, gt, s):
            if gt in ("balancer", "catcher", "cognitive_balancer", "cognitive_catcher"):
                return random.choice(["TILT_LEFT", "TILT_RIGHT", "HOLD"])
            if gt == "pong": return random.choice(["UP", "DOWN"])
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            
    runner = BenchmarkRunner(verbose=False)
    res = runner.run_game(game_type, RobustRandom(), "random", episodes)
    return res.scores


if __name__ == "__main__":
    main()
