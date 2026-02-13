"""
NSCK Agency Module (Phase 4)
============================
Implements Active Inference for decision making.

Core Concept: Minimize Expected Free Energy (G).
G(pi) = Pragmatic Value (Risk) + Epistemic Value (Ambiguity/InfoGain).

We approximate this for a Grid World:
- State: Agent Position (Known) + Map Beliefs (Unknown).
- Beliefs: Q[x,y] = Softmax distribution over {Unknown, Empty, Wall, Food}.
- Epistemic Value: Entropy of tiles in the predicted path (Information Gain).
- Pragmatic Value: Distance to 'Preferred' observations (not used in pure curiosity test).
"""

import numpy as np
import copy
from typing import List, Tuple, Dict

# Constants
TILE_UNKNOWN = 0
TILE_EMPTY = 1
TILE_WALL = 2
TILE_FOOD = 3
NUM_TILES = 4

ACTIONS = [
    (0, -1), # Up
    (0, 1),  # Down
    (-1, 0), # Left
    (1, 0)   # Right
]

class ActiveAgent:
    def __init__(self, w: int, h: int, horizon: int = 3):
        self.w = w
        self.h = h
        self.horizon = horizon
        
        # Belief State Q(s): [W, H, NUM_TILES]
        # Initialize as mostly Unknown
        self.belief_map = np.ones((w, h, NUM_TILES)) * 0.1
        self.belief_map[:, :, TILE_UNKNOWN] = 0.7 # High prior on Unknown
        # Normalize
        self.belief_map /= self.belief_map.sum(axis=-1, keepdims=True)
        
        # Internal State
        self.pos = (0, 0) # x, y
        
        # Preferences (Dynamic)
        self.preferences = {
            TILE_FOOD: 10.0,
            TILE_WALL: -10.0,
            TILE_EMPTY: 0.0,
            TILE_UNKNOWN: 0.0
        }
        
        # Generative Model (Physics)
        # We assume we know grid boundaries
        
    def reset(self, x: int, y: int):
        self.pos = (x, y)
        # Reset beliefs? No, agency persists.
        
    def _entropy(self, prob_dist: np.ndarray) -> float:
        """Compute Shannon Entropy H(P) in nats."""
        # Avoid log(0)
        p = np.clip(prob_dist, 1e-6, 1.0 - 1e-6)
        return -np.sum(p * np.log(p))

    def update_belief(self, view_x: int, view_y: int, tile_type: int):
        """
        Bayes Filter Update.
        Observation is deterministic: We see the tile type.
        Posterior becomes Dirac distribution (One-Hot).
        """
        if 0 <= view_x < self.w and 0 <= view_y < self.h:
            # Deterministic update
            self.belief_map[view_x, view_y, :] = 0.0
            self.belief_map[view_x, view_y, tile_type] = 1.0

    def get_action(self, current_pos: Tuple[int, int]) -> int:
        """
        Active Inference Loop:
        1. Enumerate Policies (sequences of actions)
        2. Evaluate G(pi) for each
        3. Sample action from Softmax(-G)
        """
        self.pos = current_pos
        policies = self._generate_policies(self.horizon)
        g_scores = []
        
        for pi in policies:
            g = self._evaluate_g(pi)
            g_scores.append(g)
            
        g_scores = np.array(g_scores)
        
        # Softmax selection
        # Scale for determinism vs exploration noise
        precision = 5.0
        probs = np.exp(-precision * g_scores)
        if probs.sum() == 0:
             probs = np.ones_like(probs)
        probs /= probs.sum()
        
        # Pick policy
        idx = np.random.choice(len(policies), p=probs)
        best_policy = policies[idx]
        
        # Return first action data
        # Mapping actions back to index 0..3 is needed if caller expects int
        # Our policies are lists of tuples (dx, dy).
        # We need to map back to 0-3 index for consistency if needed, 
        # or just return the (dx, dy).
        # Let's return the action tuple (dx, dy)
        return best_policy[0]

    def _generate_policies(self, depth: int) -> List[List[Tuple[int, int]]]:
        """Recursive generation of all move sequences."""
        if depth == 0:
            return [[]]
        
        sub_policies = self._generate_policies(depth - 1)
        policies = []
        for action in ACTIONS:
            for sub in sub_policies:
                policies.append([action] + sub)
        return policies

    def _evaluate_g(self, policy: List[Tuple[int, int]]) -> float:
        """
        Calculate Expected Free Energy G(pi).
        G = Risk - EpistemicValue
        """
        sim_x, sim_y = self.pos
        total_g = 0.0
        visited_in_plan = set() # To avoid double counting epistemic value for same tile
        
        for action in policy:
            dx, dy = action
            
            # 1. Transition Model (B)
            # Predict next state (Naive physics: Move unless wall)
            # Since we don't know where walls are 100%, we use "Expected" transition?
            # Creating a full POMDP tree is complex.
            # Simplified: Assume move succeeds. If we hit a wall in reality, we verify.
            # For checking "Epistemic Value", we assume we move to the tile.
            
            next_x = max(0, min(self.w - 1, sim_x + dx))
            next_y = max(0, min(self.h - 1, sim_y + dy))
            
            # Check belief about wall?
            # If Q(Wall) is high, maybe we shouldn't simulate moving there.
            # But "Checking if it's a wall" IS informative.
            # So we simulate looking at it.
            
            # 2. Epistemic Value (Information Gain)
            if (next_x, next_y) not in visited_in_plan:
                # Gain = H(Q[x,y])
                entropy = self._entropy(self.belief_map[next_x, next_y])
                total_g -= entropy * 1.0 
                visited_in_plan.add((next_x, next_y))

            # 3. Pragmatic Value (Goal Seeking)
            # Risk/Reward is accumulated EVERY STEP (Eating > finding food once)
            expected_tile_type = np.argmax(self.belief_map[next_x, next_y])
            
            reward = self.preferences.get(expected_tile_type, 0.0)
                
            total_g -= reward
            
            sim_x, sim_y = next_x, next_y
            
        return total_g
