
import sys
import os
import numpy as np
import random
from scipy import stats

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agency import ActiveAgent, TILE_UNKNOWN, TILE_EMPTY, TILE_WALL, TILE_FOOD

# --- MOCK ENVIRONMENT ---
class GridEnv:
    def __init__(self, w=10, h=10):
        self.w = w
        self.h = h
        self.grid = np.zeros((w, h), dtype=int)
        self.grid[:, :] = TILE_EMPTY
        
        # Add some walls
        for i in range(w):
             if random.random() < 0.1: self.grid[i, 2] = TILE_WALL
        
        self.agent_pos = (0, 0)
        self.visited = set()
        self.visited.add(self.agent_pos)

    def step(self, action):
        dx, dy = action
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        
        # Boundary check
        nx = max(0, min(self.w - 1, nx))
        ny = max(0, min(self.h - 1, ny))
        
        # Wall check
        if self.grid[nx, ny] != TILE_WALL:
            self.agent_pos = (nx, ny)
            
        self.visited.add(self.agent_pos)
        
        # Return observation (Current tile)
        # In full agency, we obs local window. Let's assume agent looks at current tile.
        return self.agent_pos, self.grid[nx, ny]

def run_trial(agent_type="active", steps=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    env = GridEnv(10, 10)
    agent = ActiveAgent(10, 10, horizon=3) if agent_type == "active" else None
    
    for _ in range(steps):
        if agent_type == "active":
            # Active Agent
            # 1. Update Belief (Active Inf needs to see before acting)
            x, y = env.agent_pos
            tile = env.grid[x, y]
            agent.update_belief(x, y, tile)
            
            # 2. Act
            action = agent.get_action((x, y))
        else:
            # Random Agent
            idx = random.randint(0, 3)
            # Map index to action tuple
            actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            action = actions[idx]
            
        env.step(action)
        
    return len(env.visited)

def test_agency():
    print("Initializing Agency verification...")
    
    n_seeds = 20
    steps = 40
    
    scores_active = []
    scores_random = []
    
    print(f"Running {n_seeds} trials (Active vs Random)...")
    
    for i in range(n_seeds):
        cov_a = run_trial("active", steps, seed=i)
        cov_r = run_trial("random", steps, seed=i)
        
        scores_active.append(cov_a)
        scores_random.append(cov_r)
        
    mean_a = np.mean(scores_active)
    mean_r = np.mean(scores_random)
    std_a = np.std(scores_active)
    std_r = np.std(scores_random)
    
    # Calculate Cohen's d
    # d = (M1 - M2) / pooled_std
    # pooled_std = sqrt((s1^2 + s2^2)/2)
    pooled_std = np.sqrt((std_a**2 + std_r**2) / 2)
    cohen_d = (mean_a - mean_r) / (pooled_std + 1e-6)
    
    print(f"\nResults (N={n_seeds}, Steps={steps}):")
    print(f"Active Agent Coverage: {mean_a:.1f} (+/- {std_a:.1f})")
    print(f"Random Agent Coverage: {mean_r:.1f} (+/- {std_r:.1f})")
    print(f"Effect Size (Cohen's d): {cohen_d:.2f}")
    
    # Success Criteria
    # Mean > Mean + 3*SE (Standard Error = Std / sqrt(N))
    se_combined = np.sqrt(std_a**2/n_seeds + std_r**2/n_seeds) # Approx
    
    if mean_a > mean_r + 2.0: # Simple threshold
         pass_margin = True
    else:
         pass_margin = False
         
    if cohen_d > 0.8: # Large effect size
        print(">> PASSED: Active Agent systematically explores significantly better.")
    else:
        print(">> FAILED: Insufficient improvement over random.")

    # --- Test 2: Goal Seeking ---
    print("\n[Test 2] Goal Seeking (Food)")
    
    # Setup: Agent at (0,0). Food at (2,0).
    # Agent needs to know food is there to plan for it?
    # Or stumble upon it?
    # If we set belief manually, agent should execute path to food immediately.
    
    agent = ActiveAgent(5, 5, horizon=3)
    
    # Inject Knowledge: "There is food at (2, 0)"
    agent.belief_map[2, 0, :] = 0.0
    agent.belief_map[2, 0, TILE_FOOD] = 1.0
    
    # Expected Path: (1,0) -> (2,0)
    print("Injecting belief: Food at (2,0). Agent at (0,0).")
    
    # Ask for action
    action = agent.get_action((0,0))
    print(f"Action taken: {action}")
    
    if action == (1, 0): # Move Right towards food
        print(">> PASSED: Agent moved towards known food.")
    else:
        print(f">> FAILED: Agent moved {action} instead of (1,0).")

if __name__ == "__main__":
    test_agency()
