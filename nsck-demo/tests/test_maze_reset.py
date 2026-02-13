import sys
import os
import time
import random

# Add python dir to path

from python.games.maze.maze_game import MazeGame

def test_maze_reset_stress():
    print("Starting Maze Reset Stress Test (100 iterations)...")
    game = MazeGame(width=15, height=15)
    
    start_time = time.time()
    
    for i in range(100):
        try:
            game.reset()
            # Verify basics
            if not game.state:
                print(f"FAIL: State is None on iter {i}")
                return
            if game.state.done:
                print(f"FAIL: Game starts as Done on iter {i}")
                return
            
            # Optional: Simulate a win
            game.state.player_pos = game.state.exit_pos
            game.state.done = True
            
        except Exception as e:
            print(f"CRASH on iter {i}: {e}")
            return
            
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1} resets...")
            
    end_time = time.time()
    print(f"Stress Test Passed in {end_time - start_time:.4f}s")

if __name__ == "__main__":
    test_maze_reset_stress()
