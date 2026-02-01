
import sys
import os

# Add the project path to sys.path
sys.path.insert(0, r"d:\NSCK_v1\nsck-demo\python")

from maze_game import MazeGame
from spatial_reasoning import GridPlanner

def test_movement():
    print("Initializing MazeGame...")
    game = MazeGame(width=15, height=15, wall_density=0.1)
    state_dict = game.get_state_dict()
    
    print(f"Initial Player Pos: {state_dict['player_pos']}")
    print(f"Goal Pos: {state_dict['exit_pos']}")
    
    planner = GridPlanner(grid_size=15)
    
    # Try multiple steps
    for i in range(5):
        current_state = game.get_state_dict()
        action = planner.get_next_action(current_state, "maze")
        
        if not action:
            print("No action found by planner!")
            break
            
        print(f"Step {i+1}: Planner suggested {action}")
        
        # Simulate execution
        next_state, reward, done = game.step(action)
        print(f"Moved to: {next_state.player_pos}, Reward: {reward}, Done: {done}")
        
        if next_state.player_pos == current_state['player_pos']:
            print("WARNING: Agent did not move!")
            # Check why
            x, y = current_state['player_pos']
            action_map = {"ACTION_UP": (0, -1), "ACTION_DOWN": (0, 1), "ACTION_LEFT": (-1, 0), "ACTION_RIGHT": (1, 0)}
            dx, dy = action_map.get(action, (0, 0))
            nx, ny = x + dx, y + dy
            
            if not (0 <= nx < 15 and 0 <= ny < 15):
                print(f"Reason: Out of bounds ({nx}, {ny})")
            elif game.state.maze[ny][nx] == 1: # Cell.WALL.value
                print(f"Reason: Hit wall at ({nx}, {ny})")
            else:
                print("Reason: Unknown! Step function didn't update pos?")

if __name__ == "__main__":
    test_movement()
