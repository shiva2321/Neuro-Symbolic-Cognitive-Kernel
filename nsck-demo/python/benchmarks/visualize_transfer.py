import numpy as np
import matplotlib.pyplot as plt
import cv2
from symbol_grounding import ActionSemantics

def plot_grounding():
    # Setup 2 Dummy States
    # Snake: Head at (5,5), Food at (5,2) -> GOAL is UP
    snake_state = {
        "head": (5, 5),
        "food": (5, 2),
        "body": [],
        "grid_size": 10
    }
    
    # Pong: Ball at (9, 2), Paddle at (9, 8) -> GOAL is UP (to catch)
    # Wait, coordinate system: (x, y). 
    # Snake: (5,2) is typically "higher" (lower index) or "lower"?
    # Usually grid origin (0,0) is top-left. So (5,2) is ABOVE (5,5).
    # Pong: Ball Y=2, Paddle Y=8. Ball is ABOVE puddle. Paddle must move UP (decrease Y).
    
    pong_state = {
        "ball_y": 2,
        "p1_y": 8,
        "grid_size": 10
    }
    
    # Get Vectors
    # Note: `get_goal_alignment` returns a probability distribution (prior)
    snake_prior = ActionSemantics.get_goal_alignment("snake", snake_state)
    pong_prior = ActionSemantics.get_goal_alignment("pong", pong_state)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Snake Plot
    actions_snake = ["UP", "DN", "LF", "RT"]
    axes[0].bar(actions_snake, snake_prior, color=['green', 'gray', 'gray', 'gray'])
    axes[0].set_title("Task A: Snake\nConcept: GOAL (Food)\nAction: UP")
    axes[0].set_ylim(0, 1.1)
    
    # Pong Plot
    actions_pong = ["UP", "DN"]
    axes[1].bar(actions_pong, pong_prior, color=['green', 'gray'])
    axes[1].set_title("Task B: Pong\nConcept: GOAL (Ball)\nAction: UP")
    axes[1].set_ylim(0, 1.1)
    
    plt.suptitle("Neuro-Symbolic Grounding: Zero-Shot Transfer (Predicate Verification)", fontsize=14)
    plt.tight_layout()
    plt.savefig("transfer_proof.png")
    print("Generated transfer_proof.png")

if __name__ == "__main__":
    plot_grounding()
