
import zmq
import time
import numpy as np
import base64
import cv2
import uuid
import random
import sys
import os

# Imports from our new brains
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agency import ActiveAgent, TILE_UNKNOWN, TILE_EMPTY, TILE_WALL, TILE_FOOD
from homeostasis import HomeostaticMonitor

def main():
    context = zmq.Context()
    
    # Connect to Server (Optional - we are simulating locally for verification)
    # But we keep the socket logic to be drop-in ready.
    push = context.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect("tcp://127.0.0.1:5565")
    
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5566")
    sub.setsockopt_string(zmq.SUBSCRIBE, "SNAKE:")
    
    session_id = str(uuid.uuid4())[:8]
    print(f"Starting Sentient Snake Client [{session_id}]")
    
    # --- BRAIN SETUP ---
    # 1. Biological Self
    body = HomeostaticMonitor()
    
    # 2. Cognitive Agent
    agent = ActiveAgent(w=10, h=10, horizon=3)
    
    # --- GAME STATE (MOCK) ---
    snake = [(5, 5), (5, 6), (5, 7)]
    food = (2, 2)
    score = 0
    
    # Initial Belief about Self (Agent knows where it starts)
    for x, y in snake:
        agent.update_belief(x, y, TILE_EMPTY) # Body is empty? No, body is body.
        # ActiveAgent doesn't strictly Model "Body" vs "Empty" in the TILE enums yet.
        # We'll treat Body as Wall for collision avoidance?
        # Let's say Body = Wall for navigation safety.
        agent.update_belief(x, y, TILE_WALL)
        
    # Initial Belief about Food (Cheat: Let's assume it smells the food if close?)
    # Or just let it explore.
    
    steps = 0
    max_steps = 100
    
    print("\n--- BEGIN SIMULATION ---\n")
    
    while steps < max_steps:
        # A. UPDATE BIOLOGY
        # Energy decay per step (Simulate 20 ticks per frame to speed up hunger)
        # Decay rate is 0.001. 20 * 0.001 = 0.02 loss per step.
        # 50 steps = -1.0 energy.
        body.update(tick_duration=20.0)
        
        
        # B. UPDATE DRIVES -> PREFERENCES
        # Drives updated in update() automatically
        hunger = body.drives['hunger']
        
        # Dynamic Preferences
        if hunger > 0.3: # Lower threshold to see effect sooner
            agent.preferences[TILE_FOOD] = 20.0 # High value on food
            mode = "SURVIVAL (Seeking Food)"
        else: # Satiated
            agent.preferences[TILE_FOOD] = 0.0 # Don't care
            mode = "CURIOSITY (Exploring)"
            
        # C. PERCEPTION (Update Beliefs)
        head = snake[0]
        
        # VISION: See 3x3 around head
        # We need to query the 'True' map (which we build on fly here)
        # Construct True Grid for current state
        true_grid = np.full((10, 10), TILE_EMPTY)
        true_grid[food[1], food[0]] = TILE_FOOD
        for (bx, by) in snake:
            true_grid[by, bx] = TILE_WALL # Treat body as wall
            
        # Update Beliefs
        vx, vy = head
        scan_radius = 1
        for dy in range(-scan_radius, scan_radius+1):
            for dx in range(-scan_radius, scan_radius+1):
                tx, ty = vx + dx, vy + dy
                if 0 <= tx < 10 and 0 <= ty < 10:
                    tile_type = true_grid[ty, tx]
                    agent.update_belief(tx, ty, tile_type)
                    
        # D. ACTION (Active Inference)
        # Agent decides move
        move = agent.get_action(head) # returns (dx, dy)
        
        # E. EXECUTION (Physics)
        dx, dy = move
        nx, ny = head[0] + dx, head[1] + dy
        
        # Boundary Check
        nx = max(0, min(9, nx))
        ny = max(0, min(9, ny))
        new_head = (nx, ny)
        
        # Logic
        reward = -0.1
        ate_food = False
        
        # Collision?
        if new_head in snake:
            print(f"CRASH! Hit self at {new_head}. Integrity -0.2.")
            body.consume("integrity", -0.2)
            # Respawn/Reset? Just bounce for now to keep sim running
            new_head = head # Don't move
        elif new_head == food:
            print(f"OM NOM NOM! Ate food at {new_head}.")
            score += 1
            reward = 10.0
            ate_food = True
            body.consume("energy", 0.5) # Restore energy (using consume)
            
            # Respawn food
            while True:
                fx, fy = random.randint(0,9), random.randint(0,9)
                if (fx, fy) not in snake and (fx, fy) != new_head:
                    food = (fx, fy)
                    break
            # Note: We need to set old food tile to Empty in belief? 
            # ActiveAgent doesn't Auto-Clear beliefs.
            agent.update_belief(nx, ny, TILE_EMPTY) # Now it's empty
            
        else:
            # Just moved
            pass
            
        # Move Snake Body
        if new_head != head:
            snake.insert(0, new_head)
            if not ate_food:
                snake.pop()
        
        # LOGGING
        print(f"Step {steps:03d} | Pos: {head} | Energy: {body.energy:.2f} | Hunger: {hunger:.2f} | Mode: {mode} | Action: {move}")
        
        time.sleep(0.05)
        steps += 1
        
    print("\n--- SIMULATION END ---")
    print(f"Final Score: {score}")
    print(f"Final Integrity: {body.integrity}")

if __name__ == "__main__":
    main()
