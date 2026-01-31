import numpy as np

# Constants from UI files
GRID_SIZE = 10
PADDLE_H = 6
PADDLE_SPEED = 2
BALL_SPEED_DIVISOR = 3.0  # From Pong UI get_frame logic

def sim_snake(state, action):
    """
    Simulate one step of Snake.
    
    Args:
        state (dict): {"head": (x, y), "snake": [(x, y), ...], "food": (x, y)}
                      Note: "snake" body is needed for collision check. 
                      If not provided in ZMQ state, we need to infer or pass it.
                      Current python_server receives {"head": .., "food": ..} but NOT full body in 'state' dict.
                      Only 'image' has full info. 
                      
                      *CRITICAL FIX*: We will need to update snake_ui.py to send full body, 
                      OR we parse from image (hard/slow), 
                      OR we assume 'snake' is in state for this module and fix UI later.
                      
                      Let's assume we will fix UI to send "body" list.
                      
        action (str): "UP", "DOWN", "LEFT", "RIGHT"
        
    Returns:
        dict: next_state (metadata only)
        bool: collision (True if dead)
    """
    head_x, head_y = state["head"]
    # snake_body = state.get("body", []) # Expecting list of (x,y)
    # Actually, let's use the 'snake' key if present, or just use head if we can't do full check (but full check is needed for safety)
    # For now, let's assume 'body' is passed.
    
    snake_body = state.get("body", []) 
    
    if action == "UP":
        head_y -= 1
    elif action == "DOWN":
        head_y += 1
    elif action == "LEFT":
        head_x -= 1
    elif action == "RIGHT":
        head_x += 1
        
    # STRICT BOUNDARY CHECK
    if head_x < 0 or head_x >= GRID_SIZE or head_y < 0 or head_y >= GRID_SIZE:
        return {"head": (head_x, head_y)}, True # Collision = True
        
    new_head = (head_x, head_y)
    
    # Check Self Collision
    collision = False
    if new_head in snake_body:
        collision = True
        
    return {"head": new_head}, collision

def sim_pong(state, action):
    """
    Simulate one step of Pong (Paddle movement relative to Ball).
    
    Args:
        state (dict): {"ball_y": float, "ball_dy": float, "p1_y": int}
        action (str): "UP", "DOWN", "STAY"
        
    Returns:
        dict: next_state
        bool: miss (True if ball passes paddle without intersection)
    """
    p1_y = state["p1_y"]
    ball_y = state["ball_y"]
    ball_dy = state["ball_dy"]
    # ball_x is not in state dict currently sent by UI! 
    # We need ball_x to know if we are about to miss.
    # Assuming we will add 'ball_x', 'ball_dx' to state.
    
    ball_x = state.get("ball_x", 15) # Default middle if missing
    ball_dx = state.get("ball_dx", -1) 
    
    # Move Paddle
    if action == "UP":
        p1_y -= PADDLE_SPEED
    elif action == "DOWN":
        p1_y += PADDLE_SPEED
    # STAY does nothing
    
    # Clamp Paddle
    img_h_units = 30 # 300px / 10 = 30 units roughly? 
    # UI uses p1_y in range 0..30 roughly? 
    # Check UI: p1_y=10. canvas height 300. 
    # drawing: self.canvas.create_rectangle(0, self.p1_y*10, ...) -> p1_y is in range 0..30.
    # paddle_h = 6.
    max_y = 30 - PADDLE_H
    p1_y = max(0, min(max_y, p1_y))
    
    # Move Ball
    next_ball_x = ball_x + ball_dx
    next_ball_y = ball_y + ball_dy
    
    # Wall Bounce (Y)
    if next_ball_y <= 0 or next_ball_y >= 30:
        # Simple bounce prediction
        # We don't need perfect physics, just "is it reachable?"
        pass
        
    # Check Miss
    # Miss happens if ball passes x=1 (Left paddle x)
    # AND paddle is not covering it.
    
    miss = False
    
    # Critical zone: Ball arriving at x=1
    if ball_dx < 0 and next_ball_x <= 1:
        # Check intersection
        # Paddle Y range: [p1_y, p1_y + PADDLE_H]
        # Ball Y: next_ball_y
        
        paddle_top = p1_y
        paddle_bot = p1_y + PADDLE_H
        
        if not (paddle_top <= next_ball_y <= paddle_bot):
            miss = True
            
    return {"p1_y": p1_y, "ball_x": next_ball_x, "ball_y": next_ball_y}, miss
