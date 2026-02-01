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
    snake_body = state.get("body", [])

    if action == "UP":
        head_y -= 1
    elif action == "DOWN":
        head_y += 1
    elif action == "LEFT":
        head_x -= 1
    elif action == "RIGHT":
        head_x += 1

    # Wraparound (toroidal)
    head_x %= GRID_SIZE
    head_y %= GRID_SIZE

    new_head = (head_x, head_y)

    # Self collision
    collision = new_head in snake_body

    return {"head": new_head}, collision

def sim_pong(state, action):
    """
    Simulate one step of Pong (very lightweight).

    This is used mainly for safety veto / "will we miss" heuristics.
    Some tests/callers provide partial state; we default missing dynamics.
    """
    p1_y = state["p1_y"]
    ball_y = state["ball_y"]
    ball_dy = state.get("ball_dy", 0.0)

    # Optional x dynamics (missing in some callers)
    ball_x = state.get("ball_x", 15)
    ball_dx = state.get("ball_dx", -1)

    # Move Paddle
    if action == "UP":
        p1_y -= PADDLE_SPEED
    elif action == "DOWN":
        p1_y += PADDLE_SPEED

    # Clamp Paddle
    max_y = 30 - PADDLE_H
    p1_y = max(0, min(max_y, p1_y))

    # Move Ball (very approximate)
    next_ball_x = ball_x + ball_dx
    next_ball_y = ball_y + ball_dy

    # Miss check: only meaningful when the ball is moving toward the paddle
    miss = False
    if ball_dx < 0 and next_ball_x <= 1:
        paddle_top = p1_y
        paddle_bot = p1_y + PADDLE_H
        if not (paddle_top <= next_ball_y <= paddle_bot):
            miss = True

    return {"p1_y": p1_y, "ball_x": next_ball_x, "ball_y": next_ball_y, "ball_dx": ball_dx, "ball_dy": ball_dy}, miss
