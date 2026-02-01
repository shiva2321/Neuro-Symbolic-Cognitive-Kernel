import zmq
import time
import numpy as np
import base64
import cv2
import uuid
import random

def main():
    context = zmq.Context()
    
    # Connect to Server
    push = context.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0) # Don't hang on close, but we want to send data...
    push.connect("tcp://127.0.0.1:5565")
    
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5566")
    sub.setsockopt_string(zmq.SUBSCRIBE, "SNAKE:")
    
    session_id = str(uuid.uuid4())[:8]
    print(f"Starting Headless Snake Client [{session_id}]")
    
    # Mock State
    snake = [(5, 5), (5, 6), (5, 7)]
    food = (2, 2)
    
    steps = 0
    while steps < 200: # detailed test
        # Create Dummy Image (10x10)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[food[1], food[0]] = 128
        for x, y in snake:
            img[y, x] = 255
            
        _, buf = cv2.imencode('.png', img)
        b64 = base64.b64encode(buf).decode('utf-8')
        
        payload = {
            "game": "snake",
            "session_id": session_id,
            "image": b64,
            "state": {
                "head": snake[0],
                "food": food,
                "body": snake
            },
            "score": 0,
            "reward": -0.1, # Step penalty
            "done": False
        }
        
        push.send_json(payload)
        if steps % 10 == 0: print(f"Sent step {steps}")
        
        # Wait for Action
        try:
            msg = sub.recv_string(flags=zmq.NOBLOCK)
            # format "SNAKE:UP"
            print(f"Server sent: {msg}")
        except zmq.Again:
            pass
            
        time.sleep(0.05) # fast simulation
        steps += 1
        
        # Simple random walk for the "snake" to change state
        hx, hy = snake[0]
        hx += random.choice([-1, 0, 1])
        hy += random.choice([-1, 0, 1])
        snake.insert(0, (max(0, min(9, hx)), max(0, min(9, hy))))
        snake.pop()

if __name__ == "__main__":
    main()
