import tkinter as tk
import zmq
import threading
import numpy as np
import cv2
import base64
import json
import random
import uuid
import time

class PongGame:
    def __init__(self, root):
        self.root = root
        self.session_id = str(uuid.uuid4())[:8]
        self.root.title(f"NSCK Pong (Student vs AI) [{self.session_id}]")
        self.canvas = tk.Canvas(root, width=300, height=300, bg="black")
        self.canvas.pack()
        
        self.reset_ball()
        self.paddle_h = 6
        self.p1_y = 10 
        self.p2_y = 10 
        self.score_p1 = 0
        self.score_p2 = 0
        self.rally_count = 0
        
        # RL Metrics
        self.current_reward = 0.0
        self.done = False
        
        self.context = zmq.Context()
        self.push = self.context.socket(zmq.PUSH)
        self.push.connect("tcp://127.0.0.1:5565")
        
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:5566")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "PONG:")
        self.sub.setsockopt(zmq.RCVTIMEO, 100) # 100ms Timeout (Frame is ~30ms, so 3 frames dropped max)
        
        self.last_key = None
        
        # Bind Keys for Manual Teacher Override
        self.root.bind("<Up>", lambda e: self.set_key("UP"))
        self.root.bind("<Down>", lambda e: self.set_key("DOWN"))
        
        threading.Thread(target=self.network_loop, daemon=True).start()
        self.game_loop()

    def set_key(self, key):
        self.last_key = key

    def reset_ball(self):
        self.ball_x = 15
        self.ball_y = 15
        self.ball_dx = 1 if random.random() > 0.5 else -1
        self.ball_dy = random.choice([-1, -0.5, 0.5, 1])
        self.rally_count = 0

    def get_frame(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        # ALIGNED: Ball=128 (Target), Paddle=255 (Self)
        bx, by = int(self.ball_x / 3), int(self.ball_y / 3)
        if 0 <= bx < 10 and 0 <= by < 10:
            img[by, bx] = 128
            
        py_start = int(self.p1_y / 3)
        py_end = int((self.p1_y + self.paddle_h) / 3)
        for y in range(max(0, py_start), min(10, py_end + 1)):
            img[y, 0] = 255
        
        # Enemy
        p2_start = int(self.p2_y / 3)
        p2_end = int((self.p2_y + self.paddle_h) / 3)
        for y in range(max(0, p2_start), min(10, p2_end + 1)):
            img[y, 9] = 50
            
        return img

    def network_loop(self):
        print("PONG: Network Loop Started")
        while True:
            try:
                img = self.get_frame()
                _, buf = cv2.imencode('.png', img)
                b64 = base64.b64encode(buf).decode('utf-8')
                
                state = {
                    "ball_y": self.ball_y,
                    "ball_dy": self.ball_dy,
                    "p1_y": self.p1_y,
                    "p2_y": self.p2_y, # Opponent paddle
                    "ball_x": self.ball_x,
                    "ball_dx": self.ball_dx
                }
                
                payload = {
                    "game": "pong", 
                    "session_id": self.session_id,
                    "image": b64, 
                    "state": state,
                    "score": self.rally_count, # Metric for transfer success
                    "reward": self.current_reward,
                    "done": self.done,
                    "teacher_voice": self.last_key # Manual Override
                }
                self.push.send_json(payload)
                self.last_key = None
                
                # Receive with Timeout
                try:
                    msg = self.sub.recv_string()
                    cmd = msg.split(":")[1]
                    if cmd == "UP": self.p1_y -= 2
                    elif cmd == "DOWN": self.p1_y += 2
                    self.p1_y = max(0, min(30 - self.paddle_h, self.p1_y))
                except zmq.Again:
                    # Timeout: Just continue loop (resend frame)
                    pass
                    
            except Exception as e:
                print(f"PONG NETWORK ERROR: {e}")
                time.sleep(1) # Prevent log spam on crash 



    def move_ai_paddle(self):
        if random.random() < 0.85: 
            if self.p2_y + self.paddle_h/2 < self.ball_y:
                self.p2_y += 1
            elif self.p2_y + self.paddle_h/2 > self.ball_y:
                self.p2_y -= 1
        self.p2_y = max(0, min(30 - self.paddle_h, self.p2_y))

    def game_loop(self):
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        if self.ball_y <= 0 or self.ball_y >= 30: self.ball_dy *= -1
        
        # Reset params
        self.current_reward = 0.0
        self.done = False
        
        # AI Opponent Move
        self.move_ai_paddle()

        if self.ball_x <= 1:
            if self.p1_y <= self.ball_y <= self.p1_y + self.paddle_h:
                self.ball_dx *= -1
                self.ball_dx = abs(self.ball_dx)
                self.rally_count += 1
                self.current_reward = 1.0 # Hit Reward
            else:
                self.score_p2 += 1
                self.rally_count = 0
                self.current_reward = -10.0 # Miss Penalty
                self.done = True
                self.reset_ball()
                
        if self.ball_x >= 29:
            if self.p2_y <= self.ball_y <= self.p2_y + self.paddle_h:
                self.ball_dx *= -1
                self.ball_dx = -abs(self.ball_dx) 
            else:
                self.score_p1 += 1
                self.current_reward = 10.0 # Win Reward
                self.done = True
                self.reset_ball()

        self.move_ai_paddle()
        
        self.canvas.delete("all")
        self.canvas.create_oval(self.ball_x*10, self.ball_y*10, (self.ball_x+1)*10, (self.ball_y+1)*10, fill="white")
        self.canvas.create_rectangle(0, self.p1_y*10, 10, (self.p1_y+self.paddle_h)*10, fill="#00FF00") 
        self.canvas.create_rectangle(290, self.p2_y*10, 300, (self.p2_y+self.paddle_h)*10, fill="#FF0000") 
        self.canvas.create_text(150, 20, text=f"{self.score_p1} - {self.score_p2} (Rally: {self.rally_count})", fill="white", font=("Arial", 16))
        
        self.root.after(40, self.game_loop)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Hide the redundant engine window
    game = PongGame(root)
    root.mainloop()
