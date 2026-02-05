import tkinter as tk
import zmq
import threading
import numpy as np
import cv2
import base64
import random
import uuid
import time

class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.session_id = str(uuid.uuid4())[:8]
        self.root.title(f"NSCK Snake (Student) [{self.session_id}]")
        self.canvas = tk.Canvas(root, width=300, height=300, bg="black")
        self.canvas.pack()
        
        self.snake = [(5, 5), (5, 6), (5, 7)]
        self.food = (2, 2)
        self.direction = "UP"
        
        # RL Metrics
        self.current_reward = 0.0
        self.done = False
        
        self.context = zmq.Context()
        self.push = self.context.socket(zmq.PUSH)
        self.push.connect("tcp://127.0.0.1:5565")
        
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:5566")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "SNAKE:")
        
        self.last_key = None
        
        # Bind Keys for Manual Teacher Override
        self.root.bind("<Up>", lambda e: self.set_key("UP"))
        self.root.bind("<Down>", lambda e: self.set_key("DOWN"))
        self.root.bind("<Left>", lambda e: self.set_key("LEFT"))
        self.root.bind("<Right>", lambda e: self.set_key("RIGHT"))
        
        threading.Thread(target=self.network_loop, daemon=True).start()
        self.game_loop()

    def set_key(self, key):
        self.last_key = key

    def get_frame(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        # ALIGNED: Self=255, Target=128
        img[self.food[1], self.food[0]] = 128
        for x, y in self.snake:
            img[y, x] = 255
        return img

    def network_loop(self):
        while True:
            img = self.get_frame()
            _, buf = cv2.imencode('.png', img)
            b64 = base64.b64encode(buf).decode('utf-8')
            
            # ORACLE STATE
            state = {
                "head": self.snake[0],
                "food": self.food,
                "body": self.snake
            }
            
            payload = {
                "game": "snake", 
                "session_id": self.session_id,
                "image": b64, 
                "state": state,
                "score": len(self.snake) - 3, # Score = Apples Eaten
                "reward": self.current_reward,
                "done": self.done,
                "teacher_voice": self.last_key # Manual Override
            }
            self.push.send_json(payload)
            self.last_key = None # Reset after sending
            
            msg = self.sub.recv_string() 
            self.direction = msg.split(":")[1]

    def game_loop(self):
        head_x, head_y = self.snake[0]
        if self.direction == "UP": head_y -= 1
        elif self.direction == "DOWN": head_y += 1
        elif self.direction == "LEFT": head_x -= 1
        elif self.direction == "RIGHT": head_x += 1
        
        # WALL DEATH (Strict Transfer Mode)
        if head_x < 0 or head_x >= 10 or head_y < 0 or head_y >= 10:
            self.current_reward = -10.0 # Death Penalty
            self.done = True
            self.snake = [(5,5)] 
        else:
            new_head = (head_x, head_y)
            self.current_reward = -0.1 # Step Penalty
            self.done = False
            
            if new_head in self.snake: 
                self.current_reward = -10.0 # Death Penalty
                self.done = True
                self.snake = [(5,5)] 
            else:
                self.snake.insert(0, new_head)
                if new_head == self.food:
                    self.current_reward = 10.0 # Goal Reward
                    self.food = (random.randint(0,9), random.randint(0,9))
                else:
                    self.snake.pop()
        
        self.canvas.delete("all")
        self.canvas.create_rectangle(self.food[0]*30, self.food[1]*30, (self.food[0]+1)*30, (self.food[1]+1)*30, fill="red")
        for x, y in self.snake:
            self.canvas.create_rectangle(x*30, y*30, (x+1)*30, (y+1)*30, fill="green")
            
        self.root.after(100, self.game_loop)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Hide the redundant engine window
    SnakeGame(root)
    root.mainloop()
