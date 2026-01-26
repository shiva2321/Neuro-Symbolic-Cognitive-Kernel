import zmq
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import json
import base64
import cv2
import random
from collections import defaultdict, deque

# --- CONFIGURATION ---
MODEL_PATH = "snn_task_aware.pth" 
SAVE_INTERVAL = 1000 
LEARNING_RATE = 0.0005
GRID_SIZE = 10
ADVERSARIAL_RATE = 0.20
SLEEP_INTERVAL = 1000
SLEEP_EPOCHS = 5
REPLAY_BATCH_SIZE = 32

# Import SNN
from snn_qat import TaskAwareSNN 

class LogAggregator:
    def __init__(self, interval=5.0, zmq_pub=None):
        self.interval = interval
        self.last_print = time.time()
        self.stats = defaultdict(lambda: {"agreements": 0, "interventions": 0, "loss_sum": 0.0, "steps": 0})
        self.zmq_pub = zmq_pub

    def update(self, game, agreed, loss):
        self.stats[game]["steps"] += 1
        self.stats[game]["loss_sum"] += loss
        if agreed:
            self.stats[game]["agreements"] += 1
        else:
            self.stats[game]["interventions"] += 1

    def check_print(self):
        if time.time() - self.last_print > self.interval:
            status_strs = []
            for game, data in self.stats.items():
                total = data["steps"]
                if total == 0: continue
                agree_pct = (data["agreements"] / total) * 100
                avg_loss = data["loss_sum"] / total
                status_strs.append(f"{game.upper()}: AGREE {agree_pct:.1f}% (Loss {avg_loss:.4f})")
                
                # Broadcast Telemetry
                if self.zmq_pub:
                    telemetry = {
                        "game": game,
                        "agree_pct": float(agree_pct),
                        "loss": float(avg_loss),
                        "steps": total,
                        "timestamp": time.time()
                    }
                    self.zmq_pub.send_string(f"STATS:{json.dumps(telemetry)}")
            
            if status_strs:
                print(f"[{time.strftime('%H:%M:%S')}] " + " | ".join(status_strs))
            
            self.stats = defaultdict(lambda: {"agreements": 0, "interventions": 0, "loss_sum": 0.0, "steps": 0})
            self.last_print = time.time()

# --- MEMORY (REPLAY BUFFER) ---
class ReplayBuffer:
    def __init__(self, capacity_per_quadrant=2500):
        self.capacity = capacity_per_quadrant
        # Stratified Buffers: [Game][Agreed?]
        self.buffers = {
            "snake": {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)},
            "pong":  {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)}
        }
    
    def push(self, state, task_id, target, agreed, game_type):
        # Detach and move to CPU to save GPU RAM
        state_cpu = state.detach().cpu()
        
        # Store tuple
        experience = (state_cpu, task_id, target, agreed)
        
        # Route to correct quadrant
        self.buffers[game_type][agreed].append(experience)
        
    def sample(self, batch_size):
        # Stratified Sampling: Attempt to get equal mix
        # Goal: batch_size // 4 from each of the 4 quadrants
        target_per_q = batch_size // 4
        if target_per_q == 0: target_per_q = 1
        
        batch = []
        
        for game in ["snake", "pong"]:
            for agreed in [True, False]:
                buf = self.buffers[game][agreed]
                if len(buf) > 0:
                    # Random sampling from deque
                    # Optimization: If deque is large, random.sample might be slow if converted to list.
                    # But for 2500 items, list conversion is fast enough (~ms).
                    count = min(len(buf), target_per_q)
                    batch.extend(random.sample(buf, count))
                    
        if len(batch) == 0: return None
        
        random.shuffle(batch) # Shuffle mixed batch
        
        # Collate
        states, task_ids, targets, _ = zip(*batch)
        
        return (
            torch.cat(states, dim=0), 
            torch.tensor(task_ids, dtype=torch.float), # Used for task tensor const
            torch.tensor(targets, dtype=torch.long)
        )
        
    def count(self):
        total = 0
        for game in self.buffers:
            for ag in self.buffers[game]:
                total += len(self.buffers[game][ag])
        return total

def sleep_cycle(model, optimizer, buffer, device, epochs=SLEEP_EPOCHS):
    if buffer.count() < REPLAY_BATCH_SIZE: return
    
    print(">> [SLEEP] Consolidating Memories...")
    model.train()
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()
    
    # Revised Sleep Logic: Iterate per buffer to match SNN scalar input expectation
    for _ in range(epochs):
        # Sample Snake
        snake_batch = []
        buf_s_a = buffer.buffers["snake"][True]
        buf_s_d = buffer.buffers["snake"][False]
        if len(buf_s_a) > 0: snake_batch.extend(random.sample(buf_s_a, min(len(buf_s_a), REPLAY_BATCH_SIZE//4)))
        if len(buf_s_d) > 0: snake_batch.extend(random.sample(buf_s_d, min(len(buf_s_d), REPLAY_BATCH_SIZE//4)))
        
        if snake_batch:
            # Snake FWD
            obs, tasks, targs, _ = zip(*snake_batch)
            inp = torch.cat(obs, dim=0).to(device)
            lbl = torch.tensor(targs, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            out = model(inp, 1) # task_id 1 = Snake
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            
        # Sample Pong
        pong_batch = []
        buf_p_a = buffer.buffers["pong"][True]
        buf_p_d = buffer.buffers["pong"][False]
        if len(buf_p_a) > 0: pong_batch.extend(random.sample(buf_p_a, min(len(buf_p_a), REPLAY_BATCH_SIZE//4)))
        if len(buf_p_d) > 0: pong_batch.extend(random.sample(buf_p_d, min(len(buf_p_d), REPLAY_BATCH_SIZE//4)))
        
        if pong_batch:
            # Pong FWD
            obs, tasks, targs, _ = zip(*pong_batch)
            inp = torch.cat(obs, dim=0).to(device)
            lbl = torch.tensor(targs, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            out = model(inp, 0) # task_id 0 = Pong
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            
    if steps > 0:
        print(f"   AVG SLEEP LOSS: {total_loss/steps:.4f}")

# --- HEURISTICS (TEACHERS) ---
def manhattan_snake_move(state, obstacles):
    # GREEDY TEACHER: Move towards food, avoiding immediate collision.
    hx, hy = state["head"]
    fx, fy = state["food"]
    full_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    best_move = 0
    min_dist = 999
    
    for i, (dx, dy) in enumerate(full_deltas):
        nx, ny = (hx + dx) % GRID_SIZE, (hy + dy) % GRID_SIZE
        if (nx, ny) in obstacles: continue 
        dist = abs(nx - fx) + abs(ny - fy)
        if dist < min_dist:
            min_dist = dist
            best_move = i
            
    if min_dist == 999: # Trapped
        for i, (dx, dy) in enumerate(full_deltas):
            nx, ny = (hx + dx) % GRID_SIZE, (hy + dy) % GRID_SIZE
            if (nx, ny) not in obstacles: return i
        
    return best_move

def get_snake_oracle(state, img):
    hx, hy = state["head"]
    fx, fy = state["food"]
    obstacles = set() 
    body = np.where(img == 255)
    for i in range(len(body[0])):
        px, py = body[1][i], body[0][i]
        if px != hx or py != hy: obstacles.add((px, py))
    return manhattan_snake_move(state, obstacles)

def get_pong_oracle(state):
    by = state["ball_y"]
    py = state["p1_y"]
    if py + 3 < by - 1: return 1 # DOWN
    if py + 3 > by + 1: return 0 # UP
    return 0 

def main():
    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL); pull_sock.bind("tcp://127.0.0.1:5555")
    pub_sock = context.socket(zmq.PUB); pub_sock.bind("tcp://127.0.0.1:5556")
    # New: Telemetry Socket
    pub_sock_stats = context.socket(zmq.PUB); pub_sock_stats.bind("tcp://127.0.0.1:5557")
    
    device = torch.device("cpu")
    
    # NEW: TaskAwareSNN (Late Fusion)
    model = TaskAwareSNN(beta=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("loaded weights.")
        except Exception as e:
            print(f"Starting fresh (New Arch: {e})")
    else:
         print("Starting fresh.")
    
    logger = LogAggregator(zmq_pub=pub_sock_stats) # Pass socket
    buffer = ReplayBuffer() # Initialize Memory
    
    history = {"snake": deque(maxlen=4), "pong": deque(maxlen=4)}
    
    print(">> NEURO-SYMBOLIC SNN: DASHBOARD ENABLED (Port 5557)...")
    
    steps_total = 0
    last_game_type = None

    while True:
        try:
            msg = pull_sock.recv_json()
            
            # --- CHECK FOR ADMIN COMMANDS ---
            if msg.get("type") == "admin":
                cmd = msg.get("cmd")
                print(f"⚠️ ADMIN COMMAND RECEIVED: {cmd}")
                if cmd == "force_sleep":
                    sleep_cycle(model, optimizer, buffer, device)
                elif cmd == "reset_memory":
                     buffer = ReplayBuffer()
                continue # Skip game logic
            
            game_type = msg.get("game", "unknown")
            
            # --- SLEEP TRIGGER: TASK SWITCH ---
            if last_game_type is not None and game_type != last_game_type:
                # Context Switch detected: SLEEP to consolidate previous context
                print(f"🔁 Task Switch ({last_game_type}->{game_type}): Triggering Sleep...")
                sleep_cycle(model, optimizer, buffer, device)
            
            last_game_type = game_type
            
            state_data = msg.get("state", {})
            img_bytes = base64.b64decode(msg["image"])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img.shape != (10, 10): img = cv2.resize(img, (10, 10))
            
            # Adversarial (Snake)
            obstacles_extra = []
            if game_type == "snake" and random.random() < ADVERSARIAL_RATE:
                hx, hy = state_data["head"]
                fx, fy = state_data["food"]
                mx, my = (hx+fx)//2, (hy+fy)//2
                if (mx!=hx or my!=hy) and (mx!=fx or my!=fy): obstacles_extra.append((mx,my))

            # Teacher
            teacher_idx = 0
            task_id = 0 
            
            if game_type == "snake":
                task_id = 1
                snake_obs = set()
                body = np.where(img==255)
                for i in range(len(body[0])): snake_obs.add((body[1][i], body[0][i]))
                for obs in obstacles_extra: snake_obs.add(obs)
                teacher_idx = manhattan_snake_move(state_data, snake_obs)
            elif game_type == "pong":
                task_id = 0
                teacher_idx = get_pong_oracle(state_data)

            # Input Processing
            curr_np = img.astype(np.float32) / 255.0
            for obs in obstacles_extra:
                 if 0<=obs[1]<10 and 0<=obs[0]<10: curr_np[obs[1], obs[0]] = 0.5
            
            q = history[game_type]
            if len(q) == 0:
                for _ in range(3): q.append(curr_np)
            q.append(curr_np)
            frames_list = list(q)
            while len(frames_list) < 4: frames_list.insert(0, frames_list[0])
            stacked_np = np.stack(frames_list, axis=0) # [4, 10, 10]
            
            input_tensor = torch.from_numpy(stacked_np).float().to(device) 
            input_tensor = input_tensor.unsqueeze(0) # [1, 4, 10, 10]
            
            # Forward
            optimizer.zero_grad()
            rate_out = model(input_tensor, task_id) 
            
            # --- DISAGREEMENT GATE ---
            probs = torch.softmax(rate_out, dim=1)
            _, student_idx_tensor = torch.max(probs, dim=1)
            student_idx = student_idx_tensor.item()
            
            # 1. Check Agreement
            agreed = (student_idx == teacher_idx)
            
            loss_val = 0.0
            
            if agreed:
                # Student is Correct -> Let Student Drive, NO Training
                action_idx = student_idx
                loss_val = 0.0 
            else:
                # Student is Wrong -> Intervene, TEACHING Signal
                action_idx = teacher_idx
                target = torch.tensor([teacher_idx], dtype=torch.long, device=device)
                loss = criterion(rate_out, target)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                
            # MEMORY: Store Experience
            buffer.push(input_tensor, task_id, teacher_idx, agreed, game_type)
            
            logger.update(game_type, agreed, loss_val)
            logger.check_print()
            
            # --- VISUALIZATION TELEMETRY (Every 10 Steps) ---
            if steps_total % 10 == 0:
                # Prepare Data
                grid_list = curr_np.tolist() # 10x10 list
                probs_list = probs[0].tolist() # List of floats
                vis_payload = {
                    "grid": grid_list,
                    "probs": probs_list,
                    "teacher": teacher_idx,
                    "student": student_idx,
                    "agreed": bool(agreed),
                    "task": game_type
                }
                pub_sock_stats.send_string(f"VIS:{json.dumps(vis_payload)}")

            actions = ["UP", "DOWN", "LEFT", "RIGHT"]
            cmd = "UP"
            if game_type == "snake":
                cmd = actions[action_idx] if action_idx < 4 else "UP"
            else: # Pong
                cmd = "UP" if action_idx == 0 else "DOWN"
            
            pub_sock.send_string(f"{game_type.upper()}:{cmd}")
            
            steps_total += 1
            
            if steps_total % SAVE_INTERVAL == 0: 
                 torch.save(model.state_dict(), MODEL_PATH)
            
            # --- SLEEP TRIGGER: INTERVAL ---
            if steps_total % SLEEP_INTERVAL == 0:
                 sleep_cycle(model, optimizer, buffer, device)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()




