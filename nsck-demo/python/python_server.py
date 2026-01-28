import zmq
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import threading
import json
import base64
import cv2
import argparse
import random
from collections import defaultdict, deque
from symbol_grounding import ActionSemantics
from simulation import sim_snake, sim_pong
from character_dataset import get_dataloader # New Import
from concept_mapper import ConceptMapper
import threading
import queue

# --- ABLATION FLAGS (DEFAULTS) ---
ENABLE_SNN = True
ENABLE_VSA = True
ENABLE_SLEEP = True
FREEZE_PONG = False
NO_TEACHER = False # If True, we never fallback to Teacher. System must survive on its own.

# --- HYPERPARAMETERS ---
SLEEP_EPOCHS = 5
REPLAY_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MODEL_PATH = "snn_task_aware.pth"
SAVE_INTERVAL = 60.0
SLEEP_INTERVAL = 10.0
ADVERSARIAL_RATE = 0.05
GRID_SIZE = 10
VSA_STRENGTH = 5.0 
CONFIDENCE_THRESHOLD = 0.6 # Low Entropy = High Confidence. Threshold for "Confusion".
# If Entropy > 0.6, we consider the SNN "Confused" and apply VSA. 
# Max entropy for 4 classes is ln(4) ~= 1.38. 
# 0.6 is reasonably confident. 

# Import SNN
from snn_qat import TaskAwareSNN 
from simulation import sim_snake, sim_pong # Import Simulation Logic
from symbol_grounding import ActionSemantics # Import Grounding Logic

class LogAggregator:
    def __init__(self, interval=5.0, zmq_pub=None):
        self.interval = interval
        self.last_print = time.time()
        self.stats = defaultdict(lambda: {"agreements": 0, "interventions": 0, "loss_sum": 0.0, "steps": 0, "session_id": "unknown", "score": 0})
        self.zmq_pub = zmq_pub

    def update(self, game, agreed, loss, score, session_id="unknown"):
        self.stats[game]["session_id"] = session_id
        self.stats[game]["steps"] += 1
        self.stats[game]["loss_sum"] += loss
        self.stats[game]["score"] = max(self.stats[game]["score"], score) # Keep max score seen in interval

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
                score = data["score"]
                
                # Format: SNAKE: AGREE 95.0% (Loss 0.1234) | Score: 15
                status_strs.append(f"{game.upper()}: AGREE {agree_pct:.1f}% (Loss {avg_loss:.4f} | Score {score})")
                
                # Broadcast Telemetry
                if self.zmq_pub:
                    telemetry = {
                        "game": game,
                        "session_id": data.get("session_id", "unknown"),
                        "agree_pct": float(agree_pct),
                        "loss": float(avg_loss),
                        "score": int(score),
                        "steps": total,
                        "timestamp": time.time()
                    }
                    self.zmq_pub.send_string(f"STATS:{json.dumps(telemetry)}")
            
            if status_strs:
                print(f"[{time.strftime('%H:%M:%S')}] " + " | ".join(status_strs))
            
            self.stats = defaultdict(lambda: {"agreements": 0, "interventions": 0, "loss_sum": 0.0, "steps": 0, "session_id": "unknown", "score": 0})
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
    
    def push(self, state, task_id, action_idx, reward, agreed, game_type):
        # Detach and move to CPU to save GPU RAM
        state_cpu = state.detach().cpu()
        
        # Store tuple
        # We store: (State, TaskID, Action, Reward)
        task_id = 1.0 if game_type == "snake" else 0.0
        experience = (state_cpu, task_id, int(action_idx), float(reward))
        
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
                    count = min(len(buf), target_per_q)
                    batch.extend(random.sample(buf, count))
                    
        if len(batch) == 0: return None
        
        random.shuffle(batch) # Shuffle mixed batch
        
        # Collate
        states, task_ids, actions, rewards = zip(*batch)
        
        return (
            torch.cat(states, dim=0), 
            torch.tensor(task_ids, dtype=torch.float), 
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float)
        )
        
    def count(self):
        total = 0
        for game in self.buffers:
            for ag in self.buffers[game]:
                total += len(self.buffers[game][ag])
        return total

    def count(self):
        total = 0
        for game in self.buffers:
            for ag in self.buffers[game]:
                total += len(self.buffers[game][ag])
        return total

def sleep_cycle(model, optimizer, buffer, device, model_lock, target_game="all", epochs=SLEEP_EPOCHS):
    if buffer.count() < REPLAY_BATCH_SIZE: return
    
    print(f">> [SLEEP] Consolidating Memories ({target_game.upper()})...")
    
    # We run the training loop here. 
    # CRITICAL: We DO NOT hold the lock for the entire duration if we want to be "nice".
    # But for safety/simplicity in PyTorch (simultaneous backward/forward is bad), we SHOULD hold it.
    # To be "non-blocking-ish", we could sleep briefly between batches? 
    # No, let's just hold it. Frame drops in game are acceptable for "Sleep".
    # User said: "other game plays should be able to play". 
    # If we hold lock for 5 epochs * N batches, it might block for seconds.
    # We must lock PER STEP.
    
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()
    
    games_to_train = ["snake", "pong"] if target_game == "all" else [target_game]
    
    for _ in range(epochs):
        for game in games_to_train:
            # Sample (No Lock needed for buffer read if careful, but buffer isn't thread safe either really. 
            # We assume main thread only PUSHES, this thread only SAMPLES. Deque is thread-safe for append/pop, but sample?)
            # Random.sample on deque is not atomic.
            # Let's risk it or lock buffer? 
            # Ideally strict lock, but let's try fine-grained.
            
            # 1. Prepare Batch (CPU work, no lock needed mostly)
            batch = []
            try:
                buf_a = buffer.buffers[game][True]
                buf_d = buffer.buffers[game][False]
                if len(buf_a) > 0: batch.extend(random.sample(buf_a, min(len(buf_a), REPLAY_BATCH_SIZE//4)))
                if len(buf_d) > 0: batch.extend(random.sample(buf_d, min(len(buf_d), REPLAY_BATCH_SIZE//4)))
            except:
                continue # dict change size during iteration?

            if not batch: continue

            obs, tasks, actions, rewards = zip(*batch)
            inp = torch.cat(obs, dim=0).to(device)
            a_lbl = torch.tensor(actions, dtype=torch.long).to(device)
            r_val = torch.tensor(rewards, dtype=torch.float).to(device)
            
            task_id = 1 if game == "snake" else 0
            
            # 2. Train Step (Model Mutating -> LOCK REQUIRED)
            with model_lock:
                # Re-set training mode just in case main thread set it to Eval? 
                model.train() 
                optimizer.zero_grad()
                out = model(inp, task_id)
                
                if NO_TEACHER:
                     # RL Loss (Policy Gradient)
                     dist = torch.distributions.Categorical(logits=out)
                     log_probs = dist.log_prob(a_lbl)
                     loss = -(log_probs * r_val).mean()
                else:
                     loss = criterion(out, a_lbl)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            
            # Sleep tiny bit to let Main Thread breathe?
            time.sleep(0.01) 
            
    if steps > 0:
        print(f"   AVG SLEEP LOSS ({target_game.upper()}): {total_loss/steps:.4f}")

def run_sleep_thread(model, optimizer, buffer, device, model_lock, target_game):
    t = threading.Thread(target=sleep_cycle, args=(model, optimizer, buffer, device, model_lock, target_game))
    t.daemon = True
    t.start()

# --- CHARACTER TRAINING THREAD ---
def train_character_thread(model, optimizer_global, device, model_lock, epochs=1, mode="handwritten", text=None):
    if text:
        print(f">> [CHAR_TRAIN] Training on TYPED Text: '{text}'...")
    else:
        print(f">> [CHAR_TRAIN] Starting Background Training ({mode.upper()})...")
        
    try:
        # 1. FREEZE ALL EXCEPT CHARACTER HEAD
        with model_lock:
            for name, param in model.named_parameters():
                if "head_chars" not in name:
                    param.requires_grad = False
        
        # 2. Setup Local Optimizer (Only for the Character Head)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_local = torch.optim.Adam(trainable_params, lr=0.001)
        
        if text or mode == "typed":
            from character_dataset import get_text_dataloader
            target_text = text if text else "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            dl = get_text_dataloader(target_text, batch_size=32, num_repeats=200)
        else:
            # handwritten mode: digit (mnist) vs alphanumeric (emnist)
            # Use 4 workers for speed if possible
            dl = get_dataloader(batch_size=32, split="train", num_samples=2000, num_workers=4, mode="alphanumeric") 
            
        criterion = nn.CrossEntropyLoss()
        model.train()
        
        total_loss = 0.0
        steps = 0
        
        for epoch in range(epochs):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                task_id = 2 
                
                with model_lock:
                    optimizer_local.zero_grad()
                    out = model(x, task_id)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer_local.step()
                
                total_loss += loss.item()
                steps += 1
                time.sleep(0) # Minimal yield for "Fast-Pass"
                
        # 3. UNFREEZE for future game adaptive training
        with model_lock:
            for param in model.parameters():
                param.requires_grad = True
                
        tag = f"'{text}'" if text else mode.upper()
        print(f">> [CHAR_TRAIN] {tag} Complete. Avg Loss: {total_loss/max(1,steps):.4f}")
        
    except Exception as e:
        print(f"[ERROR] Char Training Failed: {e}")
        # Ensure unfreeze on error
        with model_lock:
            for param in model.parameters():
                param.requires_grad = True
        

def run_char_train(model, optimizer, device, model_lock, epochs=1, mode="handwritten", text=None):
    t = threading.Thread(target=train_character_thread, args=(model, optimizer, device, model_lock, epochs, mode, text))
    t.daemon = True
    t.start()


def calculate_entropy(probs_tensor):
    # probs: [batch, classes]
    # H(p) = - sum p * log(p)
    # Clamp to avoid log(0)
    p = torch.clamp(probs_tensor, 1e-6, 1.0)
    entropy = -torch.sum(p * torch.log(p), dim=1)
    return entropy

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
    global ENABLE_SNN, ENABLE_VSA, ENABLE_SLEEP, FREEZE_PONG, NO_TEACHER
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluation Mode: VSA OFF, Sleep OFF, Training OFF")
    parser.add_argument("--no-vsa", action="store_true", help="Disable VSA Rescue")
    parser.add_argument("--no-snn", action="store_true", help="Disable SNN (Baseline)")
    parser.add_argument("--freeze-pong", action="store_true", help="Freeze weights during Pong (Transfer Test)")
    parser.add_argument("--no-teacher", action="store_true", help="Disable Teacher Override (Sink or Swim Mode)")
    parser.add_argument("--check-syntax", action="store_true", help="Check syntax only")
    args = parser.parse_args()
    
    if args.check_syntax: return

    if args.eval:
        print(">> [EVAL] EVALUATION MODE ENABLED (VSA=OFF, SLEEP=OFF, TRAINING=OFF)")
        ENABLE_VSA = False
        ENABLE_SLEEP = False
        # Note: We technically keep SNN enabled to test it, but we disable *updates* below.
    
    if args.no_vsa: ENABLE_VSA = False
    if args.no_snn: ENABLE_SNN = False
    if args.freeze_pong: FREEZE_PONG = True
    if args.no_teacher: NO_TEACHER = True

    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL)
    pull_sock.setsockopt(zmq.LINGER, 0)
    pull_sock.bind("tcp://127.0.0.1:5565")
    
    pub_sock = context.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.LINGER, 0)
    pub_sock.bind("tcp://127.0.0.1:5566")
    
    # New: Telemetry Socket
    pub_sock_stats = context.socket(zmq.PUB)
    pub_sock_stats.setsockopt(zmq.LINGER, 0)
    pub_sock_stats.bind("tcp://127.0.0.1:5567")
    
    device = torch.device("cpu")
    
    # NEW: TaskAwareSNN (Late Fusion)
    model = TaskAwareSNN(beta=0.5).to(device)
    model_lock = threading.Lock() # Lock for model/optimizer access
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, weights_only=True)
            
            # --- WEIGHT SURGERY ---
            # If the saved model has 10 outputs but current has 62, copy the 10
            if "head_chars.weight" in state_dict:
                saved_size = state_dict["head_chars.weight"].shape[0]
                current_size = model.head_chars.weight.shape[0]
                
                if saved_size == 10 and current_size == 62:
                    print(f">> [SURGERY] Migrating weights: {saved_size} -> {current_size} classes")
                    # Create new weight/bias with current size
                    new_weight = model.head_chars.weight.clone()
                    new_bias = model.head_chars.bias.clone()
                    
                    # Copy old weights into the top
                    new_weight[:10] = state_dict["head_chars.weight"]
                    new_bias[:10] = state_dict["head_chars.bias"]
                    
                    # Update state_dict so strict loading works (or just set them manually)
                    state_dict["head_chars.weight"] = new_weight
                    state_dict["head_chars.bias"] = new_bias
            
            model.load_state_dict(state_dict, strict=False)
            print("loaded weights (with Surgery if needed).")
        except Exception as e:
            print(f"Starting fresh (New Arch: {e})")
    else:
         print("Starting fresh.")
    
    logger = LogAggregator(zmq_pub=pub_sock_stats) # Pass socket
    concept_mapper = ConceptMapper() # Initialize Mapper
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
                print(f"[ADMIN] COMMAND RECEIVED: {cmd}")
                if cmd == "admin_sleep_snake":
                    run_sleep_thread(model, optimizer, buffer, device, model_lock, "snake")
                elif cmd == "admin_sleep_pong":
                     run_sleep_thread(model, optimizer, buffer, device, model_lock, "pong")
                elif cmd == "force_sleep":
                    run_sleep_thread(model, optimizer, buffer, device, model_lock, "all")
                elif cmd == "reset_memory":
                     buffer = ReplayBuffer()
                elif cmd == "toggle_teacher":
                     NO_TEACHER = not NO_TEACHER
                     print(f">> TEACHER STATUS: {'OFF' if NO_TEACHER else 'ON'}")
                elif cmd == "strict_transfer_cfg":
                     NO_TEACHER = True
                     FREEZE_PONG = True
                     print(">> STRICT TRANSFER CONFIG APPLIED (Teacher=OFF, Pong=Frozen)")           
                elif cmd == "train_char":
                    epochs = msg.get("epochs", 1)
                    mode = msg.get("mode", "handwritten")
                    text = msg.get("text", None)
                    run_char_train(model, optimizer, device, model_lock, epochs=epochs, mode=mode, text=text)
                elif cmd == "set_device":
                    new_device_str = msg.get("device", "cpu")
                    if new_device_str == "gpu" and torch.cuda.is_available():
                        new_device = torch.device("cuda")
                    else:
                        new_device = torch.device("cpu")
                    
                    if new_device != device:
                        print(f">> [DEVICE] Switching from {device} to {new_device}...")
                        with model_lock:
                            device = new_device
                            model.to(device)
                            # Re-init optimizer if moving between CPU/GPU to ensure state is on correct device
                            # Actually, Adam state can be moved, but it's often safer to re-init or use a helper
                            # For SNN-QAT, re-init with same params is fine since we are mostly doing Live training
                            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                            print(f">> [DEVICE] Model migrated to {device}")
                continue # Skip game logic

            # --- PREDICTION REQUEST (CHARACTERS) ---
            if msg.get("type") == "predict_char":
                # Handle Character Prediction
                try:
                    # 1. Decode Image (Expects base64 of 28x28 or similar)
                    img_bytes = base64.b64decode(msg["image"])
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                    
                    # 2. Resize to 10x10
                    img_10 = cv2.resize(img, (10, 10), interpolation=cv2.INTER_AREA)
                    
                    # 3. Preprocess
                    img_float = img_10.astype(np.float32) / 255.0
                    frames = np.stack([img_float]*4, axis=0) # [4, 10, 10]
                    inp = torch.tensor(frames).unsqueeze(0).float().to(device) # [1, 4, 10, 10]
                    
                    # 4. Inference (Task 2)
                    with model_lock:
                        model.eval() # Temp Eval
                        out = model(inp, 2)
                        model.train() # Resume Train mode default?
                        
                    probs = torch.softmax(out, dim=1).detach().cpu().numpy()[0]
                    pred_class = int(np.argmax(probs))
                    # Fix warning: ensure probs is treated efficiently
                    entropy = calculate_entropy(torch.from_numpy(probs).unsqueeze(0)).item()
                    
                    explanation = concept_mapper.get_explanation(pred_class)
                    
                    # 5. Send Result (VIS Channel)
                    res_payload = {
                         "task": "char_recognition", 
                         "grid": img_float.tolist(),
                         "probs": probs.tolist(), 
                         "prediction": pred_class,
                         "explanation": explanation,
                         "entropy": entropy,
                         "teacher": -1, "student": pred_class, "agreed": True
                    }
                    pub_sock_stats.send_string(f"VIS:{json.dumps(res_payload)}")
                    
                    print(f"[CHAR] Predicted: {pred_class} -> {explanation} (Conf: {probs[pred_class]:.2f})")
                    
                except Exception as e:
                    print(f"Predict Error: {e}")
                
                continue
            
            game_type = msg.get("game", "unknown")
            session_id = msg.get("session_id", "unknown")
            
            # --- SLEEP TRIGGER: TASK SWITCH ---
            # DISABLED AUTO-SLEEP AS PER USER REQUEST ("after sessions i will start sleep")
            # We only sleep when manually requested via dashboard.
            # if ENABLE_SLEEP and last_game_type is not None and game_type != last_game_type:
            #     pass 
            
            last_game_type = game_type
            
            state_data = msg.get("state", {})
            reward = msg.get("reward", 0.0)
            done = msg.get("done", False)
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
            
            # Use the 'device' variable which can now be updated
            input_tensor = torch.from_numpy(stacked_np).float().to(device) 
            input_tensor = input_tensor.unsqueeze(0) # [1, 4, 10, 10]
            
            # Forward (WITH LOCK)
            with model_lock:
                 optimizer.zero_grad()
                 rate_out = model(input_tensor, task_id) 
            
            # --- VSA PRIOR MASKING (GATED TRANSFER) ---
            snn_probs = torch.softmax(rate_out, dim=1)
            entropy = calculate_entropy(snn_probs).item()
            
            # Default to SNN
            probs = snn_probs
            prior_active = False 
            
            # GATED RESCUE
            if ENABLE_VSA and entropy > CONFIDENCE_THRESHOLD:
                # System 1 is Confused -> System 2 (VSA) Intervention
                vsa_prior = ActionSemantics.get_goal_alignment(game_type, state_data)
                
                if len(vsa_prior) > 0:
                    prior_tensor = torch.from_numpy(vsa_prior).to(device).unsqueeze(0) # [1, Actions]
                    
                    # Apply Bias
                    biased_probs = snn_probs * (1.0 + VSA_STRENGTH * prior_tensor)
                    probs = biased_probs / biased_probs.sum(dim=1, keepdim=True)
                    prior_active = True
                    # Log internally if needed, or via 'veto' field (abusing it slightly for visualization or adding new field)
                    # We'll use a new field "vsa_rescue" in telemetry.

            if not ENABLE_SNN:
                # Ablation: Pure Random or Pure VSA (if enabled)
                # If SNN disabled, we just use Uniform * Prior ?
                # Or just Uniform.
                # Let's say: If DISABLE_SNN, rate_out is ignored/random.
                # But actually, best way to disable SNN learning is stop optim step.
                # To disable SNN *inference*, we replace snn_probs with Uniform.
                probs = torch.ones_like(probs) / probs.shape[1]
                if ENABLE_VSA:
                     # Re-apply VSA logic on Uniform
                     vsa_prior = ActionSemantics.get_goal_alignment(game_type, state_data)
                     if len(vsa_prior) > 0:
                        prior_tensor = torch.from_numpy(vsa_prior).to(device).unsqueeze(0)
                        biased_probs = probs * (1.0 + VSA_STRENGTH * prior_tensor)
                        probs = biased_probs / biased_probs.sum(dim=1, keepdim=True)
            
            # 1. Determine SNN's Raw Intent (for Training)
            _, snn_raw_idx_t = torch.max(snn_probs, dim=1)
            snn_raw_idx = snn_raw_idx_t.item()
            
            # 2. Determine Final Action (for Execution/Safety)
            # 'student_idx' will track the *executed* action (after VSA/Veto)
            # Note: 'student_idx' variable name is kept for compatibility with existing logic below
            _, active_idx_t = torch.max(probs, dim=1)
            student_idx = active_idx_t.item()
            
            # --- SYSTEM 2: SIMULATION & SAFETY VETO ---
            # "Reasoning": Check if proposed action is suicidal/bad, if so, override.
            reasoning_override = False
            veto_log = ""
            
            if game_type == "snake":
                snake_act_names = ["UP", "DOWN", "LEFT", "RIGHT"]
                if student_idx < 4:
                    proposed = snake_act_names[student_idx]
                    _, dead = sim_snake(state_data, proposed)
                    
                    if dead:
                        # CRITICAL: Proposed action kills. Search for safe alternative.
                        original_idx = student_idx
                        for i, alt in enumerate(snake_act_names):
                            _, alt_dead = sim_snake(state_data, alt)
                            if not alt_dead:
                                student_idx = i # OVERRIDE
                                reasoning_override = True
                                veto_log = f"VETO: Snake {proposed}->{alt} (Reason: Predicted Self-Collision)"
                                break
            
            elif game_type == "pong":
                # Pong Actions: 0=UP, 1=DOWN
                pong_act_names = ["UP", "DOWN"]
                if student_idx < 2:
                    proposed = pong_act_names[student_idx]
                    # We need ball_x for simulation to be accurate.
                    if "ball_x" in state_data:
                        _, miss = sim_pong(state_data, proposed)
                        
                        if miss:
                            # CRITICAL: Proposed moves leads to miss. Check if other move saves.
                            original_idx = student_idx
                            other_idx = 1 - student_idx
                            other_act = pong_act_names[other_idx]
                            _, other_miss = sim_pong(state_data, other_act)
                            
                            if not other_miss:
                                student_idx = other_idx # OVERRIDE
                                reasoning_override = True
                                veto_log = f"VETO: Pong {proposed}->{other_act} (Reason: Predicted Miss)"
            
            if reasoning_override:
                print(f"[REASONING] [SYSTEM 2] Counterfactual Analysis: {veto_log}")
            
            # 3. Training & Agreement Logic
            # CRITICAL FIX: Train if SNN was wrong, even if System was right.
            
            # Did the SNN get it right natively?
            snn_agreed = (snn_raw_idx == teacher_idx)
            
            # Did the Final System get it right?
            system_agreed = (student_idx == teacher_idx)
            
            # DEFINE FINAL ACTION (Moved up for RL Training)
            if NO_TEACHER:
                final_action_idx = student_idx
            else:
                final_action_idx = student_idx if system_agreed else teacher_idx
            
            loss_val = 0.0
            
            # Gated Training Rule:
            # - If SNN is correct: No loss.
            # - If SNN is wrong: Train (unless Eval mode).
            # - If NO_TEACHER: We can't train, because we don't have a teacher target!
            
            if not args.eval: # Only train if NOT in eval mode
                if NO_TEACHER:
                     # In No-Teacher mode, we can only do Reinforcement Learning (not implemented yet)
                     # or unsupervized Hebbian. For now, Training is OFF in No-Teacher mode.
                     loss_val = 0.0
                elif snn_agreed:
                    pass # SNN effectively mastered this state
                else:
                    # SNN was wrong (or confused). 
                    # If Teacher ON: Train on Teacher.
                    # If Teacher OFF: Train on Reward (RL) or Skip?
                    
                    if NO_TEACHER:
                        # RL UPDATE (Live)
                        # We executed 'final_action_idx' (Student).
                        # We received 'reward'.
                        # Loss = -log_prob(action) * reward
                        dist = torch.distributions.Categorical(logits=rate_out)
                        log_prob = dist.log_prob(torch.tensor([final_action_idx], device=device))
                        loss = -(log_prob * reward)
                    else:
                        # IMITATION UPDATE (Live)
                        target = torch.tensor([teacher_idx], dtype=torch.long, device=device)
                        loss = criterion(rate_out, target)
                    
                    if not (game_type == "pong" and FREEZE_PONG):
                        with model_lock:
                            if loss.requires_grad:
                                loss.backward()
                                optimizer.step()
                        
                    loss_val = loss.item()
                
            # MEMORY: Store Experience
            # We store what we *did* (final_action_idx) and the result (reward)
            buffer.push(input_tensor, task_id, final_action_idx, reward, system_agreed, game_type)
            
            # Retrieve score from message payload
            current_score = msg.get("score", 0)
            logger.update(game_type, system_agreed, loss_val, current_score, session_id)
            logger.check_print()
            
            # --- VISUALIZATION TELEMETRY (Every 10 Steps) ---
            if steps_total % 10 == 0:
                # 1. Prepare Telemetry Payload
                grid_list = curr_np.tolist()
                probs_list = probs[0].tolist()
                vis_payload = {
                    "grid": grid_list,
                    "probs": probs_list,
                    "reward": float(reward),
                    "game": game_type,
                    "teacher": teacher_idx,
                    "student": student_idx,
                    "agreed": bool(system_agreed),
                    "task": game_type,
                    "veto": reasoning_override,
                    "veto_log": veto_log,
                    "vsa_prior": vsa_prior.tolist() if 'vsa_prior' in locals() and len(vsa_prior) > 0 else [],
                    "vsa_rescue": prior_active,
                    "entropy": entropy,
                    "session_id": session_id,
                    "teacher_active": not NO_TEACHER,
                    "score": msg.get("score", 0)
                }
                pub_sock_stats.send_string(f"VIS:{json.dumps(vis_payload)}")

                # 2. Prepare Console Output
                ts = time.strftime('%H:%M:%S')
                t_act = vis_payload['teacher']
                s_act = vis_payload['student']
                
                if game_type == "char_recognition":
                     label_map = [str(i) for i in range(10)]
                elif game_type == "snake":
                     label_map = ["UP", "DN", "LF", "RT"]
                else: # Pong
                     label_map = ["UP", "DN"]
                
                t_str = label_map[t_act] if 0 <= t_act < len(label_map) else f"UNK({t_act})"
                s_str = label_map[s_act] if 0 <= s_act < len(label_map) else f"UNK({s_act})"
                
                # Logic for "PREDICT" status
                status = "PREDICT" if game_type == "char_recognition" else ("AGREE" if vis_payload['agreed'] else "INTERVENE")
                probs_str = [f"{p:.2f}" for p in vis_payload['probs']]
                
                if game_type == "char_recognition":
                    explanation = concept_mapper.get_explanation(s_act)
                    print(f"[{ts}] {game_type.upper()} | {status} | Brain: {s_str} ({explanation}) {probs_str} (H={vis_payload['entropy']:.2f})")
                else:
                    print(f"[{ts}] {game_type.upper()} | {status} | Brain: {s_str} {probs_str} (H={vis_payload['entropy']:.2f})")

            actions = ["UP", "DOWN", "LEFT", "RIGHT"]
            cmd = "UP"
            # Use action_idx which is the TEACHER's advice if SNN failed, or SNN's active choice if agreed.
            # Actually, we should execute 'student_idx' (the System's final choice).
            # Wait, if !system_agreed, we might want to broadcast the TEACHER's correction?
            # Standard imitation learning: You drive, but if you fail, I reset you?
            # Here: We broadcast the executed action.
            # If !system_agreed, it usually means SNN+VSA both failed compared to Oracle.
            # If we want to behave nicely, we should arguably output the Teacher Action if we are training?
            # BUT: We want to see the failure.
            # Let's execute the System's Decision (student_idx), even if wrong.
            # UNLESS: It causes instant death? No, sim_snake handles veto.
            
            # Correction: Previously we used:
            # if agreed: action_idx = student_idx; else: action_idx = teacher_idx
            # This means we forced the Teacher action on failure. This is "Student Forcing".
            # Let's keep Student Forcing for stability.
            # UNLESS: --no-teacher is set. Then we MUST execute student_idx.
            
            if NO_TEACHER:
                final_action_idx = student_idx
            else:
                final_action_idx = student_idx if system_agreed else teacher_idx
            
            if game_type == "snake":
                cmd = actions[final_action_idx] if final_action_idx < 4 else "UP"
            else: # Pong
                cmd = "UP" if final_action_idx == 0 else "DOWN"
            
            pub_sock.send_string(f"{game_type.upper()}:{cmd}")
            
            steps_total += 1
            
            if steps_total % SAVE_INTERVAL == 0: 
                 torch.save(model.state_dict(), MODEL_PATH)
            
            # --- SLEEP TRIGGER: INTERVAL ---
            # DISABLED AUTO-SLEEP (Manual Only)
            # if ENABLE_SLEEP and steps_total % SLEEP_INTERVAL == 0:
            #      sleep_cycle(model, optimizer, buffer, device, model_lock, "all")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()




