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
import torch.nn.functional as F
import torch.serialization

# [CRITICAL] Fix for LazyLinear loading in Torch 2.4+
# This allows 'weights_only=True' (default) to load UninitializedParameter
try:
    from torch.nn.parameter import UninitializedParameter
    torch.serialization.add_safe_globals([UninitializedParameter])
except (ImportError, AttributeError):
    pass
 # Import F
from collections import defaultdict, deque
from symbol_grounding import ActionSemantics
from simulation import sim_snake, sim_pong
from maze_game import sim_maze  # For maze veto logic
from character_dataset import get_dataloader # New Import
from concept_mapper import ConceptMapper
import threading
import queue
import hypervec_shim as hypervec_rs
from curiosity import CuriosityModule
from symbol_grounding import GLOBAL_PRIMITIVES_MAP
from intelligent_buffer import IntelligentReplayBuffer, Experience # [NEW] Import Buffer
from saliency import SaliencyVisualizer # [NEW] Import Saliency
from intrinsic_motivation import CombinedIntrinsicMotivation  # [AGI] Phase 1: Intrinsic Motivation
from teacher_interface import TeacherInterface, HeuristicTeacher, NullTeacher # [AGI] Phase 1: Modular Teacher
from learning_progress import LearningProgressTracker # [AGI] Phase 1.4: Self-Curriculum
from logger_service import get_logger # [AGI] Phase 1.5: Central Logging

# [AGI] Phase 1.3: Telemetry Logging
import csv
import os
LOG_FILE = "training_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "step", "task", "extrinsic", "intrinsic_icm", "intrinsic_count", "total_reward"])



import sys

# --- ABLATION FLAGS (DEFAULTS) ---
ENABLE_SNN = True
ENABLE_VSA = True
ENABLE_SLEEP = True
FREEZE_PONG = False
FREEZE_ALL = False  # If True, NO weight updates for ANY game/task
# Check command line args
NO_TEACHER = True if "--no-teacher" in sys.argv else False 
# If True, we never fallback to Teacher. System must survive on its own.
AUTO_DREAM = False # If True, brain automatically sleeps to consolidate memories
RL_CONTEXT = {} # Stores {session_id: log_prob_of_prev_action} for REINFORCE (Policy Gradient)

# --- HYPERPARAMETERS ---
SLEEP_EPOCHS = 5
REPLAY_BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Reduced from 1e-3 to prevent overfitting
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
from snn_qat import TaskAwareSNN, ternarize_weight
from simulation import sim_snake, sim_pong # Import Simulation Logic
from symbol_grounding import ActionSemantics # Import Grounding Logic

class LogAggregator:
    def __init__(self, interval=5.0, zmq_pub=None):
        self.interval = interval
        self.last_print = time.time()
        self.stats = defaultdict(lambda: {
            "agreements": 0, 
            "interventions": 0, 
            "vetoes": 0,
            "loss_sum": 0.0, 
            "conf_sum": 0.0,
            "steps": 0, 
            "session_id": "unknown", 
            "score": 0
        })
        self.zmq_pub = zmq_pub

    def update(self, game, agreed, loss, score, session_id="unknown", veto=False, confidence=0.5):
        self.stats[game]["session_id"] = session_id
        self.stats[game]["steps"] += 1
        self.stats[game]["loss_sum"] += loss
        self.stats[game]["conf_sum"] += confidence
        self.stats[game]["score"] = max(self.stats[game]["score"], score) # Keep max score seen in interval

        if agreed:
            self.stats[game]["agreements"] += 1
        else:
            self.stats[game]["interventions"] += 1
            
        if veto:
            self.stats[game]["vetoes"] += 1

    def check_print(self):
        if time.time() - self.last_print > self.interval:
            status_strs = []
            for game, data in self.stats.items():
                total = data["steps"]
                if total == 0: continue
                agree_pct = (data["agreements"] / total) * 100
                avg_loss = data["loss_sum"] / total
                avg_conf = data["conf_sum"] / total
                score = data["score"]
                veto_pct = (data["vetoes"] / total) * 100
                
                # Format: SNAKE: AGREE 95.0% (Loss 0.1234) | Score: 15
                status_strs.append(f"{game.upper()}: AGREE {agree_pct:.1f}% (Veto {veto_pct:.1f}%) | Score {score}")
                
                # Broadcast Telemetry
                if self.zmq_pub:
                    telemetry = {
                        "game": game,
                        "session_id": data.get("session_id", "unknown"),
                        "agree_pct": float(agree_pct),
                        "veto_pct": float(veto_pct),
                        "avg_conf": float(avg_conf),
                        "loss": float(avg_loss),
                        "score": int(score),
                        "timestamp": time.time()
                    }
                    self.zmq_pub.send_string(f"TELEMETRY:{json.dumps(telemetry)}")
            
            if status_strs:
                print(f"[{time.strftime('%H:%M:%S')}] " + " | ".join(status_strs))
            
            self.stats = defaultdict(lambda: {
                "agreements": 0, 
                "interventions": 0, 
                "vetoes": 0,
                "loss_sum": 0.0, 
                "conf_sum": 0.0,
                "steps": 0, 
                "session_id": "unknown", 
                "score": 0
            })
            self.last_print = time.time()

# --- INTELLIGENT ARCHIVAL (GLOBAL) ---
# Global Replay Buffer (Shared across tasks for now, but stores task_name)
# Global Replay Buffer (Shared across tasks for now, but stores task_name)
# RAM Limit: 10,000 transitions (~2MB for 10x10x1 states)
REPLAY_BUFFER = IntelligentReplayBuffer(ram_capacity=10000, archival_threshold=0.5)

# --- [AGI] INTRINSIC MOTIVATION (Phase 1) ---
# Provides internal rewards for curiosity and exploration without teacher
INTRINSIC_MOTIVATION = None  # Initialized in main() after device selection
CURIOSITY_TRACKER = LearningProgressTracker(window_size=50) # [AGI] Phase 1.4

# [AGI] Phase 2: Cognitive Engine
from cognitive_engine import create_cognitive_engine
COGNITIVE_ENGINE = create_cognitive_engine()

# [AGI] Phase 3: Dashboard Broadcasting
# Moved to main() to access ZMQ and Logger


# --- GLOBAL LOCK ---
model_lock = threading.RLock() # Protects Model & Optimizer

# --- SALIENCY ---
SALIENCY = None # Initialized after model creation


# --- MEMORY (REPLAY BUFFER) ---
class ReplayBuffer:
    def __init__(self, capacity_per_quadrant=2500):
        self.capacity = capacity_per_quadrant
        # Stratified Buffers: [Game][Agreed?]
        self.buffers = {
            "snake": {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)},
            "pong":  {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)},
            "maze":  {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)}
        }
    
    def push(self, state, task_id, action_idx, reward, agreed, game_type, compass=None):
        # Detach and move to CPU to save GPU RAM
        state_cpu = state.detach().cpu()
        
        # Store tuple
        # We store: (State, TaskID, Action, Reward, Compass)
        if game_type == "snake" or game_type == "maze":
            tid = 1.0
        else:
            tid = 0.0
            
        if compass is None:
            compass = torch.zeros(8)  # 8-bit compass (4 Goal + 4 Blocked)
        else:
            compass = compass.detach().cpu()
            
        experience = (state_cpu, tid, int(action_idx), float(reward), compass)
        
        # Route to correct quadrant (fallback to snake if unknown)
        if game_type not in self.buffers:
            game_type = "snake"
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
        states, task_ids, actions, rewards, compasses = zip(*batch)
        
        return (
            torch.cat(states, dim=0), 
            torch.tensor(task_ids, dtype=torch.float), 
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(compasses, dim=0)
        )
    
    def sample_game(self, batch_size, game_type):
        """Sample from a specific game only (for sleep cycle training)."""
        if game_type not in self.buffers:
            return None
        
        batch = []
        target_per_q = batch_size // 2
        if target_per_q == 0: target_per_q = 1
        
        for agreed in [True, False]:
            buf = self.buffers[game_type][agreed]
            if len(buf) > 0:
                count = min(len(buf), target_per_q)
                batch.extend(random.sample(buf, count))
        
        if len(batch) == 0: return None
        
        random.shuffle(batch)
        
        # Collate
        states, task_ids, actions, rewards, compasses = zip(*batch)
        
        return (
            torch.cat(states, dim=0), 
            torch.tensor(task_ids, dtype=torch.float), 
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(compasses, dim=0)
        )
        
    def count(self):
        total = 0
        for game in self.buffers:
            for agreed in self.buffers[game]:
                total += len(self.buffers[game][agreed])
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
    
    games_to_train = ["snake", "pong", "maze"] if target_game == "all" else [target_game]
    
    for _ in range(epochs):
        for game in games_to_train:
            # 1. Prepare Batch (GAME-SPECIFIC to avoid action space mismatch)
            batch_data = buffer.sample_game(REPLAY_BATCH_SIZE, game)
            if batch_data is None: continue
            
            x, tid_batch, y, r, comp = batch_data
            # Fix compass dimension: buffer returns [batch, 1, 4], model expects [batch, 4]
            if comp.dim() == 3:
                comp = comp.squeeze(1)
            x, y, r, comp = x.to(device), y.to(device), r.to(device), comp.to(device)
            tid = tid_batch[0].item() # Use first task_id in batch (stratified sampling should ensure consistency here or we iterate)

            # 2. Train Step (Model Mutating -> LOCK REQUIRED)
            with model_lock:
                model.train() 
                optimizer.zero_grad()
                
                # Forward (Universal Signature)
                # x is [B, 4, 10, 10], tid is scalar. Map tid to task_name.
                t_name = "snake"
                if game == "pong": t_name = "pong"
                if game == "maze": t_name = "maze"
                
                logits, value = model(x, task_name=t_name)
                
                if NO_TEACHER:
                     # A2C Loss during Sleep (Dreaming) -> "Representation Learning"
                     # We can't easily do A2C offline without next_state V(s').
                     # User Plan: "Dreaming trains Critic and Encoder".
                     # We will train Critic to match Reward (assuming terminal/sparse).
                     # Simple: value ~ reward
                     critic_loss = F.mse_loss(value.squeeze(), r)
                     
                     # Optional: Entailment/Contrastive?
                     # For now, just train Critic.
                     loss = critic_loss 
                else:
                     # Supervised Loss (Actor mimics recorded actions)
                     loss_actor = criterion(logits, y)
                     # Critic Loss (Value predicts Reward)
                     loss_critic = F.mse_loss(value.squeeze(), r)
                     loss = loss_actor + 0.5 * loss_critic
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            
            # Sleep tiny bit to let Main Thread breathe
            time.sleep(0.01) 
            
    if steps > 0:
        avg_loss = total_loss/steps
        print(f"   AVG SLEEP LOSS ({target_game.upper()}): {avg_loss:.4f}")
        
        # CRITICAL: Save model after sleep to persist learned knowledge
        with model_lock:
            torch.save(model.state_dict(), MODEL_PATH)
        print(f"   [DISK] MODEL SAVED: {MODEL_PATH}")

def generative_dream_cycle(model, optimizer, device, model_lock, task_tag="snake", num_samples=32):
    """
    [AGI] Phase 5.3: Generative Dreaming
    Uses CognitiveEngine's imagination to generate synthetic training targets for the SNN.
    """
    print(f">> [DREAM] Starting Generative Dreaming ({task_tag.upper()})...")
    
    # 1. Generate Dreams from Cognitive Engine
    with model_lock:
        try:
            dreams = COGNITIVE_ENGINE.dream(num_samples=num_samples, task_tag=task_tag)
        except Exception as e:
            print(f"[DREAM] Error generating dreams: {e}")
            return
            
    if not dreams:
        print("[DREAM] No dreams generated.")
        return
        
    print(f"   [DREAM] Generated {len(dreams)} synthetic experiences.")
    
    # 2. Train SNN on Dreams
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0
    
    for dream in dreams:
        if dream.get("image") is None: continue
        
        # 1. Prepare Input Image [1, 4, 10, 10]
        # Since we only have one frame, we stack it for the SNN
        img = dream["image"]
        frames = np.stack([img]*4, axis=0)
        x = torch.from_numpy(frames).float().to(device).unsqueeze(0)
        
        # 2. Prepare Target Action
        act_names = ["UP", "DOWN", "LEFT", "RIGHT"] if task_tag != "pong" else ["UP", "DOWN"]
        if dream["action"] not in act_names: continue
        y = torch.tensor([act_names.index(dream["action"])], device=device)
        
        # 3. Train Step
        with model_lock:
            model.train()
            optimizer.zero_grad()
            logits, val = model(x, task_name=task_tag)
            
            # Hybrid Loss: Imitation + Value refinement
            loss_actor = criterion(logits, y)
            loss_critic = F.mse_loss(val.squeeze(), torch.tensor(dream["reward"], device=device))
            loss = loss_actor + 0.5 * loss_critic
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    # [AGI] Note: To truly train the SNN, we need the 10x10 image.
    if steps > 0:
        print(f"   [DREAM] Phase 5.3 Distillation Complete. Avg Loss: {total_loss/steps:.4f}")
    else:
        print("   [DREAM] No valid imagery available for distillation.")

def hypothetical_practice_cycle(model, optimizer, device, model_lock, task_tag="snake", num_anchors=10):
    """
    [AGI] Phase 5.4: Hypothetical Scenario Generation
    Discovers potential death-traps or rewards through imagination and 'practices' them.
    """
    print(f">> [HYPO] Starting Hypothetical Practice ({task_tag.upper()})...")
    
    with model_lock:
        try:
            lessons = COGNITIVE_ENGINE.generate_hypothetical_lessons(num_anchors=num_anchors, task_tag=task_tag)
        except Exception as e:
            print(f"[HYPO] Error generating lessons: {e}")
            return
            
    if not lessons:
        print("[HYPO] No significant hypothetical scenarios discovered.")
        return
        
    print(f"   [HYPO] Discovered {len(lessons)} high-impact scenarios.")
    
    # Train SNN on these lessons
    # Since hypothetical paths can be multi-step, for now we treat the LAST step 
    # (the one with the reward) as the primary lesson anchor.
    # We use the anchor image if available.
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0
    
    for lesson in lessons:
        if lesson.get("anchor_image") is None: continue
        
        # 1. Prepare Input Image
        img = lesson["anchor_image"]
        frames = np.stack([img]*4, axis=0)
        x = torch.from_numpy(frames).float().to(device).unsqueeze(0)
        
        # 2. Prepare Target
        act_names = ["UP", "DOWN", "LEFT", "RIGHT"] if task_tag != "pong" else ["UP", "DOWN"]
        if lesson["action"] not in act_names: continue
        y = torch.tensor([act_names.index(lesson["action"])], device=device)
        
        # 3. Train Step
        with model_lock:
            model.train()
            optimizer.zero_grad()
            logits, val = model(x, task_name=task_tag)
            
            # If it's a PENALTY (e.g. death), we want to minimize probability?
            # Actually, standard CrossEntropy expects 'target'. 
            # If reward is POSITIVE, we teach the action.
            # If reward is NEGATIVE, we should ideally teach the DIFFERENT action.
            # For now, simple Policy Refinement: only teach POSITIVE rewards, 
            # OR use a penalty loss for negative ones.
            
            if lesson["reward"] > 0:
                loss_actor = criterion(logits, y)
                loss_critic = F.mse_loss(val.squeeze(), torch.tensor(lesson["reward"], device=device))
                loss = loss_actor + 0.5 * loss_critic
            else:
                # Negative reward: Penalty to avoid this action
                # We can use a custom loss that maximizes entropy or pushes AWAY from y.
                log_probs = F.log_softmax(logits, dim=1)
                loss_actor = log_probs[0, y.item()] # We want to MINIMIZE this, so maximize -loss
                # loss_actor is negative, so adding it increases total loss? 
                # actually, log_probs are negative. minimize probability -> maximize -log_prob.
                # So loss = -log_probs[0, y.item()] would maximize it.
                # We want: loss = log_probs[0, y.item()] (which is e.g. -5.0).
                # Minimize -5.0 -> make it e.g. -10.0.
                loss_actor = log_probs[0, y.item()] 
                loss_critic = F.mse_loss(val.squeeze(), torch.tensor(lesson["reward"], device=device))
                loss = loss_actor + 0.5 * loss_critic
                
            loss.backward()
            optimizer.step()
            total_loss += abs(loss.item())
            steps += 1
            
    if steps > 0:
        print(f"   [HYPO] Phase 5.4 Practice Complete. Avg Activity: {total_loss/steps:.4f}")


def run_sleep_thread(model, optimizer, buffer, device, model_lock, target_game):
    # BLOCKING SLEEP (Synchronous)
    # The user requested: "while dreaming stop the game then restart when ready".
    # Running this in the Main Thread pauses the game loop implicitly.
    print(f">> [SLEEP] Pausing Game Loop for Dreaming Cycle ({target_game})...")
    
    # Standard Replay
    sleep_cycle(model, optimizer, buffer, device, model_lock, target_game)
    
    # [AGI] Phase 5.3: Generative Dreaming (Imagination)
    if target_game == "all":
        generative_dream_cycle(model, optimizer, device, model_lock, "snake")
        # [AGI] Phase 5.4: Hypothetical Scenarios
        hypothetical_practice_cycle(model, optimizer, device, model_lock, "snake")
    else:
        generative_dream_cycle(model, optimizer, device, model_lock, target_game)
        # [AGI] Phase 5.4: Hypothetical Scenarios
        hypothetical_practice_cycle(model, optimizer, device, model_lock, target_game)
        
    print(f">> [SLEEP] Waking up. Resuming Game Loop.")

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
                if "heads.char_recognition" not in name:
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
                    # Forward returns (logits, value)
                    out, _ = model(x, task_name="char_recognition")
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
        # STRICT BOUNDARY CHECK (No Modulo)
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE: continue
        
        if (nx, ny) in obstacles: continue 
        dist = abs(nx - fx) + abs(ny - fy)
        if dist < min_dist:
            min_dist = dist
            best_move = i
            
    if min_dist == 999: # Trapped
        for i, (dx, dy) in enumerate(full_deltas):
            nx, ny = hx + dx, hy + dy
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE: continue
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
    global ENABLE_SNN, ENABLE_VSA, ENABLE_SLEEP, FREEZE_PONG, FREEZE_ALL, NO_TEACHER, AUTO_DREAM
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluation Mode: VSA OFF, Sleep OFF, Training OFF")
    parser.add_argument("--no-vsa", action="store_true", help="Disable VSA Rescue")
    parser.add_argument("--no-snn", action="store_true", help="Disable SNN (Baseline)")
    parser.add_argument("--freeze-pong", action="store_true", help="Freeze weights during Pong (Transfer Test)")
    parser.add_argument("--no-teacher", action="store_true", help="Disable Teacher Override (Sink or Swim Mode)")
    parser.add_argument("--check-syntax", action="store_true", help="Check syntax only")
    args = parser.parse_args()

    # --- INTELLIGENT ARCHIVAL INITIALIZED GLOBALLY ---
    # REPLAY_BUFFER = IntelligentReplayBuffer(ram_capacity=10000, archival_threshold=0.5)
    
    # --- ZMQ SETUP ---
    # 5565: INPUT (PULL)
    # 5566: VISUALS (PUB)
    # 5567: DATA/LOGS (PUB)
    context = zmq.Context()
    
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
    
    # New: Telemetry & Logs Socket
    pub_sock_stats = context.socket(zmq.PUB)
    pub_sock_stats.setsockopt(zmq.LINGER, 0)
    pub_sock_stats.bind("tcp://127.0.0.1:5567")
    
    # Initialize Logger
    logger = get_logger()
    logger.log("SERVER", "NSCK Brain Online. Listening on 5565...", level="INFO")

    # [AGI] Phase 3: Dashboard Broadcasting Setup
    def broadcast_brain_state(msg_dict):
        try:
            pub_sock_stats.send_string(f"BRAIN_STATE:{json.dumps(msg_dict)}")
        except Exception as e:
            print(f"Broadcast Error: {e}")
            
    COGNITIVE_ENGINE.register_broadcaster(broadcast_brain_state)
    logger.log("SERVER", "Brain Broadcaster Registered", level="INFO")
    
    # Startup Broadcast (Must be after socket bind)
    try:
        startup_msg = {'source': 'SERVER', 'message': 'System Online', 'level': 'INFO', 'timestamp': time.time()}
        pub_sock_stats.send_string(f"LOG:{json.dumps(startup_msg)}")
    except Exception: pass
    
    
    device = torch.device("cpu")
    
    # NEW: TaskAwareSNN (Universal Actor-Critic)
    model = TaskAwareSNN(beta=0.5).to(device)
    
    # REGISTER TASKS (Grow Brain)
    model.register_task("maze", 4)
    model.register_task("char_recognition", 62)
    
    # model_lock is now GLOBAL
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added weight decay for regularization
    criterion = nn.CrossEntropyLoss()
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, weights_only=True)
            
            # --- WEIGHT SURGERY ---
            # NSCK v2 uses self.heads (ModuleDict). 
            # We migrate old weights to the new structure if found.
            
            # 1. Character Head Migration
            head_chars_key = "heads.char_recognition.actor.weight" # New Structure
            legacy_head_key = "head_chars.weight" # Old Structure
            
            if legacy_head_key in state_dict or head_chars_key in state_dict:
                # Use whichever is available
                w_key = head_chars_key if head_chars_key in state_dict else legacy_head_key
                b_key = w_key.replace(".weight", ".bias")
                
                saved_size = state_dict[w_key].shape[0]
                current_size = model.heads["char_recognition"]["actor"].weight.shape[0]
                
                if saved_size == 10 and current_size == 62:
                    print(f">> [SURGERY] Migrating weights: {saved_size} -> {current_size} classes")
                    new_weight = model.heads["char_recognition"]["actor"].weight.clone()
                    new_bias = model.heads["char_recognition"]["actor"].bias.clone()
                    
                    new_weight[:10] = state_dict[w_key]
                    new_bias[:10] = state_dict[b_key]
                    
                    # Update state_dict to use NEW keys and NEW values
                    state_dict[head_chars_key] = new_weight
                    state_dict[head_chars_key.replace(".actor.weight", ".actor.bias")] = new_bias
                    
                    # Also handle Critic if it exists
                    legacy_critic = "head_chars_value.weight"
                    if legacy_critic in state_dict:
                         state_dict["heads.char_recognition.critic.weight"] = state_dict[legacy_critic]
                         state_dict["heads.char_recognition.critic.bias"] = state_dict[legacy_critic.replace(".weight", ".bias")]

            # NEW: Compass Surgery (Migrate old 4-bit to new 8-bit compass)
            if "fc_shared.weight" in state_dict:
                saved_in = state_dict["fc_shared.weight"].shape[1]
                current_in = model.fc_shared.weight.shape[1]  # Should be 297 (288+1+8)
                if saved_in != current_in:
                    print(f">> [SURGERY] Compass Migration: {saved_in} -> {current_in} inputs")
                    new_fc_w = model.fc_shared.weight.clone()
                    # Copy shared weights up to the minimum common dimension
                    min_dim = min(saved_in, current_in)
                    new_fc_w[:, :min_dim] = state_dict["fc_shared.weight"][:, :min_dim]
                    # New compass bits remain freshly initialized (zero is safe)
                    state_dict["fc_shared.weight"] = new_fc_w
            
            model.load_state_dict(state_dict, strict=False)
            print("loaded weights (with Surgery if needed).")
        except Exception as e:
            print(f"Starting fresh (New Arch: {e})")
    else:
         print("Starting fresh.")
    
    logger = LogAggregator(zmq_pub=pub_sock_stats) # Pass socket
    concept_mapper = ConceptMapper() # Initialize Mapper
    buffer = ReplayBuffer() # Initialize Memory
    curiosity = CuriosityModule() # Initialize Curiosity
    
    # [AGI] Phase 1: Initialize Intrinsic Motivation System
    global INTRINSIC_MOTIVATION
    INTRINSIC_MOTIVATION = CombinedIntrinsicMotivation(num_actions=4, device=str(device))
    print(">> [AGI] Intrinsic Motivation Module ENABLED (ICM + Count-Based Exploration)")

    # [AGI] Phase 1: Modular Teacher Initialization
    # Re-check flag in case it was modified
    current_teacher: TeacherInterface = NullTeacher() if NO_TEACHER else HeuristicTeacher()
    print(f">> [AGI] Active Teacher: {current_teacher.name} (Flag: {NO_TEACHER})")
    
    history = {"snake": deque(maxlen=4), "pong": deque(maxlen=4), "maze": deque(maxlen=4)}
    
    print(">> NEURO-SYMBOLIC SNN: DASHBOARD ENABLED (Port 5557)...")
    
    steps_total = 0
    last_game_type = None

    while True:
        try:
            msg = pull_sock.recv_json()
            
            # Robust check: if msg is a string, wrap it in a dummy dict or handle it
            if isinstance(msg, str):
                msg = {"type": "raw", "data": msg}
                
            # check message type if dict
            # Standard inputs: {'x':.., 'type':..}
            # New Chat input: {'type': 'CHAT_INPUT', 'text': 'Hello'}
            
            msg_type = msg.get("type", "game_state") or "game_state" # handle empty type
            
            if msg_type == "CHAT_INPUT":
                user_text = msg.get("text", "")
                if user_text:
                    teach_mode = msg.get("teach_mode", False)
                    print(f"[{time.strftime('%H:%M:%S')}] [CHAT] USER: {user_text} (Teach: {teach_mode})")
                    reply = COGNITIVE_ENGINE.process_dialogue(user_text, teach_mode=teach_mode)
                    print(f"[{time.strftime('%H:%M:%S')}] [CHAT] AGENT: {reply}")
                    
                    # Broadcast reply to dashboard
                    pub_sock_stats.send_string(f"CHAT_RESPONSE:{json.dumps({'text': reply})}")
                continue
            
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
                     print(">> REPLAY BUFFER CLEARED")
                elif cmd == "full_reset":
                     # Full brain reset: reinit weights, clear buffer, delete saved model
                     with model_lock:
                         # Reinitialize model with fresh random weights
                         model.__init__(beta=0.5)
                         model.to(device)
                         # Reinitialize optimizer for new model params
                         optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
                     # Clear replay buffer
                     buffer = ReplayBuffer()
                     # Delete saved model file
                     if os.path.exists(MODEL_PATH):
                         os.remove(MODEL_PATH)
                         print(f">> DELETED SAVED MODEL: {MODEL_PATH}")
                     # Clear frame history
                     for key in history:
                         history[key].clear()
                     print(">> *** FULL BRAIN RESET COMPLETE *** - Starting from scratch!")
                elif cmd == "toggle_teacher":
                     NO_TEACHER = not NO_TEACHER
                     current_teacher = NullTeacher() if NO_TEACHER else HeuristicTeacher()
                     print(f">> TEACHER STATUS: {current_teacher.name}")
                elif cmd == "strict_transfer_cfg":
                     NO_TEACHER = True
                     current_teacher = NullTeacher()
                     FREEZE_PONG = True  # This also freezes Maze (see line ~1078)
                     print(">> STRICT TRANSFER CONFIG APPLIED:")
                     print("   - Teacher: OFF")
                     print("   - Maze/Pong Learning: FROZEN (weights will NOT update)")
                     print("   - Snake Learning: ACTIVE (can still train)")
                elif cmd == "transfer_test":
                     # DEFINITIVE TRANSFER TEST MODE
                     NO_TEACHER = True
                     FREEZE_PONG = True  # Freezes Maze weights
                     print("="*50)
                     print(">> TRANSFER TEST MODE ACTIVATED")
                     print("   - Teacher: OFF (Brain on its own)")
                     print("   - Maze Weights: FROZEN (no learning, pure transfer)")
                     print("   - Snake Knowledge Only: What it learned STAYS")
                     print("="*50)
                elif cmd == "freeze_all":
                     FREEZE_ALL = not FREEZE_ALL
                     if FREEZE_ALL:
                         print("="*50)
                         print(">> WEIGHTS FROZEN: Brain is in INFERENCE-ONLY mode")
                         print("   - NO learning will occur (any game/task)")
                         print("   - Brain operates purely on existing knowledge")
                         print("="*50)
                     else:
                         print(">> WEIGHTS UNFROZEN: Brain can learn again")
                elif cmd == "dump_causal_graph":
                    # [AGI] Phase 4.1 Verification
                    path = msg.get("path", "causal_dump.json")
                    print(f">> [MISSION] Dumping Causal Graph to {path}...")
                    try:
                        graph_data = {}
                        if hasattr(COGNITIVE_ENGINE, 'causal_graphs'):
                            for task, graph in COGNITIVE_ENGINE.causal_graphs.items():
                                edges = []
                                for link in graph.all_links:
                                    edges.append({
                                        "cause": link.cause, 
                                        "effect": link.effect, 
                                        "strength": link.strength,
                                        "type": link.relation.value
                                    })
                                graph_data[task] = edges
                        
                        with open(path, 'w') as f:
                            json.dump(graph_data, f, indent=2)
                        print(f"   [MISSION] Dumped {len(graph_data)} contexts to {path}")
                    except Exception as e:
                        print(f"   [MISSION] Error dumping graph: {e}")
                elif cmd == "toggle_dream":
                     AUTO_DREAM = not AUTO_DREAM
                     print(f">> AUTO DREAMING: {'ENABLED' if AUTO_DREAM else 'DISABLED'}")           
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
                            # Keep intrinsic motivation module on the same device
                            try:
                                if INTRINSIC_MOTIVATION is not None and hasattr(INTRINSIC_MOTIVATION, "set_device"):
                                    INTRINSIC_MOTIVATION.set_device(str(device))
                            except Exception as e:
                                print(f">> [DEVICE] Warning: failed to move intrinsic motivation to {device}: {e}")
                            # Re-init optimizer if moving between CPU/GPU to ensure state is on correct device
                            # Actually, Adam state can be moved, but it's often safer to re-init or use a helper
                            # For SNN-QAT, re-init with same params is fine since we are mostly doing Live training
                            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                            print(f">> [DEVICE] Model migrated to {device}")
                elif cmd == "assign_task":
                    # Manually set the active task from dashboard
                    target_task = msg.get("task", "snake")
                    print(f">> [MISSION] Task Assigned: {target_task.upper()}")
                    # This allows the dashboard to force the server into a specific task mode
                    # if needed, though usually the client PULLS the task.
                    # We can use this to bias dreaming or other background processes.
                elif cmd == "set_goal":
                    goal = msg.get("goal", "maximize_score")
                    val = msg.get("value", 0)
                    print(f">> [MISSION] Goal Set: {goal} ({val})")
                    # Store in CognitiveEngine or local mission state
                    if hasattr(COGNITIVE_ENGINE, 'set_mission_goal'):
                        COGNITIVE_ENGINE.set_mission_goal(goal, val)
                elif cmd == "toggle_teacher_for_task":
                    task = msg.get("task", "snake")
                    # Handle per-task teacher toggles if we want to be granular
                    # For now, we'll just log and use the global NO_TEACHER
                    print(f">> [MISSION] Teacher Toggle for {task}: {'OFF' if NO_TEACHER else 'ON'}")
                elif cmd == "export_trace":
                    # Export the last N cognitive traces for evaluation
                    print(">> [MISSION] Exporting Cognitive Trace Logs...")
                    # Implementation detail: CE handles the actual export
                    COGNITIVE_ENGINE.export_traces("mission_trace.json")
                continue # Skip game logic

            # --- PREDICTION REQUEST (CHARACTERS) ---
            if msg.get("type") == "predict_char":
                # Handle Character Prediction
                try:
                    # 1. Decode Image (Expects base64 of 28x28 or similar)
                    img_bytes = base64.b64decode(msg["image"])
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        raise ValueError("Decoded image is None")

                    # 2. Resize to 10x10
                    img_10 = cv2.resize(img, (10, 10), interpolation=cv2.INTER_AREA)

                    # 3. Preprocess
                    # Normalize
                    img_float = img_10.astype(np.float32) / 255.0

                    # Canvas is black ink on white background -> invert to match EMNIST-like polarity
                    img_float = 1.0 - img_float

                    # Light thresholding to keep strokes after aggressive downscale
                    img_float = (img_float > 0.2).astype(np.float32) * img_float

                    def _center_10x10(x: np.ndarray) -> np.ndarray:
                        ys, xs = np.where(x > 0.1)
                        if len(xs) == 0 or len(ys) == 0:
                            return x
                        x0, x1 = int(xs.min()), int(xs.max())
                        y0, y1 = int(ys.min()), int(ys.max())
                        crop = x[y0:y1 + 1, x0:x1 + 1]
                        canvas = np.zeros((10, 10), dtype=np.float32)
                        ch, cw = crop.shape
                        if ch > 0 and cw > 0 and ch <= 10 and cw <= 10:
                            oy = (10 - ch) // 2
                            ox = (10 - cw) // 2
                            canvas[oy:oy + ch, ox:ox + cw] = crop
                            return canvas
                        return x

                    # Two candidate orientations:
                    # - as-is (for UI canvases that already match training orientation)
                    # - rotated+flipped (common EMNIST transpose/rotation mismatch)
                    cand_a = _center_10x10(img_float)
                    cand_b = _center_10x10(np.fliplr(np.rot90(img_float, k=1)))

                    # 4. Inference
                    # IMPORTANT:
                    # The SNN's generic forward returns *summed output spikes*.
                    # For char classification, that can be all-zeros early on (flat softmax).
                    # Instead, use a raw actor linear readout for decoding.
                    def _infer_logits(grid_10: np.ndarray) -> torch.Tensor:
                        frames = np.stack([grid_10] * 4, axis=0)  # [4, 10, 10]
                        inp_local = torch.tensor(frames).unsqueeze(0).float().to(device)  # [1, 4, 10, 10]
                        latent_local = model.encoder(inp_local)
                        w_shared_local = ternarize_weight(model.fc_shared.weight)
                        mem_shared_local = model.lif_shared.init_leaky()
                        cur_shared_local = torch.nn.functional.linear(latent_local, w_shared_local, model.fc_shared.bias)
                        spk_shared_local, mem_shared_local = model.lif_shared(cur_shared_local, mem_shared_local)

                        # NOTE:
                        # Using spikes as a feature vector is often too sparse for 62-way char decoding,
                        # especially early in training. Use the analog membrane/current instead.
                        shared_feat = mem_shared_local

                        head_local = model.heads["char_recognition"]
                        w_actor_local = ternarize_weight(head_local["actor"].weight)
                        out = torch.nn.functional.linear(shared_feat, w_actor_local, head_local["actor"].bias)

                        # Return logits + lightweight debug stats
                        dbg = {
                            "cur_mean": float(cur_shared_local.detach().mean().cpu().item()),
                            "mem_mean": float(mem_shared_local.detach().mean().cpu().item()),
                            "spk_mean": float(spk_shared_local.detach().mean().cpu().item()),
                            "mem_abs_mean": float(mem_shared_local.detach().abs().mean().cpu().item()),
                        }
                        return out, dbg

                    with model_lock:
                        model.eval()
                        out_a, dbg_a = _infer_logits(cand_a)
                        out_b, dbg_b = _infer_logits(cand_b)
                        model.train()

                    probs_a = torch.softmax(out_a, dim=1).detach().cpu()[0]
                    probs_b = torch.softmax(out_b, dim=1).detach().cpu()[0]

                    # Pick the orientation with higher peak probability (more confident)
                    a_max = float(torch.max(probs_a).item())
                    b_max = float(torch.max(probs_b).item())
                    if b_max > a_max:
                        probs_t = probs_b
                        img_float = cand_b
                        orientation = "rot90+fliplr"
                        dbg = dbg_b
                    else:
                        probs_t = probs_a
                        img_float = cand_a
                        orientation = "asis"
                        dbg = dbg_a

                    probs = probs_t.numpy()
                    pred_class = int(torch.argmax(probs_t).item())

                    entropy = calculate_entropy(probs_t.unsqueeze(0)).item()
                    explanation = concept_mapper.get_explanation(pred_class)

                    # Debug: top-k to spot uniform outputs / dead heads
                    topk_vals, topk_idx = torch.topk(probs_t, k=min(5, probs_t.numel()))
                    topk = [(int(i.item()), float(v.item())) for v, i in zip(topk_vals, topk_idx)]

                    # 5. Send Result (VIS Channel)
                    res_payload = {
                        "task": "char_recognition",
                        "grid": img_float.tolist(),
                        "probs": probs.tolist(),
                        "prediction": pred_class,
                        "explanation": explanation,
                        "entropy": entropy,
                        "debug": {
                            "img_mean": float(np.mean(img_float)),
                            "img_min": float(np.min(img_float)),
                            "img_max": float(np.max(img_float)),
                            "orientation": orientation,
                            "shared": dbg,
                            "topk": topk,
                        },
                        "teacher": -1,
                        "student": pred_class,
                        "agreed": True,
                    }
                    pub_sock_stats.send_string(f"VIS:{json.dumps(res_payload)}")

                    print(
                        f"[CHAR] Predicted: {pred_class} -> {explanation} "
                        f"(Conf: {probs[pred_class]:.2f}, Entropy: {entropy:.2f}, Ori: {orientation}, Top5: {topk})"
                    )

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
            
            # Robust image handling - fallback to blank if missing
            img_b64 = msg.get("image", "")
            if img_b64:
                img_bytes = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape != (10, 10):
                    img = cv2.resize(img, (10, 10)) if img is not None else np.zeros((10, 10), dtype=np.uint8)
            else:
                # No image provided - create blank
                img = np.zeros((10, 10), dtype=np.uint8)
            
            # Adversarial (Snake)
            obstacles_extra = []
            if game_type == "snake" and random.random() < ADVERSARIAL_RATE:
                hx, hy = state_data["head"]
                fx, fy = state_data["food"]
                mx, my = (hx+fx)//2, (hy+fy)//2
                if (mx!=hx or my!=hy) and (mx!=fx or my!=fy): obstacles_extra.append((mx,my))

            # Teacher (Modular)
            teacher_idx = 0
            task_id = 0 
            
            # 1. Ask Teacher for Advice
            advice = current_teacher.get_advice(state_data, game_type)
            if advice is not None:
                teacher_idx = advice
            else:
                teacher_idx = 0 # Default if teacher abstains (autonomous fallback?)
            
            # 2. Add Task IDs
            if game_type == "snake":
                task_id = 1
            elif game_type == "pong":
                task_id = 0
            elif game_type == "maze":
                task_id = 1 # Transfer from snake
            
            # Debug Oracle (Optional logging)
            if False and steps_total % 100 == 0 and isinstance(current_teacher, HeuristicTeacher):
                 print(f"[TEACHER] {game_type.upper()} suggests: {teacher_idx}")

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
            
            # --- COMPASS CALCULATION (8-bit: 4 Goal + 4 Blocked) ---
            # Bits 0-3: Goal Direction | Bits 4-7: Obstacle Blocked
            compass_bits = [0, 0, 0, 0, 0, 0, 0, 0]
            
            if game_type in ["snake", "maze"]:
                hx, hy = state_data.get("head", state_data.get("player_pos", (0, 0)))
                fx, fy = state_data.get("food", state_data.get("exit_pos", (0, 0)))
                w = state_data.get("width", 10)
                h = state_data.get("height", 10)
                
                # Goal Direction (bits 0-3)
                if fy < hy: compass_bits[0] = 1  # Goal Above
                if fy > hy: compass_bits[1] = 1  # Goal Below
                if fx < hx: compass_bits[2] = 1  # Goal Left
                if fx > hx: compass_bits[3] = 1  # Goal Right
                
                # Obstacle Awareness (bits 4-7) - Check adjacent cells in image
                # CRITICAL: Scale hx, hy (raw coords) to img coords (10x10)
                img_h, img_w = img.shape
                def is_blocked(rx, ry):
                    if rx < 0 or rx >= w or ry < 0 or ry >= h: return True
                    # Map raw coord to image coord
                    ix = int(rx * img_w / w)
                    iy = int(ry * img_h / h)
                    # Clip just in case of rounding
                    ix = max(0, min(img_w - 1, ix))
                    iy = max(0, min(img_h - 1, iy))
                    return img[iy, ix] == 255

                if is_blocked(hx, hy - 1): compass_bits[4] = 1
                if is_blocked(hx, hy + 1): compass_bits[5] = 1
                if is_blocked(hx - 1, hy): compass_bits[6] = 1
                if is_blocked(hx + 1, hy): compass_bits[7] = 1
                    
            elif game_type == "pong":
                # Ball relative to paddle
                paddle_center = state_data.get("p1_y", 0) + 3
                ball_y = state_data.get("ball_y", 0)
                if ball_y < paddle_center - 1: compass_bits[0] = 1
                if ball_y > paddle_center + 1: compass_bits[1] = 1
            
            compass_tensor = torch.tensor([compass_bits], dtype=torch.float, device=device)

            # Forward (WITH LOCK)
            # Forward (WITH LOCK)
            with model_lock:
                 optimizer.zero_grad()
                 model.train() # Default to train mode
                 # Universal Call: No compass needed, just State + TaskName
                 rate_out, value_out = model(input_tensor, task_name=game_type)
            
            # --- VSA PRIOR MASKING (GATED TRANSFER) ---
            snn_probs = torch.softmax(rate_out, dim=1)
            entropy = calculate_entropy(snn_probs).item()
            
            # ACCUMULATE LOSSES FOR SINGLE UPDATE
            total_step_loss = 0.0
            loss_source = []
            
            # Default to SNN
            probs = snn_probs
            prior_active = False
            vsa_prior = np.array([])

            # GATED RESCUE
            if ENABLE_VSA and entropy > CONFIDENCE_THRESHOLD:
                # System 1 is Confused -> System 2 (VSA) Intervention
                vsa_prior = ActionSemantics.get_goal_alignment(game_type, state_data)
                
                if len(vsa_prior) > 0:
                    prior_tensor = torch.from_numpy(vsa_prior).to(device).unsqueeze(0) # [1, Actions]
                    
                    # Apply Bias - but only if dimensions match
                    if prior_tensor.shape[1] == snn_probs.shape[1]:
                        biased_probs = snn_probs * (1.0 + VSA_STRENGTH * prior_tensor)
                        probs = biased_probs / biased_probs.sum(dim=1, keepdim=True)
                        prior_active = True
                    else:
                        # Dimension mismatch - skip VSA rescue for this game
                        pass
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
            student_idx = None
            
            # [AGI] Phase 2: Cognitive Engine Integration
            # We consult the Unified Engine for Reasoning/Planning/Rules/Metacognition
            snn_conf_val = torch.max(probs).item()
            cog_decision = COGNITIVE_ENGINE.decide(
                state_data, 
                game_type, 
                {"confidence": snn_conf_val, "action": "ACTION_" + ["UP", "DOWN", "LEFT", "RIGHT"][snn_raw_idx]}
            )
            
            # Extract Intent from Cognitive Engine
            # We respect the Engine if it has a Plan, a Rule, or specific Veto logic.
            active_mode = cog_decision.trace.get("mode", "default")
            veto_active = cog_decision.trace.get("veto_confusion", False)
            
            planned_idx = None
            
            # Priority: Plan > Rule/Veto > SNN
            if active_mode in ["PLANNER", "RULES", "planned_spatial_goal", "learned_rule"] or veto_active:
                act_str = cog_decision.chosen_action.replace("ACTION_", "")
                # Map to index
                act_names = ["UP", "DOWN", "LEFT", "RIGHT"] if game_type != "pong" else ["UP", "DOWN"]
                
                if act_str in act_names:
                    planned_idx = act_names.index(act_str)
                    
                    # Log source
                    trigger = active_mode.upper()
                    if veto_active: trigger += " (VETO)"
                    detail = cog_decision.explanation.details if cog_decision.explanation else ""
                    print(f"[{game_type.upper()}] COGNITIVE OVERRIDE: {act_str} [{trigger}] {detail}")

            
            # --- CURIOSITY & EXPLORATION ---
            # Driven by Novelty (VSA) and Stagnation
            should_explore = False
            exploration_reason = ""
            
            # If Planner has a suggestion, we prioritize it (System 2 Override)
            if planned_idx is not None:
                student_idx = planned_idx
                # We skip random exploration if we have a plan
            elif NO_TEACHER: # Only explore if Teacher is OFF (Autonomy Mode)
                # 1. Generate Situation Hypervector (Neuro-Symbolic State)
                # We reuse primitives from symbol_grounding
                situation_hv = hypervec_rs.HyperVector(0) # Null start
                
                # Goal Relations (Snake/Maze)
                if game_type in ["snake", "maze"]:
                    hx, hy = state_data.get("head", state_data.get("player_pos", (0, 0)))
                    fx, fy = state_data.get("food", state_data.get("exit_pos", (0, 0)))
                    
                    if fy < hy: situation_hv = situation_hv.bundle(hypervec_rs.HyperVector(GLOBAL_PRIMITIVES_MAP["REL_ABOVE"]))
                    if fy > hy: situation_hv = situation_hv.bundle(hypervec_rs.HyperVector(GLOBAL_PRIMITIVES_MAP["REL_BELOW"]))
                    if fx < hx: situation_hv = situation_hv.bundle(hypervec_rs.HyperVector(GLOBAL_PRIMITIVES_MAP["REL_LEFT"]))
                    if fx > hx: situation_hv = situation_hv.bundle(hypervec_rs.HyperVector(GLOBAL_PRIMITIVES_MAP["REL_RIGHT"]))
                    
                    # Obstacle Awareness (Wall Detection)
                    # Check adjacent cells using scaled coordinates
                    img_h, img_w = img.shape
                    w = state_data.get("width", 10)
                    h = state_data.get("height", 10)
                    dirs = [(0, -1, 201), (0, 1, 202), (-1, 0, 203), (1, 0, 204)]
                    for dx, dy, code in dirs:
                        nx, ny = hx + dx, hy + dy
                        if nx < 0 or nx >= w or ny < 0 or ny >= h:
                            situation_hv = situation_hv.bundle(hypervec_rs.HyperVector(code))
                        else:
                            ix = max(0, min(img_w - 1, int(nx * img_w / w)))
                            iy = max(0, min(img_h - 1, int(ny * img_h / h)))
                            if img[iy, ix] == 255:
                                situation_hv = situation_hv.bundle(hypervec_rs.HyperVector(code))
                    
                # 2. Consult Curiosity Module
                # Confidence = (1 - Entropy) or Max Prob? 
                # Entropy 0 = High Conf. Entropy 1.3 = Low Conf.
                # Let's use Max Prob as confidence proxy.
                max_conf = torch.max(probs).item()
                
                decision = curiosity.should_explore(situation_hv, game_type, max_conf)
                
                if decision.should_explore:
                    should_explore = True
                    exploration_reason = decision.reason
                    
                    # Update Prototype Memory (if novel)
                    curiosity.update_prototype(situation_hv, game_type)
            
            if should_explore:
                # Ask Curiosity Module for a recommended action (Softmax exploration)
                # It needs probabilities to bias against best (if strictly exploring) or just sample.
                p_list = probs[0].tolist() # Batch 0
                act_names = ["UP", "DOWN", "LEFT", "RIGHT"] if game_type != "pong" else ["UP", "DOWN"]
                
                rec_act_name = curiosity.get_exploration_action(act_names, p_list, explore_rate=0.4) # 40% temp boost
                
                # Map back to index
                if rec_act_name in act_names:
                    student_idx = act_names.index(rec_act_name)
                    print(f"[{game_type.upper()}] CURIOSITY: {rec_act_name} ({exploration_reason})")
                    trace_mode = "CURIOSITY"
                else:
                    # Fallback to SNN if curiosity couldn't recommend a valid action
                    _, active_idx_t = torch.max(probs, dim=1)
                    student_idx = active_idx_t.item()
                    trace_mode = "SNN_FALLBACK"
            else:
                # Standard SNN prediction
                if game_type == "char_recognition":
                    _, active_idx_t = torch.max(probs, dim=1)
                    if student_idx is None:
                        student_idx = active_idx_t.item()
                    trace_mode = "SNN"
                else:
                    _, active_idx_t = torch.max(probs, dim=1)
                    snn_idx = active_idx_t.item()
                    if student_idx is None:
                        student_idx = snn_idx
                        trace_mode = "SNN"
                    else:
                        # Keep existing student_idx (from Planner or Curiosity)
                        # but we still log the SNN's opinion for the dashboard
                        pass
                
            # Log Outcome for Curiosity (Learning Progress)
            # We don't know "Success" yet. We find out next frame? 
            # Or we use Reward? 
            # Current frame reward is for LAST action.
            # Curiosity module tracks window.
            # We record outcome of *this* step in the *next* loop iteration?
            # actually, let's just feed current Reward as proxy for *previous* action success.
            is_success = (reward > 0)
            curiosity.record_outcome(game_type, is_success)
            
            # --- [AGI 6.2] COGNITIVE VETO & SUPER-LEARN ---
            # Replace hardcoded simulations with Unified Engine's vetoes (World Model Simulation)
            vetoed_actions = cog_decision.trace.get("vetoes", [])
            reasoning_override = (student_idx != snn_raw_idx)
            
            if vetoed_actions and NO_TEACHER and not args.eval:
                with model_lock:
                    # RE-FORWARD TO GET FRESH GRAPH
                    safe_logits, _ = model(input_tensor, task_name=game_type)
                    dist_safety = torch.distributions.Categorical(logits=safe_logits)
                    
                    for vetoed_act in vetoed_actions:
                        act_name = vetoed_act.replace("ACTION_", "")
                        if act_name in act_names:
                            unsafe_idx = act_names.index(act_name)
                            # A. Penalty for Bad Thought (Imagined Death/Failure)
                            log_prob_unsafe = dist_safety.log_prob(torch.tensor([unsafe_idx], device=device))
                            loss_penalty = -(log_prob_unsafe * -0.5) 
                            total_step_loss += loss_penalty
                            loss_source.append("COG_VETO")
                            
                    # B. Reward for Safe Action (Winner of Global Workspace)
                    if reasoning_override:
                        log_prob_safe = dist_safety.log_prob(torch.tensor([student_idx], device=device))
                        loss_teach = -(log_prob_safe * 1.0)
                        total_step_loss += loss_teach
                        loss_source.append("COG_TEACH")
                        
                feedback_msg = f"Vetoed: {vetoed_actions}" if vetoed_actions else ""
                if reasoning_override: feedback_msg += f" Taught: {act_names[student_idx]}"
                print(f"[{game_type.upper()}] CROSS-MODULE FEEDBACK: {feedback_msg}")

            # 3. Training & Agreement Logic
            # CRITICAL FIX: Train if SNN was wrong, even if System was right.
            
            # Did the SNN get it right natively?
            snn_agreed = (snn_raw_idx == teacher_idx)
            
            # Did the Final System get it right?
            system_agreed = (student_idx == teacher_idx)
            
            # DEFINE FINAL ACTION (Moved up for RL Training)
            # [AGI] Safety: If teacher is requested but none provided by client, act independently
            # This is critical for Maze as it currently has no UI-based AI teacher.
            frame_no_teacher = NO_TEACHER or (teacher_idx == -1)
            
            if frame_no_teacher:
                final_action_idx = student_idx
                system_agreed = True # Effectively agree with self when alone
            else:
                final_action_idx = student_idx if system_agreed else teacher_idx
            
            loss_val = 0.0
            
            # --- TRAINING (HYBRID) ---
            # Gated Training Rule:
            # - If SNN is correct: No loss.
            # - If SNN is wrong: Train (unless Eval mode).
            # - If NO_TEACHER: RL Mode (Policy Gradient).
            
            if not args.eval: 
                # [LOCK CRITICAL SECTION] 
                # We must hold the lock during the ENTIRE Forward->Backward process
                # otherwise the Sleep Thread can update weights and invalidate our graph.
                with model_lock:
                    # [AGI] Universal Learning (Causal Discovery)
                    # We now have the complete transition: State(t-1) -> Action(t-1) -> Reward(t)
                    if session_id in RL_CONTEXT:
                        prev_state, prev_task, prev_action, prev_val_old, prev_state_data, prev_img = RL_CONTEXT[session_id]
                        
                        act_names = ["UP", "DOWN", "LEFT", "RIGHT"] if prev_task != "pong" else ["UP", "DOWN"]
                        act_str = "ACTION_" + act_names[prev_action] if prev_action < len(act_names) else "ACTION_STAY"
                        
                        COGNITIVE_ENGINE.learn(
                            state=prev_state_data,
                            action=act_str,
                            reward=reward,
                            task_tag=prev_task,
                            outcome="success" if reward > 0 else ("failure" if reward < 0 else "neutral"),
                            next_state=state_data,
                            image=prev_img 
                        )

                    if NO_TEACHER:
                        # A2C UPDATE (Online)
                        if session_id in RL_CONTEXT:
                            # Re-use already unpacked variables
                            pass # Actual A2C logic follows below in the file

                            # This generates gradients for the CURRENT weights w.r.t the previous decision
                            # This avoids the "Inplace Operation" error because we build a NEW graph.
                            
                            prev_logits, prev_val_new = model(prev_state, task_name=prev_task)
                            
                            # [AGI] Phase 1: Compute Intrinsic Motivation Reward
                            # Augments extrinsic reward with curiosity bonus for autonomous learning
                            total_reward = reward  # Start with extrinsic
                            if INTRINSIC_MOTIVATION is not None and NO_TEACHER:
                                # Compute intrinsic reward from state transition
                                intrinsic_r, breakdown = INTRINSIC_MOTIVATION.compute_reward(
                                    prev_state.squeeze(0),  # Previous state
                                    prev_action,             # Action taken
                                    input_tensor.squeeze(0), # Current state (next state)
                                    extrinsic_reward=reward
                                )
                                total_reward = intrinsic_r
                               # Log occasionally
                                if steps_total % 50 == 0:
                                    print(f"[AGI] Intrinsic: ext={reward:.2f} icm={breakdown['icm_bonus']:.4f} count={breakdown['count_bonus']:.4f} total={total_reward:.2f}")
                                
                                # [AGI] Log to CSV
                                with open(LOG_FILE, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        time.time(), 
                                        steps_total, 
                                        game_type, 
                                        reward, 
                                        breakdown['icm_bonus'], 
                                        breakdown['count_bonus'], 
                                        total_reward
                                    ])
                                    
                                # [AGI] Phase 1.4: Update Learning Progress
                                CURIOSITY_TRACKER.update(game_type, total_reward)
                                
                                # Check for Plateau (Self-Curriculum)
                                # For Phase 1, we just LOG it. In Phase 2, we switch tasks.
                                lp = CURIOSITY_TRACKER.get_progress(game_type)
                                if steps_total % 200 == 0:
                                    is_bored = CURIOSITY_TRACKER.is_plateaued(game_type)
                                    note = " [BORED]" if is_bored else ""
                                    print(f"[AGI] Learning Progress ({game_type}): {lp:.4f}{note}")

                            
                            # 1. Critic Target
                            # Target = Reward + Gamma * V(Current_State)
                            # value_out is V(Current_State). We detach it as it's just a number for the target.
                            gamma = 0.99
                            target_value = total_reward + gamma * value_out.detach().squeeze()
                            if done: target_value = torch.tensor(total_reward).float().to(device)
                            
                            # 2. Advantage
                            # Adv = Target - V(Previous_State)
                            # We use the detached old value for the baseline? Or the new one?
                            # Standard A2C uses the value estimate from the graph we are training. 
                            # So Adv = Target - prev_val_new.
                            # Wait, we want to maximize Advantage. 
                            advantage = target_value - prev_val_new.detach().squeeze()
                            
                            # 3. Actor Loss
                            dist_prev = torch.distributions.Categorical(logits=prev_logits)
                            log_prob_prev = dist_prev.log_prob(torch.tensor([prev_action], device=device))
                            loss_actor = -(log_prob_prev * advantage)
                            
                            # 4. Critic Loss: MSE(V(prev), Target)
                            loss_critic = F.mse_loss(prev_val_new.squeeze(), target_value)
                            
                            loss = loss_actor + 0.5 * loss_critic
                            
                            total_step_loss = total_step_loss + loss 
                            loss_source.append(f"A2C(R={reward})")
                            
                            print(f"[{game_type.upper()}] A2C: R {reward:+.1f} | Val {prev_val_old:.2f}->{value_out.item():.2f} | Adv {advantage.item():.2f}")
                                
                            # [NEW] STORE EXPERIENCE IN INTELLIGENT BUFFER
                            # We store the *previous* state transition because we now know the reward and next state (current input_tensor)
                            # Priority = |Advantage| (TD Error approximation)
                            priority_val = float(abs(advantage).detach().cpu().item())
                            exp = Experience(
                                state=prev_state.detach().cpu(),
                                action_idx=prev_action,
                                reward=reward,
                                next_state=input_tensor.detach().cpu(),
                                done=False, # We don't strictly track 'done' here yet, but Snake dies on -1 usually
                                task_name=prev_task,
                                priority=priority_val,
                                timestamp=time.time()
                            )
                            REPLAY_BUFFER.add(exp)
    
                            del RL_CONTEXT[session_id]
                        
                        # --- SALIENCY MAP GENERATION (Every 100 frames) ---
                        # Already inside lock now
                        if random.random() < 0.01: # 1% chance per step
                             generate_attention_map(model, input_tensor, game_type)
    
                    elif snn_agreed:
                        pass # SNN effectively mastered this state
                    else:
                        # IMITATION LEARNING (Teacher active)
                        target = torch.tensor([teacher_idx], dtype=torch.long, device=device)
                        
                        # RE-FORWARD (Safe from Sleep Thread)
                        with model_lock:
                            imit_logits, _ = model(input_tensor, task_name=game_type)
                            # Use Actor Loss (Classification)
                            loss_actor = criterion(imit_logits, target)
                        
                        # Train Critic too?
                        # Yes, teach Critic that V(s) should be R(s)? No, in imitation we don't have R everywhere.
                        # Let's just train Actor in Imitation mode.
                        loss = loss_actor
                        
                        # STRICT TRANSFER: Freeze weights for Pong AND Maze, or ALL if FREEZE_ALL
                        should_train = True
                        if FREEZE_ALL:
                            should_train = False  # Complete freeze - no learning at all
                        elif (game_type == "pong" or game_type == "maze") and FREEZE_PONG:
                            should_train = False  # Selective freeze for transfer test
                        
                        if should_train:
                             total_step_loss += loss
                             loss_source.append("IMITATION")

                    # PERFORM SINGLE OPTIMIZER STEP FOR ALL ACCUMULATED LOSSES
                    if isinstance(total_step_loss, torch.Tensor) and total_step_loss.requires_grad:
                        optimizer.zero_grad()
                        total_step_loss.backward()
                        optimizer.step()

                    # [AGI] Universal Context Update: Store T info for T+1 transition
                    # This enables Causal Discovery to learn from Teacher observations too!
                    RL_CONTEXT[session_id] = (input_tensor.clone(), game_type, final_action_idx, value_out.item(), state_data, curr_np)
                    if done:
                        if session_id in RL_CONTEXT: del RL_CONTEXT[session_id]
            
            # 4. Memory (Experience Replay)
            # We push the CURRENT state and action we just calculated.
            buffer.push(input_tensor, task_id, final_action_idx, reward, system_agreed, game_type, compass=compass_tensor)
            
            # Retrieve score from message payload
            current_score = msg.get("score", 0)
            
            # Extract metrics for detailed logging
            is_veto = False
            conf_val = 0.5
            if cog_decision:
                conf_val = cog_decision.confidence
                if cog_decision.explanation and "VETO" in cog_decision.explanation.summary:
                    is_veto = True
            
            logger.update(game_type, system_agreed, loss_val, current_score, session_id, veto=is_veto, confidence=conf_val)
            logger.check_print()
            
            # --- MISSION TELEMETRY (Every 10 Steps) ---
            if steps_total % 10 == 0:
                # 1. Global Workspace Competition
                workspace_data = COGNITIVE_ENGINE.get_workspace_telemetry()
                pub_sock_stats.send_string(f"WORKSPACE:{json.dumps(workspace_data)}")
                
                # 2. Causal Graph States
                causal_data = COGNITIVE_ENGINE.get_causal_telemetry(game_type)
                pub_sock_stats.send_string(f"CAUSAL:{json.dumps(causal_data)}")
                
                # 3. Vision & Prediction (Already existed, but ensuring veto_log is safe)
                veto_log = cog_decision.explanation.summary if cog_decision.explanation else "VETO"
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
                    "score": msg.get("score", 0),
                    "image": msg.get("image") # Pass through base64 image for dashboard
                }
                pub_sock.send_string(f"VIS:{json.dumps(vis_payload)}")

                # 2. Prepare Console Output
                ts = time.strftime('%H:%M:%S')
                t_act = vis_payload['teacher']
                s_act = vis_payload['student']
                
                if game_type == "char_recognition":
                     label_map = [str(i) for i in range(10)]
                elif game_type == "snake" or game_type == "maze":
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
            # If we want to be behaved nicely, we should arguably output the Teacher Action if we are training?
            # BUT: We want to see the failure.
            # Let's execute the System's Decision (student_idx), even if wrong.
            # UNLESS: It causes instant death? No, sim_snake handles veto.
            
            # Correction: Previously we used:
            # if agreed: action_idx = student_idx; else: action_idx = teacher_idx
            # This means we forced the Teacher action on failure. This is "Student Forcing".
            # Let's keep Student Forcing for stability.
            # UNLESS: --no-teacher is set. Then we MUST execute student_idx.
            
            if student_idx is None: student_idx = 0 # Fallback safety

            if NO_TEACHER:
                final_action_idx = student_idx
            else:
                final_action_idx = student_idx if system_agreed else teacher_idx
            
            if game_type == "snake" or game_type == "maze":
                # Both use 4-directional actions: UP, DOWN, LEFT, RIGHT
                cmd = actions[final_action_idx] if final_action_idx < 4 else "UP"
            else:  # Pong
                cmd = "UP" if final_action_idx == 0 else "DOWN"
            
            pub_sock.send_string(f"{game_type.upper()}:{cmd}")
            
            steps_total += 1
            
            if steps_total % SAVE_INTERVAL == 0: 
                 torch.save(model.state_dict(), MODEL_PATH)
            
            # --- SLEEP TRIGGER: INTERVAL ---
            # AUTO-SLEEP ENABLED VIA DASHBOARD
            if AUTO_DREAM and steps_total % 1000 == 0:
                 run_sleep_thread(model, optimizer, buffer, device, model_lock, "all")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")

# --- SALIENCY HELPER ---
def generate_attention_map(model, input_tensor, task_name):
    """
    Generate and save generic Saliency Map for debugging.
    """
    global SALIENCY
    if SALIENCY is None:
        SALIENCY = SaliencyVisualizer(model)
        
    try:
        heatmap = SALIENCY.generate_heatmap(input_tensor, task_name)
        if heatmap is not None:
             # Save to debug file (overwriting usually, or timestamped)
             # debug/saliency_latest.png
             if not os.path.exists("debug"): os.makedirs("debug")
             cv2.imwrite("debug/saliency_latest.png", heatmap)
    except Exception as e:
        print(f"[SALIENCY] Error: {e}")


# --- DREAMING FUNCTION ---
def perform_dreaming_cycle(model, optimizer, device):
    """
    Deep Dreaming: Train on archived memories (Disk) to consolidate knowledge.
    This prevents catastrophic forgetting by replaying efficient 'sketches' of the past.
    """
    print("\n[BRAIN] Entering REM Sleep (Deep Dreaming)...")
    # model.train() # Already in train mode usually
    
    # 1. Sample from Disk (Cold Storage)
    batch_data = REPLAY_BUFFER.sample_from_disk(batch_size=REPLAY_BATCH_SIZE)
    
    if batch_data is None:
        print("[BRAIN] ...Woke up (No dreams available)")
        return
        
    states, actions, rewards, next_states, dones, tasks = batch_data
    states = states.to(device)
    rewards = rewards.to(device)
    
    # 2. Dream Training
    # We treat archived states as individual samples for reinforcement
    with model_lock:
        optimizer.zero_grad()
        total_dream_loss_val = 0.0
        
        for i in range(len(states)):
            s = states[i].unsqueeze(0)
            a = actions[i].item()
            r = rewards[i].item()
            task = tasks[i]
            
            logits, val = model(s, task_name=task)
            
            # Critic Loss (Target = r)
            val_loss = F.mse_loss(val, torch.tensor([[r]], device=device))
            
            # Actor Loss
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor([a], device=device))
            
            advantage = r - val.item()
            actor_loss = -log_prob * advantage
            
            loss = actor_loss + 0.5 * val_loss
            
            # Backward IMMEDIATELY to free graph
            # We scale by 1.0/BatchSize if we wanted true mean, but here sum is fine (learning rate absorbs it)
            # Actually let's normalize by batch size for stability
            loss = loss / len(states) 
            loss.backward()
            
            total_dream_loss_val += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
            
        print(f"[BRAIN] Dreamt {len(states)} episodes. Loss: {total_dream_loss_val:.4f}")
        print("[BRAIN] ...Waking up refreshed.")

# --- CLEANUP ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSHUTDOWN REQUESTED")
    finally:
        if 'REPLAY_BUFFER' in globals():
            REPLAY_BUFFER.close()
        print("Brain Offline.")


