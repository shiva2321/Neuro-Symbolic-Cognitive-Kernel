"""
NSCK Learning Module
Neural network training utilities: replay buffers, sleep cycles, and stratified sampling.

Provides training infrastructure for multi-task SNNs including experience replay,
sleep-based consolidation, and priority sampling for continual learning.
"""
import torch
import torch.nn as nn
import time
import random
import threading
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience for replay."""
    state: torch.Tensor
    task_id: float
    action: int
    reward: float


class ReplayBuffer:
    """
    Stratified replay buffer for multi-task learning.
    
    Maintains separate buffers per game and agreement status
    to ensure balanced sampling.
    """
    
    def __init__(self, capacity_per_quadrant: int = 2500):
        """
        Initialize replay buffer.
        
        Args:
            capacity_per_quadrant: Max experiences per (game, agreed) pair
        """
        self.capacity = capacity_per_quadrant
        self.buffers: Dict[str, Dict[bool, deque]] = {
            "snake": {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)},
            "pong": {True: deque(maxlen=self.capacity), False: deque(maxlen=self.capacity)},
        }
    
    def push(
        self,
        state: torch.Tensor,
        task_id: int,
        action_idx: int,
        reward: float,
        agreed: bool,
        game_type: str
    ):
        """
        Add experience to buffer.
        
        Args:
            state: State tensor (will be detached and moved to CPU)
            task_id: Task ID
            action_idx: Action taken
            reward: Reward received
            agreed: Whether teacher and student agreed
            game_type: Which game ("snake" or "pong")
        """
        # Detach and move to CPU to save GPU RAM
        state_cpu = state.detach().cpu()
        
        # Normalize task_id
        tid = 1.0 if game_type == "snake" else 0.0
        
        experience = (state_cpu, tid, int(action_idx), float(reward))
        
        if game_type in self.buffers:
            self.buffers[game_type][agreed].append(experience)
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a stratified batch from all quadrants.
        
        Args:
            batch_size: Total batch size
            
        Returns:
            Tuple of (states, task_ids, actions, rewards) or None if empty
        """
        target_per_q = max(1, batch_size // 4)
        batch = []
        
        for game in ["snake", "pong"]:
            for agreed in [True, False]:
                buf = self.buffers[game][agreed]
                if len(buf) > 0:
                    count = min(len(buf), target_per_q)
                    batch.extend(random.sample(list(buf), count))
        
        if len(batch) == 0:
            return None
        
        random.shuffle(batch)
        
        states, task_ids, actions, rewards = zip(*batch)
        
        return (
            torch.stack(states, dim=0),
            torch.tensor(task_ids, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
        )
    
    def count(self) -> int:
        """Total experiences in buffer."""
        total = 0
        for game in self.buffers:
            for agreed in self.buffers[game]:
                total += len(self.buffers[game][agreed])
        return total
    
    def count_per_game(self, game: str) -> int:
        """Experiences for a specific game."""
        return sum(len(self.buffers[game][a]) for a in [True, False])


def sleep_cycle(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    model_lock: threading.Lock,
    target_game: str = "all",
    epochs: int = 5,
    batch_size: int = 32,
    no_teacher: bool = False
):
    """
    Run offline consolidation (sleep cycle).
    
    Replays experiences from buffer to consolidate learning.
    
    Args:
        model: SNN model
        optimizer: Optimizer
        buffer: Replay buffer
        device: Torch device
        model_lock: Threading lock for model access
        target_game: Which game to train ("all", "snake", or "pong")
        epochs: Number of training epochs
        batch_size: Batch size for training
        no_teacher: If True, use RL loss; otherwise use imitation loss
    """
    if buffer.count() < batch_size:
        return
    
    print(f">> [SLEEP] Consolidating Memories ({target_game.upper()})...")
    
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()
    
    games_to_train = ["snake", "pong"] if target_game == "all" else [target_game]
    
    for _ in range(epochs):
        for game in games_to_train:
            # Sample from specific game buffers
            batch = []
            try:
                buf_a = buffer.buffers[game][True]
                buf_d = buffer.buffers[game][False]
                if len(buf_a) > 0:
                    batch.extend(random.sample(list(buf_a), min(len(buf_a), batch_size // 4)))
                if len(buf_d) > 0:
                    batch.extend(random.sample(list(buf_d), min(len(buf_d), batch_size // 4)))
            except RuntimeError:
                continue  # Handle concurrent modification
            
            if not batch:
                continue
            
            obs, tasks, actions, rewards = zip(*batch)
            inp = torch.cat(obs, dim=0).to(device)
            a_lbl = torch.tensor(actions, dtype=torch.long).to(device)
            r_val = torch.tensor(rewards, dtype=torch.float).to(device)
            
            task_id = 1 if game == "snake" else 0
            
            # Training step with lock
            with model_lock:
                model.train()
                optimizer.zero_grad()
                out = model(inp, task_id)
                
                if no_teacher:
                    # RL Loss (Policy Gradient)
                    dist = torch.distributions.Categorical(logits=out)
                    log_probs = dist.log_prob(a_lbl)
                    loss = -(log_probs * r_val).mean()
                else:
                    # Imitation Loss
                    loss = criterion(out, a_lbl)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            
            # Brief yield to allow main thread to run
            time.sleep(0.01)
    
    if steps > 0:
        print(f"   AVG SLEEP LOSS ({target_game.upper()}): {total_loss / steps:.4f}")


def run_sleep_thread(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    model_lock: threading.Lock,
    target_game: str = "all",
    **kwargs
) -> threading.Thread:
    """
    Start sleep cycle in background thread.
    
    Returns:
        Started daemon thread
    """
    t = threading.Thread(
        target=sleep_cycle,
        args=(model, optimizer, buffer, device, model_lock, target_game),
        kwargs=kwargs,
        daemon=True
    )
    t.start()
    return t


class LiveTrainer:
    """
    Handles live imitation/RL training during gameplay.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_lock: threading.Lock
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_lock = model_lock
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(
        self,
        logits: torch.Tensor,
        target_action: int,
        reward: float = 0.0,
        use_rl: bool = False
    ) -> float:
        """
        Single training step.
        
        Args:
            logits: Model output logits
            target_action: Target action (teacher or executed)
            reward: Reward for RL mode
            use_rl: If True, use policy gradient; else use cross-entropy
            
        Returns:
            Loss value
        """
        with self.model_lock:
            self.optimizer.zero_grad()
            
            if use_rl:
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(
                    torch.tensor([target_action], device=self.device)
                )
                loss = -(log_prob * reward)
            else:
                target = torch.tensor(
                    [target_action], dtype=torch.long, device=self.device
                )
                loss = self.criterion(logits, target)
            
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
            
            return loss.item()


class SleepConsolidator:
    """
    Consolidates episodic experiences into semantic knowledge.
    Extracts stable patterns and causal relationships.
    """
    def __init__(self, episodic_memory, semantic_memory):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        
    def consolidate_during_sleep(self, task_tag: str):
        """Perform semantic consolidation."""
        print(f">> [SLEEP] Extracting Semantic Knowledge for {task_tag}...")
        
        # 1. Retrieve salient episodes
        episodes = self.episodic.retrieve_salient(task_tag, limit=100)
        if not episodes: return
        
        # 2. Extract Causal Patterns
        # Pattern: (Situation, Action) -> Reward > 0
        patterns: Dict[Tuple[str, str], int] = {}
        for ep in episodes:
            if ep.reward > 0.05:
                # Use task_tag and emotion as a rough situation key for now
                # In more advanced versions, we'd use predicates.
                key = (f"{task_tag}_{ep.emotion}", ep.action)
                patterns[key] = patterns.get(key, 0) + 1
                
        # 3. Promote stable patterns to Semantic Memory
        for (situation, action), count in patterns.items():
            if count >= 3:  # Rule of three
                # "Action X in Situation Y causes Reward"
                effect = f"reward_{task_tag}"
                
                # Ensure concepts exist
                if situation not in self.semantic.concept_graph:
                    self.semantic.add_concept(situation, {"type": "context"})
                if action not in self.semantic.concept_graph:
                    self.semantic.add_concept(action, {"type": "action"})
                if effect not in self.semantic.concept_graph:
                    self.semantic.add_concept(effect, {"type": "outcome"})
                    
                # Add causal link
                # (Situation, Action) -> Outcome
                # For simplicity in graph, we just link Action -> Outcome in Situation
                self.semantic.add_relation(action, "causes", effect)
                print(f"   [SEMANTIC] Discovered: {action} causes {effect} in {situation}")
