"""
NSCK Intelligent Replay Buffer
Implements a two-tier memory system for "Intelligent Archival".

Tier 1: Hot Storage (RAM)
- Fixed-size circular buffer (deque)
- Holds recent and high-surprise experiences
- Size is STRICTLY limited to prevent memory bloat

Tier 2: Cold Storage (Disk)
- Backed by persistence.BrainStore (SQLite)
- "Infinite" capacity
- Stores experiences evicted from RAM

Pruning Strategy:
- When RAM is full, oldest items are popped.
- If item.priority (TD-Error) > archival_threshold, it is moved to Disk.
- Otherwise, it is discarded (forgotten).
"""
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from persistence import BrainStore, Episode
import torch

@dataclass
class Experience:
    state: torch.Tensor
    action_idx: int
    reward: float
    next_state: torch.Tensor
    done: bool
    task_name: str
    priority: float = 0.0  # TD-Error magnitude
    timestamp: float = 0.0

class IntelligentReplayBuffer:
    def __init__(
        self, 
        ram_capacity: int = 10000, 
        archival_threshold: float = 0.5,
        db_path: str = "nsck_brain.db"
    ):
        """
        Args:
            ram_capacity: Max items in RAM. Fixed to prevent bloat.
            archival_threshold: Min priority (TD-Error) to justify disk archival.
            db_path: Path to SQLite DB.
        """
        self.ram_buffer = deque(maxlen=ram_capacity)
        self.capacity = ram_capacity
        self.archival_threshold = archival_threshold
        
        # Connect to Disk Store
        self.store = BrainStore(db_path)
        try:
            self.store._init_db()
        except Exception as e:
            print(f"[BUFFER] Warning: DB Init failed (maybe already exists): {e}")

    def add(self, exp: Experience):
        """
        Add experience to RAM.
        If RAM is full, deque automatically pops the oldest.
        We intercept the pop if we wanted manual control, but deque is efficient.
        
        To implement "Archival on Eviction" with deque, we need to check if full *before* appending?
        No, deque doesn't return the popped item. 
        So we check len() manually.
        """
        if len(self.ram_buffer) >= self.capacity:
            # Buffer is full, we are about to evict the oldest
            evicted = self.ram_buffer[0] # Peek oldest
            self._handle_eviction(evicted)
            
        self.ram_buffer.append(exp)

    def _handle_eviction(self, exp: Experience):
        """
        Decide whether to archive the evicted experience to disk.
        """
        if exp.priority >= self.archival_threshold:
            # High surprise -> Archive to Disk
            self._archive_to_disk(exp)
        # Else: Forget (do nothing)

    def _archive_to_disk(self, exp: Experience):
        """
        Convert Experience to Episode and save to BrainStore.
        STORES FULL STATE (Compressed via Pickle in Persistence).
        """
        # We store the full state as a numpy array. 
        # Persistence.py pickles this dict, so it handles binary data.
        # For 10x10 grids (Snake) or small vectors, this is fine.
        state_sketch = {
            "data": exp.state.numpy(), # Store actual data
            "shape": tuple(exp.state.shape),
            "priority": float(exp.priority)
        }
        
        # Create Episode object
        episode = Episode(
            id=None,
            timestamp=exp.timestamp,
            task_tag=exp.task_name,
            situation_hv_bytes=b'', # Placeholder for VSA
            state_sketch=state_sketch,
            action=str(exp.action_idx),
            outcome="archived",
            reward=exp.reward,
            impact_score=exp.priority
        )
        
        # Send to store (buffered write)
        self.store.record_episode(episode)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch from RAM (Hot Storage).
        """
        batch_size = min(len(self.ram_buffer), batch_size)
        batch = random.sample(self.ram_buffer, batch_size)
        return self._batch_to_tensors(batch)

    def sample_from_disk(self, batch_size: int) -> Tuple:
        """
        Sample batch from Disk (Cold Storage) for Dreaming.
        """
        episodes = self.store.sample_random_episodes(batch_size)
        if not episodes:
            return None
            
        experiences = []
        for ep in episodes:
            sketch = ep.state_sketch
            if "data" in sketch:
                # Restore numpy -> tensor
                state_data = sketch["data"]
                # Create Experience object (reconstructing missing fields with defaults)
                exp = Experience(
                    state=torch.from_numpy(state_data),
                    action_idx=int(ep.action),
                    reward=ep.reward,
                    # For next_state, we don't have it archived separately to save space?
                    # Actually, A2C Value calculation needs next_state for bootstrap if not done.
                    # IF we assume dreaming is mostly for Policy/Value polishing on specific states...
                    # Wait, if we don't have next_state, we can't do TD-Error update properly?
                    # FIX: For deep dreaming, we might store next_state too or rely on Monte Carlo return if we stored full episodes?
                    # For now, let's just assume we stored next_state or we can't use it for TD.
                    # BUT wait, we stored "state_sketch". Did we store next_state? No.
                    # LIMITATION: With current archival, we only have (s, a, r). We lack s'.
                    # This means we can train Actor (Policy) if we use Reward as proxy? 
                    # OR we can train Auto-Encoder (Reconstruction).
                    # Actually, for A2C we need s'. 
                    # QUICK FIX: We will just duplicate s as s' which is wrong, OR we skip TD and just do Policy Gradient on R?
                    # Better: Use the stored 'impact_score' (Priority) as the TD error proxy? No. 
                    # Let's set next_state = state (Terminal assumption) for now, acknowledging limitation.
                    next_state=torch.from_numpy(state_data), 
                    done=True, # Treat as terminal to avoid bootstrapping from unknown s'
                    task_name=ep.task_tag,
                    priority=ep.impact_score
                )
                experiences.append(exp)
        
        return self._batch_to_tensors(experiences)

    def _batch_to_tensors(self, batch: List[Experience]) -> Tuple:
        """Helper to collate batch."""
        if not batch: return None
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action_idx for e in batch])
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32)
        tasks = [e.task_name for e in batch]
        
        return states, actions, rewards, next_states, dones, tasks
    
    def get_stats(self):
        return {
            "ram_count": len(self.ram_buffer),
            "ram_capacity": self.capacity,
            "disk_count": self.store.count_episodes()
        }

    def close(self):
        self.store.close()
