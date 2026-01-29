"""
NSCK Episodic Memory Module
Experience storage and retrieval using VSA for similarity search.

NO ATTENTION MECHANISMS - uses LSH and Hamming similarity only.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import hypervec_rs
from .persistence import BrainStore, Episode


@dataclass
class LiveEpisode:
    """In-memory episode before compression."""
    timestamp: float
    task_tag: str
    situation_hv: hypervec_rs.HyperVector
    state: Dict[str, Any]  # Full state (only recent episodes keep this)
    action: str
    outcome: str
    reward: float
    
    def to_stored(self) -> Episode:
        """Convert to storage format with compressed state."""
        # Compress state to sketch (task-specific key fields only)
        sketch = self._extract_sketch(self.state, self.task_tag)
        
        return Episode(
            id=None,
            timestamp=self.timestamp,
            task_tag=self.task_tag,
            situation_hv_bytes=bytes(self.situation_hv.bits) if hasattr(self.situation_hv, 'bits') else b'',
            state_sketch=sketch,
            action=self.action,
            outcome=self.outcome,
            reward=self.reward
        )
    
    def _extract_sketch(self, state: Dict, task_tag: str) -> Dict[str, Any]:
        """Extract minimal state representation."""
        if task_tag == "snake":
            return {
                "head": state.get("head"),
                "food": state.get("food"),
                "body_len": len(state.get("body", [])),
            }
        elif task_tag == "pong":
            return {
                "ball_y": state.get("ball_y"),
                "ball_dy": state.get("ball_dy"),
                "p1_y": state.get("p1_y"),
            }
        else:
            # Generic: keep up to 5 scalar fields
            sketch = {}
            for k, v in state.items():
                if isinstance(v, (int, float, str, bool)):
                    sketch[k] = v
                    if len(sketch) >= 5:
                        break
            return sketch


class EpisodicMemory:
    """
    Experience memory with VSA-based retrieval.
    
    Architecture:
    - Recent episodes (hot): Full state, in-memory deque
    - Older episodes (warm): Compressed sketch, indexed in SQLite
    - LSH index for fast similarity search
    
    NO TRANSFORMERS, NO ATTENTION - uses Hamming distance and LSH.
    """
    
    def __init__(
        self,
        store: Optional[BrainStore] = None,
        recent_capacity: int = 1000,
        total_capacity: int = 10000,
        consolidation_threshold: int = 500
    ):
        """
        Initialize episodic memory.
        
        Args:
            store: Persistence store (SQLite)
            recent_capacity: Max episodes to keep with full state
            total_capacity: Max episodes per task in storage
            consolidation_threshold: Consolidate when recent buffer hits this
        """
        self.store = store
        self.recent_capacity = recent_capacity
        self.total_capacity = total_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Recent episodes per task (full state)
        self.recent: Dict[str, deque] = {}
        
        # LSH index for fast retrieval: task -> {lsh_hash -> [episode_ids]}
        self.lsh_index: Dict[str, Dict[int, List[int]]] = {}
        self.lsh_bits = 64  # Number of hash bits
    
    def record(self, episode: LiveEpisode):
        """
        Record a new episode.
        
        Args:
            episode: Live episode with full state
        """
        task = episode.task_tag
        
        # Initialize task buffers if needed
        if task not in self.recent:
            self.recent[task] = deque(maxlen=self.recent_capacity)
            self.lsh_index[task] = {}
        
        # Add to recent buffer
        self.recent[task].append(episode)
        
        # Update LSH index
        if hasattr(episode.situation_hv, 'lsh_hash'):
            lsh = episode.situation_hv.lsh_hash(42, self.lsh_bits)
            if lsh not in self.lsh_index[task]:
                self.lsh_index[task][lsh] = []
            self.lsh_index[task][lsh].append(len(self.recent[task]) - 1)
        
        # Check if consolidation needed
        if len(self.recent[task]) >= self.consolidation_threshold:
            self._consolidate(task)
    
    def _consolidate(self, task_tag: str):
        """Compress old episodes and move to storage."""
        if not self.store:
            return
        
        recent = self.recent.get(task_tag)
        if not recent:
            return
        
        # Keep most recent half in memory, move rest to storage
        keep_count = len(recent) // 2
        to_store = []
        
        while len(recent) > keep_count:
            ep = recent.popleft()
            to_store.append(ep.to_stored())
        
        # Batch write to storage
        for stored_ep in to_store:
            self.store.record_episode(stored_ep)
        
        # Prune storage if over capacity
        self.store.flush_episodes()
        self.store.prune_old_episodes(task_tag, self.total_capacity)
        
        # Rebuild LSH index for remaining recent episodes
        self.lsh_index[task_tag] = {}
        for i, ep in enumerate(recent):
            if hasattr(ep.situation_hv, 'lsh_hash'):
                lsh = ep.situation_hv.lsh_hash(42, self.lsh_bits)
                if lsh not in self.lsh_index[task_tag]:
                    self.lsh_index[task_tag][lsh] = []
                self.lsh_index[task_tag][lsh].append(i)
        
        print(f"[MEMORY] Consolidated {len(to_store)} episodes for {task_tag}")
    
    def recall_similar(
        self,
        query_hv: hypervec_rs.HyperVector,
        task_tag: str,
        k: int = 5
    ) -> List[LiveEpisode]:
        """
        Find episodes most similar to query.
        
        Uses LSH for candidate selection, then exact similarity ranking.
        
        Args:
            query_hv: Query hypervector
            task_tag: Which task's memory to search
            k: Number of episodes to return
            
        Returns:
            List of similar episodes, sorted by similarity
        """
        recent = self.recent.get(task_tag, deque())
        
        if not recent:
            return []
        
        # 1. Get LSH hash for query
        candidates = set()
        
        if hasattr(query_hv, 'lsh_hash'):
            query_lsh = query_hv.lsh_hash(42, self.lsh_bits)
            
            # Find episodes with same LSH hash
            lsh_idx = self.lsh_index.get(task_tag, {})
            if query_lsh in lsh_idx:
                candidates.update(lsh_idx[query_lsh])
            
            # Also check nearby hashes (1-bit flips)
            for bit in range(min(8, self.lsh_bits)):
                nearby = query_lsh ^ (1 << bit)
                if nearby in lsh_idx:
                    candidates.update(lsh_idx[nearby])
        
        # 2. If not enough candidates, add recent episodes
        if len(candidates) < k * 2:
            candidates.update(range(max(0, len(recent) - k * 3), len(recent)))
        
        # 3. Compute exact similarity for candidates
        scored = []
        for idx in candidates:
            if 0 <= idx < len(recent):
                ep = recent[idx]
                sim = query_hv.similarity(ep.situation_hv)
                scored.append((ep, sim))
        
        # 4. Sort by similarity and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:k]]
    
    def recall_recent(self, task_tag: str, n: int = 10) -> List[LiveEpisode]:
        """Get most recent episodes."""
        recent = self.recent.get(task_tag, deque())
        return list(recent)[-n:]
    
    def recall_by_outcome(
        self,
        task_tag: str,
        outcome: str = "success",
        n: int = 10
    ) -> List[LiveEpisode]:
        """Get episodes with specific outcome."""
        recent = self.recent.get(task_tag, deque())
        matches = [ep for ep in recent if ep.outcome == outcome]
        return matches[-n:]
    
    def recall_by_reward(
        self,
        task_tag: str,
        min_reward: float = 0.0,
        n: int = 10
    ) -> List[LiveEpisode]:
        """Get high-reward episodes."""
        recent = self.recent.get(task_tag, deque())
        matches = sorted(
            [ep for ep in recent if ep.reward >= min_reward],
            key=lambda ep: ep.reward,
            reverse=True
        )
        return matches[:n]
    
    def get_statistics(self, task_tag: str) -> Dict[str, Any]:
        """Get memory statistics for a task."""
        recent = self.recent.get(task_tag, deque())
        
        rewards = [ep.reward for ep in recent]
        positive = sum(1 for r in rewards if r > 0)
        negative = sum(1 for r in rewards if r < 0)
        
        return {
            "recent_count": len(recent),
            "stored_count": self.store.count_episodes(task_tag) if self.store else 0,
            "positive_rate": positive / len(rewards) if rewards else 0,
            "negative_rate": negative / len(rewards) if rewards else 0,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "lsh_buckets": len(self.lsh_index.get(task_tag, {})),
        }
    
    def create_situation_hv(
        self,
        state: Dict[str, Any],
        task_tag: str,
        active_predicates: List[str]
    ) -> hypervec_rs.HyperVector:
        """
        Create a situation hypervector from state and predicates.
        
        Args:
            state: Game state dict
            task_tag: Which task
            active_predicates: List of true predicates
            
        Returns:
            Bundled hypervector representing the situation
        """
        # Bundle all active predicate HVs
        if not active_predicates:
            # Create from state hash if no predicates
            return hypervec_rs.HyperVector(hash(str(state)) % (2**32))
        
        # Create HV for each predicate and bundle
        hvs = [hypervec_rs.HyperVector(hash(pred) % (2**32)) for pred in active_predicates]
        
        result = hvs[0]
        for hv in hvs[1:]:
            result = result.bundle(hv)
        
        return result
