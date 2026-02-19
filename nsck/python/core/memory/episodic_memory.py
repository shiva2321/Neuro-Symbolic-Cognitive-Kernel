"""
NSCK Episodic Memory Module
Experience storage and retrieval using VSA for similarity search.

Implements two-tier architecture: hot tier (in-memory) and warm tier (SQLite)
with LSH indexing for fast approximate nearest-neighbor retrieval.
"""
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.integration.persistence import BrainStore, Episode
import random

# --- Configurable sketch extractors per task ---
_SKETCH_EXTRACTORS: Dict[str, Any] = {}  # task_tag -> callable(state, task_tag) -> dict


def register_sketch_extractor(task_tag: str, extractor):
    """Register a domain-specific sketch extractor.
    
    Args:
        task_tag: Task identifier (e.g., "hvac", "robot")
        extractor: callable(state: dict, task_tag: str) -> dict
    """
    _SKETCH_EXTRACTORS[task_tag] = extractor


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
    
    # Phase 3.1 Upgrade: Context
    emotion: str = "neutral"
    tom_beliefs: Optional[Dict[str, Any]] = None # Snapshot of ToM beliefs
    image: Optional[np.ndarray] = None # [AGI] Added for SNN generative dreaming
    impact_score: float = 0.0  # |reward| + novelty bonus for salient pruning
    
    def to_stored(self) -> Episode:
        """Convert to storage format with compressed state."""
        # Compress state to sketch (task-specific key fields only)
        sketch = self._extract_sketch(self.state, self.task_tag)
        
        # [Phase 3.1] Store emotion/ToM in sketch
        sketch["emotion"] = self.emotion
        if self.tom_beliefs:
            sketch["tom_beliefs"] = self.tom_beliefs
        
        return Episode(
            id=None,
            timestamp=self.timestamp,
            task_tag=self.task_tag,
            situation_hv_bytes=bytes(self.situation_hv.bits) if hasattr(self.situation_hv, 'bits') else b'',
            state_sketch=sketch,
            action=self.action,
            outcome=self.outcome,
            reward=self.reward,
            impact_score=self.impact_score
        )
    
    @staticmethod
    def _extract_sketch(state: Dict, task_tag: str) -> Dict[str, Any]:
        """Extract minimal state representation.
        
        Uses registered extractors first, falls back to built-in
        game-specific extractors, then generic scalar extraction.
        """
        # 1. Check for registered domain extractor
        if task_tag in _SKETCH_EXTRACTORS:
            return _SKETCH_EXTRACTORS[task_tag](state, task_tag)
        
        # 2. Built-in game extractors (legacy support)
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
        
        # 3. Generic: keep all scalar fields (up to max_sketch_fields)
        max_sketch_fields = 10
        sketch = {}
        for k, v in state.items():
            if isinstance(v, (int, float, str, bool)):
                sketch[k] = v
                if len(sketch) >= max_sketch_fields:
                    break
            elif isinstance(v, (list, tuple)) and len(v) <= 4:
                # Keep small list/tuple fields (e.g., coordinates)
                sketch[k] = v
                if len(sketch) >= max_sketch_fields:
                    break
        return sketch


class EpisodicMemory:
    """
    Experience memory with VSA-based retrieval.
    
    Architecture:
    - Recent episodes (hot): Full state, in-memory deque
    - Older episodes (warm): Compressed sketch, indexed in SQLite
    - LSH index for fast similarity search
    
    Automatically uses Rust EpisodicMemoryConcurrent when available for 10-100x speedup.
    NO TRANSFORMERS, NO ATTENTION - uses Hamming distance and LSH.
    """
    
    def __init__(
        self,
        store: Optional[BrainStore] = None,
        recent_capacity: int = 1000,
        total_capacity: int = 10000,
        consolidation_threshold: int = 500,
        use_rust: bool = True
    ):
        """
        Initialize episodic memory.
        
        Args:
            store: Persistence store (SQLite)
            recent_capacity: Max episodes to keep with full state
            total_capacity: Max episodes per task in storage
            consolidation_threshold: Consolidate when recent buffer hits this
            use_rust: Use Rust backend if available (default True)
        """
        self.store = store
        self.recent_capacity = recent_capacity
        self.total_capacity = total_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Try to use Rust backend if available and requested
        self._rust_backend = None
        if use_rust and hypervec_rs.EpisodicMemoryConcurrent is not None:
            try:
                self._rust_backend = hypervec_rs.EpisodicMemoryConcurrent(recent_capacity)
                print("EpisodicMemory Initialized with Rust backend (concurrent, optimized).")
            except Exception as e:
                print(f"[WARNING] Failed to initialize Rust backend: {e}")
                print("Falling back to Python implementation.")
        else:
            print("EpisodicMemory Initialized with Python backend.")
        
        # Recent episodes per task (full state) - Python fallback
        self.recent: Dict[str, deque] = {}
        
        # LSH index for fast retrieval
        # Multi-table LSH: use several independent hash tables for better recall
        self.lsh_bits = 10   # Bits per hash (~1024 buckets per table)
        self.lsh_num_tables = 4  # Number of independent hash tables
        self.lsh_seeds = list(range(42, 42 + self.lsh_num_tables))  # One seed per table
        # task -> table_idx -> {lsh_hash -> [episode_ids]}
        self.lsh_index: Dict[str, List[Dict[int, List[int]]]] = {}
    
    def reset(self):
        """Clear all episodic memories and re-initialize."""
        self.recent = {}
        self.lsh_index = {}
        if self.store:
            # We don't delete the DB here, KnowledgeIntegration will handle file deletion,
            # but we should ensure the store's in-memory state is cleared if any.
            pass
        print("[EPISODIC] Memory reset complete.")
    
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
            self.lsh_index[task] = [{} for _ in range(self.lsh_num_tables)]
            self._lsh_inserts_since_rebuild: Dict[str, int] = getattr(self, "_lsh_inserts_since_rebuild", {})
            self._lsh_inserts_since_rebuild[task] = 0
        
        # Track whether the deque is full (next append will evict the oldest entry).
        deque_was_full = len(self.recent[task]) == self.recent_capacity
        
        # Add to recent buffer (Python primary store)
        self.recent[task].append(episode)
        # Use timestamp as stable LSH key — deque integer indices go stale
        # once the deque reaches maxlen and begins evicting front elements.
        ep_ts = episode.timestamp
        
        if not hasattr(self, "_lsh_inserts_since_rebuild"):
            self._lsh_inserts_since_rebuild: Dict[str, int] = {}
        self._lsh_inserts_since_rebuild[task] = self._lsh_inserts_since_rebuild.get(task, 0) + 1

        # Mirror to Rust backend for fast parallel kNN when compiled.
        # Rust holds a copy for rayon-accelerated search; Python self.recent is authoritative.
        if self._rust_backend is not None and hypervec_rs.Episode is not None:
            try:
                rust_ep = hypervec_rs.Episode(
                    episode.timestamp,
                    episode.task_tag,
                    episode.situation_hv,
                    episode.action,
                    episode.outcome,
                    episode.reward,
                    episode.impact_score,
                )
                self._rust_backend.add_episode(rust_ep)
            except Exception:
                pass  # Silently fall back if HV type is incompatible (Python fallback HV).

        # Update LSH index (insert into all tables)
        if hasattr(episode.situation_hv, 'lsh_hash'):
            for t, seed in enumerate(self.lsh_seeds):
                lsh = episode.situation_hv.lsh_hash(seed, self.lsh_bits)
                if lsh not in self.lsh_index[task][t]:
                    self.lsh_index[task][t][lsh] = []
                self.lsh_index[task][t][lsh].append(ep_ts)
        
        # Stale-index fix: when the deque was at capacity an eviction just occurred.
        # After every recent_capacity/10 evictions, rebuild the LSH from current
        # deque contents so stale timestamps don't degrade recall.
        rebuild_interval = max(1, self.recent_capacity // 10)
        if deque_was_full and self._lsh_inserts_since_rebuild.get(task, 0) >= rebuild_interval:
            self._rebuild_lsh(task)
            self._lsh_inserts_since_rebuild[task] = 0
        
        # Check if consolidation needed
        if len(self.recent[task]) >= self.consolidation_threshold:
            self._consolidate(task)
    
    def _rebuild_lsh(self, task_tag: str):
        """Rebuild LSH index from scratch using only episodes currently in the deque."""
        self.lsh_index[task_tag] = [{} for _ in range(self.lsh_num_tables)]
        for ep in self.recent.get(task_tag, deque()):
            if hasattr(ep.situation_hv, 'lsh_hash'):
                for t, seed in enumerate(self.lsh_seeds):
                    lsh = ep.situation_hv.lsh_hash(seed, self.lsh_bits)
                    if lsh not in self.lsh_index[task_tag][t]:
                        self.lsh_index[task_tag][t][lsh] = []
                    self.lsh_index[task_tag][t][lsh].append(ep.timestamp)

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
        
        # Rebuild LSH index for remaining recent episodes (use timestamp as key)
        self.lsh_index[task_tag] = [{} for _ in range(self.lsh_num_tables)]
        for ep in recent:
            if hasattr(ep.situation_hv, 'lsh_hash'):
                for t, seed in enumerate(self.lsh_seeds):
                    lsh = ep.situation_hv.lsh_hash(seed, self.lsh_bits)
                    if lsh not in self.lsh_index[task_tag][t]:
                        self.lsh_index[task_tag][t][lsh] = []
                    self.lsh_index[task_tag][t][lsh].append(ep.timestamp)
        
        print(f"[MEMORY] Consolidated {len(to_store)} episodes for {task_tag}")
    
    def sample(self, task_tag: str, n: int = 32) -> List[LiveEpisode]:
        """
        Get random episodes for dreaming/replay.
        
        Args:
            task_tag: Task to sample from
            n: Number of samples, defaults to 32
            
        Returns:
            List of random episodes
        """
        recent = self.recent.get(task_tag, deque())
        if not recent:
            return []
            
        candidates = list(recent)
        k = min(n, len(candidates))
        return random.sample(candidates, k)

    def recall_similar(
        self,
        query_hv: hypervec_rs.HyperVector,
        task_tag: str,
        k: int = 5
    ) -> List[LiveEpisode]:
        """
        Find episodes most similar to query.
        
        Uses Rust parallel KNN search when available (10-100x faster), falls back to Python LSH.
        
        Args:
            query_hv: Query hypervector
            task_tag: Which task's memory to search
            k: Number of episodes to return
            
        Returns:
            List of similar episodes, sorted by similarity
        """
        # Try Rust backend first (parallel rayon kNN — 10-100x faster than Python LSH).
        if self._rust_backend is not None and hasattr(self._rust_backend, 'parallel_knn_search'):
            try:
                # Returns [(idx, similarity, Episode), ...] sorted by similarity desc.
                rust_results = self._rust_backend.parallel_knn_search(
                    query_hv, k, task_tag
                )
                if rust_results:
                    # Build timestamp → LiveEpisode map so we return the Python object
                    # (with full state, emotion, ToM, etc.) when still in recent.
                    recent_list = list(self.recent.get(task_tag, deque()))
                    ts_map: Dict[float, LiveEpisode] = {
                        ep.timestamp: ep for ep in recent_list
                    }
                    matched: List[LiveEpisode] = []
                    for (_idx, _sim, rust_ep) in rust_results:
                        ts = rust_ep.timestamp
                        if ts in ts_map:
                            matched.append(ts_map[ts])
                        else:
                            # Evicted from Python recent — reconstruct minimal LiveEpisode.
                            matched.append(LiveEpisode(
                                timestamp=ts,
                                task_tag=rust_ep.task_tag,
                                situation_hv=rust_ep.get_situation_hv(),
                                state={},
                                action=rust_ep.action,
                                outcome=rust_ep.outcome,
                                reward=rust_ep.reward,
                                impact_score=rust_ep.impact_score,
                            ))
                    return matched
            except Exception:
                pass  # Fall through to Python LSH on any error.
        
        # Python fallback with LSH
        recent = self.recent.get(task_tag, deque())
        
        if not recent:
            return []
        
        # 1. Get LSH hash for query
        candidates = set()
        
        if hasattr(query_hv, 'lsh_hash'):
            tables = self.lsh_index.get(task_tag, [])
            for t, seed in enumerate(self.lsh_seeds):
                if t >= len(tables):
                    break
                query_lsh = query_hv.lsh_hash(seed, self.lsh_bits)
                table = tables[t]
                
                # Exact bucket match
                if query_lsh in table:
                    candidates.update(table[query_lsh])
                
                # 1-bit flips for neighboring buckets
                for bit in range(self.lsh_bits):
                    nearby = query_lsh ^ (1 << bit)
                    if nearby in table:
                        candidates.update(table[nearby])
        
        # 2. If not enough candidates, fall back to recent tail
        recent_list = list(recent)
        if len(candidates) < k * 2:
            tail = recent_list[max(0, len(recent_list) - k * 3):]
            candidates.update(ep.timestamp for ep in tail)

        # 3. Build timestamp → episode map for O(1) lookup
        ts_map = {ep.timestamp: ep for ep in recent_list}

        # 4. Compute exact similarity for candidate timestamps
        scored = []
        for ts in candidates:
            ep = ts_map.get(ts)
            if ep is not None:
                sim = query_hv.similarity(ep.situation_hv)
                scored.append((ep, sim))
        
        # 5. Sort by similarity and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:k]]
    
    def recall_recent(self, task_tag: str, n: int = 10) -> List[LiveEpisode]:
        """Get most recent episodes."""
        recent = self.recent.get(task_tag, deque())
        return list(recent)[-n:]

    def retrieve_salient(self, task_tag: str, limit: int = 100) -> List[LiveEpisode]:
        """Retrieve high-impact episodes (high absolute reward or novelty)."""
        recent = list(self.recent.get(task_tag, deque()))
        # Sort by absolute reward + novelty (simplified)
        recent.sort(key=lambda ep: abs(ep.reward) + ep.impact_score, reverse=True)
        return recent[:limit]
    
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
            "lsh_buckets": sum(len(t) for t in self.lsh_index.get(task_tag, [])),
            "lsh_tables": self.lsh_num_tables,
            "lsh_bits": self.lsh_bits,
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
