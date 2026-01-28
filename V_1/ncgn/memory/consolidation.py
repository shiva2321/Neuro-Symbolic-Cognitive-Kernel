import torch
from typing import List, Tuple, Dict, Any, Optional
import time

class ShortTermMemory:
    """
    Episodic Buffer (Hippocampus-like).
    Stores raw experiences (Hypervectors + Metadata) for later replay.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        # Buffer stores: (Hypervector, Label, Intensity/Energy)
        self.buffer: List[Tuple[torch.Tensor, str, float]] = []
        
    def add(self, vector: torch.Tensor, label: str, energy: float):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0) # FIFO (forget oldest)
        self.buffer.append((vector, label, energy))
        
    def clear(self):
        self.buffer.clear()
        
    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

class SleepManager:
    """
    Sleep Consolidation System.
    
    Orchestrates the transfer from ShortTermMemory (Buffer) to LongTermMemory (Graph & VSA).
    """
    def __init__(self, stm: ShortTermMemory, brain: Any):
        self.stm = stm
        self.brain = brain
        self.sleep_threshold = 0.8 # Capacity fraction to trigger auto-sleep suggestion
        
    def run_sleep_cycle(self) -> Dict[str, int]:
        """
        Execute Sleep Consolidation.
        
        Replays the buffer and consolidates memories.
        """
        stats = {"consolidated": 0, "strengthened": 0, "pruned": 0}
        
        print("\n[SleepManager] Starting Sleep Cycle...")
        
        # 1. Replay Buffer
        # In a full model, this might involve generative replay using the SNN.
        # Here we do direct algebraic replay.
        
        # Group by label to simulate "frequency" effect
        # Frequent concepts in the day get prioritized
        memory_trace: Dict[str, List[float]] = {}
        vectors: Dict[str, torch.Tensor] = {}
        
        for vec, label, energy in self.stm.buffer:
            if label not in memory_trace:
                memory_trace[label] = []
                vectors[label] = vec
            memory_trace[label].append(energy)
            
        # 2. Consolidation Logic
        for label, energies in memory_trace.items():
            avg_energy = sum(energies) / len(energies)
            freq = len(energies)
            
            # Resonance Check: Does it exist in LTM?
            exists_in_vsa = (label in self.brain.associative.codebook)
            exists_in_graph = self.brain.has_concept(label)
            
            if exists_in_graph and exists_in_vsa:
                # Strengthening (LTP)
                # Inject energy to "fire" it and reinforce connected edges via Hebbian
                self.brain.inject(label, avg_energy)
                stats["strengthened"] += 1
            else:
                # Novelty Processing
                # Only consolidate if "significant" (freq > 1 or High Energy)
                if freq > 1 or avg_energy > 0.5:
                    if not exists_in_vsa:
                         self.brain.VSA.add_concept(label, vectors[label]) # Direct access hack? or expose method
                         # Wait, brain.associative is the wrapper
                         self.brain.associative.add_concept(label, vectors[label])
                    
                    if not exists_in_graph:
                        self.brain.add_concept(label, initial_energy=0.2)
                        
                    stats["consolidated"] += 1
                else:
                    stats["pruned"] += 1
                    
        # 3. Clear Buffer
        self.stm.clear()
        
        # 4. Trigger one "Dream" tick (Propagate) to finalize Hebbian updates
        self.brain.think(steps=5)
        
        print(f"[SleepManager] Sleep Complete. Stats: {stats}")
        return stats
