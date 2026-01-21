"""
NCGN System 1 Engine - The Physics Engine

Implements the immutable 8-phase tick pipeline:
- Phase 0: Transduction (Input Buffer)
- Phase 1: Passive Decay (The Leak)
- Phase 2: Firing Determination
- Phase 3: Refractory & Reset
- Phase 4: Propagation (Spike Transmission)
- Phase 5: Integration (Update State)
- Phase 6: Lateral Inhibition (k-WTA Attention)
- Phase 7: The Bridge (Surprise Monitor)
- Phase 8: Maintenance

The execution order is STRICTLY SERIALIZED to prevent
"ghost" signals and race conditions.
"""

import math
import heapq
from typing import Dict, Set, List, Optional, Callable, Tuple
from .memory import GraphMemory, ConceptNode


class System1Engine:
    """
    The physics engine that executes the dynamical system.
    
    Time is discretized into ticks. Each tick follows the
    immutable 8-phase execution order.
    """
    
    # Default parameters (biologically grounded)
    DEFAULT_DECAY_ALPHA = 0.9       # Energy decay per tick
    DEFAULT_THRESHOLD = 0.75        # Firing threshold
    DEFAULT_REFRACTORY_PERIOD = 3   # Ticks until re-fire
    DEFAULT_K_WINNERS = 10          # k-WTA attention limit
    DEFAULT_NOVELTY_DECAY = 0.999   # Novelty decay per tick
    DEFAULT_SURPRISE_GATE = 0.45     # PATCHED v5.1: Lowered from 0.5 for higher sensitivity
    
    def __init__(
        self,
        memory: GraphMemory,
        decay_alpha: float = DEFAULT_DECAY_ALPHA,
        k_winners: int = DEFAULT_K_WINNERS,
        refractory_period: int = DEFAULT_REFRACTORY_PERIOD,
        surprise_threshold: float = DEFAULT_SURPRISE_GATE,
        novelty_decay: float = DEFAULT_NOVELTY_DECAY
    ):
        self.memory = memory
        self.decay_alpha = decay_alpha
        self.k_winners = k_winners
        self.refractory_period = refractory_period
        self.surprise_threshold = surprise_threshold
        self.novelty_decay = novelty_decay
        
        # State tracking
        self.current_tick: int = 0
        self.sensory_buffer: Dict[str, float] = {}  # Input buffer for Phase 0
        self.pending_energy: Dict[str, float] = {}  # Buffered propagation
        self.firing_set: Set[str] = set()           # Nodes firing this tick
        
        # Expected state for surprise calculation (captured before tick)
        self.expected_state: Dict[str, float] = {}
        
        # Surprise monitoring
        self.surprise_level: float = 0.0
        self.system2_triggered: bool = False
        self.system2_callback: Optional[Callable[[float], None]] = None
        
        # Flags
        self.paused: bool = False  # Set by System 2 to pause execution
        
    # =====================
    # Input Interface
    # =====================
    
    def inject_energy(self, node_id: str, energy: float):
        """
        Queue external energy to be applied in Phase 0.
        
        This is the ONLY way external energy enters the system.
        """
        if node_id not in self.memory.nodes:
            self.memory.add_node(node_id)
        
        if node_id in self.sensory_buffer:
            self.sensory_buffer[node_id] += energy
        else:
            self.sensory_buffer[node_id] = energy
    
    def set_system2_callback(self, callback: Callable[[float], None]):
        """Set callback for System 2 interrupt."""
        self.system2_callback = callback
    
    # =====================
    # The Tick Pipeline
    # =====================
    
    def tick(self) -> bool:
        """
        Execute one complete tick of the dynamical system.
        
        Returns True if tick completed, False if paused.
        Executes phases in STRICT order 0-8.
        """
        if self.paused:
            return False
        
        # Capture expected state for surprise calculation BEFORE any changes
        self._capture_expected_state()
        
        # THE IMMUTABLE TICK ORDER
        self._phase_0_transduction()
        self._phase_1_decay()
        self._phase_2_firing_determination()
        self._phase_3_refractory_reset()
        self._phase_4_propagation()
        self._phase_5_integration()
        self._phase_6_kwta_inhibition()
        self._phase_7_surprise_monitor()
        self._phase_8_maintenance()
        
        self.current_tick += 1
        return True
    
    def run(self, num_ticks: int) -> int:
        """Run multiple ticks. Returns number of completed ticks."""
        completed = 0
        for _ in range(num_ticks):
            if self.tick():
                completed += 1
            else:
                break  # Paused by System 2
        return completed
    
    # =====================
    # Phase Implementations
    # =====================
    
    def _capture_expected_state(self):
        """Capture current state as the "expected" state for surprise calc."""
        self.expected_state = {
            node_id: node.energy * node.threshold
            for node_id, node in self.memory.nodes.items()
            if node.energy > 0
        }
        # print(f"DEBUG: Captured Expected State: {self.expected_state}")
    
    def _phase_0_transduction(self):
        """
        Phase 0: Apply external energy from Sensory Buffer.
        
        Constraint: This is the ONLY time external energy enters the system.
        """
        for node_id, energy in self.sensory_buffer.items():
            node = self.memory.get_node(node_id)
            if node:
                node.energy = min(1.0, node.energy + energy)
                self.memory.mark_active(node_id)
        
        # Clear the buffer after applying
        self.sensory_buffer.clear()
    
    def _phase_1_decay(self):
        """
        Phase 1: Passive Decay (The Leak).
        
        For ALL active nodes: E_new = E_old × α
        Physics: Without reinforcement, thoughts die.
        """
        to_deactivate = []
        
        for node_id in self.memory.get_active_nodes():
            node = self.memory.get_node(node_id)
            if node:
                node.energy *= self.decay_alpha
                
                # Deactivate if energy is negligible
                if node.energy < 0.001:
                    node.energy = 0.0
                    to_deactivate.append(node_id)
        
        for node_id in to_deactivate:
            self.memory.mark_inactive(node_id)
    
    def _phase_2_firing_determination(self):
        """
        Phase 2: Identify nodes that will fire this tick.
        
        Criteria: E > Threshold AND Refractory_Timer == 0
        CRUCIAL: Do not update targets yet - capture state first.
        """
        self.firing_set.clear()
        
        for node_id in self.memory.get_active_nodes():
            node = self.memory.get_node(node_id)
            if node and node.can_fire():
                self.firing_set.add(node_id)
    
    def _phase_3_refractory_reset(self):
        """
        Phase 3: Reset firing nodes and set refractory period.
        
        Energy is "spent" to create the spike.
        """
        for node_id in self.firing_set:
            node = self.memory.get_node(node_id)
            if node:
                node.reset_after_fire(self.current_tick, self.refractory_period)
                self.memory.mark_inactive(node_id)  # Energy is now 0
    
    def _phase_4_propagation(self):
        """
        Phase 4: Calculate spike transmission.
        
        For every synapse from firing nodes:
        - Calculate Input_target = Weight × Spike_Magnitude
        - Accumulate into Pending_Energy buffer
        
        Constraint: We buffer updates to prevent cascades in same tick.
        """
        self.pending_energy.clear()
        spike_magnitude = 1.0  # Standard spike amplitude
        
        for source_id in self.firing_set:
            for synapse in self.memory.get_outgoing(source_id):
                target_id = synapse.target_id
                energy_transfer = synapse.transmit(spike_magnitude)
                
                if target_id in self.pending_energy:
                    self.pending_energy[target_id] += energy_transfer
                else:
                    self.pending_energy[target_id] = energy_transfer
                
                # Update synapse activity
                synapse.last_active = self.current_tick
    
    def _phase_5_integration(self):
        """
        Phase 5: Apply pending energy to target nodes.
        """
        for node_id, energy in self.pending_energy.items():
            node = self.memory.get_node(node_id)
            if node:
                # Only integrate if not in refractory period
                if node.refractory_timer == 0:
                    node.energy = min(1.0, node.energy + energy)
                    if node.energy > 0:
                        self.memory.mark_active(node_id)
    
    def _phase_6_kwta_inhibition(self):
        """
        Phase 6: Lateral Inhibition using k-WTA.
        
        Algorithm: Min-Heap Selection
        - Keep top K most energetic nodes
        - Hard-set all others to E=0
        
        Complexity: O(M log K) where M is active nodes
        """
        active_nodes = self.memory.get_active_nodes()
        
        if len(active_nodes) <= self.k_winners:
            return  # No inhibition needed
        
        # Build min-heap of size K
        # Heap contains (energy, node_id) tuples
        min_heap: List[Tuple[float, str]] = []
        
        for node_id in active_nodes:
            node = self.memory.get_node(node_id)
            if not node:
                continue
            
            if len(min_heap) < self.k_winners:
                heapq.heappush(min_heap, (node.energy, node_id))
            elif node.energy > min_heap[0][0]:
                heapq.heapreplace(min_heap, (node.energy, node_id))
        
        # Build winner set
        winners = {node_id for _, node_id in min_heap}
        
        # Suppress losers
        to_deactivate = []
        for node_id in active_nodes:
            if node_id not in winners:
                node = self.memory.get_node(node_id)
                if node:
                    node.energy = 0.0
                    to_deactivate.append(node_id)
        
        for node_id in to_deactivate:
            self.memory.mark_inactive(node_id)
    
    def _phase_7_surprise_monitor(self):
        """
        Phase 7: The Bridge - Calculate surprise and trigger System 2.
        
        Surprise = penalty for violated high-confidence expectations.
        
        PATCHED v5.1: Uses RMS (sqrt) for proper Euclidean distance.
        
        Formula:
        S = sqrt(Σ ((E_pred(n) - E_obs(n)) × Confidence(n))²)
        
        Logic:
        - Missing expectations cause high surprise
        - Novel (low confidence) inputs don't cause surprise
        """
        self.surprise_level = self._calculate_surprise()
        
        if self.surprise_level > self.surprise_threshold:
            self.system2_triggered = True
            if self.system2_callback:
                self.system2_callback(self.surprise_level)
        else:
            self.system2_triggered = False
    
    def _calculate_surprise(self) -> float:
        """
        Calculate set-based surprise.
        
        PATCHED v5.1: Uses RMS (sqrt) for proper Euclidean distance.
        
        Penalizes:
        - Expected nodes that are missing
        - Based on confidence of the violated expectation
        """
        sum_of_squares = 0.0
        
        # Current observed state
        observed_state = {
            node_id: node.energy
            for node_id, node in self.memory.nodes.items()
            if node.energy > 0
        }
        
        # Check each expected node
        for node_id, expected_energy in self.expected_state.items():
            observed_energy = observed_state.get(node_id, 0.0)
            
            # Get confidence from incoming synapses
            confidence = self._get_node_confidence(node_id)
            
            # Difference weighted by confidence
            diff = expected_energy - observed_energy
            if diff > 0:  # Only penalize missing expectations
                surprise_contrib = (diff * confidence) ** 2
                sum_of_squares += surprise_contrib
        
        # CRITICAL FIX v5.1: Apply sqrt for Root Mean Square (Euclidean distance)
        total_surprise = math.sqrt(sum_of_squares)
        
        return min(1.0, total_surprise)
    
    def _get_node_confidence(self, node_id: str) -> float:
        """Get average confidence of synapses pointing to this node."""
        incoming = self.memory.get_incoming(node_id)
        if not incoming:
            return 0.0  # Novel node - no confidence
        
        total = sum(synapse.confidence for _, synapse in incoming)
        return total / len(incoming)
    
    def _phase_8_maintenance(self):
        """
        Phase 8: Maintenance tasks.
        
        - Decrement refractory timers
        - Decay novelty scores
        """
        # Decrement refractory timers
        for node in self.memory.nodes.values():
            if node.refractory_timer > 0:
                node.refractory_timer -= 1
            
            # Decay novelty
            node.novelty_score *= self.novelty_decay
    
    # =====================
    # System 2 Interface
    # =====================
    
    def pause(self):
        """Pause the tick loop (called by System 2)."""
        self.paused = True
    
    def resume(self):
        """Resume the tick loop."""
        self.paused = False
        self.system2_triggered = False
    
    def get_firing_set(self) -> Set[str]:
        """Get the set of nodes that fired in the last tick."""
        return self.firing_set.copy()
    
    def get_active_energies(self) -> Dict[str, float]:
        """Get current energy levels of active nodes."""
        return {
            node_id: self.memory.get_node(node_id).energy
            for node_id in self.memory.get_active_nodes()
        }
    
    # =====================
    # State Inspection
    # =====================
    
    def get_state_summary(self) -> dict:
        """Get a summary of the current system state."""
        return {
            'tick': self.current_tick,
            'active_nodes': len(self.memory.get_active_nodes()),
            'total_nodes': self.memory.node_count(),
            'surprise': self.surprise_level,
            'system2_triggered': self.system2_triggered,
            'paused': self.paused
        }
