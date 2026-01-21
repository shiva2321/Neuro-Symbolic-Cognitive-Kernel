"""
NCGN v6.0 System 1 Engine - Valence-Driven Physics

Implements the physics engine with:
- Softmax Inhibition (replaces k-WTA)
- Homeostatic Normalization (entropy control)
- Dynamic Temperature Annealing
- Eligibility trace updates (Phase 2)
- Global energy monitoring (seizure prevention)

The execution order is strictly serialized to prevent
"ghost" signals and race conditions.
"""

import math
import random
from typing import Dict, Set, List, Optional, Callable, Tuple
from .memory import GraphMemory, ConceptNode, Synapse, ClusterType


class System1Engine:
    """
    The physics engine for NCGN v6.0.
    
    Time is discretized into ticks. Each tick follows an
    8-phase execution order with Softmax inhibition for
    action selection and homeostatic energy regulation.
    """
    
    # Default parameters (biologically grounded)
    DEFAULT_DECAY_ALPHA = 0.9       # Energy decay per tick
    DEFAULT_THRESHOLD = 0.75        # Firing threshold
    DEFAULT_REFRACTORY_PERIOD = 3   # Ticks until re-fire
    DEFAULT_TRACE_DECAY = 0.95      # Eligibility trace decay per tick
    DEFAULT_NOVELTY_DECAY = 0.999   # Novelty decay per tick
    
    # Thermodynamic parameters
    DEFAULT_BASE_TEMPERATURE = 0.5  # Softmax temperature baseline
    DEFAULT_ENERGY_CAP = 10.0       # Maximum cluster energy (entropy control)
    DEFAULT_GLOBAL_ENERGY_THRESHOLD = 50.0  # Seizure threshold
    DEFAULT_SEIZURE_DAMPING = 0.5   # Damping factor on seizure
    
    def __init__(
        self,
        memory: GraphMemory,
        decay_alpha: float = DEFAULT_DECAY_ALPHA,
        refractory_period: int = DEFAULT_REFRACTORY_PERIOD,
        trace_decay: float = DEFAULT_TRACE_DECAY,
        novelty_decay: float = DEFAULT_NOVELTY_DECAY,
        base_temperature: float = DEFAULT_BASE_TEMPERATURE,
        energy_cap: float = DEFAULT_ENERGY_CAP,
        k_winners: int = None,  # v5 compatibility
        surprise_threshold: float = None  # v5 compatibility
    ):
        self.memory = memory
        self.decay_alpha = decay_alpha
        self.refractory_period = refractory_period
        self.trace_decay = trace_decay
        self.novelty_decay = novelty_decay
        self.base_temperature = base_temperature
        self.energy_cap = energy_cap
        
        # v5 compatibility state
        self.system2_triggered = False
        self.surprise_level = 0.0
        
        # State tracking
        self.current_tick: int = 0
        self.sensory_buffer: Dict[str, float] = {}  # Input buffer for Phase 0
        self.pending_energy: Dict[str, float] = {}  # Buffered propagation
        self.firing_set: Set[str] = set()           # Nodes firing this tick
        
        # Dynamic temperature (affected by global energy)
        self.temperature: float = base_temperature
        
        # Surprise monitoring (for System 2 integration)
        self.surprise_level: float = 0.0
        self.expected_state: Dict[str, float] = {}
        
        # Callbacks
        self.on_surprise: Optional[Callable[[float], None]] = None
        
        # Flags
        self.paused: bool = False
    
    # =====================
    # Input Interface
    # =====================
    
    def inject_energy(self, node_id: str, energy: float, cluster: ClusterType = ClusterType.SENSORY):
        """
        Queue external energy to be applied in Phase 0.
        
        This is the ONLY way external energy enters the system.
        """
        if node_id not in self.memory.nodes:
            self.memory.add_node(node_id, cluster=cluster)
        
        if node_id in self.sensory_buffer:
            self.sensory_buffer[node_id] += energy
        else:
            self.sensory_buffer[node_id] = energy
    
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
        
        # Capture expected state for surprise calculation
        self._capture_expected_state()
        
        # THE IMMUTABLE TICK ORDER
        self._phase_0_transduction()
        self._phase_1_decay()
        self._phase_2_trace_update()          # NEW: Update eligibility traces
        self._phase_3_firing_determination()
        self._phase_4_refractory_reset()
        self._phase_5_propagation()
        self._phase_6_integration()
        self._phase_7_softmax_inhibition()    # CHANGED: Softmax instead of k-WTA
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
                break
        return completed
    
    # =====================
    # Phase Implementations
    # =====================
    
    def _capture_expected_state(self):
        """Capture current state as the 'expected' state for surprise calc."""
        self.expected_state = {
            node_id: node.energy
            for node_id, node in self.memory.nodes.items()
            if node.energy > 0
        }
    
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
                
                if node.energy < 0.001:
                    node.energy = 0.0
                    to_deactivate.append(node_id)
        
        for node_id in to_deactivate:
            self.memory.mark_inactive(node_id)
    
    def _phase_2_trace_update(self):
        """
        Phase 2: Update eligibility traces based on Hebbian coincidence.
        
        NEW in v6.0: For every synapse where Pre and Post are both active,
        set the eligibility trace to 1.0. This flags the synapse for
        potential learning when reward arrives.
        """
        active_nodes = self.memory.get_active_nodes()
        
        for post_id in active_nodes:
            post_node = self.memory.get_node(post_id)
            if not post_node or post_node.energy < 0.1:
                continue
            
            # Check incoming synapses for Pre/Post coincidence
            for source_id, synapse in self.memory.get_incoming(post_id):
                if source_id in active_nodes:
                    source_node = self.memory.get_node(source_id)
                    if source_node and source_node.energy > 0.1:
                        # Hebbian coincidence: both Pre and Post active
                        synapse.set_eligible()
                        self.memory.mark_synapse_traced(source_id, post_id)
    
    def _phase_3_firing_determination(self):
        """
        Phase 3: Identify nodes that will fire this tick.
        
        Criteria: E > Threshold AND Refractory_Timer == 0
        """
        self.firing_set.clear()
        
        for node_id in self.memory.get_active_nodes():
            node = self.memory.get_node(node_id)
            if node and node.can_fire():
                self.firing_set.add(node_id)
    
    def _phase_4_refractory_reset(self):
        """
        Phase 4: Reset firing nodes and set refractory period.
        
        Energy is "spent" to create the spike.
        """
        for node_id in self.firing_set:
            node = self.memory.get_node(node_id)
            if node:
                node.reset_after_fire(self.current_tick, self.refractory_period)
                self.memory.mark_inactive(node_id)
    
    def _phase_5_propagation(self):
        """
        Phase 5: Calculate spike transmission.
        
        For every synapse from firing nodes:
        - Calculate Input_target = Weight × Spike_Magnitude
        - Accumulate into Pending_Energy buffer
        
        Constraint: We buffer updates to prevent cascades in same tick.
        """
        self.pending_energy.clear()
        spike_magnitude = 1.0
        
        for source_id in self.firing_set:
            for synapse in self.memory.get_outgoing(source_id):
                target_id = synapse.target_id
                energy_transfer = synapse.transmit(spike_magnitude)
                
                if target_id in self.pending_energy:
                    self.pending_energy[target_id] += energy_transfer
                else:
                    self.pending_energy[target_id] = energy_transfer
                
                synapse.last_active = self.current_tick
    
    def _phase_6_integration(self):
        """
        Phase 6: Apply pending energy to target nodes.
        """
        for node_id, energy in self.pending_energy.items():
            node = self.memory.get_node(node_id)
            if node:
                if node.refractory_timer == 0:
                    node.energy = min(1.0, node.energy + energy)
                    if node.energy > 0:
                        self.memory.mark_active(node_id)
    
    def _phase_7_softmax_inhibition(self):
        """
        Phase 7: Softmax Inhibition (replaces k-WTA).
        
        CHANGED in v6.0: Instead of hard winner selection, apply
        probabilistic Softmax inhibition to motor clusters.
        
        Algorithm:
        1. Update dynamic temperature based on global energy
        2. For each cluster (especially MOTOR):
           a. Apply Homeostatic Normalization (divisive)
           b. Compute Softmax probabilities
           c. Sample one winner (for MOTOR) or scale energies (for HIDDEN)
        
        Entropy Explosion Prevention:
        - Divisive normalization bounds cluster energy
        - Temperature feedback loop prevents runaway activation
        - Gating threshold excludes near-zero nodes from Softmax
        """
        # 1. Update temperature based on global energy
        self._update_temperature()
        
        # 2. Check for seizure condition (global energy too high)
        total_energy = self.memory.update_total_energy()
        if total_energy > self.DEFAULT_GLOBAL_ENERGY_THRESHOLD:
            self._apply_seizure_damping()
            return
        
        # 3. Apply Softmax to MOTOR cluster (action selection)
        self._apply_cluster_softmax(ClusterType.MOTOR, sample=True)
        
        # 4. Apply graded inhibition to HIDDEN cluster (competition)
        self._apply_cluster_softmax(ClusterType.HIDDEN, sample=False)
    
    def _update_temperature(self):
        """
        Dynamic Temperature Annealing.
        
        Temperature responds to global energy:
        - High energy → Low temperature (decisive, exploitation)
        - Low energy → High temperature (exploratory, creativity)
        
        Formula: τ = τ_base + α × (E_target - E_total)
        """
        total_energy = self.memory.get_total_energy()
        target_energy = 5.0  # Homeostatic target
        
        # Temperature increases when system is quiet (exploration)
        # Temperature decreases when system is loud (exploitation)
        alpha = 0.1
        self.temperature = self.base_temperature + alpha * (target_energy - total_energy)
        
        # Clamp to reasonable range
        self.temperature = max(0.1, min(2.0, self.temperature))
    
    def _apply_cluster_softmax(self, cluster: ClusterType, sample: bool = True):
        """
        Apply Softmax inhibition to a specific cluster.
        
        Args:
            cluster: Which cluster to process
            sample: If True, sample one winner (for MOTOR).
                    If False, scale energies by probability (for HIDDEN).
        
        Entropy Explosion Prevention (explicit comment as required):
        - GATING: Nodes with E < 0.05 are excluded from Softmax
        - HOMEOSTATIC NORMALIZATION: If cluster sum > E_cap, scale down
        - These mechanisms prevent irrelevant nodes from accumulating
          probability mass and causing entropy explosion.
        """
        cluster_nodes = self.memory.get_cluster_nodes(cluster)
        if not cluster_nodes:
            return
        
        # Gather active energies with GATING (exclude near-zero)
        active_energies: Dict[str, float] = {}
        for node_id in cluster_nodes:
            node = self.memory.get_node(node_id)
            if node and node.energy > 0.05:  # GATING THRESHOLD
                active_energies[node_id] = node.energy
        
        if not active_energies:
            return
        
        # HOMEOSTATIC NORMALIZATION
        # If cluster sum exceeds metabolic cap, scale down proportionally
        # This prevents entropy explosion by bounding total activation
        cluster_sum = sum(active_energies.values())
        if cluster_sum > self.energy_cap:
            scale = self.energy_cap / cluster_sum
            active_energies = {k: v * scale for k, v in active_energies.items()}
            # Apply scaling to actual nodes
            for node_id, scaled_e in active_energies.items():
                node = self.memory.get_node(node_id)
                if node:
                    node.energy = scaled_e
        
        # Compute Softmax
        # Subtract max for numerical stability
        max_e = max(active_energies.values())
        exp_values: Dict[str, float] = {}
        
        for node_id, energy in active_energies.items():
            exp_values[node_id] = math.exp((energy - max_e) / self.temperature)
        
        exp_sum = sum(exp_values.values())
        if exp_sum == 0:
            return
        
        probabilities = {k: v / exp_sum for k, v in exp_values.items()}
        
        if sample:
            # Sample one winner (for MOTOR cluster - action selection)
            winner = self._sample_from_distribution(probabilities)
            
            # Suppress all non-winners
            for node_id in cluster_nodes:
                node = self.memory.get_node(node_id)
                if node:
                    if node_id == winner:
                        node.energy = 1.0  # Winner takes all
                    else:
                        node.energy = 0.0
                        self.memory.mark_inactive(node_id)
        else:
            # Scale energies by probability (graded competition)
            for node_id in cluster_nodes:
                node = self.memory.get_node(node_id)
                if node:
                    prob = probabilities.get(node_id, 0.0)
                    node.energy *= prob
                    if node.energy < 0.001:
                        node.energy = 0.0
                        self.memory.mark_inactive(node_id)
    
    def _sample_from_distribution(self, probabilities: Dict[str, float]) -> Optional[str]:
        """Sample a node ID from probability distribution."""
        if not probabilities:
            return None
        
        r = random.random()
        cumulative = 0.0
        
        for node_id, prob in probabilities.items():
            cumulative += prob
            if r <= cumulative:
                return node_id
        
        return list(probabilities.keys())[-1]
    
    def _apply_seizure_damping(self):
        """
        Global damping when total energy exceeds threshold.
        
        Seizure Prevention (explicit comment as required):
        This is the "emergency brake" for entropy explosion.
        When global energy exceeds the critical threshold, we
        apply massive damping to ALL nodes, effectively resetting
        the system to a quiet state. This mimics the post-ictal
        refractory period after a biological seizure.
        """
        for node_id in self.memory.get_active_nodes():
            node = self.memory.get_node(node_id)
            if node:
                node.energy *= self.DEFAULT_SEIZURE_DAMPING
                if node.energy < 0.01:
                    node.energy = 0.0
                    self.memory.mark_inactive(node_id)
    
    def _phase_8_maintenance(self):
        """
        Phase 8: Maintenance tasks.
        
        - Decrement refractory timers
        - Decay novelty scores
        - Decay eligibility traces
        - Update energy tracking
        """
        # Decrement refractory timers
        for node in self.memory.nodes.values():
            if node.refractory_timer > 0:
                node.refractory_timer -= 1
            
            # Decay novelty
            node.novelty_score *= self.novelty_decay
        
        # Decay all eligibility traces
        self.memory.decay_all_traces(self.trace_decay)
        
        # Update energy tracking
        self.memory.update_total_energy()
    
    # =====================
    # Action Selection API
    # =====================
    
    def select_action(self, motor_node_ids: List[str]) -> Optional[str]:
        """
        Get the currently selected action from motor cluster.
        
        After running tick(), the motor cluster has been processed
        by Softmax. This method returns the winning motor node.
        
        Returns:
            The node_id of the winning motor node, or None if no action.
        """
        best_node = None
        best_energy = 0.0
        
        for node_id in motor_node_ids:
            node = self.memory.get_node(node_id)
            if node and node.energy > best_energy:
                best_energy = node.energy
                best_node = node_id
        
        return best_node
    
    def get_action_probabilities(self, motor_node_ids: List[str]) -> Dict[str, float]:
        """
        Get the current probability distribution over actions.
        
        Useful for debugging and visualization.
        """
        energies = {}
        for node_id in motor_node_ids:
            node = self.memory.get_node(node_id)
            if node and node.energy > 0.05:
                energies[node_id] = node.energy
        
        if not energies:
            return {}
        
        max_e = max(energies.values())
        exp_values = {k: math.exp((v - max_e) / self.temperature) 
                      for k, v in energies.items()}
        exp_sum = sum(exp_values.values())
        
        if exp_sum == 0:
            return {}
        
        return {k: v / exp_sum for k, v in exp_values.items()}
    
    # =====================
    # System 2 Interface
    # =====================
    
    def pause(self):
        """Pause the tick loop (called by System 2)."""
        self.paused = True
    
    def resume(self):
        """Resume the tick loop."""
        self.paused = False
    
    def get_firing_set(self) -> Set[str]:
        """Get the set of nodes that fired in the last tick."""
        return self.firing_set.copy()
    
    def get_state_summary(self) -> dict:
        """Get a summary of the current system state."""
        return {
            'tick': self.current_tick,
            'active_nodes': len(self.memory.get_active_nodes()),
            'total_nodes': self.memory.node_count,
            'total_energy': self.memory.get_total_energy(),
            'temperature': self.temperature,
            'traced_synapses': self.memory.traced_count,
            'paused': self.paused
        }
