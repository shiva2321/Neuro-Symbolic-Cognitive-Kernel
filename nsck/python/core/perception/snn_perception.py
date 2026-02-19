"""
SNN Perception Module - Spike-based sensory processing with VSA integration

Combines:
- VSA-SNN bridge (spike encoding/decoding)
- Leaky Integrate-and-Fire neurons
- Hebbian learning for concept formation
- Direct hypervector output for cognitive processing

Architecture:
    Input → Spike Encoding → SNN Layer → Hebbian Learning → Concept HVs
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import sys
from pathlib import Path

# Add workspace to path for imports
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from python.core.perception.vsa_snn_bridge import RateCoder, TemporalCoder
from python.core.learning.hebbian import VSAHebbianLearner
from python.core.vsa.hypervec_shim import HyperVector


class SimpleConceptMapper:
    """
    Simple pattern-based concept mapper
    Maps neuron activation patterns to concept hypervectors
    """
    
    def __init__(self, dimension: int = 1024, n_concepts: int = 50):
        self.dimension = dimension
        self.n_concepts = n_concepts
        self.concepts: Dict[int, Tuple[set, HyperVector]] = {}  # id → (neurons, hv)
        self.next_id = 0
    
    def recognize_pattern(self, active_neurons: List[int], threshold: float = 0.6) -> Tuple[int, float]:
        """
        Recognize a pattern by comparing with known concepts
        Returns: (concept_id, similarity) or (-1, 0.0) if novel
        """
        if not self.concepts or not active_neurons:
            return -1, 0.0
        
        active_set = set(active_neurons)
        best_match_id = -1
        best_similarity = 0.0
        
        for concept_id, (neuron_set, _) in self.concepts.items():
            # Jaccard similarity
            intersection = len(active_set & neuron_set)
            union = len(active_set | neuron_set)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match_id = concept_id
        
        return best_match_id, best_similarity
    
    def register_concept(self, active_neurons: List[int]) -> int:
        """Register a new concept from active neurons"""
        concept_id = self.next_id
        self.next_id += 1
        
        # Create unique HV for this concept
        concept_hv = HyperVector(seed=concept_id + 1000)
        self.concepts[concept_id] = (set(active_neurons), concept_hv)
        
        return concept_id
    
    def get_concept_hv(self, concept_id: int) -> HyperVector:
        """Get hypervector for a concept"""
        if concept_id in self.concepts:
            return self.concepts[concept_id][1]
        # Default for unknown concept
        return HyperVector(seed=9999)


class LIFNeuronLayer:
    """
    Leaky Integrate-and-Fire neurons with spike-timing dynamics
    
    Membrane potential equation: τ * dv/dt = -(v - v_rest) + R*I
    Spike when v >= v_thresh, then reset to v_reset
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau: float = 20.0,  # ms
        v_rest: float = -70.0,  # mV
        v_reset: float = -75.0,  # mV
        v_thresh: float = -55.0,  # mV
        refractory_period: float = 2.0,  # ms
        dt: float = 1.0  # ms timestep
    ):
        self.n_neurons = n_neurons
        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.refractory_period = refractory_period
        self.dt = dt
        
        # State variables
        self.v = np.ones(n_neurons) * v_rest  # Membrane potentials
        self.refractory_timer = np.zeros(n_neurons)  # Refractory countdown
        
        # Spike history
        self.spike_times: List[np.ndarray] = []
        self.spike_history: List[np.ndarray] = []
    
    def step(self, input_current: np.ndarray) -> np.ndarray:
        """
        Single timestep update
        
        Args:
            input_current: (n_neurons,) synaptic input current
            
        Returns:
            spikes: (n_neurons,) binary spike vector
        """
        # Decay membrane potential toward rest
        dv = (-(self.v - self.v_rest) + input_current) / self.tau * self.dt
        
        # Only update non-refractory neurons
        active_mask = self.refractory_timer == 0
        self.v[active_mask] += dv[active_mask]
        
        # Detect spikes
        spikes = (self.v >= self.v_thresh).astype(np.float32)
        
        # Reset spiked neurons
        self.v[spikes > 0] = self.v_reset
        self.refractory_timer[spikes > 0] = self.refractory_period
        
        # Update refractory timers
        self.refractory_timer = np.maximum(0, self.refractory_timer - self.dt)
        
        # Record spikes
        self.spike_history.append(spikes.copy())
        
        return spikes
    
    def reset(self):
        """Reset layer state"""
        self.v = np.ones(self.n_neurons) * self.v_rest
        self.refractory_timer = np.zeros(self.n_neurons)
        self.spike_history = []
        self.spike_times = []
    
    def get_spike_train(self) -> np.ndarray:
        """Get full spike train history as (time, n_neurons) array"""
        if not self.spike_history:
            return np.zeros((0, self.n_neurons))
        return np.array(self.spike_history)


class SNNPerceptionModule:
    """
    Complete SNN-based perception module with VSA output
    
    Pipeline:
    1. Encode sensory input as spike trains (rate or temporal coding)
    2. Process through LIF neuron layer
    3. Learn associations via Hebbian plasticity
    4. Map spike patterns to concept hypervectors
    5. Output semantic representations for cognitive processing
    """
    
    def __init__(
        self,
        input_dim: int,
        snn_size: int = 256,
        hv_dimension: int = 1024,
        n_concepts: int = 50,
        encoding_mode: str = "rate",  # "rate" or "temporal"
        simulation_time_ms: float = 50.0,
        hebbian_lr: float = 0.001,
        stdp_enabled: bool = True,  # NEW: Enable STDP learning
        stdp_lr: float = 0.0001,  # NEW: STDP learning rate
        tau_stdp: float = 20.0,  # NEW: STDP time constant (ms)
        a_plus: float = 0.001,  # NEW: LTP magnitude
        a_minus: float = 0.0005  # NEW: LTD magnitude
    ):
        self.input_dim = input_dim
        self.snn_size = snn_size
        self.hv_dimension = hv_dimension
        self.n_concepts = n_concepts
        self.encoding_mode = encoding_mode
        self.simulation_time_ms = simulation_time_ms
        
        # STDP parameters
        self.stdp_enabled = stdp_enabled
        self.stdp_lr = stdp_lr
        self.tau_stdp = tau_stdp
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Components
        if encoding_mode == "rate":
            self.encoder = RateCoder(n_neurons=snn_size,  # Match SNN size!
                                     dimension=hv_dimension)
        else:
            self.encoder = TemporalCoder(n_neurons=snn_size,  # Match SNN size!
                                         dimension=hv_dimension,
                                         time_bins=10)
        
        self.snn_layer = LIFNeuronLayer(n_neurons=snn_size,
                                        tau=10.0,  # Faster dynamics
                                        v_thresh=-50.0)  # Lower threshold
        
        self.hebbian_learner = VSAHebbianLearner(
            dimension=hv_dimension,
            n_concepts=n_concepts,
            learning_rate=hebbian_lr
        )
        
        self.concept_mapper = SimpleConceptMapper(
            dimension=hv_dimension,
            n_concepts=n_concepts
        )
        
        # Input → SNN connectivity (stronger initialization for spiking)
        self.input_weights = np.random.randn(snn_size, input_dim) * 5.0  # Increased from 0.1
        
        # STDP spike time tracking
        self.pre_spike_times = np.full(input_dim, -1000.0)  # Last spike time for each input
        self.post_spike_times = np.full(snn_size, -1000.0)  # Last spike time for each neuron
        self.current_time = 0.0  # Current simulation time (ms)
        
        # Statistics (initialize BEFORE calling _normalize_weights)
        self.processing_times = []
        self.weight_updates = 0
        self.stdp_updates = 0
        
        # Normalize weights initially (prevent runaway during learning)
        self._normalize_weights()
    
    def _normalize_weights(self):
        """
        Normalize input weights to prevent runaway growth during learning.
        Uses L2 normalization per neuron (each row normalized independently).
        """
        for i in range(self.snn_size):
            norm = np.linalg.norm(self.input_weights[i])
            if norm > 0:
                self.input_weights[i] /= norm
        self.weight_updates += 1
    
    def update_weights(self, delta_w: np.ndarray, normalize: bool = True):
        """
        Apply weight update with optional normalization.
        
        Args:
            delta_w: Weight change matrix (snn_size, input_dim)
            normalize: If True, normalize weights after update
        """
        self.input_weights += delta_w
        if normalize:
            self._normalize_weights()
    
    def _apply_stdp(self, input_spikes: np.ndarray, output_spikes: np.ndarray):
        """
        Apply Spike-Timing-Dependent Plasticity (STDP) weight updates.
        
        STDP Rule:
        - If pre-synaptic spike before post-synaptic: LTP (strengthen)
        - If post-synaptic spike before pre-synaptic: LTD (weaken)
        
        Weight change: Δw = η * A * exp(-|Δt| / τ) * sign(Δt)
        
        Args:
            input_spikes: (input_dim,) binary vector of pre-synaptic spikes
            output_spikes: (snn_size,) binary vector of post-synaptic spikes
        """
        if not self.stdp_enabled:
            return
        
        # Update spike times
        input_spike_idx = np.where(input_spikes > 0)[0]
        output_spike_idx = np.where(output_spikes > 0)[0]
        
        if len(input_spike_idx) > 0:
            self.pre_spike_times[input_spike_idx] = self.current_time
        if len(output_spike_idx) > 0:
            self.post_spike_times[output_spike_idx] = self.current_time
        
        # Calculate weight changes for all synapses
        # Only update if both pre and post have spiked recently (within 5*tau_stdp)
        delta_w = np.zeros_like(self.input_weights)
        
        for post_idx in range(self.snn_size):
            if self.post_spike_times[post_idx] > -1000:  # Neuron has spiked at least once
                for pre_idx in range(self.input_dim):
                    if self.pre_spike_times[pre_idx] > -1000:  # Input has spiked
                        # Calculate spike time difference
                        delta_t = self.post_spike_times[post_idx] - self.pre_spike_times[pre_idx]
                        
                        # Only apply STDP if spikes are close in time
                        if abs(delta_t) < 5 * self.tau_stdp:
                            if delta_t > 0:
                                # Post after pre → LTP (strengthen)
                                dw = self.a_plus * np.exp(-delta_t / self.tau_stdp)
                            else:
                                # Pre after post → LTD (weaken)
                                dw = -self.a_minus * np.exp(delta_t / self.tau_stdp)
                            
                            delta_w[post_idx, pre_idx] += dw
        
        # Apply weight update with learning rate
        self.input_weights += self.stdp_lr * delta_w
        
        # Normalize to prevent runaway growth
        self._normalize_weights()
        self.stdp_updates += 1
    
    def perceive(
        self,
        sensory_input: np.ndarray,
        learn: bool = True
    ) -> Dict:
        """
        Process sensory input through full SNN pipeline
        
        Args:
            sensory_input: (input_dim,) normalized sensory vector
            learn: Whether to update Hebbian associations
            
        Returns:
            Dict with:
                - concept_hv: Output hypervector representing perceived concept
                - concept_id: Recognized concept ID (or -1 if novel)
                - spike_train: (time, snn_size) spike activity
                - processing_time_ms: Total processing latency
        """
        start_time = time.perf_counter()
        
        # 1. Simulate SNN dynamics
        self.snn_layer.reset()
        n_steps = int(self.simulation_time_ms / self.snn_layer.dt)
        self.current_time = 0.0  # Reset simulation time
        
        for step in range(n_steps):
            # Generate input spikes from sensory signal (Poisson-like)
            # Higher values → higher spike probability
            spike_prob = np.clip(sensory_input, 0, 1)  # Ensure [0,1]
            input_spikes = (np.random.rand(self.input_dim) < spike_prob * 0.5).astype(np.float32)
            
            # Generate input current from sensory signal
            input_current = self.input_weights @ sensory_input
            
            # Amplify current and add noise for robustness
            input_current = input_current * 2.0 + np.random.randn(self.snn_size) * 1.0
            
            # Step SNN forward
            output_spikes = self.snn_layer.step(input_current)
            
            # Apply STDP learning
            if learn and self.stdp_enabled:
                self._apply_stdp(input_spikes, output_spikes)
            
            # Advance simulation time
            self.current_time += self.snn_layer.dt
        
        spike_train = self.snn_layer.get_spike_train()
        
        # 2. Encode spike train using rate or temporal coding
        if self.encoding_mode == "rate":
            encoding = self.encoder.encode(spike_train, time_window_ms=self.simulation_time_ms)
        else:
            encoding = self.encoder.encode(spike_train, dt_ms=self.snn_layer.dt)
        
        encoding_hv = encoding.hv
        active_neurons = encoding.active_neurons
        
        # 3. Recognize or register concept
        concept_id, strength = self.concept_mapper.recognize_pattern(active_neurons)
        
        if concept_id == -1 and len(active_neurons) > 0:
            # Novel pattern - register it
            concept_id = self.concept_mapper.register_concept(active_neurons)
            strength = 1.0  # Perfect match to itself
        
        # 4. Update Hebbian associations (if learning)
        if learn and len(active_neurons) > 0:
            # Form associations between co-active concepts
            active_concept_ids = [concept_id]
            self.hebbian_learner.update_associations(active_concept_ids)
        
        # 5. Get final concept hypervector
        concept_hv = self.concept_mapper.get_concept_hv(concept_id)
        
        processing_time = (time.perf_counter() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        return {
            "concept_hv": concept_hv,
            "concept_id": concept_id,
            "strength": strength,
            "spike_train": spike_train,
            "encoding_hv": encoding_hv,
            "active_neurons": active_neurons,
            "processing_time_ms": processing_time,
            "n_spikes": int(spike_train.sum())
        }
    
    def get_stats(self) -> Dict:
        """Get module statistics"""
        if not self.processing_times:
            return {"avg_latency_ms": 0, "n_processed": 0}
        
        # Calculate weight statistics
        weight_norms = [np.linalg.norm(self.input_weights[i]) for i in range(self.snn_size)]
        
        return {
            "avg_latency_ms": np.mean(self.processing_times),
            "max_latency_ms": np.max(self.processing_times),
            "n_processed": len(self.processing_times),
            "n_concepts_learned": len(self.concept_mapper.concepts),
            "weight_updates": self.weight_updates,
            "stdp_updates": self.stdp_updates,
            "stdp_enabled": self.stdp_enabled,
            "avg_weight_norm": np.mean(weight_norms),
            "max_weight_norm": np.max(weight_norms),
            "min_weight_norm": np.min(weight_norms)
        }
    
    def reset_stats(self):
        """Clear statistics"""
        self.processing_times = []


# ============================================================================
# Integration Interface for CognitiveEngine
# ============================================================================

class SNNWorkspaceModule:
    """
    Wrapper to make SNN perception compatible with Global Workspace architecture
    
    Provides WorkspaceModule interface for broadcasting perceptual concepts
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        snn_size: int = 256,
        hv_dimension: int = 1024
    ):
        self.module_name = "SNN_Perception"
        self.perception = SNNPerceptionModule(
            input_dim=input_dim,
            snn_size=snn_size,
            hv_dimension=hv_dimension
        )
        
        # Activation tracking
        self.last_activation = 0.0
        self.last_concept_id = -1
    
    def process(self, sensory_data: np.ndarray) -> Tuple[HyperVector, float]:
        """
        Process sensory input and return concept + activation for workspace
        
        Args:
            sensory_data: Raw sensory input
            
        Returns:
            (concept_hv, activation_strength)
        """
        result = self.perception.perceive(sensory_data, learn=True)
        
        # Activation strength from recognition confidence + spike activity
        spike_activity = result["n_spikes"] / (self.perception.snn_size * 50)  # Normalized
        activation = (result["strength"] + spike_activity) / 2.0
        
        self.last_activation = activation
        self.last_concept_id = result["concept_id"]
        
        return result["concept_hv"], activation
    
    def get_status(self) -> Dict:
        """Get module status for monitoring"""
        stats = self.perception.get_stats()
        return {
            "module": self.module_name,
            "last_activation": self.last_activation,
            "last_concept_id": self.last_concept_id,
            **stats
        }


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("=== SNN Perception Module Test ===\n")
    
    # Create module
    module = SNNPerceptionModule(
        input_dim=64,
        snn_size=128,
        hv_dimension=1024,
        n_concepts=20,
        encoding_mode="rate",
        simulation_time_ms=50.0
    )
    
    print(f"Configuration:")
    print(f"  Input: {module.input_dim} dimensions")
    print(f"  SNN: {module.snn_size} LIF neurons")
    print(f"  HV: {module.hv_dimension} bits")
    print(f"  Simulation: {module.simulation_time_ms} ms\n")
    
    # Test pattern 1
    pattern1 = np.random.randn(64) * 0.5
    result1 = module.perceive(pattern1, learn=True)
    
    print(f"Pattern 1:")
    print(f"  Concept ID: {result1['concept_id']}")
    print(f"  Recognition strength: {result1['strength']:.3f}")
    print(f"  Spikes: {result1['n_spikes']}")
    print(f"  Active neurons: {len(result1['active_neurons'])}")
    print(f"  Processing time: {result1['processing_time_ms']:.3f} ms\n")
    
    # Test pattern 2 (similar to pattern 1)
    pattern2 = pattern1 + np.random.randn(64) * 0.1
    result2 = module.perceive(pattern2, learn=True)
    
    print(f"Pattern 2 (similar):")
    print(f"  Concept ID: {result2['concept_id']}")
    print(f"  Recognition strength: {result2['strength']:.3f}")
    print(f"  Spikes: {result2['n_spikes']}")
    print(f"  Processing time: {result2['processing_time_ms']:.3f} ms\n")
    
    # Test pattern 3 (different)
    pattern3 = np.random.randn(64) * 0.5
    result3 = module.perceive(pattern3, learn=True)
    
    print(f"Pattern 3 (different):")
    print(f"  Concept ID: {result3['concept_id']}")
    print(f"  Recognition strength: {result3['strength']:.3f}")
    print(f"  Spikes: {result3['n_spikes']}")
    print(f"  Processing time: {result3['processing_time_ms']:.3f} ms\n")
    
    # Stats
    stats = module.get_stats()
    print(f"Module Statistics:")
    print(f"  Average latency: {stats['avg_latency_ms']:.3f} ms")
    print(f"  Concepts learned: {stats['n_concepts_learned']}")
    print(f"  Patterns processed: {stats['n_processed']}\n")
    
    print("✅ SNN Perception Module: READY")
