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

# ── Rust SNN acceleration (mirrors hypervec_shim try/fallback pattern) ──────
_SNN_USE_RUST: bool = False
_snn_rs_ext = None
try:
    import snn_rs as _snn_rs_ext  # compiled Rust extension
    _SNN_USE_RUST = True
    print(">> [SNN]  Rust backend active (snn_rs)")
except ImportError:
    print(">> [SNN]  Rust backend not found — pure Python SNN active")


class SimpleConceptMapper:
    """
    Pattern-based concept mapper with VSA cleanup-memory grounding.

    Maps neuron activation patterns to concept hypervectors and, when a
    SemanticMemory reference is available, resolves concept IDs to human-
    readable predicate names via nearest-neighbour HV lookup (cleanup
    memory).  This closes the SNN→predicate bridge so that downstream
    GWT coalitions carry meaningful symbolic content.
    """
    
    def __init__(self, dimension: int = 1024, n_concepts: int = 50):
        self.dimension = dimension
        self.n_concepts = n_concepts
        self.concepts: Dict[int, Tuple[set, HyperVector]] = {}  # id → (neurons, hv)
        self.concept_labels: Dict[int, str] = {}  # id → semantic name
        self.next_id = 0
        self._semantic_memory = None  # Optional SemanticMemory for HV lookup
        self._registered: Dict[str, np.ndarray] = {}  # V4: named concept prototypes
    
    # ── optional SemanticMemory hookup ──────────────────────────────────
    def attach_semantic_memory(self, semantic_memory) -> None:
        """Attach a SemanticMemory for VSA cleanup-based concept naming."""
        self._semantic_memory = semantic_memory

    def register(self, name: str, bits: np.ndarray) -> None:
        """Register a named concept prototype for cleanup memory lookup (V4)."""
        self._registered[name] = bits

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

        # Try to ground concept via semantic memory HV cleanup
        label = self._resolve_label(concept_id, concept_hv)
        self.concept_labels[concept_id] = label
        
        return concept_id

    def label_concept(self, concept_id: int, label: str) -> None:
        """Manually assign a human-readable label to a concept."""
        self.concept_labels[concept_id] = label

    def get_concept_label(self, concept_id: int) -> str:
        """Return human-readable label, falling back to ``SNN_Concept_<id>``."""
        return self.concept_labels.get(concept_id, f"SNN_Concept_{concept_id}")

    def _resolve_label(self, concept_id: int, concept_hv: HyperVector) -> str:
        """Nearest-neighbour HV lookup in SemanticMemory (cleanup memory)."""
        if self._semantic_memory is None:
            return f"SNN_Concept_{concept_id}"
        try:
            # SemanticMemory stores concept_hvs: Dict[name, HyperVector]
            best_name, best_sim = None, 0.0
            concept_hvs = getattr(self._semantic_memory, "concept_hvs", {})
            for name, hv in concept_hvs.items():
                sim = concept_hv.similarity(hv)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            if best_name and best_sim > 0.55:
                return best_name
        except Exception:
            pass
        return f"SNN_Concept_{concept_id}"
    
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


# ============================================================================
# PythonSnnCore — pure-NumPy drop-in for ``snn_rs.SnnCore``
# ============================================================================

class PythonSnnCore:
    """
    Pure-NumPy implementation of SnnCore with the same interface as the Rust
    ``snn_rs.SnnCore`` extension.

    ``simulate(sensory_input, n_steps, learn)`` runs the full LIF + STDP
    inner loop and returns a list-of-lists spike train (n_steps × snn_size).
    The STDP update uses a vectorised outer-product so there are no nested
    Python loops -- only the ``n_steps`` outer loop remains in Python.
    """

    def __init__(
        self,
        input_dim: int,
        snn_size: int,
        tau: float = 10.0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        v_thresh: float = -50.0,
        refractory_period: float = 2.0,
        dt: float = 1.0,
        stdp_enabled: bool = True,
        stdp_lr: float = 0.0001,
        tau_stdp: float = 20.0,
        a_plus: float = 0.001,
        a_minus: float = 0.0005,
        seed=None,
    ):
        self.input_dim        = input_dim
        self.snn_size         = snn_size
        self.tau              = tau
        self.v_rest           = v_rest
        self.v_reset          = v_reset
        self.v_thresh         = v_thresh
        self.refractory_period = refractory_period
        self.dt               = dt
        self.stdp_enabled     = stdp_enabled
        self.stdp_lr          = stdp_lr
        self.tau_stdp         = tau_stdp
        self.a_plus           = a_plus
        self.a_minus          = a_minus

        rng = np.random.default_rng(seed)
        self.input_weights: np.ndarray = (
            rng.standard_normal((snn_size, input_dim)).astype(np.float64) * 0.1
        )
        norms = np.linalg.norm(self.input_weights, axis=1, keepdims=True)
        self.input_weights /= np.maximum(norms, 1e-12)

        self.v              = np.full(snn_size, v_rest,  dtype=np.float64)
        self.refractory     = np.zeros(snn_size,          dtype=np.float64)
        self.pre_spike_times  = np.full(input_dim, -1e4, dtype=np.float64)
        self.post_spike_times = np.full(snn_size,  -1e4, dtype=np.float64)
        self.current_time: float = 0.0
        self.stdp_updates:   int = 0
        self.weight_updates: int = 0

    def simulate(self, sensory_input: list, n_steps: int, learn: bool) -> list:
        """Run n_steps LIF + STDP steps; return spike train list-of-lists."""
        x      = np.asarray(sensory_input, dtype=np.float64)
        window = 5.0 * self.tau_stdp
        spike_train: list = []

        for _ in range(n_steps):
            # Input projection + Gaussian noise
            current = self.input_weights @ x * 2.0 + np.random.randn(self.snn_size)

            # LIF membrane update (vectorized)
            active = self.refractory <= 0.0
            self.v[active] += (
                self.dt / self.tau
                * (-(self.v[active] - self.v_rest) + current[active])
            )
            self.refractory = np.maximum(0.0, self.refractory - self.dt)

            spikes = (self.v >= self.v_thresh).astype(np.float64)
            self.v[spikes > 0]          = self.v_reset
            self.refractory[spikes > 0] = self.refractory_period

            # STDP: vectorized outer-product (no nested Python loops)
            if learn and self.stdp_enabled:
                in_sp = (
                    np.random.rand(self.input_dim)
                    < np.clip(x, 0.0, 1.0) * 0.5
                ).astype(np.float64)
                t = self.current_time
                self.pre_spike_times[in_sp > 0.5]   = t
                self.post_spike_times[spikes > 0.5] = t

                vp     = self.post_spike_times > -999.0   # (snn_size,)
                vr     = self.pre_spike_times  > -999.0   # (input_dim,)
                dt_mat = (
                    self.post_spike_times[:, None]
                    - self.pre_spike_times[None, :]
                )
                mask   = vp[:, None] & vr[None, :] & (np.abs(dt_mat) < window)
                ltp    = mask & (dt_mat > 0)
                ltd    = mask & (dt_mat <= 0)

                dw = np.zeros_like(self.input_weights)
                dw[ltp] =  self.a_plus  * np.exp(-dt_mat[ltp] / self.tau_stdp)
                dw[ltd] = -self.a_minus * np.exp( dt_mat[ltd] / self.tau_stdp)
                self.input_weights += self.stdp_lr * dw
                # Soft max-norm clipping: rows exceeding max_norm are scaled
                # down; rows below budget are untouched.  This preserves the
                # effective weight scale so large inputs continue to drive
                # reliable spiking (avoids the unit-norm collapse).
                _max_norm = 4.0
                norms = np.linalg.norm(self.input_weights, axis=1, keepdims=True)
                self.input_weights /= np.where(norms > _max_norm, norms / _max_norm, 1.0)
                self.stdp_updates  += 1
                self.current_time  += self.dt

            spike_train.append(spikes.tolist())

        return spike_train

    def get_weights(self) -> list:
        return self.input_weights.ravel().tolist()

    def set_weights(self, weights: list) -> None:
        self.input_weights = np.asarray(weights, dtype=np.float64).reshape(
            self.snn_size, self.input_dim
        )

    def normalize_weights(self) -> None:
        # Soft max-norm clipping: only scale rows exceeding budget.
        # This is consistent with the per-step STDP normalization and
        # prevents the unit-norm collapse that silences all spikes.
        _max_norm = 4.0
        norms = np.linalg.norm(self.input_weights, axis=1, keepdims=True)
        self.input_weights /= np.where(norms > _max_norm, norms / _max_norm, 1.0)
        self.weight_updates += 1

    def __repr__(self) -> str:
        return (
            f"<SnnCore(Python) input={self.input_dim} "
            f"snn={self.snn_size} stdp={self.stdp_enabled}>"
        )


# ============================================================================
# PythonStdpEngine — pure-NumPy drop-in for ``snn_rs.StdpEngine``
# ============================================================================

class PythonStdpEngine:
    """
    Standalone STDP engine with the same interface as the Rust ``snn_rs.StdpEngine``.

    ``apply(input_spikes, output_spikes)`` returns a flat delta_w list
    (snn_size × input_dim, row-major) already multiplied by stdp_lr so the
    caller does: ``weights += engine.apply(...)``.
    """

    def __init__(
        self,
        input_dim: int,
        snn_size: int,
        stdp_lr: float,
        tau_stdp: float,
        a_plus: float,
        a_minus: float,
    ):
        self.input_dim  = input_dim
        self.snn_size   = snn_size
        self.stdp_lr    = stdp_lr
        self.tau_stdp   = tau_stdp
        self.a_plus     = a_plus
        self.a_minus    = a_minus
        self.pre_spike_times  = np.full(input_dim, -1e4, dtype=np.float64)
        self.post_spike_times = np.full(snn_size,  -1e4, dtype=np.float64)
        self.current_time: float = 0.0
        self.updates: int = 0

    def advance_time(self, dt: float) -> None:
        self.current_time += dt

    def reset(self) -> None:
        self.pre_spike_times.fill(-1e4)
        self.post_spike_times.fill(-1e4)
        self.current_time = 0.0

    def apply(self, input_spikes: list, output_spikes: list) -> list:
        """Compute STDP delta_w (flat, row-major, scaled by stdp_lr)."""
        in_sp  = np.asarray(input_spikes,  dtype=np.float64)
        out_sp = np.asarray(output_spikes, dtype=np.float64)
        t = self.current_time
        self.pre_spike_times[in_sp > 0.5]   = t
        self.post_spike_times[out_sp > 0.5] = t

        window = 5.0 * self.tau_stdp
        vp     = self.post_spike_times > -999.0
        vr     = self.pre_spike_times  > -999.0
        dt_mat = (
            self.post_spike_times[:, None]
            - self.pre_spike_times[None, :]
        )
        mask = vp[:, None] & vr[None, :] & (np.abs(dt_mat) < window)
        ltp  = mask & (dt_mat > 0)
        ltd  = mask & (dt_mat <= 0)

        dw = np.zeros((self.snn_size, self.input_dim), dtype=np.float64)
        dw[ltp] =  self.a_plus  * np.exp(-dt_mat[ltp] / self.tau_stdp) * self.stdp_lr
        dw[ltd] = -self.a_minus * np.exp( dt_mat[ltd] / self.tau_stdp) * self.stdp_lr

        self.updates += 1
        return dw.ravel().tolist()

    def __repr__(self) -> str:
        return (
            f"<StdpEngine(Python) input={self.input_dim} "
            f"snn={self.snn_size}>"
        )


# ============================================================================


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
        simulation_time_ms: float = 20.0,  # 20 ms nominal (was 50 ms — 2.5x faster)
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
                                         dimension=hv_dimension)
        
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
        # Use scale 5.0 and DO NOT normalize at init — normalization collapses
        # every row to unit-norm, making input currents ~0.8 mV which is far
        # too weak to drive a -70 mV resting neuron past the -50 mV threshold.
        # Normalization is applied only after STDP updates to prevent runaway.
        self.input_weights = np.random.randn(snn_size, input_dim) * 5.0

        # Cached mean row-norm for adaptive gain; re-computed only after learning
        # calls so every perceive() call avoids the 256×64 norm computation.
        # Initial value: norm of fresh scale-5 rows ≈ 5 × sqrt(input_dim).
        self._cached_avg_w_norm: float = float(
            np.linalg.norm(self.input_weights, axis=1).mean()
        )
        # STDP spike time tracking
        self.pre_spike_times = np.full(input_dim, -1000.0)  # Last spike time for each input
        self.post_spike_times = np.full(snn_size, -1000.0)  # Last spike time for each neuron
        self.current_time = 0.0  # Current simulation time (ms)
        
        # Statistics
        self.processing_times = []
        self.weight_updates = 0
        self.stdp_updates = 0

        # ── SNN Core: always available (Rust when compiled, Python otherwise) ─────────
        # SnnCore.simulate() replaces the entire per-step Python loop in perceive().
        _core_cls = _snn_rs_ext.SnnCore if _SNN_USE_RUST else PythonSnnCore
        try:
            self._snn_core = _core_cls(
                input_dim         = self.input_dim,
                snn_size          = self.snn_size,
                tau               = 10.0,
                v_rest            = -70.0,
                v_reset           = -75.0,
                v_thresh          = -50.0,
                refractory_period = 2.0,
                dt                = 1.0,
                stdp_enabled      = self.stdp_enabled,
                stdp_lr           = self.stdp_lr,
                tau_stdp          = self.tau_stdp,
                a_plus            = self.a_plus,
                a_minus           = self.a_minus,
                seed              = None,
            )
            # Seed with the same normalised weights Python just built.
            self._snn_core.set_weights(self.input_weights.flatten().tolist())
        except Exception as _e:
            self._snn_core = None  # Rare; perceive() falls back to inline Python loop.

    def _normalize_weights(self):
        """
        Soft max-norm clipping: only scale rows that exceed the norm budget.
        Rows below budget are untouched so early-stage weights can grow.
        Consistent with the PythonSnnCore STDP normalization.
        """
        _max_norm = 4.0
        norms = np.linalg.norm(self.input_weights, axis=1, keepdims=True)
        self.input_weights /= np.where(norms > _max_norm, norms / _max_norm, 1.0)
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
        
        # Vectorized STDP weight update (outer-product, same as PythonStdpEngine.apply()).
        # NOTE: this path is only reached when both the Rust SnnCore and PythonSnnCore
        # are unavailable (i.e. essentially never in production).
        window = 5.0 * self.tau_stdp
        vp = self.post_spike_times > -999.0
        vr = self.pre_spike_times > -999.0
        dt_mat = self.post_spike_times[:, None] - self.pre_spike_times[None, :]
        mask = vp[:, None] & vr[None, :] & (np.abs(dt_mat) < window)
        ltp = mask & (dt_mat > 0)
        ltd = mask & (dt_mat <= 0)
        delta_w = np.zeros_like(self.input_weights)
        delta_w[ltp] = self.a_plus * np.exp(-dt_mat[ltp] / self.tau_stdp)
        delta_w[ltd] = -self.a_minus * np.exp(dt_mat[ltd] / self.tau_stdp)

        # Apply weight update with learning rate
        self.input_weights += self.stdp_lr * delta_w
        
        # Normalize to prevent runaway growth
        self._normalize_weights()
        self.stdp_updates += 1
    
    # Target L×gain product that gives ~25% per-step spiking probability.
    # For a row w with ||w||=L and input x~N(0, gain), the dv per LIF step is
    # N(0, 0.2·L·gain).  P(first-step spike) ≈ P(Z > 20) where Z~N(0, 0.2·L·gain).
    # Setting 0.2·L·gain = 30 → L·gain = 150 gives P ~ 25% (comfortable sweet spot).
    # With adaptive gain = 150 / avg(||w_row||) this automatically scales to:
    #   - Unit-norm Rust weights (L=1)   → gain = 150
    #   - Soft-max-norm(4) Python weights → gain =  37.5
    #   - Fresh scale-5   init weights   → gain ≈   3.75
    _TARGET_LG: float = 150.0

    def perceive(
        self,
        sensory_input: np.ndarray,
        learn: bool = True
    ) -> Dict:
        """
        Process sensory input through full SNN pipeline
        
        Args:
            sensory_input: (input_dim,) sensory vector (any scale)
            learn: Whether to update Hebbian associations
            
        Returns:
            Dict with:
                - concept_hv: Output hypervector representing perceived concept
                - concept_id: Recognized concept ID (or -1 if novel)
                - spike_train: (time, snn_size) spike activity
                - processing_time_ms: Total processing latency
        """
        start_time = time.perf_counter()

        # ── Adaptive input standardization ───────────────────────────────
        # Rust SNN keeps unit-norm weights; Python soft-max-norm keeps ~4.0.
        # Fresh scale-5 init has norm ~40.  A fixed gain cannot serve all cases.
        # Solution: compute gain = _TARGET_LG / mean(||w_row||) so that
        #   0.2 × avg_weight_norm × gain ≈ 30 mV → ~25% per-step firing rate.
        # We use _cached_avg_w_norm (updated after learning calls) to avoid
        # a 256×64 norm computation on every inference call.
        x_raw = np.asarray(sensory_input, dtype=np.float64)

        # ── Weber-Fechner logarithmic compression ───────────────────────
        # Psychophysics (Fechner 1860): perceived intensity S = k·log(I/I₀)
        # Applied to raw input BEFORE standardization to compress the dynamic
        # range of sensory signals — improving discrimination at low
        # intensities and preventing saturation at high intensities.
        # The sign-preserving form: x_wf = sign(x) · log(1 + |x|)
        x_raw = np.sign(x_raw) * np.log1p(np.abs(x_raw))

        _mu, _sigma = x_raw.mean(), x_raw.std()

        avg_w_norm = self._cached_avg_w_norm  # cheap: just an attribute read
        effective_gain = self._TARGET_LG / max(avg_w_norm, 1e-4)

        if _sigma > 1e-6:
            x_proc = (x_raw - _mu) / _sigma * effective_gain
        else:
            # Constant (zero-variance) input: uniform drive at effective_gain.
            x_proc = np.ones_like(x_raw) * effective_gain

        # 1. Simulate SNN dynamics
        n_steps = int(self.simulation_time_ms / self.snn_layer.dt)

        if self._snn_core is not None:
            # ── RUST FAST PATH ───────────────────────────────────────────────
            # Entire loop (LIF + noise + STDP) runs in one Rust call.
            # Returns list-of-lists (n_steps × snn_size); convert to ndarray.
            spike_train_nested = self._snn_core.simulate(
                x_proc.tolist(),
                n_steps,
                learn,
            )
            spike_train = np.array(spike_train_nested, dtype=np.float32)
            # Sync weights back only after LEARNING calls (when Rust STDP may
            # have modified them).  Skipping on pure-inference calls avoids the
            # expensive 16,384-element Python-list allocation + numpy conversion
            # on every non-learning step (≈80% of calls with lazy STDP).
            if learn:
                rust_w = self._snn_core.get_weights()
                self.input_weights = np.array(rust_w, dtype=np.float32).reshape(
                    self.snn_size, self.input_dim
                )
                # Recompute cached norm from the freshly-synced weights.
                self._cached_avg_w_norm = float(
                    np.linalg.norm(self.input_weights, axis=1).mean()
                )
        else:
            # ── PYTHON FALLBACK PATH ─────────────────────────────────────────
            self.snn_layer.reset()
            self.current_time = 0.0
            for _step in range(n_steps):
                # Map x_proc (N(0, effective_gain)) to [0,1] for spike-prob estimate
                spike_prob   = np.clip((x_proc + effective_gain) / (2 * effective_gain), 0, 1)
                input_spikes = (np.random.rand(self.input_dim) < spike_prob * 0.5).astype(np.float32)
                input_current = self.input_weights @ x_proc
                input_current = input_current * 2.0 + np.random.randn(self.snn_size) * 1.0
                output_spikes = self.snn_layer.step(input_current)
                if learn and self.stdp_enabled:
                    self._apply_stdp(input_spikes, output_spikes)
                self.current_time += self.snn_layer.dt
            spike_train = self.snn_layer.get_spike_train()
            spike_train = np.array(spike_train, dtype=np.float32)
        
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
        
        # Pull STDP / weight update counts from core (Rust or Python)
        rust_stdp    = self._snn_core.stdp_updates   if self._snn_core else self.stdp_updates
        rust_w_upd   = self._snn_core.weight_updates if self._snn_core else self.weight_updates
        backend      = "rust" if _SNN_USE_RUST and self._snn_core is not None else "python"

        return {
            "avg_latency_ms": np.mean(self.processing_times),
            "max_latency_ms": np.max(self.processing_times),
            "n_processed": len(self.processing_times),
            "n_concepts_learned": len(self.concept_mapper.concepts),
            "weight_updates": rust_w_upd,
            "stdp_updates": rust_stdp,
            "stdp_enabled": self.stdp_enabled,
            "avg_weight_norm": np.mean(weight_norms),
            "max_weight_norm": np.max(weight_norms),
            "min_weight_norm": np.min(weight_norms),
            "backend": backend,
        }
    
    def reset_stats(self):
        """Clear statistics"""
        self.processing_times = []

    def register_concepts_from_memory(self, semantic_memory) -> int:
        """
        Populate the concept mapper from SemanticMemory (V4 SNN grounding).
        Call this after SemanticMemory has been populated.
        Returns number of concepts registered.
        """
        count = 0
        for concept_name, concept_hv in semantic_memory.concept_hvs.items():
            try:
                bits = np.asarray(concept_hv.bits, dtype=np.float32)
                if len(bits) != self.hv_dimension:
                    # Resize via random projection (Gaussian, std=0.1 keeps magnitudes small
                    # before thresholding). Occurs when SemanticMemory was built with a
                    # different HV dimension than the SNN perception module.
                    proj = np.random.randn(self.hv_dimension, len(bits)).astype(np.float32) * 0.1
                    bits = (proj @ bits > 0).astype(np.float32)
                self.concept_mapper.register(concept_name, bits)
                count += 1
            except Exception:
                pass
        return count


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
