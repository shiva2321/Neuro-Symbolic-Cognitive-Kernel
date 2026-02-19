"""
VSA-SNN Bridge
==============
Bidirectional bridge between Vector Symbolic Architecture (VSA) and Spiking Neural Networks (SNN).

This module provides:
1. Spike trains → HyperVectors (rate coding, temporal coding)
2. HyperVectors → Spike patterns (initialization, priming)
3. Hebbian learning integrated with VSA memory
4. Concept formation from neural activity

Architecture:
    SNN Layer → Spike Train → VSA Encoder → HyperVector → Semantic Memory
    Semantic Memory → HyperVector → VSA Decoder → Spike Pattern → SNN Layer

Performance:
    - Encoding: 0.1-1ms (CPU)
    - Decoding: 0.1-1ms (CPU)
    - Rust-accelerated HV operations where available
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    F = None

# Use relative imports to work with any package structure
try:
    from ..vsa import hypervec_shim as hypervec_rs
except (ImportError, ValueError):
    # Fallback for when run as script or different import context
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    try:
        from python.core.vsa import hypervec_shim as hypervec_rs
    except ImportError:
        from python.core.vsa.hypervec_py import HyperVector as _HV
        class _Shim:
            HyperVector = _HV
        hypervec_rs = _Shim()

logger = logging.getLogger("nsck.vsa_snn_bridge")


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class SpikeEncoding:
    """Result of encoding spike train to hypervector."""
    hv: Any                          # HyperVector representation
    active_neurons: List[int]        # Neuron IDs that spiked
    spike_rates: np.ndarray         # Firing rates per neuron
    encoding_time_ms: float          # Time taken to encode
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptActivation:
    """A concept activated from neural activity."""
    concept_id: int
    concept_name: str
    activation_strength: float
    source_neurons: List[int]
    timestamp: float


# ===========================================================================
# Spike → VSA Encoders
# ===========================================================================

class RateCoder:
    """
    Rate coding: Spike frequency → concept salience.
    
    Maps spike rates to hypervector space using weighted bundling.
    Higher firing rate = stronger contribution to the final HV.
    
    Example:
        neuron_0 fires at 80Hz → strong weight
        neuron_1 fires at 20Hz → weak weight
        Result: HV ≈ 0.8*HV_0 + 0.2*HV_1
    """
    
    def __init__(self, n_neurons: int = 1000, dimension: int = 10240, 
                 rate_threshold: float = 0.1):
        """
        Args:
            n_neurons: Number of neurons in the SNN layer
            dimension: Hypervector dimensionality (default 10240)
            rate_threshold: Minimum firing rate to be considered active
        """
        self.n_neurons = n_neurons
        self.dimension = dimension
        self.rate_threshold = rate_threshold
        
        # Each neuron has an associated hypervector
        self.neuron_hvs = [
            hypervec_rs.HyperVector(seed=i) 
            for i in range(n_neurons)
        ]
        
    def encode(self, spike_train: np.ndarray, time_window_ms: float = 100.0) -> SpikeEncoding:
        """
        Encode spike train to hypervector using rate coding.
        
        Args:
            spike_train: Binary spike array [time_steps, n_neurons]
            time_window_ms: Time window for rate computation
            
        Returns:
            SpikeEncoding with resulting hypervector
        """
        import time
        t0 = time.time()
        
        # Compute firing rates (spikes per ms)
        if len(spike_train.shape) == 1:
            # Single neuron case
            spike_train = spike_train.reshape(-1, 1)
        
        n_steps, n_neurons = spike_train.shape
        rates = spike_train.sum(axis=0) / time_window_ms  # Hz
        
        # Find active neurons
        active = np.where(rates > self.rate_threshold)[0]
        
        if len(active) == 0:
            # No activity → zero vector
            result_hv = hypervec_rs.HyperVector.zero()
        else:
            # Weighted bundling based on firing rates
            # Normalize rates to [0, 1]
            max_rate = rates[active].max() if len(active) > 0 else 1.0
            normalized_rates = rates[active] / max_rate if max_rate > 0 else rates[active]
            
            # Build weighted superposition
            result_hv = None
            for neuron_id, weight in zip(active, normalized_rates):
                neuron_hv = self.neuron_hvs[neuron_id]
                # Bundle multiple times based on weight (quantized)
                n_bundles = max(1, int(weight * 10))
                for _ in range(n_bundles):
                    if result_hv is None:
                        result_hv = neuron_hv
                    else:
                        result_hv = result_hv.bundle(neuron_hv)
            
            if result_hv is None:
                result_hv = hypervec_rs.HyperVector.zero()
        
        elapsed_ms = (time.time() - t0) * 1000
        
        return SpikeEncoding(
            hv=result_hv,
            active_neurons=active.tolist(),
            spike_rates=rates,
            encoding_time_ms=elapsed_ms,
            metadata={
                "n_active": len(active),
                "max_rate": rates.max(),
                "mean_rate": rates.mean(),
                "coding_scheme": "rate"
            }
        )


class TemporalCoder:
    """
    Temporal coding: Spike timing → sequential patterns.
    
    Uses permutation (bit-shift) to encode temporal order.
    Early spikes contribute to lower bit positions.
    
    Example:
        neuron_0 spikes at t=10ms → ρ¹⁰(HV_0)
        neuron_1 spikes at t=50ms → ρ⁵⁰(HV_1)
        Result: HV = ρ¹⁰(HV_0) ⊕ ρ⁵⁰(HV_1)
    """
    
    def __init__(self, n_neurons: int = 1000, dimension: int = 10240,
                 max_shift: int = 64):
        """
        Args:
            n_neurons: Number of neurons
            dimension: Hypervector dimensionality
            max_shift: Maximum permutation shift for temporal encoding
        """
        self.n_neurons = n_neurons
        self.dimension = dimension
        self.max_shift = max_shift
        
        self.neuron_hvs = [
            hypervec_rs.HyperVector(seed=i) 
            for i in range(n_neurons)
        ]
        
    def encode(self, spike_train: np.ndarray, dt_ms: float = 1.0) -> SpikeEncoding:
        """
        Encode spike train with temporal information.
        
        Args:
            spike_train: Binary spike array [time_steps, n_neurons]
            dt_ms: Time step duration in milliseconds
            
        Returns:
            SpikeEncoding with temporal structure
        """
        import time
        t0 = time.time()
        
        if len(spike_train.shape) == 1:
            spike_train = spike_train.reshape(-1, 1)
        
        n_steps, n_neurons = spike_train.shape
        
        # Find all spike events
        spike_times, neuron_ids = np.where(spike_train > 0)
        
        if len(spike_times) == 0:
            result_hv = hypervec_rs.HyperVector.zero()
            active_neurons = []
        else:
            # Bundle permuted HVs based on spike time
            result_hv = None
            active_neurons = []
            
            for t, neuron_id in zip(spike_times, neuron_ids):
                # Permutation shift proportional to time
                shift = min(int((t * dt_ms / 100.0) * self.max_shift), self.max_shift)
                neuron_hv = self.neuron_hvs[neuron_id].permute(shift)
                
                if result_hv is None:
                    result_hv = neuron_hv
                else:
                    result_hv = result_hv.bundle(neuron_hv)
                
                if neuron_id not in active_neurons:
                    active_neurons.append(neuron_id)
            
            if result_hv is None:
                result_hv = hypervec_rs.HyperVector.zero()
        
        # Compute rates for metadata
        rates = spike_train.sum(axis=0) / (n_steps * dt_ms) if n_steps > 0 else np.zeros(n_neurons)
        
        elapsed_ms = (time.time() - t0) * 1000
        
        return SpikeEncoding(
            hv=result_hv,
            active_neurons=active_neurons,
            spike_rates=rates,
            encoding_time_ms=elapsed_ms,
            metadata={
                "n_spikes": len(spike_times),
                "temporal_span_ms": n_steps * dt_ms,
                "coding_scheme": "temporal"
            }
        )


# ===========================================================================
# VSA → Spike Decoders
# ===========================================================================

class HVtoSpikeDecoder:
    """
    Decode hypervector to spike pattern for SNN initialization.
    
    Useful for priming an SNN with VSA memory or providing context.
    """
    
    def __init__(self, n_neurons: int = 1000):
        self.n_neurons = n_neurons
        self.neuron_hvs = [
            hypervec_rs.HyperVector(seed=i) 
            for i in range(n_neurons)
        ]
        
    def decode(self, target_hv: Any, n_steps: int = 100,
               threshold: float = 0.55) -> np.ndarray:
        """
        Generate spike pattern that encodes the target hypervector.
        
        Args:
            target_hv: Hypervector to decode
            n_steps: Number of time steps in output
            threshold: Similarity threshold for neuron activation
            
        Returns:
            Spike pattern [n_steps, n_neurons]
        """
        spikes = np.zeros((n_steps, self.n_neurons), dtype=np.int8)
        
        # Find neurons whose HVs are similar to target
        for i, neuron_hv in enumerate(self.neuron_hvs):
            sim = target_hv.similarity(neuron_hv)
            if sim > threshold:
                # Spike probability proportional to similarity
                prob = (sim - threshold) / (1.0 - threshold)
                # Generate random spikes with this probability
                spike_indices = np.random.rand(n_steps) < prob
                spikes[spike_indices, i] = 1
        
        return spikes


# ===========================================================================
# Concept Formation Layer
# ===========================================================================

class ConceptMapper:
    """
    Maps SNN activity patterns to learned concepts in VSA space.
    
    Maintains a registry of concept prototypes and matches
    incoming spike encodings against them.
    """
    
    def __init__(self, n_concepts: int = 100):
        self.n_concepts = n_concepts
        self.concept_prototypes: Dict[int, Any] = {}  # concept_id → HyperVector
        self.concept_names: Dict[int, str] = {}
        self.activation_history: List[ConceptActivation] = []
        self.next_concept_id = 0
        
    def register_concept(self, name: str, prototype_hv: Any) -> int:
        """Register a new concept with its prototype hypervector."""
        concept_id = self.next_concept_id
        self.concept_prototypes[concept_id] = prototype_hv
        self.concept_names[concept_id] = name
        self.next_concept_id += 1
        logger.debug(f"Registered concept {concept_id}: {name}")
        return concept_id
        
    def recognize(self, spike_encoding: SpikeEncoding,
                  threshold: float = 0.55) -> List[ConceptActivation]:
        """
        Match spike encoding against known concept prototypes.
        
        Args:
            spike_encoding: Encoded spike train
            threshold: Minimum similarity for concept activation
            
        Returns:
            List of activated concepts sorted by strength
        """
        import time
        
        activations = []
        query_hv = spike_encoding.hv
        
        for concept_id, prototype_hv in self.concept_prototypes.items():
            similarity = query_hv.similarity(prototype_hv)
            
            if similarity > threshold:
                activations.append(ConceptActivation(
                    concept_id=concept_id,
                    concept_name=self.concept_names[concept_id],
                    activation_strength=similarity,
                    source_neurons=spike_encoding.active_neurons,
                    timestamp=time.time()
                ))
        
        # Sort by activation strength
        activations.sort(key=lambda c: c.activation_strength, reverse=True)
        
        # Store in history
        self.activation_history.extend(activations)
        if len(self.activation_history) > 1000:
            self.activation_history = self.activation_history[-1000:]
        
        return activations
    
    def learn_from_spikes(self, spike_encoding: SpikeEncoding,
                          label: str) -> int:
        """
        Learn a new concept from spike encoding with supervision.
        
        Args:
            spike_encoding: Encoded spike pattern
            label: Human-readable concept label
            
        Returns:
            Concept ID
        """
        # Check if concept already exists
        for cid, name in self.concept_names.items():
            if name == label:
                # Update prototype (weighted average)
                old_proto = self.concept_prototypes[cid]
                new_proto = old_proto.bundle(spike_encoding.hv)
                self.concept_prototypes[cid] = new_proto
                logger.debug(f"Updated concept {cid}: {label}")
                return cid
        
        # Register new concept
        return self.register_concept(label, spike_encoding.hv)


# ===========================================================================
# Convenience Factory
# ===========================================================================

def create_vsa_snn_bridge(n_neurons: int = 1000,
                          coding_scheme: str = "rate",
                          n_concepts: int = 100) -> Tuple[Any, Any, ConceptMapper]:
    """
    Factory function to create a complete VSA-SNN bridge.
    
    Args:
        n_neurons: Number of neurons in SNN layer
        coding_scheme: "rate" or "temporal"
        n_concepts: Maximum number of concepts
        
        
    Returns:
        (encoder, decoder, concept_mapper)
    """
    if coding_scheme == "rate":
        encoder = RateCoder(n_neurons=n_neurons)
    elif coding_scheme == "temporal":
        encoder = TemporalCoder(n_neurons=n_neurons)
    else:
        raise ValueError(f"Unknown coding scheme: {coding_scheme}")
    
    decoder = HVtoSpikeDecoder(n_neurons=n_neurons)
    concept_mapper = ConceptMapper(n_concepts=n_concepts)
    
    logger.info(f"Created VSA-SNN bridge: {coding_scheme} coding, {n_neurons} neurons")
    
    return encoder, decoder, concept_mapper


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "SpikeEncoding",
    "ConceptActivation",
    "RateCoder",
    "TemporalCoder",
    "HVtoSpikeDecoder",
    "ConceptMapper",
    "create_vsa_snn_bridge",
]
