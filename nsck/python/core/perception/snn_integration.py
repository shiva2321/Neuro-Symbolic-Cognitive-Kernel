"""
SNN Perception Integration for CognitiveEngine
===============================================

Provides adapters to integrate SNN perception module with:
- Global Workspace Theory (GWT) broadcast system
- Semantic memory concept registration
- Hebbian association learning
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from python.core.perception.snn_perception import SNNPerceptionModule
from python.core.reasoning.global_workspace import WorkspaceModule
from python.core.memory.semantic_memory import SemanticMemory


class SNNWorkspaceAdapter(WorkspaceModule):
    """
    Adapter to connect SNN perception module to Global Workspace
    
    Responsibilities:
    - Process sensory input through SNN pipeline
    - Submit perceptual concepts to workspace competition
    - Receive broadcasts and update perceptual priors
    - Track perception statistics
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        snn_size: int = 256,
        hv_dimension: int = 1024,
        semantic_memory: Optional[SemanticMemory] = None
    ):
        self.snn_module = SNNPerceptionModule(
            input_dim=input_dim,
            snn_size=snn_size,
            hv_dimension=hv_dimension,
            n_concepts=100,
            encoding_mode="rate",
            simulation_time_ms=50.0
        )
        
        self.semantic_memory = semantic_memory
        self.last_percept = None
        self.last_activation = 0.0
        self.broadcast_history = []
        
    def perceive(self, sensory_data: np.ndarray, learn: bool = True) -> Dict:
        """
        Process sensory input through SNN pipeline
        
        Args:
            sensory_data: Raw sensory vector (input_dim,)
            learn: Whether to update associations
            
        Returns:
            perception_result: Dict with concept_hv, concept_id, strength, etc.
        """
        result = self.snn_module.perceive(sensory_data, learn=learn)
        
        self.last_percept = result
        self.last_activation = result["strength"]
        
        # Register new concepts in semantic memory if available
        if self.semantic_memory and result["concept_id"] not in self.semantic_memory.concept_hvs:
            concept_name = f"percept_{result['concept_id']}"
            self.semantic_memory.add_concept(
                concept_name,
                properties={"type": "sensory_percept", "spike_count": result["n_spikes"]},
                hv_override=result["concept_hv"]
            )
        
        return result
    
    def get_coalition_proposal(self) -> Optional[Dict]:
        """
        Generate coalition proposal for Global Workspace competition
        
        Returns:
            None if no recent percept, otherwise dict with:
                - source: "snn_perception"
                - content: concept_hv
                - base_salience: strength of recognition
                - metadata: spike counts, timing, etc.
        """
        if self.last_percept is None:
            return None
        
        # Salience based on recognition strength + spike activity
        spike_rate = self.last_percept["n_spikes"] / (self.snn_module.snn_size * 0.05)  # Spikes per neuron per ms
        salience = (self.last_percept["strength"] + min(spike_rate, 1.0)) / 2.0
        
        return {
            "source": "snn_perception",
            "content": self.last_percept["concept_hv"],
            "base_salience": salience,
            "metadata": {
                "concept_id": self.last_percept["concept_id"],
                "n_spikes": self.last_percept["n_spikes"],
                "active_neurons": len(self.last_percept["active_neurons"]),
                "processing_time_ms": self.last_percept["processing_time_ms"]
            }
        }
    
    def receive_broadcast(self, content: Any):
        """
        Receive GWT broadcast and update perceptual expectations
        
        When another module wins consciousness, use that to prime/bias
        the SNN perception for contextual processing
        """
        self.broadcast_history.append(content)
        
        # Keep only recent broadcasts
        if len(self.broadcast_history) > 10:
            self.broadcast_history.pop(0)
        
        # TODO: Use broadcasts to modulate SNN dynamics (attention, priming)
        #       - Strengthen connections to neurons associated with broadcast content
        #       - Increase excitability for related concepts
        #       - This implements top-down attention
    
    def get_stats(self) -> Dict:
        """Get perception statistics for telemetry"""
        snn_stats = self.snn_module.get_stats()
        return {
            "module": "SNN_Perception",
            "last_activation": self.last_activation,
            "broadcasts_received": len(self.broadcast_history),
            **snn_stats
        }


def add_snn_perception_to_engine(
    engine,
    input_dim: int = 64,
    snn_size: int = 256,
    hv_dimension: int = 1024
):
    """
    Add SNN perception module to existing CognitiveEngine
    
    Args:
        engine: CognitiveEngine instance
        input_dim: Dimensionality of sensory input
        snn_size: Number of LIF neurons in SNN layer
        hv_dimension: Hypervector dimension (should match engine's VSA dimension)
        
    Usage:
        engine = CognitiveEngine(config)
        add_snn_perception_to_engine(engine, input_dim=64, snn_size=256)
        
        # Now engine has engine.snn_perception
        result = engine.snn_perception.perceive(sensory_data)
    """
    # Create adapter
    adapter = SNNWorkspaceAdapter(
        input_dim=input_dim,
        snn_size=snn_size,
        hv_dimension=hv_dimension,
        semantic_memory=engine.semantic_memory
    )
    
    # Register with Global Workspace
    engine.global_workspace.register_module("snn_perception", adapter)
    
    # Attach to engine for direct access
    engine.snn_perception = adapter
    
    # Add to stats tracking
    engine.stats["snn_perceptions"] = 0
    
    print(f"✅ SNN Perception integrated: {snn_size} neurons, {hv_dimension}-bit HVs")
    return adapter


# ============================================================================
# Example Integration
# ============================================================================

if __name__ == "__main__":
    print("=== SNN Perception Integration Test ===\n")
    
    # Mock CognitiveEngine components for testing
    from python.core.reasoning.global_workspace import GlobalWorkspace
    from python.core.memory.semantic_memory import SemanticMemory
    
    class MockEngine:
        def __init__(self):
            self.global_workspace = GlobalWorkspace()
            self.semantic_memory = SemanticMemory()
            self.stats = {}
    
    engine = MockEngine()
    
    # Add SNN perception
    snn_adapter = add_snn_perception_to_engine(
        engine,
        input_dim=64,
        snn_size=128,
        hv_dimension=1024
    )
    
    # Test perception
    sensory_input = np.random.randn(64) * 0.5
    result = snn_adapter.perceive(sensory_input)
    
    print(f"\nPerception Result:")
    print(f"  Concept ID: {result['concept_id']}")
    print(f"  Spikes: {result['n_spikes']}")
    print(f"  Processing: {result['processing_time_ms']:.3f} ms")
    
    # Get coalition proposal
    proposal = snn_adapter.get_coalition_proposal()
    print(f"\nCoalition Proposal:")
    print(f"  Source: {proposal['source']}")
    print(f"  Salience: {proposal['base_salience']:.3f}")
    print(f"  Active neurons: {proposal['metadata']['active_neurons']}")
    
    # Test broadcast receiving
    snn_adapter.receive_broadcast("ACTION_MOVE")
    print(f"\nBroadcasts received: {len(snn_adapter.broadcast_history)}")
    
    # Stats
    stats = snn_adapter.get_stats()
    print(f"\nStatistics:")
    print(f"  Avg latency: {stats['avg_latency_ms']:.3f} ms")
    print(f"  Concepts learned: {stats['n_concepts_learned']}")
    
    print("\n✅ SNN Integration: WORKING")
