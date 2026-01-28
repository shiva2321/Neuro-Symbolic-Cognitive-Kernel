"""
NCGN v7 Brain Module

The unified cognitive architecture coordinator.

Integrates:
- System 1: Fast associative propagation (matrix ops)
- System 2: Slow deliberative reasoning (LLM)
- Learning: 3-Factor Hebbian plasticity
- Semantics: Embedding-based entry points

This is the main interface for interacting with NCGN v7.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .topology import GraphTopology
from .state import CognitiveState
from .engine import PropagationEngine
from .learner import HebbianLearner, RewardModulator
from .embeddings import SemanticLayer
# from .reasoner import LLMReasoner, MockReasoner  # Deprecated in v2.0

# v2.0 Components
from .confidence import ConfidenceTracker
from .conflicts import ConflictDetector, ConflictType
from .decision import DecisionEngine
from .linguistic_processor import LinguisticProcessor
from .formatter import NaturalLanguageFormatter

# Phase 2: Perception & Memory
from .memory.vsa_core import VSAEngine
from .memory.associative import AssociativeMemory
from .perception.snn_encoder import SNNEncoder
from .perception.bridge import VSABridge

# Phase 3: Control
from .control.act_inf import ActiveInferenceController
from .control.decider import Decider

# Phase 4: Sleep
from .memory.consolidation import ShortTermMemory, SleepManager

class Brain:
    """
    The unified cognitive architecture (NCGN v2.0).
    
    Coordinates:
    - GraphTopology: Rustworkx-based graph structure
    - PropagationEngine: System 1 dynamics
    - HebbianLearner: Associative learning
    - Logic Core: Confidence, Conflicts, Decision Engine
    - Peripherals: LinguisticProcessor, NaturalLanguageFormatter
    """
    
    def __init__(
        self,
        config: Config = None,
        llm_model_path: Optional[str] = None,
        use_embeddings: bool = True,
        use_mock_reasoner: bool = False
    ):
        self.config = config or DEFAULT_CONFIG
        
        # Override LLM path from config if provided
        if llm_model_path:
            self.config.llm_model_path = llm_model_path
        
        # Core components
        self.topology = GraphTopology()
        self.state = CognitiveState(self.config)
        self.engine = PropagationEngine(self.state, self.topology, self.config)
        
        # v2.0: Confidence System
        self.confidence = ConfidenceTracker()
        
        # Learner
        self.learner = HebbianLearner(
            self.state, 
            self.topology, 
            self.config,
            confidence_tracker=self.confidence
        )
        self.modulator = RewardModulator()
        
        # v2.0: Logic Engines
        self.conflicts = ConflictDetector(self.topology, self.confidence)
        self.decision = DecisionEngine(self.confidence)
        
        # v2.0: Peripherals (Linguistic)
        self.linguist = None
        self.formatter = None
        
        if self.config.llm_model_path and not use_mock_reasoner:
            try:
                self.linguist = LinguisticProcessor(self.config.llm_model_path, self.config)
                self.formatter = NaturalLanguageFormatter(self.config.llm_model_path, self.config)
            except Exception as e:
                print(f"Warning: Could not initialize Linguistic peripherals: {e}")
        
        # Semantic layer (optional)
        self._semantics: Optional[SemanticLayer] = None
        self._use_embeddings = use_embeddings
        
        # Properties knowledge base
        self._properties: Dict[str, Dict[str, bool]] = {}
        
        # Interaction history
        self._history: List[Dict[str, Any]] = []

        # Phase 2: Perception-Memory Loop
        try:
            self.vsa = VSAEngine(dimensions=10000)
            device = self.vsa.device
            
            self.associative = AssociativeMemory(self.vsa)
            self.snn = SNNEncoder(output_size=128).to(device)
            self.bridge = VSABridge(input_dim=128, vsa_dim=10000, device=device)
            self._has_perception = True
            print(f"[Brain] Perception initialized on {device}")
            
            # Phase 3: Control
            # Simple setup: 1 Modality (ConceptID), 1 State Factor (HiddenConcept), 1 Control Factor (Action)
            # Obs dim = 100 (Arbitrary cap for prototype)
            self.act_inf = ActiveInferenceController(
                num_obs=[100], 
                num_states=[100], 
                num_controls=[3] # Wait, Look, Ask
            )
            self.decider = Decider(self.act_inf)
            
            # Phase 4: Sleep
            self.stm = ShortTermMemory(capacity=100)
            self.sleep_manager = SleepManager(self.stm, self)
            
        except Exception as e:
            print(f"Warning: Perception/Control modules failed to load: {e}")
            self._has_perception = False
    
    @property
    def semantics(self) -> Optional[SemanticLayer]:
        """Lazy-load semantic layer."""
        if not self._use_embeddings:
            return None
        
        if self._semantics is None:
            try:
                self._semantics = SemanticLayer(config=self.config)
            except ImportError:
                print("Warning: SentenceTransformers not available")
                self._use_embeddings = False
                return None
        
        return self._semantics

    # ==================== Concept Management ====================
    
    def add_concept(
        self,
        label: str,
        initial_energy: float = 0.0,
        threshold: float = None,
        properties: Dict[str, bool] = None
    ) -> int:
        idx = self.topology.add_concept(label)
        self.state.ensure_capacity(idx)
        
        if initial_energy > 0:
            self.state.set_activation(idx, initial_energy)
        
        if threshold is not None:
            self.state.thresholds[idx] = threshold
        
        if properties:
            self._properties[label] = properties
        
        # Compute and store embedding if semantics available
        if self.semantics is not None:
            try:
                embedding = self.semantics.encode_single(label)
                self.state.set_embedding(idx, embedding)
            except Exception:
                pass
        
        return idx
    
    def remove_concept(self, label: str) -> bool:
        idx = self.topology.remove_concept(label)
        if idx is None:
            return False
        
        self.state.zero_index(idx)
        self._properties.pop(label, None)
        return True
    
    def has_concept(self, label: str) -> bool:
        return self.topology.registry.has_label(label)
    
    def get_concept_energy(self, label: str) -> Optional[float]:
        idx = self.topology.registry.get_index(label)
        if idx is None:
            return None
        return float(self.state.activations[idx])
    
    def set_concept_energy(self, label: str, energy: float) -> bool:
        idx = self.topology.registry.get_index(label)
        if idx is None:
            return False
        self.state.set_activation(idx, energy)
        return True
    
    # ==================== Connection Management ====================
    
    def connect(
        self,
        source: str,
        target: str,
        weight: Optional[float] = None,
        create_nodes: bool = True
    ) -> bool:
        """Connect two concepts."""
        if not create_nodes:
            if not self.has_concept(source) or not self.has_concept(target):
                return False
        
        if weight is None:
            if self.semantics is not None:
                weight = self.semantics.compute_semantic_weight(source, target)
            else:
                weight = 0.5
        
        self.topology.add_connection(source, target, weight)
        return True

    def disconnect(self, source: str, target: str) -> bool:
        return self.topology.remove_connection(source, target)
    
    def get_connection_weight(self, source: str, target: str) -> Optional[float]:
        return self.topology.get_edge_weight(source, target)
    
    def set_connection_weight(self, source: str, target: str, weight: float) -> bool:
        return self.topology.set_edge_weight(source, target, weight)

    
    # ==================== System 1: Propagation ====================
    
    # ==================== System 1: Propagation ====================
    
    def inject(self, label: str, energy: float) -> bool:
        """Inject energy into a concept."""
        return self.engine.inject_by_label(label, energy)
    
    def inject_multiple(self, injections: Dict[str, float]) -> int:
        """Inject energy into multiple concepts."""
        return self.engine.inject_multiple(injections)
    
    def think(self, steps: int = 10) -> Dict[str, float]:
        """Run System 1 propagation."""
        return self.engine.propagate(steps)
    
    def get_active_concepts(self, threshold: float = None) -> Dict[str, float]:
        """Get currently active concepts."""
        return self.engine.get_active_concepts(threshold)
    
    def get_top_concepts(self, k: int = 10) -> Dict[str, float]:
        """Get top K most active concepts."""
        return self.engine.get_top_k_active(k)
    
    # ==================== System 2: Reasoning (v2.0 Logic) ====================
    
    def _apply_graph_update(
        self, 
        update: Any
    ) -> Dict[str, Any]:
        """
        Apply a knowledge graph update with Conflict checking.
        """
        results = {
            "accepted_nodes": [],
            "accepted_edges": [],
            "conflicts": [],
            "rejected": []
        }
        
        # 1. Process Nodes (Safe, usually)
        for node in update.new_nodes:
            # Check implicit conflicts? (e.g. node already exists but with different property)
            # For now, just add/merge.
            self.add_concept(node.label, initial_energy=0.3, properties=node.properties)
            results["accepted_nodes"].append(node.label)
        
        # 2. Process Edges (Critical for Conflicts)
        for edge in update.new_edges:
            # Check for conflict
            conflict = self.conflicts.check_conflict(edge.source, edge.relation_type, edge.target)
            
            # Decide
            decision = self.decision.decide_acceptance(conflict)
            
            if decision == "accept":
                self.connect(edge.source, edge.target, edge.weight)
                # Reinforce if it's new
                if self.confidence:
                    self.confidence.reinforce(edge.source, edge.target, source_type="user_input")
                results["accepted_edges"].append((edge.source, edge.target))
            
            elif decision == "curiosity":
                results["conflicts"].append(conflict)
            
            else: # reject
                results["rejected"].append(edge)
        
        return results

    def ask(self, question: str, propagation_steps: int = 5) -> str:
        """
        Ask the brain a question.
        v2.0: Uses Formatter to explain active state.
        """
        # 1. Association (System 1)
        if self.semantics is not None:
            # Find entry points
            entry_points = self.semantics.find_entry_points(question, self.topology, top_k=5)
            for label, sim in entry_points:
                self.inject(label, sim * 0.5)
        
        # Propagate
        active = self.engine.propagate(propagation_steps)
        
        # 2. Format Response
        if self.formatter:
            state = {
                "user_input": question,
                "active_concepts": active,
                "decision": "ANSWER"
            }
            return self.formatter.format_response(state)
        
        return f"Active concepts: {', '.join(list(active.keys())[:10])}"
    
    # ==================== Learning ====================
    
    def learn(self, reward: float) -> int:
        """Apply reward-modulated learning."""
        # Calculate RPE
        rpe = self.modulator.calculate_signal(reward)
        # Apply to synapses
        return self.learner.apply_reward(rpe)
    
    def weaken(self, source: str, target: str, factor: float = 0.5) -> bool:
        """Apply LTD to a specific connection."""
        return self.learner.apply_targeted_ltd(source, target, factor)
    
    def strengthen(self, source: str, target: str, factor: float = 1.5) -> bool:
        """Apply LTP to a specific connection."""
        return self.learner.apply_targeted_ltp(source, target, factor)
    
    # ==================== Properties (System 2 Constraints) ====================
    
    def set_property(self, label: str, prop: str, value: bool) -> None:
        """Set a property for a concept."""
        if label not in self._properties:
            self._properties[label] = {}
        self._properties[label][prop] = value
    
    def get_property(self, label: str, prop: str) -> Optional[bool]:
        """Get a property value."""
        return self._properties.get(label, {}).get(prop)
    
    def get_all_properties(self, label: str) -> Dict[str, bool]:
        """Get all properties for a concept."""
        return self._properties.get(label, {}).copy()

    # ==================== Full Cognitive Cycle ====================
    
    def process_input(
        self,
        user_input: str,
        propagation_steps: int = 10,
        apply_updates: bool = True
    ) -> Dict[str, Any]:
        """
        Full cognitive cycle (v2.0):
        1. Parse (Linguistic)
        2. Associate (System 1)
        3. Logic Check (Conflict/Decision)
        4. Apply & Respond
        """
        result = {
            "input": user_input,
            "entry_points": [],
            "active_before": {},
            "active_after": {},
            "graph_updates": {},
            "formatted_response": "",
            "surprise": 0.0,
        }
        
        # Step 1: Semantic Mapping & Injection
        if self.semantics is not None:
            entry_points = self.semantics.find_entry_points(
                user_input, self.topology, top_k=5, min_similarity=0.2
            )
            result["entry_points"] = entry_points
            for label, similarity in entry_points:
                self.inject(label, similarity * 0.5)
        
        result["active_before"] = self.get_active_concepts()
        
        # Step 2: System 1 Propagation
        result["active_after"] = self.engine.propagate(propagation_steps)
        result["surprise"] = self.engine.last_surprise
        
        # Step 3: Linguistic Parsing (if ready)
        if self.linguist and apply_updates:
            updates = self.linguist.extract_triplets(user_input)
            
            if updates:
                # Step 4: Logic Decision & Application
                update_results = self._apply_graph_update(updates)
                result["graph_updates"] = update_results
                
                # Determine overall decision state for formatter
                decision_state = "ACCEPT"
                if update_results["conflicts"]:
                    decision_state = "CURIOSITY"
                elif update_results["rejected"]:
                    decision_state = "REJECT"
                
                # Step 5: Format Response
                if self.formatter:
                    fmt_state = {
                        "user_input": user_input,
                        "active_concepts": result["active_after"],
                        "decision": decision_state,
                        "conflict": update_results["conflicts"][0] if update_results["conflicts"] else None
                    }
                    result["formatted_response"] = self.formatter.format_response(fmt_state)
        
        # Fallback response if no updates or no formatter
        if not result["formatted_response"] and self.formatter:
             # Just a chat/query
             fmt_state = {
                "user_input": user_input,
                "active_concepts": result["active_after"],
                "decision": "CHAT"
            }
             result["formatted_response"] = self.formatter.format_response(fmt_state)

        # Record History
        self._history.append({
            "input": user_input,
            "active_count": len(result["active_after"]),
            "updates": result.get("graph_updates"),
        })
        
        return result

    # ==================== Phase 2: Perception Loop ====================

    def process_perception(self, image_input: Any) -> Dict[str, Any]:
        """
        Run the Fast Loop (System 1 Perception):
        Image -> SNN (Spikes) -> Bridge (Hypervector) -> VSA (Concept).
        
        Args:
            image_input: Tensor of shape (1, 1, 28, 28) or similar.
            
        Returns:
            Dict containing label, confidence, and vector details.
        """
        if not self._has_perception:
            return {"error": "Perception disabled"}
            
        # 1. Encode (SNN) -> Rate Vector
        # SNNEncoder expects (Batch, C, H, W). Ensure batch dim.
        if len(image_input.shape) == 3:
             image_input = image_input.unsqueeze(0)
             
        spike_rate = self.snn(image_input) 
        
        # 2. Bridge (Project) -> Hypervector
        hypervector = self.bridge(spike_rate)
        
        # 3. Query (VSA) -> Label & Confidence
        # flatten hypervector if batch size is 1
        hv_query = hypervector[0] if hypervector.shape[0] == 1 else hypervector
        
        label, score, confidence = self.associative.query(hv_query)
        
        # 4. Inject into System 1 (Graph)
        # We inject energy proportional to confidence if it exceeds noise floor
        # "Z < 2.0" is considered noise/unknown (R5)
        # 4. Inject into System 1 (Graph)
        # We inject energy proportional to confidence if it exceeds noise floor
        # "Z < 2.0" is considered noise/unknown (R5)
        if label != "None" and confidence > 2.0:
            # Inject energy. Research suggests confidence-weighted injection.
            # Z=2 -> 0.2, Z=10 -> 1.0?
            energy = min(0.1 * confidence, 1.0)
            success = self.inject(label, energy=energy)
            
            # Phase 4: Add to Short Term Memory for consolidation
            if hasattr(self, 'stm'):
                self.stm.add(hv_query, label, energy)
        
        # DEBUG
        print(f"Perception: Label={label}, Z={confidence:.2f}")
            
        return {
            "label": label,
            "similarity": score,
            "confidence_z": confidence,
            "injected_energy": energy if confidence > 2.0 else 0.0,
            "decision": None
        }

    def step(self, image_input: Any) -> Dict[str, Any]:
        """
        Execute one full cognitive time-step (System 1 + System 2).
        
        Flow:
        1. Perception (SNN->VSA)
        2. Decision (Decider)
           - If Confident: System 1 Propagate
           - If Ambiguous: System 2 Active Inference
        """
        # 1. Perception
        perc_result = self.process_perception(image_input)
        
        # 2. Decision
        if "error" not in perc_result:
            decision_str = self.decider.decide(perc_result)
            perc_result["decision"] = decision_str
            
            # If System 1 Reflex (implied by high confidence injection in process_perception)
            # We also run a graph tick
            if "SYSTEM_1" in decision_str:
                self.think(1)
                
        return perc_result
                
        return perc_result
        
    def sleep(self) -> Dict[str, int]:
        """
        Trigger Sleep Consolidation Cycle.
        Replays Short Term Memory -> Long Term Memory.
        """
        if hasattr(self, 'sleep_manager'):
            return self.sleep_manager.run_sleep_cycle()
        return {}
    
    # ==================== Utilities ====================
    
    def clear(self) -> None:
        """Reset the brain to empty state."""
        self.topology.clear()
        self.state.clear_all()
        self.engine.reset()
        self.learner.reset_stats()
        self.modulator.reset()
        self._properties.clear()
        self._history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "topology": self.topology.get_stats(),
            "state": self.state.get_stats(),
            "engine": self.engine.get_stats(),
            "learner": self.learner.get_stats(),
            "modulator": self.modulator.get_stats(),
            "semantics": self.semantics.get_stats() if self.semantics else None,
            "linguist": True if self.linguist else False,
            "properties_count": len(self._properties),
            "history_length": len(self._history),
        }
    
    def __repr__(self) -> str:
        stats = self.topology.get_stats()
        return (
            f"Brain(nodes={stats['num_nodes']}, "
            f"edges={stats['num_edges']}, "
            f"linguist={'yes' if self.linguist else 'no'})"
        )
