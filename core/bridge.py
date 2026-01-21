"""
NCGN Bridge Module - The System 1/2 Interface

Implements the Surprise Monitor that bridges reactive (System 1)
and deliberative (System 2) processing.

Key Concepts:
- Set-based surprise calculation
- Violated expectations trigger System 2
- Novelty suppresses false alarms on new tokens
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from .memory import GraphMemory, ConceptNode


class SurpriseMonitor:
    """
    Monitors the gap between prediction and observation.
    
    Surprise is NOT triggered by novel inputs - only by
    violated high-confidence expectations.
    """
    
    def __init__(self, memory: GraphMemory):
        self.memory = memory
        
        # Prediction state (captured before observation)
        self.prediction_set: Dict[str, Tuple[float, float]] = {}  # node_id -> (energy, confidence)
        
        # Configuration
        self.novelty_threshold = 0.5  # Below this, node is considered "new"
    
    def capture_predictions(self, active_nodes: Set[str]):
        """
        Capture current predictions before new input arrives.
        
        Stores (energy, confidence) for each active node.
        """
        self.prediction_set.clear()
        
        for node_id in active_nodes:
            node = self.memory.get_node(node_id)
            if node and node.energy > 0:
                confidence = self._get_prediction_confidence(node_id)
                self.prediction_set[node_id] = (node.energy, confidence)
    
    def calculate_surprise(
        self,
        observed_nodes: Set[str],
        observed_energies: Dict[str, float],
        debug: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate surprise based on prediction-observation mismatch.
        
        PATCHED v5.1: Uses Root Mean Square (sqrt) for proper Euclidean distance.
        
        Formula:
        S = sqrt(Σ_n∈Predicted ((E_pred(n) - E_obs(n)) × Confidence(n))²)
        
        Returns:
            Tuple of (total_surprise, per_node_surprise)
        """
        per_node_surprise: Dict[str, float] = {}
        sum_of_squares = 0.0
        
        if debug:
            print("\n[Bridge] Calculating Surprise...")
        
        # Check each prediction
        for node_id, (pred_energy, confidence) in self.prediction_set.items():
            obs_energy = observed_energies.get(node_id, 0.0)
            
            # Only penalize missing expectations (positive diff)
            diff = pred_energy - obs_energy
            if diff > 0:
                node = self.memory.get_node(node_id)
                
                # Skip if node is novel (hasn't been learned yet)
                if node and node.novelty_score > self.novelty_threshold:
                    continue
                
                penalty = diff * confidence
                surprise_contrib = penalty ** 2
                per_node_surprise[node_id] = surprise_contrib
                sum_of_squares += surprise_contrib
                
                if debug and penalty > 0.1:
                    print(f"  > Node '{node_id}': Expected {pred_energy:.2f}, Got {obs_energy:.2f}, Conf {confidence:.2f} -> Penalty {penalty:.2f}")
        
        # CRITICAL FIX v5.1: Apply sqrt for Root Mean Square (Euclidean distance)
        # This boosts small values: sqrt(0.52) = 0.72, which triggers the gate.
        total_surprise = math.sqrt(sum_of_squares)
        
        # Cap at 1.0
        total_surprise = min(1.0, total_surprise)
        
        if debug:
            print(f"[Bridge] Total Surprise (RMS): {total_surprise:.4f}")
        
        return total_surprise, per_node_surprise
    
    def _get_prediction_confidence(self, node_id: str) -> float:
        """
        Get confidence of prediction for a node.
        
        Based on the strength of synapses that activated this node.
        """
        incoming = self.memory.get_incoming(node_id)
        if not incoming:
            return 0.0
        
        # Weight by synapse confidence
        total_conf = sum(synapse.confidence for _, synapse in incoming)
        return min(1.0, total_conf / len(incoming))
    
    def get_violated_expectations(
        self,
        observed_energies: Dict[str, float]
    ) -> List[Tuple[str, float, float]]:
        """
        Get list of expectations that were violated.
        
        Returns:
            List of (node_id, expected_energy, observed_energy)
        """
        violations = []
        
        for node_id, (pred_energy, confidence) in self.prediction_set.items():
            obs_energy = observed_energies.get(node_id, 0.0)
            
            # Consider violated if prediction was significant and not met
            if pred_energy > 0.1 and obs_energy < pred_energy * 0.5:
                if confidence > 0.3:  # Only report confident predictions
                    violations.append((node_id, pred_energy, obs_energy))
        
        return violations
    
    def get_unexpected_observations(
        self,
        observed_nodes: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Get observations that were not predicted.
        
        Returns:
            List of (node_id, novelty_score)
        """
        unexpected = []
        predicted_ids = set(self.prediction_set.keys())
        
        for node_id in observed_nodes:
            if node_id not in predicted_ids:
                node = self.memory.get_node(node_id)
                if node:
                    unexpected.append((node_id, node.novelty_score))
        
        return unexpected


class PredictionBuffer:
    """
    Rolling buffer of predictions for temporal surprise.
    
    Allows System 2 to diagnose "What did we expect to happen?"
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Tuple[int, Dict[str, Tuple[float, float]]]] = []
    
    def record(self, tick: int, predictions: Dict[str, Tuple[float, float]]):
        """Record predictions for a tick."""
        self.history.append((tick, predictions.copy()))
        
        # Trim old history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_predictions_at(self, tick: int) -> Optional[Dict[str, Tuple[float, float]]]:
        """Get predictions from a specific tick."""
        for t, preds in self.history:
            if t == tick:
                return preds
        return None
    
    def get_recent_predictions(self, n: int = 3) -> List[Tuple[int, Dict[str, Tuple[float, float]]]]:
        """Get the N most recent predictions."""
        return self.history[-n:]
