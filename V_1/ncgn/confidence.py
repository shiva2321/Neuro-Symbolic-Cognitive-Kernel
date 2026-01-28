"""
NCGN v2.0 Confidence & Verification System

Tracks the reliability of knowledge in the graph.
Confidence is derived from:
1. Repetition: How often has this been observed?
2. Consistency: Is it stable over time?
3. Source: Is it from a reliable source (training data > user > guess)?
"""

from typing import Dict, Tuple, Optional
import time

class ConfidenceTracker:
    def __init__(self):
        # Edge confidence: (source, target) -> float
        self.edge_confidence: Dict[Tuple[str, str], float] = {}
        
        # Edge provenance: (source, target) -> {source_type, count, last_verified}
        self.edge_provenance: Dict[Tuple[str, str], Dict] = {}
        
        # Node confidence (concept stability)
        self.node_confidence: Dict[str, float] = {}

    def get_confidence(self, source: str, target: str) -> float:
        """Get confidence score for an edge (0.0 - 1.0)."""
        return self.edge_confidence.get((source, target), 0.0)

    def reinforce(self, source: str, target: str, source_type: str = "user") -> float:
        """
        Reinforce a connection (Hebbian or Explicit).
        Returns new confidence score.
        """
        key = (source, target)
        
        if key not in self.edge_provenance:
            self.edge_provenance[key] = {
                "count": 0,
                "source_type": source_type,
                "first_seen": time.time(),
                "last_verified": 0
            }
            self.edge_confidence[key] = 0.1  # Initial low confidence
        
        meta = self.edge_provenance[key]
        meta["count"] += 1
        meta["last_verified"] = time.time()
        
        # Confidence Formula:
        # Base confidence varies by source type
        base = 0.5 if source_type == "training" else 0.2
        
        # Repetition boost (logarithmic-ish)
        # 1 -> 0.2
        # 5 -> 0.7
        # 10 -> 0.9
        count_factor = min(0.8, meta["count"] * 0.1)
        
        new_conf = base + count_factor
        
        # Clamp to 0.0 - 1.0
        new_conf = min(max(new_conf, 0.0), 1.0)
        
        self.edge_confidence[key] = new_conf
        return new_conf

    def decay(self, rate: float = 0.99) -> None:
        """Apply passive decay to confidence (forgetting curve)."""
        for key in self.edge_confidence:
            # Don't decay foundational knowledge (training data) too much
            if self.edge_provenance[key]["source_type"] == "training":
                 self.edge_confidence[key] *= (1 - (1-rate)*0.1)
            else:
                 self.edge_confidence[key] *= rate

    def get_stats(self) -> Dict:
        return {
            "tracked_edges": len(self.edge_confidence),
            "avg_confidence": sum(self.edge_confidence.values()) / max(1, len(self.edge_confidence))
        }
