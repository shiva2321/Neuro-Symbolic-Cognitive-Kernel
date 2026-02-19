"""
SecurityMonitor - Anomaly Detection Module
===========================================

Example module demonstrating safety and anomaly detection.

Features:
- Monitors workspace state for unusual patterns
- Vetoes dangerous actions
- Tracks safety violations
- Reports security telemetry

Use Case:
When NSCK is controlling a robot or game agent, SecurityMonitor can veto
actions that violate safety constraints (e.g., cliff edges, forbidden zones).

Author: NSCK SDK Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import hashlib

from global_workspace import WorkspaceModule, Coalition
import hypervec_shim as hv


@dataclass
class SafetyRule:
    """
    A safety constraint that can veto actions.
    
    Attributes:
        pattern_hv: Hypervector representing dangerous state pattern
        threshold: Similarity threshold for triggering (>= this = dangerous)
        description: Human-readable rule description
        severity: Priority level (1-10, higher = more critical)
    """
    pattern_hv: np.ndarray
    threshold: float
    description: str
    severity: int = 5


class SecurityMonitor(WorkspaceModule):
    """
    Monitors workspace state and vetoes dangerous actions.
    
    This module demonstrates:
    - Pattern matching with hypervectors
    - Veto mechanisms (negative bids)
    - Safety constraint enforcement
    - Telemetry for debugging
    
    Example:
        # Create safety rules
        monitor = SecurityMonitor()
        
        # Add rule: Veto actions near cliffs
        cliff_pattern = encode("near_cliff")
        monitor.add_rule(SafetyRule(
            pattern_hv=cliff_pattern,
            threshold=0.6,
            description="Too close to cliff edge",
            severity=9
        ))
        
        # Register with CognitiveEngine
        engine.register_module(monitor)
    """
    
    __author__ = "NSCK SDK Team"
    __version__ = "1.0.0"
    
    def __init__(self, base_bid: float = -100.0):
        """
        Initialize SecurityMonitor.
        
        Args:
            base_bid: Negative bid value for vetoes (default -100.0)
        """
        self.base_bid = base_bid
        self.rules: List[SafetyRule] = []
        
        # Telemetry
        self.total_proposals = 0
        self.total_vetoes = 0
        self.violations_by_rule: Dict[str, int] = {}
        self.last_veto_reason = ""
    
    def add_rule(self, rule: SafetyRule):
        """Add a safety rule to monitor."""
        self.rules.append(rule)
        self.violations_by_rule[rule.description] = 0
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        """
        Evaluate state for safety violations.
        
        Returns a VETO coalition (negative bid) if state matches dangerous pattern.
        
        Args:
            state_hv: Current workspace state hypervector (10240-bit np.ndarray)
            
        Returns:
            Coalition with negative bid if unsafe, None if safe
        """
        self.total_proposals += 1
        
        # Check each safety rule
        for rule in self.rules:
            # Compute similarity using XOR + bit count
            diff = np.bitwise_xor(state_hv, rule.pattern_hv)
            hamming_dist = np.sum(diff)
            sim = 1.0 - (hamming_dist / len(state_hv))
            
            if sim >= rule.threshold:
                # SAFETY VIOLATION DETECTED
                self.total_vetoes += 1
                self.violations_by_rule[rule.description] += 1
                self.last_veto_reason = rule.description
                
                # Negative salience for veto (scaled by severity)
                # base_salience in [0.0, 1.0], so we use low values to indicate veto
                veto_salience = 0.0  # Minimum salience = veto
                confidence = 1.0 - (rule.severity / 10.0)  # Higher severity = lower confidence allows veto
                
                return Coalition(
                    source="SecurityMonitor",
                    content={
                        "type": "veto",
                        "reason": rule.description,
                        "similarity": float(sim),
                        "threshold": rule.threshold,
                        "severity": rule.severity
                    },
                    base_salience=veto_salience,
                    relevance=0.0,
                    sender_confidence=confidence
                )
        
        # No violations - safe to proceed
        return None
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        """
        Security monitor doesn't learn from feedback (rules are fixed).
        
        In a more advanced implementation, this could:
        - Adjust thresholds based on false positives
        - Learn new dangerous patterns
        - Update severity scores
        """
        pass
    
    def receive_broadcast(self, content: Any):
        """
        React to global broadcasts.
        
        Could be used for:
        - Emergency shutdown signals
        - Security alerts from other modules
        - Policy updates
        """
        if isinstance(content, dict) and content.get("type") == "security_alert":
            print(f"⚠️  SecurityMonitor received alert: {content.get('message')}")
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Return security monitoring statistics.
        
        Returns:
            Dict with:
            - total_proposals: How many states evaluated
            - total_vetoes: How many actions vetoed
            - veto_rate: Percentage of proposals that were vetoed
            - violations_by_rule: Breakdown by rule
            - last_veto_reason: Most recent veto description
            - rules_count: Number of active safety rules
        """
        veto_rate = (self.total_vetoes / self.total_proposals * 100 
                     if self.total_proposals > 0 else 0.0)
        
        return {
            "total_proposals": self.total_proposals,
            "total_vetoes": self.total_vetoes,
            "veto_rate": round(veto_rate, 2),
            "violations_by_rule": dict(self.violations_by_rule),
            "last_veto_reason": self.last_veto_reason,
            "rules_count": len(self.rules)
        }


# Example usage
if __name__ == "__main__":
    def _encode(label: str) -> hv.HyperVector:
        """Simple deterministic encoding."""
        h = hashlib.sha256(label.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        return hv.HyperVector(seed)
    
    print("SecurityMonitor Demo")
    print("=" * 50)
    
    # Create monitor
    monitor = SecurityMonitor(base_bid=-100.0)
    
    # Add safety rules
    cliff_pattern = _encode("near_cliff")
    monitor.add_rule(SafetyRule(
        pattern_hv=cliff_pattern.bits,
        threshold=0.6,
        description="Too close to cliff edge",
        severity=9
    ))
    
    danger_pattern = _encode("danger")
    monitor.add_rule(SafetyRule(
        pattern_hv=danger_pattern.bits,
        threshold=0.5,
        description="Generic danger detected",
        severity=5
    ))
    
    # Test safe state
    safe_state = _encode("walking").bundle(_encode("grass"))
    result = monitor.propose(safe_state.bits)
    print(f"\n✅ Safe state: {result}")
    
    # Test dangerous state
    dangerous_state = _encode("near_cliff").bundle(_encode("moving_forward"))
    result = monitor.propose(dangerous_state.bits)
    print(f"\n⚠️  Dangerous state: {result}")
    if result:
        print(f"   Salience: {result.base_salience}")
        print(f"   Reason: {result.content['reason']}")
    
    # Print telemetry
    print(f"\n📊 Telemetry: {monitor.get_telemetry()}")
