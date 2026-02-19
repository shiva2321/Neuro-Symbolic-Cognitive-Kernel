"""
DomainExpert - Specialized Knowledge Module
============================================

Example module demonstrating domain-specific reasoning.

Features:
- Encapsulates expert knowledge for a specific domain
- Pattern matching with domain concepts
- Specialized bidding strategies
- Knowledge base management

Use Case:
When NSCK needs expert reasoning in a specific domain (chess, medical diagnosis,
code review), DomainExpert can provide specialized advice with high confidence.

This example implements a simple chess domain expert.

Author: NSCK SDK Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import hashlib

from global_workspace import WorkspaceModule, Coalition
import hypervec_shim as hv


@dataclass
class DomainPattern:
    """
    A domain-specific pattern with associated action.
    
    Attributes:
        pattern_hv: Hypervector representing the pattern
        action_hv: Recommended action for this pattern
        confidence: How confident we are in this pattern-action pair
        description: Human-readable description
        win_count: Number of times this led to success
        loss_count: Number of times this led to failure
    """
    pattern_hv: np.ndarray
    action_hv: np.ndarray
    confidence: float
    description: str
    win_count: int = 0
    loss_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate empirical success rate."""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.5  # Unknown
        return self.win_count / total


class DomainExpert(WorkspaceModule):
    """
    Domain-specific expert with specialized knowledge patterns.
    
    This module demonstrates:
    - Pattern-based knowledge representation
    - Confidence-weighted bidding
    - Learning from outcomes
    - Domain concept encoding
    
    Example (Chess Domain):
        # Create chess expert
        expert = DomainExpert(domain_name="chess")
        
        # Add opening patterns
        expert.add_pattern(
            pattern="king_exposed",
            action="castle",
            confidence=0.9,
            description="Castle when king is exposed"
        )
        
        # Register with CognitiveEngine
        engine.register_module(expert)
    """
    
    __author__ = "NSCK SDK Team"
    __version__ = "1.0.0"
    
    def __init__(self, domain_name: str = "general", 
                 match_threshold: float = 0.6,
                 base_bid: float = 50.0):
        """
        Initialize DomainExpert.
        
        Args:
            domain_name: Name of the domain (e.g., "chess", "medical", "code_review")
            match_threshold: Minimum similarity for pattern matching
            base_bid: Base bid value (scaled by confidence)
        """
        self.domain_name = domain_name
        self.match_threshold = match_threshold
        self.base_bid = base_bid
        
        # Knowledge base
        self.patterns: List[DomainPattern] = []
        
        # Telemetry
        self.total_proposals = 0
        self.patterns_matched = 0
        self.patterns_executed = 0
        self.total_reward = 0.0
        self.last_pattern_matched = ""
    
    def add_pattern(self, pattern: str, action: str, 
                   confidence: float, description: str):
        """
        Add a pattern-action pair to knowledge base.
        
        Args:
            pattern: Pattern identifier (will be encoded)
            action: Action identifier (will be encoded)
            confidence: Initial confidence (0.0-1.0)
            description: Human-readable description
        """
        # Deterministic encoding
        h1 = hashlib.sha256(f"{self.domain_name}_{pattern}".encode("utf-8")).digest()
        seed1 = int.from_bytes(h1[:4], "little")
        pattern_hv = hv.HyperVector(seed1).bits
        
        h2 = hashlib.sha256(f"{self.domain_name}_{action}".encode("utf-8")).digest()
        seed2 = int.from_bytes(h2[:4], "little")
        action_hv = hv.HyperVector(seed2).bits
        
        self.patterns.append(DomainPattern(
            pattern_hv=pattern_hv,
            action_hv=action_hv,
            confidence=confidence,
            description=description,
            win_count=0,
            loss_count=0
        ))
    
    def add_pattern_hv(self, pattern_hv: np.ndarray, action_hv: np.ndarray,
                      confidence: float, description: str):
        """
        Add a pattern using pre-encoded hypervectors.
        
        Useful for complex patterns assembled from multiple concepts.
        """
        self.patterns.append(DomainPattern(
            pattern_hv=pattern_hv,
            action_hv=action_hv,
            confidence=confidence,
            description=description,
            win_count=0,
            loss_count=0
        ))
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        """
        Match state against domain patterns and propose expert action.
        
        Args:
            state_hv: Current workspace state hypervector (10240-bit np.ndarray)
            
        Returns:
            Coalition with domain expert's recommendation, or None if no match
        """
        self.total_proposals += 1
        
        # Find best matching pattern
        best_pattern = None
        best_similarity = self.match_threshold
        
        for pattern in self.patterns:
            # Compute similarity
            diff = np.bitwise_xor(state_hv, pattern.pattern_hv)
            hamming_dist = np.sum(diff)
            sim = 1.0 - (hamming_dist / len(state_hv))
            
            if sim > best_similarity:
                best_similarity = sim
                best_pattern = pattern
        
        # No pattern matched
        if best_pattern is None:
            return None
        
        self.patterns_matched += 1
        self.last_pattern_matched = best_pattern.description
        
        # Bid based on:
        # 1. Pattern confidence
        # 2. Empirical success rate
        # 3. Pattern match quality
        
        confidence_factor = best_pattern.confidence
        success_factor = best_pattern.success_rate
        match_factor = best_similarity
        
        # Map expert metrics to Coalition salience
        base_salience = match_factor  # How well the pattern matches
        relevance = confidence_factor  # Pattern confidence
        sender_confidence = success_factor  # Empirical success rate
        
        return Coalition(
            source=f"DomainExpert.{self.domain_name}",
            content={
                "type": "expert_action",
                "action_hv": best_pattern.action_hv,
                "pattern": best_pattern.description,
                "confidence": best_pattern.confidence,
                "success_rate": best_pattern.success_rate,
                "similarity": float(best_similarity),
                "win_count": best_pattern.win_count,
                "loss_count": best_pattern.loss_count
            },
            base_salience=base_salience,
            relevance=relevance,
            sender_confidence=sender_confidence
        )
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        """
        Learn from outcomes to update pattern success rates.
        
        Args:
            feedback_hv: Actual resulting state
            reward: Reward received
            info: Additional information (should contain 'pattern_used')
        """
        self.total_reward += reward
        
        # Find which pattern was used (if any)
        used_pattern_desc = info.get("pattern")
        if used_pattern_desc is None:
            return
        
        # Update pattern statistics
        for pattern in self.patterns:
            if pattern.description == used_pattern_desc:
                self.patterns_executed += 1
                
                if reward > 0:
                    pattern.win_count += 1
                elif reward < 0:
                    pattern.loss_count += 1
                
                # Adjust confidence based on success rate
                # (Slowly converge to empirical success rate)
                alpha = 0.1  # Learning rate
                pattern.confidence = (
                    (1 - alpha) * pattern.confidence +
                    alpha * pattern.success_rate
                )
                
                break
    
    def receive_broadcast(self, content: Any):
        """
        React to broadcasts.
        
        Could be used for:
        - Learning new patterns from other modules
        - Receiving domain updates
        - Debugging commands
        """
        if isinstance(content, dict):
            if content.get("type") == "learn_pattern":
                # Another module suggests a new pattern
                print(f"📚 DomainExpert learning new pattern: {content.get('description')}")
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Return domain expert statistics.
        
        Returns:
            Dict with:
            - domain_name: The domain this expert specializes in
            - total_proposals: How many times propose() called
            - patterns_matched: How many times a pattern matched
            - patterns_executed: How many times pattern action was executed
            - match_rate: Percentage of proposals that matched a pattern
            - execution_rate: Percentage of matches that were executed
            - avg_reward: Average reward per executed pattern
            - patterns_count: Number of patterns in knowledge base
            - last_pattern_matched: Most recent pattern description
            - top_patterns: Top 3 patterns by success rate
        """
        match_rate = (self.patterns_matched / self.total_proposals * 100 
                     if self.total_proposals > 0 else 0.0)
        
        execution_rate = (self.patterns_executed / self.patterns_matched * 100 
                         if self.patterns_matched > 0 else 0.0)
        
        avg_reward = (self.total_reward / self.patterns_executed 
                     if self.patterns_executed > 0 else 0.0)
        
        # Top patterns by success rate
        sorted_patterns = sorted(
            [p for p in self.patterns if p.win_count + p.loss_count > 0],
            key=lambda p: p.success_rate,
            reverse=True
        )
        top_patterns = [
            {
                "description": p.description,
                "success_rate": round(p.success_rate, 3),
                "confidence": round(p.confidence, 3),
                "usage_count": p.win_count + p.loss_count
            }
            for p in sorted_patterns[:3]
        ]
        
        return {
            "domain_name": self.domain_name,
            "total_proposals": self.total_proposals,
            "patterns_matched": self.patterns_matched,
            "patterns_executed": self.patterns_executed,
            "match_rate": round(match_rate, 2),
            "execution_rate": round(execution_rate, 2),
            "avg_reward": round(avg_reward, 2),
            "patterns_count": len(self.patterns),
            "last_pattern_matched": self.last_pattern_matched,
            "top_patterns": top_patterns
        }


# Example usage (Chess Domain)
if __name__ == "__main__":
    def _encode(label: str) -> hv.HyperVector:
        """Simple deterministic encoding."""
        h = hashlib.sha256(label.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        return hv.HyperVector(seed)
    
    print("DomainExpert Demo - Chess Domain")
    print("=" * 50)
    
    # Create chess expert
    expert = DomainExpert(domain_name="chess", base_bid=60.0)
    
    # Add chess patterns (simplified)
    expert.add_pattern(
        pattern="king_exposed",
        action="castle",
        confidence=0.9,
        description="Castle when king is exposed"
    )
    
    expert.add_pattern(
        pattern="center_control",
        action="advance_pawn",
        confidence=0.8,
        description="Control center with pawn advance"
    )
    
    expert.add_pattern(
        pattern="queen_hanging",
        action="defend_queen",
        confidence=0.95,
        description="Defend hanging queen"
    )
    
    expert.add_pattern(
        pattern="fork_opportunity",
        action="knight_fork",
        confidence=0.85,
        description="Execute knight fork"
    )
    
    # Test pattern matching
    print("\n🔍 Testing pattern matching...")
    
    # Simulate exposed king situation
    state = _encode("chess_king_exposed").bits
    coalition = expert.propose(state)
    if coalition:
        print(f"\n✅ Pattern matched: {coalition.content['pattern']}")
        print(f"   Salience: {coalition.base_salience:.2f}")
        print(f"   Confidence: {coalition.content['confidence']:.2f}")
        
        # Simulate successful outcome
        expert.update(state, reward=10.0, info={"pattern": coalition.content['pattern']})
    
    # Simulate fork opportunity
    state = _encode("chess_fork_opportunity").bits
    coalition = expert.propose(state)
    if coalition:
        print(f"\n✅ Pattern matched: {coalition.content['pattern']}")
        print(f"   Salience: {coalition.base_salience:.2f}")
        
        # Simulate successful outcome
        expert.update(state, reward=15.0, info={"pattern": coalition.content['pattern']})
    
    # Simulate no match
    state = _encode("chess_endgame_pawn_race").bits
    coalition = expert.propose(state)
    print(f"\n❌ No pattern matched for endgame pawn race: {coalition}")
    
    # Print telemetry
    print(f"\n📊 Telemetry:")
    telemetry = expert.get_telemetry()
    for key, value in telemetry.items():
        print(f"   {key}: {value}")
