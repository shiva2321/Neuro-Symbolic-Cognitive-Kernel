"""
NSCK Rule Extraction Module (Phase 1.1)
=======================================
Implements rule extraction from neural policies for interpretability.
Enables dual inference: neural (fast) + symbolic (safe).

Based on:
  - ECLAIRE (Extracting CLassification Rules from ANNs)
  - DeepRED (Deep Rule Extraction and Discovery)
  - Decision tree approximation

Design Principles:
  - Extract symbolic rules from trained neural policies
  - Use rules to guide neural exploration
  - Symbolic rules can override dangerous neural actions
  - Maintain interpretability while leveraging neural learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict


@dataclass
class Rule:
    """
    Symbolic rule extracted from neural policy.
    
    Format: IF conditions THEN action (with confidence)
    """
    conditions: List[Tuple[str, str, float]]  # [(feature, operator, threshold), ...]
    action: str
    confidence: float
    support: int  # Number of samples supporting this rule
    
    def __str__(self) -> str:
        """Human-readable rule representation."""
        cond_str = " AND ".join([
            f"{feat} {op} {thresh:.2f}" 
            for feat, op, thresh in self.conditions
        ])
        return f"IF {cond_str} THEN {self.action} (conf={self.confidence:.2f}, support={self.support})"
    
    def matches(self, state_features: Dict[str, float]) -> bool:
        """Check if this rule applies to the given state."""
        for feature, operator, threshold in self.conditions:
            if feature not in state_features:
                return False
            
            value = state_features[feature]
            
            if operator == ">":
                if not (value > threshold):
                    return False
            elif operator == "<":
                if not (value < threshold):
                    return False
            elif operator == ">=":
                if not (value >= threshold):
                    return False
            elif operator == "<=":
                if not (value <= threshold):
                    return False
            elif operator == "==":
                if not (abs(value - threshold) < 1e-6):
                    return False
            else:
                return False
        
        return True


@dataclass
class RuleSet:
    """Collection of extracted rules with priority ordering."""
    rules: List[Rule] = field(default_factory=list)
    
    def add_rule(self, rule: Rule):
        """Add a rule to the set."""
        self.rules.append(rule)
    
    def get_matching_rules(self, state_features: Dict[str, float]) -> List[Rule]:
        """Get all rules that match the given state."""
        return [rule for rule in self.rules if rule.matches(state_features)]
    
    def get_best_action(self, state_features: Dict[str, float], 
                       default: str = "STAY") -> Tuple[str, float]:
        """
        Get best action from matching rules.
        
        Returns:
            Tuple of (action, confidence)
        """
        matching = self.get_matching_rules(state_features)
        
        if not matching:
            return default, 0.0
        
        # Sort by confidence
        best_rule = max(matching, key=lambda r: r.confidence)
        return best_rule.action, best_rule.confidence
    
    def __len__(self) -> int:
        return len(self.rules)


class NeuralRuleExtractor:
    """
    Extract symbolic rules from trained neural networks.
    
    Uses decision tree approximation and clustering to find
    interpretable decision boundaries in the neural policy.
    """
    
    def __init__(self, model: nn.Module, feature_names: List[str],
                 action_names: List[str]):
        """
        Args:
            model: Trained neural network (policy)
            feature_names: Names of input features
            action_names: Names of possible actions
        """
        self.model = model
        self.feature_names = feature_names
        self.action_names = action_names
        self.model.eval()
        
    def extract_rules(self, states: torch.Tensor, 
                     max_rules: int = 20,
                     min_support: int = 5) -> RuleSet:
        """
        Extract rules from neural policy.
        
        Args:
            states: Sample states for rule extraction [N, state_dim]
            max_rules: Maximum number of rules to extract
            min_support: Minimum samples required to support a rule
            
        Returns:
            RuleSet containing extracted rules
        """
        rule_set = RuleSet()
        
        # Get neural predictions
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                # Handle different model architectures
                outputs = self.model(states)
                if isinstance(outputs, tuple):
                    action_logits = outputs[0]
                else:
                    action_logits = outputs
            else:
                action_logits = self.model(states)
            
            predicted_actions = torch.argmax(action_logits, dim=-1)
        
        # Convert to numpy for easier manipulation
        states_np = states.cpu().numpy()
        actions_np = predicted_actions.cpu().numpy()
        
        # Extract rules per action
        for action_idx, action_name in enumerate(self.action_names):
            # Find samples where this action is predicted
            action_mask = actions_np == action_idx
            action_states = states_np[action_mask]
            
            if len(action_states) < min_support:
                continue
            
            # Extract decision boundaries for this action
            rules = self._extract_action_rules(
                action_states, 
                action_name,
                max_rules=max_rules // len(self.action_names),
                min_support=min_support
            )
            
            for rule in rules:
                rule_set.add_rule(rule)
        
        return rule_set
    
    def _extract_action_rules(self, states: np.ndarray, action: str,
                             max_rules: int = 5,
                             min_support: int = 5) -> List[Rule]:
        """Extract rules for a specific action."""
        rules = []
        
        if len(states) < min_support:
            return rules
        
        # Simple rule extraction: find feature ranges
        # For each feature, find common ranges
        for feat_idx, feat_name in enumerate(self.feature_names):
            feature_values = states[:, feat_idx]
            
            # Skip if feature doesn't vary
            if np.std(feature_values) < 1e-6:
                continue
            
            # Find median threshold
            threshold = np.median(feature_values)
            
            # Create rules for > threshold
            above_mask = feature_values > threshold
            if np.sum(above_mask) >= min_support:
                conditions = [(feat_name, ">", float(threshold))]
                confidence = np.sum(above_mask) / len(feature_values)
                
                rule = Rule(
                    conditions=conditions,
                    action=action,
                    confidence=confidence,
                    support=int(np.sum(above_mask))
                )
                rules.append(rule)
            
            # Create rules for <= threshold
            below_mask = feature_values <= threshold
            if np.sum(below_mask) >= min_support:
                conditions = [(feat_name, "<=", float(threshold))]
                confidence = np.sum(below_mask) / len(feature_values)
                
                rule = Rule(
                    conditions=conditions,
                    action=action,
                    confidence=confidence,
                    support=int(np.sum(below_mask))
                )
                rules.append(rule)
        
        # Sort by confidence and support
        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        
        return rules[:max_rules]


class DualInferenceEngine:
    """
    Dual inference system: Neural (fast) + Symbolic (safe).
    
    The neural policy provides fast, generalizable decisions.
    The symbolic rules provide safety guarantees and interpretability.
    
    Strategy:
    1. Neural policy makes prediction
    2. Symbolic rules check for safety violations
    3. If conflict, symbolic rules override
    4. Track conflicts for retraining
    """
    
    def __init__(self, neural_model: nn.Module, rule_set: RuleSet,
                 safety_rules: Optional[RuleSet] = None):
        """
        Args:
            neural_model: Trained neural policy
            rule_set: Extracted symbolic rules
            safety_rules: Optional high-priority safety rules
        """
        self.neural_model = neural_model
        self.rule_set = rule_set
        self.safety_rules = safety_rules or RuleSet()
        
        self.neural_model.eval()
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "neural_decisions": 0,
            "symbolic_overrides": 0,
            "safety_overrides": 0,
            "conflicts": []
        }
    
    def decide(self, state: torch.Tensor, 
              state_features: Dict[str, float],
              require_safe: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Make decision using dual inference.
        
        Args:
            state: Raw state tensor for neural model
            state_features: Extracted features for symbolic rules
            require_safe: Whether to enforce safety rules
            
        Returns:
            Tuple of (action, metadata)
        """
        self.stats["total_decisions"] += 1
        
        # 1. Neural inference (fast path)
        with torch.no_grad():
            outputs = self.neural_model(state)
            if isinstance(outputs, tuple):
                action_logits = outputs[0]
            else:
                action_logits = outputs
            
            neural_probs = F.softmax(action_logits, dim=-1)
            neural_action_idx = torch.argmax(neural_probs, dim=-1).item()
            neural_confidence = neural_probs[0, neural_action_idx].item()
        
        neural_action = f"ACTION_{neural_action_idx}"
        
        # 2. Check safety rules (highest priority)
        if require_safe and len(self.safety_rules) > 0:
            safety_action, safety_conf = self.safety_rules.get_best_action(state_features)
            
            if safety_conf > 0.8 and safety_action != neural_action:
                self.stats["safety_overrides"] += 1
                self.stats["conflicts"].append({
                    "type": "safety",
                    "neural": neural_action,
                    "symbolic": safety_action
                })
                
                return safety_action, {
                    "mode": "SAFETY",
                    "neural_action": neural_action,
                    "neural_confidence": neural_confidence,
                    "symbolic_confidence": safety_conf,
                    "override": True
                }
        
        # 3. Check symbolic rules (interpretability)
        symbolic_action, symbolic_conf = self.rule_set.get_best_action(state_features)
        
        # 4. Arbitration
        if symbolic_conf > 0.7 and symbolic_action != neural_action:
            # Symbolic rules are confident and disagree
            if neural_confidence < 0.6:
                # Neural is uncertain, trust symbolic
                self.stats["symbolic_overrides"] += 1
                self.stats["conflicts"].append({
                    "type": "symbolic",
                    "neural": neural_action,
                    "symbolic": symbolic_action
                })
                
                return symbolic_action, {
                    "mode": "SYMBOLIC",
                    "neural_action": neural_action,
                    "neural_confidence": neural_confidence,
                    "symbolic_confidence": symbolic_conf,
                    "override": True
                }
        
        # 5. Default: use neural
        self.stats["neural_decisions"] += 1
        
        return neural_action, {
            "mode": "NEURAL",
            "neural_confidence": neural_confidence,
            "symbolic_confidence": symbolic_conf,
            "override": False
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.stats.copy()
        if stats["total_decisions"] > 0:
            stats["neural_rate"] = stats["neural_decisions"] / stats["total_decisions"]
            stats["override_rate"] = (stats["symbolic_overrides"] + stats["safety_overrides"]) / stats["total_decisions"]
        return stats


# Example usage
def create_dual_inference_system(model: nn.Module, 
                                sample_states: torch.Tensor,
                                feature_names: List[str],
                                action_names: List[str]) -> DualInferenceEngine:
    """
    Create a dual inference system from a trained model.
    
    Args:
        model: Trained neural policy
        sample_states: Sample states for rule extraction
        feature_names: Names of state features
        action_names: Names of actions
        
    Returns:
        DualInferenceEngine ready for use
    """
    # Extract rules from neural policy
    extractor = NeuralRuleExtractor(model, feature_names, action_names)
    rule_set = extractor.extract_rules(sample_states)
    
    # Create dual inference engine
    engine = DualInferenceEngine(model, rule_set)
    
    return engine
