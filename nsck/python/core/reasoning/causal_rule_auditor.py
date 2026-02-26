"""
CausalRuleAuditor — bridges symbolic ILP rules with causal graph structure.

Each rule is scored based on how well its condition→consequence aligns with
the causal graph edges. The audit trace provides a glass-box explanation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AuditedRule:
    """A rule decorated with causal audit metadata."""
    condition: str
    consequence: str
    original_confidence: float
    causal_score: float          # [0.0, 1.0] — higher = more causally grounded
    audit_trace: List[str]       # Glass-box explanation of how score was computed
    combined_score: float        # weighted combination of original confidence + causal_score


class CausalRuleAuditor:
    """
    Audits symbolic rules against a causal graph.

    causal_weight controls how much the causal score matters vs the original
    rule confidence when computing combined_score.
    """
    def __init__(self, causal_weight: float = 0.4):
        self.causal_weight = causal_weight
        self._causal_edges: Dict[Tuple[str, str], float] = {}  # (cause, effect) → strength
        self._audit_history: List[AuditedRule] = []

    def add_causal_edge(self, cause: str, effect: str, strength: float = 1.0) -> None:
        """Register a known causal edge."""
        self._causal_edges[(cause, effect)] = float(np.clip(strength, 0.0, 1.0))

    def audit_rule(
        self,
        condition: str,
        consequence: str,
        confidence: float,
    ) -> AuditedRule:
        """Score one symbolic rule against the causal graph."""
        trace: List[str] = []
        trace.append(f"Rule: {condition} → {consequence} (conf={confidence:.3f})")

        # Direct edge lookup
        direct = self._causal_edges.get((condition, consequence), 0.0)
        if direct > 0:
            trace.append(f"  Direct causal edge found: strength={direct:.3f}")
        else:
            trace.append("  No direct causal edge found")

        # Indirect: check if condition is a cause of any intermediate that causes consequence
        indirect_max = 0.0
        for (c1, e1), s1 in self._causal_edges.items():
            for (c2, e2), s2 in self._causal_edges.items():
                if c1 == condition and e1 == c2 and e2 == consequence:
                    path_strength = s1 * s2
                    if path_strength > indirect_max:
                        indirect_max = path_strength
                        trace.append(
                            f"  Indirect path: {condition}→{e1}→{consequence} "
                            f"(strength={path_strength:.3f})"
                        )

        causal_score = float(max(direct, indirect_max))
        if causal_score == 0.0:
            trace.append("  Causal score: 0.0 (no causal support)")
        else:
            trace.append(f"  Causal score: {causal_score:.3f}")

        combined = (1 - self.causal_weight) * confidence + self.causal_weight * causal_score
        trace.append(
            f"  Combined = {1-self.causal_weight:.2f}×conf + "
            f"{self.causal_weight:.2f}×causal = {combined:.3f}"
        )

        rule = AuditedRule(
            condition=condition,
            consequence=consequence,
            original_confidence=float(confidence),
            causal_score=causal_score,
            audit_trace=trace,
            combined_score=float(combined),
        )
        self._audit_history.append(rule)
        return rule

    def audit_rules(
        self,
        rules: List[Tuple[str, str, float]],
    ) -> List[AuditedRule]:
        """Audit a list of (condition, consequence, confidence) triples."""
        return [self.audit_rule(c, e, conf) for c, e, conf in rules]

    def top_rules(self, n: int = 5) -> List[AuditedRule]:
        """Return the n highest combined_score rules from history."""
        return sorted(self._audit_history, key=lambda r: r.combined_score, reverse=True)[:n]

    def get_audit_history(self) -> List[AuditedRule]:
        return list(self._audit_history)
