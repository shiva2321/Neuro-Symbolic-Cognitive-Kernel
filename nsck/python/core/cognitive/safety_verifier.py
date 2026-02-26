"""
Safety Rule Formal Verifier
============================
Formal safety properties and verification for NSCK rules.
"""
from __future__ import annotations

import sys
import os
from typing import List, Tuple, Optional, Any

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

_FORBIDDEN_ACTIONS = ["__import__", "exec", "eval", "os.system", "subprocess"]


class SafetyProperty:
    """A named safety property expressed as a Python expression formula."""

    def __init__(self, name: str, formula: str, severity: str = "critical"):
        self.name = name
        self.formula = formula
        self.severity = severity

    def check(self, rule) -> Tuple[bool, str]:
        """Evaluate formula against rule. Returns (satisfied, reason)."""
        confidence = getattr(rule, "confidence", 1.0)
        support = getattr(rule, "support_count", getattr(rule, "support", 0))
        conditions = getattr(rule, "condition", getattr(rule, "conditions", set()))
        action = getattr(rule, "consequence", getattr(rule, "action", ""))
        fire_count = getattr(rule, "fire_count", 0)
        forbidden_actions = _FORBIDDEN_ACTIONS  # lowercase local var in eval namespace

        try:
            result = bool(eval(  # noqa: S307 — formula is system-defined, not user input
                self.formula,
                {
                    "confidence": confidence,
                    "support": support,
                    "conditions": conditions,
                    "action": action,
                    "fire_count": fire_count,
                    "FORBIDDEN_ACTIONS": forbidden_actions,  # kept for backward compat
                    "len": len,
                    "__builtins__": {},  # restrict builtins for safety
                },
            ))
            if result:
                return (True, "OK")
            else:
                return (False, f"Property '{self.name}' violated: {self.formula}")
        except Exception as e:
            return (False, f"Property '{self.name}' error: {e}")


class SafetyRuleVerifier:
    """Verifies rules against a set of safety properties."""

    DEFAULT_PROPERTIES: List[SafetyProperty] = [
        SafetyProperty("min_confidence", "confidence > 0.3", severity="warning"),
        SafetyProperty("min_support", "support >= 2", severity="warning"),
        SafetyProperty("no_runaway", "fire_count < 10000", severity="critical"),
        SafetyProperty(
            "no_code_injection",
            "action not in FORBIDDEN_ACTIONS",
            severity="critical",
        ),
    ]

    def __init__(self):
        self._properties: List[SafetyProperty] = list(self.DEFAULT_PROPERTIES)

    def add_property(self, prop: SafetyProperty):
        """Add a safety property."""
        self._properties.append(prop)

    def verify_rule(self, rule) -> dict:
        """Check all properties against rule."""
        rule_id = getattr(rule, "id", None)
        violations = []
        for prop in self._properties:
            satisfied, reason = prop.check(rule)
            if not satisfied:
                violations.append({
                    "property": prop.name,
                    "reason": reason,
                    "severity": prop.severity,
                })
        safe = len(violations) == 0
        critical_count = sum(1 for v in violations if v["severity"] == "critical")
        score = 1.0 - (len(violations) / max(len(self._properties), 1))
        return {
            "rule_id": rule_id,
            "safe": safe,
            "violations": violations,
            "score": float(score),
            "critical_violations": critical_count,
        }

    def verify_ruleset(self, rules: List) -> dict:
        """Verify all rules; return aggregate stats."""
        results = [self.verify_rule(r) for r in rules]
        safe_count = sum(1 for r in results if r["safe"])
        total_violations = sum(len(r["violations"]) for r in results)
        return {
            "total_rules": len(rules),
            "safe_rules": safe_count,
            "unsafe_rules": len(rules) - safe_count,
            "total_violations": total_violations,
            "results": results,
        }


class SafetyGateVerifier:
    """Combines SafetyRuleVerifier with NSCKConfig for decision gating."""

    def __init__(self, config=None):
        self._config = config
        self._verifier = SafetyRuleVerifier()
        # Add default properties
        for prop in SafetyRuleVerifier.DEFAULT_PROPERTIES:
            if prop not in self._verifier._properties:
                pass  # already added in __init__

    def gate_decision(
        self,
        action: str,
        confidence: float,
        active_rules: List,
    ) -> Tuple[bool, str]:
        """
        Gate a decision.
        Returns (allowed: bool, reason: str).
        """
        # Check all active rules
        for rule in active_rules:
            result = self._verifier.verify_rule(rule)
            if not result["safe"]:
                critical = [v for v in result["violations"] if v["severity"] == "critical"]
                if critical:
                    return (False, f"Safety violation: {critical[0]['reason']}")

        # Check confidence
        if confidence < 0.3 and action != "explore":
            return (False, "Low confidence")

        return (True, "OK")
