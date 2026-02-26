"""
Safety Rule Formal Verifier
============================
Formal safety properties and verification for NSCK rules and decisions.

Provides a declarative, expression-based safety layer that checks symbolic
rules and actions before they influence behaviour:

- ``SafetyProperty`` encodes a named safety invariant as a Python expression
  string evaluated in a restricted namespace (no arbitrary builtins).
- ``SafetyRuleVerifier`` checks all registered properties against a ``Rule``
  object and returns a detailed violation report.
- ``SafetyGateVerifier`` wraps ``SafetyRuleVerifier`` and adds a ``gate_decision``
  method used by ``CognitiveEngine`` to block actions that violate critical
  properties (e.g. code injection, runaway fire counts, low confidence).

Default safety properties:
- ``min_confidence`` (warning): rule confidence must exceed 0.3.
- ``min_support`` (warning): rule must have ≥ 2 supporting examples.
- ``no_runaway`` (critical): fire count must be < 10 000.
- ``no_code_injection`` (critical): action string must not be in ``FORBIDDEN_ACTIONS``.

Custom properties can be added via ``SafetyRuleVerifier.add_property()``.

Integration: ``CognitiveEngine`` instantiates ``SafetyGateVerifier`` and calls
``gate_decision`` after coalition selection to veto unsafe actions before they
are broadcast to the environment.
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
    """A named safety property expressed as a Python expression formula.

    The formula is evaluated in a restricted namespace.  The following
    variables are available inside the formula string:

    - ``confidence`` (float): rule confidence in [0, 1]
    - ``support`` (int): number of observed supporting examples
    - ``conditions`` (set): set of symbolic preconditions
    - ``action`` (str): rule consequence / action string
    - ``fire_count`` (int): number of times the rule has fired
    - ``FORBIDDEN_ACTIONS`` (list): list of banned action strings
    - ``len`` (builtin): the built-in ``len`` function

    All other builtins are restricted (``__builtins__`` is ``{}``) to
    prevent code injection via crafted formula strings.
    """

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
