"""Belief Revision module for NSCK V3 - Bayesian free-energy belief scoring."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BeliefMetadata:
    evidence_count: int = 1
    contradiction_count: int = 0
    complexity: float = 1.0
    first_seen: float = field(default_factory=time.time)
    last_confirmed: float = field(default_factory=time.time)
    status: str = "active"  # active | contested | retracted


class BeliefScorer:
    def free_energy(self, meta: BeliefMetadata, lambda_complexity: float = 0.1) -> float:
        """F = -log(evidence / (evidence + contradiction + 1)) + lambda * complexity"""
        e = max(0, meta.evidence_count)
        c = max(0, meta.contradiction_count)
        ratio = e / (e + c + 1.0)
        if ratio <= 0:
            ratio = 1e-10
        return -math.log(ratio) + lambda_complexity * meta.complexity

    def should_revise(self, old_meta: BeliefMetadata, new_evidence_supports: bool) -> Tuple[bool, str]:
        """Determine if belief should be revised given new evidence.

        Returns (should_revise, reason).
        """
        if not new_evidence_supports:
            new_contra = old_meta.contradiction_count + 1
            if new_contra >= old_meta.evidence_count:
                return True, "contradictions_exceed_evidence"
            return False, "mark_contested"
        else:
            if old_meta.status == "retracted":
                return False, "belief_retracted"
            return False, "evidence_supported"
