"""
ConceptDriftDetector — monitors semantic memory stability.

Tracks concept HVs over time; raises a drift alarm when a concept's HV
changes more than `drift_threshold` from its reference snapshot.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod


@dataclass
class DriftEvent:
    """Record of a detected drift for one concept."""
    concept: str
    similarity_to_reference: float
    drift_magnitude: float        # 1.0 - similarity
    timestamp: float
    alarm: bool                   # True if above drift_threshold


class ConceptDriftDetector:
    """
    Monitors concept HVs and detects semantic drift.

    Usage
    -----
    1. Call ``snapshot(concept, hv)`` to register a reference HV.
    2. Call ``check(concept, hv)`` periodically to detect drift.
    3. ``get_drift_report()`` returns all detected drift events.
    """

    def __init__(self, drift_threshold: float = 0.2) -> None:
        """
        Parameters
        ----------
        drift_threshold : float
            Drift magnitude (1.0 - similarity) above which an alarm is raised.
        """
        self.drift_threshold = drift_threshold
        self._snapshots: Dict[str, hv_mod.HyperVector] = {}
        self._drift_log: List[DriftEvent] = []

    def snapshot(self, concept: str, hv: hv_mod.HyperVector) -> None:
        """Register a reference HV for a concept."""
        self._snapshots[concept] = hv

    def check(self, concept: str, hv: hv_mod.HyperVector) -> DriftEvent:
        """Check current HV against snapshot; returns a DriftEvent."""
        if concept not in self._snapshots:
            self.snapshot(concept, hv)
            return DriftEvent(
                concept=concept,
                similarity_to_reference=1.0,
                drift_magnitude=0.0,
                timestamp=time.time(),
                alarm=False,
            )
        ref = self._snapshots[concept]
        sim = float(ref.similarity(hv))
        drift = 1.0 - sim
        alarm = drift > self.drift_threshold
        event = DriftEvent(
            concept=concept,
            similarity_to_reference=sim,
            drift_magnitude=drift,
            timestamp=time.time(),
            alarm=alarm,
        )
        self._drift_log.append(event)
        return event

    def update_snapshot(self, concept: str, hv: hv_mod.HyperVector) -> None:
        """Update the reference snapshot (after drift is accepted/resolved)."""
        self._snapshots[concept] = hv

    def get_drift_report(self) -> List[DriftEvent]:
        """Return all drift events in chronological order."""
        return list(self._drift_log)

    def get_alarmed_concepts(self) -> List[str]:
        """Return list of concepts that triggered an alarm."""
        return list({e.concept for e in self._drift_log if e.alarm})

    def get_statistics(self) -> Dict[str, Any]:
        alarms = [e for e in self._drift_log if e.alarm]
        return {
            "snapshots": len(self._snapshots),
            "total_checks": len(self._drift_log),
            "total_alarms": len(alarms),
            "alarmed_concepts": self.get_alarmed_concepts(),
        }
