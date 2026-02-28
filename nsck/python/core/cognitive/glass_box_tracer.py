"""
GlassBoxTracer — V17

Captures a step-by-step, human-readable trace of the NSCK decision loop.
Every module that participates in a decision can append a TraceEntry so that
the full reasoning chain is available for post-hoc explanation, debugging,
and the transparency guarantee.

Usage::

    from python.core.cognitive.glass_box_tracer import GlassBoxTracer

    tracer = GlassBoxTracer()
    with tracer.span("perception"):
        tracer.record("TextAdapter", "encoded 'hello world'", confidence=0.9)
    with tracer.span("reasoning"):
        tracer.record("CognitiveEngine", "selected rule R42", confidence=0.75)

    trace = tracer.export()
    print(tracer.format_trace(trace))
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import logging

log = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    span: str
    module: str
    message: str
    confidence: Optional[float] = None
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    decision_id: str
    entries: List[TraceEntry] = field(default_factory=list)
    start_ms: float = field(default_factory=lambda: time.time() * 1000)
    end_ms: Optional[float] = None

    @property
    def elapsed_ms(self) -> Optional[float]:
        if self.end_ms is not None:
            return self.end_ms - self.start_ms
        return None


class GlassBoxTracer:
    """
    Glass-box tracer for NSCK decision steps.

    Parameters
    ----------
    max_history : int
        Maximum number of completed decision traces to keep in memory.
    enabled : bool
        When False, all recording is a no-op (production fast-path).
    """

    def __init__(self, max_history: int = 100, enabled: bool = True):
        self.enabled = enabled
        self.max_history = max_history
        self._active: Optional[DecisionTrace] = None
        self._current_span: str = "root"
        self._history: List[DecisionTrace] = []
        self._decision_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_decision(self, decision_id: Optional[str] = None) -> str:
        """
        Begin recording a new decision trace.

        Parameters
        ----------
        decision_id : str, optional
            Human-readable ID. Auto-generated if omitted.

        Returns
        -------
        str
            The decision ID used.
        """
        self._decision_counter += 1
        did = decision_id or f"D{self._decision_counter:06d}"
        self._active = DecisionTrace(decision_id=did)
        self._current_span = "root"
        return did

    def end_decision(self) -> Optional[DecisionTrace]:
        """
        Finalise and archive the current decision trace.

        Returns
        -------
        DecisionTrace or None
        """
        if self._active is None:
            return None
        self._active.end_ms = time.time() * 1000
        completed = self._active
        self._active = None
        self._history.append(completed)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        return completed

    def record(
        self,
        module: str,
        message: str,
        confidence: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """
        Record a single trace entry in the current span.

        Parameters
        ----------
        module : str
            Name of the module making the record.
        message : str
            Human-readable description of the step.
        confidence : float, optional
            Confidence score [0, 1] associated with this step.
        **metadata
            Arbitrary key-value metadata.
        """
        if not self.enabled or self._active is None:
            return
        entry = TraceEntry(
            span=self._current_span,
            module=module,
            message=message,
            confidence=confidence,
            metadata=dict(metadata),
        )
        self._active.entries.append(entry)

    @contextmanager
    def span(self, name: str):
        """Context manager to group entries under a named span."""
        previous = self._current_span
        self._current_span = name
        try:
            yield
        finally:
            self._current_span = previous

    def export(self) -> Optional[DecisionTrace]:
        """Return the *active* trace (before end_decision is called)."""
        return self._active

    def last_trace(self) -> Optional[DecisionTrace]:
        """Return the most recently completed trace."""
        return self._history[-1] if self._history else None

    def history(self) -> List[DecisionTrace]:
        """Return all archived traces."""
        return list(self._history)

    @staticmethod
    def format_trace(trace: Optional[DecisionTrace]) -> str:
        """
        Format a DecisionTrace as a human-readable multi-line string.

        Parameters
        ----------
        trace : DecisionTrace or None

        Returns
        -------
        str
        """
        if trace is None:
            return "(no trace)"
        lines = [f"=== Decision {trace.decision_id} ==="]
        for e in trace.entries:
            conf_str = f" [{e.confidence:.2f}]" if e.confidence is not None else ""
            lines.append(f"  [{e.span}] {e.module}: {e.message}{conf_str}")
        elapsed = trace.elapsed_ms
        if elapsed is not None:
            lines.append(f"  elapsed: {elapsed:.1f} ms")
        return "\n".join(lines)
