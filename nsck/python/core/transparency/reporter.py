"""
TransparencyReporter — helper for printing and saving ThoughtTrace objects (V30).
"""
from __future__ import annotations

import json
import os
from typing import List

from python.core.transparency.thought_trace import ThoughtTrace, REQUIRED_STAGES


class TransparencyReporter:
    """Helper for printing, saving, and comparing ThoughtTrace objects."""

    # ANSI colour codes — disabled automatically when stdout is not a TTY
    _CYAN = "\033[96m"
    _GREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _BOLD = "\033[1m"
    _RESET = "\033[0m"

    def __init__(self, use_colour: bool = True) -> None:
        import sys
        self._colour = use_colour and sys.stdout.isatty()

    def _c(self, text: str, code: str) -> str:
        if self._colour:
            return f"{code}{text}{self._RESET}"
        return text

    def print_trace(self, trace: ThoughtTrace) -> None:
        """Print a ThoughtTrace to stdout with optional colour."""
        print(self._c(f"\n{'='*60}", self._BOLD))
        print(self._c(f"ThoughtTrace: {trace.query_id}", self._BOLD + self._CYAN))
        print(self._c(f"{'='*60}", self._BOLD))
        print(f"Input      : {trace.input_text[:80]}")
        print(f"Modality   : {trace.input_modality}")
        print(f"Timestamp  : {trace.timestamp}")
        print(f"Duration   : {trace.total_duration_ms:.1f} ms")
        print(f"Action     : {self._c(trace.final_action, self._GREEN)}")
        print(f"Confidence : {trace.confidence:.4f}")
        print(f"Emotion    : {trace.emotion_state}")
        print(f"Novelty    : {trace.novelty_score:.4f}")
        print(f"Rust used  : {trace.rust_used}")
        print(self._c(f"\n--- Cognitive Stages ---", self._YELLOW))
        for step in trace.steps:
            dur = f"({step.duration_ms:.1f}ms)" if step.duration_ms > 0 else ""
            print(f"  [{step.stage:22s}] {step.summary[:70]} {dur}")
        print(self._c(f"{'='*60}\n", self._BOLD))

    def save_trace(self, trace: ThoughtTrace, path: str) -> None:
        """Save a ThoughtTrace as JSON to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        trace.to_json(path)

    def compare_traces(self, t1: ThoughtTrace, t2: ThoughtTrace) -> str:
        """Return a textual diff report between two ThoughtTrace objects."""
        lines = [
            f"ThoughtTrace diff: {t1.query_id} vs {t2.query_id}",
            f"  confidence  : {t1.confidence:.4f} → {t2.confidence:.4f}  "
            f"Δ={t2.confidence - t1.confidence:+.4f}",
            f"  emotion     : {t1.emotion_state} → {t2.emotion_state}",
            f"  novelty     : {t1.novelty_score:.4f} → {t2.novelty_score:.4f}  "
            f"Δ={t2.novelty_score - t1.novelty_score:+.4f}",
            f"  total_ms    : {t1.total_duration_ms:.1f} → {t2.total_duration_ms:.1f}  "
            f"Δ={t2.total_duration_ms - t1.total_duration_ms:+.1f}",
            f"  rust_used   : {t1.rust_used} → {t2.rust_used}",
            "  Stages:",
        ]
        for stage in REQUIRED_STAGES:
            s1 = t1.get_step(stage)
            s2 = t2.get_step(stage)
            sum1 = s1.summary if s1 else "N/A"
            sum2 = s2.summary if s2 else "N/A"
            changed = "X" if sum1 != sum2 else "="
            lines.append(f"    [{changed}] {stage:22s}: {sum1[:30]} -> {sum2[:30]}")
        return "\n".join(lines)

    def batch_report(self, traces: List[ThoughtTrace], path: str) -> None:
        """Write a Markdown batch report for a list of traces to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lines = [
            "# NSCK ThoughtTrace Batch Report",
            f"",
            f"Total traces: {len(traces)}",
            f"",
        ]
        for trace in traces:
            lines.append(trace.to_markdown())
            lines.append("\n---\n")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
