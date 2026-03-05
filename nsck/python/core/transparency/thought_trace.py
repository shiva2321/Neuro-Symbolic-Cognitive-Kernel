"""
ThoughtTrace — Structured transparency report for NSCK cognitive decisions (V30).

Every NSCKSubstrate.process() and NSCKSubstrate.ingest() call can return a
ThoughtTrace covering all 11 cognitive stages.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# The 11 required cognitive stages
# ---------------------------------------------------------------------------

REQUIRED_STAGES = [
    "encoding",
    "emotion",
    "concept_extraction",
    "semantic_search",
    "episodic_recall",
    "causal_inference",
    "global_workspace",
    "planning",
    "self_model",
    "societal_context",
    "response_generation",
]


@dataclass
class TraceStep:
    """A single stage in the cognitive trace."""

    stage: str           # one of REQUIRED_STAGES
    summary: str         # one-line human description
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "summary": self.summary,
            "data": self.data,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class ThoughtTrace:
    """
    Full structured trace of a NSCK cognitive decision cycle.

    Contains one TraceStep per cognitive stage (all 11 required stages are
    always present, even when a stage is a no-op — those are recorded as
    ``summary="N/A"``).
    """

    query_id: str
    input_text: str
    input_modality: str
    timestamp: str
    total_duration_ms: float
    steps: List[TraceStep] = field(default_factory=list)
    final_action: str = ""
    confidence: float = 0.0
    emotion_state: str = "neutral"
    novelty_score: float = 0.0
    societal_context: Dict[str, Any] = field(default_factory=dict)
    rust_used: bool = False

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_cognitive_trace(
        cls,
        query_id: str,
        input_text: str,
        input_modality: str,
        trace: Dict[str, Any],
        final_action: str,
        confidence: float,
        total_duration_ms: float,
        rust_used: bool = False,
    ) -> "ThoughtTrace":
        """Build a ThoughtTrace from a raw CognitiveEngine trace dict."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        steps: List[TraceStep] = []

        # --- encoding ---
        enc = trace.get("encoding", {})
        steps.append(TraceStep(
            stage="encoding",
            summary=f"Encoded as {enc.get('modality', input_modality)} HV "
                    f"(dim={enc.get('hv_dimension', 10240)}, "
                    f"projector={enc.get('projector_used', 'hash')})",
            data=enc,
            duration_ms=float(enc.get("encoding_time_ms", 0.0)),
        ))

        # --- emotion ---
        emo = trace.get("emotion", {})
        emo_label = emo.get("label", "neutral")
        steps.append(TraceStep(
            stage="emotion",
            summary=f"Emotion: {emo_label} "
                    f"(valence={emo.get('valence', 0.0):.2f}, "
                    f"arousal={emo.get('arousal', 0.0):.2f})",
            data=emo,
            duration_ms=float(emo.get("duration_ms", 0.0)),
        ))

        # --- concept_extraction ---
        cex = trace.get("concept_extraction", {})
        concepts = cex.get("concepts", [])
        steps.append(TraceStep(
            stage="concept_extraction",
            summary=f"Extracted {len(concepts)} concepts; "
                    f"{len(cex.get('svo_triples', []))} SVO triples",
            data=cex,
            duration_ms=float(cex.get("duration_ms", 0.0)),
        ))

        # --- semantic_search ---
        ss = trace.get("semantic_search", {})
        top_matches = ss.get("top_matches", [])
        steps.append(TraceStep(
            stage="semantic_search",
            summary=f"Top match: {top_matches[0][0] if top_matches else 'none'} "
                    f"(sim={top_matches[0][1]:.3f})" if top_matches else "No matches found",
            data=ss,
            duration_ms=float(ss.get("duration_ms", 0.0)),
        ))

        # --- episodic_recall ---
        er = trace.get("episodic_recall", {})
        steps.append(TraceStep(
            stage="episodic_recall",
            summary=f"Recalled {er.get('n_episodes_recalled', 0)} episodes "
                    f"(best_sim={er.get('best_similarity', 0.0):.3f})",
            data=er,
            duration_ms=float(er.get("duration_ms", 0.0)),
        ))

        # --- causal_inference ---
        ci = trace.get("causal_inference", {})
        steps.append(TraceStep(
            stage="causal_inference",
            summary=f"Fired {ci.get('rules_fired', 0)} causal rules; "
                    f"{len(ci.get('forward_chains', []))} forward chains",
            data=ci,
            duration_ms=float(ci.get("duration_ms", 0.0)),
        ))

        # --- global_workspace ---
        gw = trace.get("global_workspace", {})
        winner = gw.get("winning_coalition", trace.get("winner", "DEFAULT"))
        steps.append(TraceStep(
            stage="global_workspace",
            summary=f"Winner: {winner} "
                    f"(KLE={gw.get('kle_uncertainty', 0.0):.4f})",
            data=gw,
            duration_ms=float(gw.get("duration_ms", 0.0)),
        ))

        # --- planning ---
        pl = trace.get("planning", {})
        plan_steps = pl.get("plan_steps", [])
        triggered = pl.get("triggered", False)
        n_steps = len(plan_steps)
        plan_summary = (
            f"{n_steps} step(s): {plan_steps[0]!r}..." if n_steps > 1
            else (f"1 step: {plan_steps[0]!r}" if n_steps == 1 else "[]")
        )
        steps.append(TraceStep(
            stage="planning",
            summary=f"Plan ({plan_summary})" if triggered else "not triggered",
            data=pl,
            duration_ms=float(pl.get("duration_ms", 0.0)),
        ))

        # --- self_model ---
        sm = trace.get("self_model", {})
        steps.append(TraceStep(
            stage="self_model",
            summary=f"Confidence calibration error={sm.get('calibration_error', 0.0):.4f}; "
                    f"novelty={sm.get('novelty_score', 0.0):.4f}",
            data=sm,
            duration_ms=float(sm.get("duration_ms", 0.0)),
        ))

        # --- societal_context ---
        sc = trace.get("societal_context_stage", {})
        steps.append(TraceStep(
            stage="societal_context",
            summary=f"Community: {sc.get('active_community', 'N/A')}; "
                    f"n_concepts={sc.get('n_concepts', 0)}; "
                    f"bonds={sc.get('bond_count', 0)}",
            data=sc,
            duration_ms=float(sc.get("duration_ms", 0.0)),
        ))

        # --- response_generation ---
        rg = trace.get("response_generation", {})
        steps.append(TraceStep(
            stage="response_generation",
            summary=f"Strategy: {rg.get('strategy', 'retrieval')}; "
                    f"{len(rg.get('source_sentences', []))} source sentences",
            data=rg,
            duration_ms=float(rg.get("duration_ms", 0.0)),
        ))

        # Fill in any missing stages with N/A placeholders
        present = {s.stage for s in steps}
        for stage in REQUIRED_STAGES:
            if stage not in present:
                steps.append(TraceStep(
                    stage=stage,
                    summary="N/A",
                    data={},
                    duration_ms=0.0,
                ))

        # Sort steps in canonical order
        stage_order = {s: i for i, s in enumerate(REQUIRED_STAGES)}
        steps.sort(key=lambda s: stage_order.get(s.stage, 99))

        emotion_state = emo_label
        novelty_score = float(
            trace.get("self_model", {}).get("novelty_score", 0.0)
        )
        societal_ctx = dict(trace.get("societal_context_stage", {}))

        return cls(
            query_id=query_id,
            input_text=input_text,
            input_modality=input_modality,
            timestamp=timestamp,
            total_duration_ms=round(total_duration_ms, 3),
            steps=steps,
            final_action=final_action,
            confidence=round(confidence, 4),
            emotion_state=emotion_state,
            novelty_score=round(novelty_score, 4),
            societal_context=societal_ctx,
            rust_used=rust_used,
        )

    # ------------------------------------------------------------------
    # Stage lookup
    # ------------------------------------------------------------------

    def get_step(self, stage: str) -> Optional[TraceStep]:
        """Return the TraceStep for the given stage, or None."""
        for s in self.steps:
            if s.stage == stage:
                return s
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "input_text": self.input_text,
            "input_modality": self.input_modality,
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "final_action": self.final_action,
            "confidence": self.confidence,
            "emotion_state": self.emotion_state,
            "novelty_score": self.novelty_score,
            "societal_context": self.societal_context,
            "rust_used": self.rust_used,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, path: str) -> None:
        """Write the trace as JSON to *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def summary(self) -> str:
        """One-liner summary."""
        return (
            f"[{self.query_id}] '{self.input_text[:40]}...' → {self.final_action} "
            f"(conf={self.confidence:.2f}, {len(self.steps)} stages, "
            f"{self.total_duration_ms:.1f}ms)"
        )

    def to_markdown(self) -> str:
        """Full human-readable Markdown report."""
        lines = [
            f"# ThoughtTrace — {self.query_id}",
            f"",
            f"**Input:** `{self.input_text}`",
            f"**Modality:** {self.input_modality}",
            f"**Timestamp:** {self.timestamp}",
            f"**Total duration:** {self.total_duration_ms:.1f} ms",
            f"**Final action:** {self.final_action}",
            f"**Confidence:** {self.confidence:.4f}",
            f"**Emotion:** {self.emotion_state}",
            f"**Novelty:** {self.novelty_score:.4f}",
            f"**Rust used:** {self.rust_used}",
            f"",
            f"## Cognitive Stages",
            f"",
        ]
        for step in self.steps:
            lines.append(f"### {step.stage}")
            lines.append(f"*{step.summary}*  ")
            lines.append(f"Duration: {step.duration_ms:.3f} ms")
            if step.data:
                lines.append(f"```json")
                lines.append(json.dumps(step.data, indent=2, default=str))
                lines.append(f"```")
            lines.append("")
        return "\n".join(lines)
