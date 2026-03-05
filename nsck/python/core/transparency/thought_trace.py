"""
ThoughtTrace — Structured transparency report for NSCK cognitive decisions (V31).

Every NSCKSubstrate.process() and NSCKSubstrate.ingest() call can return a
ThoughtTrace covering all 11 cognitive stages with RICH data populated from
the cognitive engine's V31 trace enrichment.
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

    V31: All stages are now richly populated from the cognitive engine's
    live data — semantic search results, concept extraction, episodic recall,
    causal chains, emotion analysis, planning steps, etc.
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
        """Build a ThoughtTrace from a raw CognitiveEngine trace dict (V31)."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        steps: List[TraceStep] = []

        # ── encoding ──────────────────────────────────────────────────
        enc = trace.get("encoding", {})
        modality = enc.get("modality", input_modality)
        projector = enc.get("projector_used", "hash")
        dim = enc.get("hv_dimension", 10240)
        adapter = enc.get("adapter_name", "inline")
        transplant_domains = enc.get("transplant_domains", [])
        enc_summary = (
            f"Modality={modality}, HV-dim={dim}, projector={projector}, "
            f"adapter={adapter}"
        )
        if transplant_domains:
            enc_summary += f", transplant_domains={transplant_domains}"
        steps.append(TraceStep(
            stage="encoding",
            summary=enc_summary,
            data=enc,
            duration_ms=float(enc.get("encoding_time_ms", 0.0)),
        ))

        # ── emotion ────────────────────────────────────────────────────
        emo = trace.get("emotion", {})
        emo_label = emo.get("label", "neutral")
        valence = emo.get("valence", 0.0)
        arousal = emo.get("arousal", 0.0)
        emo_summary = (
            f"Label={emo_label}  |  valence={valence:+.2f}  |  arousal={arousal:.2f}"
        )
        steps.append(TraceStep(
            stage="emotion",
            summary=emo_summary,
            data=emo,
            duration_ms=float(emo.get("duration_ms", 0.0)),
        ))

        # ── concept_extraction ─────────────────────────────────────────
        cex = trace.get("concept_extraction", {})
        concepts = cex.get("concepts", [])
        svo_triples = cex.get("svo_triples", [])
        intent = cex.get("intent", "unknown")
        entities = cex.get("entities", [])
        frames = cex.get("frames", {})
        n_concepts = cex.get("n_concepts", len(concepts))
        parse_conf = cex.get("parse_confidence", 0.0)
        # Build rich summary
        svo_str = ""
        if svo_triples:
            t = svo_triples[0]
            svo_str = f" | SVO=[{t.get('S','?')} → {t.get('V','?')} → {t.get('O','?')}]"
        frames_str = ""
        if frames:
            frame_parts = [f"{k}={v}" for k, v in list(frames.items())[:3] if v]
            if frame_parts:
                frames_str = " | frames=[" + ", ".join(frame_parts) + "]"
        cex_summary = (
            f"intent={intent}, n_concepts={n_concepts}, "
            f"entities={entities[:5]}, conf={parse_conf:.2f}"
            f"{svo_str}{frames_str}"
        )
        steps.append(TraceStep(
            stage="concept_extraction",
            summary=cex_summary,
            data=cex,
            duration_ms=float(cex.get("duration_ms", 0.0)),
        ))

        # ── semantic_search ────────────────────────────────────────────
        ss = trace.get("semantic_search", {})
        top_matches = ss.get("top_matches", [])
        n_total = ss.get("n_total_concepts", 0)
        best_sim = ss.get("best_similarity", 0.0)
        if top_matches:
            top3 = ", ".join(
                f"{m[0]}({m[1]:.3f})" for m in top_matches[:3]
            )
            ss_summary = (
                f"Found {len(top_matches)}/{n_total} concepts | "
                f"best_sim={best_sim:.4f} | top: {top3}"
            )
        else:
            ss_summary = f"No semantic matches found (total_concepts={n_total})"
        steps.append(TraceStep(
            stage="semantic_search",
            summary=ss_summary,
            data=ss,
            duration_ms=float(ss.get("duration_ms", 0.0)),
        ))

        # ── episodic_recall ────────────────────────────────────────────
        er = trace.get("episodic_recall", {})
        n_recalled = er.get("n_episodes_recalled", 0)
        best_ep_sim = er.get("best_similarity", 0.0)
        recalled_eps = er.get("recalled", [])
        if recalled_eps:
            ep_str = ", ".join(
                f"{ep.get('action','?')}(sim={ep.get('similarity',0):.3f},r={ep.get('reward',0):.2f})"
                for ep in recalled_eps[:3]
            )
            er_summary = (
                f"Recalled {n_recalled} episodes | best_sim={best_ep_sim:.4f} | "
                f"top: {ep_str}"
            )
        else:
            er_summary = f"No past episodes recalled (episodic_mem_empty or low_sim)"
        steps.append(TraceStep(
            stage="episodic_recall",
            summary=er_summary,
            data=er,
            duration_ms=float(er.get("duration_ms", 0.0)),
        ))

        # ── causal_inference ───────────────────────────────────────────
        ci = trace.get("causal_inference", {})
        rules_fired = ci.get("rules_fired", 0)
        chains = ci.get("forward_chains", [])
        applicable_rules = ci.get("applicable_rules", [])
        cg_nodes = ci.get("causal_graph_nodes", 0)
        ci_summary = (
            f"rules_fired={rules_fired}, "
            f"applicable={applicable_rules[:3]}, "
            f"forward_chains={len(chains)}, "
            f"causal_graph_nodes={cg_nodes}"
        )
        if chains:
            chain_str = "; ".join(
                f"{ch.get('cause','?')}→[{','.join(e['effect'] for e in ch.get('effects',[])[:2])}]"
                for ch in chains[:2]
            )
            ci_summary += f" | chains: {chain_str}"
        steps.append(TraceStep(
            stage="causal_inference",
            summary=ci_summary,
            data=ci,
            duration_ms=float(ci.get("duration_ms", 0.0)),
        ))

        # ── global_workspace ───────────────────────────────────────────
        gw = trace.get("global_workspace", {})
        gw_winner = gw.get("winning_coalition", trace.get("winner", "DEFAULT"))
        gw_action = gw.get("winning_action", "")
        gw_activation = gw.get("winning_activation", 0.0)
        n_coals = gw.get("n_coalitions", 0)
        coal_sources = gw.get("coalition_sources", [])
        kle = gw.get("kle_uncertainty", 0.0)
        sys_used = gw.get("system_used", "unknown")
        gw_summary = (
            f"winner={gw_winner} (activation={gw_activation:.3f}) | "
            f"n_coalitions={n_coals} [{', '.join(coal_sources)}] | "
            f"KLE={kle:.4f} | system={sys_used}"
        )
        if gw_action and gw_action != gw_winner:
            gw_summary += f" | action='{gw_action[:60]}'"
        steps.append(TraceStep(
            stage="global_workspace",
            summary=gw_summary,
            data=gw,
            duration_ms=float(gw.get("duration_ms", 0.0)),
        ))

        # ── planning ───────────────────────────────────────────────────
        pl = trace.get("planning", {})
        plan_steps = pl.get("plan_steps", [])
        triggered = pl.get("triggered", False)
        planner_active = pl.get("planner_active", False)
        goal = pl.get("goal")
        if triggered and plan_steps:
            plan_summary = (
                f"ACTIVE | goal={goal} | {len(plan_steps)} steps: "
                f"{' → '.join(str(s) for s in plan_steps[:4])}"
            )
        elif planner_active:
            plan_summary = f"Planner active (goal={goal}) but no plan generated yet"
        else:
            plan_summary = "Not triggered (no mission goal set)"
        steps.append(TraceStep(
            stage="planning",
            summary=plan_summary,
            data=pl,
            duration_ms=float(pl.get("duration_ms", 0.0)),
        ))

        # ── self_model ─────────────────────────────────────────────────
        sm = trace.get("self_model", {})
        cal_err = sm.get("calibration_error", 0.0)
        novelty = sm.get("novelty_score", 0.0)
        task_conf = sm.get("task_confidence", confidence)
        n_novel = sm.get("n_novel_concepts", 0)
        sm_summary = (
            f"calibration_error={cal_err:.4f} | task_confidence={task_conf:.3f} | "
            f"novelty={novelty:.4f} | n_novel_concepts={n_novel}"
        )
        if novelty > 0.5:
            sm_summary += " ⚡ HIGH-NOVELTY INPUT"
        elif cal_err > 0.3:
            sm_summary += " ⚠ CONFIDENCE MISCALIBRATION"
        steps.append(TraceStep(
            stage="self_model",
            summary=sm_summary,
            data=sm,
            duration_ms=float(sm.get("duration_ms", 0.0)),
        ))

        # ── societal_context ───────────────────────────────────────────
        sc = trace.get("societal_context_stage", {})
        community = sc.get("active_community") or sc.get("active_cluster")
        n_soc_concepts = sc.get("n_concepts", sc.get("activated_count", 0))
        bonds = sc.get("bond_count", 0)
        top_soc = sc.get("top_concept")
        top_soc_sim = sc.get("top_similarity", 0.0)
        matches = sc.get("matches", [])
        if community is not None or n_soc_concepts > 0:
            sc_summary = (
                f"community={community} | concepts_activated={n_soc_concepts} | "
                f"bonds={bonds}"
            )
            if top_soc:
                sc_summary += f" | top_concept={top_soc}(sim={top_soc_sim:.3f})"
            if matches:
                match_str = ", ".join(
                    f"{m.get('concept_id','?')}({m.get('similarity',0):.3f})"
                    for m in matches[:3]
                )
                sc_summary += f" | matches=[{match_str}]"
        else:
            sc_summary = "Societal router inactive (requires NSCKConfig.societal() + init_societal_world())"
        steps.append(TraceStep(
            stage="societal_context",
            summary=sc_summary,
            data=sc,
            duration_ms=float(sc.get("duration_ms", 0.0)),
        ))

        # ── response_generation ─────────────────────────────────────────
        rg = trace.get("response_generation", {})
        strategy = rg.get("strategy", "retrieval")
        source_sentences = rg.get("source_sentences", [])
        rg_conf = rg.get("confidence", confidence)
        rg_intent = rg.get("intent", intent)
        rg_summary = (
            f"strategy={strategy} | intent={rg_intent} | "
            f"confidence={rg_conf:.3f} | sources={len(source_sentences)}"
        )
        if source_sentences:
            rg_summary += f" | top: {str(source_sentences[0])[:60]}"
        steps.append(TraceStep(
            stage="response_generation",
            summary=rg_summary,
            data=rg,
            duration_ms=float(rg.get("duration_ms", 0.0)),
        ))

        # Fill in any missing stages with N/A placeholders
        present = {s.stage for s in steps}
        for stage in REQUIRED_STAGES:
            if stage not in present:
                steps.append(TraceStep(
                    stage=stage,
                    summary="N/A — stage not populated",
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
        emo = self.emotion_state
        novelty_tag = " ⚡novel" if self.novelty_score > 0.3 else ""
        return (
            f"[{self.query_id}] '{self.input_text[:40]}...' → {self.final_action} "
            f"(conf={self.confidence:.2f}, emotion={emo}{novelty_tag}, "
            f"{len(self.steps)} stages, {self.total_duration_ms:.1f}ms)"
        )

    def to_markdown(self) -> str:
        """Full human-readable Markdown report."""
        lines = [
            f"# 🧠 ThoughtTrace — {self.query_id}",
            f"",
            f"| Field | Value |",
            f"|---|---|",
            f"| **Input** | `{self.input_text}` |",
            f"| **Modality** | {self.input_modality} |",
            f"| **Timestamp** | {self.timestamp} |",
            f"| **Total duration** | {self.total_duration_ms:.1f} ms |",
            f"| **Final action** | `{self.final_action}` |",
            f"| **Confidence** | {self.confidence:.4f} |",
            f"| **Emotion** | {self.emotion_state} |",
            f"| **Novelty** | {self.novelty_score:.4f} |",
            f"| **Rust accelerated** | {'✅ Yes' if self.rust_used else '❌ No'} |",
            f"",
            f"## 🔍 Cognitive Pipeline (11 Stages)",
            f"",
        ]
        _icons = {
            "encoding":           "📥",
            "emotion":            "💭",
            "concept_extraction": "🏷️",
            "semantic_search":    "🔎",
            "episodic_recall":    "💾",
            "causal_inference":   "🔗",
            "global_workspace":   "🏆",
            "planning":           "📋",
            "self_model":         "🪞",
            "societal_context":   "🌐",
            "response_generation":"📤",
        }
        for i, step in enumerate(self.steps, 1):
            icon = _icons.get(step.stage, "⚙️")
            lines.append(f"### {i}. {icon} `{step.stage}`")
            lines.append(f"**{step.summary}**")
            lines.append(f"*(duration: {step.duration_ms:.3f} ms)*")
            if step.data and step.stage not in ("encoding",):
                # Show a condensed version of data — not full JSON dump to keep readable
                compact = {k: v for k, v in step.data.items()
                           if k not in ("duration_ms",) and v not in (None, [], {}, "")}
                if compact:
                    lines.append(f"<details><summary>Raw data</summary>")
                    lines.append(f"")
                    lines.append(f"```json")
                    lines.append(json.dumps(compact, indent=2, default=str))
                    lines.append(f"```")
                    lines.append(f"")
                    lines.append(f"</details>")
            lines.append("")

        # Summary table
        lines.append("## 📊 Stage Summary")
        lines.append("")
        lines.append("| Stage | Summary |")
        lines.append("|---|---|")
        for step in self.steps:
            short = step.summary[:100].replace("|", "\\|")
            lines.append(f"| `{step.stage}` | {short} |")
        lines.append("")

        return "\n".join(lines)

