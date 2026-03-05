"""Tests for ThoughtTrace transparency module (V30)."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _make_trace(n_steps: int = 0) -> "ThoughtTrace":
    """Helper: build a minimal ThoughtTrace with all required stages."""
    from python.core.transparency.thought_trace import ThoughtTrace, REQUIRED_STAGES
    return ThoughtTrace.from_cognitive_trace(
        query_id="test_q_001",
        input_text="What is photosynthesis?",
        input_modality="text",
        trace={},
        final_action="explain",
        confidence=0.75,
        total_duration_ms=12.3,
        rust_used=False,
    )


class TestThoughtTrace:
    def test_thought_trace_has_all_11_stages(self):
        """ThoughtTrace must contain exactly the 11 required cognitive stages."""
        from python.core.transparency.thought_trace import REQUIRED_STAGES
        trace = _make_trace()
        stage_names = {s.stage for s in trace.steps}
        for required in REQUIRED_STAGES:
            assert required in stage_names, (
                f"Missing required stage '{required}' in ThoughtTrace.steps"
            )
        assert len(REQUIRED_STAGES) == 11, "Expected exactly 11 required stages"

    def test_thought_trace_to_dict_roundtrip(self):
        """ThoughtTrace.to_dict() must produce a JSON-serialisable dict with all keys."""
        import json
        trace = _make_trace()
        d = trace.to_dict()
        # Should be JSON-serialisable
        serialised = json.dumps(d, default=str)
        assert len(serialised) > 0
        # Required keys
        for key in ("query_id", "input_text", "input_modality", "timestamp",
                    "total_duration_ms", "steps", "final_action", "confidence"):
            assert key in d, f"Key '{key}' missing from to_dict()"
        # steps should be a list of dicts with stage/summary/data/duration_ms
        assert isinstance(d["steps"], list)
        assert len(d["steps"]) == 11
        for step in d["steps"]:
            for k in ("stage", "summary", "data", "duration_ms"):
                assert k in step, f"Key '{k}' missing from step dict"

    def test_thought_trace_to_markdown_not_empty(self):
        """ThoughtTrace.to_markdown() must return a non-empty string."""
        trace = _make_trace()
        md = trace.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100, "Markdown output is suspiciously short"
        assert "ThoughtTrace" in md
        # All 11 stage names should appear as headings
        from python.core.transparency.thought_trace import REQUIRED_STAGES
        for stage in REQUIRED_STAGES:
            assert stage in md, f"Stage '{stage}' not in Markdown output"

    def test_thought_trace_attached_to_substrate_result(self):
        """NSCKSubstrate.process() with enable_transparency=True attaches a ThoughtTrace."""
        from python.core.substrate import NSCKSubstrate, SubstrateResult
        from python.core.integration.config import NSCKConfig
        cfg = NSCKConfig()
        cfg.enable_transparency = True
        substrate = NSCKSubstrate(cfg)
        substrate.register_task("test_transparency")
        result = substrate.process("photosynthesis uses sunlight", "test_transparency")
        assert isinstance(result, SubstrateResult)
        # thought_trace should be populated
        assert result.thought_trace is not None, (
            "Expected thought_trace to be populated when enable_transparency=True"
        )
        from python.core.transparency.thought_trace import ThoughtTrace
        assert isinstance(result.thought_trace, ThoughtTrace)
        # Must have all 11 stages
        from python.core.transparency.thought_trace import REQUIRED_STAGES
        stage_names = {s.stage for s in result.thought_trace.steps}
        for req in REQUIRED_STAGES:
            assert req in stage_names

    def test_thought_trace_summary(self):
        """ThoughtTrace.summary() returns a one-line string."""
        trace = _make_trace()
        summary = trace.summary()
        assert isinstance(summary, str)
        assert "\n" not in summary, "summary() should be a single line"
        assert "explain" in summary or "q_" in summary.lower() or "test_q" in summary

    def test_thought_trace_get_step(self):
        """ThoughtTrace.get_step() returns the correct TraceStep."""
        trace = _make_trace()
        step = trace.get_step("encoding")
        assert step is not None
        assert step.stage == "encoding"
        assert trace.get_step("nonexistent_stage") is None
