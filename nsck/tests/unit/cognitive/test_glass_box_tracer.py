"""Tests for GlassBoxTracer (V17)."""
import pytest
from python.core.cognitive.glass_box_tracer import GlassBoxTracer, TraceEntry, DecisionTrace


def test_begin_end_decision():
    tracer = GlassBoxTracer()
    did = tracer.begin_decision()
    assert did.startswith("D")
    trace = tracer.end_decision()
    assert isinstance(trace, DecisionTrace)
    assert trace.decision_id == did


def test_record_entry():
    tracer = GlassBoxTracer()
    tracer.begin_decision("test-1")
    tracer.record("MyModule", "did something", confidence=0.9)
    trace = tracer.end_decision()
    assert len(trace.entries) == 1
    assert trace.entries[0].module == "MyModule"
    assert trace.entries[0].confidence == pytest.approx(0.9)


def test_span_context_manager():
    tracer = GlassBoxTracer()
    tracer.begin_decision()
    with tracer.span("perception"):
        tracer.record("TextAdapter", "encoded input")
    with tracer.span("reasoning"):
        tracer.record("CognitiveEngine", "selected action")
    trace = tracer.end_decision()
    spans = [e.span for e in trace.entries]
    assert "perception" in spans
    assert "reasoning" in spans


def test_elapsed_ms():
    tracer = GlassBoxTracer()
    tracer.begin_decision()
    trace = tracer.end_decision()
    assert trace.elapsed_ms is not None
    assert trace.elapsed_ms >= 0


def test_last_trace():
    tracer = GlassBoxTracer()
    tracer.begin_decision("X")
    tracer.end_decision()
    last = tracer.last_trace()
    assert last is not None
    assert last.decision_id == "X"


def test_history_max():
    tracer = GlassBoxTracer(max_history=3)
    for i in range(10):
        tracer.begin_decision(f"D{i}")
        tracer.end_decision()
    assert len(tracer.history()) == 3


def test_disabled_tracer_no_record():
    tracer = GlassBoxTracer(enabled=False)
    tracer.begin_decision("off")
    tracer.record("M", "msg")
    trace = tracer.end_decision()
    assert len(trace.entries) == 0


def test_format_trace():
    tracer = GlassBoxTracer()
    tracer.begin_decision("FMT")
    tracer.record("Mod", "step one", confidence=0.5)
    trace = tracer.end_decision()
    text = GlassBoxTracer.format_trace(trace)
    assert "FMT" in text
    assert "Mod" in text
    assert "step one" in text


def test_format_trace_none():
    text = GlassBoxTracer.format_trace(None)
    assert "no trace" in text.lower()


def test_end_without_begin():
    tracer = GlassBoxTracer()
    result = tracer.end_decision()
    assert result is None


def test_export_active():
    tracer = GlassBoxTracer()
    tracer.begin_decision("active")
    active = tracer.export()
    assert active is not None
    tracer.end_decision()
    assert tracer.export() is None
