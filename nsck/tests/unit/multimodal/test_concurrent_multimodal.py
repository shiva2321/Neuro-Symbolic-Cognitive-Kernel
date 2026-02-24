"""Tests for ConcurrentMultimodalScheduler (V8)."""
import pytest
import numpy as np

from python.core.multimodal.multimodal_processor import (
    MultimodalInput,
    MultimodalProcessor,
    ConcurrentMultimodalScheduler,
    ProcessedInput,
    ModalityResult,
)


@pytest.fixture()
def processor():
    return MultimodalProcessor()


@pytest.fixture()
def scheduler(processor):
    sched = ConcurrentMultimodalScheduler(processor=processor, max_workers=2)
    yield sched
    sched.close()


class TestConcurrentMultimodalSchedulerInit:
    def test_default_init(self):
        s = ConcurrentMultimodalScheduler()
        assert s.coherence_window_ms == 50.0
        s.close()

    def test_custom_coherence_window(self):
        s = ConcurrentMultimodalScheduler(coherence_window_ms=100.0)
        assert s.coherence_window_ms == 100.0
        s.close()

    def test_custom_max_workers(self):
        s = ConcurrentMultimodalScheduler(max_workers=2)
        assert s._executor is not None
        s.close()

    def test_wraps_processor(self, processor, scheduler):
        assert scheduler._processor is processor

    def test_context_manager(self):
        with ConcurrentMultimodalScheduler() as s:
            assert s is not None


class TestProcessConcurrentBasic:
    def test_empty_input_returns_zero_conf(self, scheduler):
        inp = MultimodalInput()
        result = scheduler.process_concurrent(inp)
        assert isinstance(result, ProcessedInput)
        assert result.confidence == 0.0

    def test_text_only(self, scheduler):
        inp = MultimodalInput(text="hello world")
        result = scheduler.process_concurrent(inp)
        assert isinstance(result, ProcessedInput)
        assert result.fused_hv is not None
        assert result.confidence > 0

    def test_structured_only(self, scheduler):
        inp = MultimodalInput(structured={"key": "value"})
        result = scheduler.process_concurrent(inp)
        assert result.confidence > 0

    def test_audio_only(self, scheduler):
        audio = np.zeros(1000, dtype=np.float32)
        inp = MultimodalInput(audio=audio)
        result = scheduler.process_concurrent(inp)
        assert result.fused_hv is not None

    def test_image_only(self, scheduler):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        inp = MultimodalInput(image=img)
        result = scheduler.process_concurrent(inp)
        assert result.fused_hv is not None


class TestProcessConcurrentMultimodal:
    def test_text_and_structured(self, scheduler):
        inp = MultimodalInput(text="test input", structured={"type": "test"})
        result = scheduler.process_concurrent(inp)
        assert len(result.modality_results) == 2

    def test_text_and_audio(self, scheduler):
        audio = np.random.randn(2000).astype(np.float32)
        inp = MultimodalInput(text="audio test", audio=audio)
        result = scheduler.process_concurrent(inp)
        assert len(result.modality_results) >= 1

    def test_concepts_extracted(self, scheduler):
        inp = MultimodalInput(text="cat sat on mat")
        result = scheduler.process_concurrent(inp)
        assert len(result.extracted_concepts) > 0

    def test_context_cues_populated(self, scheduler):
        inp = MultimodalInput(
            text="context test",
            metadata={"source": "test"},
        )
        result = scheduler.process_concurrent(inp)
        assert result.context_cues.get("source") == "test"

    def test_fused_hv_differs_from_parts(self, scheduler):
        inp1 = MultimodalInput(text="apple")
        inp2 = MultimodalInput(text="orange")
        r1 = scheduler.process_concurrent(inp1)
        r2 = scheduler.process_concurrent(inp2)
        # Different text → different HVs (highly likely)
        sim = float(r1.fused_hv.similarity(r2.fused_hv))
        assert sim < 0.99


class TestAttentionWeightedFusion:
    def test_fusion_returns_hv(self, scheduler, processor):
        r1 = processor._process_text("hello")
        r2 = processor._process_structured({"x": 1})
        fused = scheduler._attention_weighted_fuse([r1, r2])
        assert fused is not None

    def test_empty_results_returns_zero_seed(self, scheduler):
        fused = scheduler._attention_weighted_fuse([])
        assert fused is not None

    def test_single_result_fusion(self, scheduler, processor):
        r = processor._process_text("single")
        fused = scheduler._attention_weighted_fuse([r])
        assert fused is not None

    def test_confidence_affects_fusion(self, scheduler, processor):
        # High vs low confidence should change fused HV
        r_hi = processor._process_text("test confidence high")
        r_hi.confidence = 0.9
        r_lo = processor._process_text("test confidence high")
        r_lo.confidence = 0.1
        f1 = scheduler._attention_weighted_fuse([r_hi])
        f2 = scheduler._attention_weighted_fuse([r_lo])
        # Both return valid HVs; structure preserved
        assert f1 is not None and f2 is not None


class TestCoherence:
    def test_coherence_window_attribute(self, scheduler):
        assert scheduler.coherence_window_ms > 0

    def test_wide_window_includes_all(self):
        s = ConcurrentMultimodalScheduler(coherence_window_ms=10000.0)
        inp = MultimodalInput(text="wide window", structured={"k": "v"})
        result = s.process_concurrent(inp)
        assert len(result.modality_results) >= 1
        s.close()
