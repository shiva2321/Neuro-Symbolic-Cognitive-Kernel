"""Unit tests for TransplantPipeline."""
from __future__ import annotations

import sys
import os
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from python.core.transplant.pipeline import TransplantPipeline
from python.core.transplant.validator import TransplantReport


# ---------------------------------------------------------------------------
# Minimal mock model
# ---------------------------------------------------------------------------

class _FakeParam:
    def __init__(self, arr):
        self._arr = arr.astype(np.float32)
    @property
    def data(self): return self
    def detach(self): return self
    def numpy(self): return self._arr
    @property
    def shape(self): return self._arr.shape
    @property
    def ndim(self): return self._arr.ndim


class _FakeEmbedding:
    def __init__(self, vocab, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.weight = _FakeParam(rng.standard_normal((vocab, dim)))


class _MockLM:
    """Minimal mock with word_embeddings (transformer LM pattern)."""
    def __init__(self, vocab=50, dim=16, seed=0):
        self.word_embeddings = _FakeEmbedding(vocab, dim, seed)
    def named_parameters(self):
        yield "word_embeddings.weight", self.word_embeddings.weight


class _StructuredMockLM:
    """Mock LM with clustered embeddings for quality testing."""
    def __init__(self, n_clusters=4, per_cluster=15, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.standard_normal((n_clusters, dim)) * 4.0
        emb = np.vstack([
            centers[k] + rng.standard_normal((per_cluster, dim)) * 0.2
            for k in range(n_clusters)
        ]).astype(np.float32)
        self._emb = emb
        self.word_embeddings = type("_WE", (), {
            "weight": _FakeParam(emb)
        })()
    def named_parameters(self):
        yield "word_embeddings.weight", self.word_embeddings.weight


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransplantPipeline:

    def test_run_returns_report(self):
        model = _MockLM(vocab=30, dim=16)
        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="test",
            strategy="random",
            calibration_epochs=0,
        )
        assert isinstance(report, TransplantReport)

    def test_run_with_svd_strategy(self):
        model = _MockLM(vocab=30, dim=16)
        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="svd_test",
            strategy="svd_factored",
            calibration_epochs=0,
        )
        assert isinstance(report, TransplantReport)

    def test_run_with_learned_strategy(self):
        model = _MockLM(vocab=20, dim=8)
        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="learned_test",
            strategy="learned",
            calibration_epochs=0,
        )
        assert isinstance(report, TransplantReport)

    def test_unknown_strategy_raises(self):
        model = _MockLM(vocab=10, dim=4)
        pipe = TransplantPipeline()
        with pytest.raises(ValueError, match="Unknown projection strategy"):
            pipe.run(model, domain_name="x", strategy="unknown_strategy")

    def test_invalid_model_raises_value_error(self):
        """A model without named_parameters should raise ValueError."""
        class _Bad:
            pass
        pipe = TransplantPipeline()
        with pytest.raises(ValueError):
            pipe.run(_Bad(), domain_name="bad")

    def test_projector_stored_after_run(self):
        model = _MockLM(vocab=20, dim=8)
        pipe = TransplantPipeline()
        pipe.run(model, domain_name="mydom", strategy="random", calibration_epochs=0)
        assert "mydom" in pipe._projectors

    def test_calibration_epochs_honored(self):
        model = _MockLM(vocab=20, dim=8)
        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="cal",
            strategy="random",
            calibration_epochs=1,
        )
        assert report.calibration_quality_curve is not None
        assert len(report.calibration_quality_curve) >= 1

    def test_calibration_skipped_when_zero(self):
        model = _MockLM(vocab=20, dim=8)
        pipe = TransplantPipeline()
        report = pipe.run(
            model, domain_name="nocal",
            strategy="random",
            calibration_epochs=0,
        )
        assert report.calibration_quality_curve is None

    def test_save_load_knowledge_pack(self):
        model = _MockLM(vocab=15, dim=8)
        pipe = TransplantPipeline()
        with tempfile.NamedTemporaryFile(suffix=".kp", delete=False) as f:
            path = f.name
        try:
            pipe.run(
                model, domain_name="savetest",
                strategy="random",
                calibration_epochs=0,
                save_pack_path=path,
            )
            from python.core.integration.knowledge_pack import KnowledgePack
            pack = KnowledgePack.load(path)
            assert pack.name == "savetest"
            assert len(pack._concepts) == 15
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_integration_injects_concepts(self):
        """Concepts should be injected into a mock cognitive engine."""
        model = _StructuredMockLM(n_clusters=2, per_cluster=5, dim=8)
        pipe = TransplantPipeline()

        # Very relaxed thresholds so validation passes
        from python.core.transplant.validator import TransplantValidator
        pipe._validator = TransplantValidator(
            rho_threshold=0.0,
            recall10_threshold=0.0,
            recall50_threshold=0.0,
            ari_threshold=0.0,
        )

        class _MockMemory:
            def __init__(self):
                self.concept_hvs = {}
                self._added = []
            def add_concept(self, name, props):
                self._added.append(name)
            def add_relation(self, src, rel, dst):
                pass

        class _MockEngine:
            def __init__(self):
                self.semantic_memory = _MockMemory()

        engine = _MockEngine()
        pipe.run(
            model, domain_name="inject",
            strategy="random",
            calibration_epochs=0,
            cognitive_engine=engine,
        )
        assert len(engine.semantic_memory._added) == 10  # 2 clusters × 5

    def test_validation_failure_skips_integration(self):
        """If validation fails, concepts must NOT be injected."""
        model = _MockLM(vocab=10, dim=4)
        pipe = TransplantPipeline()
        # Strict thresholds to force failure
        from python.core.transplant.validator import TransplantValidator
        pipe._validator = TransplantValidator(
            rho_threshold=0.9999,
            recall10_threshold=0.9999,
            recall50_threshold=0.9999,
            ari_threshold=0.9999,
        )

        class _MockMemory:
            def __init__(self):
                self.concept_hvs = {}
                self._added = []
            def add_concept(self, name, props):
                self._added.append(name)
            def add_relation(self, *_):
                pass

        class _MockEngine:
            def __init__(self):
                self.semantic_memory = _MockMemory()

        engine = _MockEngine()
        pipe.run(
            model, domain_name="noinject",
            strategy="random",
            calibration_epochs=0,
            cognitive_engine=engine,
        )
        assert len(engine.semantic_memory._added) == 0

    def test_config_flags_respected(self):
        class _Cfg:
            transplant_rho_threshold = 0.0
            transplant_recall10_threshold = 0.0
            transplant_recall50_threshold = 0.0
            transplant_ari_threshold = 0.0

        pipe = TransplantPipeline(config=_Cfg())
        model = _MockLM(vocab=10, dim=4)
        report = pipe.run(model, domain_name="cfg", strategy="random", calibration_epochs=0)
        assert report.passed is True  # zero thresholds → always pass
