"""
Tests for NSCK V18 Semantic Bootstrap
======================================
Tests SemanticBootstrapper and associated DistributionalCodebook enhancements.
"""
from __future__ import annotations

import os
import sys
import pytest

# Ensure NSCK is importable (conftest.py in nsck/ adds it to sys.path automatically
# when tests are run with `python -m pytest nsck/tests/`; this guard handles direct
# execution from other working directories).
_nsck_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _nsck_root not in sys.path:
    sys.path.insert(0, _nsck_root)

from python.core.language.semantic_bootstrap import SemanticBootstrapper
from python.core.language.distributional_semantics import DistributionalCodebook


class TestSemanticBootstrapper:

    def test_build_codebook_corpus_strategy(self):
        """strategy='corpus' must always work (no external deps)."""
        cb = SemanticBootstrapper.build_codebook(strategy="corpus")
        assert len(cb._codebook) >= 100

    def test_build_codebook_auto_fallback(self):
        """strategy='auto' must not crash even without sentence-transformers."""
        cb = SemanticBootstrapper.build_codebook(strategy="auto")
        assert len(cb._codebook) >= 100

    def test_benchmark_returns_dict(self):
        """benchmark() returns a dict with float similarities in [0, 1]."""
        cb = SemanticBootstrapper.build_codebook(strategy="corpus")
        results = SemanticBootstrapper.benchmark(cb)
        assert isinstance(results, dict)
        for (w1, w2), sim in results.items():
            assert isinstance(w1, str)
            assert isinstance(w2, str)
            assert 0.0 <= sim <= 1.0

    def test_save_and_load_roundtrip(self, tmp_path):
        """Similarities must be preserved after save → load roundtrip."""
        cb = SemanticBootstrapper.build_codebook(strategy="corpus")
        path = str(tmp_path / "test_codebook.pkl")
        SemanticBootstrapper.save_codebook(cb, path)
        cb2 = SemanticBootstrapper.load_codebook(path)
        for word in ["brain", "memory", "cat", "dog"]:
            hv1 = cb.get_hv(word)
            hv2 = cb2.get_hv(word)
            if hv1 is not None and hv2 is not None:
                assert hv1.similarity(hv2) > 0.99

    def test_cognitive_vocabulary_coverage(self):
        """COGNITIVE_VOCABULARY must have ≥ 400 entries and include key words."""
        from python.core.language.cognitive_vocabulary import COGNITIVE_VOCABULARY
        assert len(COGNITIVE_VOCABULARY) >= 400
        assert "brain" in COGNITIVE_VOCABULARY
        assert "memory" in COGNITIVE_VOCABULARY
        assert "learn" in COGNITIVE_VOCABULARY

    def test_build_default_strategy_corpus(self):
        """build_default(strategy='corpus') must return a valid codebook."""
        cb = DistributionalCodebook.build_default(strategy="corpus")
        assert len(cb._codebook) >= 100

    def test_build_semantic_classmethod(self):
        """build_semantic() must return a valid codebook (bridge → corpus fallback)."""
        cb = DistributionalCodebook.build_semantic()
        assert len(cb._codebook) >= 100

    def test_build_default_auto_strategy(self):
        """build_default() with strategy='auto' must not crash."""
        cb = DistributionalCodebook.build_default(strategy="auto")
        assert len(cb._codebook) >= 100

    def test_corpus_strategy_words_in_corpus(self):
        """Words from BUILTIN_CORPUS should be present in corpus-strategy codebook."""
        cb = SemanticBootstrapper.build_codebook(strategy="corpus")
        # "brain" and "memory" both appear in BUILTIN_CORPUS
        assert cb.get_hv("brain") is not None
        assert cb.get_hv("memory") is not None

    def test_similarity_nonzero_for_related_words(self):
        """Related words should have positive similarity after corpus build."""
        cb = SemanticBootstrapper.build_codebook(strategy="corpus")
        sim = cb.similarity("brain", "memory")
        assert sim > 0.0

    def test_load_codebook_missing_file_raises(self, tmp_path):
        """load_codebook with a non-existent path raises an exception."""
        with pytest.raises(Exception):
            SemanticBootstrapper.load_codebook(str(tmp_path / "nonexistent.pkl"))

    def test_build_default_backward_compat(self):
        """build_default() with no arguments (default strategy='auto') still works."""
        cb = DistributionalCodebook.build_default()
        assert isinstance(cb, DistributionalCodebook)
        assert len(cb._codebook) > 0
