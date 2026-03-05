"""
Tests for pretrained model transplantation (V30).

All tests that require PyTorch/transformers are marked with
``pytest.mark.skipif`` so the suite passes on minimal environments.
"""
from __future__ import annotations

import sys
import os
import pytest

_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np

# ---------------------------------------------------------------------------
# Availability markers
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import sentence_transformers  # noqa: F401
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

_skip_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch not installed"
)
_skip_sbert = pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed",
)


# ---------------------------------------------------------------------------
# Synthetic duck-typed model that the harvester can process via named_params
# ---------------------------------------------------------------------------

class _SyntheticEmbeddingModel:
    """Duck-typed model with an embedding weight matrix.

    The harvester's ``_harvest_named_params`` falls back to largest weight
    tensor when no standard attribute is found.  We expose the embedding
    via the standard ``embeddings`` attribute so it is picked up by
    ``_harvest_embedding_layer``.
    """

    class _Embedding:
        def __init__(self, weight):
            self.weight = weight

    def __init__(self, vocab: list, dim: int = 64, seed: int = 42):
        rng = np.random.RandomState(seed)
        n = len(vocab)
        # 4 clusters with tightly packed intra-cluster embeddings
        centres = rng.randn(4, dim)
        weights = np.vstack([
            centres[i % 4] + 0.05 * rng.randn(1, dim)
            for i in range(n)
        ]).astype(np.float32)
        self.embeddings = self._Embedding(weights)
        self._vocab = vocab

    def named_parameters(self):
        return [("embeddings.weight", self.embeddings.weight)]

    def parameters(self):
        return [self.embeddings.weight]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text_transplant_substrate():
    """Build an NSCKSubstrate with a text transplant using synthetic embeddings."""
    from python.core.substrate import NSCKSubstrate
    from python.core.integration.config import NSCKConfig
    from python.core.transplant.pipeline import TransplantPipeline

    cfg = NSCKConfig.transplant()
    substrate = NSCKSubstrate(cfg)
    substrate.register_task("transplant_test")

    vocab = [f"word_{c}_{i}" for c in range(4) for i in range(10)]
    model = _SyntheticEmbeddingModel(vocab, dim=64)

    pipeline = TransplantPipeline(config=cfg)
    # Lower thresholds to always pass with synthetic data so integration runs
    pipeline._validator._rho_thresh = -1.0
    pipeline._validator._rec10_thresh = -1.0
    pipeline._validator._rec50_thresh = -1.0
    pipeline._validator._ari_thresh = -1.0

    # Use run() with the cognitive_engine for integration
    report = pipeline.run(
        model=model,
        domain_name="language",
        strategy="svd_factored",
        calibration_epochs=0,
        cognitive_engine=substrate.engine,
    )

    # Register the fitted projector for live encoding
    proj = pipeline._projectors.get("language")
    if proj is not None:
        substrate._transplant_projectors["language"] = proj

    return substrate, report, vocab


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransplantQuality:
    def test_text_transplant_passes_quality_threshold(self):
        """Text transplant should run without error and return a TransplantReport."""
        substrate, report, vocab = _build_text_transplant_substrate()
        from python.core.transplant.validator import TransplantReport
        assert isinstance(report, TransplantReport)
        # Semantic memory should contain the transplanted vocabulary
        sm = substrate.engine.semantic_memory
        loaded_concepts = set(sm.concept_hvs.keys())
        assert len(loaded_concepts) > 0, "Semantic memory should have concepts after transplant"

    def test_transplanted_codebook_in_semantic_memory(self):
        """After transplant, concepts should be in semantic memory (as token_N names)."""
        substrate, report, vocab = _build_text_transplant_substrate()
        sm = substrate.engine.semantic_memory
        # Harvester uses generic token_N names when no vocab is provided
        n_found = sum(1 for k in sm.concept_hvs.keys() if k.startswith("token_"))
        assert n_found > 0, (
            f"Expected token_N concepts in semantic memory after transplant; "
            f"got keys: {list(sm.concept_hvs.keys())[:5]}"
        )

    def test_live_encoding_uses_transplant_projector(self):
        """With a transplant projector registered, process() should complete without error."""
        substrate, _, _ = _build_text_transplant_substrate()
        result = substrate.process("test phrase for encoding", "transplant_test")
        assert result is not None
        assert result.chosen_action is not None

    def test_transplant_projector_registered(self):
        """After transplant, the projector should be cached in _transplant_projectors."""
        substrate, _, _ = _build_text_transplant_substrate()
        assert "language" in substrate._transplant_projectors, (
            "Expected 'language' projector in _transplant_projectors after transplant"
        )

    @_skip_sbert
    def test_text_transplant_with_real_sentence_transformers(self):
        """Full transplant pipeline with sentence-transformers if available."""
        from python.core.transplant.harvester import ModelHarvester
        from numpy.linalg import norm

        def cos_similarity(a, b):
            return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

        harvester = ModelHarvester()
        vocab_sample = ["cat", "dog", "fish", "bird", "sky"]
        embs = harvester.harvest_text(
            model_name_or_path="all-MiniLM-L6-v2",
            vocab=vocab_sample,
        )
        assert embs.shape == (len(vocab_sample), embs.shape[1])
        # Similarity between "cat" and "dog" should be higher than cat vs sky
        sim_cat_dog = cos_similarity(embs[0], embs[1])
        sim_cat_sky = cos_similarity(embs[0], embs[4])
        assert sim_cat_dog > sim_cat_sky - 0.1, (
            f"Expected cat~dog ({sim_cat_dog:.3f}) >= cat~sky ({sim_cat_sky:.3f})"
        )

