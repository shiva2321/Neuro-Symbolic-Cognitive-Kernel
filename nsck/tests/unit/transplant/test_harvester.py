"""Unit tests for ModelHarvester."""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Ensure nsck package root on path
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from python.core.transplant.harvester import ModelHarvester, HarvestResult


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------

class _FakeParam:
    """Mimics a PyTorch Parameter with .data and .detach().numpy()."""

    def __init__(self, array: np.ndarray) -> None:
        self._arr = array.astype(np.float32)

    @property
    def data(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim


class _FakeEmbedding:
    def __init__(self, vocab: int, dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weight = _FakeParam(rng.standard_normal((vocab, dim)).astype(np.float32))


class _TransformerLM:
    """Fake transformer LM model with word_embeddings attribute."""

    def __init__(self, vocab: int = 100, dim: int = 32) -> None:
        self.word_embeddings = _FakeEmbedding(vocab, dim, seed=1)

    def named_parameters(self):
        yield "word_embeddings.weight", self.word_embeddings.weight


class _TransformerBert:
    """Fake BERT-like model: model.embeddings.word_embeddings."""

    class _Embeddings:
        def __init__(self, vocab: int, dim: int) -> None:
            self.word_embeddings = _FakeEmbedding(vocab, dim, seed=2)

    def __init__(self, vocab: int = 80, dim: int = 16) -> None:
        self.embeddings = self._Embeddings(vocab, dim)

    def named_parameters(self):
        yield "embeddings.word_embeddings.weight", self.embeddings.word_embeddings.weight


class _VisionModel:
    """Fake ViT-like model with patch_embed."""

    class _PatchEmbed:
        class _Proj:
            def __init__(self) -> None:
                rng = np.random.default_rng(3)
                self.weight = _FakeParam(rng.standard_normal((64, 3, 4, 4)).astype(np.float32))

        def __init__(self) -> None:
            self.proj = self._Proj()

    def __init__(self) -> None:
        self.patch_embed = self._PatchEmbed()

    def named_parameters(self):
        yield "patch_embed.proj.weight", self.patch_embed.proj.weight


class _GenericModel:
    """Model with only named_parameters — no recognizable attribute."""

    def __init__(self) -> None:
        self._params = {
            "fc1.weight": _FakeParam(np.random.default_rng(5).standard_normal((200, 50)).astype(np.float32)),
            "fc2.weight": _FakeParam(np.random.default_rng(6).standard_normal((50, 50)).astype(np.float32)),
        }

    def named_parameters(self):
        yield from self._params.items()


class _NoParamsModel:
    """Object that has no named_parameters at all."""
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHarvestResult:
    def test_fields_present(self):
        hr = HarvestResult(
            embeddings=np.zeros((10, 8), dtype=np.float32),
            vocab_mapping={"a": 0},
            model_type="generic",
            embedding_dim=8,
            vocab_size=10,
            source_model="<Model>",
        )
        assert hr.embedding_dim == 8
        assert hr.vocab_size == 10
        assert hr.model_type == "generic"


class TestModelHarvester:

    def test_transformer_lm_word_embeddings(self):
        model = _TransformerLM(vocab=100, dim=32)
        result = ModelHarvester().harvest(model)
        assert result.model_type == "transformer_lm"
        assert result.embeddings.shape == (100, 32)
        assert result.embedding_dim == 32
        assert result.vocab_size == 100
        assert result.embeddings.dtype == np.float32

    def test_bert_style_nested_embeddings(self):
        model = _TransformerBert(vocab=80, dim=16)
        result = ModelHarvester().harvest(model)
        assert result.model_type == "transformer_lm"
        assert result.embeddings.shape == (80, 16)

    def test_vision_model_patch_embed(self):
        model = _VisionModel()
        result = ModelHarvester().harvest(model)
        assert result.model_type == "transformer_vision"
        # patch_embed.proj.weight: (64, 3, 4, 4) → reshaped to (64, 48)
        assert result.embeddings.shape[0] == 64

    def test_generic_fallback_named_params(self):
        model = _GenericModel()
        result = ModelHarvester().harvest(model)
        # Largest (N>D) matrix is fc1.weight (200, 50)
        assert result.model_type == "generic"
        assert result.embeddings.shape == (200, 50)

    def test_vocab_mapping_auto_generated(self):
        model = _TransformerLM(vocab=10, dim=4)
        result = ModelHarvester().harvest(model)
        assert len(result.vocab_mapping) == 10
        assert all(f"token_{i}" in result.vocab_mapping for i in range(10))

    def test_error_result_for_no_params_model(self):
        model = _NoParamsModel()
        result = ModelHarvester().harvest(model)
        assert result.model_type == "error"
        assert "error" in result.metadata

    def test_named_params_method_explicit(self):
        model = _GenericModel()
        result = ModelHarvester().harvest(model, method="named_params")
        assert result.model_type == "generic"
        assert result.embeddings.ndim == 2

    def test_embedding_layer_method_explicit(self):
        model = _TransformerLM(vocab=50, dim=8)
        result = ModelHarvester().harvest(model, method="embedding_layer")
        assert result.embeddings.shape == (50, 8)

    def test_encoder_decoder_model(self):
        class _EncDecModel:
            class _Encoder:
                def __init__(self):
                    self.embed_tokens = _FakeEmbedding(60, 12, seed=7)
                def named_parameters(self):
                    yield "embed_tokens.weight", self.embed_tokens.weight
            class _Decoder:
                pass
            def __init__(self):
                self.encoder = self._Encoder()
                self.decoder = self._Decoder()
            def named_parameters(self):
                yield from self.encoder.named_parameters()

        model = _EncDecModel()
        result = ModelHarvester().harvest(model)
        assert result.model_type == "encoder_decoder"
        assert result.embeddings.shape == (60, 12)

    def test_unknown_method_returns_error(self):
        model = _TransformerLM()
        result = ModelHarvester().harvest(model, method="nonexistent")
        assert result.model_type == "error"

    def test_deterministic(self):
        model = _TransformerLM(vocab=20, dim=8)
        r1 = ModelHarvester().harvest(model)
        r2 = ModelHarvester().harvest(model)
        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)
