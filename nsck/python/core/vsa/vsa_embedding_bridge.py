"""
VSA Embedding Bridge
====================
Bridges dense embedding vectors to/from VSA HyperVectors via random projection.
"""
from __future__ import annotations

import sys
import os
from typing import List, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.vsa.hypervec_shim import HyperVector


class EmbeddingVSABridge:
    """Bridges dense embeddings (e.g. sentence-transformers) to VSA HyperVectors."""

    def __init__(self, dim_in: int = 768, hv_dim: int = 10240, seed: int = 42):
        self.dim_in = dim_in
        self.hv_dim = hv_dim
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((dim_in, hv_dim)).astype(np.float32)
        self._st_model = None

    def embed_to_hv(self, embedding: np.ndarray) -> HyperVector:
        """Project dense vector to HV space and binarize."""
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        if len(emb) != self.dim_in:
            raise ValueError(f"Expected embedding of dim {self.dim_in}, got {len(emb)}")
        projected = emb @ self._proj  # (hv_dim,)
        bits = (projected >= 0).astype(np.int8)
        return HyperVector.from_bits(bits)

    def hv_to_embed(self, hv: HyperVector) -> np.ndarray:
        """Reverse-project HV bits back to embedding space."""
        bits = np.asarray(hv.bits, dtype=np.float32)
        reconstructed = bits @ self._proj.T  # (dim_in,)
        norm = np.linalg.norm(reconstructed)
        if norm > 0:
            reconstructed = reconstructed / norm
        return reconstructed

    def batch_embed_to_hv(self, embeddings: List[np.ndarray]) -> List[HyperVector]:
        """Batch convert embeddings to HyperVectors."""
        return [self.embed_to_hv(e) for e in embeddings]

    def similarity_in_embed_space(self, hv1: HyperVector, hv2: HyperVector) -> float:
        """Cosine similarity between two HVs measured in embedding space."""
        e1 = self.hv_to_embed(hv1)
        e2 = self.hv_to_embed(hv2)
        denom = np.linalg.norm(e1) * np.linalg.norm(e2)
        if denom == 0:
            return 0.0
        return float(np.dot(e1, e2) / denom)

    def try_load_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-L2') -> bool:
        """Try to load a sentence-transformers model; return True on success."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(model_name)
            return True
        except Exception:
            return False

    def encode_text(self, text: str) -> HyperVector:
        """Encode text to HyperVector, using sentence-transformers if available."""
        if self._st_model is not None:
            emb = self._st_model.encode(text)
            emb = np.asarray(emb, dtype=np.float32)
            # Resize if needed
            if len(emb) != self.dim_in:
                tmp = np.zeros(self.dim_in, dtype=np.float32)
                n = min(len(emb), self.dim_in)
                tmp[:n] = emb[:n]
                emb = tmp
            return self.embed_to_hv(emb)
        else:
            # Fallback: character n-gram embedding
            emb = np.zeros(self.dim_in, dtype=np.float32)
            for i, ch in enumerate(text):
                emb[i % self.dim_in] += float(ord(ch))
            # Add bigrams
            for i in range(len(text) - 1):
                idx = (ord(text[i]) * 31 + ord(text[i + 1])) % self.dim_in
                emb[idx] += 1.0
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            return self.embed_to_hv(emb)
