"""
NSCK V18 — Semantic HV Bootstrap
==================================
SemanticBootstrapper builds a semantically meaningful DistributionalCodebook
using a tiered, graceful-fallback approach:

  Tier 0 (prebuilt): Load a pre-built .pkl codebook — zero deps, instant.
  Tier 1 (bridge):   sentence-transformers/all-MiniLM-L6-v2 via EmbeddingVSABridge.
  Tier 2 (hf_corpus): HuggingFace corpus download (online, no local model).
  Tier 3 (corpus):   Existing BUILTIN_CORPUS co-occurrence — always works offline.

Usage::

    from python.core.language.semantic_bootstrap import SemanticBootstrapper
    cb = SemanticBootstrapper.build_codebook(strategy="auto")
    sim = cb.similarity("brain", "memory")   # → semantically meaningful
"""
from __future__ import annotations

import logging
import pickle
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SemanticBootstrapper:
    """Tiered builder for a semantically meaningful DistributionalCodebook."""

    # Pairs used for benchmarking semantic geometry quality.
    _BENCHMARK_PAIRS = [
        ("brain", "memory"),
        ("king", "queen"),
        ("cat", "dog"),
        ("hot", "cold"),
        ("fast", "slow"),
        ("doctor", "hospital"),
        ("learn", "knowledge"),
        ("code", "program"),
    ]

    @classmethod
    def build_codebook(
        cls,
        strategy: str = "auto",
        codebook_path: str = "",
        model_name: str = "all-MiniLM-L6-v2",
        hv_dim: int = 10240,
        seed: int = 42,
    ):
        """Build a DistributionalCodebook using the chosen strategy.

        Parameters
        ----------
        strategy      : ``"auto"`` | ``"bridge"`` | ``"corpus"`` | ``"prebuilt"``
        codebook_path : Path to a pre-built ``.pkl`` codebook (Tier 0).
        model_name    : sentence-transformers model name for Tier 1.
        hv_dim        : HyperVector dimensionality.
        seed          : RNG seed for EmbeddingVSABridge.

        Returns
        -------
        DistributionalCodebook
        """
        from python.core.language.distributional_semantics import DistributionalCodebook

        # ── Tier 0: prebuilt .pkl ───────────────────────────────────────────
        if strategy == "prebuilt" or (strategy == "auto" and codebook_path):
            try:
                cb = cls.load_codebook(codebook_path)
                logger.info("[SemanticBootstrap] Tier 0: loaded prebuilt codebook from %s "
                            "(%d words)", codebook_path, len(cb._codebook))
                return cb
            except Exception as exc:
                logger.warning("[SemanticBootstrap] Tier 0 failed (%s); trying next tier", exc)
                if strategy == "prebuilt":
                    # Explicit prebuilt requested — fall through to corpus only
                    strategy = "corpus"

        # ── Tier 1: sentence-transformers bridge ────────────────────────────
        if strategy in ("auto", "bridge"):
            try:
                cb = cls._build_via_bridge(model_name=model_name, hv_dim=hv_dim, seed=seed)
                logger.info("[SemanticBootstrap] Tier 1 (bridge): built codebook with %d words",
                            len(cb._codebook))
                return cb
            except Exception as exc:
                logger.info("[SemanticBootstrap] Tier 1 (bridge) unavailable (%s); "
                            "falling back", exc)
                if strategy == "bridge":
                    # Explicit bridge requested but failed — fall to corpus
                    strategy = "corpus"

        # ── Tier 2: HuggingFace corpus download ─────────────────────────────
        # (Currently folded into build_default's enable_hf_corpus path; we skip
        #  a separate Tier 2 here and go straight to Tier 3 to keep this module
        #  dependency-free.  The caller can activate HF via enable_hf_corpus.)

        # ── Tier 3: BUILTIN_CORPUS co-occurrence (always works) ─────────────
        logger.info("[SemanticBootstrap] Tier 3 (corpus): building from BUILTIN_CORPUS")
        from python.core.language.distributional_semantics import BUILTIN_CORPUS
        cb = DistributionalCodebook(pretrain=False)
        cb.build_from_corpus(BUILTIN_CORPUS)
        logger.info("[SemanticBootstrap] Tier 3: built codebook with %d words",
                    len(cb._codebook))
        return cb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_via_bridge(
        cls,
        model_name: str = "all-MiniLM-L6-v2",
        hv_dim: int = 10240,
        seed: int = 42,
    ):
        """Tier 1 — project sentence-transformer embeddings into VSA space."""
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np
        from python.core.vsa.vsa_embedding_bridge import EmbeddingVSABridge
        from python.core.language.distributional_semantics import DistributionalCodebook, BUILTIN_CORPUS
        from python.core.language.cognitive_vocabulary import COGNITIVE_VOCABULARY

        st_model = SentenceTransformer(model_name)
        emb_dim: int = st_model.get_sentence_embedding_dimension()
        bridge = EmbeddingVSABridge(dim_in=emb_dim, hv_dim=hv_dim, seed=seed)

        # Collect vocabulary: all unique lower-case tokens from BUILTIN_CORPUS
        # plus the curated COGNITIVE_VOCABULARY list.
        vocab = set(COGNITIVE_VOCABULARY)
        for sentence in BUILTIN_CORPUS:
            for token in sentence:
                vocab.add(token.lower())

        vocab_list = sorted(vocab)
        logger.info("[SemanticBootstrap] Encoding %d words via %s ...",
                    len(vocab_list), model_name)

        embeddings = st_model.encode(vocab_list, batch_size=256, show_progress_bar=False)

        cb = DistributionalCodebook(pretrain=False)
        for word, emb in zip(vocab_list, embeddings):
            emb_arr = np.asarray(emb, dtype=np.float32)
            if len(emb_arr) != emb_dim:
                # Resize to match bridge dim_in
                tmp = np.zeros(emb_dim, dtype=np.float32)
                n = min(len(emb_arr), emb_dim)
                tmp[:n] = emb_arr[:n]
                emb_arr = tmp
            cb._codebook[word] = bridge.embed_to_hv(emb_arr)

        return cb

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def save_codebook(cb, path: str) -> None:
        """Persist a DistributionalCodebook to a .pkl file."""
        with open(path, "wb") as fh:
            pickle.dump({"window_size": cb.window_size, "codebook": cb._codebook}, fh)
        logger.info("[SemanticBootstrap] Saved codebook (%d words) → %s",
                    len(cb._codebook), path)

    @staticmethod
    def load_codebook(path: str):
        """Load a DistributionalCodebook from a .pkl file."""
        from python.core.language.distributional_semantics import DistributionalCodebook
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        cb = DistributionalCodebook(pretrain=False)
        cb.window_size = data.get("window_size", 5)
        cb._codebook = data["codebook"]
        return cb

    @classmethod
    def benchmark(cls, cb) -> Dict:
        """Return cosine similarities for a standard set of semantic pairs.

        Parameters
        ----------
        cb : DistributionalCodebook

        Returns
        -------
        dict mapping ``(word1, word2)`` → float similarity in [0, 1].
        """
        results: Dict = {}
        for w1, w2 in cls._BENCHMARK_PAIRS:
            results[(w1, w2)] = cb.similarity(w1, w2)
        return results
