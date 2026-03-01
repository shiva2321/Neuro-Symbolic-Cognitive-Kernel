"""
PretrainedModelAdapter — fuses N pretrained-model embeddings into one PerceptPacket.

Accepts feature vectors or embedding vectors from any number of pretrained
models (medical-imaging, satellite, traffic, general vision, NLP, etc.),
projects each into HyperVector space via the NSCK transplant projectors,
and emits a single fused :class:`PerceptPacket` through weighted HV bundling.

This enables NSCK to absorb knowledge from heterogeneous pretrained models
and reason across their combined representations — supporting cross-domain
knowledge transfer, causal reasoning over multi-model evidence, and
generalisation that no single model could achieve alone.

Core algorithm
--------------
1. For each registered model *m* with weight *w_m*:
   a. Project its embedding vector **e_m** → HyperVector **h_m**.
   b. Optionally bind **h_m** with a deterministic *domain role HV* so that
      contributions from different domains remain distinguishable.
2. Weighted-bundle all **h_m** into a single situation HV.
3. Emit a :class:`PerceptPacket` with ``modality="pretrained_fusion"``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.types.modality_adapter import ModalityAdapter
from python.core.types.percept_packet import PerceptPacket
from python.core.transplant.projector import (
    BaseProjector,
    RandomProjector,
    SVDFactoredProjector,
)


# ---------------------------------------------------------------------------
# Dataclass for a registered model source
# ---------------------------------------------------------------------------

@dataclass
class _RegisteredSource:
    """Metadata for one registered pretrained-model source."""
    name: str
    projector: BaseProjector
    weight: float
    domain: str
    embedding_dim: int
    role_hv: hypervec_rs.HyperVector
    fitted: bool = False


# ---------------------------------------------------------------------------
# PretrainedModelAdapter
# ---------------------------------------------------------------------------

class PretrainedModelAdapter(ModalityAdapter):
    """Modality adapter that fuses N pretrained-model embedding outputs.

    Usage
    -----
    ::

        adapter = PretrainedModelAdapter()
        adapter.register_source("resnet_medical", embedding_dim=2048, weight=1.0)
        adapter.register_source("vit_satellite",  embedding_dim=768,  weight=0.8)

        # At inference time, pass a dict of source_name → embedding vector
        pkt = adapter.encode(
            {"resnet_medical": med_vec, "vit_satellite": sat_vec},
            task_tag="diagnosis",
        )

    Parameters
    ----------
    default_strategy : str
        Projection strategy for new sources (``"random"`` or
        ``"svd_factored"``).  Default ``"random"`` since pretrained
        embeddings are typically already high-quality and JL projection
        is the cheapest safe option.
    hv_dim : int
        Target HyperVector dimensionality (must match the global HV dim).
    bind_domain_role : bool
        If True, each source's HV is XOR-bound with a per-domain role HV
        before bundling so that identical embeddings from different domains
        remain distinguishable in the fused representation.
    """

    def __init__(
        self,
        default_strategy: str = "random",
        hv_dim: int = 10_240,
        bind_domain_role: bool = True,
    ) -> None:
        if default_strategy not in ("random", "svd_factored"):
            raise ValueError(
                f"Unknown strategy {default_strategy!r}; "
                "choose 'random' or 'svd_factored'."
            )
        self._default_strategy = default_strategy
        self._hv_dim = hv_dim
        self._bind_domain_role = bind_domain_role
        self._sources: Dict[str, _RegisteredSource] = {}

    # ------------------------------------------------------------------
    # Public API — source registration
    # ------------------------------------------------------------------

    def register_source(
        self,
        name: str,
        embedding_dim: int,
        *,
        weight: float = 1.0,
        domain: str = "",
        strategy: Optional[str] = None,
        fit_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Register a pretrained model source.

        Parameters
        ----------
        name : str
            Unique name for this source (e.g. ``"resnet_medical"``).
        embedding_dim : int
            Dimensionality of the embeddings this source produces.
        weight : float
            Relative importance of this source during bundling (default 1.0).
        domain : str
            Optional domain label (e.g. ``"medical"``, ``"satellite"``).
        strategy : str or None
            Projection strategy override (``"random"`` or ``"svd_factored"``).
        fit_embeddings : np.ndarray or None
            If using ``"svd_factored"`` strategy, provide a representative
            set of embeddings to fit the SVD projection.  Ignored for
            ``"random"``.
        """
        strat = strategy or self._default_strategy
        projector = self._make_projector(strat, embedding_dim)
        fitted = False

        if fit_embeddings is not None and hasattr(projector, "fit"):
            projector.fit(fit_embeddings)
            fitted = True

        domain_label = domain or name
        role_seed = hash(domain_label) % (2 ** 32)
        role_hv = hypervec_rs.HyperVector(role_seed)

        self._sources[name] = _RegisteredSource(
            name=name,
            projector=projector,
            weight=weight,
            domain=domain_label,
            embedding_dim=embedding_dim,
            role_hv=role_hv,
            fitted=fitted,
        )

    @property
    def source_names(self) -> List[str]:
        """Names of all registered sources."""
        return list(self._sources.keys())

    @property
    def n_sources(self) -> int:
        """Number of registered sources."""
        return len(self._sources)

    # ------------------------------------------------------------------
    # ModalityAdapter interface
    # ------------------------------------------------------------------

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        """Encode pretrained-model embeddings into a fused PerceptPacket.

        Parameters
        ----------
        raw_input : dict[str, array-like]
            Mapping of ``source_name → embedding_vector``.  Each vector
            must match the ``embedding_dim`` declared at registration.
            Sources present in the dict but not registered are silently
            skipped.  At least one registered source must be present.
        task_tag : str
            Task domain identifier.

        Returns
        -------
        PerceptPacket
            Fused percept with ``modality="pretrained_fusion"``.

        Raises
        ------
        ValueError
            If *raw_input* is not a dict or contains no registered sources.
        """
        if not isinstance(raw_input, dict):
            raise ValueError(
                "PretrainedModelAdapter.encode() expects a dict mapping "
                "source_name → embedding_vector."
            )
        if not self._sources:
            raise ValueError("No sources registered. Call register_source() first.")

        hvs: List[Tuple[str, hypervec_rs.HyperVector, float]] = []
        entity_hvs: Dict[str, hypervec_rs.HyperVector] = {}
        trace_sources: List[str] = []
        total_weight = 0.0

        for src_name, src in self._sources.items():
            if src_name not in raw_input:
                continue
            embedding = np.asarray(raw_input[src_name], dtype=np.float32).ravel()
            if embedding.shape[0] != src.embedding_dim:
                raise ValueError(
                    f"Source {src_name!r}: expected dim {src.embedding_dim}, "
                    f"got {embedding.shape[0]}."
                )

            hv = src.projector.encode_new(embedding)

            if self._bind_domain_role:
                hv = hv.xor(src.role_hv)

            hvs.append((src_name, hv, src.weight))
            entity_hvs[src_name] = hv
            trace_sources.append(src_name)
            total_weight += src.weight

        if not hvs:
            raise ValueError(
                "No registered source names found in raw_input keys. "
                f"Registered: {list(self._sources.keys())}; "
                f"Provided: {list(raw_input.keys())}."
            )

        # Weighted bundling: repeat-bundle approach
        situation_hv = self._weighted_bundle(hvs, total_weight)

        predicates = frozenset(
            f"SOURCE_{name.upper()}" for name, _, _ in hvs
        )

        adapter_trace = {
            "sources_used": trace_sources,
            "n_sources": len(hvs),
            "weights": {name: w for name, _, w in hvs},
            "bind_domain_role": self._bind_domain_role,
            "strategy": self._default_strategy,
        }

        return PerceptPacket.make(
            modality="pretrained_fusion",
            situation_hv=situation_hv,
            entity_hvs=entity_hvs,
            active_predicates=predicates,
            confidence=min(1.0, total_weight / max(1, len(self._sources))),
            adapter_name="PretrainedModelAdapter",
            adapter_trace=adapter_trace,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_projector(strategy: str, embedding_dim: int) -> BaseProjector:
        if strategy == "random":
            return RandomProjector(embedding_dim)
        elif strategy == "svd_factored":
            return SVDFactoredProjector(dim_in=embedding_dim)
        else:
            raise ValueError(f"Unknown projection strategy {strategy!r}.")

    @staticmethod
    def _weighted_bundle(
        hvs: List[Tuple[str, hypervec_rs.HyperVector, float]],
        total_weight: float,
    ) -> hypervec_rs.HyperVector:
        """Bundle HVs using weight-proportional repetition.

        Higher-weight sources are bundled more times so that they
        dominate the majority vote in the final binary HV.

        The repeat count for source *i* is ``max(1, round(w_i / w_min))``.
        """
        if len(hvs) == 1:
            return hvs[0][1]

        min_weight = min(w for _, _, w in hvs)
        if min_weight <= 0:
            min_weight = 1.0

        acc = None
        for _name, hv, weight in hvs:
            reps = max(1, round(weight / min_weight))
            for _ in range(reps):
                acc = hv if acc is None else acc.bundle(hv)

        return acc  # type: ignore[return-value]
