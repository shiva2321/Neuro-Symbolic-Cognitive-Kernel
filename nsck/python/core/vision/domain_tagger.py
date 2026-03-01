"""DomainTagger — tracks model-to-domain assignments and cross-domain similarity."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class DomainTagger:
    """Registry that maps pretrained models to task domains.

    Maintains per-domain model lists, arbitrary model metadata, and a
    symmetric matrix of cross-domain link weights derived from co-absorbed
    models or explicit registrations.

    Parameters
    ----------
    (none) — all state is accumulated via ``register_model``.
    """

    def __init__(self) -> None:
        # model_id -> {domain, metadata}
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        # domain -> [model_id, ...]
        self._domain_models: Dict[str, List[str]] = {}
        # (domain_a, domain_b) -> weight  (symmetric: stored with sorted key)
        self._cross_domain_weights: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_id: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Associate *model_id* with *domain*.

        If the model was previously registered under a different domain the
        record is updated (multi-domain models are allowed; register once per
        domain).

        Parameters
        ----------
        model_id: Unique model identifier (e.g. "resnet50_imagenet").
        domain:   Task domain string (e.g. "imagenet", "coco").
        metadata: Arbitrary provenance info (architecture, paper, etc.).
        """
        entry = self._model_registry.get(model_id)
        if entry is None:
            self._model_registry[model_id] = {
                "domain": domain,
                "metadata": metadata or {},
            }
        else:
            entry["domain"] = domain
            if metadata:
                entry["metadata"].update(metadata)

        if domain not in self._domain_models:
            self._domain_models[domain] = []
        if model_id not in self._domain_models[domain]:
            self._domain_models[domain].append(model_id)

    def set_cross_domain_weight(
        self, domain_a: str, domain_b: str, weight: float
    ) -> None:
        """Explicitly set a cross-domain similarity weight in [0, 1].

        The weight is stored symmetrically so order of arguments does not
        matter.
        """
        key = tuple(sorted([domain_a, domain_b]))
        self._cross_domain_weights[key] = float(max(0.0, min(1.0, weight)))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_domain_models(self, domain: str) -> List[str]:
        """Return all model IDs registered under *domain*."""
        return list(self._domain_models.get(domain, []))

    def get_cross_domain_similarity(self, domain_a: str, domain_b: str) -> float:
        """Return the cross-domain similarity weight between two domains.

        If no explicit weight was set, a heuristic based on shared models is
        used.  Returns 1.0 when domain_a == domain_b.

        Parameters
        ----------
        domain_a, domain_b: Domain names to compare.

        Returns
        -------
        float in [0, 1].
        """
        if domain_a == domain_b:
            return 1.0

        key = tuple(sorted([domain_a, domain_b]))
        if key in self._cross_domain_weights:
            return self._cross_domain_weights[key]

        # Heuristic: Jaccard similarity of model sets
        set_a = set(self._domain_models.get(domain_a, []))
        set_b = set(self._domain_models.get(domain_b, []))
        union = set_a | set_b
        if not union:
            return 0.0
        return float(len(set_a & set_b) / len(union))

    def all_domains(self) -> List[str]:
        """Return the sorted list of all registered domain names."""
        return sorted(self._domain_models.keys())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return registration info for *model_id*, or None if unknown."""
        return self._model_registry.get(model_id)

    def domain_coverage_stats(self) -> Dict[str, Any]:
        """Return a summary of domain coverage."""
        return {
            "n_domains": len(self._domain_models),
            "n_models": len(self._model_registry),
            "domain_sizes": {d: len(ms) for d, ms in self._domain_models.items()},
            "cross_domain_weights": dict(self._cross_domain_weights),
        }
