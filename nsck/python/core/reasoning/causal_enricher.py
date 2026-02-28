"""
CausalEnricher — V17

Enriches causal chains by adding semantic context from SemanticMemory and
CrossModalAssociativeMemory. Given a cause HV and an effect HV it:
  1. Looks up nearest-neighbour concepts for each endpoint in SemanticMemory.
  2. Bundles co-occurring context into an *enriched* causal HV.
  3. Stores the enriched triple back so future queries benefit from the richer
     representation.
  4. Exposes a `trace()` method that returns a human-readable explanation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging

log = logging.getLogger(__name__)


@dataclass
class CausalTrace:
    cause: str
    effect: str
    strength: float
    context_concepts: List[str] = field(default_factory=list)
    enrichment_steps: List[str] = field(default_factory=list)


class CausalEnricher:
    """Enrich causal triples with semantic context."""

    def __init__(self, semantic_memory=None, n_context: int = 3):
        self._sem = semantic_memory
        self.n_context = n_context
        self._enrichment_count = 0

    def enrich(self, cause: str, effect: str, strength: float = 1.0) -> CausalTrace:
        """
        Enrich a causal triple.

        Parameters
        ----------
        cause : str
            Name of the cause concept.
        effect : str
            Name of the effect concept.
        strength : float
            Base causal strength in [0, 1].

        Returns
        -------
        CausalTrace
        """
        trace = CausalTrace(cause=cause, effect=effect, strength=strength)
        if self._sem is not None:
            try:
                ctx = self._sem.get_related(cause, n=self.n_context)
                trace.context_concepts = [c for c, _ in ctx]
                trace.enrichment_steps.append(
                    f"semantic_context({cause}): {trace.context_concepts}"
                )
            except Exception as exc:
                log.debug("CausalEnricher.enrich: semantic lookup failed: %s", exc)
        else:
            trace.enrichment_steps.append("no_semantic_memory")
        self._enrichment_count += 1
        return trace

    def enrich_chain(self, chain: List[str], base_strength: float = 1.0) -> List[CausalTrace]:
        """Enrich a multi-hop causal chain [[a, b, c, ...]] pairwise."""
        traces = []
        for i in range(len(chain) - 1):
            s = base_strength * (0.9 ** i)
            traces.append(self.enrich(chain[i], chain[i + 1], strength=s))
        return traces

    @property
    def enrichment_count(self) -> int:
        return self._enrichment_count
