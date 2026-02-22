"""
PMI Learner — Pointwise Mutual Information for relation-strength learning.

Theoretical grounding
---------------------
Pointwise Mutual Information (Church & Hanks, 1990; Turney & Pantel, 2010)
measures the statistical association between two events relative to what
would be expected by chance:

    PMI(a, b) = log P(a, b) / (P(a) * P(b))

This is derived directly from information theory (Shannon, 1948): it
measures the reduction in uncertainty about *b* when *a* is known.

In knowledge-graph terms: how much more likely is it that concept B
appears in a particular relation with concept A, compared with a random
concept pair?  High PMI → strong, meaningful relation.  Low/negative
PMI → spurious co-occurrence.

Unlike raw co-occurrence counts (which favour frequent words), PMI
normalises out frequency, making it robust and compact — ideal for
the low-storage-overhead goal of NSCK.

Usage
-----
    pmi = PMILearner()
    pmi.observe("cat", "chases", "mouse")
    pmi.observe("dog", "chases", "cat")
    weight = pmi.relation_weight("cat", "chases", "mouse")
    # Returns float in [0, 1] — use to modulate SemanticMemory edge weights
"""
from __future__ import annotations

import math
import logging
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger("nsck.pmi_learner")


class PMILearner:
    """
    Online PMI learner for concept-relation-concept triplets.

    Counts are accumulated incrementally (one triplet at a time, or in
    batches), making this O(1) memory per new observation.

    The computed PMI is normalised to [0, 1] via the NPMI formula
    (Bouma, 2009) so it can be used directly as an edge weight or
    confidence score in SemanticMemory.
    """

    def __init__(self, smoothing: float = 1.0):
        """
        Args:
            smoothing: Laplace smoothing constant added to all counts before
                       computing PMI.  Prevents log(0) and reduces noise from
                       rare observations.  Default=1.0 (add-one smoothing).
        """
        self.smoothing = smoothing

        # n(A, rel, B) — triplet co-occurrence count
        self._triplet: Counter[Tuple[str, str, str]] = Counter()

        # n(A, rel, *) — how often A appears as subject with this relation
        self._subject_rel: Counter[Tuple[str, str]] = Counter()

        # n(*, rel, B) — how often B appears as object with this relation
        self._object_rel: Counter[Tuple[str, str]] = Counter()

        # n(*, rel, *) — total observations for this relation type
        self._relation_total: Counter[str] = Counter()

        # Grand total of all observations
        self._total: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(
        self,
        subject: str,
        relation: str,
        obj: str,
        count: int = 1,
    ) -> None:
        """Record one (or *count*) observations of a subject-relation-object triplet."""
        s, r, o = subject.lower(), relation.lower(), obj.lower()
        self._triplet[(s, r, o)] += count
        self._subject_rel[(s, r)] += count
        self._object_rel[(o, r)] += count
        self._relation_total[r] += count
        self._total += count

    def observe_batch(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Batch version of :meth:`observe`."""
        for s, r, o in triplets:
            self.observe(s, r, o)

    def pmi(self, subject: str, relation: str, obj: str) -> float:
        """
        Compute raw PMI for the triplet (subject, relation, object).

        Returns:
            float — PMI value (can be negative for anti-correlated pairs).
        """
        s, r, o = subject.lower(), relation.lower(), obj.lower()
        N = self._total + self.smoothing * max(1, len(self._triplet))

        n_sro = self._triplet.get((s, r, o), 0) + self.smoothing
        n_sr  = self._subject_rel.get((s, r), 0)  + self.smoothing
        n_ro  = self._object_rel.get((o, r), 0)   + self.smoothing

        p_sro = n_sro / N
        p_sr  = n_sr  / N
        p_ro  = n_ro  / N

        if p_sr <= 0 or p_ro <= 0:
            return 0.0

        return math.log(p_sro / (p_sr * p_ro), 2)

    def npmi(self, subject: str, relation: str, obj: str) -> float:
        """
        Normalised PMI in [-1, 1] (Bouma, 2009).

        NPMI = PMI / -log P(a,b)

        A value of +1 means perfect co-occurrence; -1 means they never
        co-occur; 0 means independence.
        """
        s, r, o = subject.lower(), relation.lower(), obj.lower()
        raw_pmi = self.pmi(s, r, o)

        N = self._total + self.smoothing * max(1, len(self._triplet))
        n_sro = self._triplet.get((s, r, o), 0) + self.smoothing
        p_sro = n_sro / N

        denom = -math.log(p_sro, 2) if p_sro > 0 else 1.0
        if denom == 0:
            return 0.0
        return raw_pmi / denom

    def relation_weight(
        self,
        subject: str,
        relation: str,
        obj: str,
    ) -> float:
        """
        Return a weight in [0, 1] suitable for use as a SemanticMemory edge
        weight or spreading-activation modulator.

        Uses NPMI shifted and scaled: weight = (npmi + 1) / 2
        so that:
            - perfectly correlated pairs → 1.0
            - independent pairs          → 0.5
            - anti-correlated pairs      → 0.0
        """
        return (self.npmi(subject, relation, obj) + 1.0) / 2.0

    def top_relations(
        self,
        subject: str,
        relation: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-*k* objects most strongly associated with
        (subject, relation) by NPMI.
        """
        s, r = subject.lower(), relation.lower()
        candidates = {
            o: self.npmi(s, r, o)
            for (ss, rr, o) in self._triplet
            if ss == s and rr == r
        }
        return sorted(candidates.items(), key=lambda x: -x[1])[:k]

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def update_semantic_memory_weights(self, semantic_memory: object) -> None:
        """
        Push learned PMI weights back into a SemanticMemory instance.

        For each edge ``(u, relation, v)`` in the graph, replace the generic
        ``relation_weights[relation]`` with the PMI-derived per-edge weight.
        This makes spreading activation path-specific rather than
        relation-type-specific.
        """
        try:
            graph = semantic_memory.concept_graph
        except AttributeError:
            logger.warning("[PMI] semantic_memory has no concept_graph attribute.")
            return

        updated = 0
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation", "")
            if not rel:
                continue
            w = self.relation_weight(u, rel, v)
            graph[u][v]["pmi_weight"] = w
            updated += 1

        logger.debug("[PMI] Updated %d edges with PMI weights.", updated)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """Return summary statistics."""
        return {
            "total_observations": self._total,
            "unique_triplets": len(self._triplet),
            "unique_relations": len(self._relation_total),
            "relation_counts": dict(self._relation_total.most_common(10)),
        }
