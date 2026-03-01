"""VisionFusion — NSCK neuro-symbolic fusion layer for vision queries."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.vision.absorption_memory import AbsorptionMemory
from python.core.vision.domain_tagger import DomainTagger


# ---------------------------------------------------------------------------
# VisionResponse
# ---------------------------------------------------------------------------

@dataclass
class VisionResponse:
    """Complete response from the NSCK vision fusion pipeline."""

    query_id: str
    label: str
    confidence: float
    accuracy_rating: float
    top5: List[Tuple[str, float]]
    causal_chain: List[str]
    cross_domain_analogies: List[str]
    source_model_provenance: List[str]
    reference_label: Optional[str]
    reference_confidence: Optional[float]
    nsck_overrode_reference: bool
    trace: Optional[Any]
    latency_ms: float
    timestamp: str
    accuracy_breakdown: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AccuracyEstimator
# ---------------------------------------------------------------------------

class AccuracyEstimator:
    """Calibrated per-claim accuracy estimator.

    Produces a scalar rating in [0, 1] using a simple weighted average of
    multiple evidence sources.  No external calibration library is required.

    Weights
    -------
    nsck_confidence       0.30
    reference_confidence  0.20  (when available)
    agreement_flag        0.15
    domain_coverage       0.15
    episodic_hits         0.10  (capped at 1.0)
    causal_chain_depth    0.05  (capped at 1.0)
    analogy_count         0.05  (capped at 1.0)
    """

    _WEIGHTS = {
        "nsck_confidence": 0.30,
        "reference_confidence": 0.20,
        "agreement": 0.15,
        "domain_coverage": 0.15,
        "episodic_hits": 0.10,
        "causal_chain_depth": 0.05,
        "analogy_count": 0.05,
    }

    def estimate(
        self,
        nsck_confidence: float,
        reference_confidence: Optional[float] = None,
        agreement: bool = True,
        domain_coverage: float = 0.5,
        episodic_hits: int = 0,
        causal_chain_depth: int = 0,
        analogy_count: int = 0,
    ) -> float:
        """Compute calibrated accuracy rating.

        Parameters
        ----------
        nsck_confidence:      NSCK HV similarity score [0, 1].
        reference_confidence: Reference model confidence [0, 1] or None.
        agreement:            True if NSCK and reference agree on label.
        domain_coverage:      Fraction of relevant domains absorbed [0, 1].
        episodic_hits:        Number of relevant episodic memories recalled.
        causal_chain_depth:   Depth of causal reasoning chain.
        analogy_count:        Number of cross-domain analogies found.

        Returns
        -------
        float in [0, 1] representing calibrated accuracy.
        """
        w = self._WEIGHTS
        total_weight = 0.0
        score = 0.0

        score += w["nsck_confidence"] * float(max(0.0, min(1.0, nsck_confidence)))
        total_weight += w["nsck_confidence"]

        if reference_confidence is not None:
            score += w["reference_confidence"] * float(max(0.0, min(1.0, reference_confidence)))
            total_weight += w["reference_confidence"]

        score += w["agreement"] * (1.0 if agreement else 0.0)
        total_weight += w["agreement"]

        score += w["domain_coverage"] * float(max(0.0, min(1.0, domain_coverage)))
        total_weight += w["domain_coverage"]

        epi_norm = float(min(1.0, episodic_hits / max(1, episodic_hits + 1)))
        score += w["episodic_hits"] * epi_norm
        total_weight += w["episodic_hits"]

        causal_norm = float(min(1.0, causal_chain_depth / max(1, causal_chain_depth + 1)))
        score += w["causal_chain_depth"] * causal_norm
        total_weight += w["causal_chain_depth"]

        analogy_norm = float(min(1.0, analogy_count / max(1, analogy_count + 1)))
        score += w["analogy_count"] * analogy_norm
        total_weight += w["analogy_count"]

        if total_weight <= 0.0:
            return float(nsck_confidence)

        return float(score / total_weight)


# ---------------------------------------------------------------------------
# NSCKVisionFusion
# ---------------------------------------------------------------------------

class NSCKVisionFusion:
    """Neuro-symbolic fusion layer for vision query responses.

    Combines NSCK HyperVector memory with optional causal reasoning,
    analogical transfer, and Global Workspace competition to produce a rich
    VisionResponse.

    Parameters
    ----------
    absorption_memory: AbsorptionMemory with absorbed model knowledge.
    domain_tagger:     DomainTagger for cross-domain reasoning.
    semantic_memory:   Optional SemanticMemory for concept lookup.
    causal_graph:      Optional causal reasoning graph.
    analogy_engine:    Optional AnalogyEngine from python.core.reasoning.analogy.
    global_workspace:  Optional GlobalWorkspace from python.core.reasoning.global_workspace.
    """

    def __init__(
        self,
        absorption_memory: AbsorptionMemory,
        domain_tagger: DomainTagger,
        semantic_memory=None,
        causal_graph=None,
        analogy_engine=None,
        global_workspace=None,
    ) -> None:
        self._absorption_memory = absorption_memory
        self._domain_tagger = domain_tagger
        self._semantic_memory = semantic_memory
        self._causal_graph = causal_graph
        self._analogy_engine = analogy_engine
        self._global_workspace = global_workspace
        self._accuracy_estimator = AccuracyEstimator()

    def fuse(
        self,
        query_hv: Any,
        label: str,
        nsck_confidence: float,
        reference_result: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        override_threshold: float = 0.95,
        min_causal_depth: int = 1,
    ) -> VisionResponse:
        """Produce a VisionResponse for a query.

        Parameters
        ----------
        query_hv:           HyperVector representing the query.
        label:              NSCK primary label prediction.
        nsck_confidence:    NSCK similarity score [0, 1].
        reference_result:   Dict with optional keys ``label``, ``confidence``,
                            ``top5`` from a reference model.
        domain:             Domain context (used for analogy lookup).
        override_threshold: NSCK overrides reference when
                            nsck_conf >= ref_conf * override_threshold
                            AND causal_chain_depth >= min_causal_depth.
        min_causal_depth:   Minimum causal chain depth required to override.

        Returns
        -------
        VisionResponse
        """
        t0 = time.perf_counter()
        query_id = str(uuid.uuid4())

        ref_label: Optional[str] = None
        ref_confidence: Optional[float] = None
        if reference_result is not None:
            ref_label = reference_result.get("label")
            ref_confidence = reference_result.get("confidence")

        # --- Top-5 from absorption memory ---
        top_records = self._absorption_memory.query_by_hv(query_hv, top_k=5)
        top5: List[Tuple[str, float]] = [
            (r.label, float(r.hv.similarity(query_hv))) for r in top_records
        ]
        if not top5:
            top5 = [(label, float(nsck_confidence))]

        # --- Source model provenance ---
        provenance = list({r.model_id for r in top_records})

        # --- Causal chain ---
        causal_chain: List[str] = []
        if self._causal_graph is not None:
            try:
                causal_chain = self._query_causal_chain(label)
            except Exception:
                causal_chain = []

        # --- Cross-domain analogies ---
        analogies: List[str] = []
        if self._analogy_engine is not None and domain is not None:
            try:
                analogies = self._query_analogies(label, domain)
            except Exception:
                analogies = []

        # --- Global Workspace competition ---
        trace = None
        if self._global_workspace is not None:
            try:
                from python.core.reasoning.global_workspace import Coalition
                nsck_coalition = Coalition(
                    source="nsck_vision",
                    content={"label": label, "confidence": nsck_confidence},
                    base_salience=float(nsck_confidence),
                    sender_confidence=float(nsck_confidence),
                )
                coalitions = [nsck_coalition]
                if reference_result is not None and ref_confidence is not None:
                    ref_coalition = Coalition(
                        source="reference_model",
                        content=reference_result,
                        base_salience=float(ref_confidence),
                        sender_confidence=float(ref_confidence),
                    )
                    coalitions.append(ref_coalition)
                trace = self._global_workspace.compete(coalitions)
            except Exception:
                trace = None

        # --- Override logic ---
        causal_depth = len(causal_chain)
        overrides = False
        if ref_confidence is not None:
            overrides = (
                nsck_confidence >= ref_confidence * override_threshold
                and causal_depth >= min_causal_depth
            )

        # --- Domain coverage ---
        all_domains = self._domain_tagger.all_domains()
        domain_coverage = min(1.0, len(all_domains) / max(1, len(all_domains) + 1))

        # --- Accuracy estimation ---
        agreement = (ref_label is None) or (label == ref_label)
        accuracy = self._accuracy_estimator.estimate(
            nsck_confidence=nsck_confidence,
            reference_confidence=ref_confidence,
            agreement=agreement,
            domain_coverage=domain_coverage,
            episodic_hits=len(top_records),
            causal_chain_depth=causal_depth,
            analogy_count=len(analogies),
        )

        # --- Per-claim accuracy breakdown ---
        breakdown: Dict[str, float] = {
            "nsck_confidence": float(nsck_confidence),
            "domain_coverage": float(domain_coverage),
            "causal_chain_depth": float(min(1.0, causal_depth / max(1, causal_depth + 1))),
            "analogy_support": float(min(1.0, len(analogies) / max(1, len(analogies) + 1))),
            "episodic_support": float(min(1.0, len(top_records) / 5.0)),
        }
        if ref_confidence is not None:
            breakdown["reference_confidence"] = float(ref_confidence)
            breakdown["agreement"] = 1.0 if agreement else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        ts = datetime.now(timezone.utc).isoformat()

        return VisionResponse(
            query_id=query_id,
            label=label,
            confidence=float(nsck_confidence),
            accuracy_rating=accuracy,
            top5=top5,
            causal_chain=causal_chain,
            cross_domain_analogies=analogies,
            source_model_provenance=provenance,
            reference_label=ref_label,
            reference_confidence=ref_confidence,
            nsck_overrode_reference=overrides,
            trace=trace,
            latency_ms=latency_ms,
            timestamp=ts,
            accuracy_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_causal_chain(self, label: str) -> List[str]:
        """Retrieve a causal reasoning chain for *label* from the causal graph."""
        chain: List[str] = []
        try:
            # Try NetworkX-style neighbours if causal_graph is a DiGraph
            import networkx as nx
            if isinstance(self._causal_graph, nx.DiGraph):
                if label in self._causal_graph:
                    for _, nbr, data in self._causal_graph.out_edges(label, data=True):
                        rel = data.get("relation", "causes")
                        chain.append(f"{label} --[{rel}]--> {nbr}")
                return chain
        except ImportError:
            pass

        # Generic: try .successors() or .get_effects()
        for attr in ("successors", "get_effects", "children"):
            if hasattr(self._causal_graph, attr):
                try:
                    for nbr in getattr(self._causal_graph, attr)(label):
                        chain.append(f"{label} --> {nbr}")
                    break
                except Exception:
                    pass
        return chain

    def _query_analogies(self, label: str, domain: str) -> List[str]:
        """Retrieve cross-domain analogies for *label* from the analogy engine."""
        analogies: List[str] = []
        try:
            # Try AnalogyEngine.find_analogies or similar API
            if hasattr(self._analogy_engine, "find_analogies"):
                results = self._analogy_engine.find_analogies(label, source_domain=domain)
                for a in results:
                    if hasattr(a, "target_concept") and hasattr(a, "target_domain"):
                        analogies.append(
                            f"{a.source_concept} ({domain}) ≈ {a.target_concept} ({a.target_domain})"
                        )
                    else:
                        analogies.append(str(a))
            elif hasattr(self._analogy_engine, "map_concept"):
                mapped = self._analogy_engine.map_concept(label, domain)
                if mapped:
                    analogies.append(str(mapped))
        except Exception:
            pass
        return analogies
