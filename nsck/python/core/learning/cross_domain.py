"""
NSCK Cross-Domain Knowledge Transfer
=====================================
Formal mechanism for lifting knowledge from one domain and applying it in another.

Architecture
------------
The module builds on the existing AnalogyEngine (structural alignment) and
RuleLearner (rule representation) to provide:

1. **SchemaExtractor** – scans a domain's concept graph and rule base to
   identify abstract relational patterns ("schemas").
2. **StructureMapper** – uses VSA cosine similarity to find corresponding
   structures between source and target domains.
3. **RuleLifter** – converts domain-specific rules to domain-independent
   forms using VSA role substitution.
4. **TransferEngine** – orchestrates extraction → mapping → lifting →
   application and tracks provenance and confidence.

Design Principles
-----------------
* Pure VSA: no neural networks. All similarity is computed via HyperVector
  inner products, exploiting the "concentration of measure" property of
  high-dimensional spaces.
* No catastrophic interference: transferred knowledge is tagged with source
  domain provenance; it does not overwrite existing target-domain knowledge.
* Confidence-gated: a transferred inference is only accepted when structural
  overlap exceeds a configurable threshold (default 0.60).
* Integrates with AnalogyEngine, RuleLearner, and ContinualLearner.

Example
-------
>>> from python.core.learning.cross_domain import TransferEngine
>>> engine = TransferEngine()
>>> # Teach source domain (physics)
>>> engine.register_concept("physics", "force", seed=1001)
>>> engine.register_concept("physics", "mass", seed=1002)
>>> engine.register_concept("physics", "acceleration", seed=1003)
>>> engine.register_rule("physics", "force", "proportional_to", "mass", 0.9)
>>> engine.register_rule("physics", "force", "causes",        "acceleration", 0.9)
>>> # Register target domain (finance)
>>> engine.register_concept("finance", "return", seed=2001)
>>> engine.register_concept("finance", "leverage", seed=2002)
>>> engine.register_concept("finance", "risk", seed=2003)
>>> # Declare correspondences (optional — auto-discovered if omitted)
>>> engine.register_correspondence("physics", "force",        "finance", "return")
>>> engine.register_correspondence("physics", "mass",         "finance", "leverage")
>>> engine.register_correspondence("physics", "acceleration", "finance", "risk")
>>> # Transfer rules
>>> results = engine.transfer("physics", "finance")
>>> for r in results:
...     print(r.summary())
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DomainConcept:
    """A concept registered in a named domain."""
    domain: str
    name: str
    hv: Any  # HyperVector


@dataclass
class DomainRelation:
    """A directed relation (rule) between two concepts in a domain."""
    domain: str
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    # VSA encoding of the full triple: bind(bind(src_hv, rel_hv), tgt_hv)
    triple_hv: Optional[Any] = field(default=None, repr=False)


@dataclass
class ConceptCorrespondence:
    """A declared or discovered correspondence between concepts in two domains."""
    source_domain: str
    source_concept: str
    target_domain: str
    target_concept: str
    # Cosine similarity of the two concept HVs (after alignment)
    structural_similarity: float = 1.0


@dataclass
class TransferredInference:
    """
    A rule that has been transferred from a source domain to a target domain.
    """
    source_domain: str
    source_rule: DomainRelation
    target_domain: str
    target_source: str          # target-domain name for source concept
    target_relation: str        # same relation label (lifted)
    target_target: str          # target-domain name for target concept
    confidence: float           # product of mapping confidence × rule confidence
    provenance: str             # human-readable chain

    def summary(self) -> str:
        return (
            f"[{self.source_domain}→{self.target_domain}] "
            f"{self.target_source} {self.target_relation} {self.target_target} "
            f"(conf={self.confidence:.2f}) "
            f"← {self.source_rule.source} {self.source_rule.relation} {self.source_rule.target}"
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _stable_seed(domain: str, name: str) -> int:
    """Deterministic integer seed from domain+name string."""
    digest = hashlib.md5(f"{domain}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _relation_hv(relation: str) -> Any:
    """Return a deterministic HyperVector for a relation label."""
    seed = _stable_seed("__relation__", relation)
    return hypervec_rs.HyperVector(seed)


# ---------------------------------------------------------------------------
# SchemaExtractor
# ---------------------------------------------------------------------------

class SchemaExtractor:
    """
    Identifies abstract relational schemas (patterns) in a domain's rule base.

    A schema is a set of DomainRelations that share the same relation type
    (e.g. "causes", "proportional_to", "is_a") across multiple concept pairs.
    The schema is represented as a *bundled* HyperVector of all its triples.
    """

    def extract(
        self,
        relations: List[DomainRelation],
        concepts: Dict[str, DomainConcept],
    ) -> Dict[str, Any]:
        """
        Group relations by type and return one bundled HV per group.

        Returns
        -------
        Dict[relation_type → bundled_HV]
        """
        groups: Dict[str, List[DomainRelation]] = {}
        for rel in relations:
            groups.setdefault(rel.relation, []).append(rel)

        schemas: Dict[str, Any] = {}
        for rel_type, members in groups.items():
            accumulated = None
            for m in members:
                if m.triple_hv is not None:
                    hv = m.triple_hv
                else:
                    # Build triple_hv on the fly
                    src_hv = concepts[m.source].hv if m.source in concepts else hypervec_rs.HyperVector(_stable_seed(m.domain, m.source))
                    tgt_hv = concepts[m.target].hv if m.target in concepts else hypervec_rs.HyperVector(_stable_seed(m.domain, m.target))
                    rel_hv = _relation_hv(m.relation)
                    hv = src_hv.xor(rel_hv).xor(tgt_hv)
                accumulated = hv if accumulated is None else accumulated.bundle(hv)
            if accumulated is not None:
                schemas[rel_type] = accumulated
        return schemas


# ---------------------------------------------------------------------------
# StructureMapper
# ---------------------------------------------------------------------------

class StructureMapper:
    """
    Finds structural correspondences between source and target concepts
    using cosine similarity of their HyperVectors.

    If explicit correspondences are supplied they are used directly;
    otherwise, all pairwise similarities are computed and a greedy
    maximum-weight matching is returned.
    """

    def map(
        self,
        source_concepts: Dict[str, DomainConcept],
        target_concepts: Dict[str, DomainConcept],
        explicit: Optional[List[ConceptCorrespondence]] = None,
        threshold: float = 0.50,
    ) -> List[ConceptCorrespondence]:
        """
        Return a list of concept correspondences.

        Parameters
        ----------
        source_concepts : Dict[name → DomainConcept]
        target_concepts : Dict[name → DomainConcept]
        explicit : optional pre-declared correspondences (bypass auto-discovery)
        threshold : minimum cosine similarity for auto-discovered matches
        """
        if explicit:
            return explicit

        correspondences: List[ConceptCorrespondence] = []
        used_targets: set = set()

        # For each source concept, find the best-matching target
        source_items = sorted(source_concepts.items())
        for src_name, src_dc in source_items:
            best_sim = -1.0
            best_tgt = None
            for tgt_name, tgt_dc in target_concepts.items():
                if tgt_name in used_targets:
                    continue
                sim = float(src_dc.hv.similarity(tgt_dc.hv))
                if sim > best_sim:
                    best_sim = sim
                    best_tgt = tgt_name
            if best_tgt is not None and best_sim >= threshold:
                correspondences.append(
                    ConceptCorrespondence(
                        source_domain=src_dc.domain,
                        source_concept=src_name,
                        target_domain=next(iter(target_concepts.values())).domain,
                        target_concept=best_tgt,
                        structural_similarity=best_sim,
                    )
                )
                used_targets.add(best_tgt)

        return correspondences


# ---------------------------------------------------------------------------
# RuleLifter
# ---------------------------------------------------------------------------

class RuleLifter:
    """
    Lifts domain-specific rules to transferred inferences using a
    concept mapping (correspondences).
    """

    def lift(
        self,
        source_relations: List[DomainRelation],
        correspondences: List[ConceptCorrespondence],
        target_domain: str,
        confidence_threshold: float = 0.50,
    ) -> List[TransferredInference]:
        """
        For each source relation, apply the concept mapping and produce a
        TransferredInference when both endpoints have a correspondence.

        The inference confidence = source_rule.confidence ×
            min(src_mapping.structural_similarity, tgt_mapping.structural_similarity)
        """
        # Build lookup: source_concept → ConceptCorrespondence
        src_to_corr: Dict[str, ConceptCorrespondence] = {
            c.source_concept: c for c in correspondences
        }

        inferences: List[TransferredInference] = []
        for rel in source_relations:
            src_corr = src_to_corr.get(rel.source)
            tgt_corr = src_to_corr.get(rel.target)
            if src_corr is None or tgt_corr is None:
                continue

            conf = rel.confidence * min(
                src_corr.structural_similarity,
                tgt_corr.structural_similarity,
            )
            if conf < confidence_threshold:
                continue

            inferences.append(
                TransferredInference(
                    source_domain=rel.domain,
                    source_rule=rel,
                    target_domain=target_domain,
                    target_source=src_corr.target_concept,
                    target_relation=rel.relation,  # relation label is lifted as-is
                    target_target=tgt_corr.target_concept,
                    confidence=conf,
                    provenance=(
                        f"{rel.domain}.{rel.source}→{rel.relation}→{rel.target} "
                        f"mapped via ({src_corr.source_concept}↔{src_corr.target_concept}, "
                        f"{tgt_corr.source_concept}↔{tgt_corr.target_concept})"
                    ),
                )
            )
        return inferences


# ---------------------------------------------------------------------------
# TransferEngine — public API
# ---------------------------------------------------------------------------

class TransferEngine:
    """
    Orchestrates cross-domain knowledge transfer.

    Usage
    -----
    1. Register concepts in source and target domains.
    2. Register rules (relations) in the source domain.
    3. Optionally, declare explicit concept correspondences.
    4. Call ``transfer(source_domain, target_domain)`` to obtain
       TransferredInference objects.
    5. Optionally, call ``apply(inferences, rule_learner)`` to inject the
       inferences into a RuleLearner instance.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold

        # domain → {concept_name → DomainConcept}
        self._concepts: Dict[str, Dict[str, DomainConcept]] = {}

        # domain → List[DomainRelation]
        self._relations: Dict[str, List[DomainRelation]] = {}

        # (src_domain, tgt_domain) → List[ConceptCorrespondence]
        self._explicit_corrs: Dict[Tuple[str, str], List[ConceptCorrespondence]] = {}

        self._extractor = SchemaExtractor()
        self._mapper = StructureMapper()
        self._lifter = RuleLifter()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_concept(
        self,
        domain: str,
        name: str,
        seed: Optional[int] = None,
    ) -> DomainConcept:
        """Register a concept in *domain* with a deterministic HyperVector."""
        if seed is None:
            seed = _stable_seed(domain, name)
        hv = hypervec_rs.HyperVector(seed)
        dc = DomainConcept(domain=domain, name=name, hv=hv)
        self._concepts.setdefault(domain, {})[name] = dc
        return dc

    def register_rule(
        self,
        domain: str,
        source_concept: str,
        relation: str,
        target_concept: str,
        confidence: float = 1.0,
    ) -> DomainRelation:
        """Register a directed relation (rule) in *domain*."""
        concepts = self._concepts.get(domain, {})

        def _get_hv(c: str) -> Any:
            if c in concepts:
                return concepts[c].hv
            seed = _stable_seed(domain, c)
            return hypervec_rs.HyperVector(seed)

        src_hv = _get_hv(source_concept)
        tgt_hv = _get_hv(target_concept)
        rel_hv = _relation_hv(relation)
        triple_hv = src_hv.xor(rel_hv).xor(tgt_hv)

        rel = DomainRelation(
            domain=domain,
            source=source_concept,
            relation=relation,
            target=target_concept,
            confidence=confidence,
            triple_hv=triple_hv,
        )
        self._relations.setdefault(domain, []).append(rel)
        return rel

    def register_correspondence(
        self,
        source_domain: str,
        source_concept: str,
        target_domain: str,
        target_concept: str,
        structural_similarity: float = 1.0,
    ) -> ConceptCorrespondence:
        """
        Declare an explicit concept correspondence, bypassing auto-discovery.
        """
        corr = ConceptCorrespondence(
            source_domain=source_domain,
            source_concept=source_concept,
            target_domain=target_domain,
            target_concept=target_concept,
            structural_similarity=structural_similarity,
        )
        key = (source_domain, target_domain)
        self._explicit_corrs.setdefault(key, []).append(corr)
        return corr

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    def transfer(
        self,
        source_domain: str,
        target_domain: str,
    ) -> List[TransferredInference]:
        """
        Transfer all applicable rules from *source_domain* to *target_domain*.

        Returns a list of TransferredInference objects ordered by descending
        confidence.
        """
        src_concepts = self._concepts.get(source_domain, {})
        tgt_concepts = self._concepts.get(target_domain, {})
        src_relations = self._relations.get(source_domain, [])

        if not src_relations:
            return []

        # Resolve correspondences
        key = (source_domain, target_domain)
        explicit = self._explicit_corrs.get(key)
        correspondences = self._mapper.map(
            src_concepts, tgt_concepts, explicit=explicit
        )

        # Lift rules
        inferences = self._lifter.lift(
            src_relations,
            correspondences,
            target_domain,
            confidence_threshold=self.confidence_threshold,
        )

        return sorted(inferences, key=lambda x: -x.confidence)

    def discover_correspondences(
        self,
        source_domain: str,
        target_domain: str,
        threshold: float = 0.50,
    ) -> List[ConceptCorrespondence]:
        """
        Auto-discover concept correspondences between two domains using VSA
        cosine similarity, without requiring explicit declarations.
        """
        src_concepts = self._concepts.get(source_domain, {})
        tgt_concepts = self._concepts.get(target_domain, {})
        return self._mapper.map(src_concepts, tgt_concepts, threshold=threshold)

    def apply(
        self,
        inferences: List[TransferredInference],
        rule_learner: Any,
    ) -> int:
        """
        Inject transferred inferences into a RuleLearner instance.

        Returns the number of rules successfully injected.
        """
        count = 0
        for inf in inferences:
            try:
                rule_learner.add_rule(
                    inf.target_source,
                    inf.target_relation,
                    inf.target_target,
                    confidence=inf.confidence,
                )
                count += 1
            except Exception:
                pass
        return count

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def domain_summary(self, domain: str) -> Dict[str, Any]:
        """Return a summary dict for a registered domain."""
        concepts = self._concepts.get(domain, {})
        relations = self._relations.get(domain, [])
        return {
            "domain": domain,
            "num_concepts": len(concepts),
            "num_relations": len(relations),
            "concept_names": list(concepts.keys()),
            "relation_types": list({r.relation for r in relations}),
        }

    def list_domains(self) -> List[str]:
        """List all registered domain names."""
        all_domains: set = set(self._concepts.keys()) | set(self._relations.keys())
        return sorted(all_domains)
