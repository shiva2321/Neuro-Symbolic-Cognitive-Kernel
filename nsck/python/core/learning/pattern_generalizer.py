"""
PatternGeneralizer — abstracts recurring patterns across domains.

Identifies patterns in HV similarity clusters and creates abstract
prototype HVs that generalize across instances.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod


@dataclass
class Pattern:
    """An abstracted pattern prototype."""
    pattern_id: str
    prototype_hv: hv_mod.HyperVector
    source_domain: str
    member_count: int
    abstraction_level: float    # 0.0=concrete, 1.0=fully abstract
    description: str = ""


class PatternGeneralizer:
    """
    Clusters HVs into patterns and creates abstract prototypes.

    Uses simple greedy clustering: if a new HV is within `cluster_threshold`
    similarity of an existing prototype, it joins that cluster.
    Otherwise, a new cluster is created.
    """

    def __init__(
        self,
        cluster_threshold: float = 0.7,
        min_members_for_abstraction: int = 3,
    ) -> None:
        self.cluster_threshold = cluster_threshold
        self.min_members = min_members_for_abstraction
        self._patterns: List[Pattern] = []
        self._cluster_hvs: Dict[str, List[hv_mod.HyperVector]] = {}
        self._next_id: int = 0

    def observe(
        self,
        hv: hv_mod.HyperVector,
        domain: str,
        description: str = "",
    ) -> Tuple[Pattern, bool]:
        """
        Observe a new HV and assign it to a cluster.
        Returns (pattern, is_new_pattern).
        """
        best_sim = 0.0
        best_pattern: Optional[Pattern] = None

        for pattern in self._patterns:
            if pattern.source_domain != domain:
                continue
            sim = float(pattern.prototype_hv.similarity(hv))
            if sim > best_sim:
                best_sim = sim
                best_pattern = pattern

        if best_pattern is not None and best_sim >= self.cluster_threshold:
            # Join existing cluster
            pid = best_pattern.pattern_id
            self._cluster_hvs[pid].append(hv)
            # Update prototype as bundle of all members
            acc = self._cluster_hvs[pid][0]
            for member_hv in self._cluster_hvs[pid][1:]:
                acc = acc.bundle(member_hv)
            best_pattern.prototype_hv = acc
            best_pattern.member_count = len(self._cluster_hvs[pid])
            n = best_pattern.member_count
            best_pattern.abstraction_level = min(1.0, n / (self.min_members * 4))
            return best_pattern, False
        else:
            # New cluster
            pid = f"pattern_{domain}_{self._next_id}"
            self._next_id += 1
            pattern = Pattern(
                pattern_id=pid,
                prototype_hv=hv,
                source_domain=domain,
                member_count=1,
                abstraction_level=0.0,
                description=description,
            )
            self._patterns.append(pattern)
            self._cluster_hvs[pid] = [hv]
            return pattern, True

    def get_mature_patterns(self) -> List[Pattern]:
        """Return patterns with enough members to be considered abstractions."""
        return [p for p in self._patterns if p.member_count >= self.min_members]

    def match(
        self,
        query_hv: hv_mod.HyperVector,
        domain: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[Pattern, float]]:
        """Find patterns most similar to the query HV."""
        candidates = self._patterns
        if domain is not None:
            candidates = [p for p in candidates if p.source_domain == domain]
        scored = [(p, float(p.prototype_hv.similarity(query_hv))) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        mature = self.get_mature_patterns()
        return {
            "total_patterns": len(self._patterns),
            "mature_patterns": len(mature),
            "domains": list({p.source_domain for p in self._patterns}),
        }


class CrossDomainTransferPipeline:
    """
    Uses PatternGeneralizer to transfer abstract patterns across domains.

    When a mature pattern from domain A is found to match a query in domain B,
    the pattern prototype is returned as a transfer candidate.
    """

    def __init__(self, generalizer: Optional[PatternGeneralizer] = None) -> None:
        self.generalizer = generalizer or PatternGeneralizer()
        self._transfer_log: List[Dict[str, Any]] = []
        # V5: Delegate rule-level transfer to TransferEngine
        try:
            from python.core.learning.cross_domain import TransferEngine
            self._transfer_engine: Any = TransferEngine()
        except Exception:
            self._transfer_engine = None

    def register(
        self,
        hv: hv_mod.HyperVector,
        domain: str,
        description: str = "",
    ) -> Pattern:
        """Register a new observation."""
        pattern, _ = self.generalizer.observe(hv, domain, description)
        return pattern

    def transfer(
        self,
        query_hv: hv_mod.HyperVector,
        source_domain: str,
        target_domain: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find patterns from source_domain that match query, then check
        if they also match any patterns in target_domain.

        Also delegates to TransferEngine for rule-level transfer results.
        """
        source_matches = self.generalizer.match(query_hv, domain=source_domain, top_k=top_k)
        results = []
        for src_pattern, src_sim in source_matches:
            # Check if any target domain pattern is similar to this abstract pattern
            target_matches = self.generalizer.match(
                src_pattern.prototype_hv, domain=target_domain, top_k=3
            )
            for tgt_pattern, tgt_sim in target_matches:
                result = {
                    "source_pattern": src_pattern.pattern_id,
                    "source_sim": src_sim,
                    "target_pattern": tgt_pattern.pattern_id,
                    "target_sim": tgt_sim,
                    "transfer_score": src_sim * tgt_sim,
                    "abstraction_level": src_pattern.abstraction_level,
                }
                results.append(result)
                self._transfer_log.append(result)

        # V5: Supplement with TransferEngine rule-level inferences
        if self._transfer_engine is not None:
            try:
                rule_inferences = self._transfer_engine.transfer(source_domain, target_domain)
                for inf in rule_inferences:
                    result = {
                        "source_pattern": f"{source_domain}:{inf.source_rule.source}",
                        "source_sim": inf.confidence,
                        "target_pattern": f"{target_domain}:{inf.target_source}",
                        "target_sim": inf.confidence,
                        "transfer_score": inf.confidence,
                        "abstraction_level": 1,
                        "rule_lifted": inf.provenance,
                    }
                    results.append(result)
                    self._transfer_log.append(result)
            except Exception:
                pass

        results.sort(key=lambda x: x["transfer_score"], reverse=True)
        return results[:top_k]

    def get_transfer_log(self) -> List[Dict[str, Any]]:
        return list(self._transfer_log)
