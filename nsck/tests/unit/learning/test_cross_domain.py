"""
Tests for the Cross-Domain Knowledge Transfer module.

Validates that TransferEngine can:
- Register concepts and rules in multiple domains
- Discover structural correspondences
- Transfer rules from source to target domain
- Report confidence scores and provenance
"""

import sys
import os
from pathlib import Path

import pytest

# Ensure nsck root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from python.core.learning.cross_domain import (
    TransferEngine,
    DomainConcept,
    DomainRelation,
    ConceptCorrespondence,
    TransferredInference,
    SchemaExtractor,
    StructureMapper,
    RuleLifter,
)


class TestTransferEngine:
    """Test the high-level TransferEngine API."""

    def setup_method(self):
        self.engine = TransferEngine(confidence_threshold=0.30)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_register_concept(self):
        dc = self.engine.register_concept("physics", "force", seed=1001)
        assert dc.domain == "physics"
        assert dc.name == "force"
        assert dc.hv is not None

    def test_register_rule(self):
        self.engine.register_concept("physics", "force", seed=1001)
        self.engine.register_concept("physics", "acceleration", seed=1002)
        rel = self.engine.register_rule(
            "physics", "force", "causes", "acceleration", confidence=0.9
        )
        assert rel.domain == "physics"
        assert rel.source == "force"
        assert rel.relation == "causes"
        assert rel.target == "acceleration"
        assert rel.confidence == 0.9
        assert rel.triple_hv is not None

    def test_register_correspondence(self):
        corr = self.engine.register_correspondence(
            "physics", "force", "finance", "return", structural_similarity=0.85
        )
        assert corr.source_concept == "force"
        assert corr.target_concept == "return"
        assert corr.structural_similarity == 0.85

    def test_list_domains(self):
        self.engine.register_concept("physics", "mass", seed=100)
        self.engine.register_concept("biology", "mass", seed=101)
        domains = self.engine.list_domains()
        assert "physics" in domains
        assert "biology" in domains

    def test_domain_summary(self):
        self.engine.register_concept("chemistry", "acid", seed=201)
        self.engine.register_concept("chemistry", "base", seed=202)
        self.engine.register_rule("chemistry", "acid", "reacts_with", "base")
        summary = self.engine.domain_summary("chemistry")
        assert summary["domain"] == "chemistry"
        assert summary["num_concepts"] == 2
        assert summary["num_relations"] == 1
        assert "acid" in summary["concept_names"]
        assert "reacts_with" in summary["relation_types"]

    # ------------------------------------------------------------------
    # Transfer with explicit correspondences
    # ------------------------------------------------------------------

    def test_transfer_with_explicit_correspondences(self):
        """Physics → Finance transfer using manually declared correspondences."""
        # Source domain: physics
        for name, seed in [("force", 1001), ("mass", 1002), ("acceleration", 1003)]:
            self.engine.register_concept("physics", name, seed=seed)
        self.engine.register_rule("physics", "force", "causes", "acceleration", 0.9)
        self.engine.register_rule("physics", "mass", "resists", "acceleration", 0.8)

        # Target domain: finance
        for name, seed in [("return", 2001), ("leverage", 2002), ("volatility", 2003)]:
            self.engine.register_concept("finance", name, seed=seed)

        # Explicit correspondences
        self.engine.register_correspondence("physics", "force", "finance", "return", 1.0)
        self.engine.register_correspondence("physics", "mass", "finance", "leverage", 1.0)
        self.engine.register_correspondence("physics", "acceleration", "finance", "volatility", 1.0)

        results = self.engine.transfer("physics", "finance")
        assert len(results) >= 1, "At least one rule should transfer"
        # All results should be in the target domain
        for r in results:
            assert r.target_domain == "finance"
        # Confidence should be ≤ source rule confidence
        for r in results:
            assert r.confidence <= 0.9

    def test_transfer_preserves_relation_label(self):
        """Transferred rules keep the same relation type."""
        self.engine.register_concept("ecology", "predator", seed=3001)
        self.engine.register_concept("ecology", "prey", seed=3002)
        self.engine.register_rule("ecology", "predator", "eats", "prey", 0.95)

        self.engine.register_concept("market", "buyer", seed=4001)
        self.engine.register_concept("market", "seller", seed=4002)

        self.engine.register_correspondence("ecology", "predator", "market", "buyer", 1.0)
        self.engine.register_correspondence("ecology", "prey", "market", "seller", 1.0)

        results = self.engine.transfer("ecology", "market")
        assert len(results) >= 1
        assert results[0].target_relation == "eats"

    def test_transfer_returns_empty_when_no_rules(self):
        self.engine.register_concept("empty_domain", "thing", seed=5001)
        results = self.engine.transfer("empty_domain", "physics")
        assert results == []

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def test_discover_correspondences_same_seeds(self):
        """Concepts with the same seed should be highly similar."""
        # Use the same seed for both — guarantees similarity == 1.0
        self.engine.register_concept("domainA", "alpha", seed=9001)
        self.engine.register_concept("domainB", "alpha_copy", seed=9001)
        corrs = self.engine.discover_correspondences("domainA", "domainB", threshold=0.0)
        assert len(corrs) >= 1
        assert corrs[0].source_concept == "alpha"
        assert corrs[0].target_concept == "alpha_copy"

    # ------------------------------------------------------------------
    # Inference structure
    # ------------------------------------------------------------------

    def test_inference_summary_format(self):
        self.engine.register_concept("src", "A", seed=10001)
        self.engine.register_concept("src", "B", seed=10002)
        self.engine.register_rule("src", "A", "implies", "B", 0.8)

        self.engine.register_concept("tgt", "X", seed=11001)
        self.engine.register_concept("tgt", "Y", seed=11002)
        self.engine.register_correspondence("src", "A", "tgt", "X", 1.0)
        self.engine.register_correspondence("src", "B", "tgt", "Y", 1.0)

        results = self.engine.transfer("src", "tgt")
        assert len(results) == 1
        summary = results[0].summary()
        assert "src→tgt" in summary
        assert "implies" in summary
        assert "X" in summary
        assert "Y" in summary

    def test_confidence_ordering(self):
        """Results should be ordered by descending confidence."""
        for name, seed, conf in [
            ("a", 20001, 0.9),
            ("b", 20002, 0.6),
            ("c", 20003, 0.8),
        ]:
            self.engine.register_concept("d1", name, seed=seed)
        self.engine.register_rule("d1", "a", "rel1", "b", 0.9)
        self.engine.register_rule("d1", "b", "rel2", "c", 0.6)
        self.engine.register_rule("d1", "a", "rel3", "c", 0.8)

        for name, seed in [("x", 21001), ("y", 21002), ("z", 21003)]:
            self.engine.register_concept("d2", name, seed=seed)
        for src, tgt in [("a", "x"), ("b", "y"), ("c", "z")]:
            self.engine.register_correspondence("d1", src, "d2", tgt, 1.0)

        results = self.engine.transfer("d1", "d2")
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)


class TestSchemaExtractor:
    def test_extract_groups_by_relation(self):
        from python.core.vsa.hypervec_shim import HyperVector
        extractor = SchemaExtractor()
        concepts = {
            "a": DomainConcept("d", "a", HyperVector(1)),
            "b": DomainConcept("d", "b", HyperVector(2)),
            "c": DomainConcept("d", "c", HyperVector(3)),
        }
        relations = [
            DomainRelation("d", "a", "causes", "b", 0.9),
            DomainRelation("d", "b", "causes", "c", 0.8),
            DomainRelation("d", "a", "isa", "c", 0.7),
        ]
        schemas = extractor.extract(relations, concepts)
        assert "causes" in schemas
        assert "isa" in schemas


class TestStructureMapper:
    def test_explicit_correspondences_bypass_auto(self):
        from python.core.vsa.hypervec_shim import HyperVector
        mapper = StructureMapper()
        src = {"a": DomainConcept("s", "a", HyperVector(1))}
        tgt = {"x": DomainConcept("t", "x", HyperVector(2))}
        explicit = [ConceptCorrespondence("s", "a", "t", "x", 0.9)]
        result = mapper.map(src, tgt, explicit=explicit)
        assert result == explicit


class TestRuleLifter:
    def test_lift_rule_with_known_mapping(self):
        from python.core.vsa.hypervec_shim import HyperVector
        lifter = RuleLifter()
        src_rel = DomainRelation("src", "heat", "causes", "expansion", 0.9)
        corrs = [
            ConceptCorrespondence("src", "heat", "tgt", "pressure", 1.0),
            ConceptCorrespondence("src", "expansion", "tgt", "volume_increase", 1.0),
        ]
        results = lifter.lift([src_rel], corrs, "tgt", confidence_threshold=0.0)
        assert len(results) == 1
        r = results[0]
        assert r.target_source == "pressure"
        assert r.target_target == "volume_increase"
        assert r.target_relation == "causes"
        assert r.confidence == pytest.approx(0.9, abs=0.01)
