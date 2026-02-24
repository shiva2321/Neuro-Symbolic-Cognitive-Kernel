"""Tests for HierarchicalResonatorNetwork (V8)."""
import pytest

from python.core.vsa.resonator import ResonatorNetwork, HierarchicalResonatorNetwork
from python.core.memory.semantic_memory import SemanticMemory
import python.core.vsa.hypervec_shim as hv


@pytest.fixture()
def codebooks():
    """Minimal codebooks with a few concepts each."""
    agent_mem = SemanticMemory()
    for name in ["alice", "bob", "carol"]:
        agent_mem.add_concept(name, {"type": "agent"})

    verb_mem = SemanticMemory()
    for name in ["chase", "eat", "see"]:
        verb_mem.add_concept(name, {"type": "verb"})

    patient_mem = SemanticMemory()
    for name in ["dog", "cat", "fish"]:
        patient_mem.add_concept(name, {"type": "patient"})

    return {"AGENT": agent_mem, "VERB": verb_mem, "PATIENT": patient_mem}


@pytest.fixture()
def hrn(codebooks):
    return HierarchicalResonatorNetwork(codebooks)


class TestHierarchicalResonatorInit:
    def test_creates_l1(self, hrn):
        assert hrn._l1 is not None

    def test_creates_l2(self, hrn):
        assert hrn._l2 is not None

    def test_codebooks_stored(self, hrn, codebooks):
        assert hrn.codebooks is codebooks

    def test_l1_roles(self):
        assert "AGENT" in HierarchicalResonatorNetwork.L1_ROLES
        assert "VERB" in HierarchicalResonatorNetwork.L1_ROLES
        assert "PATIENT" in HierarchicalResonatorNetwork.L1_ROLES

    def test_l2_roles(self):
        assert "MODIFIER_AGENT" in HierarchicalResonatorNetwork.L2_ROLES


class TestFactorizeHierarchical:
    def test_returns_dict(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize_hierarchical(target)
        assert isinstance(result, dict)

    def test_has_l1_key(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize_hierarchical(target)
        assert "L1" in result

    def test_has_l2_key_when_depth_2(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize_hierarchical(target, depth=2)
        assert "L2" in result

    def test_no_l2_when_depth_1(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize_hierarchical(target, depth=1)
        assert "L2" not in result

    def test_l1_result_is_dict(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize_hierarchical(target)
        assert isinstance(result["L1"], dict)


class TestFactorize:
    def test_compatibility_method(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize(target)
        assert isinstance(result, dict)

    def test_factorize_with_composite(self, hrn, codebooks):
        # Build composite: agent XOR verb XOR patient
        alice = codebooks["AGENT"].concept_hvs["alice"]
        chase = codebooks["VERB"].concept_hvs["chase"]
        dog = codebooks["PATIENT"].concept_hvs["dog"]
        composite = alice.xor(chase).xor(dog)
        result = hrn.factorize(composite, max_iter=5)
        assert isinstance(result, dict)

    def test_factorize_returns_roles(self, hrn, codebooks):
        target = hv.HyperVector(99)
        result = hrn.factorize(target, max_iter=3)
        # Should have entries for AGENT, VERB, PATIENT
        for role in ["AGENT", "VERB", "PATIENT"]:
            assert role in result

    def test_factorize_confidence_in_range(self, hrn, codebooks):
        target = hv.HyperVector(42)
        result = hrn.factorize(target, max_iter=3)
        for role, (name, conf) in result.items():
            assert 0.0 <= conf <= 1.0

    def test_verbose_mode(self, codebooks):
        hrn_v = HierarchicalResonatorNetwork(codebooks, verbose=True)
        target = hv.HyperVector(42)
        result = hrn_v.factorize(target, max_iter=2)
        assert isinstance(result, dict)
