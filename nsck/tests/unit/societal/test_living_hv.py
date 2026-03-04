import pytest
import time
from python.core.vsa.hypervec_shim import HyperVector
from python.core.vsa.living_hv import LivingHyperVector, ValenceEngine

def test_living_hv_creation():
    hv = HyperVector(10240)
    lhv = LivingHyperVector(hv=hv, concept_id="test_node")
    
    assert lhv.concept_id == "test_node"
    assert lhv.valence == 4
    assert lhv.stability_class == "gas"
    assert lhv.hybridization_state == "unhybridized"
    assert lhv.activation_count == 0

def test_domain_affinity_update():
    hv = HyperVector(10240)
    lhv = LivingHyperVector(hv=hv, concept_id="test_node")
    
    lhv.update_domain_affinity("physics", 0.8)
    lhv.update_domain_affinity("chemistry", 0.2)
    
    # Should be normalized (0.8 / 1.0 = 0.8)
    assert abs(lhv.domain_affinities["physics"] - 0.8) < 1e-5
    assert lhv.primary_domain == "physics"

def test_valence_bonding():
    hv1 = HyperVector(10240)
    hv2 = hv1  # Identical vectors for strong bond
    
    lhv1 = LivingHyperVector(hv=hv1, concept_id="node1")
    lhv2 = LivingHyperVector(hv=hv2, concept_id="node2")
    
    engine = ValenceEngine(min_co_activations=1) # Ease constraint for test
    engine.record_co_activation("node1", "node2")
    
    success = engine.try_bind(lhv1, lhv2)
    
    assert success is True
    assert "node2" in lhv1.current_bonds
    assert "node1" in lhv2.current_bonds
    assert lhv1.current_bonds["node2"] > 0.9 # Should be very strong

def test_hybridization_sp3_sp():
    base_hv = HyperVector(10240)
    center = LivingHyperVector(hv=base_hv, concept_id="center")
    
    engine = ValenceEngine(min_co_activations=1, min_sim=0.1)
    
    # Bind to 4 neighbors
    for i in range(4):
        n_hv = base_hv.bundle(HyperVector(10240)) # slightly different
        n_lhv = LivingHyperVector(hv=n_hv, concept_id=f"n_{i}")
        engine.record_co_activation("center", f"n_{i}")
        engine.try_bind(center, n_lhv)
        
    assert center.hybridization_state == "sp3"
    assert len(center.current_bonds) == 4

def test_activation_history_rolling():
    lhv = LivingHyperVector(hv=HyperVector(10240), concept_id="history_test")
    # maxlen is 50
    for i in range(60):
        lhv.activate(i)
        
    assert lhv.activation_count == 60
    assert len(lhv.activation_history) == 50
    assert lhv.activation_history[0] == 10  # Dropped first 10
