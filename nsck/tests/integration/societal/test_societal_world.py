import pytest
from python.core.vsa.hypervec_shim import HyperVector
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld

def test_societal_world_ingestion_and_bonding():
    world = SocietalKnowledgeWorld(dim=10240)
    
    # Ingest concepts
    hv_cat = HyperVector(10240)
    hv_dog = HyperVector(10240)
    hv_car = HyperVector(10240)
    
    lhv_cat = world.ingest_concept("cat", hv_cat)
    lhv_dog = world.ingest_concept("dog", hv_dog)
    lhv_car = world.ingest_concept("car", hv_car)
    
    assert "cat" in world.registry
    assert "dog" in world.registry
    
    # Record co-occurrences
    world.record_co_activation(["cat", "dog"])
    world.record_co_activation(["cat", "dog"])
    
    # Tick world
    world.tick_world()
    
    # Should be bonded because of min_co_activations=2
    assert "dog" in lhv_cat.current_bonds
    assert "cat" in lhv_dog.current_bonds
    
    # Car should not be bonded
    assert "car" not in lhv_cat.current_bonds

def test_percolation_and_emergence():
    world = SocietalKnowledgeWorld(dim=10240)
    
    # We will force the epoch ticker so it triggers rebuilding on tick 100
    world.epoch_ticker = 99 
    
    # Ingest a fully connected lattice of 15 concepts
    # to guarantee percolation size ratio > 0.5
    hvs = [HyperVector(10240) for _ in range(15)]
    for i in range(15):
        world.ingest_concept(f"node_{i}", hvs[i])
        
    for i in range(14):
        # Record many co-activations to form strong bonds
        for _ in range(3):
            world.record_co_activation([f"node_{i}", f"node_{i+1}"])
            
    # Tick world (reaches 100 -> triggers rebuilding & RG percolation)
    world.tick_world()
    
    # Because it's 100th tick, it rebuilt global topology and detected percolation
    assert len(world.domains) > 0
    # There should be at least one domain formed under HNSW layers
    
    domain_names = list(world.domains.keys())
    d = world.domains[domain_names[0]]
    assert len(d.neighborhoods) > 0
    assert d.emergence_epoch == 100
    
def test_hnsw_semantic_search():
    world = SocietalKnowledgeWorld(dim=10240)
    
    query_hv = HyperVector(10240)
    world.ingest_concept("target", query_hv)
    
    for i in range(5):
        world.ingest_concept(f"noise_{i}", HyperVector(10240))
        
    results = world.semantic_search(query_hv, k=2)
    # The first result should be target since similarity is 1.0 (distance 0.0)
    assert len(results) > 0
    assert results[0][0] == "target"
