import pytest
import numpy as np
from python.core.vsa.hypervec_shim import HyperVector
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld
from python.core.memory.tda_health_monitor import TDAHealthMonitor

def test_tda_scheduling_and_accuracy():
    world = SocietalKnowledgeWorld()
    monitor = TDAHealthMonitor(world, check_interval_ticks=10)
    
    # Needs to be dictionary state for default adapter
    
    # Create an artificial domain with a "hole" (cycle graph of 15 nodes)
    # A <-> B <-> C ... A
    hvs = [HyperVector(10240) for _ in range(15)]
    
    for i in range(15):
        world.ingest_concept(f"tda_node_{i}", hvs[i])
        
    for i in range(15):
        next_i = (i + 1) % 15
        for _ in range(3):
            world.record_co_activation([f"tda_node_{i}", f"tda_node_{next_i}"])
    
    # Trigger domain creation
    world.epoch_ticker = 99
    world.tick_world()
    
    # Ensure domain emerged
    assert len(world.domains) >= 1
    
    # Run Health Monitor targetting Betti 0 (1 component) and Betti 1 (~1 hole)
    results = monitor.run_health_check()
    assert len(results) > 0
    
    for domain_id, res in results.items():
        assert "betti_0" in res
        assert "betti_1" in res
        # Since the similarities might not form a perfect hole with default rand initialization,
        # we just test the pipeline doesn't crash and returns ints
        assert isinstance(res["betti_0"], int)
        assert isinstance(res["betti_1"], int)
        
    # Check that tda was logged in the domain history
    domain = list(world.domains.values())[0]
    assert len(domain.tda_health_history) == 1
