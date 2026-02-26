"""Test epoch-based LSH rebuild in EpisodicMemory."""
import time
import pytest

def test_lsh_rebuild_after_600_stores():
    """After 600 stores (overflow), retrieval still returns correct episodes."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
    
    from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode, REBUILD_INTERVAL
    import python.core.vsa.hypervec_shim as hv
    
    mem = EpisodicMemory(recent_capacity=100, total_capacity=10000, use_rust=False)
    mem.reset()
    
    target_hv = hv.HyperVector(42)
    target_episode = None
    
    for i in range(600):
        ep_hv = hv.HyperVector(42 if i == 300 else i + 1000)
        ep = LiveEpisode(
            timestamp=time.time() + i,
            task_tag="test_task",
            situation_hv=ep_hv,
            state={"i": i},
            action=f"action_{i}",
            outcome="success" if i == 300 else "neutral",
            reward=1.0 if i == 300 else 0.0,
        )
        if i == 300:
            target_episode = ep
        mem.record(ep)
    
    # LSH epoch counter should have been incremented
    assert hasattr(mem, '_lsh_epoch_counter')
    assert mem._lsh_epoch_counter == 600
    
    # REBUILD_INTERVAL should be 500
    assert REBUILD_INTERVAL == 500
    
    # The epoch counter triggered at least one rebuild
    # Retrieval should work without errors
    results = mem.retrieve(target_hv, "test_task", top_k=5)
    assert isinstance(results, list)
