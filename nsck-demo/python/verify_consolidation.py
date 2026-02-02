"""
Verify Sleep Consolidation (Phase 3.1)
======================================
Tests the extraction of Semantic knowledge from Episodic memory.
"""

from episodic_memory import EpisodicMemory, LiveEpisode
from semantic_memory import SemanticMemory
from learning import SleepConsolidator
import time

# Mock HyperVector 
class MockHV:
    def bits(self): return b''
    def similarity(self, other): return 0

def main():
    print("--- Testing Sleep Consolidation ---")
    
    # 1. Setup Memory
    em = EpisodicMemory()
    sm = SemanticMemory()
    consolidator = SleepConsolidator(em, sm)
    
    task = "snake"
    
    # 2. Add some consistent experiences (Rule of Three)
    # Action "RIGHT" in "joy" state leads to reward
    for _ in range(3):
        ep = LiveEpisode(
            timestamp=time.time(),
            task_tag=task,
            situation_hv=MockHV(),
            state={},
            action="RIGHT",
            outcome="move",
            reward=1.0,
            emotion="joy"
        )
        em.record(ep)
        
    # 3. Run Consolidation
    consolidator.consolidate_during_sleep(task)
    
    # 4. Check Semantic Memory
    if "RIGHT" in sm.concept_graph and sm.concept_graph.has_edge("RIGHT", f"reward_{task}"):
        print("[PASS] Semantic Memory extracted causal relation (RIGHT -> reward).")
        
        # Check relation data
        edge_data = sm.concept_graph.get_edge_data("RIGHT", f"reward_{task}")
        if edge_data.get("relation") == "causes":
            print("[PASS] Relation type is 'causes'.")
    else:
        print("[FAIL] Relation not found in Semantic Memory.")
        print(f"Nodes: {sm.concept_graph.nodes}")
        print(f"Edges: {sm.concept_graph.edges}")

if __name__ == "__main__":
    main()
