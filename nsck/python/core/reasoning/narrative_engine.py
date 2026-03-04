import time
from typing import Dict, List, Optional
from python.core.vsa.hypervec_shim import HyperVector
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld

class NarrativeEngine:
    """
    Reasoning engine structuring inferences via the Monomyth (Hero's Journey) 
    template mapping over the Societal Knowledge Graph. 
    A query "travels" through contexts.
    """
    
    STAGES = [
        "Status Quo", 
        "Call to Adventure", 
        "Threshold", 
        "Abyss", 
        "Transformation", 
        "Return"
    ]
    
    def __init__(self, world: SocietalKnowledgeWorld):
        self.world = world
        
    def generate_narrative_path(self, subject_hv: HyperVector, goal_hv: HyperVector) -> List[Dict[str, str]]:
        """
        Creates a reasoning chain simulating a journey from 'subject' to 'goal'
        across different semantic domains.
        """
        # HNSW routing gives us spatial waypoints
        start_nodes = self.world.semantic_search(subject_hv, k=1)
        end_nodes = self.world.semantic_search(goal_hv, k=1)
        
        if not start_nodes or not end_nodes:
            return [{"stage": "Failed", "context": "Missing anchors"}]
            
        start_id = start_nodes[0][0]
        end_id = end_nodes[0][0]
        
        # A true journey needs intermediate 'Threshold' and 'Abyss' nodes
        # We simulate this by drifting the subject vector by 25% towards goal repeatedly
        
        path = []
        current_hv = subject_hv
        
        for i, stage in enumerate(self.STAGES):
            if i == 0:
                current_id = start_id
            elif i == len(self.STAGES) - 1:
                current_id = end_id
            else:
                # Drift the concept slightly
                alpha = float(i) / float(len(self.STAGES) - 1)
                drifted_hv = subject_hv.weighted_bundle(goal_hv, weight=1.0 - alpha)
                
                # Where are we now in the societal graph?
                landmarks = self.world.semantic_search(drifted_hv, k=1)
                current_id = landmarks[0][0] if landmarks else "Unknown Territory"
                
            path.append({
                "stage": stage,
                "domain_concept": current_id,
                "action": f"Moving through semantic field of {current_id}"
            })
            
        return path
