from typing import List, Tuple
from python.core.vsa.hypervec_shim import HyperVector
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld

class SocietalContextRouter:
    """
    Routes reasoning queries directly to the most appropriate Knowledge Neighborhoods 
    and Domains using the HNSW spatial graph.
    """
    
    def __init__(self, world: SocietalKnowledgeWorld):
        self.world = world
        
    def route_query(self, query_hv: HyperVector, top_k_neighborhoods: int = 3) -> List[Tuple[str, float]]:
        """
        Takes a complex query hypervector and routes it to the most relevant
        societal domains, returning the specific Knowledge Neighborhood IDs
        that should handle the query context.
        """
        # Search the societal semantic space
        # HNSW layer 1/2 contain the emergent domains and neighborhoods
        # While the wrapper searches all, Neighborhood IDs are prefixed with "nh_"
        results = self.world.semantic_search(query_hv, k=25)
        
        neighborhood_hits = []
        for cid, score in results:
            if cid.startswith("nh_"):
                neighborhood_hits.append((cid, score))
                if len(neighborhood_hits) >= top_k_neighborhoods:
                    break
                    
        return neighborhood_hits
        
    def expand_context(self, neighborhood_id: str) -> List[str]:
        """
        Given a neighborhood, expand the query context to all its residents.
        """
        for domain in self.world.domains.values():
            if neighborhood_id in domain.neighborhoods:
                return list(domain.neighborhoods[neighborhood_id].members)
                
        return []
