"""
NSCK Semantic Memory Module (Phase 3.1)
=======================================
Structured knowledge base of concepts and relations.
Implements abstracted knowledge (Schemas) and spreading activation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import python.core.vsa.hypervec_shim as hypervec_rs

class SemanticMemory:
    """
    Structured knowledge base of concepts and relations.
    Built from episodic memory consolidation during sleep.
    """
    def __init__(self):
        # Graph database of concepts
        self.concept_graph = nx.DiGraph()
        
        # Concept -> HyperVector mapping
        self.concept_hvs: Dict[str, hypervec_rs.HyperVector] = {}
        
        # Relation types
        self.relations = ["is_a", "has_property", "causes", "part_of", "similar_to"]
        
        print("SemanticMemory Initialized.")
    
    def reset(self):
        """Clear all semantic knowledge and re-initialize."""
        self.concept_graph = nx.DiGraph()
        self.concept_hvs = {}
        print("[SEMANTIC] Memory reset complete.")
    
    def add_concept(self, concept_name: str, properties: Dict[str, Any], hv_override: Optional[hypervec_rs.HyperVector] = None):
        """
        Add a new concept to semantic memory.
        
        Args:
            concept_name: Name of the concept (e.g. "Apple")
            properties: Dictionary of properties
            hv_override: Optional pre-calculated hypervector
        """
        if hv_override:
            hv = hv_override
        else:
            # Generate base hypervector for concept name
            hv = hypervec_rs.HyperVector(hash(concept_name) % (2**32))
            
            # Bind properties into concept HV
            for prop, value in properties.items():
                prop_hv = hypervec_rs.HyperVector(hash(prop) % (2**32))
                value_hv = hypervec_rs.HyperVector(hash(str(value)) % (2**32))
                # Role-filler binding: XOR the property role with value, then bundle into concept
                bound = prop_hv.xor(value_hv)
                hv = hv.bundle(bound)
        
        self.concept_hvs[concept_name] = hv
        self.concept_graph.add_node(concept_name, **properties)
    
    def add_relation(self, concept1: str, relation: str, concept2: str, timestamp: float = 0.0):
        """
        Add relation between concepts with temporal validation.
        Only updates if the new information is more recent or same time.
        """
        if concept1 not in self.concept_graph or concept2 not in self.concept_graph:
            return
            
        # Check if edge exists
        if self.concept_graph.has_edge(concept1, concept2):
            existing_data = self.concept_graph.get_edge_data(concept1, concept2)
            existing_time = existing_data.get('timestamp', 0.0)
            
            # If explicit timestamp provided and it's older than existing knowledge, IGNORE
            if timestamp > 0 and timestamp < existing_time:
                # [Belief Revision] Reject outdated info
                return

        # Update or create edge
        self.concept_graph.add_edge(concept1, concept2, relation=relation, timestamp=timestamp)
    
    def query(self, query_hv: hypervec_rs.HyperVector, k: int = 5) -> List[Tuple[str, float]]:
        """Find concepts most similar to query HV."""
        similarities = []
        for concept_name, concept_hv in self.concept_hvs.items():
            sim = query_hv.similarity(concept_hv)
            similarities.append((concept_name, sim))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def spread_activation(self, start_concepts: List[str], steps: int = 3, decay: float = 0.7) -> Dict[str, float]:
        """
        Spreading activation for associative retrieval.
        Returns activation levels for nodes.
        """
        activation = {c: 1.0 for c in start_concepts if c in self.concept_graph}
        
        for _ in range(steps):
            new_activation = activation.copy()
            for concept, act in activation.items():
                if act < 0.01: continue
                
                # Spread to neighbors
                neighbors = list(self.concept_graph.neighbors(concept))
                if not neighbors: continue
                
                spread_val = (act * decay) / len(neighbors)
                for neighbor in neighbors:
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread_val
            
            activation = new_activation
            
        return activation
    
    def extract_schema(self, concept: str) -> Dict[str, Any]:
        """Extract abstracted structure (Schema) for a concept."""
        if concept not in self.concept_graph:
            return {}
        
        schema = {
            "name": concept,
            "properties": dict(self.concept_graph.nodes[concept]),
            "is_a": [],
            "parts": [],
            "causes": [],
            "caused_by": []
        }
        
        # Outgoing relations
        for _, neighbor, data in self.concept_graph.out_edges(concept, data=True):
            rel = data.get("relation")
            if rel == "is_a": schema["is_a"].append(neighbor)
            elif rel == "part_of": schema["parts"].append(neighbor)
            elif rel == "causes": schema["causes"].append(neighbor)
            
        # Incoming relations
        for neighbor, _, data in self.concept_graph.in_edges(concept, data=True):
            rel = data.get("relation")
            if rel == "causes": schema["caused_by"].append(neighbor)
            
        return schema
