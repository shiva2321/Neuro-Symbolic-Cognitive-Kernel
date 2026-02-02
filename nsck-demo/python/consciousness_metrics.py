"""
NSCK Consciousness Metrics Module (Phase 5.1)
=============================================
Implements Integrated Information Theory (IIT 3.0) approximations.
Computes Phi (Φ) - the core metric of information integration.
Includes Attention Schema Theory (AST) correlates.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Any

class ConsciousnessMonitor:
    """
    Computes computational correlates of consciousness for the AGI.
    """
    def __init__(self, global_workspace: Any):
        self.gws = global_workspace
        self.phi_history = []
        self.attention_schema = {
            "focus": None,
            "intensity": 0.0,
            "latency": 0.0
        }

    def compute_phi(self, system_graph: nx.DiGraph) -> float:
        """
        Estimate Phi (Φ) using the current interaction graph of cognitive modules.
        This is a simplified version of IIT 3.0's integrated information.
        Φ = Whole_Mechanism_Information - sum(Part_Mechanism_Information)
        """
        # 1. Total Information in the Graph (Entropy approximate)
        # We use the edge weights (influence) as probabilities
        nodes = list(system_graph.nodes())
        if len(nodes) < 2: return 0.0
        
        # 2. Find Minimum Information Partition (MIP)
        # For simplicity, we look at the spectral bisection or just the weakest cut.
        # IIT requires finding the partition that 'loses' the least information.
        cut_value, partition = self._find_minimal_cut(system_graph)
        
        # 3. Phi is the value of this minimal cut (Integrated Information)
        # If the cut is 0, the system is perfectly decomposable (Phi=0)
        phi = cut_value
        
        self.phi_history.append(phi)
        return phi

    def _find_minimal_cut(self, graph: nx.DiGraph) -> Tuple[float, List[Set]]:
        """Identify the weakest link in the system's causal integrity."""
        try:
            # We treat influence weights as capacity
            undirected = graph.to_undirected()
            cut_value, partition = nx.stoer_wagner(undirected)
            return cut_value, partition
        except Exception:
            return 0.0, [set(graph.nodes()), set()]

    def update_attention_schema(self, focused_concept: str, salience: float):
        """
        Attention Schema Theory (AST): 
        The system maintains a meta-representation of its own attention.
        """
        self.attention_schema["focus"] = focused_concept
        self.attention_schema["intensity"] = salience
        
    def get_subjective_report(self) -> str:
        """Generate a simulated 'subjective' report based on consciousness metrics."""
        phi = self.phi_history[-1] if self.phi_history else 0.0
        focus = self.attention_schema["focus"]
        
        state = "HIGHLY INTEGRATED" if phi > 0.5 else "FRAGMENTED"
        if focus:
            return f"I am currently in a {state} state (Phi={phi:.2f}), primarily attending to '{focus}'."
        return f"I am in a {state} state, but my attention is currently unfocused."

class GlobalWorkspaceMetrics:
    """Helper for extracting system graph from Global Workspace."""
    @staticmethod
    def extract_influence_graph(coalitions: List[Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        # Each coalition is a node
        for c in coalitions:
            G.add_node(c.source, weight=c.base_salience)
            
        # Add edges representing shared content or cross-module influence
        # (Heuristic: Similarity in VSA space)
        for i, c1 in enumerate(coalitions):
            for j, c2 in enumerate(coalitions):
                if i != j:
                    # Assume influence is proportional to confidence and workspace salience
                    influence = c1.sender_confidence * c2.sender_confidence
                    G.add_edge(c1.source, c2.source, weight=influence)
        return G
