"""
NCGN v2.0 Conflict Detection Engine

Identifies contradictions in the knowledge graph logically, without LLM.
Conflict Types:
1. HARD_REJECT: New fact contradicts high-confidence existing fact.
2. CURIOSITY: New fact contradicts medium-confidence fact.
3. ACCEPT: No conflict or overrides low-confidence fact.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ConflictType(Enum):
    HARD_REJECT = "hard_reject"  # > 90% confidence conflict
    CURIOSITY = "curiosity"      # 50-90% confidence conflict
    ACCEPT = "accept"            # < 50% confidence (overwrite)
    NEW_KNOWLEDGE = "new"        # No conflict

@dataclass
class ConflictReport:
    type: ConflictType
    source: str
    target: str
    relation: str
    existing_target: Optional[str] = None
    existing_confidence: float = 0.0
    new_confidence: float = 0.0
    message: str = ""

class ConflictDetector:
    def __init__(self, topology, confidence_tracker):
        self.topology = topology
        self.confidence = confidence_tracker

    def check_conflict(
        self, 
        source: str, 
        relation: str, 
        target: str,
        proposed_confidence: float = 0.1
    ) -> ConflictReport:
        """
        Check if adding (source, target) with relation creates a conflict.
        
        Note: Currently NCGN v2 graph is undirected/unweighted in simple mode? 
        No, it's weighted. But relations are often implicit or just "associated".
        
        If we enforce specific relation types (e.g. edge attributes), we need 
        to lookup edges with that relation type.
        """
        
        # For v2.0 Phase 1, we treat ANY strong edge from Source as a potential conflict
        # if the context implies a singular answer (e.g. "Sun rises in [East]").
        # This is hard to do without schema. 
        # For now, we'll assume conflicts only if we try to connect incompatible nodes.
        
        # TODO: Real relation typing. For now, we check if generic connection exists
        # and if the new target is "far" from the old target in embedding space?
        # Or simpler: If edge (source, OLD_TARGET) exists and is strong, 
        # and OLD_TARGET != target, is it a conflict?
        # Not necessarily (A dog has ears, A dog has tail).
        
        # Strict Mode: If we are updating a PROPERTY (e.g. color=Red), check existing.
        # This requires node property check.
        
        return ConflictReport(
            type=ConflictType.NEW_KNOWLEDGE,
            source=source,
            target=target,
            relation=relation,
            message="No specific conflict logic implemented yet for generic edges."
        )

    def check_property_conflict(
        self,
        node: str,
        property_name: str,
        new_value: Any
    ) -> ConflictReport:
        """
        Check conflicts for node properties (e.g. is_mammal=True).
        """
        # This would interface with Brain's property store
        return ConflictReport(ConflictType.NEW_KNOWLEDGE, node, property_name, "property")
