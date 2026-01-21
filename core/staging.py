"""
NCGN Staging Module - The Hippocampus (Shadow Graph)

Temporary buffer for unverified knowledge before it enters
the main GraphMemory. Prevents polluting the brain with
unverified noise from file ingestion.

Key Principle: NEVER inject directly into main memory.
Stage → Schema-check → Flag → Wait for user commit.

Workflow:
1. User uploads biology.txt
2. System parses text into StagingBuffer
3. Each triple is schema-checked against System 2
4. System reports: "Found 150 concepts, 300 links, 12 flagged"
5. User reviews flagged items
6. User says: "Commit"
7. Consolidation: StagingBuffer merges into Main GraphMemory
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ingestion import Triple, RelationType


class ConflictType(Enum):
    """Types of conflicts that can occur during staging."""
    SCHEMA_VIOLATION = "schema_violation"    # Triple violates a schema constraint
    CONTRADICTION = "contradiction"          # Conflicts with existing knowledge
    DUPLICATE = "duplicate"                  # Already exists in main memory
    UNKNOWN_PROPERTY = "unknown_property"    # Uses unknown property in constraint


@dataclass
class ConflictInfo:
    """Information about a conflict detected during staging."""
    triple: Triple
    conflict_type: ConflictType
    reason: str
    existing_value: Optional[str] = None  # What main memory says
    resolution: Optional[str] = None       # How it was resolved


@dataclass
class MergeResult:
    """Result of merging staging buffer to main memory."""
    nodes_added: int = 0
    nodes_updated: int = 0         # Existing nodes with confidence boost
    edges_added: int = 0
    edges_updated: int = 0         # Existing edges with confidence averaged
    conflicts_skipped: int = 0     # Flagged items not merged
    total_triples: int = 0
    
    def __repr__(self) -> str:
        return (f"MergeResult(+{self.nodes_added} nodes, +{self.edges_added} edges, "
                f"~{self.nodes_updated + self.edges_updated} updated, "
                f"⚠️{self.conflicts_skipped} skipped)")


class StagingBuffer:
    """
    Temporary graph holding unverified knowledge.
    
    The "Hippocampus" - stages new knowledge before consolidation
    into long-term memory (main GraphMemory).
    
    CRITICAL RULES:
    1. No energy injection into main memory
    2. No System 1 firing during staging
    3. Schema-check everything before flagging
    4. User must explicitly commit
    """
    
    def __init__(self):
        # All staged triples
        self.triples: List[Triple] = []
        
        # Triples that passed schema check
        self.clean_triples: List[Triple] = []
        
        # Triples that failed schema check (with reason)
        self.flagged_triples: List[Tuple[Triple, ConflictInfo]] = []
        
        # Track unique nodes and edges for stats
        self._staged_nodes: Set[str] = set()
        self._staged_edges: Set[Tuple[str, str, str]] = set()  # (subj, pred, obj)
        
        # Source tracking
        self.source_file: Optional[str] = None
        self.source_description: str = ""
    
    def clear(self):
        """Clear all staged content."""
        self.triples.clear()
        self.clean_triples.clear()
        self.flagged_triples.clear()
        self._staged_nodes.clear()
        self._staged_edges.clear()
        self.source_file = None
        self.source_description = ""
    
    def add_triple(
        self,
        triple: Triple,
        system2=None,  # System2Controller
        main_memory=None  # GraphMemory for conflict detection
    ) -> bool:
        """
        Add a triple to staging, checking against schemas.
        
        Args:
            triple: The triple to stage
            system2: System2Controller for schema validation
            main_memory: Main GraphMemory for conflict detection
        
        Returns:
            True if triple is clean, False if flagged
        """
        self.triples.append(triple)
        
        # Track nodes
        self._staged_nodes.add(triple.subject)
        self._staged_nodes.add(triple.object)
        
        # Track edge
        edge_key = (triple.subject, triple.predicate, triple.object)
        self._staged_edges.add(edge_key)
        
        # Schema check if System2 available
        if system2:
            conflict = self._schema_check(triple, system2)
            if conflict:
                triple.flagged = True
                triple.flag_reason = conflict.reason
                self.flagged_triples.append((triple, conflict))
                return False
        
        # Conflict check against main memory
        if main_memory:
            conflict = self._conflict_check(triple, main_memory)
            if conflict:
                triple.flagged = True
                triple.flag_reason = conflict.reason
                self.flagged_triples.append((triple, conflict))
                return False
        
        self.clean_triples.append(triple)
        return True
    
    def add_triples(
        self,
        triples: List[Triple],
        system2=None,
        main_memory=None
    ) -> Tuple[int, int]:
        """
        Add multiple triples. Returns (clean_count, flagged_count).
        """
        clean = 0
        flagged = 0
        
        for triple in triples:
            if self.add_triple(triple, system2, main_memory):
                clean += 1
            else:
                flagged += 1
        
        return clean, flagged
    
    def _schema_check(self, triple: Triple, system2) -> Optional[ConflictInfo]:
        """
        Check triple against System 2 schemas.
        
        Example: "Dog eats Metal" would fail if schema requires
        Eat.target to have is_edible=True.
        """
        from core.system2 import Triple as S2Triple, DiagnosisType
        
        # Map to System 2's Triple format
        action = triple.predicate
        
        # Find matching schema (System2 stores by action name, not schema_action)
        schema = system2.schemas.get(action)
        if not schema:
            # No schema for this action - allow by default
            return None
        
        # Check constraints on the object
        target_constraints = schema.constraints.get('target', [])
        
        for constraint in target_constraints:
            # Get property value from System 2
            prop_value = system2.get_property(triple.object, constraint)
            
            if prop_value is None:
                # Unknown property - flag as needing clarification
                return ConflictInfo(
                    triple=triple,
                    conflict_type=ConflictType.UNKNOWN_PROPERTY,
                    reason=f"Unknown property: {triple.object}.{constraint}"
                )
            
            if prop_value is False:
                # Constraint violated
                return ConflictInfo(
                    triple=triple,
                    conflict_type=ConflictType.SCHEMA_VIOLATION,
                    reason=f"Schema violation: {triple.object} does not satisfy {constraint}"
                )
        
        return None
    
    def _conflict_check(
        self,
        triple: Triple,
        main_memory  # GraphMemory
    ) -> Optional[ConflictInfo]:
        """
        Check for conflicts with existing knowledge.
        
        Example: Staging says "Sky is Green" but Main says "Sky is Blue"
        """
        # Check if contradictory edge exists
        if triple.relation_type == RelationType.PROPERTY or triple.relation_type == RelationType.IS_A:
            # Look for existing edges from subject with same predicate
            outgoing = main_memory.get_outgoing(triple.subject)
            
            for synapse in outgoing:
                if synapse.type == triple.predicate:
                    # Same predicate, different target = potential conflict
                    if synapse.target_id != triple.object:
                        # Check if this is negation scenario
                        if triple.negated:
                            # Explicit negation - definitely a conflict
                            return ConflictInfo(
                                triple=triple,
                                conflict_type=ConflictType.CONTRADICTION,
                                reason=f"Contradiction: {triple.subject} {triple.predicate} {synapse.target_id} (existing) vs NOT {triple.object} (new)",
                                existing_value=synapse.target_id
                            )
        
        # Check for duplicate
        edge_key = (triple.subject, triple.predicate, triple.object)
        existing_synapse = main_memory.get_synapse(triple.subject, triple.object, triple.predicate)
        
        if existing_synapse:
            return ConflictInfo(
                triple=triple,
                conflict_type=ConflictType.DUPLICATE,
                reason=f"Duplicate: {triple.subject} --[{triple.predicate}]--> {triple.object} already exists"
            )
        
        return None
    
    def get_conflicts(self, main_memory) -> List[Tuple[Triple, str]]:
        """
        Find all triples that conflict with main memory.
        
        Returns list of (triple, reason) tuples.
        """
        conflicts = []
        
        for triple in self.clean_triples:
            conflict = self._conflict_check(triple, main_memory)
            if conflict:
                conflicts.append((triple, conflict.reason))
        
        return conflicts
    
    def merge_to_main(
        self,
        main_memory,  # GraphMemory
        include_flagged: bool = False,
        approved_indices: Optional[List[int]] = None
    ) -> MergeResult:
        """
        Merge staged triples to main graph memory.
        
        Args:
            main_memory: The main GraphMemory to merge into
            include_flagged: If True, also merge flagged triples
            approved_indices: If provided, only merge flagged triples at these indices
        
        Rules:
        - If node exists: increase novelty decay (it's reinforced), don't duplicate
        - If edge exists: average confidence values
        - Flagged triples require explicit approval
        
        Returns:
            MergeResult with statistics
        """
        result = MergeResult()
        result.total_triples = len(self.clean_triples)
        
        # Merge clean triples
        for triple in self.clean_triples:
            self._merge_triple(triple, main_memory, result)
        
        # Handle flagged triples
        if include_flagged:
            for triple, _ in self.flagged_triples:
                self._merge_triple(triple, main_memory, result)
                result.conflicts_skipped -= 1  # Un-skip if merging
        elif approved_indices:
            for idx in approved_indices:
                if 0 <= idx < len(self.flagged_triples):
                    triple, _ = self.flagged_triples[idx]
                    self._merge_triple(triple, main_memory, result)
        
        result.conflicts_skipped = len(self.flagged_triples) - (
            len(approved_indices) if approved_indices else 0
        )
        
        return result
    
    def _merge_triple(self, triple: Triple, main_memory, result: MergeResult):
        """Merge a single triple into main memory."""
        # Add/update subject node
        if main_memory.has_node(triple.subject):
            # Node exists - decrease novelty (reinforcement)
            node = main_memory.get_node(triple.subject)
            node.novelty_score = max(0.0, node.novelty_score * 0.9)
            result.nodes_updated += 1
        else:
            # New node
            main_memory.add_node(triple.subject, novelty_score=0.5)  # Medium novelty
            result.nodes_added += 1
        
        # Add/update object node
        if main_memory.has_node(triple.object):
            node = main_memory.get_node(triple.object)
            node.novelty_score = max(0.0, node.novelty_score * 0.9)
            result.nodes_updated += 1
        else:
            main_memory.add_node(triple.object, novelty_score=0.5)
            result.nodes_added += 1
        
        # Add/update edge
        existing = main_memory.get_synapse(triple.subject, triple.object, triple.predicate)
        
        if existing:
            # Average confidence
            existing.confidence = (existing.confidence + triple.confidence) / 2
            result.edges_updated += 1
        else:
            # New edge
            main_memory.add_synapse(
                triple.subject,
                triple.object,
                type=triple.predicate,
                weight=0.5,  # Default weight
                confidence=triple.confidence
            )
            result.edges_added += 1
    
    def get_summary(self) -> str:
        """Get human-readable summary of staged content."""
        lines = [
            f"📚 Staging Buffer Summary",
            f"   Source: {self.source_file or 'Unknown'}",
            f"   Total triples: {len(self.triples)}",
            f"   Clean: {len(self.clean_triples)} ✓",
            f"   Flagged: {len(self.flagged_triples)} ⚠️",
            f"   Unique nodes: {len(self._staged_nodes)}",
            f"   Unique edges: {len(self._staged_edges)}",
        ]
        
        if self.flagged_triples:
            lines.append("\n   ⚠️ Flagged items:")
            for i, (triple, info) in enumerate(self.flagged_triples[:5]):
                lines.append(f"   {i+1}. {triple}")
                lines.append(f"      Reason: {info.reason}")
            if len(self.flagged_triples) > 5:
                lines.append(f"   ... and {len(self.flagged_triples) - 5} more")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"StagingBuffer(triples={len(self.triples)}, clean={len(self.clean_triples)}, flagged={len(self.flagged_triples)})"


# CLI for testing
if __name__ == "__main__":
    from core.memory import GraphMemory
    from ingestion import text_to_triples
    
    print("=" * 60)
    print("NCGN Staging Buffer - Test")
    print("=" * 60)
    
    # Create staging buffer
    buffer = StagingBuffer()
    buffer.source_description = "Test sentences"
    
    # Parse some test text
    test_text = """
    Dogs eat meat.
    Cats eat fish.
    Birds fly in the sky.
    Penguins are birds.
    Dogs are animals.
    """
    
    triples = text_to_triples(test_text)
    print(f"\n📝 Parsed {len(triples)} triples from text")
    
    # Add to staging (no System2 check for basic test)
    clean, flagged = buffer.add_triples(triples)
    print(f"   Clean: {clean}, Flagged: {flagged}")
    
    print("\n" + buffer.get_summary())
    
    # Test merge
    main_memory = GraphMemory()
    result = buffer.merge_to_main(main_memory)
    
    print(f"\n📊 Merge Result: {result}")
    print(f"   Main memory now has {main_memory.node_count()} nodes, {main_memory.edge_count()} edges")
    
    print("\n" + "=" * 60)
