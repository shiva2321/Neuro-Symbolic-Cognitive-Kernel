"""
NCGN Datasets Module - Training Data Structures

Provides data structures and utilities for training curricula:
- Fact: A single knowledge fact
- Association: A directed relationship between concepts
- DatasetLoader: Load training data from various formats
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import json
import os


class RelationType(Enum):
    """Types of relationships in the knowledge graph."""
    EATS = "eats"           # Animal -> Food
    IS_A = "is_a"           # Instance -> Category
    HAS_PROPERTY = "has_property"  # Object -> Property
    CAN_DO = "can_do"       # Agent -> Action
    CAUSES = "causes"       # Event -> Event
    TEMPORAL_NEXT = "temporal_next"  # Sequence
    ASSOCIATED = "associated"  # General association


@dataclass
class Fact:
    """A single knowledge fact (property assertion)."""
    subject: str
    property_name: str
    value: bool
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "property": self.property_name,
            "value": self.value,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Fact':
        return cls(
            subject=data["subject"],
            property_name=data["property"],
            value=data["value"],
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class Association:
    """A directed relationship between two concepts."""
    source: str
    target: str
    relation_type: RelationType
    weight: float = 0.5
    confidence: float = 0.5
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Association':
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=RelationType(data.get("relation", "associated")),
            weight=data.get("weight", 0.5),
            confidence=data.get("confidence", 0.5)
        )


@dataclass
class KnowledgeBase:
    """Collection of facts and associations."""
    name: str
    facts: List[Fact] = field(default_factory=list)
    associations: List[Association] = field(default_factory=list)
    
    def add_fact(self, subject: str, property_name: str, value: bool, confidence: float = 1.0):
        """Add a fact to the knowledge base."""
        self.facts.append(Fact(subject, property_name, value, confidence))
    
    def add_association(
        self, 
        source: str, 
        target: str, 
        relation: RelationType,
        weight: float = 0.5,
        confidence: float = 0.5
    ):
        """Add an association to the knowledge base."""
        self.associations.append(Association(source, target, relation, weight, confidence))
    
    def get_facts_for(self, subject: str) -> List[Fact]:
        """Get all facts about a subject."""
        return [f for f in self.facts if f.subject == subject]
    
    def get_associations_from(self, source: str) -> List[Association]:
        """Get all associations from a source."""
        return [a for a in self.associations if a.source == source]
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "facts": [f.to_dict() for f in self.facts],
            "associations": [a.to_dict() for a in self.associations]
        }
    
    def save(self, filepath: str):
        """Save knowledge base to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeBase':
        """Load knowledge base from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        kb = cls(name=data.get("name", "unnamed"))
        kb.facts = [Fact.from_dict(f) for f in data.get("facts", [])]
        kb.associations = [Association.from_dict(a) for a in data.get("associations", [])]
        return kb


class DatasetLoader:
    """Load and manage training datasets."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.curricula_dir = os.path.join(base_dir, "curricula")
        self.evaluation_dir = os.path.join(base_dir, "evaluation")
    
    def ensure_directories(self):
        """Create dataset directories if they don't exist."""
        os.makedirs(self.curricula_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
    
    def list_curricula(self) -> List[str]:
        """List available curriculum files."""
        if not os.path.exists(self.curricula_dir):
            return []
        return [f for f in os.listdir(self.curricula_dir) if f.endswith('.json')]
    
    def load_curriculum_file(self, filename: str) -> dict:
        """Load a single curriculum JSON file."""
        filepath = os.path.join(self.curricula_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_evaluation_set(self, filename: str = "holdout_set.json") -> List[dict]:
        """Load evaluation/holdout examples."""
        filepath = os.path.join(self.evaluation_dir, filename)
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            return json.load(f)


def create_default_knowledge_base() -> KnowledgeBase:
    """Create a default knowledge base for testing."""
    kb = KnowledgeBase("default")
    
    # Animal-food associations
    kb.add_association("dog", "meat", RelationType.EATS, weight=0.9, confidence=0.9)
    kb.add_association("cat", "fish", RelationType.EATS, weight=0.9, confidence=0.9)
    kb.add_association("rabbit", "carrot", RelationType.EATS, weight=0.9, confidence=0.9)
    kb.add_association("bird", "seed", RelationType.EATS, weight=0.9, confidence=0.9)
    kb.add_association("cow", "grass", RelationType.EATS, weight=0.9, confidence=0.9)
    
    # Category memberships
    kb.add_association("dog", "animal", RelationType.IS_A, weight=0.95, confidence=0.99)
    kb.add_association("cat", "animal", RelationType.IS_A, weight=0.95, confidence=0.99)
    kb.add_association("meat", "food", RelationType.IS_A, weight=0.95, confidence=0.99)
    kb.add_association("fish", "food", RelationType.IS_A, weight=0.95, confidence=0.99)
    
    # Edibility facts
    kb.add_fact("meat", "is_edible", True, confidence=1.0)
    kb.add_fact("fish", "is_edible", True, confidence=1.0)
    kb.add_fact("carrot", "is_edible", True, confidence=1.0)
    kb.add_fact("seed", "is_edible", True, confidence=1.0)
    kb.add_fact("grass", "is_edible", True, confidence=1.0)
    kb.add_fact("metal", "is_edible", False, confidence=1.0)
    kb.add_fact("plastic", "is_edible", False, confidence=1.0)
    kb.add_fact("rock", "is_edible", False, confidence=1.0)
    
    # Animacy facts
    kb.add_fact("dog", "is_animate", True, confidence=1.0)
    kb.add_fact("cat", "is_animate", True, confidence=1.0)
    kb.add_fact("rabbit", "is_animate", True, confidence=1.0)
    kb.add_fact("metal", "is_animate", False, confidence=1.0)
    kb.add_fact("meat", "is_animate", False, confidence=1.0)
    
    return kb
