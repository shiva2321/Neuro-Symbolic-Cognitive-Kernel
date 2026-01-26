"""
NCGN v7 Cognitive Schemas

Pydantic models for structured LLM output.
The 'reasoning' field is always FIRST to enforce Chain-of-Thought.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional


class ConceptNode(BaseModel):
    """Schema for a concept in the knowledge graph."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "dog",
                "properties": {"is_alive": True, "is_mammal": True}
            }
        }
    )
    
    label: str = Field(
        ..., 
        description="Unique name of the concept (lowercase, underscores for spaces)"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value properties of this concept (e.g., 'is_edible': True)"
    )


class RelationEdge(BaseModel):
    """Schema for a relation between concepts."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "dog",
                "target": "mammal", 
                "relation_type": "is_a",
                "weight": 0.9
            }
        }
    )
    
    source: str = Field(
        ..., 
        description="Source concept label"
    )
    target: str = Field(
        ..., 
        description="Target concept label"
    )
    relation_type: str = Field(
        ..., 
        description="Type of relation (e.g., 'is_a', 'causes', 'has_property')"
    )
    weight: float = Field(
        default=0.5,
        ge=0.0, 
        le=1.0,
        description="Strength of relation (0.0 = weak, 1.0 = strong)"
    )


class KnowledgeGraphUpdate(BaseModel):
    """
    Output schema for LLM graph updates.
    
    IMPORTANT: The 'reasoning' field MUST be generated first 
    to enforce Chain-of-Thought prompting.
    """
    
    reasoning: str = Field(
        ..., 
        description="Step-by-step explanation of why these updates are needed. "
                    "Think through the connections before proposing changes."
    )
    new_nodes: List[ConceptNode] = Field(
        default_factory=list,
        description="New concepts to add to the graph"
    )
    new_edges: List[RelationEdge] = Field(
        default_factory=list,
        description="New relations to add between concepts"
    )
    nodes_to_remove: List[str] = Field(
        default_factory=list,
        description="Concept labels to remove from the graph"
    )
    edges_to_weaken: List[RelationEdge] = Field(
        default_factory=list,
        description="Existing relations to weaken (apply LTD)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reasoning": "The user mentioned that dogs eat meat. This establishes "
                             "'dog' as an agent, 'meat' as a target, and 'eats' as a relation.",
                "new_nodes": [
                    {"label": "dog", "properties": {"is_alive": True}},
                    {"label": "meat", "properties": {"is_edible": True}}
                ],
                "new_edges": [
                    {"source": "dog", "target": "meat", "relation_type": "eats", "weight": 0.8}
                ],
                "nodes_to_remove": [],
                "edges_to_weaken": []
            }
        }
    )


class QueryResponse(BaseModel):
    """
    Schema for answering user questions.
    
    Chain-of-Thought: reasoning field is generated first.
    """
    
    reasoning: str = Field(
        ...,
        description="Step-by-step reasoning process to arrive at the answer"
    )
    answer: str = Field(
        ...,
        description="Direct, concise answer to the user's question"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0, 
        le=1.0,
        description="Confidence in this answer (0.0 = uncertain, 1.0 = certain)"
    )
    missing_knowledge: List[str] = Field(
        default_factory=list,
        description="Concepts the system doesn't know about but would need to answer fully"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Questions that could help clarify or extend the answer"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reasoning": "The user asks what dogs eat. From my knowledge, dogs are "
                             "carnivores that primarily eat meat, but can also eat some vegetables.",
                "answer": "Dogs primarily eat meat, but they can also eat some vegetables and grains.",
                "confidence": 0.85,
                "missing_knowledge": [],
                "follow_up_questions": ["What specific type of meat?", "Are there foods dogs should avoid?"]
            }
        }
    )


class DiagnosisResult(BaseModel):
    """Schema for System 2 diagnostic output."""
    
    diagnosis_type: str = Field(
        ...,
        description="Type of issue: 'constraint_violation', 'missing_knowledge', 'no_issue'"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the diagnosis"
    )
    violated_constraints: List[str] = Field(
        default_factory=list,
        description="List of constraints that were violated"
    )
    missing_properties: List[str] = Field(
        default_factory=list,
        description="Properties that need to be acquired"
    )
    suggested_action: str = Field(
        default="no_action",
        description="Suggested response: 'apply_ltd', 'query_user', 'inject_goal', 'no_action'"
    )
