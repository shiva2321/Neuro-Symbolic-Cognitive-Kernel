"""
Semantic System: Knowledge Representation and Querying
RDF triple store with natural language interface for semantic queries.

Components:
- SemanticBrain: RDF triple store with binary persistence
- ContextDriver: NLP parsing and semantic graph construction
- SemanticAssistant: Natural language query interface
"""

from .context_driver import SemanticBrain

__all__ = ['SemanticBrain']

