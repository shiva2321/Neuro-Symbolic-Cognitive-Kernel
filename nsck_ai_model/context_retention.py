#!/usr/bin/env python3
"""
Context Retention Module for NSCK AI Model
===========================================

Addresses the context retention limitation (25% score in testing).
Implements conversation context tracking and attention-like mechanisms.

Key improvements:
1. Conversation history tracking
2. Attention-like mechanism for relevant context
3. Context-aware response generation
4. Entity and pronoun resolution
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import re

# Ensure nsck-demo is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

import python.core.vsa.hypervec_shim as hypervec_rs


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation."""
    turn_id: int
    user_input: str
    response: str
    concepts: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # entity -> type
    timestamp: float = 0.0
    hypervector: Optional[Any] = None


class ContextRetentionModule:
    """
    Manages conversation context for improved multi-turn interactions.
    """
    
    def __init__(self, max_history: int = 20, attention_window: int = 5):
        """
        Initialize context retention module.
        
        Args:
            max_history: Maximum conversation turns to retain
            attention_window: Number of recent turns to pay attention to
        """
        self.max_history = max_history
        self.attention_window = attention_window
        
        # Conversation history
        self.conversation_history: deque = deque(maxlen=max_history)
        self.turn_counter = 0
        
        # Entity tracking
        self.entities: Dict[str, str] = {}  # entity name -> type/value
        self.last_mentioned_entities: List[str] = []
        
        # Context hypervector (aggregated context)
        self.context_hv: Optional[Any] = None
    
    def add_turn(
        self,
        user_input: str,
        response: str,
        concepts: List[str],
        query_hv: Optional[Any] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn.
        
        Args:
            user_input: User's input
            response: System's response
            concepts: Extracted concepts
            query_hv: Hypervector representation of query
            
        Returns:
            ConversationTurn object
        """
        self.turn_counter += 1
        
        # Extract entities from this turn
        turn_entities = self._extract_entities(user_input, response)
        self.entities.update(turn_entities)
        
        # Create turn object
        turn = ConversationTurn(
            turn_id=self.turn_counter,
            user_input=user_input,
            response=response,
            concepts=concepts,
            entities=turn_entities,
            hypervector=query_hv
        )
        
        # Add to history
        self.conversation_history.append(turn)
        
        # Update context hypervector
        if query_hv is not None:
            self._update_context_hv(query_hv)
        
        # Update last mentioned entities
        self.last_mentioned_entities = list(turn_entities.keys())[-5:]
        
        return turn
    
    def get_relevant_context(
        self,
        current_query: str,
        current_concepts: List[str],
        current_hv: Optional[Any] = None
    ) -> List[ConversationTurn]:
        """
        Get relevant context from conversation history.
        
        Args:
            current_query: Current user query
            current_concepts: Concepts in current query
            current_hv: Hypervector of current query
            
        Returns:
            List of relevant conversation turns
        """
        if not self.conversation_history:
            return []
        
        # Get recent turns (attention window)
        recent_turns = list(self.conversation_history)[-self.attention_window:]
        
        # Score turns by relevance
        scored_turns: List[Tuple[ConversationTurn, float]] = []
        
        current_concepts_set = set(c.lower() for c in current_concepts)
        
        for turn in recent_turns:
            score = 0.0
            
            # Recency boost
            position = len(list(self.conversation_history)) - list(self.conversation_history).index(turn)
            recency_score = 1.0 / position if position > 0 else 1.0
            score += recency_score * 0.3
            
            # Concept overlap
            turn_concepts_set = set(c.lower() for c in turn.concepts)
            overlap = len(current_concepts_set & turn_concepts_set)
            if turn_concepts_set:
                concept_score = overlap / len(turn_concepts_set)
                score += concept_score * 0.5
            
            # Entity mention
            if any(entity.lower() in current_query.lower() for entity in turn.entities.keys()):
                score += 0.2
            
            # Hypervector similarity (if available)
            if current_hv is not None and turn.hypervector is not None:
                try:
                    sim = current_hv.similarity(turn.hypervector)
                    score += sim * 0.3
                except:
                    pass
            
            scored_turns.append((turn, score))
        
        # Sort by score and return top relevant turns
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 relevant turns
        return [turn for turn, score in scored_turns[:3]]
    
    def resolve_references(self, text: str) -> str:
        """
        Resolve pronouns and references using conversation context.
        
        Args:
            text: Text with potential references
            
        Returns:
            Text with resolved references
        """
        if not self.last_mentioned_entities:
            return text
        
        # Simple pronoun resolution
        resolved = text
        
        # Get last mentioned entity
        last_entity = self.last_mentioned_entities[-1] if self.last_mentioned_entities else None
        
        if last_entity:
            # Replace pronouns with last entity (simple heuristic)
            patterns = [
                (r'\bit\b', last_entity),
                (r'\bthis\b', last_entity),
                (r'\bthat\b', last_entity),
            ]
            
            for pattern, replacement in patterns:
                # Only replace if it makes sense (not at start of sentence)
                resolved = re.sub(
                    r'(?<!^)' + pattern,
                    replacement,
                    resolved,
                    flags=re.IGNORECASE
                )
        
        return resolved
    
    def get_context_summary(self) -> str:
        """
        Get a summary of current conversation context.
        
        Returns:
            Context summary string
        """
        if not self.conversation_history:
            return "No conversation history."
        
        # Get recent topics
        recent_concepts = []
        for turn in list(self.conversation_history)[-3:]:
            recent_concepts.extend(turn.concepts)
        
        # Count and get top concepts
        from collections import Counter
        concept_counts = Counter(recent_concepts)
        top_concepts = [c for c, _ in concept_counts.most_common(5)]
        
        # Get active entities
        active_entities = list(self.entities.keys())[-5:]
        
        summary_parts = []
        
        if top_concepts:
            summary_parts.append(f"Topics: {', '.join(top_concepts)}")
        
        if active_entities:
            summary_parts.append(f"Entities: {', '.join(active_entities)}")
        
        summary_parts.append(f"Turns: {len(self.conversation_history)}")
        
        return " | ".join(summary_parts)
    
    def enhance_query_with_context(
        self,
        query: str,
        concepts: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Enhance query with relevant context.
        
        Args:
            query: Original query
            concepts: Extracted concepts
            
        Returns:
            Tuple of (enhanced_query, enhanced_concepts)
        """
        # Get relevant context
        relevant_turns = self.get_relevant_context(query, concepts)
        
        if not relevant_turns:
            return query, concepts
        
        # Add context concepts
        enhanced_concepts = list(concepts)
        
        for turn in relevant_turns:
            for concept in turn.concepts:
                if concept not in enhanced_concepts:
                    enhanced_concepts.append(concept)
        
        # Resolve references in query
        enhanced_query = self.resolve_references(query)
        
        return enhanced_query, enhanced_concepts[:10]  # Limit to top 10
    
    def should_use_context(self, query: str) -> bool:
        """
        Determine if context should be used for this query.
        
        Args:
            query: User query
            
        Returns:
            True if context should be used
        """
        # Use context if query has pronouns or references
        reference_words = ['it', 'this', 'that', 'these', 'those', 'they', 'them',
                          'its', 'their', 'what about', 'how about', 'tell me more']
        
        query_lower = query.lower()
        
        for word in reference_words:
            if word in query_lower:
                return True
        
        # Use context if query is short and vague
        if len(query.split()) <= 3:
            return True
        
        return False
    
    def clear_context(self):
        """Clear conversation context."""
        self.conversation_history.clear()
        self.entities.clear()
        self.last_mentioned_entities.clear()
        self.context_hv = None
        self.turn_counter = 0
    
    def _update_context_hv(self, query_hv: Any):
        """Update the aggregated context hypervector."""
        if self.context_hv is None:
            self.context_hv = query_hv
        else:
            try:
                # Bundle with previous context (moving average)
                self.context_hv = self.context_hv.bundle(query_hv)
            except:
                # If bundle fails, just use new one
                self.context_hv = query_hv
    
    def _extract_entities(self, user_input: str, response: str) -> Dict[str, str]:
        """
        Extract entities from conversation turn.
        
        Args:
            user_input: User input
            response: System response
            
        Returns:
            Dictionary of entities
        """
        entities = {}
        
        # Combine text
        combined = f"{user_input} {response}"
        
        # Extract capitalized words (simple NER)
        words = combined.split()
        for i, word in enumerate(words):
            # Check if capitalized (and not sentence start)
            if i > 0 and word and len(word) > 1 and word[0].isupper():
                # Clean punctuation
                clean_word = re.sub(r'[^\w]', '', word)
                if len(clean_word) > 1:
                    entities[clean_word] = 'ENTITY'
        
        # Extract numbers as entities
        numbers = re.findall(r'\b\d+\b', combined)
        for num in numbers:
            entities[num] = 'NUMBER'
        
        # Extract quoted text as entities
        quoted = re.findall(r'"([^"]+)"', combined)
        for quote in quoted:
            if len(quote) > 2:
                entities[quote] = 'QUOTED'
        
        return entities
    
    def get_conversation_context_for_query(self, query: str) -> str:
        """
        Get formatted conversation context relevant to query.
        
        Args:
            query: Current query
            
        Returns:
            Formatted context string
        """
        if not self.conversation_history:
            return ""
        
        # Get last 2-3 turns as context
        recent_turns = list(self.conversation_history)[-3:]
        
        context_lines = []
        for turn in recent_turns:
            context_lines.append(f"User: {turn.user_input}")
            context_lines.append(f"AI: {turn.response}")
        
        return "\n".join(context_lines)
