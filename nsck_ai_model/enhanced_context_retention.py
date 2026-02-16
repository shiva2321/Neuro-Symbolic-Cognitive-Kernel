#!/usr/bin/env python3
"""
Enhanced Context Retention Module for NSCK AI Model
===================================================

Advanced context retention to achieve 80%+ accuracy target.

Key enhancements over base version:
1. Weighted context scoring (recency + relevance + entity continuity)
2. Multi-hop entity tracking (entity chains and relationships)
3. Improved coreference resolution
4. Context summarization for long conversations
5. Semantic clustering of related turns
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque, defaultdict
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
    entity_mentions: Dict[str, int] = field(default_factory=dict)  # entity -> mention count
    timestamp: float = 0.0
    hypervector: Optional[Any] = None
    importance_score: float = 1.0  # How important is this turn


@dataclass
class EntityChain:
    """Tracks entity mentions across turns."""
    entity: str
    entity_type: str
    first_mention_turn: int
    last_mention_turn: int
    mention_count: int
    related_entities: Set[str] = field(default_factory=set)
    properties: Dict[str, str] = field(default_factory=dict)  # entity properties learned


class EnhancedContextRetentionModule:
    """
    Advanced context retention with enhanced tracking and resolution.
    Target: 80%+ context retention accuracy.
    """
    
    def __init__(self, max_history: int = 30, attention_window: int = 10):
        """
        Initialize enhanced context retention module.
        
        Args:
            max_history: Maximum conversation turns to retain (increased from 20)
            attention_window: Number of recent turns to pay attention to (increased from 5)
        """
        self.max_history = max_history
        self.attention_window = attention_window
        
        # Conversation history
        self.conversation_history: deque = deque(maxlen=max_history)
        self.turn_counter = 0
        
        # Enhanced entity tracking
        self.entities: Dict[str, EntityChain] = {}  # entity name -> EntityChain
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)  # entity -> related entities
        self.last_mentioned_entities: List[str] = []
        
        # Context hypervector (aggregated context)
        self.context_hv: Optional[Any] = None
        
        # Topic tracking
        self.current_topics: List[str] = []
        self.topic_transitions: List[Tuple[int, str]] = []  # (turn_id, topic)
        
        # Question tracking (for better context retention)
        self.pending_questions: Dict[int, str] = {}  # turn_id -> question
        self.answered_questions: Set[int] = set()
    
    def add_turn(
        self,
        user_input: str,
        response: str,
        concepts: List[str],
        query_hv: Optional[Any] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn with enhanced tracking.
        
        Args:
            user_input: User's input
            response: System's response
            concepts: Extracted concepts
            query_hv: Hypervector representation of query
            
        Returns:
            ConversationTurn object
        """
        self.turn_counter += 1
        
        # Extract entities and count mentions
        turn_entities, entity_mentions = self._extract_entities_enhanced(user_input, response)
        
        # Update entity chains
        self._update_entity_chains(self.turn_counter, turn_entities, entity_mentions)
        
        # Calculate importance score for this turn
        importance = self._calculate_turn_importance(user_input, response, concepts)
        
        # Create turn object
        turn = ConversationTurn(
            turn_id=self.turn_counter,
            user_input=user_input,
            response=response,
            concepts=concepts,
            entities=turn_entities,
            entity_mentions=entity_mentions,
            hypervector=query_hv,
            importance_score=importance
        )
        
        # Add to history
        self.conversation_history.append(turn)
        
        # Update context hypervector
        if query_hv is not None:
            self._update_context_hv(query_hv, importance)
        
        # Update last mentioned entities (with more context)
        self.last_mentioned_entities = self._get_recent_entity_mentions()
        
        # Track if this is a question
        if user_input.strip().endswith('?'):
            self.pending_questions[self.turn_counter] = user_input
        
        # Update topics
        self._update_topics(concepts)
        
        return turn
    
    def get_relevant_context(
        self,
        current_query: str,
        current_concepts: List[str],
        current_hv: Optional[Any] = None
    ) -> List[ConversationTurn]:
        """
        Get relevant context with enhanced scoring.
        
        Args:
            current_query: Current user query
            current_concepts: Concepts in current query
            current_hv: Hypervector of current query
            
        Returns:
            List of relevant conversation turns (sorted by relevance)
        """
        if not self.conversation_history:
            return []
        
        # Get extended attention window
        recent_turns = list(self.conversation_history)[-self.attention_window:]
        
        # Score turns by multiple factors
        scored_turns: List[Tuple[ConversationTurn, float, Dict[str, float]]] = []
        
        current_concepts_set = set(c.lower() for c in current_concepts)
        
        for turn in recent_turns:
            scores = self._score_turn_relevance(
                turn, current_query, current_concepts_set, current_hv
            )
            
            # Weighted total score
            total_score = (
                scores['recency'] * 0.25 +
                scores['concept_overlap'] * 0.25 +
                scores['entity_continuity'] * 0.20 +
                scores['semantic_similarity'] * 0.20 +
                scores['importance'] * 0.10
            )
            
            scored_turns.append((turn, total_score, scores))
        
        # Sort by total score
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 relevant turns (increased from 3)
        return [turn for turn, score, _ in scored_turns[:5]]
    
    def _score_turn_relevance(
        self,
        turn: ConversationTurn,
        current_query: str,
        current_concepts: Set[str],
        current_hv: Optional[Any]
    ) -> Dict[str, float]:
        """
        Score a turn's relevance using multiple factors.
        
        Returns:
            Dictionary of individual scores
        """
        scores = {}
        
        # 1. Recency score (exponential decay)
        position = len(list(self.conversation_history)) - list(self.conversation_history).index(turn)
        scores['recency'] = 0.9 ** (position - 1) if position > 0 else 1.0
        
        # 2. Concept overlap score
        turn_concepts = set(c.lower() for c in turn.concepts)
        if turn_concepts and current_concepts:
            overlap = len(current_concepts & turn_concepts)
            scores['concept_overlap'] = overlap / len(current_concepts | turn_concepts)
        else:
            scores['concept_overlap'] = 0.0
        
        # 3. Entity continuity score
        scores['entity_continuity'] = self._calculate_entity_continuity(turn, current_query)
        
        # 4. Semantic similarity (hypervector)
        if current_hv is not None and turn.hypervector is not None:
            try:
                scores['semantic_similarity'] = max(0, current_hv.similarity(turn.hypervector))
            except:
                scores['semantic_similarity'] = 0.0
        else:
            scores['semantic_similarity'] = 0.0
        
        # 5. Turn importance
        scores['importance'] = turn.importance_score
        
        return scores
    
    def _calculate_entity_continuity(self, turn: ConversationTurn, current_query: str) -> float:
        """
        Calculate entity continuity score.
        
        Checks if entities in turn are mentioned in current query or recently.
        """
        score = 0.0
        current_query_lower = current_query.lower()
        
        for entity in turn.entities.keys():
            entity_lower = entity.lower()
            
            # Direct mention in current query
            if entity_lower in current_query_lower:
                score += 0.5
            
            # Recent mention
            elif entity in self.last_mentioned_entities:
                idx = self.last_mentioned_entities.index(entity)
                recency = 1.0 - (idx / len(self.last_mentioned_entities))
                score += 0.3 * recency
            
            # Part of entity chain
            if entity in self.entities:
                chain = self.entities[entity]
                for related in chain.related_entities:
                    if related.lower() in current_query_lower:
                        score += 0.2
                        break
        
        return min(1.0, score)
    
    def resolve_references(self, text: str) -> str:
        """
        Enhanced reference resolution using entity chains.
        
        Args:
            text: Text with potential references
            
        Returns:
            Text with resolved references
        """
        if not self.last_mentioned_entities:
            return text
        
        resolved = text
        
        # Get most recent and relevant entity
        target_entity = self._get_most_relevant_entity(text)
        
        if target_entity:
            # Enhanced pronoun patterns
            pronoun_patterns = [
                (r'\bit\b', target_entity),
                (r'\bthis\b', target_entity),
                (r'\bthat\b', target_entity),
                (r'\bthey\b', target_entity),
                (r'\bthem\b', target_entity),
                (r'\bhe\b', target_entity),
                (r'\bshe\b', target_entity),
            ]
            
            for pattern, replacement in pronoun_patterns:
                # Only replace if not at sentence start and makes sense
                if re.search(pattern, resolved, re.IGNORECASE):
                    # Context-aware replacement
                    resolved = re.sub(
                        r'(?<!^)(?<!\. )' + pattern,
                        replacement,
                        resolved,
                        flags=re.IGNORECASE
                    )
        
        return resolved
    
    def enhance_query_with_context(
        self,
        query: str,
        concepts: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Enhanced query enrichment with context.
        
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
        
        # Add context concepts with relevance weighting
        enhanced_concepts = list(concepts)
        concept_weights = {}
        
        for turn in relevant_turns:
            turn_weight = turn.importance_score
            for concept in turn.concepts:
                if concept not in enhanced_concepts:
                    if concept not in concept_weights:
                        concept_weights[concept] = 0.0
                    concept_weights[concept] += turn_weight
        
        # Add top weighted concepts
        for concept, weight in sorted(concept_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            enhanced_concepts.append(concept)
        
        # Resolve references in query
        enhanced_query = self.resolve_references(query)
        
        # If query references previous context, add explicit connection
        if self.should_use_context(query):
            # Find most relevant previous turn
            if relevant_turns:
                prev_turn = relevant_turns[0]
                # Add implicit context (concepts only, not text)
                for entity in prev_turn.entities.keys():
                    if entity.lower() not in enhanced_query.lower():
                        # Entity might be implied
                        if entity not in enhanced_concepts:
                            enhanced_concepts.append(entity)
        
        return enhanced_query, enhanced_concepts[:15]  # Limit concepts
    
    def should_use_context(self, query: str) -> bool:
        """
        Enhanced decision on whether to use context.
        
        Args:
            query: User query
            
        Returns:
            True if context should be used
        """
        query_lower = query.lower()
        
        # Reference words
        reference_words = [
            'it', 'this', 'that', 'these', 'those', 'they', 'them',
            'its', 'their', 'what about', 'how about', 'tell me more',
            'also', 'and', 'too', 'as well'
        ]
        
        if any(word in query_lower for word in reference_words):
            return True
        
        # Short and vague queries
        if len(query.split()) <= 4:
            return True
        
        # Follow-up question indicators
        if query_lower.startswith(('and', 'but', 'so', 'then', 'also')):
            return True
        
        # Questions about previously mentioned entities
        for entity in self.last_mentioned_entities[:3]:
            if entity.lower() in query_lower:
                return True
        
        return False
    
    def _calculate_turn_importance(
        self,
        user_input: str,
        response: str,
        concepts: List[str]
    ) -> float:
        """
        Calculate importance score for a turn.
        
        Important turns get higher weight in context retrieval.
        """
        importance = 0.5  # Base importance
        
        # Questions are important
        if user_input.strip().endswith('?'):
            importance += 0.2
        
        # Turns with many concepts are important
        if len(concepts) > 3:
            importance += 0.1
        
        # Long responses suggest important information
        if len(response) > 100:
            importance += 0.1
        
        # Turns with entities are important
        entities, _ = self._extract_entities_enhanced(user_input, response)
        if len(entities) > 0:
            importance += 0.1 * min(len(entities), 3)
        
        return min(1.0, importance)
    
    def _update_entity_chains(
        self,
        turn_id: int,
        entities: Dict[str, str],
        mentions: Dict[str, int]
    ):
        """
        Update entity chains for tracking across turns.
        """
        for entity, entity_type in entities.items():
            if entity in self.entities:
                # Update existing chain
                chain = self.entities[entity]
                chain.last_mention_turn = turn_id
                chain.mention_count += mentions.get(entity, 1)
            else:
                # Create new chain
                chain = EntityChain(
                    entity=entity,
                    entity_type=entity_type,
                    first_mention_turn=turn_id,
                    last_mention_turn=turn_id,
                    mention_count=mentions.get(entity, 1)
                )
                self.entities[entity] = chain
            
            # Track entity relationships
            for other_entity in entities.keys():
                if other_entity != entity:
                    chain.related_entities.add(other_entity)
                    self.entity_graph[entity].add(other_entity)
    
    def _get_recent_entity_mentions(self) -> List[str]:
        """
        Get recently mentioned entities sorted by recency and frequency.
        """
        recent_entities = []
        entity_scores = {}
        
        # Get entities from recent turns
        for i, turn in enumerate(reversed(list(self.conversation_history)[-10:])):
            for entity, count in turn.entity_mentions.items():
                if entity not in entity_scores:
                    entity_scores[entity] = 0.0
                # Score based on recency and mention count
                recency_weight = 0.9 ** i
                entity_scores[entity] += recency_weight * count
        
        # Sort by score
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        return [entity for entity, score in sorted_entities[:10]]
    
    def _get_most_relevant_entity(self, text: str) -> Optional[str]:
        """
        Get the most relevant entity for reference resolution.
        """
        text_lower = text.lower()
        
        # Check recent entities in order
        for entity in self.last_mentioned_entities[:5]:
            # If entity is mentioned, use related entities
            if entity.lower() in text_lower:
                return entity
        
        # Use most recent entity
        return self.last_mentioned_entities[0] if self.last_mentioned_entities else None
    
    def _extract_entities_enhanced(
        self,
        user_input: str,
        response: str
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Enhanced entity extraction with mention counting.
        
        Returns:
            Tuple of (entities dict, mention count dict)
        """
        entities = {}
        mentions = defaultdict(int)
        
        # Combine text
        combined = f"{user_input} {response}"
        
        # Extract capitalized words (proper nouns)
        words = combined.split()
        for i, word in enumerate(words):
            if i > 0 and word and len(word) > 1 and word[0].isupper():
                clean_word = re.sub(r'[^\w]', '', word)
                if len(clean_word) > 1:
                    entities[clean_word] = 'ENTITY'
                    mentions[clean_word] += 1
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', combined)
        for num in numbers:
            entities[num] = 'NUMBER'
            mentions[num] += 1
        
        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', combined)
        for quote in quoted:
            if len(quote) > 2:
                entities[quote] = 'QUOTED'
                mentions[quote] += 1
        
        return entities, dict(mentions)
    
    def _update_context_hv(self, query_hv: Any, importance: float = 1.0):
        """
        Update aggregated context hypervector with importance weighting.
        """
        if self.context_hv is None:
            self.context_hv = query_hv
        else:
            try:
                # Weighted bundle based on importance
                if importance > 0.7:
                    # Important turns get more weight
                    self.context_hv = self.context_hv.bundle(query_hv)
                    self.context_hv = self.context_hv.bundle(query_hv)
                else:
                    self.context_hv = self.context_hv.bundle(query_hv)
            except:
                self.context_hv = query_hv
    
    def _update_topics(self, concepts: List[str]):
        """
        Track conversation topics for context.
        """
        # Update current topics (keep last 10 concepts)
        self.current_topics.extend(concepts)
        self.current_topics = self.current_topics[-10:]
        
        # Detect topic transitions
        if len(concepts) > 2:
            # Simple topic tracking
            topic = "_".join(sorted(concepts[:3]))
            self.topic_transitions.append((self.turn_counter, topic))
            self.topic_transitions = self.topic_transitions[-5:]
    
    def get_context_summary(self) -> str:
        """
        Get enhanced summary of conversation context.
        
        Returns:
            Context summary string
        """
        if not self.conversation_history:
            return "No conversation history."
        
        # Get key entities
        top_entities = self.last_mentioned_entities[:5]
        
        # Get recent topics
        from collections import Counter
        concept_counts = Counter(self.current_topics)
        top_concepts = [c for c, _ in concept_counts.most_common(5)]
        
        # Get conversation length and important turns
        total_turns = len(self.conversation_history)
        important_turns = [t for t in self.conversation_history if t.importance_score > 0.7]
        
        summary_parts = []
        
        if top_concepts:
            summary_parts.append(f"Topics: {', '.join(top_concepts)}")
        
        if top_entities:
            summary_parts.append(f"Entities: {', '.join(top_entities)}")
        
        summary_parts.append(f"Turns: {total_turns} ({len(important_turns)} important)")
        
        if self.pending_questions:
            summary_parts.append(f"Pending questions: {len(self.pending_questions)}")
        
        return " | ".join(summary_parts)
    
    def clear_context(self):
        """Clear conversation context."""
        self.conversation_history.clear()
        self.entities.clear()
        self.entity_graph.clear()
        self.last_mentioned_entities.clear()
        self.context_hv = None
        self.turn_counter = 0
        self.current_topics.clear()
        self.topic_transitions.clear()
        self.pending_questions.clear()
        self.answered_questions.clear()
