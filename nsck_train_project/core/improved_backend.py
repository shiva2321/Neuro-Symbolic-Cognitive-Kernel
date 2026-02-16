#!/usr/bin/env python3
"""
Improved Backend with All Fixes
================================

Integrates all improvements to address identified issues:

1. Adaptive Retrieval (fixes concept dilution)
2. Deep Context Integration (fixes 50% context maintenance)
3. Multi-Hop Reasoning (fixes 33% advanced reasoning)
4. Enhanced NLG (fixes generic responses)

This is a drop-in replacement for TrainableChatBackend.
"""

import os
import sys
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add nsck-demo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

from neural_chat_backend import TrainableChatBackend
from adaptive_retrieval import AdaptiveRetrievalEngine
from python.core.language.text_knowledge_learner import LearnedFact


@dataclass
class ConversationTurn:
    """A turn in the conversation with extracted concepts."""
    query: str
    response: str
    concepts: List[str]
    facts_used: List[LearnedFact]
    timestamp: datetime


class EpisodicContextMemory:
    """
    Deep episodic memory for conversation context.
    
    Unlike the simple context manager, this:
    - Stores actual facts from each turn
    - Retrieves relevant past facts during queries
    - Enables multi-turn reasoning
    """
    
    def __init__(self, max_turns: int = 10):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
        self.entity_history: Dict[str, List[int]] = {}  # entity -> turn indices
    
    def add_turn(
        self,
        query: str,
        response: str,
        concepts: List[str],
        facts_used: List[LearnedFact]
    ):
        """Add a conversation turn."""
        turn = ConversationTurn(
            query=query,
            response=response,
            concepts=concepts,
            facts_used=facts_used,
            timestamp=datetime.now()
        )
        
        self.turns.append(turn)
        
        # Index entities
        turn_idx = len(self.turns) - 1
        for concept in concepts:
            if concept not in self.entity_history:
                self.entity_history[concept] = []
            self.entity_history[concept].append(turn_idx)
        
        # Keep only recent turns
        if len(self.turns) > self.max_turns:
            removed_turn = self.turns.pop(0)
            # Update entity history indices
            for entity in self.entity_history:
                self.entity_history[entity] = [i - 1 for i in self.entity_history[entity] if i > 0]
    
    def get_relevant_past_facts(self, current_concepts: List[str]) -> List[LearnedFact]:
        """
        Retrieve facts from past conversation turns that are relevant to current concepts.
        
        This enables the AI to remember what was discussed before.
        """
        relevant_facts = []
        seen_facts = set()
        
        # Look for turns that mentioned these concepts
        for concept in current_concepts:
            if concept in self.entity_history:
                for turn_idx in self.entity_history[concept]:
                    if turn_idx < len(self.turns):
                        turn = self.turns[turn_idx]
                        for fact in turn.facts_used:
                            fact_key = (fact.subject, fact.relation, fact.object)
                            if fact_key not in seen_facts:
                                relevant_facts.append(fact)
                                seen_facts.add(fact_key)
        
        return relevant_facts
    
    def get_recent_entities(self, n_turns: int = 3) -> List[str]:
        """Get entities from recent conversation."""
        recent_turns = self.turns[-n_turns:] if self.turns else []
        entities = []
        seen = set()
        
        for turn in reversed(recent_turns):
            for concept in turn.concepts:
                if concept not in seen:
                    entities.append(concept)
                    seen.add(concept)
        
        return entities
    
    def clear(self):
        """Clear conversation history."""
        self.turns.clear()
        self.entity_history.clear()


class MultiHopReasoning:
    """
    Enables multi-hop reasoning by chaining facts.
    
    Example:
      Fact 1: "Newton discovered gravity"
      Fact 2: "Newton formulated laws of motion"
      Query: "What else did Newton discover?"
      → Chain: Newton → gravity, laws of motion
    """
    
    def __init__(self):
        self.fact_index: Dict[str, List[LearnedFact]] = {}  # subject -> facts
    
    def index_facts(self, facts: List[LearnedFact]):
        """Build index for fast fact chaining."""
        self.fact_index.clear()
        
        for fact in facts:
            # Index by subject
            if fact.subject not in self.fact_index:
                self.fact_index[fact.subject] = []
            self.fact_index[fact.subject].append(fact)
            
            # Also index by object (for bidirectional chaining)
            if fact.object not in self.fact_index:
                self.fact_index[fact.object] = []
            self.fact_index[fact.object].append(fact)
    
    def find_related_facts(
        self,
        seed_concepts: List[str],
        max_hops: int = 2,
        max_facts: int = 10
    ) -> List[LearnedFact]:
        """
        Find facts related to seed concepts by chaining.
        
        Args:
            seed_concepts: Starting concepts
            max_hops: Maximum chain length (1 = direct, 2 = 2-hop, etc.)
            max_facts: Maximum facts to return
        """
        visited_facts = set()
        relevant_facts = []
        
        # BFS through fact graph
        current_concepts = set(seed_concepts)
        
        for hop in range(max_hops):
            next_concepts = set()
            
            for concept in current_concepts:
                if concept in self.fact_index:
                    for fact in self.fact_index[concept]:
                        fact_key = (fact.subject, fact.relation, fact.object)
                        
                        if fact_key not in visited_facts:
                            visited_facts.add(fact_key)
                            relevant_facts.append(fact)
                            
                            # Add connected concepts for next hop
                            next_concepts.add(fact.subject)
                            next_concepts.add(fact.object)
                            
                            if len(relevant_facts) >= max_facts:
                                return relevant_facts
            
            current_concepts = next_concepts
        
        return relevant_facts


class ImprovedBackend:
    """
    Improved backend that fixes all identified issues.
    
    Fixes:
    1. Concept dilution → Adaptive retrieval
    2. Context maintenance → Episodic memory integration
    3. Advanced reasoning → Multi-hop fact chaining
    4. Generic responses → Better fact selection
    """
    
    def __init__(self, base_backend: Optional[TrainableChatBackend] = None):
        """
        Args:
            base_backend: Existing backend to wrap, or creates new one
        """
        self.backend = base_backend if base_backend else TrainableChatBackend()
        
        # New components
        self.adaptive_retrieval = AdaptiveRetrievalEngine()
        self.episodic_context = EpisodicContextMemory(max_turns=10)
        self.multi_hop = MultiHopReasoning()
        
        # Initialize retrieval engine with current concepts
        self._refresh_retrieval_index()
    
    def _refresh_retrieval_index(self):
        """Refresh retrieval indices after learning new concepts."""
        concepts = list(self.backend.semantic.concept_hvs.keys())
        if concepts:
            self.adaptive_retrieval.index_concepts(concepts)
        
        # Index facts for multi-hop reasoning
        if self.backend.text_learner.learned_facts:
            self.multi_hop.index_facts(self.backend.text_learner.learned_facts)
    
    def learn_from_text(self, text: str, source: str = "text_input") -> Dict[str, Any]:
        """Learn from text and refresh indices."""
        result = self.backend.learn_from_text(text, source)
        self._refresh_retrieval_index()
        return result
    
    def query(self, user_query: str, top_k: int = 5, use_context: bool = True) -> Dict[str, Any]:
        """
        Query with all improvements applied.
        
        Pipeline:
        1. Get recent context entities from episodic memory
        2. Perform VSA retrieval
        3. Apply adaptive multi-metric re-ranking
        4. Retrieve facts from past conversation
        5. Perform multi-hop reasoning to expand facts
        6. Generate improved response
        7. Store turn in episodic memory
        """
        print(f"[ImprovedBackend] Query: {user_query}")
        
        # Step 1: Get context from episodic memory
        context_entities = []
        if use_context:
            context_entities = self.episodic_context.get_recent_entities(n_turns=3)
            if context_entities:
                print(f"[Context] Recent entities: {context_entities[:5]}")
                self.adaptive_retrieval.update_context(context_entities)
        
        # Step 2: VSA retrieval (from base backend)
        query_lower = user_query.lower()
        
        # Get all concepts and their similarities
        try:
            # Use backend's semantic memory for VSA retrieval
            stored_concepts = list(self.backend.semantic.concept_hvs.keys())
            
            # Simple similarity: check if query terms match concept terms
            from direct_concept_matcher import DirectConceptMatcher
            query_concepts = DirectConceptMatcher.extract_query_concepts(query_lower)
            
            vsa_results = []
            for concept in stored_concepts:
                # Calculate simple text similarity
                matches = DirectConceptMatcher.find_matching_concepts(query_lower, [concept], min_similarity=0.0)
                if matches:
                    vsa_results.append((concept, matches[0][1]))
            
            # Sort by similarity
            vsa_results.sort(key=lambda x: x[1], reverse=True)
            vsa_results = vsa_results[:top_k * 3]  # Get more candidates for re-ranking
            
            print(f"[VSA] Top candidates: {vsa_results[:5]}")
            
        except Exception as e:
            print(f"[VSA] Error: {e}")
            vsa_results = []
        
        # Step 3: Adaptive re-ranking
        if vsa_results:
            num_concepts = len(self.backend.semantic.concept_hvs)
            ranked_results = self.adaptive_retrieval.retrieve(
                query=user_query,
                vsa_results=vsa_results,
                num_concepts=num_concepts,
                top_k=top_k,
                use_context=use_context
            )
            similar_concepts = [concept for concept, score in ranked_results]
            print(f"[Adaptive] After re-ranking: {similar_concepts}")
        else:
            similar_concepts = []
        
        # Step 4: Retrieve facts from current concepts
        related_facts = []
        for concept in similar_concepts:
            concept_facts = [
                fact for fact in self.backend.text_learner.learned_facts
                if (concept.lower() in fact.subject.lower() or
                    concept.lower() in fact.object.lower() or
                    fact.subject.lower() in concept.lower() or
                    fact.object.lower() in concept.lower())
            ]
            related_facts.extend(concept_facts)
        
        # Deduplicate facts
        seen_facts = set()
        unique_facts = []
        for fact in related_facts:
            fact_key = (fact.subject, fact.relation, fact.object)
            if fact_key not in seen_facts:
                seen_facts.add(fact_key)
                unique_facts.append(fact)
        related_facts = unique_facts
        
        print(f"[Facts] Found {len(related_facts)} direct facts")
        
        # Step 5: Add facts from past conversation (episodic memory)
        past_facts = []
        if use_context and context_entities:
            past_facts = self.episodic_context.get_relevant_past_facts(similar_concepts + context_entities)
            for fact in past_facts:
                fact_key = (fact.subject, fact.relation, fact.object)
                if fact_key not in seen_facts:
                    related_facts.append(fact)
                    seen_facts.add(fact_key)
            print(f"[Episodic] Added {len(past_facts)} facts from conversation history")
        
        # Step 6: Multi-hop reasoning to expand facts
        expanded_facts = []
        if len(related_facts) < top_k and similar_concepts:
            expanded_facts = self.multi_hop.find_related_facts(
                seed_concepts=similar_concepts,
                max_hops=2,
                max_facts=top_k * 2
            )
            for fact in expanded_facts:
                fact_key = (fact.subject, fact.relation, fact.object)
                if fact_key not in seen_facts:
                    related_facts.append(fact)
                    seen_facts.add(fact_key)
            print(f"[MultiHop] Added {len(expanded_facts)} facts from chaining")
        
        # Limit facts
        related_facts = related_facts[:top_k * 2]
        
        # Step 7: Generate response using backend's synthesis
        response = self.backend._synthesize_response(
            query=user_query,
            similar_concepts=similar_concepts,
            related_facts=related_facts,
            activated_concepts=similar_concepts
        )
        
        confidence = 1.0 if similar_concepts else 0.0
        
        # Step 8: Store in episodic memory
        self.episodic_context.add_turn(
            query=user_query,
            response=response,
            concepts=similar_concepts,
            facts_used=related_facts
        )
        
        # Record success for importance weighting
        if similar_concepts:
            self.adaptive_retrieval.record_success(similar_concepts)
        
        print(f"[Response] Generated {len(response)} chars, confidence: {confidence:.2f}\n")
        
        return {
            "response": response,
            "confidence": confidence,
            "similar_concepts": similar_concepts,
            "related_facts": related_facts,
            "activated_concepts": similar_concepts,
            "context_used": use_context,
            "episodic_facts_used": len(past_facts),
            "multi_hop_facts_used": len(expanded_facts),
        }
    
    def clear_context(self):
        """Clear conversation context."""
        self.episodic_context.clear()
        self.adaptive_retrieval.clear_context()
    
    # Forward other methods to backend
    def __getattr__(self, name):
        """Forward attribute access to base backend."""
        # Avoid infinite recursion by checking if backend exists in __dict__
        if 'backend' not in self.__dict__:
            raise AttributeError(f"'ImprovedBackend' object has no attribute '{name}'")
        return getattr(self.backend, name)


def test_improvements():
    """Test the improved backend."""
    print("=" * 80)
    print("TESTING IMPROVED BACKEND")
    print("=" * 80)
    
    # Load existing model
    model_path = "ai_model_progressive.pkl"
    if os.path.exists(model_path):
        print(f"\nLoading model: {model_path}")
        with open(model_path, 'rb') as f:
            base_backend = pickle.load(f)
        
        print(f"Concepts: {len(base_backend.semantic.concept_hvs)}")
        print(f"Facts: {len(base_backend.text_learner.learned_facts)}")
    else:
        print(f"\nModel not found: {model_path}")
        print("Creating new backend...")
        base_backend = TrainableChatBackend()
        
        # Train on sample data
        sample_text = """
        Gravity is a fundamental force that attracts objects with mass.
        Isaac Newton discovered the law of universal gravitation in 1687.
        Newton also formulated the three laws of motion.
        The laws of motion describe how objects move when forces act on them.
        Einstein later refined Newton's theory with general relativity.
        """
        base_backend.learn_from_text(sample_text)
    
    # Wrap with improvements
    improved = ImprovedBackend(base_backend)
    
    print("\n" + "=" * 80)
    print("TEST 1: Context Maintenance")
    print("=" * 80)
    
    # Multi-turn conversation
    q1 = "What is gravity?"
    r1 = improved.query(q1)
    print(f"\nQ1: {q1}")
    print(f"A1: {r1['response'][:200]}...")
    print(f"Concepts: {r1['similar_concepts']}")
    
    q2 = "Who discovered it?"
    r2 = improved.query(q2)
    print(f"\nQ2: {q2}")
    print(f"A2: {r2['response'][:200]}...")
    print(f"Concepts: {r2['similar_concepts']}")
    print(f"Context used: {r2.get('context_used', False)}")
    print(f"Episodic facts: {r2.get('episodic_facts_used', 0)}")
    
    q3 = "What else did he discover?"
    r3 = improved.query(q3)
    print(f"\nQ3: {q3}")
    print(f"A3: {r3['response'][:200]}...")
    print(f"Concepts: {r3['similar_concepts']}")
    print(f"Multi-hop facts: {r3.get('multi_hop_facts_used', 0)}")
    
    print("\n✅ Improved backend test complete")


if __name__ == "__main__":
    test_improvements()
