#!/usr/bin/env python3
"""
Interactive Chat with Trained NSCK Model
==========================================
Test the trained system's capabilities:
- Query learned facts with natural language synthesis
- See memory recalls with context awareness
- View reasoning traces and confidence scores
- Monitor internal states (emotions, confidence, memory stats)
- Watch decision-making process with conversation history
- Learn from user feedback

Run: python chat_with_trained_model.py
"""

import sys
import os
import pickle
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.reasoning.context_engine import ContextEngine


class ModelIntrospector:
    """Provides visibility into model's internal states with enhanced reasoning."""
    
    def __init__(
        self,
        semantic_memory: SemanticMemory,
        episodic_memory: EpisodicMemory,
        text_learner: TextKnowledgeLearner
    ):
        self.semantic = semantic_memory
        self.episodic = episodic_memory
        self.text_learner = text_learner
        self.conversation_history = deque(maxlen=10)  # Keep last 10 exchanges
        self.query_feedback = {}  # Track user feedback on responses
        self.min_confidence = 0.5  # Filter out low-confidence results
    
    def query_learned_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query what the model learned and explain reasoning with enhanced synthesis."""
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")
        
        # Track in conversation history
        self.conversation_history.append({"query": query, "timestamp": datetime.now().isoformat()})
        
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "query_intent": self._detect_intent(query),
            "memory_stats": self._get_memory_stats(),
            "learned_facts": self._get_learned_facts(),
            "query_results": {},
            "synthesized_answer": None
        }
        
        # Try to query learned knowledge
        try:
            if hasattr(self.text_learner, 'query_learned_knowledge'):
                query_result = self.text_learner.query_learned_knowledge(query, top_k=top_k)
                result["query_results"] = query_result
                
                # Filter and rank results
                filtered_results = self._filter_and_rank_results(query_result, query)
                
                # Synthesize natural language answer
                synthesized = self._synthesize_answer(query, filtered_results, query_result)
                result["synthesized_answer"] = synthesized
                
                print("[QUERY RESULTS]")
                print(f"  answer: {synthesized['summary']}")
                
                if synthesized['confidence'] > 0:
                    print(f"  confidence: {synthesized['confidence']:.2%}")
                    
                if synthesized['sources']:
                    print(f"\n  top_sources:")
                    for i, source in enumerate(synthesized['sources'][:3], 1):
                        print(f"    {i}. {source}")
                
                if hasattr(query_result, '__dict__'):
                    for key, value in query_result.__dict__.items():
                        if key not in ['answer'] and value:
                            print(f"  {key}: {value}")
                elif isinstance(query_result, dict):
                    for key, value in query_result.items():
                        if key not in ['answer'] and value:
                            print(f"  {key}: {value}")
            else:
                print("[!] Query method not available in text_learner")
        
        except Exception as e:
            print(f"[!] Query failed: {e}")
            result["query_error"] = str(e)
        
        return result
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        query_lower = query.lower()
        
        # Question intents
        if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'why', 'how']):
            if 'is' in query_lower or 'are' in query_lower:
                return "definition_query"
            elif 'many' in query_lower or 'count' in query_lower or 'number' in query_lower:
                return "count_query"
            else:
                return "factual_query"
        
        # Assertion intents
        elif any(word in query_lower for word in ['is', 'are', 'be']):
            return "assertion"
        
        # Command intents
        elif any(word in query_lower for word in ['list', 'show', 'tell', 'find']):
            return "command_query"
        
        # Default
        else:
            return "general_query"
    
    def _filter_and_rank_results(self, query_result: Any, query: str) -> Dict[str, Any]:
        """Filter and rank results by confidence and relevance."""
        filtered = {}
        
        if hasattr(query_result, '__dict__'):
            filtered = query_result.__dict__.copy()
        elif isinstance(query_result, dict):
            filtered = query_result.copy()
        else:
            return {"raw_result": query_result}
        
        # Filter by confidence threshold
        if 'confidence' in filtered:
            confidence = filtered.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                filtered['meets_confidence'] = confidence >= self.min_confidence
        
        # Calculate relevance score
        if 'similar_concepts' in filtered and filtered['similar_concepts']:
            relevance_scores = []
            for concept, score in filtered['similar_concepts'][:3]:
                relevance_scores.append(score)
            
            if relevance_scores:
                filtered['avg_relevance'] = sum(relevance_scores) / len(relevance_scores)
        
        return filtered
    
    def _synthesize_answer(self, query: str, filtered_results: Dict, raw_results: Any) -> Dict[str, Any]:
        """Synthesize a natural language answer from query results."""
        synthesis = {
            "summary": "",
            "confidence": 0.0,
            "sources": [],
            "evidence_count": 0
        }
        
        try:
            # Extract components from results
            confidence = 0.0
            similar_concepts = []
            related_facts = []
            activated_concepts = []
            
            if hasattr(raw_results, '__dict__'):
                obj_dict = raw_results.__dict__
                confidence = obj_dict.get('confidence', 0.0)
                similar_concepts = obj_dict.get('similar_concepts', [])
                related_facts = obj_dict.get('related_facts', [])
                activated_concepts = obj_dict.get('activated_concepts', [])
            elif isinstance(raw_results, dict):
                confidence = raw_results.get('confidence', 0.0)
                similar_concepts = raw_results.get('similar_concepts', [])
                related_facts = raw_results.get('related_facts', [])
                activated_concepts = raw_results.get('activated_concepts', [])
            
            synthesis['confidence'] = confidence
            
            # Build answer parts
            answer_parts = []
            
            # Add concept-based answer
            if similar_concepts and confidence >= self.min_confidence:
                top_concepts = [c[0] for c in similar_concepts[:3]]
                answer_parts.append(f"Based on learned concepts, this relates to: {', '.join(top_concepts)}")
                synthesis['sources'].extend(top_concepts)
            
            # Add fact-based answer
            if related_facts:
                high_conf_facts = [f for f in related_facts if f.get('confidence', 0) >= 0.8][:3]
                if high_conf_facts:
                    for fact in high_conf_facts:
                        subj = fact.get('subject', '?')
                        rel = fact.get('relation', '?')
                        obj = fact.get('object', '?')
                        answer_parts.append(f"{subj} {rel} {obj}")
                        synthesis['sources'].append(f"{subj} {rel} {obj}")
                    synthesis['evidence_count'] = len(high_conf_facts)
            
            # Add activation-based answer
            if activated_concepts:
                top_activated = sorted(activated_concepts, key=lambda x: x[1] if isinstance(x, tuple) else 0, reverse=True)[:2]
                if top_activated:
                    concepts_text = ', '.join([c[0] if isinstance(c, tuple) else str(c) for c in top_activated])
                    answer_parts.append(f"Related activated concepts: {concepts_text}")
            
            # Combine answer
            if answer_parts:
                synthesis['summary'] = " | ".join(answer_parts)
            else:
                # Fallback answer
                synthesis['summary'] = f"Query '{query}' processed with {len(related_facts)} supporting facts"
                synthesis['confidence'] = 0.3
            
            return synthesis
            
        except Exception as e:
            synthesis['summary'] = f"Unable to synthesize answer: {str(e)}"
            synthesis['confidence'] = 0.0
            return synthesis
    
    def provide_feedback(self, query: str, rating: int, notes: str = ""):
        """Record user feedback on query response."""
        if query not in self.query_feedback:
            self.query_feedback[query] = []
        
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "rating": rating,  # 1-5 scale
            "notes": notes
        }
        self.query_feedback[query].append(feedback)
        print(f"[✓] Feedback recorded (rating: {rating}/5)")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        return {
            "total_queries": len(self.conversation_history),
            "avg_feedback_rating": sum(
                sum(f['rating'] for f in self.query_feedback.get(h['query'], []))
                for h in self.conversation_history
            ) / max(len(self.conversation_history), 1),
            "recent_queries": list(self.conversation_history),
            "feedback_summary": {
                q: {
                    "count": len(f),
                    "avg_rating": sum(x['rating'] for x in f) / len(f)
                }
                for q, f in self.query_feedback.items()
            }
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            "episodic": {
                "total_episodes": len(self.episodic.recent_episodes) if hasattr(self.episodic, 'recent_episodes') else 0,
            },
            "semantic": {
                "concepts": len(self.semantic.concept_map) if hasattr(self.semantic, 'concept_map') else 0,
                "relations": len(self.semantic.relations) if hasattr(self.semantic, 'relations') else 0,
            },
            "text_learner": {
                "facts_learned": len(self.text_learner.learned_facts) if hasattr(self.text_learner, 'learned_facts') else 0,
                "learning_sessions": len(self.text_learner.learning_sessions) if hasattr(self.text_learner, 'learning_sessions') else 0,
            }
        }
        
        print("[MEMORY STATISTICS]")
        print(f"  Episodic Memory: {stats['episodic']['total_episodes']} episodes")
        print(f"  Semantic Memory: {stats['semantic']['concepts']} concepts, {stats['semantic']['relations']} relations")
        print(f"  Learned Facts: {stats['text_learner']['facts_learned']}")
        print(f"  Learning Sessions: {stats['text_learner']['learning_sessions']}")
        
        return stats
    
    def _get_learned_facts(self) -> list:
        """Get all learned facts."""
        facts = []
        
        if hasattr(self.text_learner, 'learned_facts'):
            for fact in self.text_learner.learned_facts[:10]:  # Show first 10
                fact_dict = {
                    "subject": fact.subject if hasattr(fact, 'subject') else str(fact),
                    "relation": fact.relation if hasattr(fact, 'relation') else "?",
                    "object": fact.object if hasattr(fact, 'object') else "?",
                    "confidence": fact.confidence if hasattr(fact, 'confidence') else 0.5,
                }
                facts.append(fact_dict)
        
        if facts:
            print("[LEARNED FACTS (Sample)]")
            for i, fact in enumerate(facts, 1):
                print(f"  {i}. {fact['subject']} -{fact['relation']}-> {fact['object']} ({fact['confidence']:.2f})")
        
        return facts
    
    def show_internal_state(self) -> Dict[str, Any]:
        """Display internal cognitive state."""
        return {
            "memory_stats": self._get_memory_stats(),
            "learned_facts": self._get_learned_facts(),
            "timestamp": datetime.now().isoformat(),
        }


def load_trained_model(model_path: str = "models/trained_system.pkl") -> Optional[Dict[str, Any]]:
    """Load previously trained model state."""
    full_path = Path(__file__).parent / model_path
    
    if not full_path.exists():
        print(f"[!] Model not found at {full_path}")
        print(f"[!] Run: python train_multimodal.py first")
        return None
    
    try:
        print(f"[Loading] {full_path}...")
        with open(full_path, 'rb') as f:
            state = pickle.load(f)
        print(f"[✓] Model loaded successfully")
        return state
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        return None


def interactive_chat():
    """Run interactive chat session with enhanced features."""
    print("\n" + "="*60)
    print("NSCK TRAINED MODEL - INTERACTIVE CHAT (ENHANCED)")
    print("="*60 + "\n")
    
    # Load model
    state = load_trained_model()
    if not state:
        print("[!] Creating minimal system for testing...")
        semantic_memory = SemanticMemory()
        episodic_memory = EpisodicMemory()
        text_learner = TextKnowledgeLearner(
            semantic_memory=semantic_memory,
            episodic_memory=episodic_memory
        )
    else:
        semantic_memory = state.get("semantic_memory")
        episodic_memory = state.get("episodic_memory")
        text_learner = state.get("text_learner")
    
    introspector = ModelIntrospector(semantic_memory, episodic_memory, text_learner)
    
    print("Commands:")
    print("  'stats'      - Show memory statistics")
    print("  'facts'      - Show learned facts")
    print("  'state'      - Show internal cognitive state")
    print("  'learn <file>'   - Learn from text file")
    print("  'rate <1-5>'     - Rate the last response (1=bad, 5=excellent)")
    print("  'history'    - Show conversation history")
    print("  'summary'    - Show conversation summary")
    print("  'query <text>'   - Query learned knowledge")
    print("  'quit'       - Exit")
    print("\n")
    
    last_query_result = None
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\n[Goodbye!]")
                break
            
            elif user_input.lower() == "stats":
                introspector._get_memory_stats()
            
            elif user_input.lower() == "facts":
                introspector._get_learned_facts()
            
            elif user_input.lower() == "state":
                state = introspector.show_internal_state()
                print("\n[INTERNAL COGNITIVE STATE]")
                print(json.dumps(state, indent=2, default=str))
            
            elif user_input.lower() == "history":
                print("\n[CONVERSATION HISTORY]")
                for i, exchange in enumerate(introspector.conversation_history, 1):
                    print(f"  {i}. {exchange['query']}")
            
            elif user_input.lower() == "summary":
                summary = introspector.get_conversation_summary()
                print("\n[CONVERSATION SUMMARY]")
                print(json.dumps(summary, indent=2, default=str))
            
            elif user_input.lower().startswith("rate "):
                try:
                    rating = int(user_input[5:].strip())
                    if 1 <= rating <= 5:
                        query_text = introspector.conversation_history[-1]['query'] if introspector.conversation_history else "unknown"
                        introspector.provide_feedback(query_text, rating)
                    else:
                        print("[!] Rating must be between 1 and 5")
                except (ValueError, IndexError):
                    print("[!] Usage: rate <1-5>")
            
            elif user_input.lower().startswith("learn "):
                filepath = user_input[6:].strip()
                if os.path.exists(filepath):
                    print(f"\n[Learning from: {filepath}]")
                    session = text_learner.learn_from_text_file(filepath)
                    print(f"[✓] Learned {session.concepts_learned} concepts")
                else:
                    print(f"[!] File not found: {filepath}")
            
            elif user_input.lower().startswith("query "):
                query_text = user_input[6:].strip()
                result = introspector.query_learned_knowledge(query_text)
                last_query_result = result
            
            else:
                # Treat as free-form query
                result = introspector.query_learned_knowledge(user_input)
                last_query_result = result
        
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
            break
        except Exception as e:
            print(f"[Error] {e}")


def test_scenarios():
    """Run predefined test scenarios with enhanced analysis."""
    print("\n" + "="*60)
    print("TESTING LEARNED SYSTEM CAPABILITIES (ENHANCED)")
    print("="*60 + "\n")
    
    state = load_trained_model()
    if not state:
        print("[!] Model not found. Cannot run tests.")
        return
    
    introspector = ModelIntrospector(
        state["semantic_memory"],
        state["episodic_memory"],
        state["text_learner"]
    )
    
    # Test 1: Memory statistics
    print("[TEST 1] System Initialization")
    print("-" * 40)
    introspector.show_internal_state()
    
    # Test 2: Query system with intent detection
    print("\n[TEST 2] Knowledge Queries with Intent Detection")
    print("-" * 40)
    
    test_queries = [
        "What types of animals appear in the data?",
        "What datasets were used?",
        "Tell me about concepts learned",
        "How many facts were learned?",
        "When was this trained?",
    ]
    
    results = []
    for query in test_queries:
        result = introspector.query_learned_knowledge(query, top_k=3)
        results.append(result)
        print(f"  Intent: {result.get('query_intent', 'unknown')}")
        print(f"  Synthesized: {result.get('synthesized_answer', {}).get('summary', 'N/A')[:100]}...")
        print()
    
    # Test 3: Conversation analysis
    print("\n[TEST 3] Conversation Analysis")
    print("-" * 40)
    summary = introspector.get_conversation_summary()
    print(f"  Total Queries: {summary['total_queries']}")
    print(f"  Average Feedback: {summary['avg_feedback_rating']:.2f}/5.0")
    
    # Test 4: Result filtering effectiveness
    print("\n[TEST 4] Result Quality Analysis")
    print("-" * 40)
    high_confidence = sum(1 for r in results if r.get('synthesized_answer', {}).get('confidence', 0) >= 0.6)
    evidence_based = sum(1 for r in results if r.get('synthesized_answer', {}).get('evidence_count', 0) > 0)
    print(f"  High Confidence Results: {high_confidence}/{len(results)}")
    print(f"  Evidence-Based Results: {evidence_based}/{len(results)}")
    
    print("\n[TEST COMPLETE]")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Chat with trained NSCK model")
    parser.add_argument(
        "--mode",
        choices=["interactive", "test"],
        default="interactive",
        help="Chat mode or test mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trained_system.pkl",
        help="Path to trained model"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_scenarios()
    else:
        interactive_chat()


if __name__ == "__main__":
    main()
