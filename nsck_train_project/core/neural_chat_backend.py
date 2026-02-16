#!/usr/bin/env python3
"""
Trainable backend chat model for NSCK dashboard.

This module extends the NSCK architecture (TextKnowledgeLearner, SemanticMemory, VSA)
to support:
- Learning from text using the existing TextKnowledgeLearner
- Learning from conversation history
- Learning from image-caption pairs (integrated into semantic memory)
- Natural language query with VSA-based semantic retrieval

Built on NSCK's sophisticated VSA architecture, not sklearn ML.
"""

from __future__ import annotations

import json
import os
import sys
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add nsck-demo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck-demo'))

from python.core.language.text_knowledge_learner import TextKnowledgeLearner, LearnedFact
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory
from python.core.language.lingua_cortex import get_lingua_cortex

# Import production-grade NLG engine
from production_nlg import ProductionNLG, ResponseStyle

# Import enhanced retrieval utilities
from enhanced_retrieval import (
    EnhancedConceptExtractor,
    QueryExpander,
    ImprovedFactRetrieval,
    ResponseQualityFilter,
)
from direct_concept_matcher import DirectConceptMatcher

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
except ImportError:
    from python.core.vsa.hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


@dataclass
class ConversationExample:
    """Conversation history entry for continual learning."""
    query: str
    response: str
    confidence: float
    evidence: List[str]
    timestamp: str
    source: str = "conversation"


@dataclass
class ImageCaptionExample:
    """Image-caption supervision for multimodal learning."""
    image_path: str
    caption: str
    source: str = "image"


class TrainableChatBackend:
    """
    End-to-end trainable backend using NSCK architecture.

    Pipeline:
    1. Use TextKnowledgeLearner to learn from text files.
    2. Query using VSA-based semantic memory retrieval.
    3. Add conversation history as episodic memories.
    4. Integrate image-caption data into semantic concepts.
    """

    def __init__(
        self,
        text_learner: Optional[TextKnowledgeLearner] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
    ):
        # Core NSCK components
        self.semantic = semantic_memory or SemanticMemory()
        self.episodic = episodic_memory or EpisodicMemory()
        self.text_learner = text_learner or TextKnowledgeLearner(
            semantic_memory=self.semantic,
            episodic_memory=self.episodic,
        )
        
        # Language cortex for encoding
        self.lingua = get_lingua_cortex()
        
        # Production-grade NLG engine
        self.nlg_engine = ProductionNLG(default_style=ResponseStyle.CONVERSATIONAL)
        
        # Enhanced retrieval components
        self.concept_extractor = EnhancedConceptExtractor()
        self.query_expander = QueryExpander()
        self.fact_retriever = ImprovedFactRetrieval()
        self.quality_filter = ResponseQualityFilter()
        
        # Patch text_learner's concept extraction to use enhanced extractor
        self._patch_concept_extraction()
        
        # Conversation history and image examples
        self.conversation_history: List[ConversationExample] = []
        self.image_examples: List[ImageCaptionExample] = []
        
        # Training metadata
        self.training_metadata: Dict[str, Any] = {
            "last_trained": None,
            "texts_learned": 0,
            "conversations_learned": 0,
            "images_learned": 0,
            "total_facts": 0,
        }

    def _patch_concept_extraction(self):
        """
        Monkey-patch TextKnowledgeLearner's _extract_concepts to use enhanced extraction.
        This improves concept quality during learning and querying.
        """
        # Save original method as instance variable for unpickling
        self._original_extract_concepts = self.text_learner._extract_concepts
        
        # Create bound method that we can pickle
        extractor = self.concept_extractor
        original = self._original_extract_concepts
        
        def enhanced_extract_concepts(sentence: str) -> List[str]:
            """Enhanced concept extraction using EnhancedConceptExtractor."""
            # Use our enhanced extractor
            concepts = extractor.extract_concepts(sentence)
            
            # Fall back to original if no concepts found
            if not concepts or len(concepts) == 0:
                concepts = original(sentence)
            
            return concepts
        
        # Replace the method
        self.text_learner._extract_concepts = enhanced_extract_concepts
        self._enhanced_extract_concepts = enhanced_extract_concepts
    
    def __getstate__(self):
        """Prepare object for pickling by restoring original methods."""
        state = self.__dict__.copy()
        # Restore original method before pickling
        if hasattr(self, '_original_extract_concepts'):
            self.text_learner._extract_concepts = self._original_extract_concepts
            # Remove unpicklable method
            state.pop('_enhanced_extract_concepts', None)
        return state
    
    def __setstate__(self, state):
        """Restore object after unpickling and re-patch methods."""
        self.__dict__.update(state)
        # Re-apply monkey patch after unpickling
        if hasattr(self, '_original_extract_concepts'):
            self._patch_concept_extraction()
        
    def learn_from_text_file(self, filepath: str) -> Dict[str, Any]:
        """Learn from a text file using TextKnowledgeLearner."""
        try:
            session = self.text_learner.learn_from_text_file(filepath)
            
            self.training_metadata["texts_learned"] += 1
            self.training_metadata["total_facts"] = len(self.text_learner.learned_facts)
            self.training_metadata["last_trained"] = datetime.now().isoformat()
            
            # Handle both dict and LearningSession returns
            if isinstance(session, dict):
                concepts_learned = session.get('concepts', 0)
                relations_learned = session.get('relations', 0)
                facts_stored = session.get('facts', 0)
            else:
                concepts_learned = getattr(session, 'concepts_learned', session.get('concepts', 0))
                relations_learned = getattr(session, 'relations_learned', session.get('relations', 0))
                facts_stored = getattr(session, 'facts_stored', session.get('facts', 0))
            
            return {
                "status": "success",
                "filepath": filepath,
                "concepts_learned": concepts_learned,
                "relations_learned": relations_learned,
                "facts_stored": facts_stored,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "filepath": filepath,
            }

    def learn_from_text(self, text: str, source: str = "text_input") -> Dict[str, Any]:
        """Learn from raw text using TextKnowledgeLearner."""
        try:
            session = self.text_learner.learn_from_text(text, source=source)
            
            self.training_metadata["texts_learned"] += 1
            self.training_metadata["total_facts"] = len(self.text_learner.learned_facts)
            self.training_metadata["last_trained"] = datetime.now().isoformat()
            
            # Handle both dict and LearningSession returns
            if isinstance(session, dict):
                concepts_learned = session.get('concepts', 0)
                relations_learned = session.get('relations', 0)
                facts_stored = session.get('facts', 0)
            else:
                concepts_learned = getattr(session, 'concepts_learned', session.get('concepts', 0))
                relations_learned = getattr(session, 'relations_learned', session.get('relations', 0))
                facts_stored = getattr(session, 'facts_stored', session.get('facts', 0))
            
            return {
                "status": "success",
                "concepts_learned": concepts_learned,
                "relations_learned": relations_learned,
                "facts_stored": facts_stored,
                "source": source,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "source": source,
            }

    def add_conversation_history(
        self,
        history: Sequence[Dict[str, Any]],
        min_confidence: float = 0.6,
        max_items: int = 200,
    ) -> int:
        """
        Learn from successful conversation history.
        
        High-confidence conversations are stored as episodic memories
        and used to improve future responses.
        """
        added = 0
        for item in list(history)[-max_items:]:
            query = str(item.get("query", "")).strip()
            response = str(item.get("response", "")).strip()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            
            if len(query) < 3 or len(response) < 3:
                continue
            if confidence < min_confidence:
                continue
            
            evidence = [str(e) for e in item.get("evidence", []) if isinstance(e, str)]
            timestamp = item.get("timestamp", datetime.now().isoformat())
            
            self.conversation_history.append(
                ConversationExample(
                    query=query,
                    response=response,
                    confidence=confidence,
                    evidence=evidence,
                    timestamp=timestamp,
                    source="conversation_history",
                )
            )
            
            # Also store as episodic memory
            try:
                query_hv = self._encode_text(query)
                self.episodic.store_episode(
                    concept_hv=query_hv,
                    task_tag="conversation",
                    context={"query": query, "response": response, "confidence": confidence},
                )
            except Exception:
                pass
            
            added += 1
        
        self.training_metadata["conversations_learned"] = len(self.conversation_history)
        return added

    def add_image_caption_examples(self, image_dir: str) -> int:
        """
        Add image supervision from a directory.
        
        Supported patterns:
        - image.jpg + image.txt (caption sidecar)
        - image.jpg + image.json with {"caption": "..."}
        
        Images are integrated as semantic concepts with captions as properties.
        """
        root = Path(image_dir)
        if not root.exists() or not root.is_dir():
            return 0
        
        image_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        added = 0
        
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() not in image_ext:
                continue
            
            caption = self._resolve_caption_for_image(image_path)
            if len(caption) < 3:
                continue
            
            self.image_examples.append(
                ImageCaptionExample(
                    image_path=str(image_path),
                    caption=caption,
                    source="image_caption",
                )
            )
            
            # Add image as semantic concept
            concept_name = f"image_{image_path.stem}"
            try:
                self.semantic.add_concept(
                    concept_name=concept_name,
                    properties={
                        "type": "image",
                        "path": str(image_path),
                        "caption": caption,
                    },
                )
                
                # Learn caption text to connect image concept with language
                self.learn_from_text(
                    f"The image {image_path.name} shows {caption}.",
                    source=f"image_caption:{image_path.name}",
                )
            except Exception:
                pass
            
            added += 1
        
        self.training_metadata["images_learned"] = len(self.image_examples)
        return added

    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer user query using NSCK architecture with enhanced retrieval.
        
        Uses TextKnowledgeLearner.query_learned_knowledge() which:
        - Encodes query to hypervector
        - Searches semantic memory with spreading activation
        - Recalls episodic memories
        - Returns related facts and reasoning trace
        
        Enhanced with:
        - Query expansion for better semantic matching
        - Improved fact retrieval and ranking
        - Response quality filtering
        """
        q = (user_query or "").strip()
        if not q:
            return {"status": "error", "error": "Empty query"}
        
        try:
            # Initialize variables
            expanded_terms = []
            
            # Step 1: Direct concept matching (text-based) - HIGH PRIORITY
            query_concepts = DirectConceptMatcher.extract_query_concepts(q)
            stored_concepts = list(self.semantic.concept_hvs.keys()) if hasattr(self.semantic, 'concept_hvs') else []
            
            # Find matching concepts using direct text matching
            direct_matches = []
            for qc in query_concepts:
                matches = DirectConceptMatcher.find_matching_concepts(qc, stored_concepts, min_similarity=0.6)
                direct_matches.extend(matches)
            
            # Deduplicate and rank matches
            unique_matches = {}
            for concept, score in direct_matches:
                if concept not in unique_matches or score > unique_matches[concept]:
                    unique_matches[concept] = score
            
            matched_concepts = sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)[:top_k]
            similar_concepts = [concept for concept, score in matched_concepts]
            
            # Step 2: Retrieve facts related to matched concepts
            related_facts = []
            if similar_concepts:
                for concept in similar_concepts:
                    concept_facts = [
                        fact for fact in self.text_learner.learned_facts
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
                related_facts = unique_facts[:top_k * 3]
                
                confidence = matched_concepts[0][1] if matched_concepts else 0.0
                activated_concepts = similar_concepts
            else:
                # Fallback: Use NSCK's hypervector-based query
                expanded_terms = self.query_expander.expand_query(q)
                expanded_query = q
                if expanded_terms and len(expanded_terms) > 1:
                    expanded_query = f"{q} {' '.join(expanded_terms[:3])}"
                
                result = self.text_learner.query_learned_knowledge(expanded_query, top_k=top_k * 2)
                similar_concepts = result.get("similar_concepts", [])
                related_facts = result.get("related_facts", [])
                activated_concepts = result.get("activated_concepts", [])
                confidence = result.get("confidence", 0.0)
            
            # Step 3: Re-rank facts using improved retrieval
            if related_facts:
                ranked_facts = self.fact_retriever.retrieve_and_rank_facts(
                    facts=related_facts,
                    query=q,
                    top_k=top_k,
                )
                # Extract just the facts (remove scores)
                related_facts = [fact for fact, score in ranked_facts]
            
            # Step 4: Build response using NSCK's knowledge
            response = self._synthesize_response(
                query=q,
                similar_concepts=similar_concepts,
                related_facts=related_facts,
                activated_concepts=activated_concepts,
            )
            
            # Step 5: Validate response quality
            quality_result = self.quality_filter.validate_response(response, related_facts)
            if not quality_result['is_quality']:
                # Try to improve low-quality response
                if related_facts:
                    # Regenerate with more emphasis on facts
                    response = self._synthesize_response_with_facts(
                        query=q,
                        related_facts=related_facts,
                        concepts=similar_concepts
                    )
                    quality_result = self.quality_filter.validate_response(response, related_facts)
            
            # Extract evidence and sources
            evidence = self._build_evidence(related_facts)
            sources = self._build_sources(similar_concepts, activated_concepts)
            
            # Determine intent from query
            intent = self._detect_intent_simple(q)
            
            # Add quality metadata
            metadata = {
                "query_expanded": len(expanded_terms) > 0,
                "expanded_terms": expanded_terms[:3],
                "facts_reranked": len(related_facts) > 0,
                "quality_score": quality_result.get('quality_score', 0.5),
                "is_quality_response": quality_result.get('is_quality', False),
            }
            
            return {
                "status": "success",
                "query": q,
                "response": response,
                "confidence": float(confidence),
                "intent": intent,
                "intent_confidence": 0.85,
                "sources": sources,
                "evidence": evidence,
                "nlg_type": "nsck_architecture_backend_enhanced",
                "facts_used": len(related_facts),
                "training": self.training_metadata,
                "enhancement_metadata": metadata,
            }
        except Exception as e:
            return {
                "status": "error",
                "query": q,
                "error": str(e),
                "response": f"Error processing query: {str(e)}",
            }

    def save(self, output_path: str) -> bool:
        """Persist trained backend model."""
        try:
            payload = {
                "semantic_memory": self.semantic,
                "episodic_memory": self.episodic,
                "text_learner": self.text_learner,
                "conversation_history": self.conversation_history,
                "image_examples": self.image_examples,
                "training_metadata": self.training_metadata,
            }
            with open(output_path, "wb") as f:
                pickle.dump(payload, f)
            return True
        except Exception:
            return False

    def load(self, input_path: str) -> bool:
        """Load persisted backend model."""
        path = Path(input_path)
        if not path.exists():
            return False
        
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            
            self.semantic = payload.get("semantic_memory", self.semantic)
            self.episodic = payload.get("episodic_memory", self.episodic)
            self.text_learner = payload.get("text_learner", self.text_learner)
            self.conversation_history = payload.get("conversation_history", [])
            self.image_examples = payload.get("image_examples", [])
            self.training_metadata = payload.get("training_metadata", {})
            
            # Reconnect text_learner to loaded memories
            if self.text_learner:
                self.text_learner.semantic = self.semantic
                self.text_learner.episodic = self.episodic
            
            return True
        except Exception:
            return False

    def export_training_summary(self) -> Dict[str, Any]:
        """Get current backend status and learned coverage."""
        return {
            "training": self.training_metadata,
            "total_facts": len(self.text_learner.learned_facts),
            "total_concepts": len(self.semantic.concept_hvs),
            "conversation_examples": len(self.conversation_history),
            "image_examples": len(self.image_examples),
            "semantic_relations": self.semantic.concept_graph.number_of_edges() if hasattr(self.semantic, 'concept_graph') else 0,
        }

    def _encode_text(self, text: str) -> hypervec_rs.HyperVector:
        """Encode text to hypervector using LinguaCortex."""
        # Use the text_learner's internal encoding method
        return self.text_learner._encode_sentence(text)

    def _synthesize_response(
        self,
        query: str,
        similar_concepts: List,
        related_facts: List,
        activated_concepts: List,
    ) -> str:
        """Synthesize natural language response using production NLG engine."""
        
        # Convert to format expected by ProductionNLG
        # similar_concepts: list of tuples (name, similarity) or strings
        concept_tuples = []
        for item in similar_concepts:
            if isinstance(item, tuple) and len(item) >= 2:
                concept_tuples.append((str(item[0]), float(item[1])))
            elif isinstance(item, tuple):
                concept_tuples.append((str(item[0]), 0.7))
            else:
                concept_tuples.append((str(item), 0.7))
        
        # Convert facts to dicts with subject, relation, object
        fact_dicts = []
        for fact in related_facts:
            if hasattr(fact, 'subject') and hasattr(fact, 'relation') and hasattr(fact, 'object'):
                fact_dicts.append({
                    'subject': str(fact.subject),
                    'relation': str(fact.relation),
                    'object': str(fact.object),
                    'confidence': getattr(fact, 'confidence', 0.8)
                })
        
        # Convert activated concepts to strings
        activated_names = []
        for item in activated_concepts:
            if isinstance(item, tuple):
                activated_names.append(str(item[0]))
            else:
                activated_names.append(str(item))
        
        # Generate response using production NLG
        nlg_result = self.nlg_engine.generate_response(
            query=query,
            retrieved_concepts=concept_tuples,
            retrieved_facts=fact_dicts,
            activated_concepts=activated_names,
            style=ResponseStyle.CONVERSATIONAL,
        )
        
        return nlg_result['response']

    def _synthesize_response_with_facts(
        self,
        query: str,
        related_facts: List,
        concepts: List,
    ) -> str:
        """
        Synthesize response with emphasis on facts when initial response is low quality.
        """
        # Convert facts to dicts
        fact_dicts = []
        for fact in related_facts[:5]:  # Use top 5 facts
            if hasattr(fact, 'subject') and hasattr(fact, 'relation') and hasattr(fact, 'object'):
                fact_dicts.append({
                    'subject': str(fact.subject),
                    'relation': str(fact.relation),
                    'object': str(fact.object),
                    'confidence': getattr(fact, 'confidence', 0.8)
                })
        
        if not fact_dicts:
            # Fallback to basic response if no facts
            return "Based on the available information, I can provide some insights on this topic."
        
        # Build fact-heavy response
        response_parts = []
        
        # Add main facts as sentences
        for i, fact in enumerate(fact_dicts[:3]):
            subj = fact['subject']
            rel = fact['relation']
            obj = fact['object']
            
            # Format as natural sentence
            if rel.lower() in ['is', 'are', 'was', 'were']:
                sentence = f"{subj} {rel} {obj}."
            elif rel.lower() in ['has', 'have', 'contains', 'includes']:
                sentence = f"{subj} {rel} {obj}."
            elif rel.lower() == 'defined_as':
                sentence = f"{subj} is defined as {obj}."
            elif rel.lower() == 'means':
                sentence = f"{subj} means {obj}."
            else:
                sentence = f"{subj} {rel} {obj}."
            
            response_parts.append(sentence)
        
        # Add additional facts if available
        if len(fact_dicts) > 3:
            extra_info = []
            for fact in fact_dicts[3:5]:
                extra_info.append(f"{fact['subject']} {fact['relation']} {fact['object']}")
            if extra_info:
                response_parts.append(f"Additionally, {', and '.join(extra_info)}.")
        
        return " ".join(response_parts)

    def _build_evidence(self, related_facts: List) -> List[str]:
        """Extract evidence strings from facts."""
        evidence = []
        for fact in related_facts[:6]:
            if hasattr(fact, 'subject') and hasattr(fact, 'relation') and hasattr(fact, 'object'):
                evidence.append(f"{fact.subject} {fact.relation} {fact.object}")
        return evidence

    def _build_sources(self, similar_concepts: List, activated_concepts: List) -> List[str]:
        """Extract source concept names."""
        sources = []
        for item in similar_concepts[:5]:
            if isinstance(item, tuple):
                sources.append(str(item[0]))
            else:
                sources.append(str(item))
        
        for item in activated_concepts[:3]:
            if isinstance(item, tuple):
                name = str(item[0])
                if name not in sources:
                    sources.append(name)
        
        return sources[:8]

    def _detect_intent_simple(self, query: str) -> str:
        """Simple intent detection from query keywords."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "define", "who is", "what does"]):
            return "definition"
        elif any(word in query_lower for word in ["how does", "how is", "process", "explain how"]):
            return "process"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "causality"
        elif any(word in query_lower for word in ["properties", "characteristics", "features"]):
            return "property"
        elif any(word in query_lower for word in ["relationship", "relate", "connect"]):
            return "relationship"
        else:
            return "generic"

    def _resolve_caption_for_image(self, image_path: Path) -> str:
        """Resolve caption from image sidecar files."""
        # Try .txt sidecar
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                return txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        
        # Try .json sidecar
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    caption = str(data.get("caption", "")).strip()
                    if caption:
                        return caption
            except Exception:
                pass
        
        # Fallback to filename
        stem = image_path.stem.replace("_", " ").replace("-", " ").strip()
        return stem
