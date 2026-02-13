"""
NSCK Text Knowledge Learner
============================
Learns from text files using the NSCK cognitive architecture.

CRITICAL: This module does NOT use an LLM for learning/reasoning.
The LLM (if present) is ONLY for natural language <-> VSA translation.

Core Learning Process:
1. Parse text into sentences and extract concepts/relations
2. Encode concepts into hypervectors using LinguaCortex (VSA)
3. Store concepts in SemanticMemory (graph structure + HVs)
4. Store experiences in EpisodicMemory for retrieval
5. Build causal and contextual relationships
6. Enable semantic search and reasoning over learned knowledge

Architecture:
- LinguaCortex: Text -> Semantic Fingerprints (SDR/VSA)
- SemanticMemory: Concept graph + Hypervector index
- EpisodicMemory: Experience storage with similarity search
- ContextEngine: Contextual disambiguation
- CausalReasoner: Causal inference over learned facts
"""

import re
import time
import hashlib
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()

from lingua_cortex import get_lingua_cortex, SemanticFingerprint
from semantic_memory import SemanticMemory
from episodic_memory import EpisodicMemory, LiveEpisode
from context_engine import ContextEngine
from causal_reasoning import CausalGraph, CausalReasoner


@dataclass
class LearnedFact:
    """A fact extracted from text and stored in memory."""
    subject: str
    relation: str
    object: str
    source_text: str
    confidence: float
    timestamp: float


@dataclass
class LearningSession:
    """Metadata about a learning session."""
    session_id: str
    filename: str
    start_time: float
    end_time: float
    sentences_processed: int
    concepts_learned: int
    relations_learned: int
    facts_stored: int


class TextKnowledgeLearner:
    """
    Learns from text files using VSA and symbolic reasoning.
    
    NO LLM-BASED LEARNING. Uses:
    - Semantic Folding (LinguaCortex) for text encoding
    - Hypervector representations for concepts
    - Graph-based semantic memory
    - Episodic memory for experiences
    - Rule-based relation extraction
    """
    
    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        context_engine: Optional[ContextEngine] = None
    ):
        # Core cognitive modules
        self.semantic = semantic_memory or SemanticMemory()
        self.episodic = episodic_memory or EpisodicMemory()
        self.context = context_engine or ContextEngine(self.semantic)
        
        # Language cortex for text encoding (NOT an LLM!)
        self.lingua = get_lingua_cortex()
        
        # Causal reasoning
        self.causal_graph = CausalGraph()
        self.causal_reasoner = CausalReasoner(self.causal_graph)
        
        # Learning state
        self.learned_facts: List[LearnedFact] = []
        self.learning_sessions: List[LearningSession] = []
        self.concept_frequencies: Counter = Counter()
        self.relation_patterns: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Semantic folding state for relation discovery
        self.concept_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self.concept_context_hvs: Dict[str, hypervec_rs.HyperVector] = {}  # Accumulated context
        self.folding_window_size = 7  # Words before/after for context
        self.relation_threshold = 0.65  # Similarity threshold for implicit relations
        self.min_cooccurrence = 3  # Minimum co-occurrences to consider relation
        
        print("[TextLearner] Initialized with VSA-based cognitive architecture + semantic folding")
    
    def learn_from_text_file(self, filepath: str, max_sentences: Optional[int] = None) -> LearningSession:
        """
        Learn from a text file by encoding knowledge into the cognitive system.
        
        Args:
            filepath: Path to text file to learn from
            
        Returns:
            LearningSession metadata about what was learned
        """
        session_id = hashlib.md5(f"{filepath}{time.time()}".encode()).hexdigest()[:8]
        start_time = time.time()
        
        print(f"\n[TextLearner] Starting learning session {session_id}")
        print(f"[TextLearner] Reading file: {filepath}")
        
        # Read text file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Learn from text
        stats = self._learn_from_text(
            text,
            source=filepath,
            session_id=session_id,
            max_sentences=max_sentences,
        )
        
        # Create session record
        session = LearningSession(
            session_id=session_id,
            filename=filepath,
            start_time=start_time,
            end_time=time.time(),
            sentences_processed=stats['sentences'],
            concepts_learned=stats['concepts'],
            relations_learned=stats['relations'],
            facts_stored=stats['facts']
        )
        
        self.learning_sessions.append(session)
        
        duration = session.end_time - session.start_time
        print(f"[TextLearner] Session {session_id} complete in {duration:.2f}s")
        print(f"[TextLearner]   Sentences: {session.sentences_processed}")
        print(f"[TextLearner]   Concepts: {session.concepts_learned}")
        print(f"[TextLearner]   Relations: {session.relations_learned}")
        print(f"[TextLearner]   Facts: {session.facts_stored}")
        
        return session
    
    def _learn_from_text(
        self,
        text: str,
        source: str,
        session_id: str,
        max_sentences: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Core learning loop: extract and encode knowledge from text.
        
        Process:
        1. Split into sentences
        2. Extract concepts (nouns, named entities)
        3. Extract relations (verb phrases, patterns)
        4. Encode as hypervectors
        5. Store in semantic memory
        6. Record in episodic memory
        """
        stats = {'sentences': 0, 'concepts': 0, 'relations': 0, 'facts': 0}
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if max_sentences is not None and max_sentences > 0:
            sentences = sentences[:max_sentences]
        stats['sentences'] = len(sentences)
        
        print(f"[TextLearner] Processing {len(sentences)} sentences...")
        
        for sent_idx, sentence in enumerate(sentences):
            # Clean sentence
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            
            # Learn from sentence
            sent_stats = self._learn_from_sentence(sentence, source, session_id, sent_idx)
            stats['concepts'] += sent_stats['concepts']
            stats['relations'] += sent_stats['relations']
            stats['facts'] += sent_stats['facts']
        
        return stats
    
    def _learn_from_sentence(
        self,
        sentence: str,
        source: str,
        session_id: str,
        sent_idx: int
    ) -> Dict[str, int]:
        """
        Learn from a single sentence.
        
        Steps:
        1. Encode sentence using LinguaCortex (semantic folding)
        2. Extract concepts (nouns, entities)
        3. Extract relations (patterns, verb phrases)
        4. Store in semantic and episodic memory
        """
        stats = {'concepts': 0, 'relations': 0, 'facts': 0}
        
        # Encode sentence to hypervector using LinguaCortex
        sentence_hv = self._encode_sentence(sentence)
        
        # Extract concepts (simple noun extraction + named entities)
        concepts = self._extract_concepts(sentence)
        
        # Store concepts in semantic memory
        for concept in concepts:
            if concept not in self.semantic.concept_hvs:
                # Create hypervector for concept
                concept_hv = hypervec_rs.HyperVector(hash(concept) % (2**32))
                
                # Add to semantic memory
                self.semantic.add_concept(
                    concept_name=concept,
                    properties={'source': source, 'session': session_id},
                    hv_override=concept_hv
                )
                
                stats['concepts'] += 1
                self.concept_frequencies[concept] += 1
        
        # Extract relations between concepts
        relations = self._extract_relations(sentence, concepts)
        
        # Store relations in semantic memory and causal graph
        for subj, rel, obj in relations:
            # Ensure concepts exist before adding relation
            for concept in (subj, obj):
                if concept not in self.semantic.concept_hvs:
                    concept_hv = hypervec_rs.HyperVector(hash(concept) % (2**32))
                    self.semantic.add_concept(
                        concept_name=concept,
                        properties={'source': source, 'session': session_id},
                        hv_override=concept_hv
                    )

            # Add relation to semantic memory
            self.semantic.add_relation(subj, rel, obj)
            
            # If causal relation, add to causal graph
            if rel in ['causes', 'results_in', 'leads_to', 'produces']:
                try:
                    self.causal_graph.add_causes(
                        cause=subj,
                        effect=obj,
                        mechanism=f"learned from text: {sentence[:50]}",
                        confidence=0.7
                    )
                except Exception as e:
                    # If causal graph fails, just skip it
                    pass
            
            # Store fact
            fact = LearnedFact(
                subject=subj,
                relation=rel,
                object=obj,
                source_text=sentence,
                confidence=0.7,  # Base confidence for text extraction
                timestamp=time.time()
            )
            self.learned_facts.append(fact)
            
            # Track relation patterns
            self.relation_patterns[rel].append((subj, obj))
            
            stats['relations'] += 1
            stats['facts'] += 1
        
        # Store experience in episodic memory
        episode = LiveEpisode(
            timestamp=time.time(),
            task_tag='text_learning',
            situation_hv=sentence_hv,
            state={
                'sentence': sentence,
                'source': source,
                'session_id': session_id,
                'sent_idx': sent_idx,
                'concepts': concepts,
                'relations_count': len(relations)
            },
            action='learn',
            outcome='knowledge_acquired',
            reward=len(concepts) * 0.1 + len(relations) * 0.2,
            emotion='curious',
            impact_score=len(concepts) + len(relations) * 2
        )
        self.episodic.record(episode)
        
        return stats
    
    def _encode_sentence(self, sentence: str) -> hypervec_rs.HyperVector:
        """
        Encode sentence to hypervector using LinguaCortex.
        This uses Semantic Folding (SDR/VSA), NOT embeddings.
        """
        # Learn text in lingua cortex
        self.lingua.learn_text_snippet(sentence)
        
        # Get semantic fingerprint (SDR)
        words = self._tokenize(sentence)
        
        # Bundle word hypervectors
        sentence_hv = None
        for word in words:
            word = word.lower().strip()
            if len(word) < 2:
                continue
            
            # Get or create word hypervector
            word_hv = hypervec_rs.HyperVector(hash(word) % (2**32))
            
            if sentence_hv is None:
                sentence_hv = word_hv
            else:
                sentence_hv = sentence_hv.bundle(word_hv)
        
        return sentence_hv if sentence_hv else hypervec_rs.HyperVector(0)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenizer
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _extract_concepts(self, sentence: str) -> List[str]:
        """
        Extract concepts from sentence.
        Uses simple heuristics (capitalized words, nouns).
        """
        concepts = []
        
        # Extract capitalized words (potential named entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        concepts.extend(capitalized)
        
        # Extract quoted concepts
        quoted = re.findall(r'"([^"]+)"', sentence)
        concepts.extend(quoted)
        
        # Extract words that appear important (longer words, domain terms)
        words = self._tokenize(sentence)
        for word in words:
            if len(word) >= 5 and word not in concepts:
                concepts.append(word.capitalize())
        
        # Deduplicate
        return list(set(concepts))
    
    def _extract_relations(
        self,
        sentence: str,
        concepts: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relations using semantic folding and co-occurrence analysis.
        
        Phase 1 (Bootstrapping): Uses lightweight regex patterns for obvious relations
        Phase 2 (Semantic Folding): Discovers relations via context similarity
        
        Returns list of (subject, relation, object) tuples.
        """
        relations = []
        
        # Phase 1: Bootstrap with explicit linguistic patterns (only for unambiguous cases)
        explicit_patterns = [
            (r'(\w+)\s+is\s+(?:a|an|the)?\s*(?:type\s+of\s+)?(\w+)', 'is_a'),
            (r'(\w+)\s+are\s+(?:a|an|the)?\s*(\w+)', 'is_a'),
            (r'(\w+)\s+causes\s+(\w+)', 'causes'),
            (r'(\w+)\s+produces\s+(\w+)', 'produces'),
        ]
        
        sentence_lower = sentence.lower()
        
        for pattern, relation_type in explicit_patterns:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                subj = match.group(1).capitalize()
                obj = match.group(2).capitalize()
                
                if (subj in concepts or len(subj) > 3) and (obj in concepts or len(obj) > 3):
                    relations.append((subj, relation_type, obj))
        
        # Phase 2: Semantic folding - discover implicit relations via co-occurrence
        folded_relations = self._discover_relations_via_folding(sentence, concepts)
        relations.extend(folded_relations)
        
        return relations
    
    def _discover_relations_via_folding(
        self,
        sentence: str,
        concepts: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Discover relations through semantic folding and co-occurrence analysis.
        
        Algorithm:
        1. For each concept pair in sentence, increment co-occurrence count
        2. Update context hypervectors by bundling with neighboring words
        3. Check similarity between concept context vectors
        4. If similarity > threshold AND sufficient co-occurrence, infer relation
        
        Returns discovered (subject, relation_type, object) tuples.
        """
        relations = []
        
        if len(concepts) < 2:
            return relations
        
        # Tokenize for context window analysis
        words = self._tokenize(sentence)
        word_positions = {word.capitalize(): [] for word in words}
        for idx, word in enumerate(words):
            word_cap = word.capitalize()
            if word_cap in concepts:
                word_positions[word_cap].append(idx)
        
        # Update co-occurrence counts and context vectors
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                # Increment co-occurrence
                pair = tuple(sorted([concept_a, concept_b]))
                self.concept_cooccurrence[pair] += 1
                
                # Update context hypervectors using surrounding words
                self._update_context_vectors(concept_a, concept_b, words, word_positions)
        
        # Discover relations based on accumulated context similarity
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                pair = tuple(sorted([concept_a, concept_b]))
                
                # Check if we have enough evidence
                if self.concept_cooccurrence[pair] < self.min_cooccurrence:
                    continue
                
                # Check context similarity
                if concept_a in self.concept_context_hvs and concept_b in self.concept_context_hvs:
                    try:
                        similarity = self.concept_context_hvs[concept_a].similarity(
                            self.concept_context_hvs[concept_b]
                        )
                        
                        if similarity > self.relation_threshold:
                            # High similarity + co-occurrence = implicit relation
                            relation_type = self._infer_relation_type(
                                concept_a, concept_b, similarity, sentence
                            )
                            relations.append((concept_a, relation_type, concept_b))
                    except Exception:
                        pass  # Skip if similarity calculation fails
        
        return relations
    
    def _update_context_vectors(
        self,
        concept_a: str,
        concept_b: str,
        words: List[str],
        word_positions: Dict[str, List[int]]
    ):
        """
        Update context hypervectors for concepts based on surrounding words.
        
        Uses semantic folding: incrementally bundle context words into concept's
        context vector, weighted by distance.
        """
        for concept in [concept_a, concept_b]:
            if concept not in word_positions:
                continue
            
            # Initialize context vector if first time
            if concept not in self.concept_context_hvs:
                self.concept_context_hvs[concept] = hypervec_rs.HyperVector(hash(concept) % (2**32))
            
            # Extract context window around each occurrence
            for pos in word_positions[concept]:
                window_start = max(0, pos - self.folding_window_size)
                window_end = min(len(words), pos + self.folding_window_size + 1)
                
                # Bundle context words into concept's context vector
                for ctx_pos in range(window_start, window_end):
                    if ctx_pos == pos:
                        continue  # Skip the concept itself
                    
                    context_word = words[ctx_pos].capitalize()
                    distance = abs(ctx_pos - pos)
                    
                    # Create context word vector
                    ctx_hv = hypervec_rs.HyperVector(hash(context_word) % (2**32))
                    
                    # Permute based on distance (encodes position)
                    ctx_hv = ctx_hv.permute(distance)
                    
                    # Bundle into concept's context vector
                    self.concept_context_hvs[concept] = self.concept_context_hvs[concept].bundle(ctx_hv)
    
    def _infer_relation_type(
        self,
        concept_a: str,
        concept_b: str,
        similarity: float,
        sentence: str
    ) -> str:
        """
        Infer relation type based on context and similarity.
        
        Uses heuristics and linguistic cues to categorize the relation.
        Falls back to generic 'semantically_related' if no specific type found.
        """
        sentence_lower = sentence.lower()
        a_lower = concept_a.lower()
        b_lower = concept_b.lower()
        
        # Check for causal indicators in surrounding context
        if any(word in sentence_lower for word in ['cause', 'because', 'due to', 'result', 'lead']):
            return 'causes'
        
        # Check for taxonomic indicators
        if any(word in sentence_lower for word in ['type of', 'kind of', 'is a', 'are']):
            return 'is_a'
        
        # Check for part-whole indicators
        if any(word in sentence_lower for word in ['part of', 'contain', 'compos', 'include']):
            return 'part_of'
        
        # Check for similarity indicators
        if any(word in sentence_lower for word in ['similar', 'like', 'resemble', 'analogous']):
            return 'similar_to'
        
        # Very high similarity suggests strong semantic relation
        if similarity > 0.8:
            return 'strongly_related'
        
        # Default: generic semantic relation
        return 'semantically_related'
    
    def get_folding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about semantic folding and relation discovery.
        
        Returns telemetry about co-occurrence patterns and discovered relations.
        """
        total_pairs = len(self.concept_cooccurrence)
        high_cooccurrence = sum(1 for count in self.concept_cooccurrence.values() if count >= self.min_cooccurrence)
        
        # Find most co-occurring concepts
        top_cooccurrences = sorted(
            self.concept_cooccurrence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Analyze relation types discovered
        relation_types = Counter()
        for fact in self.learned_facts:
            relation_types[fact.relation] += 1
        
        return {
            'total_concept_pairs': total_pairs,
            'high_cooccurrence_pairs': high_cooccurrence,
            'context_vectors_tracked': len(self.concept_context_hvs),
            'folding_window_size': self.folding_window_size,
            'relation_threshold': self.relation_threshold,
            'min_cooccurrence': self.min_cooccurrence,
            'top_cooccurring_pairs': [(f"{a}~{b}", count) for (a, b), count in top_cooccurrences],
            'relation_types_discovered': dict(relation_types),
            'emergent_relations': sum(1 for r in relation_types if r in ['semantically_related', 'strongly_related']),
            'explicit_relations': sum(1 for r in relation_types if r in ['is_a', 'causes', 'produces'])
        }
    
    def discover_emergent_relations(self, min_similarity: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Discover emergent relations by analyzing all concept context vectors.
        
        This is a batch analysis that finds implicit relations across the entire
        learned knowledge base, not just within sentences.
        
        Args:
            min_similarity: Minimum context similarity to consider a relation
            
        Returns:
            List of (concept_a, concept_b, similarity) tuples
        """
        emergent_relations = []
        concepts = list(self.concept_context_hvs.keys())
        
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                try:
                    similarity = self.concept_context_hvs[concept_a].similarity(
                        self.concept_context_hvs[concept_b]
                    )
                    
                    if similarity >= min_similarity:
                        pair = tuple(sorted([concept_a, concept_b]))
                        cooccurrence = self.concept_cooccurrence.get(pair, 0)
                        
                        # Only add if we have some cooccurrence evidence
                        if cooccurrence >= self.min_cooccurrence:
                            emergent_relations.append((concept_a, concept_b, similarity))
                except Exception:
                    continue
        
        # Sort by similarity descending
        emergent_relations.sort(key=lambda x: x[2], reverse=True)
        return emergent_relations
    
    def query_learned_knowledge(
        self,
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query learned knowledge using semantic search.
        
        Returns:
        - Relevant concepts
        - Related facts
        - Episodic memories
        - Reasoning trace
        """
        print(f"\n[TextLearner] Querying: '{query_text}'")
        
        # Encode query to hypervector
        query_hv = self._encode_sentence(query_text)
        
        # Extract query concepts
        query_concepts = self._extract_concepts(query_text)
        
        # Search semantic memory
        similar_concepts = self.semantic.query(query_hv, k=top_k)
        
        # Spreading activation from query concepts
        activated = self.semantic.spread_activation(query_concepts, steps=3)
        top_activated = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Search episodic memory
        recalled_episodes = []
        try:
            recalled_episodes = self.episodic.recall_similar(
                query_hv, 
                task_tag='text_learning', 
                k=top_k
            )
        except Exception as e:
            # If recall fails, just continue without episodes
            pass
        
        # Find related facts
        related_facts = []
        for fact in self.learned_facts:
            # Check if fact involves query concepts or similar concepts
            fact_concepts = [fact.subject, fact.object]
            for qc in query_concepts:
                if any(qc.lower() in fc.lower() or fc.lower() in qc.lower() for fc in fact_concepts):
                    related_facts.append(fact)
                    break
        
        # Build reasoning trace
        reasoning_trace = [
            f"Query encoded to hypervector",
            f"Extracted query concepts: {query_concepts}",
            f"Found {len(similar_concepts)} similar concepts in semantic memory",
            f"Activated {len(top_activated)} related concepts via spreading activation",
            f"Recalled {len(recalled_episodes)} relevant episodes",
            f"Found {len(related_facts)} related facts"
        ]
        
        # Build answer
        answer_parts = []
        
        if similar_concepts:
            answer_parts.append("**Relevant Concepts:**")
            for concept, sim in similar_concepts[:3]:
                answer_parts.append(f"- {concept} (similarity: {sim:.3f})")
        
        if related_facts:
            answer_parts.append("\n**Learned Facts:**")
            for fact in related_facts[:5]:
                answer_parts.append(f"- {fact.subject} {fact.relation} {fact.object}")
                answer_parts.append(f"  Source: '{fact.source_text[:60]}...'")
        
        if top_activated:
            answer_parts.append("\n**Associated Concepts:**")
            for concept, activation in top_activated[:5]:
                if concept not in [c for c, _ in similar_concepts[:3]]:
                    answer_parts.append(f"- {concept} (activation: {activation:.3f})")
        
        answer = "\n".join(answer_parts) if answer_parts else "No relevant knowledge found."
        
        return {
            'answer': answer,
            'confidence': self._calculate_confidence(similar_concepts, related_facts),
            'similar_concepts': [(c, float(s)) for c, s in similar_concepts],
            'activated_concepts': [(c, float(a)) for c, a in top_activated],
            'related_facts': [
                {
                    'subject': f.subject,
                    'relation': f.relation,
                    'object': f.object,
                    'source': f.source_text[:100],
                    'confidence': f.confidence
                }
                for f in related_facts[:10]
            ],
            'recalled_episodes': len(recalled_episodes),
            'reasoning_trace': reasoning_trace
        }
    
    def _calculate_confidence(
        self,
        similar_concepts: List[Tuple[str, float]],
        related_facts: List[LearnedFact]
    ) -> float:
        """Calculate confidence in query response."""
        if not similar_concepts and not related_facts:
            return 0.1
        
        # Base confidence on similarity scores and fact count
        avg_similarity = sum(s for _, s in similar_concepts[:3]) / max(len(similar_concepts[:3]), 1)
        fact_score = min(len(related_facts) * 0.1, 0.5)
        
        confidence = (avg_similarity * 0.7) + fact_score
        return min(confidence, 0.95)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_sessions': len(self.learning_sessions),
            'total_concepts': len(self.semantic.concept_hvs),
            'total_facts': len(self.learned_facts),
            'total_episodes': len(self.episodic.recent),
            'concept_frequencies': dict(self.concept_frequencies.most_common(20)),
            'relation_types': {
                rel: len(pairs)
                for rel, pairs in self.relation_patterns.items()
            },
            'sessions': [
                {
                    'session_id': s.session_id,
                    'filename': s.filename,
                    'duration': s.end_time - s.start_time,
                    'concepts': s.concepts_learned,
                    'relations': s.relations_learned,
                    'facts': s.facts_stored
                }
                for s in self.learning_sessions
            ]
        }
    
    def export_learned_knowledge(self) -> str:
        """Export learned knowledge as formatted text."""
        output = []
        output.append("="*80)
        output.append("NSCK Text Knowledge Learner - Learned Knowledge Export")
        output.append("="*80)
        output.append("")
        
        # Statistics
        stats = self.get_statistics()
        output.append("STATISTICS:")
        output.append(f"  Total Learning Sessions: {stats['total_sessions']}")
        output.append(f"  Total Concepts Learned: {stats['total_concepts']}")
        output.append(f"  Total Facts Stored: {stats['total_facts']}")
        output.append(f"  Total Episodes: {stats['total_episodes']}")
        output.append("")
        
        # Sessions
        output.append("LEARNING SESSIONS:")
        for session in self.learning_sessions:
            output.append(f"  Session {session.session_id}:")
            output.append(f"    File: {session.filename}")
            output.append(f"    Duration: {session.end_time - session.start_time:.2f}s")
            output.append(f"    Concepts: {session.concepts_learned}")
            output.append(f"    Relations: {session.relations_learned}")
            output.append(f"    Facts: {session.facts_stored}")
        output.append("")
        
        # Top concepts
        output.append("TOP CONCEPTS (by frequency):")
        for concept, freq in stats['concept_frequencies'].items():
            output.append(f"  {concept}: {freq} occurrences")
        output.append("")
        
        # Learned facts
        output.append("LEARNED FACTS:")
        for fact in self.learned_facts[:50]:  # Limit to 50 for readability
            output.append(f"  {fact.subject} --[{fact.relation}]--> {fact.object}")
            output.append(f"    Source: '{fact.source_text[:80]}...'")
            output.append(f"    Confidence: {fact.confidence:.2f}")
        
        if len(self.learned_facts) > 50:
            output.append(f"  ... and {len(self.learned_facts) - 50} more facts")
        output.append("")
        
        output.append("="*80)
        
        return "\n".join(output)
