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

# V3: pronouns used for coreference scanning — module-level constant (not recreated per sentence)
_COREFERENCE_PRONOUNS: frozenset = frozenset({
    'he', 'him', 'his', 'she', 'her', 'it', 'its',
    'they', 'them', 'their', 'this', 'that',
})

# Particles, adverbs and prepositions that should NEVER become concept nodes.
# These slip through when a sentence fragment ends with them (e.g. "move away").
_STOP_CONCEPTS: frozenset = frozenset({
    'away', 'back', 'down', 'off', 'out', 'over', 'up', 'through', 'along',
    'around', 'aside', 'ahead', 'behind', 'below', 'above', 'beside', 'between',
    'beyond', 'within', 'without', 'toward', 'towards', 'inside', 'outside',
    'across', 'against', 'among', 'amongst', 'upon', 'onto', 'into', 'unto',
    'via', 'per', 'thus', 'hence', 'still', 'even', 'else', 'ever',
    'never', 'often', 'always', 'usually', 'sometimes', 'rarely', 'already',
    'soon', 'later', 'once', 'twice', 'again', 'much', 'many', 'few', 'less',
    'more', 'most', 'very', 'quite', 'rather', 'nearly', 'almost', 'too',
    'well', 'just', 'only', 'also', 'both', 'either', 'neither',
    'but', 'and', 'or', 'not', 'nor', 'so', 'yet', 'for',
})

# Generic relation types — require stricter similarity threshold to avoid noise
_GENERIC_RELATION_TYPES: frozenset = frozenset({'semantically_related', 'strongly_related'})

# Similarity threshold applied to generic (co-occurrence only) relations.
# Must be higher than the regular relation_threshold to suppress noisy edges.
_GENERIC_RELATION_THRESHOLD: float = 0.62

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
except ImportError:
    from python.core.vsa.hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()

from python.core.language.lingua_cortex import get_lingua_cortex, SemanticFingerprint
from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.reasoning.context_engine import ContextEngine
from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner
# [AGI] Phase 4: Language Integration
try:
    from python.core.language.language_module import LanguageModule
except ImportError:
    LanguageModule = None

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
        context_engine: Optional[ContextEngine] = None,
        language_module: Optional[Any] = None, # [AGI] Phase 4: NLU Parser
        causal_graph: Optional[CausalGraph] = None,
        config=None,
    ):
        self._config = config
        # Core cognitive modules
        self.semantic = semantic_memory or SemanticMemory()
        self.episodic = episodic_memory or EpisodicMemory()
        self.context = context_engine or ContextEngine(self.semantic)
        self.language_module = language_module
        
        # Language cortex for text encoding (NOT an LLM!)
        self.lingua = get_lingua_cortex()
        
        # Causal reasoning — use the shared graph when provided so that
        # causal facts learned here flow directly into the engine's reasoner.
        self.causal_graph = causal_graph if causal_graph is not None else CausalGraph()
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
        self.relation_threshold = 0.55  # [Refactor] Lowered threshold for greater recall
        self.min_cooccurrence = 1  # [Refactor] Lowered to 1 for One-Shot Learning

        # O(1) fact-duplicate index: (subject, relation, object) → index in learned_facts
        self._fact_index: Dict[Tuple[str, str, str], int] = {}

        # V3: optional language modules (lazy-loaded when flags are set)
        self._construction_matcher = None
        self._frame_library = None
        self._entity_register = None
        self._distrib_codebook = None
        self._config = config
        if config is not None:
            if getattr(config, 'enable_construction_grammar', False):
                try:
                    from python.core.language.construction_grammar import ConstructionMatcher
                    self._construction_matcher = ConstructionMatcher()
                except Exception as e:
                    print(f"[TextLearner] Construction grammar unavailable: {e}")
            if getattr(config, 'enable_frame_semantics', False):
                try:
                    from python.core.language.frame_semantics import FrameLibrary
                    self._frame_library = FrameLibrary()
                except Exception as e:
                    print(f"[TextLearner] Frame semantics unavailable: {e}")
            if getattr(config, 'enable_coreference', False):
                try:
                    from python.core.language.coreference import EntityRegister
                    self._entity_register = EntityRegister()
                except Exception as e:
                    print(f"[TextLearner] Coreference unavailable: {e}")
            if getattr(config, 'enable_distributional_semantics', False):
                try:
                    from python.core.language.distributional_semantics import DistributionalCodebook
                    use_hf = getattr(config, 'enable_hf_corpus', False)
                    # build_default pre-trains on BUILTIN_CORPUS (+HF when enabled)
                    self._distrib_codebook = DistributionalCodebook.build_default(
                        enable_hf_corpus=use_hf
                    )
                except Exception as e:
                    print(f"[TextLearner] Distributional semantics unavailable: {e}")
        
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

    def learn_from_text(self, text: str, source: str = "string_input") -> LearningSession:
        """Convenience method to learn from a string."""
        import hashlib
        import time
        
        session_id = hashlib.md5(f"{time.time()}_{source}".encode()).hexdigest()[:8]
        return self._learn_from_text(text, source, session_id)

    # Alias for backward compatibility
    learn_text = learn_from_text
    
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
        # Configurable soft cap — warn rather than silently truncate.
        # Default max is 1000 sentences; override via max_sentences param.
        _SOFT_CAP = 1000
        if len(sentences) > _SOFT_CAP:
            print(f"[TextLearner] WARNING: {len(sentences)} sentences; truncating to {_SOFT_CAP}. "
                  "Pass max_sentences= to change limit.")
            sentences = sentences[:_SOFT_CAP]
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
          - [NEW] If LanguageModule available, use deep parsing for frames/conditions
        4. Store in semantic and episodic memory
        """
        stats = {'concepts': 0, 'relations': 0, 'facts': 0}
        
        # Encode sentence to hypervector using LinguaCortex
        sentence_hv = self._encode_sentence(sentence)
        
        # Extract concepts (simple noun extraction + named entities)
        concepts = self._extract_concepts(sentence)

        # V3: Coreference resolution — scan for pronouns and resolve before relation extraction
        if self._entity_register is not None:
            words = self._tokenize(sentence)
            resolved_entities: List[str] = []
            for word in words:
                if word.lower() in _COREFERENCE_PRONOUNS:
                    mention = self._entity_register.resolve(word.lower())
                    if mention:
                        resolved = mention.name.capitalize()
                        resolved_entities.append(resolved)
            # Remove pronoun forms from concepts (they are sentence-initial capitals, not entities)
            concepts = [c for c in concepts if c.lower() not in _COREFERENCE_PRONOUNS]
            # Add resolved entities
            for resolved in resolved_entities:
                if resolved not in concepts:
                    concepts.append(resolved)
            # Register new named entities (capitalized proper nouns, excluding pronouns)
            for concept in concepts:
                if concept[0].isupper() and concept.lower() not in _COREFERENCE_PRONOUNS:
                    c_hv = hypervec_rs.HyperVector(abs(hash(concept)) % (2**32))
                    self._entity_register.register(concept, c_hv, {})
        
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
        relations = []

        # [V3-A] Frame Semantics — identify verb, lookup frame, fill roles
        # Simple positional heuristic: assign concepts to frame roles in order.
        if self._frame_library is not None and len(concepts) >= 2:
            try:
                words_list = self._tokenize(sentence)
                for word in words_list:
                    frame = self._frame_library.find_frame(word)
                    if frame is not None:
                        roles_list = list(frame.roles.keys())
                        # Assign concepts to roles positionally (concept[0] → role[0], etc.)
                        role_to_concept = {}
                        for idx, role in enumerate(roles_list):
                            if idx < len(concepts):
                                role_to_concept[role] = concepts[idx]
                        if len(role_to_concept) >= 2:
                            subj_concept = role_to_concept[roles_list[0]]
                            obj_concept = role_to_concept[roles_list[1]]
                            if subj_concept and obj_concept and subj_concept != obj_concept:
                                relations.append((subj_concept, frame.name.lower(), obj_concept))
                        break  # Use first frame match per sentence
            except Exception:
                pass  # graceful fallback
        
        # [A] Deep Parsing via LanguageModule (Prioritize if available)
        if self.language_module:
            try:
                understanding = self.language_module.understand(sentence)
                structured = understanding.get("structured_output", {})
                frames = structured.get("frames", {})
                
                # 1. Standard Agent-Action-Patient
                if frames.get("agent") and frames.get("action") and frames.get("patient"):
                     relations.append((frames["agent"].capitalize(), frames["action"], frames["patient"].capitalize()))
                
                # 2. Conditional Logic (The nuance we missed before)
                if frames.get("condition"):
                    condition_text = frames["condition"]
                    
                    # Handle None values safely
                    ag = frames.get('agent') or 'something'
                    ac = frames.get('action') or 'does'
                    pa = frames.get('patient') or ''
                    
                    consequence_text = f"{ag} {ac} {pa}".strip()
                    
                    relations.append((condition_text.capitalize(), "implies", consequence_text.capitalize()))
                    
                    cond_concepts = self._extract_concepts(condition_text)
                    for cc in cond_concepts:
                         relations.append((cc, "leads_to", consequence_text.capitalize()))

                # 3. Causal Logic (Because...)
                if frames.get("cause"):
                    cause_text = frames["cause"]
                    effect_text = f"{frames.get('agent')} {frames.get('action')} {frames.get('patient')}"
                    relations.append((cause_text.capitalize(), "causes", effect_text.capitalize()))

            except Exception as e:
                print(f"[TextLearner] Deep parsing failed: {e}")
        
        # [B] Fallback / Augment with Heuristics
        heuristic_relations = self._extract_relations(sentence, concepts)
        relations.extend(heuristic_relations)
        
        # Deduplicate matches
        relations = list(set(relations))

        # V3: If coreference is active, strip pronoun-role fillers from relations
        # (they appear as subjects/objects from construction grammar matches)
        if self._entity_register is not None:
            relations = [
                (s, r, o) for s, r, o in relations
                if s.lower() not in _COREFERENCE_PRONOUNS
                and o.lower() not in _COREFERENCE_PRONOUNS
            ]
        
        # Store relations in semantic memory and causal graph
        current_time = time.time()
        sim_time = getattr(self, 'current_simulated_time', 0.0)
        ts = sim_time if sim_time > 0 else current_time
        
        for subj, rel, obj in relations:
            # Ensure concepts exist before adding relation
            for concept in (subj, obj):
                if concept not in self.semantic.concept_hvs:
                    # Use distributional HV when available — gives better semantic
                    # similarity than a raw hash (words in similar contexts cluster)
                    concept_hv = None
                    if self._distrib_codebook is not None:
                        concept_hv = self._distrib_codebook.get_hv(concept.lower())
                    if concept_hv is None:
                        concept_hv = hypervec_rs.HyperVector(hash(concept) % (2**32))
                    self.semantic.add_concept(
                        concept_name=concept,
                        properties={'source': source, 'session': session_id},
                        hv_override=concept_hv
                    )

            # Add relation to semantic memory (with timestamp)
            self.semantic.add_relation(subj, rel, obj, timestamp=ts)
            
            # Causal Graph
            if rel in ["causes", "leads_to", "implies", "results_in"]:
                try:
                    self.causal_graph.add_causes(
                        cause=subj,
                        effect=obj,
                        strength=0.7,
                        context='text_learning',
                    )
                except Exception:
                    pass
            
            # Learn Fact (Symbolic)
            new_fact = LearnedFact(
                subject=subj,
                relation=rel,
                object=obj,
                confidence=0.9,
                source_text=f"{source[:30]}...",
                timestamp=ts
            )
            self._store_fact(new_fact)
            
            # Track relation patterns
            self.relation_patterns[rel].append((subj, obj))
            
            stats['relations'] += 1
            stats['facts'] += 1

        # V4: transitive inference — derive new facts after each sentence
        # (run lazily every 10 sentences to amortise cost)
        if (self._config is not None and
                getattr(self._config, 'enable_transitive_inference', False) and
                getattr(self, '_tkl_sentence_count', 0) % 10 == 0):
            try:
                for rel_type in ("is_a", "part_of", "causes", "located_in"):
                    self.semantic.infer_transitive(rel_type, max_hops=2)
            except Exception:
                pass
        self._tkl_sentence_count = getattr(self, '_tkl_sentence_count', 0) + 1
        
        # Store experience in episodic memory
        episode = LiveEpisode(
            timestamp=ts, # Use the simulated timestamp for episode too
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

        V3 enhancements (gated by config flags):
        - ``enable_contextual_encoding``: uses positional permutation encoding
        - ``enable_distributional_semantics``: uses distributional codebook HVs
        """
        # Learn text in lingua cortex
        self.lingua.learn_text_snippet(sentence)
        
        # Get semantic fingerprint (SDR)
        words = self._tokenize(sentence)
        words_clean = [w.lower().strip() for w in words if len(w.strip()) >= 2]

        # V3: Feed words to distributional codebook for online learning
        if self._distrib_codebook is not None:
            self._distrib_codebook.build_from_corpus([words_clean])

        use_contextual = (self._config is not None and
                          getattr(self._config, 'enable_contextual_encoding', False))

        # Bundle word hypervectors
        sentence_hv = None
        for idx, word in enumerate(words_clean):
            # Choose HV source: distributional > contextual > hash
            if self._distrib_codebook is not None:
                word_hv = self._distrib_codebook.get_hv(word)
                if word_hv is None:
                    word_hv = hypervec_rs.HyperVector(abs(hash(word)) % (2**32))
            elif use_contextual:
                prev_w = words_clean[idx - 1] if idx > 0 else None
                next_w = words_clean[idx + 1] if idx < len(words_clean) - 1 else None
                word_hv = self.encode_word_in_context(word, prev_w, next_w)
            else:
                word_hv = hypervec_rs.HyperVector(hash(word) % (2**32))
            
            if sentence_hv is None:
                sentence_hv = word_hv
            else:
                sentence_hv = sentence_hv.bundle(word_hv)
        
        return sentence_hv if sentence_hv else hypervec_rs.HyperVector(0)

    def encode_word_in_context(
        self,
        word: str,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None,
    ) -> hypervec_rs.HyperVector:
        """V3: Contextual word encoding using positional permutation.

        context_hv = base_hv.bundle(prev_hv.permute(1)).bundle(next_hv.permute(-1))
        """
        base_hv = hypervec_rs.HyperVector(hash(word.lower()) % (2**32))
        if prev_word is not None:
            prev_hv = hypervec_rs.HyperVector(hash(prev_word.lower()) % (2**32))
            if hasattr(prev_hv, 'permute'):
                base_hv = base_hv.bundle(prev_hv.permute(1))
        if next_word is not None:
            next_hv = hypervec_rs.HyperVector(hash(next_word.lower()) % (2**32))
            if hasattr(next_hv, 'permute'):
                base_hv = base_hv.bundle(next_hv.permute(-1))
        return base_hv
    
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
        stop_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'but', 'not', 'are', 'was', 'were', 'has', 'had', 'can', 'may', 'its', 'his', 'her', 'our', 'their', 'she', 'him', 'you', 'who', 'how', 'why', 'any', 'all', 'one', 'two', 'six', 'ten', 'yes', 'no', 'nor', 'yet', 'per', 'via', 'etc', 'viz', 'i.e', 'e.g', 'non', 'sub', 'pre', 'pro', 'con', 'sur', 'co',
            'which', 'also', 'other', 'because', 'after', 'before', 'have', 'they', 'there', 'some', 'been', 'would', 'could', 'should', 'than', 'then', 'now', 'only', 'just', 'more', 'most', 'such', 'into', 'over', 'under', 'again', 'further', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        for word in words:
            w_lower = word.lower()
            if w_lower in stop_words:
                continue
            # V7: skip particle/adverb pseudo-concepts that produce noisy edges
            if w_lower in _STOP_CONCEPTS:
                continue

            # Allow length >= 3 (e.g. Sky, Red, Sun, Eye, Ear)
            if len(word) >= 3 and word.capitalize() not in concepts:
                concepts.append(word.capitalize())
        
        # Deduplicate while preserving order
        seen = set()
        ordered_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                ordered_concepts.append(c)

        # Cap concepts per sentence to avoid O(k^2) blow-up in
        # _discover_relations_via_folding.  Keep capitalized / longer
        # concepts first (more likely to be meaningful entities).
        _MAX_CONCEPTS = 20
        if len(ordered_concepts) > _MAX_CONCEPTS:
            # Sort: prefer capitalized words, then longer words
            ordered_concepts.sort(
                key=lambda c: (c[0].isupper(), len(c)), reverse=True)
            ordered_concepts = ordered_concepts[:_MAX_CONCEPTS]

        return ordered_concepts
    
    def _extract_relations(
        self,
        sentence: str,
        concepts: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relations using semantic folding and co-occurrence analysis.
        
        Phase 1 (Bootstrapping): Uses linguistic patterns and S-V-O heuristics
        Phase 2 (Semantic Folding): Discovers relations via context similarity
        
        Returns list of (subject, relation, object) tuples.
        """
        relations = []
        
        # V3: Construction Grammar — try first if enabled; fall through to regex if no match
        if self._construction_matcher is not None:
            try:
                words_list = self._tokenize(sentence)
                cg_matches = self._construction_matcher.match(words_list)
                for match in cg_matches:
                    fillers = match.role_fillers
                    rel = match.construction.relation
                    cname = match.construction.name

                    # V4: skip negation constructions unless the flag is on
                    is_negation = rel.startswith("not_") or rel in ("lacks", "cannot_do", "never_does", "acts_without", "lacks_relation")
                    if is_negation:
                        if not (self._config is not None and getattr(self._config, 'enable_negation_handling', False)):
                            continue

                    # V4: skip temporal constructions unless the flag is on
                    is_temporal = rel in ("precedes", "follows", "since_event", "until_event", "triggered_by", "occurs_during")
                    if is_temporal:
                        if not (self._config is not None and getattr(self._config, 'enable_temporal_reasoning', False)):
                            continue

                    # V4: skip conditional constructions unless the flag is on
                    is_conditional = rel in ("implies", "conditional_on", "unless_condition", "when_then", "only_if", "leads_to", "results_in")
                    if is_conditional:
                        if not (self._config is not None and getattr(self._config, 'enable_conditional_logic', False)):
                            continue

                    # Extract subject/object from role fillers — V4 extended role name list
                    subj = None
                    obj = None
                    for role_a in (
                        'subject', 'cause', 'owner', 'entity', 'agent',
                        # V4:
                        'antecedent', 'event1', 'early_subject', 'entity1',
                        'condition_subject', 'negated_subject', 'dependent',
                        'tool', 'part', 'person', 'artifact', 'left',
                        'enabler', 'preventer', 'whole', 'term', 'quantity',
                        'producer', 'container',
                    ):
                        if role_a in fillers:
                            subj = fillers[role_a].capitalize()
                            break
                    for role_b in (
                        'attribute', 'effect', 'owned', 'location', 'patient', 'object',
                        # V4:
                        'consequent', 'event2', 'late_subject', 'entity2',
                        'condition_object', 'negated_attribute', 'negated_quality',
                        'missing', 'alias', 'definition', 'purpose', 'material',
                        'required', 'enabled', 'prevented', 'product', 'contained',
                        'right', 'birthplace', 'dependency', 'responsibility',
                        'capability', 'absent', 'excluded_location',
                    ):
                        if role_b in fillers:
                            obj = fillers[role_b].capitalize()
                            break
                    if subj and obj and len(subj) >= 2 and len(obj) >= 2:
                        relations.append((subj, rel, obj))
                if relations:
                    # Construction grammar matched — still run heuristics to augment
                    pass
            except Exception as e:
                import logging as _log
                _log.getLogger("nsck.tkl").debug("[TKL] Construction grammar matching failed: %s", e)

        sentence_lower = sentence.lower()
        
        # --- Phase 1: Explicit Linguistic Patterns ---
        
        # 1. Definitive patterns (High confidence)
        explicit_patterns = [
            (r'(\w+)\s+is\s+(?:a|an|the)?\s*(?:type\s+of\s+)?(\w+)', 'is_a'),
            (r'(\w+)\s+are\s+(?:a|an|the)?\s*(\w+)', 'is_a'),
            (r'(\w+)\s+causes\s+(\w+)', 'causes'),
            (r'(\w+)\s+produces\s+(\w+)', 'produces'),
            (r'(\w+)\s+leads\s+to\s+(\w+)', 'leads_to'),
            (r'(\w+)\s+results\s+in\s+(\w+)', 'results_in'),
            (r'(\w+)\s+made\s+(?:entirely\s+|partially\s+)?of\s+(\w+)', 'made_of'),
        ]
        
        for pattern, relation_type in explicit_patterns:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                subj = match.group(1).capitalize()
                obj = match.group(2).capitalize()
                
                # Validation: Concepts must be somewhat significant (length >= 3 or known)
                if (len(subj) >= 3) and (len(obj) >= 3):
                    relations.append((subj, relation_type, obj))

        # 2. Generic S-V-O Heuristic (Medium confidence)
        # Look for "Noun Verb Noun" patterns where Nouns are in our extracted concepts
        # This allows capturing "Resonance attracts space_whales" -> (Resonance, attracts, Space_whales)
        
        words = self._tokenize(sentence)
        # Map words to concepts if possible
        tokens = []
        for w in words:
            cap = w.capitalize()
            if cap in concepts:
                tokens.append({'text': cap, 'type': 'CONCEPT'})
            elif w in ['is', 'are', 'was', 'were']:
                tokens.append({'text': w, 'type': 'AUX'})
            elif w in ['a', 'an', 'the']:
                tokens.append({'text': w, 'type': 'DET'})
            elif len(w) >= 3: # Potential verb or other
                tokens.append({'text': w, 'type': 'WORD'})
        
        # Scan for Concept - Word(Verb) - Concept
        for i in range(len(tokens) - 2):
            t1 = tokens[i]
            t2 = tokens[i+1]
            t3 = tokens[i+2]
            
            # Pattern: Concept - [Verb/Relation] - Concept
            if t1['type'] == 'CONCEPT' and t3['type'] == 'CONCEPT':
                
                # Check middle token
                is_valid_relation = False
                relation_text = t2['text']
                
                # 1. It's a generic word
                if t2['type'] == 'WORD':
                    is_valid_relation = True
                    
                # 2. It's a concept (verbs like 'Attracts' might be auto-tagged as concepts)
                # But we must ensure it's not just a list like "Apples, Oranges, Bananas"
                elif t2['type'] == 'CONCEPT':
                    # Heuristic: If it ends in 's' or 'ed' or 'ing', likely a verb
                    if relation_text.endswith('s') or relation_text.endswith('ed') or relation_text.endswith('ing'):
                        is_valid_relation = True
                        
                # 3. It's 'is_a' helper
                elif t2['type'] == 'AUX' and (i+2 < len(tokens)):
                     # Handle "X is Y" where Y is concept
                     relations.append((t1['text'], 'is_a', t3['text']))
                     continue

                if is_valid_relation:
                    relations.append((t1['text'], relation_text, t3['text']))
        
        # Deduplicate
        relations = list(set(relations))

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
                        
                        # [Refactor] Relation Inference with Override
                        # 1. Infer potential relationship type from context
                        relation_type = self._infer_relation_type(
                            concept_a, concept_b, similarity, sentence
                        )

                        # 2. Check criteria — generic relations need a stricter
                        #    threshold to suppress noise ("memory related to away")
                        is_strong_relation = relation_type not in _GENERIC_RELATION_TYPES
                        threshold = (
                            _GENERIC_RELATION_THRESHOLD
                            if not is_strong_relation
                            else self.relation_threshold
                        )
                        if is_strong_relation or (similarity > threshold):
                            # If specific linguistic marker exists, trust it even if
                            # similarity is low; otherwise require high-confidence sim.
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
        
        # [Refactor] Check for composition/material indicators (Priority)
        if any(word in sentence_lower for word in ['made of', 'made from', 'consist', 'material']):
            return 'made_of'

        # Check for causal indicators in surrounding context
        if any(word in sentence_lower for word in ['cause', 'because', 'due to', 'result', 'lead']):
            return 'causes'
        
        # Check for part-whole indicators
        if any(word in sentence_lower for word in ['part of', 'contain', 'compos', 'include']):
            return 'part_of'
        
        # Check for similarity indicators
        if any(word in sentence_lower for word in ['similar', 'like', 'resemble', 'analogous']):
            return 'similar_to'
        
        # Check for taxonomic indicators (Low priority, 'are' is common)
        if any(word in sentence_lower for word in ['type of', 'kind of', 'is a', 'are']):
            return 'is_a'
        
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
        
        # 2. Spreading Activation
        # Activate concepts strongly associated with query terms
        activation = self.semantic.spread_activation(
            query_concepts,
            steps=6,  # [Adv Reasoning] Increased from 3 to 6 for 5-step transitive chains
            decay=0.7 # [Adv Reasoning] Reduced decay slightly (0.6 -> 0.7) to keep signal alive
        )
        top_activated = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
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
        # Only scan the most recent facts to avoid O(n*m) on huge fact stores
        _MAX_FACTS_SCAN = 5000
        facts_to_scan = self.learned_facts[-_MAX_FACTS_SCAN:] if len(self.learned_facts) > _MAX_FACTS_SCAN else self.learned_facts
        for fact in facts_to_scan:
            # Check if fact involves query concepts or similar concepts
            fact_concepts = [fact.subject, fact.object]
            for qc in query_concepts:
                if any(qc.lower() in fc.lower() or fc.lower() in qc.lower() for fc in fact_concepts):
                    related_facts.append(fact)
                    break
            if len(related_facts) >= 50:  # cap results
                break
        
        # Sort facts by timestamp (most recent first) to prioritize current beliefs
        related_facts.sort(key=lambda x: x.timestamp, reverse=True)
        
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
    
    def _store_fact(self, new_fact: LearnedFact):
        """Store a fact, handling updates and duplicates based on timestamp.

        Uses ``_fact_index`` (dict) for O(1) duplicate lookup instead of
        scanning the entire ``learned_facts`` list.
        """
        key = (new_fact.subject, new_fact.relation, new_fact.object)

        existing_idx = self._fact_index.get(key, -1)
        if existing_idx >= 0:
            # Fact exists. Only update if new one is more recent.
            existing_fact = self.learned_facts[existing_idx]
            if new_fact.timestamp >= existing_fact.timestamp:
                self.learned_facts[existing_idx] = new_fact
        else:
            # New fact
            idx = len(self.learned_facts)
            self.learned_facts.append(new_fact)
            self._fact_index[key] = idx

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

if __name__ == "__main__":
    # Internal Demo / Test
    import os
    
    # 1. Initialize
    lang_mod = LanguageModule()
    learner = TextKnowledgeLearner(language_module=lang_mod)
    
    # 2. Find planetary corpus
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_corpus")
    corpus_file = os.path.join(data_dir, "xylophone_planets.txt")
    
    if os.path.exists(corpus_file):
        print(f"\n[Demo] Learning from {os.path.basename(corpus_file)}...")
        learner.learn_from_text_file(corpus_file)
        
        # 3. Test Query (The one that used to fail)
        query = "What are Xylophone planets made of?"
        print(f"\n[Demo] Query: '{query}'")
        response = learner.query_learned_knowledge(query)
        print(f"Result:\n{response['answer']}")
        
        # 4. Check status of the internal graph
        print(f"\n[Demo] Memory Stats:")
        print(f"  Concepts: {len(learner.semantic.concept_hvs)}")
        print(f"  Facts:    {len(learner.learned_facts)}")
    else:
        print(f"ERROR: Corpus not found at {corpus_file}")
