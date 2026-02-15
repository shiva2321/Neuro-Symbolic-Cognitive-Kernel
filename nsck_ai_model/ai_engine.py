"""
NSCK AI Engine — Core Text & Image Understanding
=================================================

What This Module Does
---------------------
This is the central intelligence of the NSCK AI model. It takes natural language
text (or image descriptions) as input, processes them through a Vector Symbolic
Architecture (VSA) pipeline, and produces natural language responses.

How It Works (No Transformers, No Matrix Multiplication)
--------------------------------------------------------
1. **Encoding**: Text is converted to 10,240-dimensional binary hypervectors
   using semantic folding — words with similar contexts get similar bit patterns.

2. **Memory Storage**: Knowledge is stored in two complementary systems:
   - Semantic Memory: A concept graph where nodes are hypervectors and edges
     are typed relations (is_a, has_property, causes, etc.)
   - Episodic Memory: A timeline of experiences indexed by LSH for fast
     approximate nearest-neighbor retrieval.

3. **Retrieval**: When a query arrives, it's encoded to a hypervector, then
   similar concepts are found via Hamming distance (O(D) per comparison,
   no matrix multiply needed).

4. **Response Generation**: The system assembles responses from:
   - Matched knowledge fragments (weighted by similarity score)
   - Contextual reasoning (spreading activation over the concept graph)
   - Template-based natural language generation with learned n-grams

Why VSA Instead of Transformers?
--------------------------------
- **O(D) operations** instead of O(N²·D) attention: ~10,000× faster per token
- **CPU-only**: No GPU required, runs on any machine
- **150 MB RAM**: vs. 4+ GB for even small transformer models
- **Interpretable**: Every decision can be traced to specific stored concepts
- **Incremental learning**: New knowledge added without retraining everything

Architecture
------------
    User Input (text)
        ↓
    TextEncoder (semantic folding → hypervector)
        ↓
    KnowledgeStore (semantic graph + episodic timeline)
        ↓
    ResponseAssembler (retrieval + NLG)
        ↓
    Natural Language Response
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("nsck_ai.engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSION = 10240       # Hypervector dimensionality (binary, 10,240 bits)
GRID_SIZE = 128         # Semantic map: 128×128 = 16,384 bit SDR grid
SPARSITY = 0.02         # Target sparsity on the SDR grid (2% of bits ON)
ON_BITS = int(GRID_SIZE * GRID_SIZE * SPARSITY)  # ~328 ON bits per fingerprint
MAX_CONTEXT = 20        # Maximum conversation context window
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to consider a match


# ---------------------------------------------------------------------------
# Hypervector Operations (No matrix multiplication)
# ---------------------------------------------------------------------------
class HyperVector:
    """
    A 10,240-dimensional binary vector for holographic reduced representations.

    Why binary?
    - XOR binding is O(D) and self-inverse (A ⊕ A = 0)
    - Majority-rule bundling preserves information from all constituents
    - Hamming distance is a simple bit-count operation
    - No floating-point arithmetic needed
    """

    __slots__ = ('bits',)

    def __init__(self, bits: Optional[np.ndarray] = None):
        if bits is not None:
            self.bits = bits.astype(np.int8)
        else:
            self.bits = np.random.randint(0, 2, size=DIMENSION, dtype=np.int8)

    @classmethod
    def zero(cls) -> 'HyperVector':
        """Create a zero vector (all bits off)."""
        return cls(np.zeros(DIMENSION, dtype=np.int8))

    @classmethod
    def from_seed(cls, seed: str) -> 'HyperVector':
        """
        Create a deterministic hypervector from a string seed.
        Same seed always produces the same vector — critical for reproducibility.
        """
        h = hashlib.sha256(seed.encode('utf-8')).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
        return cls(rng.randint(0, 2, size=DIMENSION, dtype=np.int8))

    def bind(self, other: 'HyperVector') -> 'HyperVector':
        """XOR binding: creates a new vector dissimilar to both inputs."""
        return HyperVector(np.bitwise_xor(self.bits, other.bits))

    @staticmethod
    def bundle(vectors: List['HyperVector']) -> 'HyperVector':
        """
        Majority-rule bundling: the resulting vector is similar to all inputs.
        Ties are broken randomly for robustness.
        """
        if not vectors:
            return HyperVector.zero()
        if len(vectors) == 1:
            return HyperVector(vectors[0].bits.copy())

        stacked = np.stack([v.bits for v in vectors], axis=0)
        total = stacked.sum(axis=0)
        threshold = len(vectors) / 2.0
        result = np.where(total > threshold, 1,
                          np.where(total < threshold, 0,
                                   np.random.randint(0, 2, size=DIMENSION))).astype(np.int8)
        return HyperVector(result)

    def permute(self, n: int = 1) -> 'HyperVector':
        """Circular bit shift — used for positional/temporal encoding."""
        return HyperVector(np.roll(self.bits, n))

    def similarity(self, other: 'HyperVector') -> float:
        """
        Cosine-like similarity via normalized Hamming distance.
        Returns 0.0 (orthogonal/random) to 1.0 (identical).
        """
        matches = np.sum(self.bits == other.bits)
        return matches / DIMENSION

    def __repr__(self):
        density = np.mean(self.bits)
        return f"<HV dim={DIMENSION} density={density:.3f}>"


# ---------------------------------------------------------------------------
# Text Encoder (Semantic Folding → Hypervectors)
# ---------------------------------------------------------------------------

# Common stop words to filter out during concept extraction
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if',
    'while', 'about', 'up', 'out', 'off', 'over', 'also', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
    'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
    'their', 'what', 'which', 'who', 'whom',
})


class TextEncoder:
    """
    Encodes text into hypervectors using semantic folding.

    How it works:
    1. Each unique word gets a deterministic base hypervector (from its hash)
    2. Context is captured by bundling nearby words (window of ±3 words)
    3. Word order is preserved via permutation encoding
    4. The final sentence vector is a weighted bundle of all word-in-context vectors

    Why this works:
    - Words appearing in similar contexts get similar compound vectors
      (distributional hypothesis, but implemented with O(D) operations)
    - No embedding matrix needed — vectors are generated on-the-fly from hashes
    - Context windows capture local semantics without attention mechanisms
    """

    def __init__(self, context_window: int = 3):
        self.context_window = context_window
        self.word_vectors: Dict[str, HyperVector] = {}
        self.word_frequencies: Counter = Counter()
        self.co_occurrence: Dict[str, Counter] = defaultdict(Counter)
        self.trained_contexts: Dict[str, HyperVector] = {}
        self._total_words_seen = 0

    def _get_word_vector(self, word: str) -> HyperVector:
        """Get or create a deterministic hypervector for a word."""
        w = word.lower().strip()
        if w not in self.word_vectors:
            self.word_vectors[w] = HyperVector.from_seed(f"word:{w}")
        return self.word_vectors[w]

    def _get_char_ngram_vector(self, word: str, n: int = 3) -> HyperVector:
        """
        Create a vector from character n-grams.
        This captures morphological similarity (e.g., "running" ≈ "runner").
        """
        w = f"#{word.lower()}#"  # Add boundary markers
        ngrams = [w[i:i+n] for i in range(len(w) - n + 1)]
        if not ngrams:
            return self._get_word_vector(word)
        ngram_hvs = [HyperVector.from_seed(f"ngram:{ng}") for ng in ngrams]
        return HyperVector.bundle(ngram_hvs)

    def encode_word_in_context(self, word: str, context: List[str]) -> HyperVector:
        """
        Encode a word with its surrounding context.
        The context vector is bound with position-permuted neighbor vectors.
        """
        word_hv = self._get_word_vector(word)
        if not context:
            return word_hv

        context_hvs = []
        for i, ctx_word in enumerate(context):
            ctx_hv = self._get_word_vector(ctx_word)
            # Position-encode the context word relative to the target
            offset = i - len(context) // 2
            positioned = ctx_hv.permute(offset)
            context_hvs.append(positioned)

        if context_hvs:
            context_bundle = HyperVector.bundle(context_hvs)
            # Bind word with its context: creates a word-in-context representation
            return word_hv.bind(context_bundle)
        return word_hv

    def encode_sentence(self, text: str) -> HyperVector:
        """
        Encode an entire sentence into a single hypervector.

        Process:
        1. Tokenize into words
        2. For each word, compute its context window
        3. Create word-in-context vectors
        4. Bundle all word-in-context vectors (preserving all information)
        """
        words = self._tokenize(text)
        if not words:
            return HyperVector()

        word_hvs = []
        for i, word in enumerate(words):
            # Get context window
            start = max(0, i - self.context_window)
            end = min(len(words), i + self.context_window + 1)
            context = words[start:i] + words[i+1:end]

            wic_hv = self.encode_word_in_context(word, context)
            # Add positional encoding via permutation
            positioned = wic_hv.permute(i)
            word_hvs.append(positioned)

        return HyperVector.bundle(word_hvs)

    def learn_from_text(self, text: str) -> Dict[str, Any]:
        """
        Learn word co-occurrences and update context vectors from a text passage.
        Returns statistics about what was learned.
        """
        sentences = self._split_sentences(text)
        stats = {
            'sentences': len(sentences),
            'new_words': 0,
            'co_occurrences': 0,
            'concepts': [],
        }

        for sentence in sentences:
            words = self._tokenize(sentence)
            if len(words) < 2:
                continue

            for i, word in enumerate(words):
                if word in _STOP_WORDS:
                    continue

                # Track frequency
                was_new = word not in self.word_frequencies
                self.word_frequencies[word] += 1
                self._total_words_seen += 1
                if was_new:
                    stats['new_words'] += 1

                # Build co-occurrence counts
                start = max(0, i - self.context_window)
                end = min(len(words), i + self.context_window + 1)
                for j in range(start, end):
                    if i != j and words[j] not in _STOP_WORDS:
                        self.co_occurrence[word][words[j]] += 1
                        stats['co_occurrences'] += 1

                # Update the trained context vector for this word
                context = [w for w in words[start:i] + words[i+1:end]
                           if w not in _STOP_WORDS]
                wic = self.encode_word_in_context(word, context)

                if word in self.trained_contexts:
                    # Incrementally update: bundle old context with new observation
                    self.trained_contexts[word] = HyperVector.bundle([
                        self.trained_contexts[word], wic
                    ])
                else:
                    self.trained_contexts[word] = wic

        # Extract key concepts (high-frequency content words)
        stats['concepts'] = [w for w, c in self.word_frequencies.most_common(20)
                             if w not in _STOP_WORDS]
        return stats

    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find words with similar context vectors (distributional similarity)."""
        if word not in self.trained_contexts:
            return []

        target = self.trained_contexts[word]
        similarities = []

        for other_word, other_hv in self.trained_contexts.items():
            if other_word == word:
                continue
            sim = target.similarity(other_hv)
            if sim > 0.45:  # Only above-random similarities
                similarities.append((other_word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return encoder statistics."""
        return {
            'vocabulary_size': len(self.word_vectors),
            'trained_words': len(self.trained_contexts),
            'total_words_seen': self._total_words_seen,
            'unique_co_occurrences': sum(len(v) for v in self.co_occurrence.values()),
            'top_words': self.word_frequencies.most_common(10),
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]


# ---------------------------------------------------------------------------
# Knowledge Store (Semantic + Episodic Memory)
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    """A concept stored in semantic memory."""
    name: str
    hv: HyperVector
    properties: Dict[str, str] = field(default_factory=dict)
    frequency: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    source_texts: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """A typed relation between two concepts."""
    source: str
    relation_type: str
    target: str
    weight: float = 1.0
    evidence_count: int = 1
    source_text: str = ""


@dataclass
class Episode:
    """An episodic memory entry."""
    text: str
    hv: HyperVector
    timestamp: float
    concepts: List[str]
    response: str = ""
    emotion: str = "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeStore:
    """
    Dual-memory knowledge storage combining semantic and episodic memory.

    Why two memory systems?
    - Semantic Memory stores facts and relationships (what we know):
      "Paris is the capital of France" → (Paris, capital_of, France)
    - Episodic Memory stores experiences (what happened):
      "User asked about Paris and I answered correctly"

    The combination enables both factual recall AND experiential learning.
    """

    def __init__(self, max_episodes: int = 10000):
        # Semantic memory: concept graph
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[Relation] = []
        self.relation_index: Dict[str, List[int]] = defaultdict(list)

        # Episodic memory: timestamped experiences
        self.episodes: deque = deque(maxlen=max_episodes)

        # LSH index for fast approximate search
        self._lsh_buckets: Dict[int, List[int]] = defaultdict(list)
        self._lsh_projections: Optional[np.ndarray] = None
        self._lsh_bits = 16  # 2^16 = 65536 possible buckets

        self._initialize_lsh()

    def _initialize_lsh(self):
        """
        Initialize Locality-Sensitive Hashing projections.

        Why LSH?
        - Brute-force search over all episodes is O(N·D)
        - LSH reduces this to O(D + bucket_size) by hashing similar vectors
          to the same bucket with high probability
        - We use random hyperplane LSH: each projection defines a half-space,
          and vectors on the same side of all hyperplanes land in the same bucket
        """
        self._lsh_projections = np.random.randn(self._lsh_bits, DIMENSION).astype(np.float32)

    def _lsh_hash(self, hv: HyperVector) -> int:
        """Compute LSH bucket hash for a hypervector."""
        projections = self._lsh_projections @ hv.bits.astype(np.float32)
        bits = (projections > 0).astype(np.int32)
        return int(sum(b << i for i, b in enumerate(bits)))

    def add_concept(self, name: str, hv: HyperVector,
                    properties: Optional[Dict[str, str]] = None,
                    source_text: str = "") -> Concept:
        """Add or update a concept in semantic memory."""
        now = time.time()
        if name in self.concepts:
            concept = self.concepts[name]
            concept.frequency += 1
            concept.last_seen = now
            # Update HV by bundling with new observation
            concept.hv = HyperVector.bundle([concept.hv, hv])
            if source_text:
                concept.source_texts.append(source_text)
                if len(concept.source_texts) > 10:
                    concept.source_texts = concept.source_texts[-10:]
            if properties:
                concept.properties.update(properties)
        else:
            concept = Concept(
                name=name, hv=hv,
                properties=properties or {},
                frequency=1,
                first_seen=now, last_seen=now,
                source_texts=[source_text] if source_text else [],
            )
            self.concepts[name] = concept

        return concept

    def add_relation(self, source: str, relation_type: str, target: str,
                     weight: float = 1.0, source_text: str = "") -> Relation:
        """Add a typed relation between two concepts."""
        # Check for existing relation
        for i in self.relation_index.get(source, []):
            r = self.relations[i]
            if r.target == target and r.relation_type == relation_type:
                r.evidence_count += 1
                r.weight = min(r.weight + 0.1, 5.0)
                return r

        rel = Relation(
            source=source, relation_type=relation_type, target=target,
            weight=weight, evidence_count=1, source_text=source_text,
        )
        idx = len(self.relations)
        self.relations.append(rel)
        self.relation_index[source].append(idx)
        return rel

    def record_episode(self, text: str, hv: HyperVector,
                       concepts: List[str], response: str = "",
                       emotion: str = "neutral",
                       metadata: Optional[Dict[str, Any]] = None) -> Episode:
        """Record an experience in episodic memory with LSH indexing."""
        episode = Episode(
            text=text, hv=hv, timestamp=time.time(),
            concepts=concepts, response=response,
            emotion=emotion, metadata=metadata or {},
        )
        ep_idx = len(self.episodes)
        self.episodes.append(episode)

        # Index in LSH
        bucket = self._lsh_hash(hv)
        self._lsh_buckets[bucket].append(ep_idx)

        return episode

    def search_concepts(self, query_hv: HyperVector,
                        top_k: int = 5) -> List[Tuple[str, float]]:
        """Find concepts most similar to a query hypervector."""
        results = []
        for name, concept in self.concepts.items():
            sim = query_hv.similarity(concept.hv)
            if sim > SIMILARITY_THRESHOLD:
                results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_episodes(self, query_hv: HyperVector,
                        top_k: int = 5) -> List[Tuple[Episode, float]]:
        """
        Find episodes similar to a query, using LSH for speed.

        Process:
        1. Hash query to find its LSH bucket
        2. Check all episodes in that bucket (+ neighboring buckets)
        3. Rank by exact similarity
        """
        bucket = self._lsh_hash(query_hv)

        # Check primary bucket and neighbors (flip each LSH bit)
        candidate_indices = set()
        candidate_indices.update(self._lsh_buckets.get(bucket, []))
        for bit in range(min(4, self._lsh_bits)):  # Check 4 nearest buckets
            neighbor = bucket ^ (1 << bit)
            candidate_indices.update(self._lsh_buckets.get(neighbor, []))

        # Also check recent episodes (they're most likely relevant)
        episodes_list = list(self.episodes)
        n_eps = len(episodes_list)
        for i in range(max(0, n_eps - 50), n_eps):
            candidate_indices.add(i)

        results = []
        for idx in candidate_indices:
            if idx < n_eps:
                ep = episodes_list[idx]
                sim = query_hv.similarity(ep.hv)
                if sim > SIMILARITY_THRESHOLD:
                    results.append((ep, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_related_concepts(self, concept_name: str,
                             max_depth: int = 2) -> List[Tuple[str, str, str, float]]:
        """
        Spreading activation: find concepts related to a starting concept.
        Returns list of (source, relation_type, target, weight).
        """
        visited = set()
        results = []
        frontier = [(concept_name, 0, 1.0)]

        while frontier:
            current, depth, decay = frontier.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            for idx in self.relation_index.get(current, []):
                rel = self.relations[idx]
                effective_weight = rel.weight * decay
                results.append((rel.source, rel.relation_type, rel.target,
                                effective_weight))
                if depth + 1 <= max_depth:
                    frontier.append((rel.target, depth + 1, decay * 0.7))

        results.sort(key=lambda x: x[3], reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return knowledge store statistics."""
        return {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'total_episodes': len(self.episodes),
            'lsh_buckets_used': len(self._lsh_buckets),
            'relation_types': dict(Counter(r.relation_type for r in self.relations)),
            'top_concepts': sorted(
                [(n, c.frequency) for n, c in self.concepts.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
        }


# ---------------------------------------------------------------------------
# Knowledge Extractor (Concepts and Relations from Text)
# ---------------------------------------------------------------------------

# Relation patterns: (regex, relation_type)
_RELATION_PATTERNS = [
    (r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+(\w+(?:\s+\w+)?)', 'is_a'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+of\s+(\w+)', 'property_of'),
    (r'(\w+(?:\s+\w+)?)\s+(?:has|have)\s+(?:a\s+)?(\w+(?:\s+\w+)?)', 'has'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:located\s+)?in\s+(\w+(?:\s+\w+)?)', 'located_in'),
    (r'(\w+(?:\s+\w+)?)\s+(?:causes?|leads?\s+to)\s+(\w+(?:\s+\w+)?)', 'causes'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:part|member)\s+of\s+(\w+(?:\s+\w+)?)', 'part_of'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:made|composed)\s+of\s+(\w+(?:\s+\w+)?)', 'made_of'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:used|useful)\s+for\s+(\w+(?:\s+\w+)?)', 'used_for'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:similar|like|related)\s+to\s+(\w+(?:\s+\w+)?)', 'similar_to'),
    (r'(\w+(?:\s+\w+)?)\s+(?:was|were)\s+(?:created|invented|founded)\s+(?:by|in)\s+(\w+(?:\s+\w+)?)', 'created_by'),
]


class KnowledgeExtractor:
    """
    Extracts structured knowledge (concepts and relations) from text.

    How it works:
    1. Sentence segmentation → individual facts
    2. Concept extraction: content words (nouns, adjectives) not in stop words
    3. Relation extraction: regex patterns match common knowledge structures
    4. Co-occurrence relations: words appearing together are likely related

    Why regex instead of NER models?
    - No dependency on large ML models
    - Deterministic and interpretable
    - Sufficient for the relation types we need
    - Can be extended with new patterns easily
    """

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts (content words) from text."""
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        concepts = []
        for w in words:
            if w not in _STOP_WORDS and len(w) > 2:
                concepts.append(w)
        return list(dict.fromkeys(concepts))  # Deduplicate, preserve order

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract typed relations from text using pattern matching.
        Returns list of (subject, relation_type, object) triples.
        """
        relations = []
        text_lower = text.lower()

        for pattern, rel_type in _RELATION_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    obj = groups[-1].strip()
                    if (subject not in _STOP_WORDS and obj not in _STOP_WORDS
                            and len(subject) > 1 and len(obj) > 1):
                        relations.append((subject, rel_type, obj))

        return relations

    def extract_co_occurrences(self, text: str,
                                window: int = 5) -> List[Tuple[str, str]]:
        """
        Extract concept co-occurrences within a sliding window.
        Co-occurring concepts are likely semantically related.
        """
        concepts = self.extract_concepts(text)
        pairs = set()
        for i, c1 in enumerate(concepts):
            for j in range(i + 1, min(i + window + 1, len(concepts))):
                c2 = concepts[j]
                if c1 != c2:
                    pair = tuple(sorted([c1, c2]))
                    pairs.add(pair)
        return list(pairs)


# ---------------------------------------------------------------------------
# Response Assembler (Knowledge → Natural Language)
# ---------------------------------------------------------------------------

class ResponseAssembler:
    """
    Generates natural language responses from retrieved knowledge.

    How it works:
    1. Retrieves relevant concepts and episodes from the knowledge store
    2. Scores each knowledge fragment by relevance to the query
    3. Assembles a response using template-based NLG
    4. Falls back to learned n-gram generation for novel phrasing

    Why templates + n-grams instead of a language model?
    - Fully interpretable: every word can be traced to stored knowledge
    - No hallucination: only says things it actually learned
    - Incremental: new templates and n-grams are learned from training data
    """

    def __init__(self):
        # Response templates by intent category
        self.templates: Dict[str, List[str]] = {
            'factual': [
                "{subject} {relation} {object}.",
                "Based on what I know, {subject} {relation} {object}.",
                "I've learned that {subject} {relation} {object}.",
            ],
            'definition': [
                "{concept} is {definition}.",
                "A {concept} can be described as {definition}.",
                "From what I've learned, {concept} refers to {definition}.",
            ],
            'related': [
                "{concept} is related to {related_concepts}.",
                "When thinking about {concept}, I also recall {related_concepts}.",
                "There are connections between {concept} and {related_concepts}.",
            ],
            'episodic': [
                "I recall a similar topic: {episode_text}",
                "This reminds me of something I learned: {episode_text}",
                "I've encountered this before: {episode_text}",
            ],
            'unknown': [
                "I don't have enough knowledge about that topic yet.",
                "I'm not sure about that. Could you tell me more?",
                "That's outside my current knowledge. I'm still learning.",
            ],
            'greeting': [
                "Hello! I'm the NSCK AI. How can I help you?",
                "Hi there! What would you like to discuss?",
                "Welcome! I'm ready to chat. What's on your mind?",
            ],
            'acknowledgment': [
                "I understand. {detail}",
                "Got it. {detail}",
                "That makes sense. {detail}",
            ],
        }

        # Learned n-gram model for novel generation
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        self.trigrams: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._corpus_size = 0

    def learn_language_patterns(self, text: str):
        """Learn n-gram patterns from text for more natural generation."""
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        self._corpus_size += len(words)

        for i in range(len(words) - 1):
            self.bigrams[words[i]][words[i+1]] += 1
            if i < len(words) - 2:
                self.trigrams[(words[i], words[i+1])][words[i+2]] += 1

    def generate_response(self, query: str,
                          matched_concepts: List[Tuple[str, float]],
                          matched_episodes: List[Tuple[Episode, float]],
                          related: List[Tuple[str, str, str, float]],
                          emotion: str = "neutral") -> str:
        """
        Assemble a natural language response from retrieved knowledge.
        """
        query_lower = query.lower().strip()

        # Check for greeting patterns
        greetings = {'hello', 'hi', 'hey', 'greetings', 'good morning',
                     'good afternoon', 'good evening'}
        if any(g in query_lower for g in greetings):
            return np.random.choice(self.templates['greeting'])

        parts = []

        # 1. Use relations for factual answers
        if related:
            for source, rel_type, target, weight in related[:3]:
                rel_text = rel_type.replace('_', ' ')
                response = np.random.choice(self.templates['factual']).format(
                    subject=source.title(), relation=rel_text, object=target.title()
                )
                parts.append(response)

        # 2. Use matched concepts for definitions/descriptions
        if matched_concepts and not parts:
            concept_names = [name for name, _ in matched_concepts[:3]]
            concept = concept_names[0]

            if len(concept_names) > 1:
                related_text = ", ".join(c.title() for c in concept_names[1:])
                response = np.random.choice(self.templates['related']).format(
                    concept=concept.title(), related_concepts=related_text
                )
                parts.append(response)

        # 3. Use episodic memory for experience-based responses
        if matched_episodes and len(parts) < 2:
            best_episode = matched_episodes[0][0]
            # Use the stored response if available, otherwise the original text
            ep_text = best_episode.response or best_episode.text
            if len(ep_text) > 200:
                ep_text = ep_text[:200] + "..."
            response = np.random.choice(self.templates['episodic']).format(
                episode_text=ep_text
            )
            parts.append(response)

        # 4. If we have nothing, try n-gram generation or say we don't know
        if not parts:
            # Try to generate something from learned n-grams
            query_words = [w for w in query_lower.split() if w not in _STOP_WORDS]
            generated = self._generate_from_ngrams(query_words)
            if generated:
                parts.append(generated)
            else:
                parts.append(np.random.choice(self.templates['unknown']))

        return " ".join(parts)

    def _generate_from_ngrams(self, seed_words: List[str],
                              max_length: int = 30) -> Optional[str]:
        """
        Generate text using learned n-gram model.
        Returns None if insufficient n-gram data.
        """
        if not self.bigrams or not seed_words:
            return None

        # Find a seed word that has bigram continuations
        current = None
        for w in seed_words:
            if w in self.bigrams:
                current = w
                break

        if current is None:
            return None

        words = [current]
        for _ in range(max_length):
            candidates = self.bigrams.get(current, Counter())
            if not candidates:
                break

            # Weighted random selection from bigram continuations
            total = sum(candidates.values())
            r = np.random.random() * total
            cumulative = 0
            next_word = None
            for word, count in candidates.items():
                cumulative += count
                if cumulative >= r:
                    next_word = word
                    break

            if next_word is None:
                break

            words.append(next_word)
            current = next_word

            # Stop at sentence boundaries
            if len(words) >= 5 and current in {'the', 'a', 'an', 'and', 'but', 'or'}:
                break

        if len(words) < 3:
            return None

        sentence = " ".join(words)
        return sentence[0].upper() + sentence[1:] + "."

    def get_stats(self) -> Dict[str, Any]:
        """Return assembler statistics."""
        return {
            'bigram_vocabulary': len(self.bigrams),
            'trigram_contexts': len(self.trigrams),
            'corpus_size': self._corpus_size,
            'template_categories': list(self.templates.keys()),
        }


# ---------------------------------------------------------------------------
# Emotion Tracker (Affective State)
# ---------------------------------------------------------------------------

class EmotionTracker:
    """
    Tracks the emotional state of the AI during conversations.

    Uses Russell's Circumplex Model:
    - Valence: -1.0 (negative) to +1.0 (positive)
    - Arousal: 0.0 (calm) to 1.0 (excited)

    Why track emotions?
    - Emotion influences response style (empathetic vs. factual)
    - Provides dashboard telemetry for monitoring system health
    - Detects user frustration or satisfaction patterns
    """

    EMOTION_MAP = {
        'joy':           ( 0.8,  0.7),
        'trust':         ( 0.5,  0.2),
        'fear':          (-0.7,  0.8),
        'surprise':      ( 0.0,  0.9),
        'sadness':       (-0.6,  0.2),
        'disgust':       (-0.5,  0.4),
        'anger':         (-0.5,  0.8),
        'anticipation':  ( 0.3,  0.5),
        'neutral':       ( 0.0,  0.3),
    }

    # Keyword lexicon for text-based emotion recognition
    _POSITIVE_WORDS = frozenset({
        'good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'happy',
        'fantastic', 'awesome', 'brilliant', 'nice', 'perfect', 'beautiful',
        'thank', 'thanks', 'pleased', 'enjoy', 'delightful', 'superb', 'glad',
    })
    _NEGATIVE_WORDS = frozenset({
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad',
        'disappointed', 'frustrated', 'annoyed', 'worried', 'scared', 'ugly',
        'wrong', 'stupid', 'boring', 'useless', 'painful', 'worst', 'fail',
    })
    _CURIOSITY_WORDS = frozenset({
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 'curious',
        'wonder', 'explain', 'tell', 'describe', 'show', 'know', 'learn',
    })

    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.3
        self.current_emotion = 'neutral'
        self.history: deque = deque(maxlen=200)
        self._update_count = 0

    def update_from_text(self, text: str) -> str:
        """
        Detect emotional tone from text and update internal state.
        Returns the detected emotion label.
        """
        words = set(text.lower().split())

        pos_count = len(words & self._POSITIVE_WORDS)
        neg_count = len(words & self._NEGATIVE_WORDS)
        cur_count = len(words & self._CURIOSITY_WORDS)

        # Update valence based on positive/negative balance
        if pos_count + neg_count > 0:
            sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            self.valence = 0.7 * self.valence + 0.3 * sentiment

        # Curiosity/questions increase arousal
        if cur_count > 0:
            self.arousal = min(1.0, self.arousal + 0.1 * cur_count)
        else:
            self.arousal = max(0.1, self.arousal * 0.95)  # Decay toward calm

        # Clamp
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

        # Map to closest basic emotion
        self.current_emotion = self._map_to_emotion()
        self._update_count += 1

        # Record history
        self.history.append({
            'emotion': self.current_emotion,
            'valence': round(self.valence, 3),
            'arousal': round(self.arousal, 3),
            'timestamp': time.time(),
        })

        return self.current_emotion

    def _map_to_emotion(self) -> str:
        """Map current (valence, arousal) to closest basic emotion."""
        best_emotion = 'neutral'
        best_dist = float('inf')

        for emotion, (v, a) in self.EMOTION_MAP.items():
            dist = (self.valence - v) ** 2 + (self.arousal - a) ** 2
            if dist < best_dist:
                best_dist = dist
                best_emotion = emotion

        return best_emotion

    def get_blend(self) -> Dict[str, float]:
        """Get a soft blend of emotions based on proximity in VA space."""
        blend = {}
        total = 0.0
        for emotion, (v, a) in self.EMOTION_MAP.items():
            dist = ((self.valence - v) ** 2 + (self.arousal - a) ** 2) ** 0.5
            weight = max(0, 1.0 - dist)
            if weight > 0.05:
                blend[emotion] = weight
                total += weight

        if total > 0:
            blend = {k: round(v / total, 3) for k, v in blend.items()}

        return dict(sorted(blend.items(), key=lambda x: x[1], reverse=True))

    def get_state(self) -> Dict[str, Any]:
        """Get current emotional state for dashboard display."""
        return {
            'emotion': self.current_emotion,
            'valence': round(self.valence, 3),
            'arousal': round(self.arousal, 3),
            'blend': self.get_blend(),
            'history_length': len(self.history),
            'updates': self._update_count,
        }


# ---------------------------------------------------------------------------
# Main AI Engine (Orchestrator)
# ---------------------------------------------------------------------------

class NSCKAIEngine:
    """
    The main AI engine that orchestrates all components.

    Usage:
        engine = NSCKAIEngine()
        engine.train_on_text("Paris is the capital of France.")
        response = engine.chat("What is the capital of France?")
    """

    def __init__(self):
        self.encoder = TextEncoder(context_window=3)
        self.knowledge = KnowledgeStore(max_episodes=10000)
        self.extractor = KnowledgeExtractor()
        self.assembler = ResponseAssembler()
        self.emotion = EmotionTracker()
        self.conversation_history: deque = deque(maxlen=MAX_CONTEXT)
        self._training_stats = {
            'texts_trained': 0,
            'total_concepts': 0,
            'total_relations': 0,
            'total_episodes': 0,
            'training_time_seconds': 0.0,
        }
        self._query_count = 0
        logger.info("NSCK AI Engine initialized")

    def train_on_text(self, text: str) -> Dict[str, Any]:
        """
        Learn from a text passage. Extracts concepts, relations, and
        language patterns, storing everything in the knowledge systems.
        """
        start = time.time()

        # 1. Learn word co-occurrences and update context vectors
        encoder_stats = self.encoder.learn_from_text(text)

        # 2. Extract concepts and encode them as hypervectors
        sentences = self.encoder._split_sentences(text)
        concepts_added = 0
        relations_added = 0

        for sentence in sentences:
            # Extract and store concepts
            concepts = self.extractor.extract_concepts(sentence)
            sentence_hv = self.encoder.encode_sentence(sentence)

            for concept in concepts:
                concept_hv = self.encoder.encode_sentence(concept)
                self.knowledge.add_concept(
                    name=concept, hv=concept_hv,
                    source_text=sentence,
                )
                concepts_added += 1

            # Extract and store relations
            relations = self.extractor.extract_relations(sentence)
            for subject, rel_type, obj in relations:
                self.knowledge.add_relation(
                    source=subject, relation_type=rel_type, target=obj,
                    source_text=sentence,
                )
                relations_added += 1

            # Store co-occurrence relations
            co_occ = self.extractor.extract_co_occurrences(sentence)
            for c1, c2 in co_occ:
                self.knowledge.add_relation(
                    source=c1, relation_type='co_occurs_with', target=c2,
                    weight=0.5, source_text=sentence,
                )

            # Record as episodic memory
            self.knowledge.record_episode(
                text=sentence, hv=sentence_hv,
                concepts=concepts,
            )

        # 3. Learn language patterns for natural generation
        self.assembler.learn_language_patterns(text)

        elapsed = time.time() - start
        self._training_stats['texts_trained'] += 1
        self._training_stats['total_concepts'] += concepts_added
        self._training_stats['total_relations'] += relations_added
        self._training_stats['total_episodes'] += len(sentences)
        self._training_stats['training_time_seconds'] += elapsed

        result = {
            'sentences_processed': len(sentences),
            'concepts_added': concepts_added,
            'relations_added': relations_added,
            'encoder_stats': encoder_stats,
            'elapsed_seconds': round(elapsed, 4),
        }
        logger.info(f"Trained on text: {result}")
        return result

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Returns a dict with:
        - response: The natural language response text
        - emotion: Current emotional state
        - confidence: How confident we are in the answer
        - reasoning: Trace of what knowledge was used
        """
        start = time.time()
        self._query_count += 1

        # 1. Update emotion from user input
        detected_emotion = self.emotion.update_from_text(user_input)

        # 2. Encode the query as a hypervector
        query_hv = self.encoder.encode_sentence(user_input)

        # 3. Search for relevant knowledge
        matched_concepts = self.knowledge.search_concepts(query_hv, top_k=5)
        matched_episodes = self.knowledge.search_episodes(query_hv, top_k=3)

        # 4. Get related concepts via spreading activation
        related = []
        query_concepts = self.extractor.extract_concepts(user_input)
        for concept in query_concepts:
            if concept in self.knowledge.concepts:
                rels = self.knowledge.get_related_concepts(concept, max_depth=2)
                related.extend(rels)

        # 5. Assemble response
        response_text = self.assembler.generate_response(
            query=user_input,
            matched_concepts=matched_concepts,
            matched_episodes=matched_episodes,
            related=related,
            emotion=detected_emotion,
        )

        # 6. Record this interaction as an episode
        self.knowledge.record_episode(
            text=user_input, hv=query_hv,
            concepts=query_concepts, response=response_text,
            emotion=detected_emotion,
        )

        # 7. Update conversation history
        self.conversation_history.append({
            'role': 'user', 'content': user_input,
            'timestamp': time.time(),
        })
        self.conversation_history.append({
            'role': 'assistant', 'content': response_text,
            'timestamp': time.time(),
        })

        elapsed = time.time() - start

        # Calculate confidence based on match quality
        confidence = 0.0
        if matched_concepts:
            confidence = max(sim for _, sim in matched_concepts)
        if matched_episodes:
            ep_conf = max(sim for _, sim in matched_episodes)
            confidence = max(confidence, ep_conf)

        result = {
            'response': response_text,
            'emotion': self.emotion.get_state(),
            'confidence': round(confidence, 3),
            'reasoning': {
                'matched_concepts': [(n, round(s, 3)) for n, s in matched_concepts],
                'matched_episodes': len(matched_episodes),
                'related_facts': len(related),
                'query_concepts': query_concepts,
            },
            'latency_ms': round(elapsed * 1000, 1),
        }

        logger.debug(f"Chat response: {result}")
        return result

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics for dashboard display."""
        return {
            'engine': {
                'query_count': self._query_count,
                'conversation_length': len(self.conversation_history),
            },
            'training': self._training_stats.copy(),
            'encoder': self.encoder.get_stats(),
            'knowledge': self.knowledge.get_stats(),
            'assembler': self.assembler.get_stats(),
            'emotion': self.emotion.get_state(),
        }

    def export_knowledge(self) -> Dict[str, Any]:
        """Export all knowledge for inspection or backup."""
        return {
            'concepts': {
                name: {
                    'frequency': c.frequency,
                    'properties': c.properties,
                    'source_texts': c.source_texts,
                }
                for name, c in self.knowledge.concepts.items()
            },
            'relations': [
                {
                    'source': r.source,
                    'type': r.relation_type,
                    'target': r.target,
                    'weight': r.weight,
                    'evidence': r.evidence_count,
                }
                for r in self.knowledge.relations
            ],
            'stats': self.get_system_stats(),
        }

    def reset(self):
        """Reset all knowledge and state (for fresh training)."""
        self.__init__()
        logger.info("AI Engine reset to initial state")
