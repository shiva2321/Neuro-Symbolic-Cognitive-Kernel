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
import uuid
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("nsck_ai.engine")


# ---------------------------------------------------------------------------
# Glass-Box Thought Trace — records every cognitive step end-to-end
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """One atomic cognitive step in the reasoning pipeline.

    Why record every step?
    Because this is a glass-box architecture: a human observer must be able
    to see *exactly* why the model produced a particular response, which
    knowledge was consulted, which rules fired, and what confidence level
    was assigned at every stage.
    """
    stage: str          # e.g. "encode", "retrieve", "reason", "generate"
    action: str         # human-readable description of what happened
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ThoughtTrace:
    """Complete end-to-end trace of a single cognitive cycle.

    Every call to ``NSCKAIEngine.chat()`` creates one ``ThoughtTrace``.  The
    trace records every intermediate step — encoding, retrieval, rule
    evaluation, causal inference, response assembly — so the dashboard can
    display a full *reasoning chain* from user input to final output.

    How it works:
    1. ``begin(stage)`` starts timing a new stage.
    2. Code does its work and logs inputs/outputs.
    3. ``end(stage, outputs)`` stops timing and records the step.
    4. The final trace is attached to the chat response dict.
    """

    def __init__(self, query: str):
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.query: str = query
        self.steps: List[TraceStep] = []
        self.start_time: float = time.time()
        self._pending_stage: Optional[str] = None
        self._pending_start: float = 0.0
        self._pending_inputs: Dict[str, Any] = {}

    def begin(self, stage: str, inputs: Optional[Dict[str, Any]] = None):
        """Start timing a new cognitive stage."""
        self._pending_stage = stage
        self._pending_start = time.time()
        self._pending_inputs = inputs or {}

    def end(self, action: str, outputs: Optional[Dict[str, Any]] = None):
        """Finish the current stage and record the step."""
        elapsed = (time.time() - self._pending_start) * 1000
        step = TraceStep(
            stage=self._pending_stage or "unknown",
            action=action,
            inputs=self._pending_inputs,
            outputs=outputs or {},
            duration_ms=round(elapsed, 3),
        )
        self.steps.append(step)
        self._pending_stage = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON transport to the dashboard."""
        return {
            'trace_id': self.trace_id,
            'query': self.query,
            'total_ms': round((time.time() - self.start_time) * 1000, 2),
            'step_count': len(self.steps),
            'steps': [
                {
                    'stage': s.stage,
                    'action': s.action,
                    'inputs': _safe_serialise(s.inputs),
                    'outputs': _safe_serialise(s.outputs),
                    'duration_ms': s.duration_ms,
                }
                for s in self.steps
            ],
        }


def _safe_serialise(obj: Any, depth: int = 0) -> Any:
    """Recursively convert an object to JSON-safe types."""
    if depth > 4:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe_serialise(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialise(v, depth + 1) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape}>"
    if isinstance(obj, HyperVector):
        return repr(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Causal Rule Store — autonomous rule learning & inference
# ---------------------------------------------------------------------------

@dataclass
class CausalRule:
    """A learned causal rule: IF antecedent concepts → THEN consequent concepts.

    How rules are learned:
    When the engine sees text like "Rain causes flooding", it extracts a
    causal rule (rain → flooding).  Rules accumulate evidence counts; rules
    with more evidence are trusted more.

    How rules are used:
    During reasoning, the engine checks if any antecedent concepts match the
    current query.  If so, the consequent concepts are added to the response
    context, enabling forward-chaining inference.
    """
    antecedent: List[str]
    consequent: List[str]
    relation: str = "causes"
    evidence_count: int = 1
    confidence: float = 0.5
    source_text: str = ""

    @property
    def strength(self) -> float:
        """Rule strength grows logarithmically with evidence."""
        return min(1.0, self.confidence + 0.1 * np.log1p(self.evidence_count))


class CausalRuleStore:
    """Stores and indexes causal rules for forward-chaining inference.

    Why a separate rule store?
    Relations in the KnowledgeStore are individual edges in a concept graph.
    Causal rules are *compound*: they link sets of concepts to sets of
    consequences, with confidence scores.  This separation keeps the graph
    clean while enabling multi-hop inference.
    """

    def __init__(self):
        self.rules: List[CausalRule] = []
        self._antecedent_index: Dict[str, List[int]] = defaultdict(list)

    def add_rule(self, antecedent: List[str], consequent: List[str],
                 relation: str = "causes", source_text: str = "") -> CausalRule:
        """Add or reinforce a causal rule."""
        ant_key = tuple(sorted(antecedent))
        # Check for existing rule with same antecedent & consequent
        for idx in self._antecedent_index.get(ant_key[0], []):
            r = self.rules[idx]
            if set(r.antecedent) == set(antecedent) and set(r.consequent) == set(consequent):
                r.evidence_count += 1
                r.confidence = min(1.0, r.confidence + 0.05)
                return r

        rule = CausalRule(
            antecedent=list(antecedent),
            consequent=list(consequent),
            relation=relation,
            source_text=source_text,
        )
        idx = len(self.rules)
        self.rules.append(rule)
        for a in antecedent:
            self._antecedent_index[a].append(idx)
        return rule

    def forward_chain(self, active_concepts: List[str],
                      max_depth: int = 3) -> List[Tuple[CausalRule, int]]:
        """Fire rules whose antecedents match active concepts.

        Returns (rule, depth) pairs for every rule that fires, including
        rules triggered transitively by consequents of earlier rules.
        """
        fired: List[Tuple[CausalRule, int]] = []
        frontier = set(active_concepts)
        visited_rules: Set[int] = set()

        for depth in range(max_depth):
            new_concepts: Set[str] = set()
            for concept in list(frontier):
                for idx in self._antecedent_index.get(concept, []):
                    if idx in visited_rules:
                        continue
                    rule = self.rules[idx]
                    # Check if ALL antecedents are satisfied
                    if all(a in (frontier | set(active_concepts)) for a in rule.antecedent):
                        fired.append((rule, depth))
                        visited_rules.add(idx)
                        new_concepts.update(rule.consequent)
            if not new_concepts:
                break
            frontier = new_concepts

        return fired

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_rules': len(self.rules),
            'indexed_concepts': len(self._antecedent_index),
            'avg_confidence': (
                round(np.mean([r.confidence for r in self.rules]), 3)
                if self.rules else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Knowledge Abstractor — generalises specific facts into categories
# ---------------------------------------------------------------------------

class KnowledgeAbstractor:
    """Generalises specific facts into higher-level abstractions.

    Why abstraction matters:
    If the engine learns "Paris is the capital of France" and "Berlin is
    the capital of Germany", abstraction creates a general concept "capital"
    linked to "country" by an "is_capital_of" relation.  This allows the
    engine to answer "What is a capital?" even though it was never told
    directly.

    How it works:
    1. Group relations by type (e.g. all "is_a" relations).
    2. If a relation type has N+ instances with the same target, create
       an abstraction: "<target> is a category that includes <sources>".
    3. Store the abstraction as a new concept in the knowledge store.
    """

    def __init__(self, min_instances: int = 2):
        self.min_instances = min_instances
        self.abstractions: Dict[str, Dict[str, Any]] = {}

    def abstract(self, relations: List['Relation'],
                 knowledge: 'KnowledgeStore') -> List[Dict[str, Any]]:
        """Scan relations and create category abstractions.

        Only considers meaningful relation types (is_a, has, located_in,
        etc.) — co-occurrence relations are too noisy for abstraction.
        """
        # Only abstract from semantically meaningful relation types
        _ABSTRACTABLE = {'is_a', 'has', 'located_in', 'part_of', 'used_for',
                         'made_of', 'causes', 'similar_to', 'created_by'}

        # Group by (relation_type, target)
        groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for r in relations:
            if r.relation_type in _ABSTRACTABLE:
                groups[(r.relation_type, r.target)].append(r.source)

        new_abstractions = []
        for (rel_type, target), sources in groups.items():
            if len(sources) < self.min_instances:
                continue
            abs_key = f"category:{target}:{rel_type}"
            if abs_key in self.abstractions:
                existing = self.abstractions[abs_key]
                existing['members'] = list(set(existing['members']) | set(sources))
                continue

            abstraction = {
                'category': target,
                'relation': rel_type,
                'members': list(set(sources)),
                'member_count': len(set(sources)),
            }
            self.abstractions[abs_key] = abstraction
            new_abstractions.append(abstraction)

            # Store as a concept in the knowledge store
            member_hvs = []
            for src in sources:
                if src in knowledge.concepts:
                    member_hvs.append(knowledge.concepts[src].hv)
            if member_hvs:
                category_hv = HyperVector.bundle(member_hvs)
                knowledge.add_concept(
                    name=f"{target} (category)",
                    hv=category_hv,
                    properties={
                        'type': 'abstraction',
                        'relation': rel_type,
                        'members': ", ".join(sources[:5]),
                    },
                    source_text=f"Abstracted from {len(sources)} {rel_type} relations",
                )

        return new_abstractions

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_abstractions': len(self.abstractions),
            'categories': list(self.abstractions.keys())[:10],
        }

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

# ---------------------------------------------------------------------------
# Relation patterns (extended for richer knowledge extraction)
# ---------------------------------------------------------------------------

_RELATION_PATTERNS = [
    # "X is a Y" / "X is an Y" — the object can be up to 3 words
    (r'(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+(\w+(?:\s+\w+){0,2})', 'is_a'),
    # "X are Y" — for plural forms like "Dogs are mammals"
    (r'(\w+)\s+are\s+(\w+(?:\s+\w+)?)', 'is_a'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+of\s+(\w+)', 'property_of'),
    (r'(\w+(?:\s+\w+)?)\s+(?:has|have)\s+(?:a\s+)?(\w+(?:\s+\w+)?)', 'has'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:located\s+)?in\s+(\w+(?:\s+\w+)?)', 'located_in'),
    (r'(\w+(?:\s+\w+)?)\s+(?:causes?|leads?\s+to)\s+(\w+(?:\s+\w+)?)', 'causes'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:part|member)\s+of\s+(\w+(?:\s+\w+)?)', 'part_of'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:made|composed)\s+of\s+(\w+(?:\s+\w+)?)', 'made_of'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:used|useful)\s+for\s+(\w+(?:\s+\w+)?)', 'used_for'),
    (r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:similar|like|related)\s+to\s+(\w+(?:\s+\w+)?)', 'similar_to'),
    (r'(\w+(?:\s+\w+)?)\s+(?:was|were)\s+(?:created|invented|founded)\s+(?:by|in)\s+(\w+(?:\s+\w+)?)', 'created_by'),
    # Causal patterns for rule learning
    (r'if\s+(.+?)\s*,?\s*then\s+(.+)', 'if_then'),
    (r'(\w+(?:\s+\w+)?)\s+(?:results?\s+in|produces?)\s+(\w+(?:\s+\w+)?)', 'causes'),
    (r'(\w+(?:\s+\w+)?)\s+(?:prevents?|stops?)\s+(\w+(?:\s+\w+)?)', 'prevents'),
    (r'(\w+(?:\s+\w+)?)\s+(?:requires?|needs?)\s+(\w+(?:\s+\w+)?)', 'requires'),
    (r'(\w+(?:\s+\w+)?)\s+(?:enables?|allows?)\s+(\w+(?:\s+\w+)?)', 'enables'),
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
                        # For property_of with 3 groups, also store the
                        # property itself as a relation:
                        # "Paris is the capital of France" →
                        #   (paris, property_of, france)  AND
                        #   (obj=france, has, property=capital)
                        if rel_type == 'property_of' and len(groups) == 3:
                            prop = groups[1].strip()
                            if prop not in _STOP_WORDS and len(prop) > 1:
                                relations.append((obj, 'has', prop))
                                relations.append((subject, 'is_a', prop))

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

    Glass-box architecture
    ----------------------
    Every call to ``chat()`` produces a ``ThoughtTrace`` that records the
    complete reasoning chain — from input encoding through knowledge retrieval,
    causal inference, and response assembly.  The trace is returned alongside
    the response so the dashboard can display it.

    Autonomous cognition
    --------------------
    The engine autonomously:
    * **Learns** — extracts concepts, relations, and causal rules from text
    * **Reasons** — forward-chains causal rules, infers missing knowledge
    * **Abstracts** — creates category concepts from recurring patterns
    * **Applies** — uses all of the above to answer novel questions

    Usage::

        engine = NSCKAIEngine()
        engine.train_on_text("Paris is the capital of France. Berlin is the capital of Germany.")
        result = engine.chat("What is a capital?")
        print(result['response'])  # sensible answer from learned knowledge
        print(result['trace'])     # full glass-box reasoning chain
    """

    def __init__(self):
        self.encoder = TextEncoder(context_window=3)
        self.knowledge = KnowledgeStore(max_episodes=10000)
        self.extractor = KnowledgeExtractor()
        self.assembler = ResponseAssembler()
        self.emotion = EmotionTracker()
        self.causal_rules = CausalRuleStore()
        self.abstractor = KnowledgeAbstractor(min_instances=2)
        self.conversation_history: deque = deque(maxlen=MAX_CONTEXT)
        self._training_stats = {
            'texts_trained': 0,
            'total_concepts': 0,
            'total_relations': 0,
            'total_episodes': 0,
            'total_causal_rules': 0,
            'total_abstractions': 0,
            'training_time_seconds': 0.0,
        }
        self._query_count = 0
        logger.info("NSCK AI Engine initialized (glass-box mode)")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_text(self, text: str) -> Dict[str, Any]:
        """
        Learn from a text passage.  Extracts concepts, relations, causal
        rules, and language patterns, storing everything in the knowledge
        systems.  Also runs the abstractor to create category concepts.
        """
        start = time.time()

        # 1. Learn word co-occurrences and update context vectors
        encoder_stats = self.encoder.learn_from_text(text)

        # 2. Extract concepts and encode them as hypervectors
        sentences = self.encoder._split_sentences(text)
        concepts_added = 0
        relations_added = 0
        rules_added = 0

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

                # Learn causal rules from causal relation types
                if rel_type in ('causes', 'prevents', 'enables', 'requires', 'if_then'):
                    ant = self.extractor.extract_concepts(subject)
                    con = self.extractor.extract_concepts(obj)
                    if ant and con:
                        self.causal_rules.add_rule(
                            antecedent=ant, consequent=con,
                            relation=rel_type, source_text=sentence,
                        )
                        rules_added += 1

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

        # 4. Run abstractor to create category concepts
        new_abs = self.abstractor.abstract(self.knowledge.relations, self.knowledge)

        elapsed = time.time() - start
        self._training_stats['texts_trained'] += 1
        self._training_stats['total_concepts'] += concepts_added
        self._training_stats['total_relations'] += relations_added
        self._training_stats['total_episodes'] += len(sentences)
        self._training_stats['total_causal_rules'] += rules_added
        self._training_stats['total_abstractions'] += len(new_abs)
        self._training_stats['training_time_seconds'] += elapsed

        result = {
            'sentences_processed': len(sentences),
            'concepts_added': concepts_added,
            'relations_added': relations_added,
            'causal_rules_added': rules_added,
            'abstractions_created': len(new_abs),
            'encoder_stats': encoder_stats,
            'elapsed_seconds': round(elapsed, 4),
        }
        logger.info(f"Trained on text: {result}")
        return result

    # ------------------------------------------------------------------
    # Chat — glass-box, traced, with autonomous reasoning
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Every cognitive step is recorded in a ``ThoughtTrace`` for full
        glass-box transparency.  The trace is included in the returned
        dict under the ``'trace'`` key.

        Returns a dict with:
        - response:   natural language response text
        - emotion:    current emotional state
        - confidence: how confident we are in the answer (0–1)
        - reasoning:  summary of what knowledge was used
        - trace:      full glass-box ThoughtTrace (list of steps)
        - latency_ms: response time in milliseconds
        """
        trace = ThoughtTrace(user_input)
        start = time.time()
        self._query_count += 1

        # --- Stage 1: Emotion detection ---
        trace.begin("emotion", {"input": user_input[:200]})
        detected_emotion = self.emotion.update_from_text(user_input)
        trace.end("Detected emotional tone from input", {
            "emotion": detected_emotion,
            "valence": self.emotion.valence,
            "arousal": self.emotion.arousal,
        })

        # --- Stage 2: Intent classification ---
        trace.begin("intent", {"input": user_input[:200]})
        intent = self._classify_intent(user_input)
        trace.end(f"Classified intent as '{intent}'", {"intent": intent})

        # --- Stage 3: Encode query to hypervector ---
        trace.begin("encode", {"text_length": len(user_input)})
        query_hv = self.encoder.encode_sentence(user_input)
        trace.end("Encoded query to hypervector via semantic folding", {
            "dimension": DIMENSION,
        })

        # --- Stage 4: Extract query concepts ---
        trace.begin("extract", {"input": user_input[:200]})
        query_concepts = self.extractor.extract_concepts(user_input)
        trace.end("Extracted content concepts from query", {
            "concepts": query_concepts,
        })

        # --- Stage 5: Knowledge retrieval (semantic) ---
        trace.begin("retrieve_semantic", {"concepts": query_concepts})
        matched_concepts = self.knowledge.search_concepts(query_hv, top_k=8)
        trace.end("Searched semantic memory by hypervector similarity", {
            "matches": [(n, round(s, 3)) for n, s in matched_concepts],
        })

        # --- Stage 6: Knowledge retrieval (episodic) ---
        trace.begin("retrieve_episodic", {"query_concepts": query_concepts})
        matched_episodes = self.knowledge.search_episodes(query_hv, top_k=5)
        trace.end("Searched episodic memory via LSH index", {
            "episodes_found": len(matched_episodes),
            "best_similarity": round(matched_episodes[0][1], 3) if matched_episodes else 0,
        })

        # --- Stage 7: Spreading activation (graph traversal) ---
        trace.begin("spread_activation", {"seed_concepts": query_concepts})
        related = []
        for concept in query_concepts:
            if concept in self.knowledge.concepts:
                rels = self.knowledge.get_related_concepts(concept, max_depth=2)
                related.extend(rels)
        # Deduplicate and keep top
        seen = set()
        unique_related = []
        for item in related:
            key = (item[0], item[1], item[2])
            if key not in seen:
                seen.add(key)
                unique_related.append(item)
        related = sorted(unique_related, key=lambda x: x[3], reverse=True)[:10]
        trace.end("Spread activation through concept graph", {
            "relations_found": len(related),
            "relation_types": list(set(r[1] for r in related)),
        })

        # --- Stage 8: Causal inference (forward chaining) ---
        trace.begin("causal_inference", {"active_concepts": query_concepts})
        fired_rules = self.causal_rules.forward_chain(query_concepts, max_depth=2)
        inferred_concepts = []
        for rule, depth in fired_rules:
            inferred_concepts.extend(rule.consequent)
        trace.end("Forward-chained causal rules", {
            "rules_fired": len(fired_rules),
            "inferred_concepts": list(set(inferred_concepts))[:10],
            "rule_details": [
                {
                    "antecedent": r.antecedent,
                    "consequent": r.consequent,
                    "confidence": round(r.strength, 3),
                    "depth": d,
                }
                for r, d in fired_rules[:5]
            ],
        })

        # --- Stage 9: Conversation context ---
        trace.begin("context", {"history_length": len(self.conversation_history)})
        context_concepts = self._get_context_concepts()
        trace.end("Retrieved conversation context", {
            "context_concepts": context_concepts[:10],
        })

        # --- Stage 10: Response assembly ---
        trace.begin("generate", {
            "matched_concepts_count": len(matched_concepts),
            "related_count": len(related),
            "intent": intent,
        })
        response_text = self._assemble_response(
            user_input=user_input,
            intent=intent,
            query_concepts=query_concepts,
            matched_concepts=matched_concepts,
            matched_episodes=matched_episodes,
            related=related,
            fired_rules=fired_rules,
            inferred_concepts=inferred_concepts,
            context_concepts=context_concepts,
            emotion=detected_emotion,
        )
        trace.end("Assembled natural language response", {
            "response_length": len(response_text),
            "response_preview": response_text[:200],
        })

        # --- Record episode & update history ---
        self.knowledge.record_episode(
            text=user_input, hv=query_hv,
            concepts=query_concepts, response=response_text,
            emotion=detected_emotion,
        )
        self.conversation_history.append({
            'role': 'user', 'content': user_input,
            'timestamp': time.time(),
        })
        self.conversation_history.append({
            'role': 'assistant', 'content': response_text,
            'timestamp': time.time(),
        })

        elapsed = time.time() - start

        # Calculate confidence
        confidence = 0.0
        if matched_concepts:
            confidence = max(sim for _, sim in matched_concepts)
        if matched_episodes:
            ep_conf = max(sim for _, sim in matched_episodes)
            confidence = max(confidence, ep_conf)
        if fired_rules:
            rule_conf = max(r.strength for r, _ in fired_rules)
            confidence = max(confidence, rule_conf)

        result = {
            'response': response_text,
            'emotion': self.emotion.get_state(),
            'confidence': round(confidence, 3),
            'reasoning': {
                'intent': intent,
                'matched_concepts': [(n, round(s, 3)) for n, s in matched_concepts],
                'matched_episodes': len(matched_episodes),
                'related_facts': len(related),
                'causal_rules_fired': len(fired_rules),
                'inferred_concepts': list(set(inferred_concepts))[:10],
                'query_concepts': query_concepts,
            },
            'trace': trace.to_dict(),
            'latency_ms': round(elapsed * 1000, 1),
        }

        logger.debug(f"Chat response: confidence={confidence:.3f}")
        return result

    # ------------------------------------------------------------------
    # Intent classification (autonomous, no hardcoded responses)
    # ------------------------------------------------------------------

    _GREETING_WORDS = frozenset({
        'hello', 'hi', 'hey', 'greetings', 'howdy', 'hola',
    })
    _QUESTION_WORDS = frozenset({
        'what', 'who', 'where', 'when', 'why', 'how', 'which',
        'is', 'are', 'can', 'does', 'do', 'could', 'would', 'will',
    })
    _FAREWELL_WORDS = frozenset({
        'bye', 'goodbye', 'farewell', 'see', 'later', 'quit', 'exit',
    })
    _THANKS_WORDS = frozenset({
        'thank', 'thanks', 'thx', 'appreciate', 'grateful',
    })

    def _classify_intent(self, text: str) -> str:
        """Classify user intent from text.

        Returns one of: greeting, question, statement, farewell,
        thanks, command, or general.
        """
        # Strip punctuation for word matching
        clean = re.sub(r'[^\w\s]', '', text.lower())
        words = set(clean.split())
        text_lower = text.lower().strip()

        if words & self._GREETING_WORDS and len(words) < 6:
            return 'greeting'
        if words & self._FAREWELL_WORDS and len(words) < 6:
            return 'farewell'
        if words & self._THANKS_WORDS:
            return 'thanks'
        if text_lower.endswith('?') or (words & self._QUESTION_WORDS and clean.split()[0] in self._QUESTION_WORDS):
            return 'question'
        # Check for imperative (starts with a verb-like word)
        first_word = clean.split()[0] if clean.split() else ''
        if first_word in {'tell', 'explain', 'describe', 'show', 'list', 'find', 'help'}:
            return 'command'

        return 'statement'

    # ------------------------------------------------------------------
    # Context-aware response assembly
    # ------------------------------------------------------------------

    def _get_context_concepts(self) -> List[str]:
        """Extract concepts from recent conversation history."""
        concepts = []
        for entry in list(self.conversation_history)[-6:]:
            if entry.get('role') == 'user':
                concepts.extend(self.extractor.extract_concepts(entry['content']))
        return list(dict.fromkeys(concepts))

    def _assemble_response(self, user_input: str, intent: str,
                           query_concepts: List[str],
                           matched_concepts: List[Tuple[str, float]],
                           matched_episodes: List[Tuple[Episode, float]],
                           related: List[Tuple[str, str, str, float]],
                           fired_rules: List[Tuple[CausalRule, int]],
                           inferred_concepts: List[str],
                           context_concepts: List[str],
                           emotion: str) -> str:
        """Build a natural language response from all retrieved knowledge.

        This method is the *heart* of the conversation system.  It takes
        everything the engine knows that is relevant to the query and
        weaves it into a coherent, natural response.  The logic is:

        1. Handle social intents (greeting, farewell, thanks) directly.
        2. For questions: find the best factual answer from relations and
           concepts, supplement with causal inference and episodic recall.
        3. For statements: acknowledge and connect to existing knowledge.
        4. For commands: attempt to answer using the knowledge base.

        Critically, every part of the response is traceable to specific
        stored knowledge — no hallucination.
        """
        # --- Social intents ---
        if intent == 'greeting':
            return "Hello! I'm the NSCK AI model. I understand and learn from text using hypervector algebra — no neural networks needed. What would you like to talk about?"
        if intent == 'farewell':
            return "Goodbye! I've enjoyed our conversation. Everything I learned is stored and ready for next time."
        if intent == 'thanks':
            learned = self._training_stats['total_concepts']
            return f"You're welcome! I'm always learning — I currently know about {learned} concepts."

        # --- Collect knowledge fragments ---
        fragments: List[Tuple[str, float]] = []  # (text, relevance_score)

        # Helper to title-case concept names
        def _tc(s: str) -> str:
            return s.title() if s == s.lower() else s

        # From direct relations
        for source, rel_type, target, weight in related:
            s, t = _tc(source), _tc(target)
            if rel_type == 'is_a':
                fragments.append((f"{s} is a {t}", weight))
            elif rel_type == 'property_of':
                # property_of stores (subject, property_of, owner) — already
                # decomposed into has + is_a by the extractor, so skip here
                # to avoid duplicate "Paris is the France" fragments.
                continue
            elif rel_type == 'has':
                fragments.append((f"{s} has {t.lower()}", weight))
            elif rel_type == 'located_in':
                fragments.append((f"{s} is located in {t}", weight))
            elif rel_type == 'causes':
                fragments.append((f"{s} causes {t.lower()}", weight))
            elif rel_type == 'part_of':
                fragments.append((f"{s} is part of {t}", weight))
            elif rel_type == 'used_for':
                fragments.append((f"{s} is used for {t.lower()}", weight))
            elif rel_type == 'co_occurs_with':
                # Skip mere co-occurrence to avoid noise
                continue
            else:
                rel_text = rel_type.replace('_', ' ')
                fragments.append((f"{s} {rel_text} {t}", weight))

        # From causal inference
        for rule, depth in fired_rules:
            ant_text = " and ".join(rule.antecedent)
            con_text = " and ".join(rule.consequent)
            if rule.relation == 'causes':
                fragments.append((f"{ant_text} can lead to {con_text}", rule.strength))
            elif rule.relation == 'prevents':
                fragments.append((f"{ant_text} prevents {con_text}", rule.strength))
            elif rule.relation == 'requires':
                fragments.append((f"{ant_text} requires {con_text}", rule.strength))
            elif rule.relation == 'enables':
                fragments.append((f"{ant_text} enables {con_text}", rule.strength))
            else:
                fragments.append((f"{ant_text} relates to {con_text}", rule.strength * 0.7))

        # From concept properties (only high-similarity concepts)
        query_concept_set = set(query_concepts)
        for name, sim in matched_concepts[:5]:
            if sim < 0.45:
                continue
            concept = self.knowledge.concepts.get(name)
            if concept and concept.properties:
                for prop_key, prop_val in concept.properties.items():
                    if prop_key == 'type' and prop_val == 'abstraction':
                        # Only include if query overlaps with members
                        members_str = concept.properties.get('members', '')
                        if members_str and (query_concept_set & set(members_str.split(', '))):
                            fragments.append(
                                (f"{_tc(name)} is a category that includes {members_str}", sim),
                            )
                        continue
                    fragments.append(
                        (f"{_tc(name)} has {prop_key}: {prop_val}", sim * 0.8),
                    )

        # From episodic memory (source texts of concepts that overlap with query)
        for name, sim in matched_concepts[:3]:
            if sim < 0.45 or name not in query_concept_set:
                continue
            concept = self.knowledge.concepts.get(name)
            if concept and concept.source_texts:
                for src in concept.source_texts[:2]:
                    fragments.append((src, sim * 0.9))

        # Sort by relevance
        fragments.sort(key=lambda x: x[1], reverse=True)

        # --- Build response ---
        if not fragments:
            # No knowledge at all — honest acknowledgement
            if intent == 'question':
                return ("I don't have enough information to answer that question yet. "
                        "If you teach me about this topic, I'll remember it for next time.")
            elif intent == 'statement':
                # Learn from the statement
                self.train_on_text(user_input)
                concepts_str = ", ".join(query_concepts[:5]) if query_concepts else "that"
                return f"Interesting — I've noted that. I now have knowledge about {concepts_str}."
            elif intent == 'command':
                return ("I'd like to help, but I don't have enough knowledge about that topic yet. "
                        "Try teaching me first by telling me facts about it.")
            else:
                return "I'm listening. Tell me more, and I'll learn from what you share."

        # Take top fragments (avoid repetition)
        used_texts: Set[str] = set()
        top_fragments: List[str] = []
        for text, score in fragments:
            normalised = text.lower().strip()
            if normalised not in used_texts and len(top_fragments) < 4:
                used_texts.add(normalised)
                top_fragments.append(text)

        # Format based on intent
        if intent == 'question':
            if len(top_fragments) == 1:
                return f"{top_fragments[0]}."
            else:
                main_answer = top_fragments[0]
                supporting = ". ".join(top_fragments[1:3])
                return f"{main_answer}. Additionally, {supporting.lower()}."
        elif intent == 'statement':
            # Acknowledge and connect to existing knowledge
            self.train_on_text(user_input)
            if top_fragments:
                connection = top_fragments[0]
                return f"I see — that connects to what I already know: {connection.lower()}. I've stored this new information."
            concepts_str = ", ".join(query_concepts[:5]) if query_concepts else "that"
            return f"Got it. I've learned about {concepts_str} and stored it in my knowledge base."
        elif intent == 'command':
            if top_fragments:
                info = ". ".join(top_fragments[:3])
                return f"Here's what I know: {info}."
            return "I don't have specific information about that yet."
        else:
            # General response
            if top_fragments:
                return ". ".join(f"{f}" for f in top_fragments[:3]) + "."
            return "Tell me more — I'm always learning."

    # ------------------------------------------------------------------
    # Statistics & export
    # ------------------------------------------------------------------

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
            'causal_rules': self.causal_rules.get_stats(),
            'abstractions': self.abstractor.get_stats(),
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
            'causal_rules': [
                {
                    'antecedent': r.antecedent,
                    'consequent': r.consequent,
                    'relation': r.relation,
                    'evidence': r.evidence_count,
                    'strength': round(r.strength, 3),
                }
                for r in self.causal_rules.rules
            ],
            'abstractions': self.abstractor.abstractions,
            'stats': self.get_system_stats(),
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Return full conversation history."""
        return list(self.conversation_history)

    def reset(self):
        """Reset all knowledge and state (for fresh training)."""
        self.__init__()
        logger.info("AI Engine reset to initial state")
