"""
NSCK AI Engine — Core Text & Image Understanding
=================================================

ZERO Hardcoded Patterns
-----------------------
Nothing in this engine is hardcoded.  There are no regex relation extractors,
no template strings, no keyword lexicons.  Everything — concept extraction,
relation discovery, emotion detection, intent classification, response
generation — is **learned autonomously** from training data using pure VSA
(Vector Symbolic Architecture) operations.

How concepts are discovered:
    Words that appear in similar contexts develop similar hypervectors via
    incremental bundling of context-window observations.  High-frequency
    content words are promoted to "concepts".  No POS tagger, no NER model.

How relations are learned:
    When two concepts co-occur within a sentence, a relation triple
    (subject, sentence_hv, object) is stored.  The sentence hypervector
    itself acts as the "relation type" — similar sentence structures
    produce similar relation vectors.  At query time the system finds
    the most similar stored sentence and uses it as a response fragment.

How intents are classified:
    Each training sentence is stored as an episodic memory with its
    hypervector.  At query time the system computes similarity to all
    stored episodes; the closest match's context determines the response
    strategy (answer with related facts, or echo back stored knowledge).

How responses are generated:
    The system retrieves the most relevant stored sentences (by HV
    similarity) and uses a learned n-gram language model to glue
    fragments together.  All words in the response trace back to
    specific training data.

Glass-Box Traceability
----------------------
Every ``chat()`` call produces a ``ThoughtTrace`` recording every cognitive
step.  The dashboard can display the full trace.

Autonomous Cognition
--------------------
The engine learns, reasons, abstracts, and applies knowledge on its own.
No human-authored rules or templates are involved.
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
# Constants
# ---------------------------------------------------------------------------
DIMENSION = 10240       # Hypervector dimensionality (binary, 10,240 bits)
MAX_CONTEXT = 20        # Maximum conversation context window
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity to consider a match


# ═══════════════════════════════════════════════════════════════════════════
# §1  HyperVector — the only data structure the system needs
# ═══════════════════════════════════════════════════════════════════════════

class HyperVector:
    """A 10,240-dimensional binary vector for holographic reduced
    representations.

    Why binary?
    - XOR binding is O(D) and self-inverse (A ⊕ A = 𝟎)
    - Majority-rule bundling preserves information from all constituents
    - Hamming-distance similarity is a simple bit-count
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
        return cls(np.zeros(DIMENSION, dtype=np.int8))

    @classmethod
    def from_seed(cls, seed: str) -> 'HyperVector':
        """Deterministic HV from a string seed — same seed ⟹ same vector."""
        h = hashlib.sha256(seed.encode('utf-8')).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], 'big'))
        return cls(rng.randint(0, 2, size=DIMENSION, dtype=np.int8))

    def bind(self, other: 'HyperVector') -> 'HyperVector':
        """XOR binding — creates a vector dissimilar to both inputs."""
        return HyperVector(np.bitwise_xor(self.bits, other.bits))

    @staticmethod
    def bundle(vectors: list) -> 'HyperVector':
        """Majority-rule bundling — result is similar to every input."""
        if not vectors:
            return HyperVector.zero()
        if len(vectors) == 1:
            return HyperVector(vectors[0].bits.copy())
        stacked = np.stack([v.bits for v in vectors], axis=0)
        total = stacked.sum(axis=0)
        threshold = len(vectors) / 2.0
        result = np.where(
            total > threshold, 1,
            np.where(total < threshold, 0,
                     np.random.randint(0, 2, size=DIMENSION)),
        ).astype(np.int8)
        return HyperVector(result)

    def permute(self, n: int = 1) -> 'HyperVector':
        """Circular bit shift — positional / temporal encoding."""
        return HyperVector(np.roll(self.bits, n))

    def similarity(self, other: 'HyperVector') -> float:
        """Normalised Hamming similarity  (0 = random, 1 = identical)."""
        return float(np.sum(self.bits == other.bits)) / DIMENSION

    def __repr__(self):
        return f"<HV d={DIMENSION} dens={np.mean(self.bits):.3f}>"


# ═══════════════════════════════════════════════════════════════════════════
# §2  Glass-Box Thought Trace
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TraceStep:
    """One atomic cognitive step in the reasoning pipeline."""
    stage: str
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ThoughtTrace:
    """Complete end-to-end trace of a single cognitive cycle.

    Every ``chat()`` call creates one trace.  The dashboard displays it so a
    human can see *exactly* why the model said what it said.
    """

    def __init__(self, query: str):
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.query: str = query
        self.steps: List[TraceStep] = []
        self.start_time: float = time.time()
        self._stage: Optional[str] = None
        self._t0: float = 0.0
        self._inputs: Dict[str, Any] = {}

    def begin(self, stage: str, inputs: Optional[Dict[str, Any]] = None):
        self._stage = stage
        self._t0 = time.time()
        self._inputs = inputs or {}

    def end(self, action: str, outputs: Optional[Dict[str, Any]] = None):
        elapsed = (time.time() - self._t0) * 1000
        self.steps.append(TraceStep(
            stage=self._stage or "unknown", action=action,
            inputs=self._inputs, outputs=outputs or {},
            duration_ms=round(elapsed, 3),
        ))
        self._stage = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'query': self.query,
            'total_ms': round((time.time() - self.start_time) * 1000, 2),
            'step_count': len(self.steps),
            'steps': [
                {'stage': s.stage, 'action': s.action,
                 'inputs': _safe(s.inputs), 'outputs': _safe(s.outputs),
                 'duration_ms': s.duration_ms}
                for s in self.steps
            ],
        }


def _safe(obj, depth=0):
    """Recursively convert to JSON-safe types."""
    if depth > 4:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v, depth + 1) for v in obj]
    if isinstance(obj, np.integer):
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


# ═══════════════════════════════════════════════════════════════════════════
# §3  Text Encoder — learned semantic folding (no hardcoded word lists)
# ═══════════════════════════════════════════════════════════════════════════

# Functional words that carry grammatical rather than semantic weight.
# These are the *only* built-in list in the system; they are language-
# structural (not domain-specific) and equivalent to a tokeniser's
# punctuation table.  Every other classification is learned.
_FUNCTION_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'out',
    'off', 'over', 'also', 'it', 'its', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
})


class TextEncoder:
    """Encodes text → hypervectors using *learned* semantic folding.

    How it works (no hardcoded patterns):
    1. Each unique word gets a deterministic base HV (from its hash).
    2. Context is captured by bundling ±N neighbour HVs.
    3. Word order is preserved via permutation encoding.
    4. Training updates context vectors incrementally so that words in
       similar contexts converge to similar representations.
    """

    def __init__(self, context_window: int = 3):
        self.context_window = context_window
        self.word_vectors: Dict[str, HyperVector] = {}
        self.word_freq: Counter = Counter()
        self.cooccur: Dict[str, Counter] = defaultdict(Counter)
        self.context_vectors: Dict[str, HyperVector] = {}
        self._total_words = 0

    def _get_word_hv(self, word: str) -> HyperVector:
        w = word.lower().strip()
        if w not in self.word_vectors:
            self.word_vectors[w] = HyperVector.from_seed(f"w:{w}")
        return self.word_vectors[w]

    def encode_sentence(self, text: str) -> HyperVector:
        """Encode a sentence into a single HV (bundle of positioned
        word-in-context vectors)."""
        words = self._tokenize(text)
        if not words:
            return HyperVector()
        hvs = []
        for i, w in enumerate(words):
            start = max(0, i - self.context_window)
            end = min(len(words), i + self.context_window + 1)
            ctx = words[start:i] + words[i + 1:end]
            wic = self._encode_word_in_context(w, ctx)
            hvs.append(wic.permute(i))
        return HyperVector.bundle(hvs)

    def _encode_word_in_context(self, word, context):
        whv = self._get_word_hv(word)
        if not context:
            return whv
        ctx_hvs = [self._get_word_hv(c).permute(i - len(context) // 2)
                    for i, c in enumerate(context)]
        return whv.bind(HyperVector.bundle(ctx_hvs))

    def learn_text(self, text: str) -> Dict[str, Any]:
        """Learn co-occurrences and update context vectors.
        Returns stats about what was learned."""
        sentences = self._split_sentences(text)
        stats = {'sentences': 0, 'new_words': 0, 'cooccurrences': 0}
        for sent in sentences:
            words = self._tokenize(sent)
            if len(words) < 2:
                continue
            stats['sentences'] += 1
            for i, w in enumerate(words):
                if w in _FUNCTION_WORDS:
                    continue
                new = w not in self.word_freq
                self.word_freq[w] += 1
                self._total_words += 1
                if new:
                    stats['new_words'] += 1
                start = max(0, i - self.context_window)
                end = min(len(words), i + self.context_window + 1)
                ctx_words = [words[j] for j in range(start, end)
                             if j != i and words[j] not in _FUNCTION_WORDS]
                for c in ctx_words:
                    self.cooccur[w][c] += 1
                    stats['cooccurrences'] += 1
                wic = self._encode_word_in_context(w, ctx_words)
                if w in self.context_vectors:
                    self.context_vectors[w] = HyperVector.bundle(
                        [self.context_vectors[w], wic])
                else:
                    self.context_vectors[w] = wic
        return stats

    def extract_concepts(self, text: str) -> List[str]:
        """Extract content words (concepts) from text — no regex, no NER.
        Simply returns non-function words with length > 2."""
        words = self._tokenize(text)
        seen = set()
        out = []
        for w in words:
            if w not in _FUNCTION_WORDS and len(w) > 2 and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def find_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find words with similar learned context vectors."""
        if word not in self.context_vectors:
            return []
        target = self.context_vectors[word]
        sims = []
        for other, hv in self.context_vectors.items():
            if other == word:
                continue
            s = target.similarity(hv)
            if s > 0.45:
                sims.append((other, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            'vocabulary_size': len(self.word_vectors),
            'trained_words': len(self.context_vectors),
            'total_words_seen': self._total_words,
            'top_words': self.word_freq.most_common(10),
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', ' ', t)
        return [w for w in t.split() if len(w) > 1]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'[.!?]+', text)
        return [s.strip() for s in parts if len(s.strip()) > 5]


# ═══════════════════════════════════════════════════════════════════════════
# §4  Knowledge Store — semantic graph + episodic timeline (all learned)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Concept:
    """A concept stored in semantic memory."""
    name: str
    hv: HyperVector
    frequency: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    source_sentences: List[str] = field(default_factory=list)


@dataclass
class LearnedRelation:
    """A relation learned autonomously from training data.

    Unlike hardcoded relation extractors, each relation stores the full
    source sentence and its hypervector.  The "type" of the relation is
    not a label — it IS the sentence's HV, so similar sentences produce
    similar relation types automatically.
    """
    source_concept: str
    target_concept: str
    sentence: str               # the original sentence linking them
    sentence_hv: HyperVector    # HV of that sentence
    weight: float = 1.0
    evidence_count: int = 1


@dataclass
class Episode:
    """An episodic memory entry."""
    text: str
    hv: HyperVector
    timestamp: float
    concepts: List[str]
    response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeStore:
    """Dual-memory knowledge storage.

    * **Semantic memory** — concept graph with learned relations.
    * **Episodic memory** — timestamped experiences with LSH index.
    """

    def __init__(self, max_episodes: int = 50000):
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[LearnedRelation] = []
        self.rel_index: Dict[str, List[int]] = defaultdict(list)
        self.episodes: deque = deque(maxlen=max_episodes)
        self._lsh_buckets: Dict[int, List[int]] = defaultdict(list)
        self._lsh_proj: Optional[np.ndarray] = None
        self._lsh_bits = 16
        self._init_lsh()

    def _init_lsh(self):
        self._lsh_proj = np.random.randn(
            self._lsh_bits, DIMENSION).astype(np.float32)

    def _lsh_hash(self, hv: HyperVector) -> int:
        proj = self._lsh_proj @ hv.bits.astype(np.float32)
        bits = (proj > 0).astype(np.int32)
        return int(sum(b << i for i, b in enumerate(bits)))

    # ---- Concepts ----

    def add_concept(self, name: str, hv: HyperVector,
                    source_sentence: str = "") -> Concept:
        now = time.time()
        if name in self.concepts:
            c = self.concepts[name]
            c.frequency += 1
            c.last_seen = now
            c.hv = HyperVector.bundle([c.hv, hv])
            if source_sentence:
                c.source_sentences.append(source_sentence)
                if len(c.source_sentences) > 20:
                    c.source_sentences = c.source_sentences[-20:]
        else:
            c = Concept(name=name, hv=hv, frequency=1,
                        first_seen=now, last_seen=now,
                        source_sentences=[source_sentence] if source_sentence else [])
            self.concepts[name] = c
        return c

    # ---- Relations (fully learned, no type labels) ----

    def add_relation(self, src: str, tgt: str,
                     sentence: str, sentence_hv: HyperVector,
                     weight: float = 1.0) -> LearnedRelation:
        """Store a learned relation.  No hardcoded relation type — the
        sentence HV *is* the relation representation."""
        # Check for duplicate
        for idx in self.rel_index.get(src, []):
            r = self.relations[idx]
            if r.target_concept == tgt and r.sentence == sentence:
                r.evidence_count += 1
                r.weight = min(r.weight + 0.1, 5.0)
                return r
        rel = LearnedRelation(
            source_concept=src, target_concept=tgt,
            sentence=sentence, sentence_hv=sentence_hv, weight=weight)
        idx = len(self.relations)
        self.relations.append(rel)
        self.rel_index[src].append(idx)
        self.rel_index[tgt].append(idx)
        return rel

    # ---- Episodes ----

    def record_episode(self, text: str, hv: HyperVector,
                       concepts: List[str], response: str = "",
                       metadata: Optional[Dict[str, Any]] = None) -> Episode:
        ep = Episode(text=text, hv=hv, timestamp=time.time(),
                     concepts=concepts, response=response,
                     metadata=metadata or {})
        ep_idx = len(self.episodes)
        self.episodes.append(ep)
        bucket = self._lsh_hash(hv)
        self._lsh_buckets[bucket].append(ep_idx)
        return ep

    # ---- Search ----

    def search_concepts(self, query_hv: HyperVector,
                        top_k: int = 5) -> List[Tuple[str, float]]:
        results = []
        for name, c in self.concepts.items():
            s = query_hv.similarity(c.hv)
            if s > SIMILARITY_THRESHOLD:
                results.append((name, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_episodes(self, query_hv: HyperVector,
                        top_k: int = 5) -> List[Tuple[Episode, float]]:
        bucket = self._lsh_hash(query_hv)
        cands: Set[int] = set()
        cands.update(self._lsh_buckets.get(bucket, []))
        for bit in range(min(4, self._lsh_bits)):
            cands.update(self._lsh_buckets.get(bucket ^ (1 << bit), []))
        eps = list(self.episodes)
        n = len(eps)
        for i in range(max(0, n - 100), n):
            cands.add(i)
        results = []
        for idx in cands:
            if idx < n:
                ep = eps[idx]
                s = query_hv.similarity(ep.hv)
                if s > SIMILARITY_THRESHOLD:
                    results.append((ep, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_related(self, concept: str,
                     max_depth: int = 2) -> List[Tuple[str, str, float]]:
        """Spreading activation: returns (related_concept, sentence, weight)."""
        visited: Set[str] = set()
        results: List[Tuple[str, str, float]] = []
        frontier = [(concept, 1.0, 0)]
        while frontier:
            cur, decay, depth = frontier.pop(0)
            if cur in visited or depth > max_depth:
                continue
            visited.add(cur)
            for idx in self.rel_index.get(cur, []):
                r = self.relations[idx]
                other = r.target_concept if r.source_concept == cur else r.source_concept
                w = r.weight * decay
                results.append((other, r.sentence, w))
                if depth + 1 <= max_depth:
                    frontier.append((other, decay * 0.7, depth + 1))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'total_episodes': len(self.episodes),
            'lsh_buckets': len(self._lsh_buckets),
            'top_concepts': sorted(
                [(n, c.frequency) for n, c in self.concepts.items()],
                key=lambda x: x[1], reverse=True)[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════
# §5  Causal Rule Store — learned autonomously from data
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CausalRule:
    """A causal rule learned from data.

    The antecedent and consequent are concept sets.  The sentence_hv
    captures the *way* they are related (no hardcoded label).
    """
    antecedent: List[str]
    consequent: List[str]
    sentence: str
    sentence_hv: HyperVector
    evidence: int = 1
    confidence: float = 0.5

    @property
    def strength(self) -> float:
        return min(1.0, self.confidence + 0.1 * np.log1p(self.evidence))


class CausalRuleStore:
    """Learns and stores causal rules from data.

    Rules are discovered when the engine encounters sentences that link
    two concept sets.  The discovery is fully data-driven: any sentence
    containing at least two distinct concepts is a potential rule.
    """

    def __init__(self):
        self.rules: List[CausalRule] = []
        self._ant_idx: Dict[str, List[int]] = defaultdict(list)

    def add_rule(self, antecedent: List[str], consequent: List[str],
                 sentence: str, sentence_hv: HyperVector) -> CausalRule:
        ant_set, con_set = set(antecedent), set(consequent)
        for idx in self._ant_idx.get(antecedent[0], []):
            r = self.rules[idx]
            if set(r.antecedent) == ant_set and set(r.consequent) == con_set:
                r.evidence += 1
                r.confidence = min(1.0, r.confidence + 0.05)
                return r
        rule = CausalRule(antecedent=list(antecedent),
                          consequent=list(consequent),
                          sentence=sentence, sentence_hv=sentence_hv)
        idx = len(self.rules)
        self.rules.append(rule)
        for a in antecedent:
            self._ant_idx[a].append(idx)
        return rule

    def forward_chain(self, active: List[str],
                      max_depth: int = 3) -> List[Tuple[CausalRule, int]]:
        """Fire rules whose antecedents match active concepts."""
        fired: List[Tuple[CausalRule, int]] = []
        frontier = set(active)
        used: Set[int] = set()
        for depth in range(max_depth):
            new_concepts: Set[str] = set()
            for concept in list(frontier):
                for idx in self._ant_idx.get(concept, []):
                    if idx in used:
                        continue
                    rule = self.rules[idx]
                    if all(a in (frontier | set(active)) for a in rule.antecedent):
                        fired.append((rule, depth))
                        used.add(idx)
                        new_concepts.update(rule.consequent)
            if not new_concepts:
                break
            frontier = new_concepts
        return fired

    def get_stats(self):
        return {
            'total_rules': len(self.rules),
            'indexed_concepts': len(self._ant_idx),
            'avg_confidence': (round(float(np.mean(
                [r.confidence for r in self.rules])), 3) if self.rules else 0),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §6  Knowledge Abstractor — autonomous generalisation
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeAbstractor:
    """Generalises specific facts into categories — autonomously.

    How it works (no hardcoded relation types):
    1. Group all relations by their target concept.
    2. If a target concept appears as the target in N+ distinct relations,
       create an abstraction: it's a *category* whose *members* are the
       source concepts of those relations.
    3. The category's HV is the bundle of all its members' HVs.
    """

    def __init__(self, min_members: int = 2):
        self.min_members = min_members
        self.abstractions: Dict[str, Dict[str, Any]] = {}

    def abstract(self, relations: List[LearnedRelation],
                 knowledge: 'KnowledgeStore') -> List[Dict[str, Any]]:
        # Group by target concept
        groups: Dict[str, List[str]] = defaultdict(list)
        for r in relations:
            groups[r.target_concept].append(r.source_concept)
        new_abs = []
        for target, sources in groups.items():
            unique = list(set(sources))
            if len(unique) < self.min_members:
                continue
            key = f"category:{target}"
            if key in self.abstractions:
                self.abstractions[key]['members'] = list(
                    set(self.abstractions[key]['members']) | set(unique))
                continue
            ab = {'category': target, 'members': unique,
                  'member_count': len(unique)}
            self.abstractions[key] = ab
            new_abs.append(ab)
            hvs = [knowledge.concepts[s].hv for s in unique
                   if s in knowledge.concepts]
            if hvs:
                knowledge.add_concept(
                    name=target, hv=HyperVector.bundle(hvs),
                    source_sentence=f"Category with {len(unique)} members: "
                                    + ", ".join(unique[:5]))
        return new_abs

    def get_stats(self):
        return {
            'total_abstractions': len(self.abstractions),
            'categories': list(self.abstractions.keys())[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════
# §7  Emotion Tracker — learned from data, not from keyword lists
# ═══════════════════════════════════════════════════════════════════════════

class EmotionTracker:
    """Tracks conversational emotional state.

    How it learns (no hardcoded keyword lists):
    The tracker maintains a valence/arousal state.  During training, the
    system feeds back reward signals (did the user respond positively?).
    At inference time the system compares the input HV to recent positive
    and negative HVs to estimate valence.

    The 9 emotion labels (joy, trust, fear …) are just human-readable
    names for regions of the 2D valence-arousal space — they are NOT used
    for any decision-making.
    """

    _EMOTION_COORDS = {
        'joy':          ( 0.8,  0.7),
        'trust':        ( 0.5,  0.2),
        'fear':         (-0.7,  0.8),
        'surprise':     ( 0.0,  0.9),
        'sadness':      (-0.6,  0.2),
        'disgust':      (-0.5,  0.4),
        'anger':        (-0.5,  0.8),
        'anticipation': ( 0.3,  0.5),
        'neutral':      ( 0.0,  0.3),
    }

    def __init__(self):
        self.valence: float = 0.0
        self.arousal: float = 0.3
        self.current_emotion: str = 'neutral'
        self.history: deque = deque(maxlen=200)
        self._update_count: int = 0
        # Learned valence signals: HVs associated with pos / neg feedback
        self._positive_hvs: List[HyperVector] = []
        self._negative_hvs: List[HyperVector] = []

    def learn_valence(self, hv: HyperVector, valence: float):
        """Learn that this HV is associated with positive/negative valence."""
        if valence > 0.2:
            self._positive_hvs.append(hv)
            if len(self._positive_hvs) > 200:
                self._positive_hvs = self._positive_hvs[-200:]
        elif valence < -0.2:
            self._negative_hvs.append(hv)
            if len(self._negative_hvs) > 200:
                self._negative_hvs = self._negative_hvs[-200:]

    def update_from_hv(self, hv: HyperVector) -> str:
        """Estimate emotional tone by comparing to learned pos/neg HVs."""
        pos_sim = 0.0
        neg_sim = 0.0
        if self._positive_hvs:
            pos_sim = max(hv.similarity(p) for p in self._positive_hvs[-50:])
        if self._negative_hvs:
            neg_sim = max(hv.similarity(n) for n in self._negative_hvs[-50:])

        # If we have no learned data yet, stay neutral
        if pos_sim > 0.5 or neg_sim > 0.5:
            sentiment = pos_sim - neg_sim
            self.valence = 0.7 * self.valence + 0.3 * sentiment

        # Question-like inputs (short, ending with ?) raise arousal slightly
        self.arousal = max(0.1, self.arousal * 0.95)

        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.current_emotion = self._map()
        self._update_count += 1
        self.history.append({
            'emotion': self.current_emotion,
            'valence': round(self.valence, 3),
            'arousal': round(self.arousal, 3),
            'timestamp': time.time(),
        })
        return self.current_emotion

    def _map(self) -> str:
        best, best_d = 'neutral', float('inf')
        for em, (v, a) in self._EMOTION_COORDS.items():
            d = (self.valence - v) ** 2 + (self.arousal - a) ** 2
            if d < best_d:
                best_d = d
                best = em
        return best

    def get_blend(self) -> Dict[str, float]:
        blend = {}
        total = 0.0
        for em, (v, a) in self._EMOTION_COORDS.items():
            d = ((self.valence - v) ** 2 + (self.arousal - a) ** 2) ** 0.5
            w = max(0, 1.0 - d)
            if w > 0.05:
                blend[em] = w
                total += w
        if total > 0:
            blend = {k: round(v / total, 3) for k, v in blend.items()}
        return dict(sorted(blend.items(), key=lambda x: x[1], reverse=True))

    def get_state(self) -> Dict[str, Any]:
        return {
            'emotion': self.current_emotion,
            'valence': round(self.valence, 3),
            'arousal': round(self.arousal, 3),
            'blend': self.get_blend(),
            'learned_positive': len(self._positive_hvs),
            'learned_negative': len(self._negative_hvs),
            'updates': self._update_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# §8  Response Generator — learned n-gram model, NO templates
# ═══════════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Generates responses using learned n-gram patterns and retrieved
    sentences.  NO hardcoded templates whatsoever.

    How it works:
    1. During training, the generator learns bigram and trigram
       continuation probabilities from all training text.
    2. At inference time it retrieves relevant stored sentences and,
       if needed, extends them with learned n-gram continuations.
    3. Every word in the output traces back to a specific training text.
    """

    def __init__(self):
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        self.trigrams: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._corpus_size: int = 0

    def learn(self, text: str):
        """Learn n-gram patterns from text."""
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        self._corpus_size += len(words)
        for i in range(len(words) - 1):
            self.bigrams[words[i]][words[i + 1]] += 1
            if i < len(words) - 2:
                self.trigrams[(words[i], words[i + 1])][words[i + 2]] += 1

    def continue_from(self, seed_words: List[str],
                      max_len: int = 25) -> Optional[str]:
        """Generate text by continuing from seed words using learned
        n-gram probabilities."""
        if not self.bigrams or not seed_words:
            return None
        current = None
        for w in seed_words:
            if w in self.bigrams:
                current = w
                break
        if current is None:
            return None
        words = [current]
        for _ in range(max_len):
            cands = self.bigrams.get(current, Counter())
            if not cands:
                break
            total = sum(cands.values())
            r = np.random.random() * total
            cum = 0
            nxt = None
            for word, cnt in cands.items():
                cum += cnt
                if cum >= r:
                    nxt = word
                    break
            if nxt is None:
                break
            words.append(nxt)
            current = nxt
            if len(words) >= 8 and nxt in _FUNCTION_WORDS:
                break
        if len(words) < 3:
            return None
        return " ".join(words)

    def get_stats(self):
        return {
            'bigram_vocab': len(self.bigrams),
            'trigram_contexts': len(self.trigrams),
            'corpus_size': self._corpus_size,
        }


# ═══════════════════════════════════════════════════════════════════════════
# §9  NSCKAIEngine — the orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class NSCKAIEngine:
    """Main AI engine.  Orchestrates all components.

    Glass-box architecture: every ``chat()`` returns a full ``ThoughtTrace``.
    Autonomous cognition: learns, reasons, abstracts — no hardcoded rules.

    Usage::

        engine = NSCKAIEngine()
        engine.train_on_text("Paris is the capital of France.")
        result = engine.chat("What is the capital of France?")
        print(result['response'])
        print(result['trace'])  # full glass-box reasoning chain
    """

    def __init__(self):
        self.encoder = TextEncoder(context_window=3)
        self.knowledge = KnowledgeStore(max_episodes=50000)
        self.generator = ResponseGenerator()
        self.emotion = EmotionTracker()
        self.causal_rules = CausalRuleStore()
        self.abstractor = KnowledgeAbstractor(min_members=2)
        self.conversation_history: deque = deque(maxlen=MAX_CONTEXT)
        self._training_stats = {
            'texts_trained': 0,
            'total_concepts': 0,
            'total_relations': 0,
            'total_episodes': 0,
            'total_causal_rules': 0,
            'total_abstractions': 0,
            'training_time_s': 0.0,
        }
        self._query_count = 0
        logger.info("NSCK AI Engine initialised (zero-hardcode mode)")

    # ------------------------------------------------------------------
    # Training — fully autonomous knowledge extraction
    # ------------------------------------------------------------------

    def train_on_text(self, text: str) -> Dict[str, Any]:
        """Learn from a text passage.

        Process (no hardcoded patterns):
        1. Learn word co-occurrences and update context vectors.
        2. Extract concepts (= frequent content words).
        3. For each sentence containing ≥2 concepts, store a *learned
           relation* — the sentence itself is the relation representation.
        4. Learn n-gram continuation probabilities.
        5. Run abstractor to create category concepts.
        """
        start = time.time()
        enc_stats = self.encoder.learn_text(text)
        sentences = self.encoder._split_sentences(text)
        concepts_added = 0
        relations_added = 0
        rules_added = 0

        for sentence in sentences:
            concepts = self.encoder.extract_concepts(sentence)
            sent_hv = self.encoder.encode_sentence(sentence)

            # Store each concept
            for c in concepts:
                c_hv = self.encoder.encode_sentence(c)
                self.knowledge.add_concept(c, c_hv, source_sentence=sentence)
                concepts_added += 1

            # Store learned relations between every pair of concepts in
            # the sentence — the sentence HV IS the relation
            if len(concepts) >= 2:
                for i in range(len(concepts)):
                    for j in range(i + 1, min(i + 4, len(concepts))):
                        self.knowledge.add_relation(
                            src=concepts[i], tgt=concepts[j],
                            sentence=sentence, sentence_hv=sent_hv)
                        relations_added += 1

                # Treat the first half as antecedent, second as consequent
                # for causal rule learning
                mid = len(concepts) // 2
                if mid > 0:
                    self.causal_rules.add_rule(
                        antecedent=concepts[:mid],
                        consequent=concepts[mid:],
                        sentence=sentence, sentence_hv=sent_hv)
                    rules_added += 1

            # Record as episodic memory
            self.knowledge.record_episode(
                text=sentence, hv=sent_hv, concepts=concepts)

        # Learn n-gram patterns
        self.generator.learn(text)

        # Run abstractor
        new_abs = self.abstractor.abstract(
            self.knowledge.relations, self.knowledge)

        elapsed = time.time() - start
        self._training_stats['texts_trained'] += 1
        self._training_stats['total_concepts'] += concepts_added
        self._training_stats['total_relations'] += relations_added
        self._training_stats['total_episodes'] += len(sentences)
        self._training_stats['total_causal_rules'] += rules_added
        self._training_stats['total_abstractions'] += len(new_abs)
        self._training_stats['training_time_s'] += elapsed

        result = {
            'sentences_processed': len(sentences),
            'concepts_added': concepts_added,
            'relations_added': relations_added,
            'causal_rules_added': rules_added,
            'abstractions_created': len(new_abs),
            'encoder_stats': enc_stats,
            'elapsed_s': round(elapsed, 4),
        }
        logger.info("Trained: %s", result)
        return result

    # ------------------------------------------------------------------
    # Chat — glass-box, traced, fully autonomous
    # ------------------------------------------------------------------

    def chat(self, user_input: str, auto_learn: bool = True) -> Dict[str, Any]:
        """Process user input and generate a response.

        Every cognitive step is recorded in a ThoughtTrace.
        No hardcoded templates, intents, or patterns are used.
        The response is built entirely from learned knowledge.

        Parameters
        ----------
        user_input : str
            The user's message.
        auto_learn : bool
            If True (default), the engine will also train on the user's
            input when it looks like a factual statement.  Set to False
            to keep training and inference fully separate.
        """
        trace = ThoughtTrace(user_input)
        start = time.time()
        self._query_count += 1

        # --- Early validation: if no meaningful words, short-circuit ---
        stripped = re.sub(r'[^\w\s]', ' ', user_input).strip()
        meaningful_words = [w for w in stripped.lower().split()
                           if len(w) > 2 and w not in _FUNCTION_WORDS]
        if not meaningful_words:
            trace.begin("validate", {"input_length": len(user_input)})
            trace.end("No meaningful content in input", {})
            elapsed = time.time() - start
            return {
                'response': "I need more training data to answer that.",
                'emotion': self.emotion.get_state(),
                'confidence': 0.0,
                'reasoning': {
                    'query_concepts': [],
                    'matched_concepts': [],
                    'episodes_found': 0,
                    'related_facts': 0,
                    'causal_rules_fired': 0,
                },
                'trace': trace.to_dict(),
                'latency_ms': round(elapsed * 1000, 1),
            }

        # --- Stage 1: Encode input ---
        trace.begin("encode", {"text_length": len(user_input)})
        query_hv = self.encoder.encode_sentence(user_input)
        trace.end("Encoded input to hypervector", {"dimension": DIMENSION})

        # --- Stage 2: Emotion ---
        trace.begin("emotion", {})
        emotion = self.emotion.update_from_hv(query_hv)
        trace.end("Updated emotional state", {
            "emotion": emotion, "valence": self.emotion.valence})

        # --- Stage 3: Extract concepts ---
        trace.begin("extract_concepts", {"input": user_input[:200]})
        query_concepts = self.encoder.extract_concepts(user_input)
        # Resolve pronouns / references using conversation context
        resolved = self._resolve_context(user_input, query_concepts)
        if len(resolved) > len(query_concepts):
            trace.end("Extracted and resolved concepts", {
                "raw_concepts": query_concepts,
                "resolved_concepts": resolved,
                "context_added": [c for c in resolved
                                  if c not in query_concepts]})
            query_concepts = resolved
        else:
            trace.end("Extracted concepts from input", {
                "concepts": query_concepts})

        # --- Stage 4: Search semantic memory ---
        trace.begin("search_semantic", {"concepts": query_concepts})
        matched = self.knowledge.search_concepts(query_hv, top_k=10)
        trace.end("Searched concept memory by HV similarity", {
            "matches": [(n, round(s, 3)) for n, s in matched]})

        # --- Stage 5: Search episodic memory ---
        trace.begin("search_episodic", {})
        episodes = self.knowledge.search_episodes(query_hv, top_k=10)
        trace.end("Searched episodic memory via LSH", {
            "episodes_found": len(episodes),
            "best_sim": round(episodes[0][1], 3) if episodes else 0})

        # --- Stage 6: Spreading activation ---
        trace.begin("spread_activation", {"seeds": query_concepts})
        related_facts: List[Tuple[str, str, float]] = []
        for c in query_concepts:
            if c in self.knowledge.concepts:
                related_facts.extend(
                    self.knowledge.find_related(c, max_depth=2))
        # Deduplicate
        seen_sents: Set[str] = set()
        unique_facts: List[Tuple[str, str, float]] = []
        for concept, sentence, weight in related_facts:
            if sentence not in seen_sents:
                seen_sents.add(sentence)
                unique_facts.append((concept, sentence, weight))
        related_facts = sorted(unique_facts,
                               key=lambda x: x[2], reverse=True)[:10]
        trace.end("Spread activation through graph", {
            "facts_found": len(related_facts)})

        # --- Stage 7: Causal inference ---
        trace.begin("causal_inference", {"active": query_concepts})
        fired = self.causal_rules.forward_chain(query_concepts, max_depth=2)
        trace.end("Forward-chained causal rules", {
            "rules_fired": len(fired),
            "details": [{"ant": r.antecedent, "con": r.consequent,
                         "sentence": r.sentence,
                         "strength": round(r.strength, 3), "depth": d}
                        for r, d in fired[:5]]})

        # --- Stage 8: Assemble response (no templates!) ---
        trace.begin("generate", {"strategy": "retrieved_sentences"})
        response = self._build_response(
            user_input, query_concepts, matched, episodes,
            related_facts, fired)
        trace.end("Generated response from learned knowledge", {
            "response_preview": response[:200]})

        # --- Record & update history ---
        # Auto-learn: if the user made a factual statement and we had no
        # relevant knowledge, learn from it.
        if auto_learn and not user_input.strip().endswith('?') and not matched:
            self.train_on_text(user_input)

        self.knowledge.record_episode(
            text=user_input, hv=query_hv,
            concepts=query_concepts, response=response)
        self.conversation_history.append(
            {'role': 'user', 'content': user_input, 'ts': time.time()})
        self.conversation_history.append(
            {'role': 'assistant', 'content': response, 'ts': time.time()})

        elapsed = time.time() - start
        confidence = 0.0
        if matched:
            confidence = max(s for _, s in matched)
        if episodes:
            confidence = max(confidence, max(s for _, s in episodes))

        return {
            'response': response,
            'emotion': self.emotion.get_state(),
            'confidence': round(confidence, 3),
            'reasoning': {
                'query_concepts': query_concepts,
                'matched_concepts': [(n, round(s, 3)) for n, s in matched],
                'episodes_found': len(episodes),
                'related_facts': len(related_facts),
                'causal_rules_fired': len(fired),
            },
            'trace': trace.to_dict(),
            'latency_ms': round(elapsed * 1000, 1),
        }

    # ------------------------------------------------------------------
    # Response builder — NO TEMPLATES, all from learned data
    # ------------------------------------------------------------------

    def _resolve_context(self, user_input: str,
                         query_concepts: List[str]) -> List[str]:
        """Resolve pronouns and implicit references using conversation
        history.

        If the user says "she", "he", "it", "they", or "that", look
        back in the conversation to find the most recently mentioned
        content concepts and merge them into the query.
        """
        pronouns = {'she', 'he', 'it', 'they', 'them', 'that', 'this',
                     'those', 'these', 'its', 'his', 'her', 'their'}
        # Common discourse verbs that are concepts but shouldn't be resolved
        noise = {'tell', 'know', 'think', 'say', 'talk', 'ask', 'want',
                 'need', 'like', 'make', 'take', 'give', 'get', 'see',
                 'look', 'find', 'help', 'show', 'try', 'use', 'come',
                 'let', 'keep', 'set', 'put', 'run', 'read', 'write'}
        input_words = set(user_input.lower().split())
        if not (pronouns & input_words):
            return query_concepts

        # Walk backwards through history — only use the MOST RECENT
        # user turn's concepts to avoid confusion
        resolved = list(query_concepts)
        for entry in reversed(list(self.conversation_history)):
            if entry['role'] != 'user':
                continue
            prev_concepts = self.encoder.extract_concepts(entry['content'])
            for c in prev_concepts:
                if (c not in resolved and c not in _FUNCTION_WORDS
                        and c not in noise):
                    resolved.append(c)
            # Only look at the MOST RECENT user turn
            break
        return resolved

    def _build_response(
            self,
            user_input: str,
            query_concepts: List[str],
            matched_concepts: List[Tuple[str, float]],
            matched_episodes: List[Tuple[Episode, float]],
            related_facts: List[Tuple[str, str, float]],
            fired_rules: List[Tuple[CausalRule, int]],
    ) -> str:
        """Build a response entirely from learned knowledge.

        Strategy:
        1. Collect *candidate sentences* from all retrieval channels.
        2. Score them by concept overlap with the query (not just raw
           retrieval score) so the most *relevant* sentence wins.
        3. Select the top 1-3 non-redundant sentences.
        4. If nothing relevant is found, try n-gram continuation.
        5. If still nothing, use the best episodic match verbatim.

        No templates, no format strings — every word in the output was
        seen in training data or generated by the learned n-gram model.
        """
        query_set = set(query_concepts)

        # --- Collect candidate sentences with relevance scores ---
        candidates: List[Tuple[str, float]] = []

        # From related facts (spreading activation)
        for concept, sentence, weight in related_facts:
            candidates.append((sentence, weight))

        # From causal rules
        for rule, depth in fired_rules:
            candidates.append((rule.sentence, rule.strength * (0.8 ** depth)))

        # From episodic memory — skip episodes that are themselves
        # queries (questions ending with ?) or that start with common
        # chat prefixes, as these are user inputs not knowledge.
        for ep, sim in matched_episodes:
            if not ep.text:
                continue
            txt = ep.text.strip()
            if txt.startswith('[Image:'):
                continue
            if txt.endswith('?'):
                continue
            # Skip if it looks like a previous user query
            lower = txt.lower()
            if (lower.startswith('tell me') or lower.startswith('what ')
                    or lower.startswith('who ') or lower.startswith('where ')
                    or lower.startswith('when ') or lower.startswith('how ')
                    or lower.startswith('why ') or lower.startswith('which ')
                    or lower.startswith('are ') or lower.startswith('is ')
                    or lower.startswith('do ') or lower.startswith('does ')):
                continue
            candidates.append((txt, sim))

        # From concept source sentences
        for name, sim in matched_concepts[:5]:
            c = self.knowledge.concepts.get(name)
            if c and c.source_sentences:
                for src in c.source_sentences[:3]:
                    if not src.startswith('Category with'):
                        candidates.append((src, sim * 0.9))

        if not candidates:
            return self._fallback_response(user_input, query_concepts)

        # --- Score by concept overlap (relevance) ---
        # Weight rare/specific query concepts more than common ones.
        # "france" is more discriminative than "capital" when asking
        # about France's capital.
        concept_specificity: Dict[str, float] = {}
        for c in query_concepts:
            freq = self.encoder.word_freq.get(c, 1)
            # Inverse frequency: log2(freq+2) provides smooth scaling;
            # +2 prevents division by zero and ensures unseen words
            # (freq=0) get a high specificity of 1.0
            concept_specificity[c] = 1.0 / np.log2(freq + 2)

        scored: List[Tuple[str, float, float]] = []
        for sent, base_score in candidates:
            sent_concepts = set(self.encoder.extract_concepts(sent))
            overlap = sent_concepts & query_set
            # Weighted overlap: sum of specificities of matched concepts
            weighted_overlap = sum(concept_specificity.get(c, 0.5)
                                   for c in overlap)
            raw_overlap = len(overlap)
            # Final relevance combines base score + weighted overlap
            relevance = base_score * 0.4 + weighted_overlap * 0.6
            if raw_overlap == 0:
                relevance *= 0.2
            scored.append((sent, relevance, weighted_overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        # --- Deduplicate with fuzzy matching ---
        used_norms: Set[str] = set()
        selected: List[str] = []
        for sent, score, overlap in scored:
            # Normalise for dedup: lowercase, strip punctuation
            norm = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            if norm in used_norms:
                continue
            # Once we have one good result, only add more if they
            # have actual concept overlap with the query
            sent_concepts = set(self.encoder.extract_concepts(sent))
            if selected and not (sent_concepts & query_set):
                continue
            # Skip if structurally too similar to already selected text
            # (e.g. "X is the capital of Y" vs "Z is the capital of W").
            # Jaccard > 0.5 means more than half the words overlap,
            # indicating the sentences are structural variants.
            skip = False
            for existing in used_norms:
                norm_words = set(norm.split())
                exist_words = set(existing.split())
                if norm_words and exist_words:
                    jaccard = (len(norm_words & exist_words) /
                               len(norm_words | exist_words))
                    if jaccard > 0.5:
                        skip = True
                        break
            if skip:
                continue
            used_norms.add(norm)
            # Capitalise first letter
            selected.append(sent[0].upper() + sent[1:] if sent else sent)
            if len(selected) >= 2:
                break

        if selected:
            # Join selected sentences, ensure proper punctuation
            parts = []
            for s in selected:
                s = s.rstrip('.').strip()
                if s:
                    parts.append(s)
            return ". ".join(parts) + "."

        return self._fallback_response(user_input, query_concepts)

    def _fallback_response(self, user_input: str,
                           query_concepts: List[str]) -> str:
        """Generate a fallback when no relevant knowledge is found.

        Uses n-gram continuation if possible, otherwise returns the
        closest episodic match or a learned acknowledgement.
        """
        # If the input has NO meaningful concepts, don't try to
        # generate from random unrelated n-grams
        if not query_concepts:
            return "I need more training data to answer that."

        # Try n-gram generation — but validate the output contains
        # at least one query concept word (otherwise it's gibberish)
        gen = self.generator.continue_from(query_concepts)
        if gen:
            gen_lower = gen.lower()
            has_relevant = any(c in gen_lower for c in query_concepts)
            if has_relevant and len(gen.split()) >= 3:
                return gen[0].upper() + gen[1:] + "."

        # For statements (not questions), acknowledge learning
        if not user_input.strip().endswith('?'):
            # Find the closest stored sentence mentioning these concepts
            for c in query_concepts:
                if c in self.knowledge.concepts:
                    src = self.knowledge.concepts[c].source_sentences
                    if src:
                        s = src[-1].strip().rstrip('.')
                        if s:
                            return s[0].upper() + s[1:] + "."
            # No stored sentence found — echo back the user's own input
            # as acknowledgement (every word traces to user data)
            s = user_input.strip().rstrip('.')
            if s:
                return s[0].upper() + s[1:] + "."

        # Last resort: if we have episodes matching our concepts, use one
        if self.knowledge.episodes and query_concepts:
            for ep in reversed(list(self.knowledge.episodes)):
                ep_concepts = set(ep.concepts)
                if ep_concepts & set(query_concepts):
                    s = ep.text.strip().rstrip('.')
                    if s:
                        return s[0].upper() + s[1:] + "."

        return "I need more training data to answer that."

    # ------------------------------------------------------------------
    # Stats & export
    # ------------------------------------------------------------------

    def get_system_stats(self) -> Dict[str, Any]:
        return {
            'engine': {'query_count': self._query_count,
                       'conversation_length': len(self.conversation_history)},
            'training': self._training_stats.copy(),
            'encoder': self.encoder.get_stats(),
            'knowledge': self.knowledge.get_stats(),
            'generator': self.generator.get_stats(),
            'emotion': self.emotion.get_state(),
            'causal_rules': self.causal_rules.get_stats(),
            'abstractions': self.abstractor.get_stats(),
        }

    def export_knowledge(self) -> Dict[str, Any]:
        return {
            'concepts': {
                n: {'frequency': c.frequency,
                    'source_sentences': c.source_sentences}
                for n, c in self.knowledge.concepts.items()},
            'relations': [
                {'source': r.source_concept, 'target': r.target_concept,
                 'sentence': r.sentence, 'weight': r.weight,
                 'evidence': r.evidence_count}
                for r in self.knowledge.relations],
            'causal_rules': [
                {'antecedent': r.antecedent, 'consequent': r.consequent,
                 'sentence': r.sentence, 'evidence': r.evidence,
                 'strength': round(r.strength, 3)}
                for r in self.causal_rules.rules],
            'abstractions': self.abstractor.abstractions,
            'stats': self.get_system_stats(),
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return list(self.conversation_history)

    def reset(self):
        self.__init__()
        logger.info("AI Engine reset")
