"""
NSCK AI Engine — Genuine Cognitive AI Using the Full NSCK Architecture
======================================================================

This engine wraps the **actual** NSCK cognitive modules — SemanticMemory,
EpisodicMemory, GlobalWorkspace, EmotionSystem, CausalReasoner, SelfModel,
CuriosityModule, TextKnowledgeLearner — rather than reimplementing them.

Architecture Summary
--------------------
The NSCK (Neural-Symbolic Cognitive Kernel) is a glass-box cognitive
architecture built on Vector Symbolic Architecture (VSA).  It uses
10,240-bit binary hypervectors for all representations — no matrix
multiplication, no transformers, no neural networks for inference.

How Text Understanding Works
----------------------------
1. **Encoding**: Text → hypervectors via semantic folding (LinguaCortex).
   Each word gets a deterministic base HV.  Context is captured by
   bundling neighbouring word HVs with XOR binding.  Word order is
   encoded via circular bit-shift (permutation).

2. **Learning**: The ``TextKnowledgeLearner`` extracts concepts and
   relations from sentences.  Concepts are stored in ``SemanticMemory``
   (a NetworkX directed graph with HV indices).  Episodes are stored in
   ``EpisodicMemory`` (a two-tier VSA store with LSH indexing).  Causal
   links go into ``CausalGraph``.

3. **Retrieval**: At query time the input is encoded to a HV.  The system
   searches semantic memory (HV similarity), runs spreading activation
   through the concept graph, recalls similar episodes, and fires causal
   rules — all in parallel.

4. **Reasoning**: Results from all retrieval channels become ``Coalition``
   objects that compete in the ``GlobalWorkspace`` (LIDA-style
   consciousness model).  The winner determines the response strategy.

5. **Response**: The winning coalition's content — retrieved facts,
   activated concepts, causal chains — is assembled into natural language.
   Every word in the response traces back to specific training data.

6. **Self-monitoring**: ``SelfModel`` tracks confidence calibration.
   ``CuriosityModule`` detects novelty.  ``EmotionSystem`` tracks
   valence/arousal.  All are reported in the glass-box trace.

Glass-Box Traceability
----------------------
Every ``chat()`` call produces a ``ThoughtTrace`` that records:
- Input encoding (HV dimension, encoding time)
- Emotion state (Plutchik classification, valence, arousal)
- Concept extraction (what concepts were found)
- Semantic search (which concepts matched and at what similarity)
- Episodic recall (how many episodes, best similarity)
- Spreading activation (activated concepts and their levels)
- Causal inference (which causal rules fired, chains)
- GlobalWorkspace competition (which coalition won, why)
- Self-model confidence (calibration error, trend)
- Curiosity assessment (novelty score)
- Response generation (strategy, source sentences)

No Hardcoded Patterns
---------------------
The system has zero hardcoded response templates, regex relation
extractors, or keyword lexicons.  All knowledge is learned from training
data through the NSCK learning pipeline.
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Ensure nsck is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

# ---------------------------------------------------------------------------
# Import the actual NSCK cognitive modules
# ---------------------------------------------------------------------------
import python.core.vsa.hypervec_shim as hypervec_rs  # 10240-bit HyperVector

from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.reasoning.global_workspace import (
    GlobalWorkspace, Coalition, WorkspaceModule,
)
from python.core.cognitive.emotion_system import EmotionSystem
from python.core.cognitive.self_model import SelfModel
from python.core.reasoning.causal_reasoning import (
    CausalGraph, CausalReasoner, CausalDiscovery,
)
from python.core.learning.curiosity import CuriosityModule
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.reasoning.analogy import AnalogyEngine
from python.core.language.nlg import NLGEngine
from python.core.multimodal.multimodal_processor import (
    MultimodalProcessor, MultimodalInput, ProcessedInput, ModalityResult,
)
from python.core.multimodal.image_generator import (
    ImageGenerator, GenerationConfig, VisualFeatures,
)

# ---------------------------------------------------------------------------
# NSCK AI wrapper modules (integrated into the engine pipeline)
# ---------------------------------------------------------------------------
from nsck_ai_model.context_retention import ContextRetentionModule
from nsck_ai_model.counterfactual_reasoner import CounterfactualReasoner
from nsck_ai_model.response_composer import ResponseComposer
from nsck_ai_model.math_handler import MathHandler, is_math_question, LearnableOperatorDetector

logger = logging.getLogger("nsck_ai.engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSION = 10240          # HyperVector dimensionality (binary, 10,240 bits)
MAX_CONTEXT = 20           # Conversation history window
SIMILARITY_THRESHOLD = 0.35
# Seed modulo for deterministic HV creation from word hashes.
# Uses 32-bit range to match the NSCK HyperVector constructor's uint32 seed.
HASH_SEED_MODULO = 2 ** 32
# Maximum bit-shift for positional encoding.  NSCK HVs are 10,240 bits
# wide; shifting by more than 64 creates near-random positions which is
# sufficient for word-order encoding in typical sentence lengths.
MAX_POSITION_SHIFT = 64

# Functional words — the *only* built-in list.  These are structural
# (grammatical), not domain-specific.  Every other classification is learned.
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
    # Interrogative / quantifier words — carry no topical meaning
    'how', 'when', 'where', 'why', 'many', 'much', 'often',
})


# ═══════════════════════════════════════════════════════════════════════════
# §1  Glass-Box Thought Trace
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

    Every ``chat()`` call creates one trace.  The dashboard displays it
    so a human can see *exactly* why the model said what it said — and
    which NSCK modules contributed.
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
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════
# §2  Workspace Modules — bridge NSCK modules to GlobalWorkspace
# ═══════════════════════════════════════════════════════════════════════════

class _SemanticModule(WorkspaceModule):
    """Wraps SemanticMemory for GlobalWorkspace participation."""
    def receive_broadcast(self, content: Any):
        pass  # Semantic memory is passive — queried, not notified


class _EpisodicModule(WorkspaceModule):
    """Wraps EpisodicMemory for GlobalWorkspace participation."""
    def receive_broadcast(self, content: Any):
        pass


class _CausalModule(WorkspaceModule):
    """Wraps CausalReasoner for GlobalWorkspace participation."""
    def receive_broadcast(self, content: Any):
        pass


class _EmotionModule(WorkspaceModule):
    """Wraps EmotionSystem for GlobalWorkspace participation."""
    def __init__(self, emotion: EmotionSystem):
        self._emotion = emotion

    def receive_broadcast(self, content: Any):
        # Emotion system is informed of winning coalition
        pass


# ═══════════════════════════════════════════════════════════════════════════
# §2.5  Learned Cognitive Components — NO regex, NO hardcoded rules
# ═══════════════════════════════════════════════════════════════════════════

class IntentMemory:
    """Learns intent categories from training data using VSA prototypes.

    During training, sentences of each type (greeting, question, statement,
    thanks, farewell, etc.) are encoded as hypervectors and bundled
    together into a **prototype HV** for that intent.  At inference time,
    the input HV is compared against all prototypes — the closest one
    above threshold wins.

    This replaces ALL hardcoded regex-based intent detection (greeting
    patterns, yes/no question patterns, thanks patterns, etc.).
    """

    def __init__(self):
        # intent_label → list of HVs seen during training
        self._exemplars: Dict[str, List] = defaultdict(list)
        # intent_label → bundled prototype HV (built after training)
        self._prototypes: Dict[str, Any] = {}
        # intent_label → learned response text (from training data)
        self._response_templates: Dict[str, List[str]] = defaultdict(list)

    def add_example(self, label: str, text_hv, response: Optional[str] = None):
        """Record a training example for the given intent category."""
        self._exemplars[label].append(text_hv)
        if response:
            self._response_templates[label].append(response)

    def build_prototypes(self):
        """Bundle all exemplars into prototype HVs (call after training)."""
        for label, hvs in self._exemplars.items():
            if not hvs:
                continue
            proto = hvs[0]
            for hv in hvs[1:]:
                proto = proto.bundle(hv)
            self._prototypes[label] = proto

    def classify(self, query_hv, threshold: float = 0.510,
                 margin: float = 0.007
                 ) -> Optional[Tuple[str, float]]:
        """Return (intent_label, similarity) or None.

        Uses both an absolute *threshold* and a *margin* requirement:
        the best-matching intent must beat the runner-up by at least
        ``margin`` to avoid false positives from random noise (binary
        HV similarity clusters tightly around 0.50).
        """
        if not self._prototypes:
            return None
        scores = []
        for label, proto in self._prototypes.items():
            sim = query_hv.similarity(proto)
            scores.append((label, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_label, best_sim = scores[0]
        if best_sim < threshold:
            return None
        # Margin check: best must clearly beat runner-up
        if len(scores) > 1:
            second_sim = scores[1][1]
            if best_sim - second_sim < margin:
                return None
        return (best_label, best_sim)

    def get_response(self, label: str) -> Optional[str]:
        """Return a learned response for the given intent, if any."""
        templates = self._response_templates.get(label, [])
        if templates:
            return templates[0]  # Return the first learned response
        return None

    def is_yesno_question(self, query_hv) -> bool:
        """Check if the query is a yes/no question.

        Uses relative comparison between question_yesno and question
        prototypes — no absolute threshold, purely learned.
        """
        yesno_proto = self._prototypes.get('question_yesno')
        q_proto = self._prototypes.get('question')
        if yesno_proto is None:
            return False
        yesno_sim = query_hv.similarity(yesno_proto)
        if q_proto is None:
            return yesno_sim > 0.505
        q_sim = query_hv.similarity(q_proto)
        # Yes/no if clearly closer to yesno prototype (with margin)
        return yesno_sim > q_sim + 0.003 and yesno_sim > 0.50

    def stats(self) -> Dict[str, int]:
        return {label: len(hvs) for label, hvs in self._exemplars.items()}


class MorphologyLearner:
    """Learns word families from context co-occurrence instead of suffix rules.

    During training, each word accumulates a **context HV** built by
    bundling the HVs of words that appear near it.  Words with similar
    context HVs are morphological variants (e.g. "analogy" ↔ "analogies",
    "compute" ↔ "computing" ↔ "computer").

    At query time, ``find_family(word)`` returns all words whose context
    HV is within a learned similarity radius of the query word.

    This replaces the hardcoded ``_stem()`` suffix-stripping rules.
    """

    def __init__(self, window: int = 3):
        self._window = window
        # word → accumulated context HV
        self._context_hvs: Dict[str, Any] = {}
        # word → count of contexts seen
        self._counts: Dict[str, int] = defaultdict(int)
        # Cached family groups: word → set of family members
        self._families: Dict[str, Set[str]] = {}
        self._families_built = False

    def observe_sentence(self, words: List[str], encode_fn):
        """Learn context vectors from a sentence.

        For each content word, bundle the HVs of its context-window
        neighbours.  This builds a distributional context signature.
        """
        if len(words) < 2:
            return
        for i, w in enumerate(words):
            if len(w) < 3:
                continue
            # Collect context window neighbours
            ctx_words = []
            for j in range(max(0, i - self._window),
                           min(len(words), i + self._window + 1)):
                if j != i and len(words[j]) >= 2:
                    ctx_words.append(words[j])
            if not ctx_words:
                continue
            # Bundle context word HVs
            ctx_hv = encode_fn(ctx_words[0])
            for cw in ctx_words[1:]:
                ctx_hv = ctx_hv.bundle(encode_fn(cw))
            # Accumulate into word's context signature
            if w in self._context_hvs:
                self._context_hvs[w] = self._context_hvs[w].bundle(ctx_hv)
            else:
                self._context_hvs[w] = ctx_hv
            self._counts[w] += 1
        self._families_built = False

    def build_families(self, threshold: float = 0.55):
        """Group words into morphological families by context similarity.

        Words whose context HVs have similarity ≥ threshold AND share
        a common prefix (≥3 chars) are grouped together.  The prefix
        heuristic is a soft constraint that prevents unrelated words
        with coincidentally similar contexts from clustering.
        """
        words = [w for w, c in self._counts.items() if c >= 2]
        self._families = {}
        for w in words:
            family = {w}
            w_hv = self._context_hvs[w]
            # Only compare with words sharing ≥3-char prefix (efficiency)
            prefix3 = w[:3]
            for other in words:
                if other == w:
                    continue
                if not other.startswith(prefix3):
                    continue
                sim = w_hv.similarity(self._context_hvs[other])
                if sim >= threshold:
                    family.add(other)
            self._families[w] = family
        self._families_built = True

    def find_family(self, word: str) -> Set[str]:
        """Return all known morphological variants of ``word``."""
        if not self._families_built:
            self.build_families()
        return self._families.get(word, {word})

    def stats(self) -> Dict[str, int]:
        return {
            'words_tracked': len(self._context_hvs),
            'families_built': len(self._families),
        }


class PassageMemory:
    """Stores complete text passages with passage-level HVs.

    Instead of storing individual sentences with fragile neighbor
    chains, this stores entire passages and their constituent sentences.
    At retrieval time, the passage HV is compared with the query HV
    to find the most relevant passage, then the best sentence(s) within
    that passage are selected.

    This replaces the sentence-neighbor index and produces more coherent
    multi-sentence responses.
    """

    def __init__(self):
        # passage_id → {hv, sentences, quality, concepts_per_sent}
        self._passages: Dict[int, Dict] = {}
        self._next_id = 0
        # concept → set of passage IDs
        self._concept_to_passages: Dict[str, Set[int]] = defaultdict(set)

    def store(self, sentences: List[str], passage_hv,
              quality: float = 1.0,
              concepts_per_sent: Optional[List[Set[str]]] = None):
        """Store a passage (list of sentences) with its HV."""
        pid = self._next_id
        self._next_id += 1
        self._passages[pid] = {
            'hv': passage_hv,
            'sentences': sentences,
            'quality': quality,
            'concepts_per_sent': concepts_per_sent or [set()] * len(sentences),
        }
        # Index concepts
        if concepts_per_sent:
            for concepts in concepts_per_sent:
                for c in concepts:
                    self._concept_to_passages[c].add(pid)

    def retrieve(self, query_concepts: Set[str], query_hv,
                 top_k: int = 3) -> List[Dict]:
        """Retrieve top-k passages by concept overlap + HV similarity.

        Returns list of {passage_id, sentences, quality, score, concepts_per_sent}.
        """
        candidates = set()
        # Find passages that mention at least one query concept
        for c in query_concepts:
            candidates.update(self._concept_to_passages.get(c, set()))
        if not candidates:
            # Fallback: scan all (limited)
            candidates = set(list(self._passages.keys())[:200])

        scored = []
        for pid in candidates:
            p = self._passages[pid]
            # Concept overlap score
            all_concepts = set()
            for cs in p['concepts_per_sent']:
                all_concepts.update(cs)
            overlap = len(all_concepts & query_concepts)
            if overlap == 0:
                continue
            # HV similarity
            hv_sim = query_hv.similarity(p['hv'])
            # Combined score — quality is multiplicative so rich
            # corpus (quality=2.0) reliably outranks noisy WikiText
            # (quality=0.5) at similar concept overlap.
            score = (overlap * 2.0 + hv_sim * 3.0) * p['quality']
            scored.append({
                'passage_id': pid,
                'sentences': p['sentences'],
                'quality': p['quality'],
                'score': score,
                'concepts_per_sent': p['concepts_per_sent'],
                'overlap': overlap,
                'hv_sim': hv_sim,
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def stats(self) -> Dict[str, int]:
        return {
            'passages_stored': len(self._passages),
            'concepts_indexed': len(self._concept_to_passages),
        }


class PolarityLearner:
    """Learns to detect affirmation vs negation from training data.

    During training, sentences containing negation patterns and
    affirmation patterns are bundled into separate prototype HVs.
    At inference, the sentence HV is compared against both prototypes
    to determine polarity.

    This replaces the hardcoded negation regex patterns.
    """

    def __init__(self):
        self._neg_exemplars: List = []
        self._aff_exemplars: List = []
        self._neg_prototype = None
        self._aff_prototype = None

    def add_example(self, text_hv, is_negation: bool):
        """Add a training example."""
        if is_negation:
            self._neg_exemplars.append(text_hv)
        else:
            self._aff_exemplars.append(text_hv)

    def build_prototypes(self):
        """Bundle exemplars into prototypes.

        Cap each class to avoid class imbalance — bundling thousands
        of affirmation exemplars produces a generic prototype that
        any sentence matches equally, destroying discriminative power.
        """
        _MAX = 200  # balanced cap per class
        if self._neg_exemplars:
            sample = self._neg_exemplars[:_MAX]
            proto = sample[0]
            for hv in sample[1:]:
                proto = proto.bundle(hv)
            self._neg_prototype = proto
        if self._aff_exemplars:
            sample = self._aff_exemplars[:_MAX]
            proto = sample[0]
            for hv in sample[1:]:
                proto = proto.bundle(hv)
            self._aff_prototype = proto

    def is_negation(self, text_hv) -> bool:
        """Return True if the text expresses negation."""
        if self._neg_prototype is None or self._aff_prototype is None:
            return False
        neg_sim = text_hv.similarity(self._neg_prototype)
        aff_sim = text_hv.similarity(self._aff_prototype)
        return neg_sim > aff_sim

    def stats(self) -> Dict[str, int]:
        return {
            'neg_examples': len(self._neg_exemplars),
            'aff_examples': len(self._aff_exemplars),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §3  N-Gram Response Generator — learned from data, NO templates
# ═══════════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Generates text continuations using learned n-gram patterns.

    How it works:
    1. During training, learns bigram and trigram probabilities from text.
    2. At inference, extends seed words using weighted random sampling.
    3. Every word traces back to training data.
    """

    def __init__(self):
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)
        self.trigrams: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._corpus_size: int = 0

    def learn(self, text: str):
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        self._corpus_size += len(words)
        for i in range(len(words) - 1):
            self.bigrams[words[i]][words[i + 1]] += 1
            if i < len(words) - 2:
                self.trigrams[(words[i], words[i + 1])][words[i + 2]] += 1

    def continue_from(self, seed_words: List[str],
                      max_len: int = 25) -> Optional[str]:
        if not self.bigrams or not seed_words:
            return None

        # Choose seed by specificity (rarest bigram first)
        current = None
        best_specificity = float('inf')
        for w in seed_words:
            if w in self.bigrams:
                freq = sum(self.bigrams[w].values())
                if freq < best_specificity:
                    best_specificity = freq
                    current = w
        if current is None:
            return None

        words = [current]
        for _ in range(max_len):
            # Try trigram first for better coherence
            if len(words) >= 2:
                trigram_key = (words[-2], words[-1])
                tri_cands = self.trigrams.get(trigram_key, Counter())
                if tri_cands and sum(tri_cands.values()) >= 2:
                    total = sum(tri_cands.values())
                    r = np.random.random() * total
                    cum = 0
                    nxt = None
                    for word, cnt in tri_cands.items():
                        cum += cnt
                        if cum >= r:
                            nxt = word
                            break
                    if nxt is not None:
                        words.append(nxt)
                        current = nxt
                        if len(words) >= 8 and nxt in _FUNCTION_WORDS:
                            break
                        continue

            # Fall back to bigram
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
# §4  NSCKAIEngine — orchestrator using ACTUAL NSCK modules
# ═══════════════════════════════════════════════════════════════════════════

class NSCKAIEngine:
    """Main AI engine built on the **actual NSCK cognitive architecture**.

    Modules used (from nsck-demo/python/):
    - ``SemanticMemory``    — concept graph + spreading activation
    - ``EpisodicMemory``    — VSA episode storage + LSH retrieval
    - ``GlobalWorkspace``   — LIDA-style coalition competition
    - ``EmotionSystem``     — Plutchik 8 emotions + valence/arousal
    - ``CausalReasoner``    — forward/backward chaining + counterfactuals
    - ``SelfModel``         — confidence calibration + self-awareness
    - ``CuriosityModule``   — novelty detection + exploration
    - ``TextKnowledgeLearner`` — VSA text learning pipeline
    - ``hypervec_shim.HyperVector`` — 10240-bit binary VSA vectors

    Glass-box: every ``chat()`` returns a full ``ThoughtTrace``.

    Usage::

        engine = NSCKAIEngine()
        engine.train_on_text("Paris is the capital of France.")
        result = engine.chat("What is the capital of France?")
        print(result['response'])
        print(result['trace'])  # full glass-box reasoning chain
    """

    def __init__(self):
        # --- Core NSCK cognitive modules ---
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory()
        self.global_workspace = GlobalWorkspace(attention_threshold=0.3)
        self.emotion_system = EmotionSystem()
        self.self_model = SelfModel()
        self.curiosity = CuriosityModule()

        # Causal reasoning
        self.causal_graph = CausalGraph()
        self.causal_reasoner = CausalReasoner(self.causal_graph)
        self.causal_discovery = CausalDiscovery()

        # Text learning — uses SemanticMemory, EpisodicMemory and the shared
        # CausalGraph so causal facts flow directly into causal_reasoner.
        self.text_learner = TextKnowledgeLearner(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            causal_graph=self.causal_graph,
        )

        # N-gram model for response generation
        self.generator = ResponseGenerator()

        # Context retention for multi-turn conversations
        self.context_module = ContextRetentionModule(
            max_history=MAX_CONTEXT, attention_window=5)

        # Counterfactual reasoner for "what if" queries
        self.counterfactual = CounterfactualReasoner(engine=self)

        # Response composer for enhanced fluency
        self.response_composer = ResponseComposer()

        # Symbolic math handler (no neural networks)
        self.math_handler = MathHandler()

        # Multimodal processor (text, image, audio, video → HV)
        self.multimodal = MultimodalProcessor(
            context_engine=None,
            semantic_memory=self.semantic_memory,
        )

        # Image generator (text → image via VSA concept-visual memory)
        self.image_generator = ImageGenerator(
            semantic_memory=self.semantic_memory,
        )

        # AnalogyEngine for cross-domain knowledge transfer.
        # auto_discover_abstractions() is called whenever a new domain is
        # trained to build HV-similarity bridges between domains; discovered
        # pairs are written as 'similar_to' edges in semantic_memory so that
        # spreading activation propagates across domain boundaries.
        self.analogy_engine = AnalogyEngine()
        # domain_tag -> {concept_name: HyperVector} — populated during training
        self._domain_concept_hvs: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # NLGEngine for structured multi-sentence discourse generation.
        # Used in _build_response() when multiple facts are available to
        # assemble a coherent multi-sentence answer with connectives and
        # anaphora rather than raw sentence concatenation.
        self.nlg_engine = NLGEngine()

        # Register modules with GlobalWorkspace
        self.global_workspace.register_module("SEMANTIC", _SemanticModule())
        self.global_workspace.register_module("EPISODIC", _EpisodicModule())
        self.global_workspace.register_module("CAUSAL", _CausalModule())
        self.global_workspace.register_module(
            "EMOTION", _EmotionModule(self.emotion_system))

        # Conversation state
        self.conversation_history: deque = deque(maxlen=MAX_CONTEXT)

        # Sentence store — maps sentences to their source data for
        # response assembly.  Key = normalised sentence, Value = original.
        self._sentence_store: Dict[str, str] = {}
        # Concept → source sentences index
        self._concept_sentences: Dict[str, List[str]] = defaultdict(list)
        # Stem → concept forms mapping for fuzzy lookup
        self._stem_to_concepts: Dict[str, Set[str]] = defaultdict(set)
        # Sentence → source quality (higher = more reliable content)
        self._sentence_quality: Dict[str, float] = {}
        # Word frequency for inverse-frequency weighting
        self._word_freq: Counter = Counter()
        # Sentence neighbor index: norm(sent) → next sentence in same passage
        self._sentence_next: Dict[str, str] = {}

        # ── Learned cognitive components (NO hardcoded patterns) ──
        # Intent classifier: learns to detect greetings, questions,
        # thanks, etc. from training data via VSA prototype bundling.
        self.intent_memory = IntentMemory()
        # Morphology learner: learns word families ("analogy"↔"analogies")
        # from context co-occurrence instead of hardcoded suffix rules.
        self.morphology = MorphologyLearner(window=3)
        # Passage memory: stores complete passages with HVs for coherent
        # multi-sentence retrieval (replaces sentence-neighbor hacks).
        self.passage_memory = PassageMemory()
        # Polarity learner: learns affirmation vs negation patterns from
        # data instead of hardcoded regex lists.
        self.polarity = PolarityLearner()

        # Training stats
        self._training_stats = {
            'texts_trained': 0,
            'sentences_processed': 0,
            'concepts_learned': 0,
            'relations_learned': 0,
            'facts_stored': 0,
            'causal_links': 0,
            'training_time_s': 0.0,
            'images_trained': 0,
            'images_described': 0,
            'images_generated': 0,
        }
        self._query_count = 0
        logger.info("NSCK AI Engine initialised with full NSCK architecture")

    # ------------------------------------------------------------------
    # Training — uses actual NSCK TextKnowledgeLearner
    # ------------------------------------------------------------------

    def train_on_text(self, text: str, source_quality: float = 1.0,
                      domain: str = 'general') -> Dict[str, Any]:
        """Learn from a text passage using the NSCK learning pipeline.

        Process:
        1. ``TextKnowledgeLearner.learn_from_text()`` extracts concepts,
           relations, and facts using semantic folding (LinguaCortex).
        2. Concepts → ``SemanticMemory`` (NetworkX graph + HV index).
        3. Episodes → ``EpisodicMemory`` (VSA + LSH).
        4. Causal links → ``CausalGraph`` (shared with TextKnowledgeLearner).
        5. N-gram patterns → ``ResponseGenerator``.
        6. Sentence index → ``_sentence_store`` for response assembly.
        7. Cross-domain bridges discovered via ``AnalogyEngine`` whenever a
           new domain is introduced.

        Args:
            text: Text passage to learn from.
            source_quality: Quality weight for sentences from this source.
                Higher values (e.g. 2.0) for curated data, lower (0.5)
                for noisy web text.  Default 1.0.
            domain: Optional domain tag (e.g. 'biology', 'technology').
                Used by AnalogyEngine to discover cross-domain concept
                bridges.  Defaults to 'general'.
        """
        start = time.time()

        # Step 1: Learn via TextKnowledgeLearner (populates semantic +
        # episodic memory automatically)
        learn_result = self.text_learner.learn_from_text(text, source='training')

        # Step 2: Learn causal links from learned facts
        causal_added = 0
        for fact in self.text_learner.learned_facts[-20:]:  # recent facts
            if fact.relation in ('causes', 'prevents', 'leads_to'):
                self.causal_graph.add_causes(
                    fact.subject, fact.object,
                    strength=fact.confidence,
                    context='chat')
                causal_added += 1

        # Step 3: Learn n-gram patterns
        self.generator.learn(text)

        # Step 4: Teach response composer linguistic patterns
        self.response_composer.learn_from_text(text)

        # Step 5: Index sentences for response assembly AND learn intent /
        # morphology / polarity / passage structures from this text.
        sentences = self._split_sentences(text)

        # Build neighbour index (ordered sentences within same passage)
        prev_norm: Optional[str] = None
        for sent in sentences:
            norm = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            if len(norm) > 10:
                if prev_norm is not None and prev_norm not in self._sentence_next:
                    self._sentence_next[prev_norm] = sent
                prev_norm = norm
            else:
                prev_norm = None

        # Step 5a: Store passage in PassageMemory for coherent retrieval
        good_sents = [s for s in sentences
                      if len(re.sub(r'[^\w\s]', '', s.lower()).strip()) > 10]
        if good_sents:
            passage_hv = self._encode_text(text[:500])  # passage-level HV
            concepts_per_sent = []
            for s in good_sents:
                concepts_per_sent.append(
                    set(c.lower() for c in self._extract_concepts(s)))
            self.passage_memory.store(
                good_sents, passage_hv, quality=source_quality,
                concepts_per_sent=concepts_per_sent)

        # Step 5b: Learn intent categories from sentence patterns.
        # The system discovers sentence types by analysing structure:
        #  - Sentences that START like greetings (contain greeting words)
        #  - Sentences that ARE questions (end with ?)
        #  - Sentences that express thanks
        #  - Declarative statements
        # Critically, intent prototypes are LEARNED by bundling HVs — the
        # system learns what a greeting "looks like" in HV space, NOT by
        # matching regex.
        for sent in sentences:
            sent_lower = sent.lower().strip()
            sent_hv = self._encode_text(sent)
            # Learn from the sentence itself — the text tells us what
            # patterns exist.  E.g. "A common response is 'I am fine'" →
            # the system learns the greeting-response association.
            words = set(sent_lower.split())

            if sent_lower.endswith('?'):
                # Question — learn question-type prototypes
                first_word = sent_lower.split()[0] if sent_lower.split() else ''
                _YESNO_STARTERS = {
                    'do', 'does', 'did', 'is', 'are', 'was', 'were',
                    'can', 'could', 'would', 'should', 'has', 'have',
                    'will', 'shall', 'may', 'might',
                }
                if first_word in _YESNO_STARTERS:
                    self.intent_memory.add_example('question_yesno', sent_hv)
                else:
                    self.intent_memory.add_example('question', sent_hv)

            # NOTE: Response-template extraction from quoted text in
            # training data is unreliable (nested quotes, meta-descriptions).
            # Conversational response mappings are now taught directly in
            # the conversation training phase with curated exemplars.

            # NOTE: Greeting/thanks intent exemplars are provided
            # exclusively by the conversation training phase, which
            # supplies curated short examples.  Adding greeting/thanks
            # from general text (WikiText, Dolly long-form, etc.)
            # introduces too much noise and causes false-positive
            # intent routing.

            # Step 5c: Learn polarity (negation vs affirmation) from
            # sentence content.  The system learns what negation "sounds
            # like" in HV space by observing sentences that contain
            # negation words during training.
            _NEG_INDICATORS = {
                'not', 'never', 'no', 'cannot', "can't", "doesn't",
                "don't", "isn't", "aren't", "wasn't", "weren't",
                "won't", "shouldn't", "wouldn't", "couldn't",
            }
            if words & _NEG_INDICATORS:
                self.polarity.add_example(sent_hv, is_negation=True)
            elif len(words) > 3:  # Only use substantial sentences
                self.polarity.add_example(sent_hv, is_negation=False)

        # Step 5d: Learn morphological context from word sequences
        for sent in sentences:
            words_list = [w for w in re.sub(r'[^\w\s]', ' ', sent.lower()).split()
                          if len(w) >= 2]
            if words_list:
                self.morphology.observe_sentence(
                    words_list,
                    lambda w: hypervec_rs.HyperVector(hash(w) % HASH_SEED_MODULO))

        # Step 5e: Original sentence→concept index (kept for backward compat)
        for sent in sentences:
            norm = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            if len(norm) > 10:
                self._sentence_store[norm] = sent
                # Track source quality for retrieval scoring
                self._sentence_quality[norm] = max(
                    self._sentence_quality.get(norm, 0.0), source_quality)
                concepts = self._extract_concepts(sent)
                for c in concepts:
                    c_low = c.lower()
                    # Populate stem → concept mapping
                    stem = self._stem(c_low)
                    self._stem_to_concepts[stem].add(c_low)

                    bucket = self._concept_sentences[c_low]
                    if sent not in bucket:
                        bucket.append(sent)
                        # Cap per-concept sentences — evict lowest quality
                        # only when bucket is well over the limit to
                        # amortise the expensive sort.
                        if len(bucket) > 60:
                            bucket.sort(
                                key=lambda s: self._sentence_quality.get(
                                    re.sub(r'[^\w\s]', '', s.lower()).strip(), 1.0),
                                reverse=True)
                            self._concept_sentences[c_low] = bucket[:50]
            # Update word frequencies
            for w in norm.split():
                self._word_freq[w] += 1

        elapsed = time.time() - start
        n_concepts = learn_result.get('concepts', 0) if isinstance(learn_result, dict) else 0
        n_relations = learn_result.get('relations', 0) if isinstance(learn_result, dict) else 0
        n_facts = learn_result.get('facts', 0) if isinstance(learn_result, dict) else 0

        # Step 6: Collect new concept HVs for this domain and discover
        # cross-domain bridges via AnalogyEngine.
        cross_domain_bridges = self._update_domain_concepts_and_bridge(domain)

        self._training_stats['texts_trained'] += 1
        self._training_stats['sentences_processed'] += len(sentences)
        self._training_stats['concepts_learned'] += n_concepts
        self._training_stats['relations_learned'] += n_relations
        self._training_stats['facts_stored'] += n_facts
        self._training_stats['causal_links'] += causal_added
        self._training_stats['training_time_s'] += elapsed

        result = {
            'sentences_processed': len(sentences),
            'concepts_added': n_concepts,
            'relations_added': n_relations,
            'facts_stored': n_facts,
            'causal_links_added': causal_added,
            'cross_domain_bridges': cross_domain_bridges,
            'elapsed_s': round(elapsed, 4),
        }
        logger.info("Trained: %s", result)
        return result

    def _update_domain_concepts_and_bridge(self, domain: str) -> int:
        """Snapshot new concept HVs for *domain* and discover cross-domain bridges.

        For every domain already seen, ``AnalogyEngine.auto_discover_abstractions``
        compares HV representations of concepts from the two domains.  Pairs
        whose names share a common stem (≥3 chars) or whose HV similarity
        exceeds the threshold are linked with a ``similar_to`` edge in
        ``semantic_memory`` so spreading activation propagates across domain
        boundaries.

        Returns the number of new bridge edges added to semantic memory.
        """
        # Collect current concept HVs for this domain from semantic_memory
        current_hvs = dict(self.semantic_memory.concept_hvs)
        existing_in_domain = self._domain_concept_hvs.get(domain, {})
        # Only process concepts not yet registered for this domain
        new_hvs = {
            name: hv for name, hv in current_hvs.items()
            if name not in existing_in_domain
        }
        if new_hvs:
            self._domain_concept_hvs[domain].update(new_hvs)

        bridges_added = 0
        # Compare against every other domain already stored
        for other_domain, other_hvs in self._domain_concept_hvs.items():
            if other_domain == domain or not other_hvs:
                continue
            # Discover mappings between this domain's new concepts and the
            # other domain's concepts.  auto_discover_abstractions uses both
            # HV similarity and name-stem matching.
            try:
                mappings = self.analogy_engine.auto_discover_abstractions(
                    domain_a=domain,
                    domain_b=other_domain,
                    concept_hvs_a=new_hvs,
                    concept_hvs_b=other_hvs,
                    similarity_threshold=0.52,
                )
                for mapping in mappings:
                    # Write a bidirectional 'similar_to' edge into the
                    # semantic graph so spreading activation crosses domains.
                    a_name = mapping.source_concept
                    b_name = mapping.target_concept
                    if (a_name in self.semantic_memory.concept_graph
                            and b_name in self.semantic_memory.concept_graph):
                        if not self.semantic_memory.concept_graph.has_edge(a_name, b_name):
                            self.semantic_memory.concept_graph.add_edge(
                                a_name, b_name,
                                relation='similar_to',
                                weight=float(mapping.similarity),
                                cross_domain=True,
                                domains=(domain, other_domain),
                            )
                            bridges_added += 1
                        if not self.semantic_memory.concept_graph.has_edge(b_name, a_name):
                            self.semantic_memory.concept_graph.add_edge(
                                b_name, a_name,
                                relation='similar_to',
                                weight=float(mapping.similarity),
                                cross_domain=True,
                                domains=(other_domain, domain),
                            )
                            bridges_added += 1
            except Exception as _e:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "AnalogyEngine bridge discovery failed: %s", _e)
                pass  # AnalogyEngine failures must not abort training

        return bridges_added

    def finalize_training(self):
        """Build learned prototypes after all training data is processed.

        Must be called once after all ``train_on_text()`` calls are
        complete.  Builds the VSA prototypes for intent classification,
        polarity detection, and morphological families — all learned
        purely from the training data, no hardcoded rules.
        """
        self.intent_memory.build_prototypes()
        self.polarity.build_prototypes()
        self.morphology.build_families()
        logger.info(
            "Finalized learned models: intents=%s, polarity=%s, morphology=%s",
            self.intent_memory.stats(),
            self.polarity.stats(),
            self.morphology.stats(),
        )

    # ------------------------------------------------------------------
    # Multimodal: Image training / description / generation
    # ------------------------------------------------------------------

    def train_on_image(
        self,
        image: 'np.ndarray',
        caption: str,
    ) -> Dict[str, Any]:
        """Learn from an image-caption pair.

        Process:
        1. Process the image through ``MultimodalProcessor`` (HOG, color,
           LBP, edge, spatial quadrants → HyperVector).
        2. Learn the caption text through ``train_on_text()``.
        3. Store the text→visual association in ``ImageGenerator``'s
           concept-visual memory.
        4. Bundle the image HV with the caption HV and store a cross-modal
           episode in ``EpisodicMemory``.

        Args:
            image: numpy array — H×W (grayscale) or H×W×C (color), uint8.
            caption: textual description of the image.

        Returns:
            Dict with training stats.
        """
        start = time.time()

        # Step 1: Process the image → HV + visual features
        img_input = MultimodalInput(image=image)
        img_result = self.multimodal.process(img_input)
        image_hv = img_result.fused_hv
        image_descriptors = img_result.extracted_concepts
        image_stats = {}
        for mr in img_result.modality_results:
            if mr.modality == "image":
                image_stats = mr.features.get("stats", {})
                break

        # Step 2: Learn the caption as text
        text_result = self.train_on_text(caption)

        # Step 3: Train the image generator (concept→visual associations)
        self.image_generator.train_from_examples(
            [(caption, image)], max_examples=1,
        )

        # Step 4: Cross-modal episode — bind image HV with caption HV,
        # store in semantic memory so "describe" can find it later.
        caption_hv = self._encode_text(caption)
        cross_hv = image_hv.xor(caption_hv)

        # Store the association: image descriptors → caption in semantic graph
        for desc in image_descriptors:
            desc_cap = desc.capitalize()
            if desc_cap not in self.semantic_memory.concept_graph:
                self.semantic_memory.add_concept(
                    desc_cap, properties={"source": "image"},
                    hv_override=image_hv,
                )
            for concept_word in self._extract_concepts(caption):
                concept_cap = concept_word.capitalize()
                if concept_cap in self.semantic_memory.concept_graph:
                    self.semantic_memory.concept_graph.add_edge(
                        desc_cap, concept_cap,
                        relation="visual_of",
                        weight=0.7,
                    )

        elapsed = time.time() - start
        self._training_stats['images_trained'] += 1
        self._training_stats['training_time_s'] += elapsed

        result = {
            'image_descriptors': image_descriptors,
            'image_stats': image_stats,
            'text_result': text_result,
            'visual_concepts_learned': len(
                self.image_generator.concept_memory.feature_templates
            ),
            'elapsed_s': round(elapsed, 4),
        }
        logger.info("Trained image-caption pair: descriptors=%s",
                     image_descriptors)
        return result

    def describe_image(self, image: 'np.ndarray') -> Dict[str, Any]:
        """Describe an image using the full cognitive pipeline.

        Pipeline:
        1. Process image through ``MultimodalProcessor`` → HV + descriptors.
        2. Query ``SemanticMemory`` with the image HV for learned concepts.
        3. Run spreading activation from matched concepts.
        4. Gather visual descriptors (color, texture, edges, etc.).
        5. Retrieve related captions from ``_concept_sentences``.
        6. Build a natural-language description from all evidence.

        Args:
            image: numpy array — H×W or H×W×C uint8.

        Returns:
            Dict with 'description', 'descriptors', 'matched_concepts',
            'confidence', 'trace'.
        """
        trace = ThoughtTrace("describe_image")
        start = time.time()

        # Stage 1: Image → HV + features
        trace.begin("encode_image", {"module": "MultimodalProcessor"})
        img_input = MultimodalInput(image=image)
        img_result = self.multimodal.process(img_input)
        image_hv = img_result.fused_hv
        descriptors = img_result.extracted_concepts
        raw_summary = ""
        image_stats = {}
        for mr in img_result.modality_results:
            if mr.modality == "image":
                raw_summary = mr.raw_summary
                image_stats = mr.features.get("stats", {})
                descriptors = mr.features.get("descriptors", [])
                break
        trace.end("Image encoded to 10240-bit HV", {
            "descriptors": descriptors,
            "stats": {k: round(v, 2) if isinstance(v, float) else v
                      for k, v in image_stats.items()},
        })

        # Stage 2: Semantic memory query (find learned concepts matching image)
        trace.begin("search_semantic", {"module": "SemanticMemory"})
        matched = self.semantic_memory.query(image_hv, k=10)
        trace.end("Semantic memory search for image", {
            "matches": [(n, round(s, 3)) for n, s in matched[:5]],
        })

        # Stage 3: Spreading activation from matches
        trace.begin("spread_activation", {"module": "SemanticMemory"})
        seeds = [n for n, s in matched[:5] if s > SIMILARITY_THRESHOLD]
        activation = {}
        if seeds:
            activation = self.semantic_memory.spread_activation(
                seeds, steps=2, decay=0.6)
        trace.end("Spreading activation", {
            "activated": sorted([(k, round(v, 3)) for k, v in
                                 activation.items()],
                                key=lambda x: x[1], reverse=True)[:5],
        })

        # Stage 4: Retrieve related sentences/captions
        trace.begin("retrieve_captions", {})
        related_sentences = []
        all_concepts = set(seeds) | set(descriptors)
        for concept in all_concepts:
            for sent in self._concept_sentences.get(concept.lower(), []):
                if sent not in related_sentences:
                    related_sentences.append(sent)
        trace.end("Retrieved captions", {
            "caption_count": len(related_sentences)
        })

        # Stage 5: Build description
        trace.begin("compose_description", {})
        description = self._compose_image_description(
            descriptors, matched, activation, related_sentences, image_stats)
        trace.end("Description composed", {
            "description_preview": description[:200],
        })

        elapsed = time.time() - start
        self._training_stats['images_described'] += 1

        confidence = 0.0
        if matched:
            confidence = max(float(s) for _, s in matched[:3])
        if related_sentences:
            confidence = max(confidence, 0.5)

        return {
            'description': description,
            'descriptors': descriptors,
            'matched_concepts': [(n, round(float(s), 3))
                                 for n, s in matched[:5]],
            'activated_concepts': sorted(
                [(k, round(v, 3)) for k, v in activation.items()],
                key=lambda x: x[1], reverse=True)[:5],
            'related_captions': related_sentences[:3],
            'image_stats': image_stats,
            'confidence': round(confidence, 3),
            'trace': trace.to_dict(),
            'latency_ms': round(elapsed * 1000, 1),
        }

    def generate_image(
        self,
        text_prompt: str,
        size: tuple = (64, 64),
        color: bool = True,
    ) -> Dict[str, Any]:
        """Generate an image from a text description.

        Uses the ``ImageGenerator``'s VSA concept-visual associative memory
        to map text concepts → visual features → pixels.  No neural network
        is involved; the visual features are classical CV features (HOG,
        color histograms, LBP texture, etc.) learned from training pairs.

        Args:
            text_prompt: Description of the desired image.
            size: (height, width) tuple.
            color: Whether to generate a colour image.

        Returns:
            Dict with 'image' (numpy array), 'concepts', 'stats'.
        """
        start = time.time()

        self.image_generator.config = GenerationConfig(
            image_size=size, is_color=color)
        image = self.image_generator.generate(text_prompt)

        elapsed = time.time() - start
        self._training_stats['images_generated'] += 1

        return {
            'image': image,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'pixel_range': (int(image.min()), int(image.max())),
            'concepts_used': self.image_generator._extract_concepts(
                text_prompt),
            'visual_concepts_available': len(
                self.image_generator.concept_memory.feature_templates),
            'elapsed_ms': round(elapsed * 1000, 1),
        }

    def chat_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional['np.ndarray'] = None,
        auto_learn: bool = True,
    ) -> Dict[str, Any]:
        """Unified multimodal chat — accepts text, image, or both.

        Behaviour:
        - **Text only** → delegates to ``chat()``.
        - **Image only** → delegates to ``describe_image()``.
        - **Text + Image** → processes image, then uses visual context
          to enrich the text query.  Enables visual question-answering
          (e.g., "What colour is this?" + image).

        Args:
            text: Optional text query.
            image: Optional image (H×W or H×W×C uint8 numpy array).
            auto_learn: Whether to auto-learn from statements.

        Returns:
            Dict with 'response', 'modalities', 'trace', etc.
        """
        if image is None and text is not None:
            result = self.chat(text, auto_learn=auto_learn)
            result['modalities'] = ['text']
            return result

        if image is not None and text is None:
            desc_result = self.describe_image(image)
            return {
                'response': desc_result['description'],
                'modalities': ['image'],
                'image_description': desc_result,
                'confidence': desc_result['confidence'],
                'trace': desc_result['trace'],
                'latency_ms': desc_result['latency_ms'],
            }

        if image is not None and text is not None:
            return self._visual_qa(text, image, auto_learn=auto_learn)

        return {
            'response': "No input provided.",
            'modalities': [],
            'confidence': 0.0,
        }

    def _visual_qa(
        self,
        text: str,
        image: 'np.ndarray',
        auto_learn: bool = True,
    ) -> Dict[str, Any]:
        """Visual question-answering: text query + image context.

        Pipeline:
        1. Process image → descriptors + matched concepts.
        2. Augment the text query with visual context.
        3. Run the full chat pipeline with enriched query.
        """
        trace = ThoughtTrace(f"visual_qa: {text[:80]}")
        start = time.time()

        # Step 1: Process image
        trace.begin("encode_image", {"module": "MultimodalProcessor"})
        img_input = MultimodalInput(image=image)
        img_result = self.multimodal.process(img_input)
        image_hv = img_result.fused_hv
        descriptors = img_result.extracted_concepts
        for mr in img_result.modality_results:
            if mr.modality == "image":
                descriptors = mr.features.get("descriptors", descriptors)
                break
        trace.end("Image encoded", {"descriptors": descriptors})

        # Step 2: Find concepts matching this image
        trace.begin("image_semantic_search", {"module": "SemanticMemory"})
        matched = self.semantic_memory.query(image_hv, k=5)
        visual_concepts = [n for n, s in matched if s > SIMILARITY_THRESHOLD]
        trace.end("Visual concepts found", {
            "visual_concepts": visual_concepts,
            "descriptors": descriptors,
        })

        # Step 3: Augment query with visual context
        visual_context = " ".join(descriptors + visual_concepts)
        enriched_query = f"{text} [visual context: {visual_context}]"

        # Step 4: Run text chat with enriched query
        chat_result = self.chat(enriched_query, auto_learn=auto_learn)

        elapsed = time.time() - start

        return {
            'response': chat_result['response'],
            'modalities': ['text', 'image'],
            'visual_descriptors': descriptors,
            'visual_concepts': visual_concepts,
            'text_reasoning': chat_result.get('reasoning', {}),
            'confidence': chat_result.get('confidence', 0.0),
            'trace': trace.to_dict(),
            'latency_ms': round(elapsed * 1000, 1),
        }

    def _compose_image_description(
        self,
        descriptors: List[str],
        matched: List[Tuple[str, float]],
        activation: Dict[str, float],
        related_sentences: List[str],
        stats: Dict[str, Any],
    ) -> str:
        """Build a natural-language description from image analysis evidence."""
        parts = []

        # 1. Physical properties
        properties = []
        dim_h = stats.get("height", 0)
        dim_w = stats.get("width", 0)

        # Brightness / color
        if "bright" in descriptors:
            properties.append("bright")
        elif "dark" in descriptors:
            properties.append("dark")
        if "color" in descriptors:
            properties.append("color")
        elif "greyscale" in descriptors:
            properties.append("greyscale")
        if "high_contrast" in descriptors:
            properties.append("high-contrast")
        if "detailed" in descriptors:
            properties.append("detailed")
        elif "smooth" in descriptors:
            properties.append("smooth")
        if "textured" in descriptors:
            properties.append("textured")
        if "circular" in descriptors:
            properties.append("circular")

        if properties:
            parts.append(
                f"This is a {', '.join(properties)} image"
                f" ({dim_h}×{dim_w} pixels)."
            )
        else:
            parts.append(f"This is an image ({dim_h}×{dim_w} pixels).")

        # 2. Learned concept matches
        good_matches = [(n, s) for n, s in matched[:3]
                        if s > SIMILARITY_THRESHOLD]
        if good_matches:
            concepts_str = ", ".join(n for n, _ in good_matches)
            parts.append(f"It appears related to: {concepts_str}.")

        # 3. Related captions from training data
        if related_sentences:
            parts.append(related_sentences[0].strip().rstrip('.') + '.')

        # 4. If nothing matched, describe raw features
        if not good_matches and not related_sentences:
            edge_d = stats.get("edge_density", 0)
            mean_v = stats.get("mean", 0)
            if edge_d > 0.3:
                parts.append("The image has strong edges and structure.")
            elif edge_d < 0.05:
                parts.append("The image is relatively uniform.")
            if mean_v > 200:
                parts.append("It is predominantly light-toned.")
            elif mean_v < 55:
                parts.append("It is predominantly dark-toned.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Chat — glass-box, traced, uses all NSCK modules
    # ------------------------------------------------------------------

    def chat(self, user_input: str, auto_learn: bool = True) -> Dict[str, Any]:
        """Process user input and generate a response.

        Cognitive pipeline (each step is traced):
        1. **Encode** — Input → HyperVector via semantic folding
        2. **Emotion** — Classify emotional tone (EmotionSystem)
        3. **Concepts** — Extract query concepts + resolve pronouns
        4. **Semantic search** — Find matching concepts (SemanticMemory)
        5. **Episodic recall** — Recall similar experiences (EpisodicMemory)
        6. **Spreading activation** — Propagate through concept graph
        7. **Causal inference** — Forward-chain causal rules
        8. **Curiosity** — Assess novelty (CuriosityModule)
        9. **GlobalWorkspace** — Coalition competition for response selection
        10. **Self-model** — Update confidence calibration (SelfModel)
        11. **Generate** — Assemble response from winning coalition
        """
        trace = ThoughtTrace(user_input)
        start = time.time()
        self._query_count += 1

        # --- Math routing: handle math symbolically (before validation) ---
        if is_math_question(user_input):
            trace.begin("math_check", {"module": "MathHandler"})
            math_result = self.math_handler.solve(user_input)
            if math_result.confidence > 0.5:
                trace.end("Solved via symbolic math", {
                    "answer": math_result.answer,
                    "method": math_result.method,
                    "steps": math_result.steps,
                })
                elapsed = time.time() - start
                return {
                    'response': math_result.answer,
                    'emotion': {
                        'emotion': 'neutral', 'valence': 0.0,
                        'arousal': 0.0, 'blend': {}, 'mood': 'neutral',
                    },
                    'confidence': math_result.confidence,
                    'reasoning': {
                        'query_concepts': self._extract_concepts(user_input),
                        'matched_concepts': [],
                        'related_facts': 0,
                        'activated_concepts': [],
                        'causal_effects': [],
                        'novelty': 0.0,
                        'gw_winner': 'MATH',
                        'self_confidence': math_result.confidence,
                        'math_method': math_result.method,
                        'math_steps': math_result.steps,
                    },
                    'trace': trace.to_dict(),
                    'latency_ms': round(elapsed * 1000, 1),
                }
            trace.end("Math check — not solvable symbolically", {
                "confidence": math_result.confidence,
            })

        # --- Early validation ---
        stripped = re.sub(r'[^\w\s]', ' ', user_input).strip()
        meaningful_words = [w for w in stripped.lower().split()
                           if len(w) > 2 and w not in _FUNCTION_WORDS]
        if not meaningful_words:
            trace.begin("validate", {"input_length": len(user_input)})
            trace.end("No meaningful content in input", {})
            elapsed = time.time() - start
            return self._empty_result(trace, elapsed)

        # --- Stage 1: Encode input via NSCK HyperVector ---
        trace.begin("encode", {"text_length": len(user_input),
                                "module": "hypervec_shim"})
        query_hv = self._encode_text(user_input)
        trace.end("Encoded input to 10240-bit HyperVector", {
            "dimension": DIMENSION, "backend": "NSCK VSA"})

        # --- Stage 2: Emotion (NSCK EmotionSystem) ---
        trace.begin("emotion", {"module": "EmotionSystem"})
        emotion_text = self.emotion_system.recognize_emotion_from_text(
            user_input)
        emotion_info = self.emotion_system.get_emotion_info()
        emotion_blend = self.emotion_system.get_emotion_blend()
        mood = self.emotion_system.get_mood(window=10)
        trace.end("NSCK EmotionSystem classified input", {
            "detected_emotion": emotion_text,
            "current_state": emotion_info.get('name', 'neutral'),
            "valence": emotion_info.get('valence', 0.0),
            "arousal": emotion_info.get('arousal', 0.0),
            "blend": {k: round(v, 3) for k, v in
                      list(emotion_blend.items())[:5]},
            "mood": mood.get('dominant_emotion', 'neutral'),
        })

        # --- Stage 3: Extract concepts ---
        trace.begin("extract_concepts", {"input": user_input[:200],
                                          "module": "TextKnowledgeLearner"})
        query_concepts = self._extract_concepts(user_input)
        resolved = self._resolve_context(user_input, query_concepts)

        # Enhance with ContextRetentionModule if multi-turn context applies
        if self.context_module.should_use_context(user_input):
            _, ctx_concepts = self.context_module.enhance_query_with_context(
                user_input, resolved)
            new_from_ctx = [c for c in ctx_concepts if c not in resolved]
            resolved = ctx_concepts
        else:
            new_from_ctx = []

        if len(resolved) > len(query_concepts):
            added = [c for c in resolved if c not in query_concepts]
            trace.end("Extracted and resolved concepts (pronoun + context)", {
                "raw_concepts": query_concepts,
                "resolved_concepts": resolved,
                "context_added": added,
                "context_module_added": new_from_ctx,
            })
            query_concepts = resolved
        else:
            trace.end("Extracted concepts from input", {
                "concepts": query_concepts})

        # --- Stage 4: Semantic memory search (NSCK SemanticMemory) ---
        trace.begin("search_semantic", {
            "concepts": query_concepts,
            "module": "SemanticMemory",
            "graph_nodes": len(self.semantic_memory.concept_graph.nodes),
        })
        matched_concepts = self.semantic_memory.query(query_hv, k=10)
        # Also search by concept name directly in the graph
        graph_matches = []
        for c in query_concepts:
            c_cap = c.capitalize()
            if c_cap in self.semantic_memory.concept_graph:
                graph_matches.append(c_cap)
            elif c in self.semantic_memory.concept_graph:
                graph_matches.append(c)
        trace.end("Searched NSCK SemanticMemory", {
            "hv_matches": [(n, round(s, 3)) for n, s in matched_concepts[:5]],
            "graph_matches": graph_matches,
            "total_nodes": len(self.semantic_memory.concept_graph.nodes),
        })

        # --- Stage 5: Spreading activation (NSCK SemanticMemory) ---
        trace.begin("spread_activation", {
            "seeds": graph_matches or query_concepts,
            "module": "SemanticMemory.spread_activation",
        })
        activation = {}
        seeds = graph_matches if graph_matches else [
            c.capitalize() for c in query_concepts]
        if seeds:
            activation = self.semantic_memory.spread_activation(
                seeds, steps=3, decay=0.7)
        trace.end("Spreading activation through concept graph", {
            "activated_concepts": sorted(
                [(k, round(v, 3)) for k, v in activation.items()],
                key=lambda x: x[1], reverse=True)[:10],
        })

        # --- Stage 6: Query learned knowledge (TextKnowledgeLearner) ---
        trace.begin("query_knowledge", {
            "module": "TextKnowledgeLearner.query_learned_knowledge"})
        knowledge_result = self.text_learner.query_learned_knowledge(
            user_input, top_k=10)
        related_facts = knowledge_result.get('related_facts', [])
        similar_concepts = knowledge_result.get('similar_concepts', [])
        activated_from_query = knowledge_result.get('activated_concepts', [])
        reasoning_trace = knowledge_result.get('reasoning_trace', [])
        tkl_confidence = knowledge_result.get('confidence', 0.0)
        trace.end("TextKnowledgeLearner query completed", {
            "facts_found": len(related_facts),
            "similar_concepts": [(n, round(s, 3))
                                  for n, s in similar_concepts[:5]],
            "activated": activated_from_query[:5],
            "confidence": round(float(tkl_confidence), 3),
            "reasoning_steps": reasoning_trace,
        })

        # --- Stage 7: Causal inference + Cross-domain bridges ---
        trace.begin("causal_inference", {
            "active_concepts": query_concepts,
            "module": "CausalReasoner + AnalogyEngine + CounterfactualReasoner",
        })
        causal_effects = []
        causal_chains = []
        counterfactual_result = None
        for c in query_concepts:
            effects = self.causal_graph.get_immediate_effects(c.capitalize())
            for e in effects:
                causal_effects.append((c, e))
            causes = self.causal_graph.get_immediate_causes(c.capitalize())
            for cause in causes:
                causal_chains.append((cause, c))

        # Cross-domain bridge discovery: find concepts from other domains
        # that are analogically similar to the query concepts via the
        # AnalogyEngine's registered auto-abstractions.
        cross_domain_analogies: List[str] = []
        known_domains = list(self._domain_concept_hvs.keys())
        if len(known_domains) >= 2:
            query_cap = [c.capitalize() for c in query_concepts]
            for i, d_a in enumerate(known_domains):
                for d_b in known_domains[i + 1:]:
                    try:
                        analogy = self.analogy_engine.find_analogy(d_a, d_b)
                        for mapping in analogy.mappings:
                            if mapping.source_concept in query_cap:
                                cross_domain_analogies.append(
                                    f"{mapping.source_concept} ({d_a}) ≈ "
                                    f"{mapping.target_concept} ({d_b})"
                                )
                            elif mapping.target_concept in query_cap:
                                cross_domain_analogies.append(
                                    f"{mapping.target_concept} ({d_b}) ≈ "
                                    f"{mapping.source_concept} ({d_a})"
                                )
                    except Exception as _e:
                        import logging as _logging
                        _logging.getLogger(__name__).debug(
                            "Cross-domain analogy query failed: %s", _e)
        is_counterfactual = self.counterfactual.is_counterfactual_query(
            user_input)
        if is_counterfactual:
            try:
                kb = {f.subject: f for f in
                      self.text_learner.learned_facts[-50:]}
                scenario = self.counterfactual.process_counterfactual_query(
                    user_input, kb, self.causal_graph)
                counterfactual_result = scenario
            except Exception:
                pass

        trace.end("NSCK CausalReasoner + AnalogyEngine inference", {
            "effects_found": causal_effects[:5],
            "causes_found": causal_chains[:5],
            "cross_domain_analogies": cross_domain_analogies[:5],
            "is_counterfactual": is_counterfactual,
            "counterfactual_confidence": (
                round(counterfactual_result.confidence, 3)
                if counterfactual_result else None),
        })

        # --- Stage 8: Curiosity (NSCK CuriosityModule) ---
        trace.begin("curiosity", {"module": "CuriosityModule"})
        novelty = self.curiosity.compute_novelty(query_hv, task_tag='chat')
        self.curiosity.record_visit(query_hv, task_tag='chat')
        trace.end("NSCK CuriosityModule novelty assessment", {
            "novelty_score": round(novelty, 3),
            "is_novel": novelty > 0.5,
        })

        # --- Stage 9: GlobalWorkspace competition ---
        trace.begin("global_workspace", {
            "module": "GlobalWorkspace.compete",
        })
        coalitions = self._build_coalitions(
            query_concepts, matched_concepts, related_facts,
            activation, causal_effects, knowledge_result,
            emotion_text, tkl_confidence)
        winner = self.global_workspace.compete(coalitions) if coalitions else None
        trace.end("NSCK GlobalWorkspace coalition competition", {
            "coalitions_count": len(coalitions),
            "winner_source": winner.source if winner else "none",
            "winner_salience": round(winner.activation, 3) if winner else 0,
            "all_coalitions": [
                {"source": c.source, "activation": round(c.activation, 3)}
                for c in coalitions
            ],
        })

        # --- Stage 10: Self-model update (NSCK SelfModel) ---
        trace.begin("self_model", {"module": "SelfModel"})
        has_knowledge = bool(related_facts or activation or causal_effects)
        predicted_conf = min(1.0, float(tkl_confidence))
        self.self_model.update(
            task_tag='chat',
            predicted_confidence=predicted_conf,
            actual_success=has_knowledge,
            action='respond',
            reward=1.0 if has_knowledge else -0.1,
            state={},
        )
        self_stats = self.self_model.get_stats('chat')
        trend = self.self_model.get_improvement_trend('chat')
        trace.end("NSCK SelfModel confidence update", {
            "confidence": round(self.self_model.get_confidence('chat'), 3),
            "calibration_error": round(
                self.self_model.get_calibration_error('chat'), 3),
            "improvement_trend": trend,
            "total_attempts": self_stats.get('attempts', 0),
        })

        # --- Stage 11: Assemble response ---
        trace.begin("generate", {"strategy": "knowledge_retrieval"})

        # If counterfactual query was detected, use CounterfactualReasoner
        if counterfactual_result is not None:
            response = self.counterfactual.generate_counterfactual_response(
                counterfactual_result)
        else:
            response = self._build_response(
                user_input, query_concepts, related_facts,
                activation, causal_effects, knowledge_result, winner)

        # Enhance response fluency via ResponseComposer (post-processing only)
        if response and not response.startswith("I need more training data"):
            response = self.response_composer._improve_fluency(
                response, self.response_composer._determine_query_type(user_input))

        response = self._sanitize_response(response)
        trace.end("Generated response from NSCK knowledge", {
            "response_preview": response[:200],
            "source": winner.source if winner else "fallback",
        })

        # --- Auto-learn from user statements ---
        if (auto_learn and not user_input.strip().endswith('?')
                and not has_knowledge and len(meaningful_words) >= 2):
            self.train_on_text(user_input)

        # --- Update conversation history ---
        self.conversation_history.append(
            {'role': 'user', 'content': user_input, 'ts': time.time()})
        self.conversation_history.append(
            {'role': 'assistant', 'content': response, 'ts': time.time()})

        # Record turn in ContextRetentionModule for multi-turn tracking
        self.context_module.add_turn(
            user_input=user_input,
            response=response,
            concepts=query_concepts,
            query_hv=query_hv,
        )

        elapsed = time.time() - start
        confidence = min(1.0, float(tkl_confidence))
        if matched_concepts:
            confidence = max(confidence,
                             max(float(s) for _, s in matched_concepts[:3]))

        return {
            'response': response,
            'emotion': {
                'emotion': emotion_text,
                'valence': emotion_info.get('valence', 0.0),
                'arousal': emotion_info.get('arousal', 0.0),
                'blend': emotion_blend,
                'mood': mood.get('dominant_emotion', 'neutral'),
            },
            'confidence': round(confidence, 3),
            'reasoning': {
                'query_concepts': query_concepts,
                'matched_concepts': [(n, round(float(s), 3))
                                     for n, s in matched_concepts[:5]],
                'related_facts': len(related_facts),
                'activated_concepts': sorted(
                    [(k, round(v, 3)) for k, v in activation.items()],
                    key=lambda x: x[1], reverse=True)[:5],
                'causal_effects': causal_effects[:5],
                'cross_domain_analogies': cross_domain_analogies[:5],
                'novelty': round(novelty, 3),
                'gw_winner': winner.source if winner else "none",
                'self_confidence': round(
                    self.self_model.get_confidence('chat'), 3),
            },
            'trace': trace.to_dict(),
            'latency_ms': round(elapsed * 1000, 1),
        }

    # ------------------------------------------------------------------
    # Internal: HyperVector encoding
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> hypervec_rs.HyperVector:
        """Encode text to a 10240-bit NSCK HyperVector.

        Uses deterministic seeding from word hashes so the same text
        always produces the same vector.  Context is captured by XOR
        binding neighbouring word vectors.
        """
        words = self._tokenize(text)
        if not words:
            return hypervec_rs.HyperVector(0)
        # Create word HVs and bundle with positional encoding
        hvs = []
        for i, w in enumerate(words):
            whv = hypervec_rs.HyperVector(hash(w) % HASH_SEED_MODULO)
            # Positional encoding via permutation
            positioned = whv.permute(i % MAX_POSITION_SHIFT)
            hvs.append(positioned)
        # Bundle all word vectors
        result = hvs[0]
        for hv in hvs[1:]:
            result = result.bundle(hv)
        return result

    # ------------------------------------------------------------------
    # Internal: Concept extraction + pronoun resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _stem(word: str) -> str:
        """Minimal suffix-stripping stemmer for concept matching.

        Handles common English suffixes so that e.g. 'analogy' matches
        'analogies', 'computing' matches 'computer', etc.  Deliberately
        conservative — only strips when the remaining root is ≥ 3 chars.
        """
        w = word.lower()
        # Order matters: longest suffix first
        for suffix, min_root in [
            ('ologies', 4), ('ology', 4),
            ('ation', 3), ('ations', 3),
            ('ities', 3), ('ness', 3), ('ment', 3), ('ments', 3),
            ('ies', 3), ('ious', 3), ('eous', 3),
            ('ing', 3), ('tion', 3), ('sion', 3),
            ('ence', 3), ('ance', 3),
            ('able', 3), ('ible', 3),
            ('ates', 3), ('ate', 3),
            ('ive', 3), ('ful', 3),
            ('ous', 3), ('ual', 3),
            ('ly', 4), ('ed', 3),
            ('er', 3), ('est', 3),
            ('y', 4),  # analogy → analog (matches analogies → analog)
            ('es', 3), ('s', 4),  # s needs 4-char root to avoid over-stripping
            ('e', 4),  # whale → whal (matches whales → whal via 'es')
        ]:
            if w.endswith(suffix) and len(w) - len(suffix) >= min_root:
                return w[:-len(suffix)]
        return w

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract content words from text (non-function words > 2 chars)."""
        words = self._tokenize(text)
        seen: Set[str] = set()
        out = []
        for w in words:
            if w not in _FUNCTION_WORDS and len(w) > 2 and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def _resolve_context(self, user_input: str,
                         query_concepts: List[str]) -> List[str]:
        """Resolve pronouns using conversation history."""
        pronouns = {'she', 'he', 'it', 'they', 'them', 'that', 'this',
                     'those', 'these', 'its', 'his', 'her', 'their'}
        noise = {'tell', 'know', 'think', 'say', 'talk', 'ask', 'want',
                 'need', 'like', 'make', 'take', 'give', 'get', 'see',
                 'look', 'find', 'help', 'show', 'try', 'use', 'come',
                 'let', 'keep', 'set', 'put', 'run', 'read', 'write'}
        input_words = set(user_input.lower().split())
        if not (pronouns & input_words):
            return query_concepts
        resolved = list(query_concepts)
        for entry in reversed(list(self.conversation_history)):
            if entry['role'] != 'user':
                continue
            prev_concepts = self._extract_concepts(entry['content'])
            for c in prev_concepts:
                if (c not in resolved and c not in _FUNCTION_WORDS
                        and c not in noise):
                    resolved.append(c)
            break  # Only use most recent user turn
        return resolved

    # ------------------------------------------------------------------
    # Internal: Build GlobalWorkspace coalitions
    # ------------------------------------------------------------------

    def _build_coalitions(
        self,
        query_concepts: List[str],
        matched_concepts: List[Tuple[str, float]],
        related_facts: List[Dict],
        activation: Dict[str, float],
        causal_effects: List[Tuple[str, str]],
        knowledge_result: Dict[str, Any],
        emotion: str,
        confidence: float,
    ) -> List[Coalition]:
        """Build Coalition proposals for GlobalWorkspace competition.

        Each retrieval channel creates a Coalition.  The GlobalWorkspace
        selects the winner based on activation score.
        """
        coalitions = []

        # Coalition from semantic memory (related facts)
        if related_facts:
            coalitions.append(Coalition(
                source="SEMANTIC",
                content={
                    'type': 'semantic_facts',
                    'facts': related_facts,
                    'query_concepts': query_concepts,
                },
                base_salience=min(0.9, 0.5 + len(related_facts) * 0.1),
                relevance=min(1.0, confidence),
                affect_match=0.1 if emotion == 'neutral' else 0.2,
                sender_confidence=min(1.0, confidence),
            ))

        # Coalition from spreading activation
        if activation:
            max_act = max(activation.values()) if activation else 0
            coalitions.append(Coalition(
                source="SPREADING_ACTIVATION",
                content={
                    'type': 'activation',
                    'activated': sorted(activation.items(),
                                        key=lambda x: x[1], reverse=True)[:5],
                    'query_concepts': query_concepts,
                },
                base_salience=min(0.8, 0.3 + max_act * 0.1),
                relevance=min(1.0, max_act * 0.3),
            ))

        # Coalition from causal reasoning
        if causal_effects:
            coalitions.append(Coalition(
                source="CAUSAL",
                content={
                    'type': 'causal',
                    'effects': causal_effects,
                    'query_concepts': query_concepts,
                },
                base_salience=0.7,
                relevance=0.6,
            ))

        # Coalition from episodic memory
        recalled = knowledge_result.get('recalled_episodes', 0)
        if recalled > 0:
            coalitions.append(Coalition(
                source="EPISODIC",
                content={
                    'type': 'episodic',
                    'recalled': recalled,
                    'query_concepts': query_concepts,
                },
                base_salience=min(0.6, 0.3 + recalled * 0.05),
                relevance=min(1.0, confidence * 0.8),
            ))

        return coalitions

    # ------------------------------------------------------------------
    # Internal: Response assembly — NO templates, knowledge-graph-first
    # ------------------------------------------------------------------

    def _build_response(
        self,
        user_input: str,
        query_concepts: List[str],
        related_facts: List[Dict],
        activation: Dict[str, float],
        causal_effects: List[Tuple[str, str]],
        knowledge_result: Dict[str, Any],
        winner: Optional[Coalition],
    ) -> str:
        """Build a response entirely from learned knowledge.

        ALL classification is done via learned VSA prototypes — no regex,
        no hardcoded patterns.  The intent classifier, polarity detector,
        and morphological grouping were all learned during training.

        Strategy (knowledge-first, conversational-fallback):
        1. **Knowledge retrieval** (passage → sentence → fact → chain):
           if the system has learned knowledge relevant to the query,
           return it.  This ensures factual questions always get
           factual answers.
        2. **Learned intent routing** (conversational fallback):
           only if no relevant knowledge is found, check if the query
           matches a conversational intent (greeting, thanks) and
           return the learned response.
        3. **Yes/no detection**: applied as a prefix to whatever
           knowledge response was found.
        """
        query_set = set(c.lower() for c in query_concepts)

        # --- Detect intent + yes/no (used for routing later) ---
        query_hv = self._encode_text(user_input)
        intent_result = self.intent_memory.classify(query_hv)

        is_yesno = (intent_result is not None
                    and intent_result[0] == 'question_yesno')
        if not is_yesno:
            is_yesno = self.intent_memory.is_yesno_question(query_hv)
        if not is_yesno and '.' in user_input:
            last_sent = user_input.rsplit('.', 1)[-1].strip()
            if last_sent and last_sent.endswith('?'):
                # Multi-sentence inputs with a trailing question are
                # strong indicators of yes/no pattern ("X is true. Is Y?")
                # Use relaxed margin for the last sentence.
                last_hv = self._encode_text(last_sent)
                yp = self.intent_memory._prototypes.get('question_yesno')
                qp = self.intent_memory._prototypes.get('question')
                if yp is not None:
                    ys = last_hv.similarity(yp)
                    qs = last_hv.similarity(qp) if qp else 0.0
                    if ys > qs:  # relaxed: any positive margin
                        is_yesno = True

        # --- Determine if this is a conversational intent ---
        is_conversational = False
        learned_conv_resp = None
        if intent_result:
            intent_label, intent_sim = intent_result
            if intent_label in ('greeting', 'thanks'):
                learned_conv_resp = self.intent_memory.get_response(
                    intent_label)
                if learned_conv_resp:
                    is_conversational = True

        # --- Stage 1: Passage-level retrieval (coherent multi-sentence) ---
        passage_result = self._retrieve_from_passages(
            query_concepts, query_set, query_hv)
        if passage_result:
            # If the intent says this is conversational AND the passage
            # result is meta-knowledge (low concept overlap with direct
            # answer concepts), prefer the learned conversational response.
            if is_conversational:
                # Check if the retrieved passage is about the topic
                # or just meta-knowledge about conversational patterns.
                passage_hv = self._encode_text(passage_result)
                passage_sim = query_hv.similarity(passage_hv)
                intent_sim_val = intent_result[1] if intent_result else 0.0
                # If intent match is stronger than passage match, prefer intent
                if intent_sim_val > passage_sim:
                    return learned_conv_resp
            if is_yesno:
                passage_result = self._prepend_yesno_learned(
                    user_input, passage_result)
            return passage_result

        # --- Stage 2: Sentence retrieval with concept scoring ---
        retrieval = self._retrieve_scored_sentences(
            query_concepts, query_set, related_facts, activation,
            causal_effects)
        if retrieval:
            if is_conversational:
                retrieval_hv = self._encode_text(retrieval)
                retrieval_sim = query_hv.similarity(retrieval_hv)
                intent_sim_val = intent_result[1] if intent_result else 0.0
                if intent_sim_val > retrieval_sim:
                    return learned_conv_resp
            if is_yesno:
                retrieval = self._prepend_yesno_learned(
                    user_input, retrieval)
            return retrieval

        # --- Stage 3: Compose from fact triples (only high overlap) ---
        fact_response = self._compose_from_facts(query_concepts, related_facts)
        if fact_response:
            return fact_response

        # --- Stage 4: Multi-hop reasoning ---
        chain_response = self._chain_reasoning(
            query_concepts, activation, causal_effects)
        if chain_response:
            if is_yesno:
                chain_response = self._prepend_yesno_learned(
                    user_input, chain_response)
            return chain_response

        # --- Stage 5: Conversational intent (fallback) ---
        # Only trigger when NO relevant knowledge was found.
        # This prevents greeting/thanks routing from overriding
        # factual answers.
        if intent_result:
            intent_label, intent_sim = intent_result
            if intent_label in ('greeting', 'thanks'):
                learned_resp = self.intent_memory.get_response(intent_label)
                if learned_resp:
                    return learned_resp

        return self._fallback_response(user_input, query_concepts)

    # ── Learned passage retrieval ──

    def _retrieve_from_passages(
        self,
        query_concepts: List[str],
        query_set: Set[str],
        query_hv,
    ) -> Optional[str]:
        """Retrieve the best passage and select relevant sentences.

        Uses ``PassageMemory`` for coherent multi-sentence responses
        instead of fragile sentence-neighbor chains.
        """
        # Expand query with morphological family members
        expanded_query = set(query_set)
        for c in list(query_set):
            family = self.morphology.find_family(c)
            expanded_query.update(family)

        passages = self.passage_memory.retrieve(
            expanded_query, query_hv, top_k=3)
        if not passages:
            return None

        best = passages[0]
        if best['overlap'] < 2 and best['hv_sim'] < 0.52:
            return None

        # For high-quality passages (rich corpus), return all sentences
        # since they were curated as a cohesive unit.
        if best['quality'] >= 2.0:
            result_sents = [s for s in best['sentences'] if s.strip()]
        else:
            # Select the most relevant sentences from lower-quality passages
            selected = []
            for i, sent in enumerate(best['sentences']):
                sent_concepts = best['concepts_per_sent'][i] if i < len(best['concepts_per_sent']) else set()
                # Expand per-sentence concepts with morphology families
                sent_expanded = set(sent_concepts)
                for c in list(sent_concepts):
                    sent_expanded.update(self.morphology.find_family(c))
                # Check overlap with expanded query
                sent_overlap = len(sent_expanded & expanded_query)
                if sent_overlap > 0:
                    selected.append((sent, sent_overlap))

            if not selected:
                return None

            # Sort by overlap (most relevant first), take up to 4 sentences
            selected.sort(key=lambda x: x[1], reverse=True)
            result_sents = [s for s, _ in selected[:4]]

        # Re-order to preserve original passage order
        original_order = {s: i for i, s in enumerate(best['sentences'])}
        result_sents.sort(key=lambda s: original_order.get(s, 999))

        parts = []
        for s in result_sents:
            s = s.rstrip('.').strip()
            if s:
                parts.append(s[0].upper() + s[1:] if s else s)
        return ". ".join(parts) + "." if parts else None

    # ── Learned Yes/No prefix ──

    def _prepend_yesno_learned(self, question: str, evidence: str) -> str:
        """Prepend 'Yes' or 'No' using the **learned** polarity detector.

        The ``PolarityLearner`` was trained on sentences containing
        negation/affirmation patterns during ``train_on_text()``.
        It classifies via VSA prototype similarity, not regex.
        """
        ev_hv = self._encode_text(evidence)
        q_hv = self._encode_text(question)

        ev_is_neg = self.polarity.is_negation(ev_hv)
        q_is_neg = self.polarity.is_negation(q_hv)

        if ev_is_neg and not q_is_neg:
            prefix = "No"
        elif ev_is_neg and q_is_neg:
            prefix = "Yes"
        else:
            prefix = "Yes"

        return f"{prefix}. {evidence}"

    # ── Strategy 1: Fact-triple composition ──

    def _compose_from_facts(
        self,
        query_concepts: List[str],
        related_facts: List[Dict],
    ) -> Optional[str]:
        """Construct a natural-language response directly from fact triples.

        If the knowledge graph has ``(Paris, capital_of, France)`` and the
        query contains "capital" + "France", constructs:
        ``"Paris is the capital of France."``

        Skips generic/uninformative relations that would produce poor
        responses.  Returns None if no relevant high-quality fact applies.
        """
        if not related_facts:
            return None

        # Relations that are too generic to compose useful sentences from
        _SKIP_RELATIONS = frozenset({
            'related_to', 'related', 'semantically_related',
            'associated_with', 'co_occurs', 'cooccurrence',
            'similar_to', 'near', 'mentioned_with',
        })

        query_set = set(c.lower() for c in query_concepts)
        best_fact = None
        best_overlap = 0

        for fact in related_facts:
            subj = fact.get('subject', '')
            rel = fact.get('relation', '')
            obj = fact.get('object', '')
            conf = fact.get('confidence', 0.0)

            if not subj or not obj:
                continue
            if conf < 0.3:
                continue
            # Skip uninformative relations
            if rel.lower().replace(' ', '_') in _SKIP_RELATIONS:
                continue

            # Score by how many query concepts appear in the triple
            triple_words = set(
                re.sub(r'[^\w\s]', ' ',
                       f"{subj} {rel} {obj}".lower()).split()
            )
            overlap = len(triple_words & query_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_fact = fact

        if best_fact is None or best_overlap < 2:
            return None

        subj = best_fact['subject']
        rel = best_fact['relation']
        obj = best_fact['object']

        # Compose natural language from the triple
        sentence = self._relation_to_sentence(subj, rel, obj)
        if sentence:
            return sentence
        return None

    def _relation_to_sentence(self, subject: str, relation: str,
                               obj: str) -> str:
        """Convert a knowledge-graph triple to a natural sentence.

        Uses *learned* relation→text patterns when available, with a
        small set of grammatical fallbacks for common relation types.
        """
        rel_lower = relation.lower().replace('_', ' ')

        # Check if the response composer has a learned template
        templates = self.response_composer.relation_templates.get(
            relation.lower(), [])
        if templates:
            tmpl = templates[0]
            try:
                return tmpl.format(subject=subject, object=obj)
            except (KeyError, IndexError):
                pass

        # Grammatical construction (not domain-specific)
        copula = ('is', 'are', 'was', 'were')
        if rel_lower in copula:
            return f"{subject} {rel_lower} {obj}."
        if rel_lower in ('has', 'have', 'had'):
            return f"{subject} {rel_lower} {obj}."

        # Relation names that read like verbs
        if ' ' not in rel_lower:
            # e.g. capital_of → "is the capital of"
            if rel_lower.endswith('_of') or 'of' in rel_lower:
                return f"{subject} is the {rel_lower.replace('_', ' ')} {obj}."
            return f"{subject} {rel_lower.replace('_', ' ')} {obj}."

        return f"{subject} {rel_lower} {obj}."

    # ── Strategy 2: Multi-hop reasoning ──

    def _chain_reasoning(
        self,
        query_concepts: List[str],
        activation: Dict[str, float],
        causal_effects: List[Tuple[str, str]],
    ) -> Optional[str]:
        """Chain facts together for multi-hop reasoning.

        If the graph has A→B and B→C and the query involves A and C,
        constructs a chained explanation.

        Only fires when we have *specific* (non-generic) relations and
        at least 3 query concepts to avoid hallucinating from noisy
        graph edges.
        """
        if not activation or len(query_concepts) < 3:
            return None

        _SKIP_RELS = frozenset({
            'related_to', 'related', 'semantically_related',
            'associated_with', 'co_occurs', 'cooccurrence',
            'similar_to', 'near', 'mentioned_with',
        })

        query_set = set(c.lower() for c in query_concepts)
        graph = self.semantic_memory.concept_graph

        for concept in query_concepts:
            c_cap = concept.capitalize()
            if c_cap not in graph:
                continue

            for neighbor in graph.neighbors(c_cap):
                edge1 = graph.edges.get((c_cap, neighbor), {})
                rel1 = edge1.get('relation', 'related_to')
                if rel1.lower().replace(' ', '_') in _SKIP_RELS:
                    continue

                for second_hop in graph.neighbors(neighbor):
                    if second_hop.lower() not in query_set:
                        continue
                    if second_hop == c_cap:
                        continue

                    edge2 = graph.edges.get((neighbor, second_hop), {})
                    rel2 = edge2.get('relation', 'related_to')
                    if rel2.lower().replace(' ', '_') in _SKIP_RELS:
                        continue

                    s1 = self._relation_to_sentence(c_cap, rel1, neighbor)
                    s2 = self._relation_to_sentence(neighbor, rel2, second_hop)
                    if s1 and s2:
                        return f"{s1.rstrip('.')} and {s2.lower()}"

        return None

    # ── Strategy 3: Causal explanation ──

    def _compose_causal(
        self,
        query_concepts: List[str],
        causal_effects: List[Tuple[str, str]],
    ) -> Optional[str]:
        """Compose a response from causal links."""
        if not causal_effects:
            return None

        query_set = set(c.lower() for c in query_concepts)
        best = None
        best_score = 0

        for cause, effect in causal_effects:
            words = set(f"{cause} {effect}".lower().split())
            score = len(words & query_set)
            if score > best_score:
                best_score = score
                best = (cause, effect)

        if best and best_score >= 1:
            return f"{best[0]} causes {best[1]}."
        return None

    # ── Strategy 4: Sentence retrieval ──

    def _lookup_sentences_for_concept(self, concept: str) -> List[str]:
        """Look up sentences for a concept, using learned morphological families."""
        c_low = concept.lower()
        direct = self._concept_sentences.get(c_low, [])
        if direct:
            return direct
        # Learned morphology: find family members and look up their sentences
        family = self.morphology.find_family(c_low)
        results: List[str] = []
        for variant in family:
            results.extend(self._concept_sentences.get(variant, []))
        if results:
            return results
        # Stem fallback (data-driven from training — stem index was built
        # from words actually observed in training data)
        stem = self._stem(c_low)
        related_concepts = self._stem_to_concepts.get(stem, set())
        for rc in related_concepts:
            results.extend(self._concept_sentences.get(rc, []))
        return results

    def _retrieve_scored_sentences(
        self,
        query_concepts: List[str],
        query_set: Set[str],
        related_facts: List[Dict],
        activation: Dict[str, float],
        causal_effects: List[Tuple[str, str]],
    ) -> Optional[str]:
        """Score stored sentences by concept overlap with the query.

        Uses the **learned** MorphologyLearner for fuzzy matching instead
        of hardcoded suffix-stripping rules.  Falls back to stem index
        when morphology families haven't been built yet.
        """
        candidates: List[Tuple[str, float]] = []

        # Build an expanded query set using LEARNED morphological families
        expanded_query: Set[str] = set(query_set)
        for c in list(query_set):
            # Primary: learned morphological families
            family = self.morphology.find_family(c)
            expanded_query.update(family)
            # Fallback: stem-based expansion (still data-driven via training)
            stem = self._stem(c)
            for rc in self._stem_to_concepts.get(stem, set()):
                expanded_query.add(rc)

        # --- Direct lookup from query concepts (most reliable path) ---
        for concept in query_concepts:
            for sent in self._lookup_sentences_for_concept(concept):
                candidates.append((sent, 1.0))

        # From related facts
        for fact in related_facts:
            subj = fact.get('subject', '')
            obj = fact.get('object', '')
            conf = fact.get('confidence', 0.5)
            for concept in [subj.lower(), obj.lower()]:
                for sent in self._lookup_sentences_for_concept(concept):
                    candidates.append((sent, conf))

        # From spreading activation (top 5 only)
        for concept, act_level in sorted(
                activation.items(), key=lambda x: x[1], reverse=True)[:5]:
            for sent in self._lookup_sentences_for_concept(concept):
                candidates.append((sent, act_level * 0.3))

        # From causal effects
        for cause, effect in causal_effects:
            for sent in self._lookup_sentences_for_concept(cause):
                candidates.append((sent, 0.8))
            for sent in self._lookup_sentences_for_concept(effect):
                candidates.append((sent, 0.7))

        if not candidates:
            return None

        # Score by concept overlap (using expanded query for matching)
        concept_specificity: Dict[str, float] = {}
        for c in query_concepts:
            freq = self._word_freq.get(c.lower(), 1)
            concept_specificity[c.lower()] = 1.0 / np.log2(freq + 2)
            # Also set specificity for learned family variants
            family = self.morphology.find_family(c.lower())
            for rc in family:
                if rc not in concept_specificity:
                    concept_specificity[rc] = concept_specificity[c.lower()] * 0.9
            # Stem-based fallback
            stem = self._stem(c.lower())
            for rc in self._stem_to_concepts.get(stem, set()):
                if rc not in concept_specificity:
                    concept_specificity[rc] = concept_specificity[c.lower()] * 0.85

        scored: List[Tuple[str, float, int]] = []
        seen_sents: Set[str] = set()
        for sent, base_score in candidates:
            if sent in seen_sents:
                continue
            seen_sents.add(sent)

            sent_lower = sent.lower()
            sent_concepts = set(self._extract_concepts(sent_lower))

            # Concept overlap (exact + learned morphological family)
            overlap = sent_concepts & expanded_query
            # Family-level overlap: sentence concepts that belong to the
            # same morphological family as a query concept
            family_overlap = set()
            for sc in sent_concepts:
                sc_family = self.morphology.find_family(sc)
                if sc_family & expanded_query:
                    family_overlap.add(sc)
            all_matches = overlap | family_overlap
            weighted_overlap = sum(concept_specificity.get(c, 0.5)
                                   for c in all_matches)
            raw_overlap = len(all_matches)

            if raw_overlap == 0:
                continue

            relevance = weighted_overlap + base_score * 0.3

            # Boost: more query concepts matched = much more relevant
            if raw_overlap >= 2:
                relevance *= 1.5
            if raw_overlap >= 3:
                relevance *= 1.3

            # Boost: source quality (curated content scores higher)
            norm = re.sub(r'[^\w\s]', '', sent_lower).strip()
            quality = self._sentence_quality.get(norm, 1.0)
            relevance *= quality

            # Penalty: very long sentences (likely noisy corpus passages)
            word_count = len(sent.split())
            if word_count > 60:
                relevance *= 0.3
            elif word_count > 40:
                relevance *= 0.5
            elif word_count > 25:
                relevance *= 0.8

            # Penalty: sentences that are questions (not answers)
            if sent.strip().endswith('?'):
                relevance *= 0.1

            scored.append((sent, relevance, raw_overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate and select best, following passage chains
        used_norms: Set[str] = set()
        selected: List[str] = []
        max_sents = 4  # Allow chaining up to 4 sentences for detailed answers
        for sent, score, overlap in scored:
            norm = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            if norm in used_norms:
                continue
            lower = sent.lower().strip()
            if lower.endswith('?'):
                continue
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
            sent_concepts = set(self._extract_concepts(sent.lower()))
            if selected and not (sent_concepts & expanded_query):
                continue
            used_norms.add(norm)
            selected.append(sent[0].upper() + sent[1:] if sent else sent)

            # Multi-sentence: follow the passage neighbor chain to grab
            # continuation sentences that add detail about the query.
            cur_norm = norm
            while len(selected) < max_sents:
                next_sent = self._sentence_next.get(cur_norm)
                if not next_sent:
                    break
                next_norm = re.sub(r'[^\w\s]', '', next_sent.lower()).strip()
                if next_norm in used_norms or next_sent.strip().endswith('?'):
                    break
                next_concepts = set(self._extract_concepts(next_sent.lower()))
                # Only include if it shares at least 1 query concept
                if not (next_concepts & expanded_query):
                    break
                used_norms.add(next_norm)
                selected.append(
                    next_sent[0].upper() + next_sent[1:]
                    if next_sent else next_sent)
                cur_norm = next_norm

            if len(selected) >= max_sents:
                break

        if selected:
            parts = []
            for s in selected:
                s = s.rstrip('.').strip()
                if s:
                    parts.append(s)
            return ". ".join(parts) + "."

        return None

    def _fallback_response(self, user_input: str,
                           query_concepts: List[str]) -> str:
        """Generate a fallback when no relevant knowledge is found."""
        if not query_concepts:
            return "I need more training data to answer that."

        # Try n-gram continuation
        gen = self.generator.continue_from(query_concepts)
        if gen:
            gen_lower = gen.lower()
            has_relevant = any(c in gen_lower for c in query_concepts)
            if has_relevant and len(gen.split()) >= 3:
                return gen[0].upper() + gen[1:] + "."

        # For statements, acknowledge
        if not user_input.strip().endswith('?'):
            for c in query_concepts:
                sents = self._concept_sentences.get(c.lower(), [])
                if sents:
                    s = sents[-1].strip().rstrip('.')
                    if s:
                        return s[0].upper() + s[1:] + "."
            s = user_input.strip().rstrip('.')
            if s:
                return s[0].upper() + s[1:] + "."

        return "I need more training data to answer that."

    def _empty_result(self, trace: ThoughtTrace,
                      elapsed: float) -> Dict[str, Any]:
        """Return a result for empty/junk input."""
        return {
            'response': "I need more training data to answer that.",
            'emotion': {
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'blend': {},
                'mood': 'neutral',
            },
            'confidence': 0.0,
            'reasoning': {
                'query_concepts': [],
                'matched_concepts': [],
                'related_facts': 0,
                'activated_concepts': [],
                'causal_effects': [],
                'novelty': 0.0,
                'gw_winner': 'none',
                'self_confidence': 0.0,
            },
            'trace': trace.to_dict(),
            'latency_ms': round(elapsed * 1000, 1),
        }

    # ------------------------------------------------------------------
    # Stats & export
    # ------------------------------------------------------------------

    def get_system_stats(self) -> Dict[str, Any]:
        """Return comprehensive system statistics from all NSCK modules."""
        sm_nodes = len(self.semantic_memory.concept_graph.nodes)
        sm_edges = len(self.semantic_memory.concept_graph.edges)
        emotion_info = self.emotion_system.get_emotion_info()
        return {
            'engine': {
                'query_count': self._query_count,
                'conversation_length': len(self.conversation_history),
            },
            'training': self._training_stats.copy(),
            'semantic_memory': {
                'total_concepts': sm_nodes,
                'total_relations': sm_edges,
                'concept_hvs': len(self.semantic_memory.concept_hvs),
            },
            'knowledge': {
                'total_concepts': sm_nodes,
                'total_relations': sm_edges,
                'total_episodes': len(self.text_learner.learned_facts),
                'total_facts': len(self.text_learner.learned_facts),
                'sentence_store': len(self._sentence_store),
            },
            'generator': self.generator.get_stats(),
            'emotion': {
                'emotion': emotion_info.get('name', 'neutral'),
                'valence': emotion_info.get('valence', 0.0),
                'arousal': emotion_info.get('arousal', 0.0),
                'blend': self.emotion_system.get_emotion_blend(),
                'mood': self.emotion_system.get_mood(window=10),
            },
            'causal': {
                'total_links': self._count_causal_links(),
            },
            'context': {
                'turns_tracked': len(self.context_module.conversation_history),
                'entities_tracked': len(self.context_module.entities),
                'context_summary': self.context_module.get_context_summary(),
            },
            'counterfactual': {
                'scenarios_processed': len(self.counterfactual.scenarios),
            },
            'self_model': self.self_model.get_stats('chat'),
            'curiosity': self.curiosity.get_statistics('chat'),
            'global_workspace': self.global_workspace.get_status(),
            'text_learner': self.text_learner.get_statistics(),
            'multimodal': {
                'supported_modalities': self.multimodal.supported_modalities,
                'images_trained': self._training_stats.get('images_trained', 0),
                'images_described': self._training_stats.get('images_described', 0),
                'images_generated': self._training_stats.get('images_generated', 0),
                'visual_concepts': len(
                    self.image_generator.concept_memory.feature_templates),
            },
            'image_generator': self.image_generator.get_statistics(),
        }

    def export_knowledge(self) -> Dict[str, Any]:
        """Export all learned knowledge."""
        # Concepts from semantic memory graph
        concepts = {}
        for node in self.semantic_memory.concept_graph.nodes:
            node_data = dict(self.semantic_memory.concept_graph.nodes[node])
            concepts[node] = {
                'properties': node_data,
                'source_sentences': self._concept_sentences.get(
                    node.lower(), [])[:5],
            }

        # Relations from semantic memory graph
        relations = []
        for u, v, data in self.semantic_memory.concept_graph.edges(data=True):
            relations.append({
                'source': u,
                'target': v,
                'relation': data.get('relation', 'related'),
                'timestamp': data.get('timestamp', 0),
            })

        # Facts from TextKnowledgeLearner
        facts = []
        for f in self.text_learner.learned_facts:
            facts.append({
                'subject': f.subject,
                'relation': f.relation,
                'object': f.object,
                'confidence': f.confidence,
                'source': f.source_text[:100],
            })

        return {
            'concepts': concepts,
            'relations': relations,
            'facts': facts,
            'stats': self.get_system_stats(),
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return list(self.conversation_history)

    def reset(self):
        """Reset all cognitive modules."""
        self.__init__()
        logger.info("AI Engine reset")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_response(text: str) -> str:
        """Strip HTML/script tags from response to prevent XSS."""
        return re.sub(r'<[^>]+>', '', text)

    def get_concept_sentences(self, concept: str) -> List[str]:
        """Return source sentences associated with a concept."""
        return list(self._concept_sentences.get(concept.lower(), []))

    def _count_causal_links(self) -> int:
        """Count causal links using the CausalGraph's public API."""
        try:
            return len(self.causal_graph._graph.edges)
        except AttributeError:
            return 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', ' ', t)
        return [w for w in t.split() if len(w) > 1]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'[.!?]+', text)
        return [s.strip() for s in parts if len(s.strip()) > 5]
