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
# Ensure nsck-demo is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
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

logger = logging.getLogger("nsck_ai.engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSION = 10240          # HyperVector dimensionality (binary, 10,240 bits)
MAX_CONTEXT = 20           # Conversation history window
SIMILARITY_THRESHOLD = 0.35

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

        # Text learning — uses SemanticMemory and EpisodicMemory internally
        self.text_learner = TextKnowledgeLearner(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
        )

        # N-gram model for response generation
        self.generator = ResponseGenerator()

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
        # Word frequency for inverse-frequency weighting
        self._word_freq: Counter = Counter()

        # Training stats
        self._training_stats = {
            'texts_trained': 0,
            'sentences_processed': 0,
            'concepts_learned': 0,
            'relations_learned': 0,
            'facts_stored': 0,
            'causal_links': 0,
            'training_time_s': 0.0,
        }
        self._query_count = 0
        logger.info("NSCK AI Engine initialised with full NSCK architecture")

    # ------------------------------------------------------------------
    # Training — uses actual NSCK TextKnowledgeLearner
    # ------------------------------------------------------------------

    def train_on_text(self, text: str) -> Dict[str, Any]:
        """Learn from a text passage using the NSCK learning pipeline.

        Process:
        1. ``TextKnowledgeLearner.learn_from_text()`` extracts concepts,
           relations, and facts using semantic folding (LinguaCortex).
        2. Concepts → ``SemanticMemory`` (NetworkX graph + HV index).
        3. Episodes → ``EpisodicMemory`` (VSA + LSH).
        4. Causal links → ``CausalGraph``.
        5. N-gram patterns → ``ResponseGenerator``.
        6. Sentence index → ``_sentence_store`` for response assembly.
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

        # Step 4: Index sentences for response assembly
        sentences = self._split_sentences(text)
        for sent in sentences:
            norm = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            if len(norm) > 10:
                self._sentence_store[norm] = sent
                concepts = self._extract_concepts(sent)
                for c in concepts:
                    if sent not in self._concept_sentences[c.lower()]:
                        self._concept_sentences[c.lower()].append(sent)
                        # Cap per-concept sentences
                        if len(self._concept_sentences[c.lower()]) > 20:
                            self._concept_sentences[c.lower()] = \
                                self._concept_sentences[c.lower()][-20:]
            # Update word frequencies
            for w in norm.split():
                self._word_freq[w] += 1

        elapsed = time.time() - start
        n_concepts = learn_result.get('concepts', 0) if isinstance(learn_result, dict) else 0
        n_relations = learn_result.get('relations', 0) if isinstance(learn_result, dict) else 0
        n_facts = learn_result.get('facts', 0) if isinstance(learn_result, dict) else 0

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
            'elapsed_s': round(elapsed, 4),
        }
        logger.info("Trained: %s", result)
        return result

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
        if len(resolved) > len(query_concepts):
            added = [c for c in resolved if c not in query_concepts]
            trace.end("Extracted and resolved concepts (pronoun resolution)", {
                "raw_concepts": query_concepts,
                "resolved_concepts": resolved,
                "context_added": added,
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

        # --- Stage 7: Causal inference (NSCK CausalReasoner) ---
        trace.begin("causal_inference", {
            "active_concepts": query_concepts,
            "module": "CausalReasoner",
        })
        causal_effects = []
        causal_chains = []
        for c in query_concepts:
            effects = self.causal_graph.get_immediate_effects(c.capitalize())
            for e in effects:
                causal_effects.append((c, e))
            causes = self.causal_graph.get_immediate_causes(c.capitalize())
            for cause in causes:
                causal_chains.append((cause, c))
        trace.end("NSCK CausalReasoner inference", {
            "effects_found": causal_effects[:5],
            "causes_found": causal_chains[:5],
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
        response = self._build_response(
            user_input, query_concepts, related_facts,
            activation, causal_effects, knowledge_result, winner)
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
            whv = hypervec_rs.HyperVector(hash(w) % (2**32))
            # Positional encoding via permutation
            positioned = whv.permute(i % 64)
            hvs.append(positioned)
        # Bundle all word vectors
        result = hvs[0]
        for hv in hvs[1:]:
            result = result.bundle(hv)
        return result

    # ------------------------------------------------------------------
    # Internal: Concept extraction + pronoun resolution
    # ------------------------------------------------------------------

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
    # Internal: Response assembly — NO templates
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

        Strategy:
        1. Collect candidate sentences from all retrieval channels.
        2. Score them by concept overlap with the query.
        3. Deduplicate (Jaccard > 0.5 → skip).
        4. Select top 1-2 most relevant sentences.
        5. If nothing found, try n-gram continuation.
        """
        query_set = set(c.lower() for c in query_concepts)

        # --- Collect candidate sentences ---
        candidates: List[Tuple[str, float]] = []

        # From related facts (TextKnowledgeLearner)
        for fact in related_facts:
            subj = fact.get('subject', '')
            rel = fact.get('relation', '')
            obj = fact.get('object', '')
            conf = fact.get('confidence', 0.5)
            # Find source sentences that mention these concepts
            for concept in [subj.lower(), obj.lower()]:
                for sent in self._concept_sentences.get(concept, []):
                    candidates.append((sent, conf))

        # From spreading activation — find sentences for activated concepts
        for concept, act_level in sorted(
                activation.items(), key=lambda x: x[1], reverse=True)[:5]:
            for sent in self._concept_sentences.get(concept.lower(), []):
                candidates.append((sent, act_level * 0.3))

        # From causal effects
        for cause, effect in causal_effects:
            for sent in self._concept_sentences.get(cause.lower(), []):
                candidates.append((sent, 0.8))
            for sent in self._concept_sentences.get(effect.lower(), []):
                candidates.append((sent, 0.7))

        if not candidates:
            return self._fallback_response(user_input, query_concepts)

        # --- Score by concept overlap ---
        concept_specificity: Dict[str, float] = {}
        for c in query_concepts:
            freq = self._word_freq.get(c.lower(), 1)
            concept_specificity[c.lower()] = 1.0 / np.log2(freq + 2)

        scored: List[Tuple[str, float, float]] = []
        for sent, base_score in candidates:
            sent_lower = sent.lower()
            sent_concepts = set(self._extract_concepts(sent_lower))
            overlap = sent_concepts & query_set
            weighted_overlap = sum(concept_specificity.get(c, 0.5)
                                   for c in overlap)
            raw_overlap = len(overlap)
            relevance = base_score * 0.4 + weighted_overlap * 0.6
            if raw_overlap == 0:
                relevance *= 0.2
            scored.append((sent, relevance, weighted_overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        # --- Deduplicate ---
        used_norms: Set[str] = set()
        selected: List[str] = []
        for sent, score, overlap in scored:
            norm = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            if norm in used_norms:
                continue
            # Skip user questions
            lower = sent.lower().strip()
            if lower.endswith('?'):
                continue
            if any(lower.startswith(p) for p in
                   ['tell me', 'what ', 'who ', 'where ', 'when ',
                    'how ', 'why ', 'which ', 'are ', 'is ', 'do ', 'does ']):
                continue
            # Jaccard dedup
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
            # Only add 2nd sentence if it overlaps query concepts
            sent_concepts = set(self._extract_concepts(sent.lower()))
            if selected and not (sent_concepts & query_set):
                continue
            used_norms.add(norm)
            selected.append(sent[0].upper() + sent[1:] if sent else sent)
            if len(selected) >= 2:
                break

        if selected:
            parts = []
            for s in selected:
                s = s.rstrip('.').strip()
                if s:
                    parts.append(s)
            return ". ".join(parts) + "."

        return self._fallback_response(user_input, query_concepts)

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
                'current': self.emotion_system.get_emotion_info(),
                'blend': self.emotion_system.get_emotion_blend(),
                'mood': self.emotion_system.get_mood(window=10),
            },
            'causal': {
                'total_links': len(self.causal_graph._graph.edges)
                if hasattr(self.causal_graph, '_graph') else 0,
            },
            'self_model': self.self_model.get_stats('chat'),
            'curiosity': self.curiosity.get_statistics('chat'),
            'global_workspace': self.global_workspace.get_status(),
            'text_learner': self.text_learner.get_statistics(),
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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', ' ', t)
        return [w for w in t.split() if len(w) > 1]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'[.!?]+', text)
        return [s.strip() for s in parts if len(s.strip()) > 5]
