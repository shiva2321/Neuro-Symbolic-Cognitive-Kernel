"""
NSCK Knowledge Integration Module
==================================
Unified pipeline that connects perception, memory, reasoning, learning,
and contextual understanding into a coherent cognitive loop.

This module orchestrates:
1. Input processing (multimodal → HV)
2. Memory recall (episodic + semantic)
3. Contextual disambiguation
4. Reasoning (causal + rule-based)
5. Learning from experience
6. Knowledge update and self-correction
7. Response generation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import numpy as np

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
except ImportError:
    from python.core.vsa.hypervec_py import HyperVector as _HV

    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()

from python.core.memory.semantic_memory import SemanticMemory
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.reasoning.context_engine import ContextEngine, ContextFrame, DisambiguatedMeaning

try:
    from python.core.multimodal.multimodal_processor import (
        MultimodalProcessor,
        MultimodalInput,
        ProcessedInput,
    )
except ImportError:
    # Multimodal module is optional (may be in archive/)
    MultimodalProcessor = None  # type: ignore[misc,assignment]
    MultimodalInput = None  # type: ignore[misc,assignment]
    ProcessedInput = None  # type: ignore[misc,assignment]

from python.core.reasoning.causal_reasoning import CausalGraph, CausalDiscovery, CausalReasoner


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CognitiveResponse:
    """Full response from the knowledge integration pipeline."""
    answer: str
    confidence: float
    reasoning_trace: List[str]
    recalled_episodes: int
    disambiguations: List[DisambiguatedMeaning]
    learned_facts: List[str]
    emotional_context: str = "neutral"


@dataclass
class KnowledgeEntry:
    """A piece of knowledge stored in the system."""
    concept: str
    relation: str       # "is_a", "has_property", "causes", "context_means", ...
    target: str
    confidence: float
    source: str          # "bootstrap", "learned", "corrected"
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Knowledge Integration Engine
# ---------------------------------------------------------------------------

class KnowledgeIntegration:
    """
    Central cognitive integration pipeline.

    Connects multimodal perception → memory → reasoning → learning
    into a single coherent loop that:
    - Understands multi-modal inputs
    - Remembers and recalls past experiences
    - Reasons about causes and consequences
    - Learns and self-corrects over time
    - Generalises knowledge to new situations
    - Disambiguates context-dependent meanings
    """

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        context_engine: Optional[ContextEngine] = None,
        multimodal_processor: Optional[MultimodalProcessor] = None,
    ):
        # Core modules — create defaults if not provided
        self.semantic = semantic_memory or SemanticMemory()
        self.episodic = episodic_memory or EpisodicMemory()
        self.context = context_engine or ContextEngine(self.semantic)
        self.multimodal = multimodal_processor or MultimodalProcessor(
            context_engine=self.context,
            semantic_memory=self.semantic,
        )

        # Causal reasoning
        self.causal_discovery = CausalDiscovery()
        self.causal_graph = CausalGraph()

        # Knowledge base (simple triple store)
        self.knowledge: List[KnowledgeEntry] = []
        self._concept_index: Dict[str, List[int]] = defaultdict(list)

        # Learning statistics
        self.stats = {
            "queries_processed": 0,
            "facts_learned": 0,
            "corrections_made": 0,
            "episodes_recorded": 0,
        }

        # Bootstrap with basic knowledge
        self._bootstrap_knowledge()

    def reset(self):
        """Clear all knowledge and re-initialize the entire cognitive pipeline."""
        self.knowledge = []
        self._concept_index = defaultdict(list)
        self.stats = {
            "queries_processed": 0,
            "facts_learned": 0,
            "corrections_made": 0,
            "episodes_recorded": 0,
        }
        # Reset sub-modules
        self.semantic.reset()
        self.episodic.reset()
        self.context.reset()
        self.causal_discovery.reset()
        self.causal_graph.reset()
        
        # Re-bootstrap
        self._bootstrap_knowledge()
        print("[COGNITIVE] Core integration pipeline reset.")

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_knowledge(self):
        """Seed the system with foundational common-sense knowledge."""
        bootstrap_facts = [
            ("red", "is_a", "color"),
            ("rose", "is_a", "flower"),
            ("flower", "is_a", "plant"),
            ("danger", "is_a", "concept"),
            ("beauty", "is_a", "concept"),
            ("traffic_light", "has_property", "red"),
            ("traffic_light", "has_property", "signal"),
            ("rose", "has_property", "red"),
            ("rose", "has_property", "beauty"),
            ("fire", "has_property", "red"),
            ("fire", "causes", "danger"),
            ("red_traffic_light", "causes", "stop"),
            ("red_rose", "causes", "admiration"),
            ("learning", "causes", "knowledge"),
            ("knowledge", "causes", "understanding"),
        ]

        for concept, relation, target in bootstrap_facts:
            self._add_knowledge(concept, relation, target, 0.9, "bootstrap")
            # Also add to semantic memory graph
            if concept not in self.semantic.concept_graph:
                self.semantic.add_concept(concept, {"type": "entity"})
            if target not in self.semantic.concept_graph:
                self.semantic.add_concept(target, {"type": "entity"})
            self.semantic.add_relation(concept, relation, target)

    def _add_knowledge(
        self,
        concept: str,
        relation: str,
        target: str,
        confidence: float,
        source: str,
    ):
        """Add a knowledge entry to the store."""
        entry = KnowledgeEntry(
            concept=concept,
            relation=relation,
            target=target,
            confidence=confidence,
            source=source,
            timestamp=time.time(),
        )
        idx = len(self.knowledge)
        self.knowledge.append(entry)
        self._concept_index[concept].append(idx)
        self._concept_index[target].append(idx)

    # ------------------------------------------------------------------
    # Main cognitive loop
    # ------------------------------------------------------------------

    def process_input(
        self,
        inp: MultimodalInput,
        task_tag: str = "general",
        emotional_state: str = "neutral",
    ) -> CognitiveResponse:
        """
        Full cognitive processing pipeline for any input.

        Steps:
        1. Multimodal encoding → HV + extracted concepts
        2. Memory recall (episodic + semantic)
        3. Context disambiguation
        4. Causal reasoning
        5. Response generation
        6. Episodic recording (learning)
        """
        self.stats["queries_processed"] += 1
        trace: List[str] = []
        learned: List[str] = []

        # --- 1. Encode input ---
        processed = self.multimodal.process(inp)
        trace.append(
            f"Encoded {len(processed.modality_results)} modalities, "
            f"{len(processed.extracted_concepts)} concepts extracted."
        )

        # --- 2. Recall relevant memories ---
        similar_episodes = self.episodic.recall_similar(
            processed.fused_hv, task_tag, k=5
        )
        trace.append(f"Recalled {len(similar_episodes)} similar episodes.")

        # Semantic lookup
        semantic_matches = self.semantic.query(processed.fused_hv, k=5)
        if semantic_matches:
            trace.append(
                f"Semantic matches: {', '.join(m[0] for m in semantic_matches[:3])}"
            )

        # --- 3. Context disambiguation ---
        context_frame = ContextFrame(
            domain=self._infer_domain(processed),
            active_concepts=processed.extracted_concepts,
            environment_cues=processed.context_cues,
            emotional_state=emotional_state,
            timestamp=time.time(),
        )

        disambiguations: List[DisambiguatedMeaning] = []
        for concept in processed.extracted_concepts:
            dis = self.context.disambiguate(concept, context_frame)
            if dis.confidence > 0.4:
                disambiguations.append(dis)
                trace.append(
                    f"Disambiguated '{concept}' → '{dis.meaning}' "
                    f"(domain={dis.context_domain}, conf={dis.confidence:.2f})"
                )

        # --- 4. Causal reasoning ---
        causal_inferences = []
        for concept in processed.extracted_concepts[:5]:
            chains = self.causal_graph.forward_chain(concept, max_depth=2)
            for chain in chains:
                causal_inferences.append(chain.describe())
        if causal_inferences:
            trace.append(f"Causal inferences: {'; '.join(causal_inferences[:3])}")

        # --- 5. Generate response ---
        answer = self._generate_response(
            processed, similar_episodes, semantic_matches,
            disambiguations, causal_inferences, trace
        )

        # --- 6. Record episode (learn from this interaction) ---
        episode = LiveEpisode(
            timestamp=time.time(),
            task_tag=task_tag,
            situation_hv=processed.fused_hv,
            state=processed.context_cues,
            action="process_input",
            outcome="response_generated",
            reward=0.0,  # Will be updated via feedback
            emotion=emotional_state,
        )
        self.episodic.record(episode)
        self.stats["episodes_recorded"] += 1

        # Learn new associations from input
        new_facts = self._extract_learnable_facts(processed, disambiguations)
        for fact in new_facts:
            self._add_knowledge(
                fact[0], fact[1], fact[2], 0.6, "learned"
            )
            learned.append(f"{fact[0]} {fact[1]} {fact[2]}")
            self.stats["facts_learned"] += 1

        return CognitiveResponse(
            answer=answer,
            confidence=processed.confidence,
            reasoning_trace=trace,
            recalled_episodes=len(similar_episodes),
            disambiguations=disambiguations,
            learned_facts=learned,
            emotional_context=emotional_state,
        )

    # ------------------------------------------------------------------
    # Knowledge queries
    # ------------------------------------------------------------------

    def query_knowledge(self, concept: str) -> List[KnowledgeEntry]:
        """Retrieve all knowledge about a concept."""
        indices = self._concept_index.get(concept, [])
        return [self.knowledge[i] for i in indices if i < len(self.knowledge)]

    def query_relation(
        self, concept: str, relation: str
    ) -> List[KnowledgeEntry]:
        """Retrieve knowledge about a specific relation."""
        return [
            k for k in self.query_knowledge(concept)
            if k.relation == relation and k.concept == concept
        ]

    # ------------------------------------------------------------------
    # Learning & self-correction
    # ------------------------------------------------------------------

    def correct_knowledge(
        self,
        concept: str,
        wrong_relation: str,
        wrong_target: str,
        correct_target: str,
    ):
        """
        Self-correct a piece of knowledge.
        Lowers confidence on wrong fact, adds corrected version.
        """
        # Weaken wrong knowledge
        for idx in self._concept_index.get(concept, []):
            k = self.knowledge[idx]
            if k.relation == wrong_relation and k.target == wrong_target:
                k.confidence = max(0.0, k.confidence - 0.3)

        # Add corrected fact
        self._add_knowledge(concept, wrong_relation, correct_target, 0.8, "corrected")
        self.stats["corrections_made"] += 1

        # Update context engine if applicable
        if wrong_relation == "context_means":
            # The concept-meaning was wrong in some context — re-learn
            pass  # context_engine.learn_from_feedback handles this externally

    def learn_from_feedback(
        self,
        task_tag: str,
        reward: float,
        concepts: List[str],
    ):
        """
        Adjust knowledge confidence based on outcome feedback.
        Positive reward → strengthen recent associations.
        Negative reward → weaken them.
        """
        recent = self.episodic.recall_recent(task_tag, n=1)
        if not recent:
            return

        # Strengthen or weaken causal links from concepts used
        for concept in concepts:
            for idx in self._concept_index.get(concept, []):
                k = self.knowledge[idx]
                if k.source in ("learned", "corrected"):
                    adjustment = 0.05 * reward
                    k.confidence = max(0.0, min(1.0, k.confidence + adjustment))

        # Record reward for causal discovery
        causes = [c for c in concepts if c]
        effects = ["positive_outcome"] if reward > 0 else ["negative_outcome"]
        self.causal_discovery.observe(task_tag, causes, effects)

    # ------------------------------------------------------------------
    # Transfer & generalisation
    # ------------------------------------------------------------------

    def apply_knowledge_to_new_domain(
        self,
        source_domain: str,
        target_domain: str,
    ) -> List[str]:
        """
        Attempt to transfer knowledge from source_domain to target_domain.
        Returns list of transferred facts.
        """
        transferred: List[str] = []

        for k in self.knowledge:
            if k.confidence < 0.5:
                continue
            # Check if fact is domain-specific
            if source_domain.lower() in k.concept.lower():
                # Create analogous fact for target domain
                new_concept = k.concept.replace(
                    source_domain.lower(), target_domain.lower()
                )
                # Only transfer if it's a novel fact
                existing = self.query_relation(new_concept, k.relation)
                if not any(e.target == k.target for e in existing):
                    self._add_knowledge(
                        new_concept, k.relation, k.target,
                        k.confidence * 0.7,  # lower confidence for transferred
                        "transferred",
                    )
                    transferred.append(
                        f"{new_concept} {k.relation} {k.target} "
                        f"(from {source_domain})"
                    )

        return transferred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_domain(self, processed: ProcessedInput) -> str:
        """Infer the domain from processed input."""
        if self.context:
            return self.context.infer_context_domain(
                processed.extracted_concepts,
                processed.context_cues,
            )
        return "general"

    def _generate_response(
        self,
        processed: ProcessedInput,
        episodes: List[LiveEpisode],
        semantic_matches: List[Tuple[str, float]],
        disambiguations: List[DisambiguatedMeaning],
        causal_inferences: List[str],
        trace: List[str],
    ) -> str:
        """Generate a human-readable response from all gathered information."""
        parts = []

        # Summarise understanding
        if processed.extracted_concepts:
            parts.append(
                f"I understood the following concepts: "
                f"{', '.join(processed.extracted_concepts[:10])}."
            )

        # Report disambiguations
        for d in disambiguations:
            if d.confidence > 0.5:
                parts.append(
                    f"In this context, '{d.concept}' means '{d.meaning}' "
                    f"({d.explanation})"
                )

        # Report recalled experience
        if episodes:
            parts.append(
                f"I recall {len(episodes)} similar past experience(s)."
            )

        # Report causal understanding
        if causal_inferences:
            parts.append(
                f"Based on my understanding: {'; '.join(causal_inferences[:3])}."
            )

        # Report semantic connections
        if semantic_matches:
            top = semantic_matches[0]
            parts.append(f"Closest known concept: '{top[0]}' (similarity={top[1]:.2f}).")

        if not parts:
            parts.append("I received and processed the input, but need more context to provide a detailed response.")

        return " ".join(parts)

    def _extract_learnable_facts(
        self,
        processed: ProcessedInput,
        disambiguations: List[DisambiguatedMeaning],
    ) -> List[Tuple[str, str, str]]:
        """Extract new facts that can be learned from this interaction."""
        facts: List[Tuple[str, str, str]] = []

        # Learn context-meaning associations from high-confidence disambiguations
        for d in disambiguations:
            if d.confidence > 0.7:
                facts.append((d.concept, "context_means", d.meaning))

        # Learn co-occurrence: concepts that appear together
        concepts = processed.extracted_concepts
        if len(concepts) >= 2:
            for i in range(min(len(concepts) - 1, 3)):
                facts.append(
                    (concepts[i], "co_occurs_with", concepts[i + 1])
                )

        return facts

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return system-wide statistics."""
        return {
            **self.stats,
            "knowledge_entries": len(self.knowledge),
            "semantic_concepts": len(self.semantic.concept_hvs),
            "context_stats": self.context.get_statistics(),
        }
