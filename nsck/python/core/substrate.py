"""
NSCKSubstrate — The public substrate API for NSCK (V11).
========================================================
Stable interface for third-party developers to build on top of NSCK.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from python.core.integration.config import NSCKConfig


def _stable_seed(obj: object) -> int:
    """Convert any object to a stable 32-bit seed for HyperVector construction."""
    return hash(str(obj)) % (2 ** 32)


@dataclass
class SubstrateResult:
    """Result of processing an input through the NSCK substrate."""
    chosen_action: str
    confidence: float
    explanation: str
    predicates: Set[str]
    trace: Dict[str, Any]
    modalities_processed: List[str]
    generalization_triggered: bool


class NSCKSubstrate:
    """
    The public substrate API for NSCK.

    Wraps CognitiveEngine with:
    1. A clean, stable public API
    2. A plugin/encoder registration system
    3. The PerceptPacket protocol for any input type
    4. Built-in cross-modal learning
    5. Explicit support for all input types

    Example usage::

        substrate = NSCKSubstrate()
        substrate.register_task("my_task")

        # Text input
        result = substrate.process("The sky is blue", "my_task")

        # Numeric sequence
        result = substrate.process([1.2, 1.4, 1.7, 2.1, 2.8], "my_task")

        # Multiple modalities simultaneously
        result = substrate.process_multimodal({
            "text": "fire detected",
            "sensor": [0.9, 1.1, 0.95, 1.3],
        }, "my_task")

        # Learn from outcome
        substrate.learn(state, action, reward=1.0, task_tag="my_task")

        # Register custom encoder
        substrate.register_encoder("thermal", my_encoder)
    """

    def __init__(self, config: Optional[NSCKConfig] = None) -> None:
        """Initialize with optional NSCKConfig."""
        self.config = config or NSCKConfig()
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        self._engine = CognitiveEngine(config=self.config, persistence_path=None)
        # Custom encoder registry: modality_name → encoder_fn(data, task_tag) → PerceptPacket
        self._custom_encoders: Dict[str, Callable] = {}
        self._registered_tasks: List[str] = []

    def register_task(self, task_tag: str) -> None:
        """Register a new task/domain."""
        if task_tag not in self._registered_tasks:
            self._engine.register_task(task_tag)
            self._registered_tasks.append(task_tag)

    def process(
        self,
        input_data: Any,
        task_tag: str,
        available_actions: Optional[List[str]] = None,
    ) -> SubstrateResult:
        """
        Process ANY input type and return a reasoning result.

        Input can be:
        - str: text
        - list/np.ndarray of numbers: numeric sequence
        - np.ndarray (2D/3D): image
        - dict: structured state
        - PerceptPacket: pre-encoded perception
        """
        if task_tag not in self._registered_tasks:
            self.register_task(task_tag)

        from python.core.types.percept_packet import PerceptPacket

        modalities = []
        state = input_data

        # Route based on input type
        if isinstance(input_data, str):
            state = {"text": input_data}
            modalities = ["text"]
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim >= 2:
                # Image input — use ImageAdapter for FPE-based similarity-preserving encoding
                from python.core.adapters.image_adapter import ImageAdapter
                pkt = ImageAdapter().encode(input_data, task_tag)
                state = pkt
                modalities = ["image"]
            else:
                # 1D array — numeric sequence
                state = input_data
                modalities = ["numeric_sequence"]
        elif isinstance(input_data, (list, tuple)):
            try:
                if len(input_data) > 0 and isinstance(input_data[0], (int, float)):
                    state = input_data
                    modalities = ["numeric_sequence"]
                else:
                    state = {"values": list(input_data)}
                    modalities = ["list"]
            except Exception:
                state = {"values": str(input_data)}
                modalities = ["unknown"]
        elif isinstance(input_data, dict):
            state = input_data
            modalities = ["dict"]
        elif isinstance(input_data, PerceptPacket):
            state = input_data
            modalities = [input_data.modality]
        else:
            state = {"value": str(input_data)}
            modalities = ["unknown"]

        prev_decision_count = self._engine._decision_counter
        cog_state = self._engine.decide(state, task_tag)
        generalization_triggered = (
            self._engine._decision_counter % self.config.generalization_interval == 0
            and self._engine._decision_counter > prev_decision_count
        ) if self.config.enable_continuous_generalization else False

        explanation_text = ""
        if cog_state.explanation:
            try:
                explanation_text = str(cog_state.explanation.text)
            except Exception:
                explanation_text = str(cog_state.explanation)

        return SubstrateResult(
            chosen_action=cog_state.chosen_action,
            confidence=cog_state.confidence,
            explanation=explanation_text,
            predicates=set(cog_state.active_predicates),
            trace=cog_state.trace or {},
            modalities_processed=modalities,
            generalization_triggered=generalization_triggered,
        )

    def process_multimodal(
        self,
        inputs: Dict[str, Any],
        task_tag: str,
        available_actions: Optional[List[str]] = None,
    ) -> SubstrateResult:
        """Process multiple input types simultaneously."""
        if task_tag not in self._registered_tasks:
            self.register_task(task_tag)

        from python.core.types.percept_packet import PerceptPacket
        from python.core.adapters.multimodal_fuser import MultimodalFuser

        packets = []
        modalities_processed = []

        for modality, data in inputs.items():
            # Check for custom encoder
            if modality in self._custom_encoders:
                try:
                    pkt = self._custom_encoders[modality](data, task_tag)
                    packets.append(pkt)
                    modalities_processed.append(modality)
                    continue
                except Exception:
                    pass

            # Built-in routing
            pkt = self._encode_single(data, modality, task_tag)
            packets.append(pkt)
            modalities_processed.append(modality)

        # Cross-modal learning: observe modality HVs together
        if self._engine.cross_modal is not None and len(packets) >= 2:
            try:
                modality_hvs = {
                    mod: pkt.situation_hv
                    for mod, pkt in zip(modalities_processed, packets)
                }
                self._engine.cross_modal.observe(modality_hvs)
            except Exception:
                pass

        # Fuse all packets
        if len(packets) == 1:
            fused_packet = packets[0]
        elif packets:
            fuser = MultimodalFuser()
            fused_packet = fuser.fuse(packets)
        else:
            # Empty input: create minimal packet
            import python.core.vsa.hypervec_shim as hv_mod
            fused_packet = PerceptPacket.make(
                modality="empty",
                situation_hv=hv_mod.HyperVector(0),
                active_predicates=frozenset(),
            )

        cog_state = self._engine.decide(fused_packet, task_tag)
        explanation_text = ""
        if cog_state.explanation:
            try:
                explanation_text = str(cog_state.explanation.text)
            except Exception:
                explanation_text = str(cog_state.explanation)

        return SubstrateResult(
            chosen_action=cog_state.chosen_action,
            confidence=cog_state.confidence,
            explanation=explanation_text,
            predicates=set(cog_state.active_predicates),
            trace=cog_state.trace or {},
            modalities_processed=modalities_processed,
            generalization_triggered=False,
        )

    def _encode_single(self, data: Any, modality: str, task_tag: str):
        """Encode a single modality input into a PerceptPacket."""
        from python.core.types.percept_packet import PerceptPacket
        import python.core.vsa.hypervec_shim as hv_mod

        try:
            if isinstance(data, str):
                state = {"text": data}
                adapter = self._engine.adapters.get(task_tag)
                if adapter:
                    return adapter.encode(state, task_tag)
                from python.core.perception.grounding_verifier import GroundingVerifier
                ver = GroundingVerifier()
                preds = ver.get_active_predicates(state, context=task_tag)
                shv = self._engine.episodic_memory.create_situation_hv(state, task_tag, preds)
                return PerceptPacket.make(
                    modality="text",
                    situation_hv=shv,
                    active_predicates=frozenset(preds),
                    raw_state=state,
                )
            elif modality == "image" or (
                isinstance(data, np.ndarray) and data.ndim >= 2
            ):
                from python.core.adapters.image_adapter import ImageAdapter
                return ImageAdapter().encode(data, task_tag)
            elif modality == "audio":
                from python.core.adapters.audio_adapter import AudioAdapter
                return AudioAdapter().encode(data, task_tag)
            elif isinstance(data, (list, np.ndarray)):
                try:
                    vals = [float(v) for v in data]
                    from python.core.adapters.numeric_sequence_adapter import NumericSequenceAdapter
                    return NumericSequenceAdapter(channel_name=modality).encode(vals, task_tag)
                except Exception:
                    state = {"values": list(data)}
                    shv = hv_mod.HyperVector(_stable_seed(state))
                    return PerceptPacket.make(
                        modality="numeric",
                        situation_hv=shv,
                        active_predicates=frozenset(),
                        raw_state=state,
                    )
            elif isinstance(data, dict):
                adapter = self._engine.adapters.get(task_tag)
                if adapter:
                    return adapter.encode(data, task_tag)
                shv = hv_mod.HyperVector(_stable_seed(sorted(data.items())))
                return PerceptPacket.make(
                    modality="dict",
                    situation_hv=shv,
                    active_predicates=frozenset(),
                    raw_state=data,
                )
            else:
                shv = hv_mod.HyperVector(_stable_seed(data))
                return PerceptPacket.make(
                    modality=modality,
                    situation_hv=shv,
                    active_predicates=frozenset(),
                )
        except Exception:
            shv = hv_mod.HyperVector(0)
            return PerceptPacket.make(
                modality=modality,
                situation_hv=shv,
                active_predicates=frozenset(),
            )

    def learn(
        self,
        state: Any,
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "neutral",
    ) -> None:
        """Learn from a (state, action, reward) experience."""
        if task_tag not in self._registered_tasks:
            self.register_task(task_tag)
        # Ensure state is a dict for the engine's learn method
        if not isinstance(state, dict):
            state_dict = {"value": str(state)}
        else:
            state_dict = state
        self._engine.learn(state_dict, action, reward, task_tag, outcome)

    def sleep(self, task_tag: Optional[str] = None) -> Dict[str, Any]:
        """Trigger offline consolidation and generalization."""
        self._engine.sleep(task_tag)
        return {"sleep_cycles": self._engine.stats.get("sleep_cycles", 0)}

    def remember(self, query: Any, task_tag: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """Recall similar past experiences."""
        import python.core.vsa.hypervec_shim as hv_mod
        if isinstance(query, dict):
            query_hv = self._engine.episodic_memory.create_situation_hv(
                query, task_tag or "unknown", []
            )
        elif isinstance(query, str):
            query_state = {"text": query}
            query_hv = self._engine.episodic_memory.create_situation_hv(
                query_state, task_tag or "unknown", []
            )
        else:
            query_hv = hv_mod.HyperVector(_stable_seed(query))

        tasks = [task_tag] if task_tag else self._registered_tasks
        results = []
        for t in tasks:
            try:
                episodes = self._engine.episodic_memory.retrieve(query_hv, t, top_k=top_k)
                for ep in episodes:
                    results.append({
                        "task": t,
                        "action": ep.action,
                        "outcome": ep.outcome,
                        "reward": ep.reward,
                        "timestamp": ep.timestamp,
                    })
            except Exception:
                pass
        return results[:top_k]

    def register_encoder(self, modality_name: str, encoder_fn: Callable) -> None:
        """Register a custom encoder for a new input modality.

        The encoder_fn should accept (data, task_tag) and return a PerceptPacket.
        """
        self._custom_encoders[modality_name] = encoder_fn

    def get_knowledge(self, concept: str) -> Dict[str, Any]:
        """Query the semantic knowledge graph for a concept."""
        sm = self._engine.semantic_memory
        if concept in sm.concept_hvs:
            neighbors = sm.get_similar_concepts(concept, top_k=5)
            return {
                "concept": concept,
                "known": True,
                "similar": [n for n, _ in neighbors] if neighbors else [],
            }
        return {"concept": concept, "known": False}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = dict(self._engine.stats)
        stats["registered_tasks"] = list(self._registered_tasks)
        stats["decision_counter"] = self._engine._decision_counter
        if self._engine.cross_modal is not None:
            stats["cross_modal"] = self._engine.cross_modal.get_statistics()
        return stats
