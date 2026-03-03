"""
NSCKSubstrate — The public substrate API for NSCK (V13).
========================================================
Stable interface for third-party developers to build on top of NSCK.

V13 additions:
- SignalIngestor + UniversalHVEncoder wired into ingest pipeline
- CrossModalAssociativeMemory for modality-agnostic binding
- ProceduralMemory for skill caching and fast-path decisions
- ConceptDriftDetector for semantic memory monitoring
- ConformalWrapper for calibrated uncertainty bounds
- KLE uncertainty exposed in SubstrateResult
- ingest() / feedback() clean API
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from python.core.integration.config import NSCKConfig

logger = logging.getLogger(__name__)


def _stable_seed(obj: object) -> int:
    """Convert any object to a stable 32-bit seed for HyperVector construction."""
    return hash(str(obj)) % (2 ** 32)


@dataclass
class SubstrateResult:
    """Result of processing an input through the NSCK substrate (V13)."""
    chosen_action: str
    confidence: float
    explanation: str
    predicates: Set[str]
    trace: Dict[str, Any]
    modalities_processed: List[str]
    generalization_triggered: bool
    # V13 fields
    kle_uncertainty: Optional[float] = None
    uncertainty_bounds: Optional[tuple] = None
    encoding_stats: Optional[Dict[str, Any]] = None
    procedural_hit: bool = False
    # V26 societal context
    societal_context: Optional[Dict[str, Any]] = None


class NSCKSubstrate:
    """
    The public substrate API for NSCK (V13).

    Wraps CognitiveEngine with:
    1. A clean, stable public API (ingest/feedback)
    2. A plugin/encoder registration system
    3. The PerceptPacket protocol for any input type
    4. Built-in cross-modal learning via CrossModalAssociativeMemory
    5. Skill caching via ProceduralMemory
    6. Semantic drift monitoring via ConceptDriftDetector
    7. Calibrated uncertainty via ConformalWrapper
    8. UniversalHVEncoder for signal-agnostic encoding
    9. KLE uncertainty in every result

    Example usage::

        substrate = NSCKSubstrate()
        substrate.register_task("my_task")

        # V13 clean ingest API
        result = substrate.ingest("The sky is blue", "my_task")
        substrate.feedback(result.chosen_action, reward=1.0, task_tag="my_task")

        # Legacy API still works
        result = substrate.process("The sky is blue", "my_task")
    """

    def __init__(self, config: Optional[NSCKConfig] = None) -> None:
        """Initialize with optional NSCKConfig."""
        self.config = config or NSCKConfig()
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        self._engine = CognitiveEngine(config=self.config, persistence_path=None)
        # Custom encoder registry: modality_name → encoder_fn(data, task_tag) → PerceptPacket
        self._custom_encoders: Dict[str, Callable] = {}
        self._registered_tasks: List[str] = []

        # V13 modules
        from python.core.perception.signal_ingestor import SignalIngestor
        from python.core.vsa.universal_hv_encoder import UniversalHVEncoder
        from python.core.memory.cross_modal_associative_memory import CrossModalAssociativeMemory
        from python.core.memory.procedural_memory import ProceduralMemory
        from python.core.memory.concept_drift_detector import ConceptDriftDetector
        from python.core.learning.conformal_wrapper import ConformalWrapper
        self.signal_ingestor = SignalIngestor()
        self.universal_encoder = UniversalHVEncoder()
        self.cross_modal_memory = CrossModalAssociativeMemory()
        self.procedural_memory = ProceduralMemory()
        self.drift_detector = ConceptDriftDetector()
        self.conformal = ConformalWrapper(alpha=0.1)
        self._last_ingest_hv = None  # For procedural memory lookup

        # V14: Rich perception adapters
        self._rich_adapters: Dict[str, Any] = {}
        self.distiller = None
        self._init_rich_adapters()

        # V14: Load knowledge packs
        packs = getattr(self.config, "knowledge_packs", [])
        if packs:
            self._load_knowledge_packs(packs)

        # V15: Transplant projector registry for live encoding
        self._transplant_projectors: Dict[str, Any] = {}
        # NSCK-UPMA Vision fields — lazily initialized on first call to
        # absorb_vision_model() or analyze_image() via _ensure_vision_components().
        self._vision_absorber = None
        self._absorption_memory = None
        self._domain_tagger = None

        # V16: Auto-seed if enabled
        if getattr(self.config, 'enable_seeding', False):
            self._auto_seed()

        # V26: Societal HyperVector router (lazy; created by init_societal_world)
        self._societal_manager = None
        self._societal_router = None

        # V4: Wire SNN perception to semantic memory for symbol grounding.
        # This populates the concept mapper from SemanticMemory HVs so that
        # spike patterns can resolve to named predicates (cleanup memory bridge).
        _eng_perception = getattr(self._engine, 'perception', None)
        if (_eng_perception is not None and
                hasattr(_eng_perception, 'register_concepts_from_memory')):
            try:
                _n = _eng_perception.register_concepts_from_memory(
                    self._engine.semantic_memory
                )
                logger.info("[SUBSTRATE] SNN grounded to %d semantic concepts", _n)
            except Exception as _exc:
                logger.debug("[SUBSTRATE] SNN grounding skipped: %s", _exc)

    def _init_rich_adapters(self) -> None:
        """Initialize rich perception adapters based on perception_mode."""
        mode = getattr(self.config, "perception_mode", "pure")
        if mode not in ("bridge", "hybrid"):
            return
        try:
            from python.core.adapters.rich_text_adapter import RichTextAdapter
            self._rich_adapters["text"] = RichTextAdapter(self.config)
        except Exception:
            pass
        try:
            from python.core.adapters.rich_image_adapter import RichImageAdapter
            self._rich_adapters["image"] = RichImageAdapter(self.config)
        except Exception:
            pass
        try:
            from python.core.adapters.rich_audio_adapter import RichAudioAdapter
            self._rich_adapters["audio"] = RichAudioAdapter(self.config)
        except Exception:
            pass
        if mode == "hybrid":
            try:
                from python.core.learning.perception_distiller import PerceptionDistiller
                self.distiller = PerceptionDistiller(
                    threshold=getattr(self.config, "distillation_threshold", 0.80)
                )
            except Exception:
                pass

    def _load_knowledge_packs(self, pack_paths) -> None:
        """Load and inject knowledge packs into semantic memory."""
        import logging
        _log = logging.getLogger("nsck.substrate")
        for path in pack_paths:
            try:
                from python.core.integration.knowledge_pack import KnowledgePack
                pack = KnowledgePack.load(path)
                counts = pack.inject_into(self._engine)
                _log.debug("Loaded knowledge pack '%s' from %s: %s", pack.name, path, counts)
            except FileNotFoundError:
                _log.warning("Knowledge pack not found: %s", path)
            except Exception as e:
                _log.warning("Failed to load knowledge pack '%s': %s (%s)", path, e, type(e).__name__)

    def register_task(self, task_tag: str) -> None:
        """Register a new task/domain."""
        if task_tag not in self._registered_tasks:
            self._engine.register_task(task_tag)
            self._registered_tasks.append(task_tag)

    @property
    def engine(self):
        """Expose the underlying CognitiveEngine."""
        return self._engine

    def _auto_seed(self) -> None:
        """Auto-seed substrate from configured knowledge packs and/or BERT."""
        import logging
        _log = logging.getLogger("nsck.substrate")
        try:
            from python.core.seeding.semantic_seeder import SemanticSeeder
            seeder = SemanticSeeder()
            pack_path = getattr(self.config, 'seed_conceptnet_pack', '')
            if pack_path:
                import os
                if os.path.exists(pack_path):
                    seeder.seed_from_conceptnet_pack(self, pack_path)
                else:
                    _log.warning("_auto_seed: conceptnet pack not found: %s", pack_path)
            if getattr(self.config, 'seed_bert_on_init', False):
                model_name = getattr(self.config, 'seed_bert_model_name', 'bert-base-uncased')
                try:
                    seeder.seed_from_bert(self, model_name=model_name)
                except RuntimeError as exc:
                    _log.warning("_auto_seed: BERT seeding skipped (transformers not installed): %s", exc)
            seeder.post_seed_enrich(self)
        except Exception as exc:
            _log.warning("_auto_seed failed: %s", exc)


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

        # V13: compute encoding stats + conformal uncertainty
        enc_stats = None
        kle = None
        ubounds = None
        try:
            ts = self.signal_ingestor.ingest(input_data)
            enc_result = self.universal_encoder.encode_with_stats(ts)
            enc_stats = {k: v for k, v in enc_result.items() if k != "hv"}
            self._last_ingest_hv = enc_result["hv"]
        except Exception:
            pass
        if self.conformal.is_calibrated():
            score = 1.0 - cog_state.confidence
            ubounds = self.conformal.uncertainty_bound(score)
        # KLE from global workspace
        try:
            kle = self._engine.global_workspace.get_kle_uncertainty()
        except Exception:
            pass

        # V26: societal context routing
        societal_ctx: Optional[Dict[str, Any]] = None
        try:
            router = self._ensure_societal_world() and self._societal_router
            if router is not None and self._last_ingest_hv is not None:
                societal_ctx = router.route(
                    self._last_ingest_hv, task_tag=task_tag
                )
        except Exception:
            pass

        return SubstrateResult(
            chosen_action=cog_state.chosen_action,
            confidence=cog_state.confidence,
            explanation=explanation_text,
            predicates=set(cog_state.active_predicates),
            trace=cog_state.trace or {},
            modalities_processed=modalities,
            generalization_triggered=generalization_triggered,
            kle_uncertainty=kle,
            uncertainty_bounds=ubounds,
            encoding_stats=enc_stats,
            societal_context=societal_ctx,
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

        # V14: Rich perception routing
        mode = getattr(self.config, "perception_mode", "pure")
        if mode in ("bridge", "hybrid") and self._rich_adapters:
            detected_modality = modality
            if isinstance(data, str):
                detected_modality = "text"
            elif isinstance(data, np.ndarray) and data.ndim >= 2:
                detected_modality = "image"
            elif modality == "audio":
                detected_modality = "audio"
            rich_adapter = self._rich_adapters.get(detected_modality)
            if rich_adapter is not None:
                try:
                    return rich_adapter.encode(data, task_tag)
                except Exception:
                    pass

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

    # ------------------------------------------------------------------
    # V13 clean ingest / feedback API
    # ------------------------------------------------------------------

    def ingest(
        self,
        input_data: Any,
        task_tag: str,
        available_actions: Optional[List[str]] = None,
    ) -> SubstrateResult:
        """
        V13 clean ingest API — alias for process() with procedural fast-path.

        Checks ProceduralMemory first; if a cached skill matches the encoded
        context, returns a fast result without full deliberation.
        """
        if task_tag not in self._registered_tasks:
            self.register_task(task_tag)

        # Try procedural fast-path
        try:
            ts = self.signal_ingestor.ingest(input_data)
            context_hv = self.universal_encoder.encode(ts)
            cached = self.procedural_memory.recall_action(context_hv)
            if cached is not None:
                action, sim, reward = cached
                return SubstrateResult(
                    chosen_action=action,
                    confidence=float(sim),
                    explanation=f"Procedural cache hit (similarity={sim:.3f})",
                    predicates=set(),
                    trace={"procedural_cache": True, "similarity": sim},
                    modalities_processed=[ts.source_type],
                    generalization_triggered=False,
                    kle_uncertainty=0.0,
                    procedural_hit=True,
                )
        except Exception:
            pass

        return self.process(input_data, task_tag, available_actions)

    def feedback(
        self,
        action: str,
        reward: float,
        task_tag: str,
        state: Optional[Any] = None,
        outcome: str = "neutral",
    ) -> None:
        """
        V13 feedback API — record an outcome and update procedural memory.

        Combines learn() with Hebbian weight update and skill caching.
        """
        if state is not None:
            self.learn(state, action, reward, task_tag, outcome)
        # Update Hebbian weights in universal encoder
        if state is not None:
            try:
                self.universal_encoder.hebbian_update(state, reward)
            except Exception:
                pass
        # Cache skill in procedural memory if reward is positive
        if self._last_ingest_hv is not None and reward > 0:
            try:
                self.procedural_memory.cache_skill(
                    self._last_ingest_hv, action, reward
                )
            except Exception:
                pass
        # Update conformal calibration
        try:
            self.conformal.calibrate([1.0 - reward], labels=[reward >= 0])
        except Exception:
            pass
        # V26: auto-register action concept in societal world
        try:
            router = self._societal_router
            if router is not None and self._last_ingest_hv is not None:
                router.register_action(
                    action=action,
                    hv=self._last_ingest_hv,
                    domain="actions",
                    reward=reward,
                )
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = dict(self._engine.stats)
        stats["registered_tasks"] = list(self._registered_tasks)
        stats["decision_counter"] = self._engine._decision_counter
        if self._engine.cross_modal is not None:
            stats["cross_modal"] = self._engine.cross_modal.get_statistics()
        # V13 stats
        stats["procedural_memory"] = self.procedural_memory.get_statistics()
        stats["cross_modal_memory"] = self.cross_modal_memory.get_statistics()
        stats["drift_detector"] = self.drift_detector.get_statistics()
        stats["conformal"] = self.conformal.get_statistics()
        # V15 stats
        stats["transplant_domains"] = list(self._transplant_projectors.keys())
        return stats

    # ------------------------------------------------------------------
    # V15 Model Transplantation API
    # ------------------------------------------------------------------

    def transplant(
        self,
        model: Any,
        domain_name: str = "default",
        strategy: Optional[str] = None,
        calibration_epochs: Optional[int] = None,
        save_pack: Optional[str] = None,
    ) -> Any:
        """Transplant knowledge from *model* into NSCK's HV space.

        Requires ``NSCKConfig.enable_transplant=True``.

        Parameters
        ----------
        model:
            Pretrained neural network (e.g. BERT, GPT, ViT).  Any PyTorch
            module or duck-typed object with ``named_parameters()``.
        domain_name:
            Label for the transplanted knowledge domain.
        strategy:
            Projection strategy: ``"random"``, ``"learned"``, or
            ``"svd_factored"`` (default from config).
        calibration_epochs:
            STDP calibration epochs.  ``None`` reads from config.  ``0``
            disables calibration.
        save_pack:
            Optional path to save a ``KnowledgePack`` file.

        Returns
        -------
        TransplantReport
            Quality metrics and pass/fail status.

        Raises
        ------
        RuntimeError
            If ``enable_transplant`` is False.
        """
        if not getattr(self.config, "enable_transplant", False):
            raise RuntimeError(
                "Transplant is disabled. Set NSCKConfig.enable_transplant=True."
            )

        from python.core.transplant.pipeline import TransplantPipeline  # noqa: PLC0415

        eff_strategy = strategy or getattr(
            self.config, "transplant_strategy", "svd_factored"
        )
        eff_epochs = calibration_epochs if calibration_epochs is not None else int(
            getattr(self.config, "transplant_calibration_epochs", 10)
        )

        pipeline = TransplantPipeline(config=self.config)
        report = pipeline.run(
            model=model,
            domain_name=domain_name,
            strategy=eff_strategy,
            calibration_epochs=eff_epochs,
            save_pack_path=save_pack,
            cognitive_engine=self._engine,
        )

        # Store projector for live encoding
        if domain_name in pipeline._projectors:
            self._transplant_projectors[domain_name] = pipeline._projectors[domain_name]

        return report

    # ── NSCK-UPMA Vision Absorption API ──────────────────────────────────

    def _ensure_vision_components(self) -> None:
        """Lazily initialize vision absorption components."""
        if self._vision_absorber is not None:
            return
        from python.core.vision.absorption_memory import AbsorptionMemory  # noqa: PLC0415
        from python.core.vision.domain_tagger import DomainTagger  # noqa: PLC0415
        from python.core.vision.feature_absorber import FeatureAbsorber  # noqa: PLC0415

        sem_mem = getattr(self._engine, "_semantic_memory", None)
        ep_mem = getattr(self._engine, "_episodic_memory", None)
        causal = getattr(self._engine, "_causal_graph", None)

        self._absorption_memory = AbsorptionMemory(semantic_memory=sem_mem)
        self._domain_tagger = DomainTagger()
        self._vision_absorber = FeatureAbsorber(
            absorption_memory=self._absorption_memory,
            domain_tagger=self._domain_tagger,
            semantic_memory=sem_mem,
            episodic_memory=ep_mem,
            causal_graph=causal,
        )

    def absorb_vision_model(
        self,
        model_or_name: Any,
        domain: str,
        dataset_iter=None,
        layer_names=None,
        model_id: Optional[str] = None,
        max_samples: int = 500,
        strategy: str = "svd_factored",
    ) -> Any:
        """Absorb a pretrained vision/text model into NSCK's HV space.

        Parameters
        ----------
        model_or_name:
            A PyTorch module, HuggingFace model name string, or any object
            supported by ``PretrainedModelAdapter``.
        domain:
            Semantic domain label (e.g. ``"medical"``, ``"satellite"``).
        dataset_iter:
            Iterable of ``(image_tensor, label)`` pairs for absorption.
            If ``None``, synthetic random samples are used.
        layer_names:
            Names of layers to extract features from.
        model_id:
            Unique identifier for this model.  Defaults to ``str(model_or_name)``.
        max_samples:
            Maximum number of samples to absorb.
        strategy:
            Projection strategy: ``"svd_factored"`` (default), ``"random"``.

        Returns
        -------
        AbsorptionReport
        """
        if not getattr(self.config, "enable_vision_absorption", False):
            raise RuntimeError(
                "Vision absorption is disabled. "
                "Set NSCKConfig.enable_vision_absorption=True or use NSCKConfig.vision()."
            )
        self._ensure_vision_components()
        eff_model_id = model_id or str(model_or_name)
        eff_max = max_samples or getattr(self.config, "vision_absorption_max_samples", 500)
        return self._vision_absorber.absorb(
            model=model_or_name,
            model_id=eff_model_id,
            domain=domain,
            dataset_iter=dataset_iter,
            layer_names=layer_names,
            max_samples=eff_max,
            strategy=strategy,
        )

    def absorb_vision_models_parallel(self, model_specs: List[Dict[str, Any]]) -> List[Any]:
        """Absorb multiple models simultaneously using ThreadPoolExecutor.

        Parameters
        ----------
        model_specs:
            List of dicts, each with keys: ``model`` (or ``model_or_name``),
            ``domain``, and optionally ``model_id``, ``layer_names``,
            ``max_samples``, ``strategy``.

        Returns
        -------
        List[AbsorptionReport]
        """
        if not getattr(self.config, "enable_vision_absorption", False):
            raise RuntimeError(
                "Vision absorption is disabled. "
                "Set NSCKConfig.enable_vision_absorption=True or use NSCKConfig.vision()."
            )
        self._ensure_vision_components()
        specs = []
        for s in model_specs:
            spec = dict(s)
            if "model_or_name" in spec and "model" not in spec:
                spec["model"] = spec.pop("model_or_name")
            specs.append(spec)
        return self._vision_absorber.absorb_batch(specs)

    def analyze_image(
        self,
        image: Any,
        reference_model=None,
        task_tag: str = "vision",
    ) -> Any:
        """Analyze an image using NSCK absorbed knowledge plus optional reference model.

        Parameters
        ----------
        image:
            PIL Image, numpy array, or base64 string.
        reference_model:
            Optional reference model (PretrainedModelAdapter or name string).
        task_tag:
            Semantic task tag.

        Returns
        -------
        VisionResponse
        """
        import time as _time  # noqa: PLC0415
        import uuid  # noqa: PLC0415
        t0 = _time.perf_counter()

        self._ensure_vision_components()

        from python.core.vision.vision_fusion import NSCKVisionFusion, VisionResponse  # noqa: PLC0415
        from python.core.vision.pretrained_adapter import PretrainedModelAdapter  # noqa: PLC0415

        # Encode the image into an HV using any absorbed model's projector
        query_hv = None
        nsck_label = "unknown"
        nsck_conf = 0.0
        prov: List[str] = []

        if self._absorption_memory and len(self._absorption_memory._records) > 0:
            # Use the first available projector to encode
            rec = self._absorption_memory._records[0]
            proj_key = rec.model_id
            if proj_key in self._vision_absorber._projectors:
                proj = self._vision_absorber._projectors[proj_key]
                try:
                    import numpy as _np  # noqa: PLC0415
                    if hasattr(image, "__array__"):
                        arr = _np.asarray(image, dtype=_np.float32).ravel()
                    else:
                        arr = _np.zeros(512, dtype=_np.float32)
                    # Normalize
                    norm = _np.linalg.norm(arr)
                    if norm > 1e-9:
                        arr = arr / norm
                    query_hv = proj.encode_new(arr)
                except Exception:
                    query_hv = None

            if query_hv is not None:
                results = self._absorption_memory.query_by_hv(query_hv, top_k=5)
                if results:
                    nsck_label = results[0].label
                    nsck_conf = results[0].confidence
                    prov = list({r.model_id for r in results})

        # Reference model result
        ref_label = None
        ref_conf = None
        if reference_model is not None:
            try:
                if isinstance(reference_model, str):
                    adapter = PretrainedModelAdapter.load(reference_model)
                else:
                    adapter = reference_model
                import numpy as _np  # noqa: PLC0415
                if hasattr(image, "__array__"):
                    arr = _np.asarray(image, dtype=_np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                else:
                    arr = _np.zeros((1, 3, 224, 224), dtype=_np.float32)
                preds = adapter.get_predictions(arr)
                if preds:
                    ref_label = preds[0].label
                    ref_conf = preds[0].confidence
            except Exception as e:
                logger.warning("Reference model inference failed: %s", e)

        # Create null HV if needed
        if query_hv is None:
            import python.core.vsa.hypervec_shim as hypervec_rs  # noqa: PLC0415
            query_hv = hypervec_rs.HyperVector(0)

        sem_mem = getattr(self._engine, "_semantic_memory", None)
        causal = getattr(self._engine, "_causal_graph", None)
        analogy = getattr(self._engine, "_analogy_engine", None)
        gw = getattr(self._engine, "_global_workspace", None)

        fusion = NSCKVisionFusion(
            absorption_memory=self._absorption_memory,
            domain_tagger=self._domain_tagger,
            semantic_memory=sem_mem,
            causal_graph=causal,
            analogy_engine=analogy,
            global_workspace=gw,
        )

        override_thresh = getattr(self.config, "vision_fusion_override_threshold", 0.95)
        min_depth = getattr(self.config, "vision_causal_chain_min_depth", 1)

        response = fusion.fuse(
            query_hv=query_hv,
            label=nsck_label,
            nsck_confidence=nsck_conf,
            reference_result=(ref_label, ref_conf) if ref_label else None,
            domain=None,
            override_threshold=override_thresh,
            min_causal_depth=min_depth,
        )

        # Inject provenance and latency
        import dataclasses as _dc  # noqa: PLC0415
        latency = (_time.perf_counter() - t0) * 1000
        response = _dc.replace(
            response,
            source_model_provenance=prov or response.source_model_provenance,
            latency_ms=latency,
        )
        return response

    def analyze_images(
        self,
        images: List[Any],
        reference_model=None,
        task_tag: str = "vision",
    ) -> List[Any]:
        """Analyze a batch of images."""
        return [self.analyze_image(img, reference_model=reference_model, task_tag=task_tag) for img in images]

    def get_absorption_stats(self) -> Dict[str, Any]:
        """Get statistics about absorbed models."""
        if self._vision_absorber is None:
            return {"absorbed_models": 0, "total_concepts": 0, "domains": []}
        stats = self._absorption_memory.get_stats() if self._absorption_memory else {}
        domain_stats = {}
        if self._domain_tagger:
            for d in self._domain_tagger.all_domains():
                domain_stats[d] = self._domain_tagger.get_domain_models(d)
        stats["domain_breakdown"] = domain_stats
        return stats

    # ------------------------------------------------------------------
    # V26: Societal Hypervector Knowledge Representation
    # ------------------------------------------------------------------

    def init_societal_world(
        self,
        concepts: Optional[List[Dict[str, Any]]] = None,
    ) -> "SocietyManager":
        """Initialise (or reinitialise) the societal knowledge graph.

        Parameters
        ----------
        concepts:
            Optional list of dicts with keys ``concept_id``, ``domain_path``,
            and ``role``.  Each concept's HV is drawn from the substrate's
            semantic memory (or freshly generated if absent).

        Returns
        -------
        SocietyManager
            The newly created manager, also stored as ``self._societal_manager``.
        """
        from python.core.societal.society_manager import SocietyManager
        from python.core.societal.societal_context_router import SocietalContextRouter
        from python.core.societal.living_hypervector import LivingHyperVector
        import python.core.vsa.hypervec_shim as _hv_mod

        mgr = SocietyManager(
            bond_threshold=getattr(self.config, "societal_bond_threshold", 0.65),
            max_bonds=getattr(self.config, "societal_max_bonds", 8),
            bond_decay_rate=getattr(self.config, "societal_bond_decay", 0.01),
            activation_decay_rate=getattr(self.config, "societal_activation_decay", 0.05),
            activation_spread_factor=getattr(self.config, "societal_activation_spread", 0.4),
            auto_cluster_interval=getattr(self.config, "societal_auto_cluster_interval", 10),
        )

        # Register concepts supplied by caller
        for concept_data in (concepts or []):
            cid = concept_data.get("concept_id", "")
            if not cid:
                continue
            # Try to retrieve HV from semantic memory, else create a fresh one
            hv = None
            try:
                hv = self._engine.semantic_memory.get_concept(cid)
            except Exception:
                pass
            if hv is None:
                hv = _hv_mod.HyperVector(seed=abs(hash(cid)) % (2 ** 31))

            lhv = LivingHyperVector(
                concept_id=cid,
                hv=hv,
                domain_path=concept_data.get("domain_path", []),
                role=concept_data.get("role", "leaf"),
                initial_activation=concept_data.get("initial_activation", 0.5),
                birth_epoch=mgr.epoch,
                metadata=concept_data.get("metadata", {}),
            )
            mgr.register(lhv)

        self._societal_manager = mgr
        self._societal_router = SocietalContextRouter(
            manager=mgr,
            top_k=getattr(self.config, "societal_top_k", 5),
            activation_delta=0.3,
            min_similarity=getattr(self.config, "societal_min_similarity", 0.55),
            auto_register=True,
        )
        logger.info(
            "[SUBSTRATE-V26] Societal world initialised with %d concepts", len(mgr)
        )
        return mgr

    def _ensure_societal_world(self) -> Optional["SocietyManager"]:
        """Return the societal manager, lazily creating it if needed."""
        if self._societal_manager is None and getattr(
            self.config, "enable_societal", False
        ):
            self.init_societal_world()
        return self._societal_manager

    @property
    def societal_manager(self) -> Optional["SocietyManager"]:
        """Access the societal manager (None if not initialised)."""
        return self._societal_manager

    @property
    def societal_router(self) -> Optional["SocietalContextRouter"]:
        """Access the societal context router (None if not initialised)."""
        return self._societal_router
