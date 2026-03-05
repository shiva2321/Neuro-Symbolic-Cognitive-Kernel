"""
NSCK Cognitive Engine
=====================
Central orchestrator for the Neural-Symbolic Cognitive Kernel.

This is the main entry point that integrates all cognitive modules into a
unified decision-making loop based on Global Workspace Theory (GWT).

Architecture Layers:
    1. VSA (HyperVector representation)
    2. Memory (Semantic + Episodic)
    3. Reasoning (GWT, Causal, Rules, Planning, Analogy)
    4. Language (NLU, NLG, Text Learning, Dialogue)
    5. Cognitive (Metacognition, Self-Model, Theory of Mind)
    6. Integration (Persistence, Brain Fusion)

The engine is domain-agnostic. Register task-specific verifiers and causal
graphs via register_task() to adapt it to any domain.
"""
import logging
import time
import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union

import python.core.vsa.hypervec_shim as hypervec_rs

# --- Core imports (always available) ---
from python.core.integration.config import NSCKConfig
from python.core.integration.explanation import ExplanationGenerator, Explanation
from python.core.perception.grounding_verifier import GroundingVerifier
from python.core.perception.snn_perception import SNNPerceptionModule
from python.core.reasoning.rule_learner import RuleLearner
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.memory.semantic_memory import SemanticMemory
from python.core.learning.curiosity import CuriosityModule
from python.core.reasoning.analogy import AnalogyEngine
from python.core.reasoning.math_reasoning import MathReasoner
from python.core.learning.active_inference import ActiveInferenceLearner
from python.core.reasoning.planner import STRIPSPlanner
from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition
from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner, CausalDiscovery
from python.core.cognitive.self_model import SelfModel
from python.core.cognitive.theory_of_mind import TheoryOfMind
from python.core.cognitive.metacognition import SafetyGate
from python.core.reasoning.societal_context_router import SocietalContextRouter
from python.core.reasoning.narrative_engine import NarrativeEngine
from python.core.integration.persistence import BrainStore
from python.core.integration.brain_fusion import BrainFusion, TaskBrain
from python.core.language.universal_input import UniversalInput
from python.core.language.language_module import LanguageModule
from python.core.language.dialogue_manager import DialogueManager

# V3 optional modules — imported lazily below but declared here for typing
try:
    from python.core.memory.homeostasis import MemoryHomeostasis
    _HOMEOSTASIS_AVAILABLE = True
except Exception:
    _HOMEOSTASIS_AVAILABLE = False

# V9: PerceptPacket + adapter imports
from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter
from python.core.adapters.dict_state_adapter import DictStateAdapter
from python.core.adapters.multimodal_fuser import MultimodalFuser

logger = logging.getLogger("nsck.cognitive_engine")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    """A proposal for the Global Workspace."""
    module: str
    action: str
    content: str
    salience: float = 0.0


@dataclass
class CognitiveState:
    """Snapshot of the cognitive system after a decision cycle."""
    task_tag: str
    situation_hv: Optional[hypervec_rs.HyperVector] = None
    active_predicates: List[str] = field(default_factory=list)
    chosen_action: str = "ACTION_STAY"
    confidence: float = 0.5
    self_confidence: float = 0.5
    exploration_mode: bool = False
    explanation: Optional[Explanation] = None
    trace: Dict[str, Any] = field(default_factory=dict)
    # V3 glass-box trace fields
    construction_match: Optional[Dict] = None
    frame_fill: Optional[Dict] = None
    coreference_chain: Optional[List[Dict]] = None
    belief_revision: Optional[Dict] = None
    system_used: Optional[str] = None
    system_1_confidence: Optional[float] = None
    homeostasis_actions: Optional[List[str]] = None
    # V13 KLE uncertainty (glass-box API)
    kle_uncertainty: Optional[float] = None       # KL-divergence-like uncertainty estimate [0,∞)
    uncertainty_bounds: Optional[tuple] = None    # (lower, upper) conformal prediction bounds


# ---------------------------------------------------------------------------
# CognitiveEngine
# ---------------------------------------------------------------------------

class CognitiveEngine:
    """
    Unified cognitive engine integrating all NSCK modules.

    Provides a domain-agnostic interface for:
    - decide(state, task_tag) -> CognitiveState with action + explanation
    - learn(state, action, reward, task_tag) -> update knowledge
    - transfer(source_task, target_task, state, predicates) -> adapt knowledge
    - process_dialogue(text) -> natural language response
    - explain() / why_not() / counterfactual() -> interpretability

    Register new domains via register_task(task_tag, verifier, causal_graph).
    """

    def __init__(
        self,
        config: Optional[NSCKConfig] = None,
        persistence_path: Optional[str] = None,
    ):
        self.config = config or NSCKConfig()

        # --- Persistence ---
        self.store = self._init_persistence(persistence_path)

        # --- Task registry (domain-agnostic) ---
        self.verifiers: Dict[str, GroundingVerifier] = {}
        self.causal_graphs: Dict[str, CausalGraph] = {}
        self.causal_reasoners: Dict[str, CausalReasoner] = {}
        self.task_brains: Dict[str, TaskBrain] = {}
        self.adapters: Dict[str, ModalityAdapter] = {}  # V9: modality adapters per task

        # --- Core cognitive modules ---
        self.rule_learner = RuleLearner(
            verifier=GroundingVerifier(),
            store=self.store,
            min_support=self.config.min_rule_support,
            min_success_rate=self.config.min_success_rate,
        )
        self.episodic_memory = EpisodicMemory(
            store=self.store,
            recent_capacity=self.config.memory_capacity,
        )
        self.semantic_memory = SemanticMemory()
        # V4: ProceduralMemory for System-1 fast-path
        from python.core.memory.procedural_memory import ProceduralMemory
        self.procedural_memory = ProceduralMemory(
            familiarity_threshold=getattr(self.config, 'procedural_familiarity_threshold', 0.72)
        )
        self.curiosity = CuriosityModule(
            novelty_threshold=self.config.novelty_threshold,
        )
        self.self_model = SelfModel()
        self.theory_of_mind = TheoryOfMind()
        self.global_workspace = GlobalWorkspace()
        self.causal_discovery = CausalDiscovery()
        self.planner = STRIPSPlanner()
        self.analogy = AnalogyEngine(load_defaults=True)
        self.analogy.load_sensor_domain_defaults()  # IIT: robot/env structural abstractions
        self.explainer = ExplanationGenerator()
        self.fusion = BrainFusion()
        self.math_reasoner = MathReasoner()
        
        # V5: Societal World Context
        self.societal_world = self.semantic_memory.societal_world
        
        self.active_inference = ActiveInferenceLearner(
            curiosity_module=self.curiosity,
            safety_threshold=0.7,
            societal_world=self.societal_world
        )
        
        # Narrative Reasoning
        self.societal_router = SocietalContextRouter(self.societal_world)
        self.narrative_engine = NarrativeEngine(self.societal_world)

        # --- Language ---
        self.universal_input = UniversalInput()
        # --- Language ---
        self.universal_input = UniversalInput()
        self.language = LanguageModule(semantic_memory=self.semantic_memory, use_vsa=True)
        self.dialogue = DialogueManager(self, self.language, config=self.config)

        # V11: NgramNLU integration
        self.ngram_nlu = None
        if self.config.enable_ngram_nlu:
            try:
                from python.core.language.ngram_nlu import NgramNLUAdapter
                self.ngram_nlu = NgramNLUAdapter(config=self.config)
                logger.info("NgramNLU initialized")
            except Exception as _exc:
                logger.warning("NgramNLU init failed: %s", _exc)

        # V11: Cross-modal correlation learning
        self.cross_modal = None  # type: Optional[Any]
        if self.config.enable_cross_modal_learning:
            try:
                from python.core.learning.cross_modal import CrossModalCorrelationLearner
                self.cross_modal = CrossModalCorrelationLearner()
                logger.info("CrossModalCorrelationLearner initialized")
            except Exception as _exc:
                logger.warning("CrossModal init failed: %s", _exc)
        
        # --- Perception (SNN) ---
        # Initialize the "Eyes" of the system
        try:
            self.perception = SNNPerceptionModule(
                input_dim=64,       # Standard sensory vector size
                snn_size=256,       # Number of LIF neurons
                hv_dimension=10240, # Match system VSA dimension
                n_concepts=50,
                simulation_time_ms=20.0,  # 20 ms gives 20 LIF steps — 2.5× faster than 50 ms
            )
            self._perception_call_count: int = 0   # for lazy-STDP scheduling
            logger.info("SNN Perception Module initialized (input_dim=64, snn_size=256, sim_ms=20)")
        except Exception as e:
            logger.error(f"Failed to init SNN Perception: {e}")
            self.perception = None

        # --- Mission / tracing ---
        self.mission_goal: Dict[str, Any] = {"type": "default", "threshold": 0}
        self._active_plan: Optional[List[str]] = None  # Current multi-step plan
        self._plan_goal: Optional[Set[str]] = None      # Goal the plan targets
        self.trace_history: List[Dict[str, Any]] = []
        self.max_trace_history = 1000
        
        # --- Reward-based learning (Q-values) ---
        self.q_values: Dict[Tuple[str, str], float] = {}  # (state_key, action) → Q-value
        self.state_visits: Dict[str, int] = {}  # state_key → visit count
        self.last_state_action: Optional[Tuple[str, str]] = None  # For credit assignment
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3  # Exploration rate

        # --- Stats ---
        self.stats = {
            "decisions": 0,
            "episodes_recorded": 0,
            "rules_induced": 0,
            "explorations": 0,
            "memory_recalls": 0,
            "plans_generated": 0,
            "safety_vetoes": 0,
            "gwt_broadcasts": 0,
            "sleep_cycles": 0,
            # V9 substrate stats
            "prototypes_built": 0,
            "transitive_inferences": 0,
            "auto_abstractions": 0,
            "auto_transfers": 0,
        }
        self.current_state = CognitiveState(task_tag="unknown")

        # V10: Optional neural rule scorer
        self.rule_scorer = None  # type: Optional[Any]
        self.msg_broadcaster = None

        # V11: Decision counter for continuous generalization
        self._decision_counter: int = 0

        # --- V3: Homeostasis ---
        self.homeostasis = None
        if _HOMEOSTASIS_AVAILABLE and self.config.enable_homeostasis:
            self.homeostasis = MemoryHomeostasis()

        # --- V3: Stigmergy path tracking ---
        self._last_reasoning_path: List[str] = []

        # --- Gap 1: Wire GWT broadcast subscribers ---
        self._register_gwt_subscribers()

        # --- V16: EWC Wiring ---
        self._continual_learner = None
        self._ewc_importance_buffer: Dict[str, List] = {}
        self._ewc_update_count: int = 0
        self._last_ewc_loss: float = 0.0
        if getattr(self.config, 'enable_ewc', False):
            try:
                from python.core.learning.continual_learning import ContinualLearner
                self._continual_learner = ContinualLearner(
                    ewc_lambda=getattr(self.config, 'ewc_lambda', 1000.0),
                    task_capacity=100,
                )
                logger.info("ContinualLearner (EWC) initialized")
            except Exception as _exc:
                logger.warning("ContinualLearner init failed: %s", _exc)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _register_gwt_subscribers(self):
        """Register cognitive modules as GWT broadcast subscribers.

        This is the *defining mechanism* of Global Workspace Theory:
        when a coalition wins, all subscribers hear the decision so they
        can update their internal state.
        """
        # RuleLearner already implements WorkspaceModule
        self.global_workspace.register_module("rule_learner", self.rule_learner)

        # Wrap EpisodicMemory, SemanticMemory, SelfModel, etc. as broadcast
        # listeners via lightweight adapters (they don't implement
        # WorkspaceModule natively, so we wrap them).
        from python.core.reasoning.global_workspace import WorkspaceModule

        class _EpisodicBroadcastAdapter(WorkspaceModule):
            """Primes episodic memory with last broadcast for similarity search."""
            def __init__(self, engine: 'CognitiveEngine'):
                self._engine = engine
            def receive_broadcast(self, content):
                # Track that a broadcast happened (for stats)
                self._engine.stats["gwt_broadcasts"] += 1

        class _SemanticBroadcastAdapter(WorkspaceModule):
            """Runs spreading activation when a concept is broadcast."""
            def __init__(self, semantic_memory: SemanticMemory):
                self._mem = semantic_memory
                self.last_primed: Dict[str, float] = {}
            def receive_broadcast(self, content):
                # If content is an action string, prime related concepts
                if isinstance(content, str) and content in self._mem.concept_graph:
                    self.last_primed = self._mem.spread_activation(
                        [content], steps=2, decay=0.5
                    )

        self._episodic_adapter = _EpisodicBroadcastAdapter(self)
        self._semantic_adapter = _SemanticBroadcastAdapter(self.semantic_memory)
        self.global_workspace.register_module("episodic_memory", self._episodic_adapter)
        self.global_workspace.register_module("semantic_memory", self._semantic_adapter)
        logger.info("GWT: registered %d broadcast subscribers",
                    len(self.global_workspace.modules))

    @staticmethod
    def _init_persistence(path: Optional[str]) -> Optional[BrainStore]:
        if path:
            return BrainStore(path)
        default_db = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "nsck_brain.db"
        )
        try:
            store = BrainStore(default_db)
            logger.info("Auto-wired BrainStore at %s", default_db)
            return store
        except Exception as exc:
            logger.warning("Auto-wire failed (%s), running without persistence", exc)
            return None

    # ------------------------------------------------------------------
    # Task registration (domain-agnostic)
    # ------------------------------------------------------------------

    def register_task(
        self,
        task_tag: str,
        verifier: Optional[GroundingVerifier] = None,
        causal_graph: Optional[CausalGraph] = None,
        adapter: Optional[ModalityAdapter] = None,
    ):
        """Register a new task domain with optional verifier and causal graph.

        Parameters
        ----------
        task_tag : str
            Unique identifier for the domain (e.g. ``"physics"``, ``"chess"``).
        verifier : GroundingVerifier, optional
            Extracts symbolic predicates from raw state dicts.
        causal_graph : CausalGraph, optional
            Bootstrap causal knowledge for this domain.
        adapter : ModalityAdapter, optional
            V9: Modality adapter for this task.  If not provided, a
            :class:`DictStateAdapter` is auto-created from *verifier*.
        """
        if verifier:
            self.verifiers[task_tag] = verifier
            # Wire domain verifier into rule_learner so observe() uses correct predicates
            self.rule_learner.register_verifier(task_tag, verifier)
        if causal_graph is None:
            causal_graph = CausalGraph()
        self.causal_graphs[task_tag] = causal_graph
        self.causal_reasoners[task_tag] = CausalReasoner(causal_graph)
        try:
            self.task_brains[task_tag] = TaskBrain(task_tag)
        except Exception:
            pass

        # V9: Register adapter — auto-create DictStateAdapter if none provided
        if adapter is not None:
            self.adapters[task_tag] = adapter
        else:
            _verifier = verifier or self.verifiers.get(task_tag, GroundingVerifier())
            self.adapters[task_tag] = DictStateAdapter(_verifier, self.episodic_memory)

        logger.info("Registered task '%s'", task_tag)

        # V9: Auto-transfer — try to transfer rules from existing tasks to this new one
        if self.config.enable_auto_transfer:
            self._auto_transfer_to_task(task_tag)

        # V16: Register task with EWC continual learner
        if self._continual_learner is not None:
            self._continual_learner.register_task(task_tag)
            if task_tag not in self._ewc_importance_buffer:
                self._ewc_importance_buffer[task_tag] = []

    # ------------------------------------------------------------------
    # Core cognitive loop
    # ------------------------------------------------------------------

    def get_concept_hv(self, concept: str) -> hypervec_rs.HyperVector:
        """Generate a stable VSA hypervector for a string concept."""
        return hypervec_rs.HyperVector(hash(concept) % (2**32))

    def _get_state_key(self, state: Dict[str, Any], task_tag: str) -> str:
        """Create a hashable key from state for Q-value lookup."""
        # Only include relevant state features
        if task_tag == "maze_navigation":
            return f"{task_tag}:({state.get('position_x', 0)},{state.get('position_y', 0)})"
        else:
            # Generic state key (first 5 state values)
            key_parts = [task_tag]
            for k, v in sorted(state.items())[:5]:
                key_parts.append(f"{k}={v}")
            return ":".join(key_parts)
    
    def _get_best_action_from_q(self, state_key: str, available_actions: List[str]) -> Optional[str]:
        """Get best action according to Q-values."""
        if not available_actions:
            return None
        
        # Get Q-values for all actions from this state
        q_vals = [(action, self.q_values.get((state_key, action), 0.0)) for action in available_actions]
        
        # Epsilon-greedy: sometimes explore
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        
        # Otherwise choose best action
        best_action = max(q_vals, key=lambda x: x[1])[0]
        return best_action

        return best_action

    def perceive_and_decide(
        self,
        sensory_input: np.ndarray,
        task_tag: str
    ) -> CognitiveState:
        """Full Neuro-Symbolic Cycle: SNN Perception -> Global Workspace -> Action."""
        
        # 1. Run SNN Perception
        snn_result = None
        if self.perception:
            # Lazy STDP: only run full weight update every 5th call.
            # Pure-inference calls save ~80% of the STDP outer-product cost.
            self._perception_call_count = getattr(self, '_perception_call_count', 0) + 1
            learn_this_step = (self._perception_call_count % 5 == 0)
            snn_result = self.perception.perceive(sensory_input, learn=learn_this_step)
            
        # 2. Convert SNN result to State Dict for symbol grounding.
        # Extract meaningful predicates from the sensory array and from the
        # SNN concept mapper (which may resolve to semantic-memory labels via
        # VSA cleanup-memory lookup when a SemanticMemory is attached).
        state: Dict[str, Any] = {"snn_active": True}
        
        if sensory_input is not None and hasattr(sensory_input, '__len__') and len(sensory_input) > 0:
            arr = np.asarray(sensory_input, dtype=np.float64)
            mean_val  = float(np.mean(arr))
            std_val   = float(np.std(arr))
            max_val   = float(np.max(arr))
            state["signal_mean"]   = mean_val
            state["signal_std"]    = std_val
            state["signal_max"]    = max_val
            state["high_activity"] = int(mean_val > 0.6)
            state["low_activity"]  = int(mean_val < 0.2)
            state["noisy"]         = int(std_val > 0.4)
            if snn_result:
                concept_id = snn_result.get("concept_id", 0)
                state["concept_id"]    = concept_id
                state["snn_strength"]  = float(snn_result.get("strength", 0.0))
                n_spikes = snn_result.get("n_spikes", 0)
                state["active_firing"] = int(n_spikes > 10)
                state["sparse_firing"] = int(n_spikes <= 10)
                # Resolve semantic label via concept mapper cleanup memory
                if self.perception and hasattr(self.perception, "concept_mapper"):
                    label = self.perception.concept_mapper.get_concept_label(concept_id)
                    state["concept_label"] = label
        
        # 3. Inject SNN Concept into Global Workspace (as a 'Bot-Up' Coalition)
        snn_coalition = None
        if snn_result:
            concept_id = snn_result['concept_id']
            # Resolve to a semantic label when the mapper has one
            if self.perception and hasattr(self.perception, "concept_mapper"):
                concept_name = self.perception.concept_mapper.get_concept_label(concept_id)
            else:
                concept_name = f"Percept_{concept_id}"
            activation = snn_result['strength']
            
            snn_coalition = Coalition(
                source="SNN_PERCEPTION",
                content=concept_name,
                base_salience=activation,
                relevance=0.8, # High relevance for sensory data
                sender_confidence=activation
            )
            
        # 4. Proceed with standard decision cycle, but PASS the SNN coalition
        # fast_mode=True: skips memory+planner coalitions → <5ms GWT target
        return self.decide(state, task_tag, external_coalition=snn_coalition, fast_mode=True)


    def decide(
        self,
        state: Union[Dict[str, Any], PerceptPacket],
        task_tag: str,
        metacognition_result: Optional[Dict] = None,
        external_coalition: Optional[Any] = None,
        fast_mode: bool = False  # Skip memory+planner coalitions for low-latency GWT path
    ) -> CognitiveState:
        """Run one cognitive cycle and return a decision.

        Parameters
        ----------
        state : dict or PerceptPacket
            Raw state from the environment or data source, or a pre-built
            :class:`PerceptPacket` from any modality adapter (V9).
            Passing a dict is fully backward-compatible — it is auto-wrapped
            via the task's registered :class:`DictStateAdapter`.
        task_tag : str
            Which registered task this state belongs to.
        metacognition_result : dict, optional
            External metacognition input (e.g. from SNN layer).

        Returns
        -------
        CognitiveState
            Contains chosen action, explanation, confidence, trace, etc.
        """
        self.stats["decisions"] += 1

        # V11: Route numeric sequences (list/ndarray of numbers) via NumericSequenceAdapter
        import numpy as np
        if isinstance(state, (list, np.ndarray)) and not isinstance(state, str):
            try:
                if len(state) > 0 and isinstance(state[0], (int, float, np.floating, np.integer)):
                    from python.core.adapters.numeric_sequence_adapter import NumericSequenceAdapter
                    _ns_adapter = NumericSequenceAdapter()
                    state = _ns_adapter.encode(state, task_tag)
            except (TypeError, IndexError, Exception):
                state = {"value": state}

        # V9: Convert raw dict → PerceptPacket (backward-compatible auto-wrap)
        if isinstance(state, PerceptPacket):
            percept = state
            # raw_state_dict used for legacy internal references (explanation, safety, etc.)
            raw_state_dict: Dict[str, Any] = percept.raw_state or {}
        else:
            # Legacy dict path — use registered adapter or create inline
            raw_state_dict = state if isinstance(state, dict) else {}
            _adapter = self.adapters.get(task_tag)
            if _adapter is not None:
                percept = _adapter.encode(raw_state_dict, task_tag)
            else:
                # Fallback: inline grounding (original decide() lines 457-463)
                verifier = self.verifiers.get(task_tag, GroundingVerifier())
                _active_preds = verifier.get_active_predicates(raw_state_dict, context=task_tag)
                _situation_hv = self.episodic_memory.create_situation_hv(
                    raw_state_dict, task_tag, _active_preds
                )
                percept = PerceptPacket.make(
                    modality="dict",
                    situation_hv=_situation_hv,
                    active_predicates=frozenset(_active_preds),
                    raw_state=raw_state_dict,
                    adapter_name="inline",
                )

        # Extract grounded symbols and situation HV from packet
        active_preds = list(percept.active_predicates)
        situation_hv = percept.situation_hv

        # V11: NgramNLU enrichment for text inputs
        if self.ngram_nlu is not None:
            _text_input = raw_state_dict.get("text", "") if isinstance(raw_state_dict, dict) else ""
            if isinstance(_text_input, str) and _text_input.strip():
                try:
                    _nlu_result = self.ngram_nlu.process(_text_input)
                    _intent = _nlu_result.get("intent", {})
                    _intent_label = _intent.get("label", "") if isinstance(_intent, dict) else str(_intent)
                    if _intent_label:
                        active_preds = list(active_preds) + [f"INTENT_{_intent_label.upper()}"]
                except Exception:
                    pass

        # 3. Curiosity / exploration check
        confidence = (
            metacognition_result.get("confidence", 0.5)
            if metacognition_result
            else 0.5
        )
        explore_decision = self.curiosity.should_explore(
            situation_hv, task_tag, confidence
        )

        # 4. Build coalitions for GWT competition
        coalitions: List[Coalition] = []

        # A. External metacognition proposal (e.g. SNN fast system)
        if metacognition_result and metacognition_result.get("action"):
            coalitions.append(Coalition(
                source="EXTERNAL",
                content=metacognition_result["action"],
                base_salience=confidence,
                relevance=0.0,
                sender_confidence=self.self_model.get_confidence(task_tag),
            ))

        # A2. SNN Perception Coalition
        if external_coalition:
            coalitions.append(external_coalition)

        # Math routing (V8) — detect math queries and add high-confidence coalition
        _math_text = raw_state_dict.get("math_query") or (raw_state_dict.get("text", "") if isinstance(raw_state_dict.get("text"), str) else "")
        if _math_text and self.universal_input.is_mathematical(_math_text):
            _math_result = self._solve_math(_math_text)
            if _math_result is not None:
                coalitions.append(Coalition(
                    source="MATH",
                    content=str(_math_result.get("answer", "")),
                    base_salience=0.95,
                    relevance=0.5,
                    sender_confidence=0.95,
                ))

        # B. Rule-based proposal
        applicable = self.rule_learner.get_applicable_rules(active_preds, task_tag)
        if applicable:
            # V10: re-rank with neural scorer if available
            if self.rule_scorer is not None and len(applicable) > 1:
                rules_only = [r for r, _ in applicable]
                scored = self.rule_scorer.rank_rules(rules_only)
                rule = scored[0]
                # Find original score using identity comparison (Rule is not hashable)
                score = next((s for r, s in applicable if r is rule), applicable[0][1])
            else:
                rule, score = applicable[0]
            coalitions.append(Coalition(
                source="RULES",
                content=rule.consequence,
                base_salience=score,
                relevance=0.2,
                sender_confidence=rule.confidence,
            ))

        # C. Exploration proposal
        if explore_decision.should_explore:
            explore_act = self._get_exploration_action(raw_state_dict, task_tag)
            salience = 0.6 + (0.2 if "stagnant" in explore_decision.reason else 0.0)
            coalitions.append(Coalition(
                source="EXPLORATION",
                content=explore_act,
                base_salience=salience,
                relevance=0.0,
                sender_confidence=0.5,
            ))
        
        # C2. Q-LEARNING proposal (reward-based policy)
        state_key = self._get_state_key(raw_state_dict, task_tag)
        available_actions = self.get_allowed_actions(task_tag)  # Use actual task actions
        q_action = self._get_best_action_from_q(state_key, available_actions)
        if q_action:
            # Salience based on visit count (more confident after more visits)
            visit_count = self.state_visits.get(state_key, 0)
            q_salience = min(0.9, 0.5 + (visit_count * 0.05))  # Increases with experience
            q_value = self.q_values.get((state_key, q_action), 0.0)
            
            # Only propose if Q-value is positive or we have experience
            if q_value > -1.0 or visit_count > 2:
                coalitions.append(Coalition(
                    source="Q_LEARNING",
                    content=q_action,
                    base_salience=q_salience,
                    relevance=0.1,
                    sender_confidence=min(0.9, 0.3 + visit_count * 0.1),
                ))

        # --- V3: Dual Process ---
        # When enabled, try System 1 (fast) coalitions first.
        # Only proceed to System 2 (slow) if System 1 confidence is low.
        #
        # NOTE: Coalition.activation sums several components and can exceed 1.0
        # (base_salience + relevance + affect_match + sender_confidence*0.5).
        # We normalize to [0,1] before comparing against the threshold so that
        # threshold=0.75 means "top 25% of possible activation" consistently.
        _ACTIVATION_NORM = 2.0  # theoretical max: 1.0+1.0+0.0+0.5*1.0 = 2.5, use 2.0 for practical range
        system_used: Optional[str] = None
        system_1_confidence: Optional[float] = None
        if self.config.enable_dual_process and not fast_mode:
            # System 1: already-built coalitions (Q_LEARNING, RULES, EXPLORATION)
            s1_winner = self.global_workspace.compete(coalitions)
            raw_activation = s1_winner.activation if s1_winner else 0.0
            s1_confidence = min(1.0, raw_activation / _ACTIVATION_NORM)
            system_1_confidence = s1_confidence
            if s1_confidence >= self.config.system1_confidence_threshold:
                # Use System 1 directly — skip expensive memory+planner
                system_used = "system_1"
                winner_coalition = s1_winner
                if winner_coalition:
                    winner_coalition = self._safety_check(
                        winner_coalition, coalitions, raw_state_dict, task_tag
                    )
                # Skip to action determination
                goto_action_determination = True
            else:
                # System 2: add slow coalitions
                system_used = "system_2"
                goto_action_determination = False
        else:
            goto_action_determination = False

        if not goto_action_determination:
            winner_coalition = None  # will be set below

        # D. [Gap 2] MEMORY coalition — episodic recall for case-based reasoning
        # Skip in fast_mode or when System 1 succeeded
        if not fast_mode and not goto_action_determination:
            memory_coalition = self._build_memory_coalition(
                situation_hv, task_tag, active_preds
            )
            if memory_coalition:
                coalitions.append(memory_coalition)

        # E. [Gap 4] PLANNER coalition — goal-directed multi-step planning
        # Skip in fast_mode or when System 1 succeeded
        if not fast_mode and not goto_action_determination:
            planner_coalition = self._build_planner_coalition(
                active_preds, task_tag
            )
            if planner_coalition:
                coalitions.append(planner_coalition)
                
        # E2. [V5] SOCIETAL NARRATIVE coalition
        if not fast_mode and not goto_action_determination:
            if hasattr(self, 'narrative_engine') and situation_hv is not None:
                # Goal HV - dummy target to invoke cross-domain routing
                goal_hv = self.get_concept_hv("survival_goal") 
                path = self.narrative_engine.generate_narrative_path(situation_hv, goal_hv)
                if path and path[-1].get("stage") != "Failed":
                    next_domain = path[1].get("domain_concept", "Unknown") if len(path) > 1 else path[0].get("domain_concept", "Unknown")
                    coalitions.append(Coalition(
                        source="NARRATIVE",
                        content=f"Proceed via domain: {next_domain}",
                        base_salience=0.6,
                        relevance=0.4,
                        sender_confidence=0.7,
                    ))

        # Active inference (V8) — adjust coalition salience by free energy
        if self.config.enable_active_inference and situation_hv is not None:
            _ai_weight = getattr(self.config, 'active_inference_weight', 0.2)
            for _c in coalitions:
                _fe = self.active_inference.free_energy(_c.content, situation_hv)
                _c.base_salience += _ai_weight * (0.5 - _fe)

        # 5. GWT competition (only if System 1 didn't already win)
        if not goto_action_determination:
            winner_coalition = self.global_workspace.compete(coalitions)

            # 5b. [Gap 5] Safety gate veto — check winning action before committing
            if winner_coalition:
                winner_coalition = self._safety_check(
                    winner_coalition, coalitions, raw_state_dict, task_tag
                )

        # 6. Determine final action
        trace: Dict[str, Any] = {"proposals": len(coalitions)}
        if system_used:
            trace["system_used"] = system_used
        if system_1_confidence is not None:
            trace["system_1_confidence"] = system_1_confidence

        if winner_coalition:
            action = winner_coalition.content
            # Normalize action name against allowed actions (strip or add ACTION_ prefix)
            _allowed = self.get_allowed_actions(task_tag)
            if _allowed and action not in _allowed:
                _stripped = action[7:] if action.startswith("ACTION_") else action
                if _stripped in _allowed:
                    action = _stripped
                elif f"ACTION_{action}" in _allowed:
                    action = f"ACTION_{action}"
            winner_name = winner_coalition.source
            # V4: EWC rule importance — increment gwt_win_count for winning RULES coalition
            if winner_name == "RULES":
                try:
                    for rule in self.rule_learner.learned_rules.get(task_tag, []):
                        if rule.consequence == action:
                            rule.gwt_win_count += 1
                            rule.ewc_importance = min(1.0, rule.gwt_win_count / 20.0)
                            break
                except Exception:
                    pass
            trace["mode"] = winner_name
            trace["winner"] = winner_name
            trace["reason"] = (
                f"Winner: {winner_name} "
                f"(Activation: {winner_coalition.activation:.2f})"
            )
            self.self_model.update_confidence(task_tag, confidence)
        else:
            action = self._default_action(raw_state_dict, task_tag)
            winner_name = "DEFAULT"
            trace["mode"] = "default"
            trace["winner"] = "DEFAULT"

        # 7. Generate explanation
        trace["confidence"] = confidence
        explanation = self.explainer.explain_action(action, raw_state_dict, task_tag, trace)

        # 7b. V30: Trigger CounterfactualReasoner for hypothetical queries (Bug 5.3 fix)
        _text_input = raw_state_dict.get("text", "") if isinstance(raw_state_dict, dict) else ""
        _HYPOTHETICAL_PATTERNS = (
            "if ", " if ", "what if", " would ", " unless ", " suppose ",
            "had been", "had not", "hadn't", "would have",
        )
        if isinstance(_text_input, str) and any(p in _text_input.lower() for p in _HYPOTHETICAL_PATTERNS):
            try:
                _reasoner = self.causal_reasoners.get(task_tag)
                if _reasoner is not None:
                    _cf_result = _reasoner.counterfactual(
                        action, "alternative_action", raw_state_dict, task_tag
                    )
                    trace["counterfactual"] = {
                        "triggered": True,
                        "query": _text_input[:120],
                        "original_outcome": _cf_result.original_outcome,
                        "counterfactual_outcome": _cf_result.counterfactual_outcome,
                        "confidence": _cf_result.confidence,
                    }
            except Exception as _cf_exc:
                trace["counterfactual"] = {"triggered": True, "error": str(_cf_exc)}
        else:
            if "counterfactual" not in trace:
                trace["counterfactual"] = {"triggered": False}

        # 8. Update curiosity
        self.curiosity.record_visit(situation_hv, task_tag)

        # 9. Record trace
        full_trace = {
            "timestamp": time.time(),
            "task": task_tag,
            "winner": winner_name,
            "action": action,
            "proposals": [c.source for c in coalitions],
        }
        self.trace_history.append(full_trace)
        if len(self.trace_history) > self.max_trace_history:
            self.trace_history.pop(0)

        # 10. Build cognitive state
        self.current_state = CognitiveState(
            task_tag=task_tag,
            situation_hv=situation_hv,
            active_predicates=active_preds,
            chosen_action=action,
            confidence=confidence,
            self_confidence=self.self_model.get_confidence(task_tag),
            exploration_mode=explore_decision.should_explore,
            explanation=explanation,
            trace=trace,
            system_used=system_used,
            system_1_confidence=system_1_confidence,
        )
        
        # Track state-action for Q-learning updates
        state_key = self._get_state_key(raw_state_dict, task_tag)
        self.last_state_action = (state_key, action)
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1

        # V11: Continuous generalization
        self._decision_counter += 1
        if (self.config.enable_continuous_generalization
                and self._decision_counter % self.config.generalization_interval == 0):
            self._incremental_generalize()

        # V16: EWC importance buffer update
        if self._continual_learner is not None and self.current_state.situation_hv is not None:
            _buf = self._ewc_importance_buffer.setdefault(task_tag, [])
            _buf.append(self.current_state.situation_hv)
            _window = getattr(self.config, 'ewc_importance_window', 100)
            if len(_buf) > _window:
                _buf.pop(0)
            _protected = list(self._continual_learner.get_protected_concepts(task_tag))
            trace["ewc_protected_concepts"] = _protected
            trace["ewc_loss"] = self._last_ewc_loss

        return self.current_state

    # ------------------------------------------------------------------
    # V11: Continuous generalization
    # ------------------------------------------------------------------

    def _incremental_generalize(self) -> None:
        """Incremental generalization step wired into the decide() loop (V11)."""
        try:
            self.semantic_memory.build_prototypes(min_members=3)
        except Exception:
            pass
        try:
            self.semantic_memory.infer_transitive(max_hops=1)
        except Exception:
            pass
        if hasattr(self, 'cross_domain') and self.cross_domain:
            try:
                self.cross_domain.auto_discover_abstractions()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # V9: Multimodal convenience method
    # ------------------------------------------------------------------

    def decide_multimodal(
        self,
        inputs: List[Any],
        task_tag: str,
        metacognition_result: Optional[Dict] = None,
        fast_mode: bool = False,
    ) -> CognitiveState:
        """Decide from multiple simultaneous modality inputs (V9).

        Each element in *inputs* is encoded via the task's registered adapter,
        then all resulting :class:`PerceptPacket` objects are fused into a
        single multimodal packet before the normal GWT decision loop.

        Parameters
        ----------
        inputs : list
            Raw inputs from different modalities (dict, str, ndarray, etc.).
        task_tag : str
            Registered task domain identifier.
        metacognition_result : dict, optional
            External metacognition input.
        fast_mode : bool
            Skip memory/planner coalitions for low-latency path.

        Returns
        -------
        CognitiveState
        """
        adapter = self.adapters.get(task_tag)
        packets = []
        for raw in inputs:
            if isinstance(raw, PerceptPacket):
                packets.append(raw)
            elif adapter is not None:
                packets.append(adapter.encode(raw, task_tag))
            else:
                # Fallback: inline dict encoding
                if not isinstance(raw, dict):
                    raw = {"value": raw}
                verifier = self.verifiers.get(task_tag, GroundingVerifier())
                preds = verifier.get_active_predicates(raw, context=task_tag)
                hv = self.episodic_memory.create_situation_hv(raw, task_tag, preds)
                packets.append(PerceptPacket.make(
                    modality="dict", situation_hv=hv,
                    active_predicates=frozenset(preds), raw_state=raw,
                    adapter_name="inline",
                ))

        fused = MultimodalFuser().fuse(packets)
        return self.decide(
            fused, task_tag,
            metacognition_result=metacognition_result,
            fast_mode=fast_mode,
        )

    # ------------------------------------------------------------------
    # V9: Auto-transfer helper
    # ------------------------------------------------------------------

    def _auto_transfer_to_task(self, new_task_tag: str) -> None:
        """Transfer rules from all known tasks to *new_task_tag* (V9).

        Called automatically at the end of ``register_task()`` when
        ``config.enable_auto_transfer`` is True.
        """
        transferred = 0
        for source_task, rules in self.rule_learner.learned_rules.items():
            if source_task == new_task_tag or not rules:
                continue
            # Collect concept HVs for both tasks
            hvs_source = self._collect_concept_hvs_for_task(source_task)
            hvs_target = self._collect_concept_hvs_for_task(new_task_tag)
            if not hvs_source or not hvs_target:
                continue
            # Auto-discover structural mappings
            try:
                mappings = self.analogy.auto_discover_abstractions(
                    source_task, new_task_tag, hvs_source, hvs_target
                )
            except Exception:
                mappings = []
            if not mappings:
                continue
            # Transfer each learned rule
            for rule in rules:
                try:
                    new_cond, new_action = self.analogy.transfer_rule(
                        set(rule.condition), rule.consequence,
                        source_task, new_task_tag,
                    )
                    if new_cond:
                        self.rule_learner.add_transferred_rule(
                            new_task_tag, frozenset(new_cond), new_action,
                            source_task=source_task,
                            source_confidence=rule.confidence,
                        )
                        transferred += 1
                except Exception:
                    pass
        if transferred:
            self.stats["auto_transfers"] += transferred
            logger.info(
                "[TRANSFER] Auto-transferred %d rules to '%s'", transferred, new_task_tag
            )

    def _collect_concept_hvs_for_task(
        self, task_tag: str
    ) -> Dict[str, hypervec_rs.HyperVector]:
        """Gather concept HVs associated with *task_tag* (V9).

        Collects from:
        - Learned rule conditions (predicate name → deterministic HV)
        - Causal graph node names
        - Semantic memory concepts whose name starts with ``task_tag``

        Returns
        -------
        dict mapping concept name → HyperVector
        """
        hvs: Dict[str, hypervec_rs.HyperVector] = {}

        # 1. Rule condition predicates
        for rule in self.rule_learner.learned_rules.get(task_tag, []):
            for pred in rule.condition:
                if pred not in hvs:
                    hvs[pred] = self.get_concept_hv(pred)

        # 2. Causal graph nodes
        graph = self.causal_graphs.get(task_tag)
        if graph:
            for link in graph.all_links:
                for name in (link.cause, link.effect):
                    if name not in hvs:
                        hvs[name] = self.get_concept_hv(name)

        # 3. Semantic memory concepts labelled with task prefix
        for concept in self.semantic_memory.concept_hvs:
            if concept.startswith(task_tag) or f"_{task_tag}" in concept:
                hvs[concept] = self.semantic_memory.concept_hvs[concept]

        return hvs

    def record_outcome(
        self,
        reward: float,
        task_tag: str,
        new_state: Optional[Dict[str, Any]] = None
    ):
        """
        Record outcome of last decision and provide reward feedback.
        
        This enables reward-modulated Hebbian learning and Q-learning for RL.
        
        Args:
            reward: Reward signal (positive = good, negative = bad, 0 = neutral)
            task_tag: Task identifier
            new_state: Optional new state after action (for Q-learning TD update)
        """
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        if self.last_state_action:
            state_key, action = self.last_state_action
            current_q = self.q_values.get((state_key, action), 0.0)
            
            # Compute TD target
            if new_state:
                new_state_key = self._get_state_key(new_state, task_tag)
                available_actions = self.get_allowed_actions(task_tag)  # Use actual task actions
                
                # Get max Q-value for next state
                next_q_values = [self.q_values.get((new_state_key, a), 0.0) for a in available_actions]
                max_next_q = max(next_q_values) if next_q_values else 0.0
                
                # TD update
                td_target = reward + self.discount_factor * max_next_q
            else:
                # Terminal state or no next state provided
                td_target = reward
            
            # Update Q-value
            td_error = td_target - current_q
            new_q = current_q + self.learning_rate * td_error
            self.q_values[(state_key, action)] = new_q
            
            # Decay epsilon: lower floor (0.02) + faster decay (0.98) for quicker convergence
            self.epsilon = max(0.02, self.epsilon * 0.98)
        
        # Update last episode with reward
        if hasattr(self.episodic_memory, 'last_episode') and self.episodic_memory.last_episode:
            self.episodic_memory.last_episode.reward = reward
        
        # Provide reward feedback to rule learner
        if hasattr(self.rule_learner, 'record_outcome'):
            self.rule_learner.record_outcome(reward, task_tag)
        
        # Update self-model based on outcome
        # Positive reward → increase confidence, negative → decrease
        current_conf = self.self_model.get_confidence(task_tag)
        if reward > 0:
            new_conf = min(1.0, current_conf + 0.05)
        elif reward < 0:
            new_conf = max(0.1, current_conf - 0.05)
        else:
            new_conf = current_conf
        self.self_model.update_confidence(task_tag, new_conf)
        
        # Store reward in stats
        if 'total_reward' not in self.stats:
            self.stats['total_reward'] = 0.0
            self.stats['reward_count'] = 0
        self.stats['total_reward'] += reward
        self.stats['reward_count'] += 1

        # V3: Stigmergy — mark reasoning path on positive reward
        if reward > 0 and self.config.enable_stigmergy:
            if self._last_reasoning_path and hasattr(self.semantic_memory, 'mark_path'):
                self.semantic_memory.mark_path(self._last_reasoning_path, reward)

        # V8: update active inference world model if we have state context
        if hasattr(self, 'current_state') and self.current_state is not None:
            try:
                if hasattr(self.current_state, 'situation_hv') and self.current_state.situation_hv is not None:
                    next_hv = self.current_state.situation_hv
                    action = self.current_state.chosen_action
                    if new_state is not None:
                        _nsk = self._get_state_key(new_state, task_tag)
                        next_hv_candidate = hypervec_rs.HyperVector(hash(_nsk) % (2**32))
                    else:
                        next_hv_candidate = next_hv
                    self.active_inference.update_world_model(
                        self.current_state.situation_hv, action, next_hv_candidate
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "neutral",
        next_state: Optional[Dict[str, Any]] = None,
    ):
        """Learn from a single experience tuple.

        Parameters
        ----------
        state : dict
            State when the action was taken.
        action : str
            Action taken.
        reward : float
            Scalar reward received.
        task_tag : str
            Domain identifier.
        outcome : str
            Human-readable outcome label.
        next_state : dict, optional
            Resulting state (enables causal discovery).
        """
        # 1. Rule learning observation — pass pre-computed active_preds so the
        #    rule learner uses the same domain predicates as decide() did.
        _learn_preds = (
            list(self.current_state.active_predicates)
            if self.current_state and self.current_state.active_predicates
            else None
        )
        self.rule_learner.observe(
            state, action, reward, task_tag, outcome,
            active_preds=_learn_preds,
        )

        # 2. Danger vector registration for mental rehearsal
        if (outcome == "death" or reward < -0.5) and self.current_state.situation_hv:
            self.global_workspace.register_danger(self.current_state.situation_hv)

        # 3. Episodic memory recording
        if self.current_state.situation_hv:
            episode = LiveEpisode(
                timestamp=time.time(),
                task_tag=task_tag,
                situation_hv=self.current_state.situation_hv,
                state=state,
                action=action,
                outcome=outcome,
                reward=reward,
            )
            self.episodic_memory.record(episode)
            self.stats["episodes_recorded"] += 1

        # 4. Self-model update
        actual_success = outcome == "success"
        self.curiosity.record_outcome(task_tag, reward > 0)
        self.self_model.update(
            task_tag=task_tag,
            action=action,
            predicted_confidence=self.current_state.confidence,
            actual_success=actual_success,
            reward=reward,
        )

        # 5. Periodic rule induction
        if self.stats["episodes_recorded"] % 50 == 0:
            new_rules = self.rule_learner.induce_rules(task_tag)
            self.stats["rules_induced"] += len(new_rules)

        # V16: EWC importance update
        if self._continual_learner is not None:
            self._ewc_update_count += 1
            _interval = getattr(self.config, 'ewc_consolidate_interval', 500)
            if self._ewc_update_count % _interval == 0:
                self._compute_and_record_vsa_importance(task_tag)

        # V4: Auto-populate ProceduralMemory for positive-reward familiar states
        if (reward > 0.0 and self.current_state is not None and
                self.current_state.situation_hv is not None and
                hasattr(self, 'procedural_memory')):
            try:
                self.procedural_memory.cache_skill(
                    context_hv=self.current_state.situation_hv,
                    action=action,
                    reward=float(reward),
                    label=task_tag,
                )
            except Exception:
                pass

        # 6. Causal discovery (requires next_state)
        if next_state:
            self._update_causal_model(state, action, reward, task_tag, next_state)

    def _update_causal_model(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        task_tag: str,
        next_state: Dict[str, Any],
    ):
        """Integrate a transition into the causal discovery engine."""
        verifier = self.verifiers.get(task_tag, GroundingVerifier())

        # Causes = pre-conditions + action
        if self.current_state and self.current_state.active_predicates:
            causes = list(self.current_state.active_predicates)
        else:
            causes = verifier.get_active_predicates(state, context=task_tag)
        causes.append(action)

        # Effects = post-conditions + reward signal
        effects = verifier.get_active_predicates(next_state, context=task_tag)
        if reward > 0:
            effects.append("REWARD_POS")
        elif reward < 0:
            effects.append("REWARD_NEG")

        self.causal_discovery.observe(task_tag, causes, effects)

        # Periodically induce and merge causal graph
        if self.stats["episodes_recorded"] % 10 == 0:
            induced = self.causal_discovery.induce_graph(task_tag)
            active_graph = self.causal_graphs.get(task_tag)
            if active_graph is None:
                self.causal_graphs[task_tag] = induced
                self.causal_reasoners[task_tag] = CausalReasoner(induced)
                return

            new_count = 0
            for link in induced.all_links:
                existing = active_graph.forward.get(link.cause, [])
                match = next((l for e, l in existing if e == link.effect), None)
                if not match:
                    active_graph.add_link(link)
                    new_count += 1
                elif link.strength > match.strength:
                    match.strength = link.strength
            if new_count:
                logger.info(
                    "Causal discovery: integrated %d new links for %s",
                    new_count, task_tag,
                )

    # ------------------------------------------------------------------
    # Transfer & Analogy
    # ------------------------------------------------------------------

    def transfer(
        self,
        source_task: str,
        target_task: str,
        state: Dict[str, Any],
        active_predicates: List[str],
    ) -> Optional[str]:
        """Zero-shot knowledge transfer via analogy + global rules.

        Combines:
        1. Global rules (directly applicable across domains)
        2. Source-task rules transferred via analogical mapping
        3. Auto-discovery of cross-domain structural alignments via VSA HV
           similarity (IIT-style: Importance Inversion Transfer) — seeds the
           analogy engine from concept HVs so cold-start transfer succeeds.
        4. Rules from any other known domain
        """
        active_set = set(active_predicates)

        # 1. Global rules
        for rule in self.rule_learner.get_rules("global"):
            if rule.condition.issubset(active_set):
                return rule.consequence

        # 2. Source-task analogy
        source_rules = [
            (r.condition, r.consequence)
            for r in self.rule_learner.get_rules(source_task)
        ]
        if source_rules:
            result = self.analogy.zero_shot_action(
                state=state,
                known_domain=source_task,
                new_domain=target_task,
                learned_rules=source_rules,
                active_predicates=active_set,
            )
            if result:
                return result

            # 2b. IIT: Auto-discover predicate alignments via VSA HV similarity
            # when explicit grounding is missing. Build HV maps from concept_hvs
            # for predicates that appear in learned rules.
            source_preds = {p for r_cond, _ in source_rules for p in r_cond}
            target_preds_all = set(active_predicates)

            # Ensure all predicate strings have HVs in semantic memory
            import python.core.vsa.hypervec_shim as _hv_shim
            sem_hvs = self.semantic_memory.concept_hvs if hasattr(self.semantic_memory, "concept_hvs") else {}
            
            def _pred_hv(pred: str) -> Any:
                if pred in sem_hvs:
                    return sem_hvs[pred]
                # Stable HV from name hash (deterministic)
                v = _hv_shim.HyperVector(hash(pred.upper()) % (2**32))
                self.semantic_memory.add_concept(pred, {"auto": True}, hv_override=v)
                return v

            src_hvs = {p: _pred_hv(p) for p in source_preds}
            tgt_hvs = {p: _pred_hv(p) for p in target_preds_all}

            if src_hvs and tgt_hvs:
                new_mappings = self.analogy.auto_discover_abstractions(
                    domain_a=source_task,
                    domain_b=target_task,
                    concept_hvs_a=src_hvs,
                    concept_hvs_b=tgt_hvs,
                    similarity_threshold=0.48,  # Slightly relaxed for predicate names
                )
                if new_mappings:
                    # Retry zero-shot now that new abstractions exist
                    result = self.analogy.zero_shot_action(
                        state=state,
                        known_domain=source_task,
                        new_domain=target_task,
                        learned_rules=source_rules,
                        active_predicates=active_set,
                    )
                    if result:
                        return result

            # 2c. Structural fallback: if source rule fires on a predicate that
            # subsumes the target (e.g. ACTION_NORMAL_OPERATION spans both domains),
            # apply it directly when the action is shared.
            for cond, action in source_rules:
                if action in self._get_allowed_actions_safe(target_task):
                    # Check if ANY predicate in this rule's condition has a known
                    # abstract grounding in target domain (even partial match)
                    for pred in cond:
                        abstract = self.analogy.lift_to_abstract(pred, source_task)
                        if abstract:
                            grounded = self.analogy.ground_to_domain(abstract, target_task)
                            if grounded and grounded in active_set:
                                return action
                    # Last resort: if condition is empty-like (single pred rule)
                    # and the action is valid, fire it with a default predicate match
                    if len(cond) == 1:
                        (only_pred,) = cond
                        # Accept if any target predicate has name-prefix overlap ≥3 chars
                        op = only_pred.lower()
                        for tp in active_set:
                            shared = sum(1 for a, b in zip(op, tp.lower()) if a == b)
                            if shared >= 3 and action in self._get_allowed_actions_safe(target_task):
                                return action

            # 2d. Functional role transfer (IIT-inspired default action transfer):
            # When predicate alignment fully fails (no shared structure), transfer
            # the most-supported source action that is valid in the target domain.
            # This is the "distribution prior" strategy from cross-domain research:
            # carry over what the source agent does MOST OFTEN as a safe default.
            allowed_target = set(self._get_allowed_actions_safe(target_task))
            if allowed_target:
                # Rank source rules by how many conditions are ABSENT from active_set
                # (lower abs = closer to firing in target) and action is valid
                transferable = [
                    (action, len(cond))
                    for cond, action in source_rules
                    if action in allowed_target
                ]
                if transferable:
                    # Pick valid action with smallest condition set (most general rule)
                    transferable.sort(key=lambda x: x[1])
                    return transferable[0][0]

        # 3. Other known domains
        for domain in self.rule_learner.learned_rules:
            if domain in (source_task, target_task, "global"):
                continue
            domain_rules = [
                (r.condition, r.consequence)
                for r in self.rule_learner.get_rules(domain)
            ]
            if domain_rules:
                result = self.analogy.zero_shot_action(
                    state=state,
                    known_domain=domain,
                    new_domain=target_task,
                    learned_rules=domain_rules,
                    active_predicates=active_set,
                )
                if result:
                    return result
        return None

    def _get_allowed_actions_safe(self, task_tag: str) -> List[str]:
        """Get allowed actions for a task without raising exceptions."""
        try:
            if callable(getattr(self, "get_allowed_actions", None)):
                return self.get_allowed_actions(task_tag) or []
            verifier = self.verifiers.get(task_tag)
            if verifier and hasattr(verifier, "allowed_actions"):
                return verifier.allowed_actions
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Dialogue & Language
    # ------------------------------------------------------------------

    def process_dialogue(self, user_text: str, teach_mode: bool = False) -> str:
        """Process natural-language input.

        Parameters
        ----------
        user_text : str
            User's message.
        teach_mode : bool
            If True, parse as ``Subject Relation Object`` and store fact.

        Returns
        -------
        str
            Agent's response.
        """
        if teach_mode:
            words = user_text.split()
            if len(words) >= 3:
                subj, rel, obj = words[0], words[1], " ".join(words[2:])
                self.semantic_memory.add_concept(subj, {"source": "user"})
                self.semantic_memory.add_concept(obj, {"source": "user"})
                self.semantic_memory.add_relation(subj, rel, obj)
                return f"I have learned that {subj} {rel} {obj}."
            return (
                "I couldn't extract a fact. "
                "Please use 'Subject Relation Object' format."
            )
        return self.dialogue.process_turn(user_text)

    # ------------------------------------------------------------------
    # Explanation & Interpretability
    # ------------------------------------------------------------------

    def explain(self, query_type: str = "action") -> str:
        """Get explanation for the most recent decision."""
        if self.current_state.explanation:
            return self.current_state.explanation.details
        return "No current decision to explain."

    def why_not(self, rejected_action: str) -> str:
        """Explain why *rejected_action* was not taken."""
        return self.explainer.explain_rejection(
            rejected_action,
            self.current_state.chosen_action,
            {},
            self.current_state.task_tag,
            reason="",
        ).summary

    def counterfactual(self, alternative_action: str) -> str:
        """Answer 'What if I did *alternative_action* instead?'"""
        reasoner = self.causal_reasoners.get(self.current_state.task_tag)
        if not reasoner:
            return "No causal model for this task."
        result = reasoner.counterfactual(
            self.current_state.chosen_action,
            alternative_action,
            {},
            self.current_state.task_tag,
        )
        return result.explanation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def enable_neural_rule_scoring(self):
        """Instantiate RuleNeuralScorer and attach to engine."""
        from python.core.learning.rule_neural_scorer import RuleNeuralScorer
        self.rule_scorer = RuleNeuralScorer()

    def get_allowed_actions(self, task_tag: str) -> List[str]:
        """Return symbolic action names for a task.

        Override or extend this for custom domains.
        """
        return ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]

    def _get_exploration_action(
        self, state: Dict[str, Any], task_tag: str
    ) -> str:
        possible = self.get_allowed_actions(task_tag)
        probs = [1.0 / len(possible)] * len(possible)
        return self.curiosity.get_exploration_action(
            possible, probs, explore_rate=0.7
        )

    def _solve_math(self, text: str) -> Optional[Dict]:
        """Route math text through MathReasoner."""
        try:
            result = self.math_reasoner.solve_word_problem(text)
            if result and result.get("answer") is not None:
                return result
        except Exception:
            pass
        try:
            val = self.math_reasoner.solve_expression(text)
            if val is not None:
                return {"answer": val, "expression": text, "explanation": f"{text} = {val}"}
        except Exception:
            pass
        return None

    def get_belief_summary(self, topic_hv) -> Any:
        """Return agent's belief HV about a topic via SemanticMemory."""
        if hasattr(self.semantic_memory, 'concept_hvs') and self.semantic_memory.concept_hvs:
            hvs = list(self.semantic_memory.concept_hvs.values())[:3]
            if hvs:
                summary = hvs[0]
                for h in hvs[1:]:
                    summary = summary.bundle(h)
                return summary
        return topic_hv

    def _default_action(
        self, state: Dict[str, Any], task_tag: str
    ) -> str:
        """Fallback action when no module wins competition."""
        import random
        return random.choice(self.get_allowed_actions(task_tag))

    # ------------------------------------------------------------------
    # Gap 2: Memory-guided decision making
    # ------------------------------------------------------------------

    def _build_memory_coalition(
        self,
        situation_hv: hypervec_rs.HyperVector,
        task_tag: str,
        active_preds: List[str],
    ) -> Optional[Coalition]:
        """Recall similar past episodes and propose the action that worked best.

        This is case-based reasoning: "last time I was in a similar
        situation, action X gave reward Y."
        """
        similar = self.episodic_memory.recall_similar(situation_hv, task_tag, k=5)
        if not similar:
            return None

        self.stats["memory_recalls"] += 1

        # Score each past action by reward-weighted similarity
        action_scores: Dict[str, float] = {}
        for ep in similar:
            sim = situation_hv.similarity(ep.situation_hv)
            weighted = sim * (0.5 + ep.reward)  # Reward bonus
            action_scores[ep.action] = action_scores.get(ep.action, 0.0) + weighted

        if not action_scores:
            return None

        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]

        # Only propose if there's a clear memory signal
        if best_score < 0.3:
            return None

        # Normalize salience to 0-1 range
        salience = min(0.9, best_score / max(1.0, len(similar)))

        return Coalition(
            source="MEMORY",
            content=best_action,
            base_salience=salience,
            relevance=0.15,
            sender_confidence=min(0.9, len(similar) / 5.0),
        )

    # ------------------------------------------------------------------
    # Gap 4: Goal-directed planning
    # ------------------------------------------------------------------

    def set_mission_goal(self, goal_type: str, threshold: float,
                         goal_predicates: Optional[Set[str]] = None):
        """Set high-level mission goal.

        Parameters
        ----------
        goal_type : str
            Goal category (e.g. "reach", "collect", "avoid").
        threshold : float
            Numeric threshold for goal completion.
        goal_predicates : set of str, optional
            Symbolic predicates that define the goal state for the planner
            (e.g. ``{"AT_5_6", "HAS_FOOD"}``).
        """
        self.mission_goal = {"type": goal_type, "threshold": threshold}
        self._plan_goal = goal_predicates
        self._active_plan = None  # Reset any existing plan
        logger.info("Mission goal set: %s >= %s (predicates=%s)",
                     goal_type, threshold, goal_predicates)

    def _build_planner_coalition(
        self,
        active_preds: List[str],
        task_tag: str,
    ) -> Optional[Coalition]:
        """Submit a PLANNER coalition with the next planned action.

        Only activates when a mission goal with goal_predicates is set.
        Uses the causal graph to derive STRIPS operators and A* to plan.
        """
        if not self._plan_goal:
            return None

        current_state = set(active_preds)

        # Check if goal already satisfied
        if self._plan_goal.issubset(current_state):
            self._active_plan = None
            return None

        # If no active plan, or plan is stale, generate a new one
        if not self._active_plan:
            # Try to learn operators from causal graph if available
            graph = self.causal_graphs.get(task_tag)
            if graph and graph.all_links:
                self.planner.learn_operators_from_graph(graph, context=task_tag)

            plan = self.planner.plan(current_state, self._plan_goal, max_depth=15)
            if plan:
                self._active_plan = plan
                self.stats["plans_generated"] += 1
                logger.info("Planner: generated %d-step plan for %s",
                            len(plan), task_tag)
            else:
                return None

        # Pop the next step from the plan
        if self._active_plan:
            next_action = self._active_plan.pop(0)
            confidence = 0.7 if len(self._active_plan) < 5 else 0.5

            # V4: Validate plan via multi-step imagination
            plan_salience = 0.75
            if (self.current_state is not None and
                    self.current_state.situation_hv is not None and
                    self._active_plan):
                try:
                    preview_actions = [next_action] + list(self._active_plan[:4])
                    _, plan_safe = self.imagine_rollout(
                        initial_hv=self.current_state.situation_hv,
                        action_sequence=preview_actions,
                        task_tag=task_tag,
                    )
                    if not plan_safe:
                        # Plan is potentially unsafe — reduce salience by 50%
                        plan_salience *= 0.5
                        logger.warning("[PLANNER] Imagination detected unsafe plan; salience halved")
                except Exception:
                    pass

            return Coalition(
                source="PLANNER",
                content=next_action,
                base_salience=plan_salience,
                relevance=0.3,  # Goal-directed gets relevance boost
                sender_confidence=confidence,
            )

        return None

    # ------------------------------------------------------------------
    # Gap 5: Safety gate
    # ------------------------------------------------------------------

    def _safety_check(
        self,
        winner: Coalition,
        all_coalitions: List[Coalition],
        state: Dict[str, Any],
        task_tag: str,
    ) -> Coalition:
        """Validate the winning coalition through the SafetyGate.

        If the winning action is unsafe, try alternatives in activation
        order. If all are unsafe, fall back to a safe default.
        """
        # Check winner first
        if SafetyGate.is_safe(task_tag, state, winner.content):
            return winner

        # Winner is unsafe — try alternatives
        self.stats["safety_vetoes"] += 1
        logger.warning("SafetyGate: vetoed %s from %s", winner.content, winner.source)

        ranked = sorted(all_coalitions, key=lambda c: c.activation, reverse=True)
        for candidate in ranked:
            if candidate is winner:
                continue
            if SafetyGate.is_safe(task_tag, state, candidate.content):
                logger.info("SafetyGate: substituted %s from %s",
                            candidate.content, candidate.source)
                return candidate

        # All unsafe — return a safe default (stay in place)
        return Coalition(
            source="SAFETY_FALLBACK",
            content="ACTION_STAY",
            base_salience=0.1,
            sender_confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Gap 3: Sleep / Consolidation / Dreaming
    # ------------------------------------------------------------------

    def sleep(self, task_tag: Optional[str] = None, epochs: Optional[int] = None):
        """Offline consolidation cycle: replay → rules → semantic extraction.

        Implements the neuroscience-inspired sleep cycle:
        1. **Replay**: Sample past episodes and re-observe them to
           strengthen or discover rules (like hippocampal replay).
        2. **Semantic extraction**: Frequent predicate co-occurrences
           become new concepts/relations in semantic memory.
        3. **Plan refresh**: Re-learn planner operators from the
           updated causal graph.

        Parameters
        ----------
        task_tag : str, optional
            Consolidate a specific task (default: all tasks with episodes).
        epochs : int, optional
            Number of replay epochs (default: ``config.sleep_epochs``).
        """
        if not self.config.enable_sleep:
            return

        n_epochs = epochs or self.config.sleep_epochs
        batch_size = self.config.replay_batch_size

        # Determine which tasks to consolidate
        tasks = [task_tag] if task_tag else list(self.episodic_memory.recent.keys())

        for task in tasks:
            if not self.episodic_memory.recent.get(task):
                continue

            logger.info("[SLEEP] Consolidating task '%s' for %d epochs", task, n_epochs)

            for epoch in range(n_epochs):
                # 1. Replay: sample and re-observe episodes
                episodes = self.episodic_memory.sample(task, n=batch_size)
                for ep in episodes:
                    self.rule_learner.observe(
                        ep.state, ep.action, ep.reward, task, ep.outcome
                    )

                # 2. Induce rules from replayed evidence
                new_rules = self.rule_learner.induce_rules(task)
                if new_rules:
                    self.stats["rules_induced"] += len(new_rules)

            # 3. Semantic extraction: find frequent predicate pairs
            self._consolidate_semantic(task)

            # 4. Refresh planner operators from updated causal graph
            graph = self.causal_graphs.get(task)
            if graph and graph.all_links:
                self.planner.learn_operators_from_graph(graph, context=task)

        # V9: Step 4a — Prototype building across all consolidated tasks
        try:
            prototypes = self.semantic_memory.build_prototypes(min_members=2)
            if prototypes:
                self.stats["prototypes_built"] += len(prototypes)
                logger.info("[SLEEP] Built %d concept prototypes", len(prototypes))
        except Exception as _exc:
            logger.debug("[SLEEP] Prototype building skipped: %s", _exc)

        # V9: Step 4b — Transitive inference
        try:
            n_is_a = self.semantic_memory.infer_transitive("is_a", max_hops=3)
            n_causes = self.semantic_memory.infer_transitive("causes", max_hops=2)
            n_trans = n_is_a + n_causes
            if n_trans:
                self.stats["transitive_inferences"] += n_trans
                logger.info("[SLEEP] Transitive inference: %d new edges", n_trans)
        except Exception as _exc:
            logger.debug("[SLEEP] Transitive inference skipped: %s", _exc)

        # V9: Step 4c — Cross-task auto-abstraction discovery
        if len(tasks) >= 2:
            for i, ta in enumerate(tasks):
                for tb in tasks[i + 1:]:
                    hvs_a = self._collect_concept_hvs_for_task(ta)
                    hvs_b = self._collect_concept_hvs_for_task(tb)
                    if hvs_a and hvs_b:
                        try:
                            mappings = self.analogy.auto_discover_abstractions(
                                ta, tb, hvs_a, hvs_b
                            )
                            if mappings:
                                self.stats["auto_abstractions"] += len(mappings)
                                logger.info(
                                    "[SLEEP] Auto-abstractions %s↔%s: %d",
                                    ta, tb, len(mappings),
                                )
                        except Exception:
                            pass

        self.stats["sleep_cycles"] += 1
        logger.info("[SLEEP] Consolidation complete (%d tasks)", len(tasks))

        # V16: EWC consolidation
        if self._continual_learner is not None:
            _tasks = list(self._ewc_importance_buffer.keys())
            for _t in _tasks:
                self._compute_and_record_vsa_importance(_t)
                self._continual_learner.consolidate_task(_t)
            self.stats["tasks_consolidated"] = self.stats.get("tasks_consolidated", 0) + len(_tasks)

        # V3: Homeostasis regulation after consolidation
        if self.config.enable_homeostasis and self.homeostasis is not None:
            actions = self.homeostasis.regulate(self.semantic_memory)
            if actions:
                logger.info("[SLEEP] Homeostasis: %s", actions)
            # Auto-categorization
            if self.config.enable_auto_categories:
                cat_actions = self.homeostasis._auto_categorize(self.semantic_memory)
                if cat_actions:
                    logger.info("[SLEEP] Auto-categories: %s", cat_actions)

        # V3: Stigmergy evaporation during sleep
        if self.config.enable_stigmergy:
            if hasattr(self.semantic_memory, 'evaporate_stigmergy'):
                self.semantic_memory.evaporate_stigmergy()

        # V18: Rebuild LSH index after consolidation to flush stale bucket entries
        # left by implicit deque evictions.  Stale entries cause false-positive recalls
        # (episodes that were evicted still match queries), so a full rebuild here
        # ensures recall accuracy after every sleep cycle.
        if hasattr(self.episodic_memory, '_rebuild_lsh_index'):
            try:
                self.episodic_memory._rebuild_lsh_index()
                logger.debug("[SLEEP] LSH index rebuilt after consolidation")
            except Exception as _exc:
                logger.debug("[SLEEP] LSH rebuild skipped: %s", _exc)

        # V9: Step 5 — Drift detection and rule pruning
        self._detect_rule_drift(tasks)
        if self.homeostasis is not None:
            try:
                pruned = self.homeostasis.prune_unused_rules(
                    self.rule_learner, max_idle_episodes=200
                )
                if pruned:
                    logger.info("[SLEEP] Pruned %d unused rules", pruned)
            except Exception:
                pass

    def _compute_and_record_vsa_importance(self, task_tag: str) -> None:
        """Compute VSA-based Fisher importance for EWC from buffered situation HVs."""
        if self._continual_learner is None:
            return
        buf = self._ewc_importance_buffer.get(task_tag, [])
        if not buf:
            return
        concept_hvs = self.semantic_memory.concept_hvs
        if not concept_hvs:
            return
        params: Dict[str, Any] = {}
        gradients: Dict[str, Any] = {}
        for concept_name, chv in concept_hvs.items():
            sims = []
            for shv in buf:
                try:
                    sims.append(float(shv.similarity(chv)))
                except Exception:
                    sims.append(0.0)
            if sims:
                avg_sim = float(np.mean(sims))
            else:
                avg_sim = 0.0
            params[concept_name] = np.array([avg_sim])
            gradients[concept_name] = np.array([avg_sim ** 2])
        try:
            self._continual_learner.compute_importance(task_tag, params, gradients)
        except Exception as _exc:
            logger.warning("EWC importance computation failed: %s", _exc)

        # V27: Societal-Guided EWC boost
        if getattr(self.config, "enable_societal_ewc", False):
            _soc_mgr = getattr(self, "_societal_manager", None)
            if _soc_mgr is not None:
                try:
                    from python.core.learning.societal_ewc import SocietalEWCCombiner
                    _combiner = SocietalEWCCombiner(
                        centrality_weight=getattr(
                            self.config, "societal_ewc_centrality_weight", 0.5
                        ),
                        degree_weight=getattr(
                            self.config, "societal_ewc_degree_weight", 0.4
                        ),
                        activation_weight=getattr(
                            self.config, "societal_ewc_activation_weight", 0.4
                        ),
                        cluster_weight=getattr(
                            self.config, "societal_ewc_cluster_weight", 0.2
                        ),
                    )
                    _combiner.apply_societal_boost(
                        self._continual_learner, task_tag, _soc_mgr
                    )
                    logger.debug(
                        "[V27] SoCL boost applied for task %r", task_tag
                    )
                except Exception as _exc:
                    logger.warning("SoCL boost failed: %s", _exc)

    def _detect_rule_drift(self, tasks: List[str]) -> None:
        """Mark rules as drifting when recent confidence drops below 50% of older confidence (V9)."""
        for task in tasks:
            for rule in self.rule_learner.learned_rules.get(task, []):
                history = getattr(rule, "confidence_history", [])
                if len(history) < 4:
                    continue
                mid = len(history) // 2
                older_avg = sum(history[:mid]) / mid
                recent_avg = sum(history[mid:]) / (len(history) - mid)
                if older_avg > 0 and recent_avg < 0.5 * older_avg:
                    rule.source = f"drifting:{rule.source}"
                    logger.info(
                        "[SLEEP] Rule drift detected: %s→%s in '%s'",
                        set(rule.condition), rule.consequence, task,
                    )

    # ------------------------------------------------------------------
    # V4: Multi-step imagination
    # ------------------------------------------------------------------

    def imagine_rollout(
        self,
        initial_hv,
        action_sequence: List[str],
        task_tag: str,
        gamma: float = 0.9,
        danger_threshold: float = 0.75,
        get_action_hv_fn=None,
        danger_vectors=None,
    ) -> Tuple[float, bool]:
        """
        N-step forward simulation. Returns (total_discounted_reward, is_plan_safe).

        Aborts early if any predicted state exceeds danger_threshold similarity
        to any registered danger vector.
        """
        total_reward = 0.0
        current_hv = initial_hv
        is_safe = True

        _danger_vectors = (
            danger_vectors
            if danger_vectors is not None
            else getattr(self.global_workspace, '_danger_vectors', [])
        )

        for step, action in enumerate(action_sequence):
            # Get action HV
            if get_action_hv_fn is not None:
                action_hv = get_action_hv_fn(action)
            else:
                action_hv = hypervec_rs.HyperVector(hash(action) % (2**32))

            # Predict next state via active inference world model
            try:
                next_hv = self.active_inference.predict_next_state(current_hv, action_hv)
            except Exception:
                next_hv = current_hv.xor(action_hv)

            # Estimate reward from world model
            try:
                pred_reward = float(self.active_inference.predict_reward(current_hv, action_hv))
            except Exception:
                pred_reward = 0.0

            total_reward += (gamma ** step) * pred_reward

            # Safety check against danger vectors
            for dv in _danger_vectors:
                try:
                    sim = float(next_hv.similarity(dv))
                    if sim > danger_threshold:
                        is_safe = False
                        break
                except Exception:
                    pass
            if not is_safe:
                break

            current_hv = next_hv

        return total_reward, is_safe

    def _consolidate_semantic(self, task_tag: str):
        """Extract repeating patterns from episodes into semantic memory.

        Scans recent episodes for frequently co-occurring predicates
        and promotes them to semantic concepts and relations.
        """
        from collections import Counter

        episodes = list(self.episodic_memory.recent.get(task_tag, []))
        if len(episodes) < 10:
            return

        # Count action→outcome co-occurrences
        action_outcome_counts: Counter = Counter()
        # Count predicate pair co-occurrences within episode states
        predicate_pairs: Counter = Counter()

        for ep in episodes:
            action_outcome_counts[(ep.action, ep.outcome)] += 1

            # Extract predicates from state sketch
            sketch = LiveEpisode._extract_sketch(ep.state, task_tag)
            keys = sorted(sketch.keys())
            for i, k1 in enumerate(keys):
                for k2 in keys[i + 1:]:
                    predicate_pairs[(k1, k2)] += 1

        # Promote frequent action→outcome pairs as causal concepts
        min_freq = max(3, len(episodes) // 10)
        for (action, outcome), count in action_outcome_counts.items():
            if count >= min_freq and outcome != "neutral":
                # Add to semantic memory as a relation
                if action not in self.semantic_memory.concept_graph:
                    self.semantic_memory.add_concept(action, {"type": "action"})
                if outcome not in self.semantic_memory.concept_graph:
                    self.semantic_memory.add_concept(outcome, {"type": "outcome"})
                self.semantic_memory.add_relation(
                    action, "causes", outcome, timestamp=time.time()
                )

    def get_stats(self) -> Dict[str, Any]:
        """Return cognitive engine statistics."""
        return {
            **self.stats,
            "memory_stats": self.episodic_memory.get_statistics(
                self.current_state.task_tag
            ),
            "curiosity_stats": self.curiosity.get_statistics(
                self.current_state.task_tag
            ),
            "rules_per_task": {
                task: len(rules)
                for task, rules in self.rule_learner.learned_rules.items()
            },
        }

    def get_workspace_telemetry(self) -> Dict[str, Any]:
        """Get telemetry for monitoring/dashboard."""
        active_modules = []
        if any(len(r) > 0 for r in self.rule_learner.learned_rules.values()):
            active_modules.append("RULES")
        if any(len(g.all_links) > 0 for g in self.causal_graphs.values()):
            active_modules.append("CAUSAL")
        active_modules.extend(["PLANNER", "GLOBAL", "EPISODIC", "SELF_MODEL"])

        coalitions_data = []
        if (
            hasattr(self.global_workspace, "latest_coalitions")
            and self.global_workspace.latest_coalitions
        ):
            for c in self.global_workspace.latest_coalitions:
                coalitions_data.append({
                    "source": c.source,
                    "content": str(c.content),
                    "activation": float(c.activation),
                    "salience": float(c.base_salience),
                    "confidence": float(c.sender_confidence),
                })

        return {
            "winner": (
                self.trace_history[-1]["winner"] if self.trace_history else "NONE"
            ),
            "active_modules": active_modules,
            "competition": coalitions_data,
        }

    def export_traces(self, filename: str):
        """Export decision traces to JSON."""
        with open(filename, "w") as f:
            json.dump(self.trace_history, f, indent=2)
        logger.info("Traces exported to %s", filename)

    def register_broadcaster(self, callback):
        """Register callback for real-time event broadcasting."""
        self.msg_broadcaster = callback

    def seed_domain(self, yaml_path: str) -> int:
        """Seed a domain from a YAML bootstrap kit. Returns number of rules seeded.

        Parameters
        ----------
        yaml_path : str
            Path to a domain kit YAML file (see nsck/python/core/bootstrap/domain_kits/).

        Returns
        -------
        int
            Number of rules seeded.
        """
        from python.core.bootstrap.knowledge_seeder import KnowledgeSeeder
        seeder = KnowledgeSeeder()
        return seeder.seed_from_yaml(yaml_path, self)

    def get_causal_telemetry(self, task_tag: str) -> Dict[str, Any]:
        """Get causal graph details for monitoring."""
        graph = self.causal_graphs.get(task_tag)
        if not graph:
            return {"link_count": 0, "links": []}
        sorted_links = sorted(
            graph.all_links, key=lambda x: x.strength, reverse=True
        )
        return {
            "link_count": len(graph.all_links),
            "links": [
                {
                    "cause": link.cause,
                    "effect": link.effect,
                    "strength": float(link.strength),
                }
                for link in sorted_links[:20]
            ],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_cognitive_engine(
    persistence_path: Optional[str] = None,
) -> CognitiveEngine:
    """Create a fully configured cognitive engine."""
    return CognitiveEngine(
        config=NSCKConfig(),
        persistence_path=persistence_path,
    )
