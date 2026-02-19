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
from typing import Dict, List, Optional, Any, Tuple, Set

import python.core.vsa.hypervec_shim as hypervec_rs

# --- Core imports (always available) ---
from python.core.integration.config import NSCKConfig
from python.core.integration.explanation import ExplanationGenerator, Explanation
from python.core.perception.grounding_verifier import GroundingVerifier
from python.core.reasoning.rule_learner import RuleLearner
from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode
from python.core.memory.semantic_memory import SemanticMemory
from python.core.learning.curiosity import CuriosityModule
from python.core.reasoning.analogy import AnalogyEngine
from python.core.reasoning.planner import STRIPSPlanner
from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition
from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner, CausalDiscovery
from python.core.cognitive.self_model import SelfModel
from python.core.cognitive.theory_of_mind import TheoryOfMind
from python.core.cognitive.metacognition import SafetyGate
from python.core.integration.persistence import BrainStore
from python.core.integration.brain_fusion import BrainFusion, TaskBrain
from python.core.language.universal_input import UniversalInput
from python.core.language.language_module import LanguageModule
from python.core.language.dialogue_manager import DialogueManager

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
        self.curiosity = CuriosityModule(
            novelty_threshold=self.config.novelty_threshold,
        )
        self.self_model = SelfModel()
        self.theory_of_mind = TheoryOfMind()
        self.global_workspace = GlobalWorkspace()
        self.causal_discovery = CausalDiscovery()
        self.planner = STRIPSPlanner()
        self.analogy = AnalogyEngine(load_defaults=True)
        self.explainer = ExplanationGenerator()
        self.fusion = BrainFusion()

        # --- Language ---
        self.universal_input = UniversalInput()
        self.language = LanguageModule()
        self.dialogue = DialogueManager(self, self.language)

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
        }
        self.current_state = CognitiveState(task_tag="unknown")
        self.msg_broadcaster = None

        # --- Gap 1: Wire GWT broadcast subscribers ---
        self._register_gwt_subscribers()

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
        """
        if verifier:
            self.verifiers[task_tag] = verifier
        if causal_graph is None:
            causal_graph = CausalGraph()
        self.causal_graphs[task_tag] = causal_graph
        self.causal_reasoners[task_tag] = CausalReasoner(causal_graph)
        try:
            self.task_brains[task_tag] = TaskBrain(task_tag)
        except Exception:
            pass
        logger.info("Registered task '%s'", task_tag)

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

    def decide(
        self,
        state: Dict[str, Any],
        task_tag: str,
        metacognition_result: Optional[Dict] = None,
    ) -> CognitiveState:
        """Run one cognitive cycle and return a decision.

        Parameters
        ----------
        state : dict
            Raw state from the environment or data source.
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

        # 1. Symbol grounding
        verifier = self.verifiers.get(task_tag, GroundingVerifier())
        active_preds = verifier.get_active_predicates(state, context=task_tag)

        # 2. Situation hypervector
        situation_hv = self.episodic_memory.create_situation_hv(
            state, task_tag, active_preds
        )

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

        # B. Rule-based proposal
        applicable = self.rule_learner.get_applicable_rules(active_preds, task_tag)
        if applicable:
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
            explore_act = self._get_exploration_action(state, task_tag)
            salience = 0.6 + (0.2 if "stagnant" in explore_decision.reason else 0.0)
            coalitions.append(Coalition(
                source="EXPLORATION",
                content=explore_act,
                base_salience=salience,
                relevance=0.0,
                sender_confidence=0.5,
            ))
        
        # C2. Q-LEARNING proposal (reward-based policy)
        state_key = self._get_state_key(state, task_tag)
        available_actions = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_STAY"]
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

        # D. [Gap 2] MEMORY coalition — episodic recall for case-based reasoning
        memory_coalition = self._build_memory_coalition(
            situation_hv, task_tag, active_preds
        )
        if memory_coalition:
            coalitions.append(memory_coalition)

        # E. [Gap 4] PLANNER coalition — goal-directed multi-step planning
        planner_coalition = self._build_planner_coalition(
            active_preds, task_tag
        )
        if planner_coalition:
            coalitions.append(planner_coalition)

        # 5. GWT competition
        winner_coalition = self.global_workspace.compete(coalitions)

        # 5b. [Gap 5] Safety gate veto — check winning action before committing
        if winner_coalition:
            winner_coalition = self._safety_check(
                winner_coalition, coalitions, state, task_tag
            )

        # 6. Determine final action
        trace: Dict[str, Any] = {"proposals": len(coalitions)}

        if winner_coalition:
            action = winner_coalition.content
            winner_name = winner_coalition.source
            trace["mode"] = winner_name
            trace["winner"] = winner_name
            trace["reason"] = (
                f"Winner: {winner_name} "
                f"(Activation: {winner_coalition.activation:.2f})"
            )
            self.self_model.update_confidence(task_tag, confidence)
        else:
            action = self._default_action(state, task_tag)
            winner_name = "DEFAULT"
            trace["mode"] = "default"
            trace["winner"] = "DEFAULT"

        # 7. Generate explanation
        trace["confidence"] = confidence
        explanation = self.explainer.explain_action(action, state, task_tag, trace)

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
        )
        
        # Track state-action for Q-learning updates
        state_key = self._get_state_key(state, task_tag)
        self.last_state_action = (state_key, action)
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
        
        return self.current_state

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
                available_actions = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_STAY"]
                
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
            
            # Decay epsilon (reduce exploration over time)
            self.epsilon = max(0.1, self.epsilon * 0.99)
        
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
        # 1. Rule learning observation
        self.rule_learner.observe(state, action, reward, task_tag, outcome)

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
        3. Rules from any other known domain
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
            return Coalition(
                source="PLANNER",
                content=next_action,
                base_salience=0.75,
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

        self.stats["sleep_cycles"] += 1
        logger.info("[SLEEP] Consolidation complete (%d tasks)", len(tasks))

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
