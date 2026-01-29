"""
NSCK Cognitive Engine
Unified integration of all cognitive modules for the NSCK system.

This is the main entry point that orchestrates:
- Perception (SNN + frame processing)
- Metacognition (dual inference + confidence)
- Rule Learning (frequency-based ILP)
- Episodic Memory (VSA + LSH)
- Curiosity (novelty detection)
- Causal Reasoning (forward/backward chains)
- Explanation Generation
- Cross-task Transfer (analogical reasoning)
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import hypervec_rs

# Import all cognitive modules
from .config import NSCKConfig
from .perception import PerceptionEngine, calculate_entropy
from .grounding_verifier import GroundingVerifier, create_snake_verifier, create_pong_verifier
from .rule_learner import RuleLearner
from .episodic_memory import EpisodicMemory, LiveEpisode
from .curiosity import CuriosityModule
from .causal_reasoning import CausalReasoner, CausalGraph, create_snake_causal_graph, create_pong_causal_graph
from .explanation import ExplanationGenerator, Explanation
from .semantic_coherence import SemanticCoherence
from .analogy import AnalogyEngine
from .persistence import BrainStore


@dataclass
class CognitiveState:
    """Current state of the cognitive system."""
    task_tag: str
    situation_hv: Optional[hypervec_rs.HyperVector] = None
    active_predicates: List[str] = field(default_factory=list)
    chosen_action: str = "ACTION_STAY"
    confidence: float = 0.5
    exploration_mode: bool = False
    explanation: Optional[Explanation] = None
    
    # Trace for debugging
    trace: Dict[str, Any] = field(default_factory=dict)


class CognitiveEngine:
    """
    Unified cognitive engine integrating all NSCK modules.
    
    Provides a single interface for:
    - decide(state) -> action with explanation
    - learn(state, action, reward) -> update knowledge
    - transfer(source_task, target_task) -> adapt knowledge
    - explain() -> natural language explanation
    """
    
    def __init__(
        self,
        config: Optional[NSCKConfig] = None,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize cognitive engine with all modules.
        
        Args:
            config: Configuration object
            persistence_path: Path for SQLite database
        """
        self.config = config or NSCKConfig()
        
        # Initialize persistence (optional)
        self.store = BrainStore(persistence_path) if persistence_path else None
        
        # Initialize verifiers for each task
        self.verifiers: Dict[str, GroundingVerifier] = {
            "snake": create_snake_verifier(),
            "pong": create_pong_verifier(),
        }
        
        # Initialize cognitive modules
        self.rule_learner = RuleLearner(
            verifier=self.verifiers.get("snake", GroundingVerifier()),
            store=self.store,
            min_support=self.config.min_rule_support,
            min_success_rate=self.config.min_success_rate
        )
        
        self.episodic_memory = EpisodicMemory(
            store=self.store,
            recent_capacity=self.config.memory_capacity
        )
        
        self.curiosity = CuriosityModule(
            novelty_threshold=self.config.novelty_threshold
        )
        
        # Causal graphs per task
        self.causal_graphs: Dict[str, CausalGraph] = {
            "snake": create_snake_causal_graph(),
            "pong": create_pong_causal_graph(),
        }
        
        self.causal_reasoners: Dict[str, CausalReasoner] = {}
        for task, graph in self.causal_graphs.items():
            self.causal_reasoners[task] = CausalReasoner(graph)
        
        self.explainer = ExplanationGenerator()
        self.coherence = SemanticCoherence()
        self.analogy = AnalogyEngine()
        
        # Current cognitive state
        self.current_state = CognitiveState(task_tag="snake")
        
        # Stats
        self.stats = {
            "decisions": 0,
            "explorations": 0,
            "rules_induced": 0,
            "episodes_recorded": 0,
        }
    
    def decide(
        self,
        state: Dict[str, Any],
        task_tag: str,
        metacognition_result: Optional[Dict] = None
    ) -> CognitiveState:
        """
        Make a decision given current state.
        
        Args:
            state: Game state dict
            task_tag: Which task (snake, pong, maze)
            metacognition_result: Optional result from metacognition layer
            
        Returns:
            CognitiveState with decision and explanation
        """
        self.stats["decisions"] += 1
        
        # 1. Get verifier for this task
        verifier = self.verifiers.get(task_tag, GroundingVerifier())
        
        # 2. Extract active predicates
        active_preds = verifier.get_active_predicates(state, context=task_tag)
        
        # 3. Create situation hypervector
        situation_hv = self.episodic_memory.create_situation_hv(
            state, task_tag, active_preds
        )
        
        # 4. Check curiosity for exploration
        confidence = metacognition_result.get("confidence", 0.5) if metacognition_result else 0.5
        explore_decision = self.curiosity.should_explore(
            situation_hv, task_tag, confidence
        )
        
        # 5. Get action
        if explore_decision.should_explore:
            self.stats["explorations"] += 1
            action = self._get_exploration_action(state, task_tag, active_preds)
            trace = {"mode": "exploration", "reason": explore_decision.reason}
        elif metacognition_result and metacognition_result.get("action"):
            action = metacognition_result["action"]
            trace = {"mode": "metacognition", "layer": metacognition_result.get("layer_used")}
        else:
            # Use rule learner
            applicable = self.rule_learner.get_applicable_rules(state, task_tag)
            if applicable:
                rule, score = applicable[0]
                action = rule.consequence
                trace = {"mode": "learned_rule", "rule_score": score}
            else:
                action = self._default_action(state, task_tag)
                trace = {"mode": "default"}
        
        # 6. Generate explanation
        trace["confidence"] = confidence
        trace["veto"] = metacognition_result.get("veto", False) if metacognition_result else False
        explanation = self.explainer.explain_action(action, state, task_tag, trace)
        
        # 7. Check coherence
        pred_set = set(active_preds)
        coherence_check = self.coherence.check_predicates(pred_set)
        if not coherence_check.is_coherent:
            trace["coherence_issues"] = len(coherence_check.contradictions)
        
        # 8. Update curiosity
        self.curiosity.record_visit(situation_hv, task_tag)
        
        # Build cognitive state
        self.current_state = CognitiveState(
            task_tag=task_tag,
            situation_hv=situation_hv,
            active_predicates=active_preds,
            chosen_action=action,
            confidence=confidence,
            exploration_mode=explore_decision.should_explore,
            explanation=explanation,
            trace=trace
        )
        
        return self.current_state
    
    def _get_exploration_action(
        self,
        state: Dict[str, Any],
        task_tag: str,
        active_preds: List[str]
    ) -> str:
        """Get an exploratory action."""
        # Get actions with low visit counts
        possible_actions = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
        
        # Use curiosity module to select
        probs = [0.25, 0.25, 0.25, 0.25]
        return self.curiosity.get_exploration_action(possible_actions, probs, explore_rate=0.7)
    
    def _default_action(self, state: Dict[str, Any], task_tag: str) -> str:
        """Get default action when no rules apply."""
        if task_tag == "snake":
            # Move toward food
            head = state.get("head", (5, 5))
            food = state.get("food", (5, 5))
            
            dx = food[0] - head[0]
            dy = food[1] - head[1]
            
            if abs(dx) > abs(dy):
                return "ACTION_RIGHT" if dx > 0 else "ACTION_LEFT"
            else:
                return "ACTION_UP" if dy < 0 else "ACTION_DOWN"
        
        elif task_tag == "pong":
            ball_y = state.get("ball_y", 15)
            paddle_y = state.get("p1_y", 10)
            paddle_center = paddle_y + 3
            
            if ball_y < paddle_center:
                return "ACTION_UP"
            elif ball_y > paddle_center:
                return "ACTION_DOWN"
            return "ACTION_STAY"
        
        return "ACTION_STAY"
    
    def learn(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "neutral"
    ):
        """
        Learn from experience.
        
        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
            task_tag: Which task
            outcome: Outcome label
        """
        # 1. Record observation for rule learning
        self.rule_learner.observe(state, action, reward, task_tag, outcome)
        
        # 2. Record episode in memory
        if self.current_state.situation_hv:
            episode = LiveEpisode(
                timestamp=time.time(),
                task_tag=task_tag,
                situation_hv=self.current_state.situation_hv,
                state=state,
                action=action,
                outcome=outcome,
                reward=reward
            )
            self.episodic_memory.record(episode)
            self.stats["episodes_recorded"] += 1
        
        # 3. Update curiosity
        self.curiosity.record_outcome(task_tag, reward > 0)
        
        # 4. Periodically induce rules
        if self.stats["episodes_recorded"] % 50 == 0:
            new_rules = self.rule_learner.induce_rules(task_tag)
            self.stats["rules_induced"] += len(new_rules)
    
    def transfer(
        self,
        source_task: str,
        target_task: str,
        state: Dict[str, Any],
        active_predicates: List[str]
    ) -> Optional[str]:
        """
        Transfer knowledge from source to target task (zero-shot).
        
        Args:
            source_task: Task with learned knowledge
            target_task: New task to apply knowledge
            state: Current state in target task
            active_predicates: Active predicates in target task
            
        Returns:
            Recommended action or None
        """
        # Get rules from source task
        source_rules = [
            (rule.condition, rule.consequence)
            for rule in self.rule_learner.get_rules(source_task)
        ]
        
        if not source_rules:
            return None
        
        # Use analogy engine for transfer
        return self.analogy.zero_shot_action(
            state=state,
            known_domain=source_task,
            new_domain=target_task,
            learned_rules=source_rules,
            active_predicates=set(active_predicates)
        )
    
    def explain(self, query_type: str = "action") -> str:
        """
        Get explanation for current decision.
        
        Args:
            query_type: Type of explanation (action, state, goal)
            
        Returns:
            Human-readable explanation
        """
        if self.current_state.explanation:
            return self.current_state.explanation.details
        return "No current decision to explain."
    
    def why_not(self, rejected_action: str) -> str:
        """Explain why an action was not taken."""
        return self.explainer.explain_rejection(
            rejected_action,
            self.current_state.chosen_action,
            {},
            self.current_state.task_tag,
            reason=""
        ).summary
    
    def counterfactual(self, alternative_action: str) -> str:
        """Answer 'What if I did X instead?'"""
        reasoner = self.causal_reasoners.get(self.current_state.task_tag)
        if not reasoner:
            return "No causal model for this task."
        
        result = reasoner.counterfactual(
            self.current_state.chosen_action,
            alternative_action,
            {},  # Would need actual state
            self.current_state.task_tag
        )
        return result.explanation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cognitive engine statistics."""
        return {
            **self.stats,
            "memory_stats": self.episodic_memory.get_statistics(self.current_state.task_tag),
            "curiosity_stats": self.curiosity.get_statistics(self.current_state.task_tag),
            "rules_per_task": {
                task: len(rules) 
                for task, rules in self.rule_learner.learned_rules.items()
            }
        }
    
    def register_task(self, task_tag: str, verifier: GroundingVerifier, causal_graph: Optional[CausalGraph] = None):
        """Register a new task/game with its verifier."""
        self.verifiers[task_tag] = verifier
        if causal_graph:
            self.causal_graphs[task_tag] = causal_graph
            self.causal_reasoners[task_tag] = CausalReasoner(causal_graph)


# Factory function for easy creation
def create_cognitive_engine(persistence_path: Optional[str] = None) -> CognitiveEngine:
    """Create a fully configured cognitive engine."""
    return CognitiveEngine(
        config=NSCKConfig(),
        persistence_path=persistence_path
    )
