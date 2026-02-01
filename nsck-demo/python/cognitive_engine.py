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
import hashlib
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import hypervec_shim as hypervec_rs

# Import all cognitive modules
from config import NSCKConfig
from perception import PerceptionEngine, calculate_entropy
from grounding_verifier import GroundingVerifier, create_snake_verifier, create_pong_verifier, create_maze_verifier
from rule_learner import RuleLearner
from episodic_memory import EpisodicMemory, LiveEpisode
from curiosity import CuriosityModule
from explanation import ExplanationGenerator, Explanation
from semantic_coherence import SemanticCoherence
from analogy import AnalogyEngine
from brain_fusion import BrainFusion, TaskBrain # [AGI] Phase 2: Logic Integration
from planner import STRIPSPlanner 
from spatial_reasoning import GridPlanner # [AGI] Phase 2.3: Spatial Reasoning
from persistence import BrainStore
from global_workspace import GlobalWorkspace # [AGI] Phase 3.1: Global Workspace
from self_model import SelfModel # [AGI] Phase 3.2: Self-Model
from causal_reasoning import (
    CausalGraph, CausalReasoner, create_snake_causal_graph, 
    create_pong_causal_graph, create_maze_causal_graph, CausalDiscovery # [AGI] Phase 4.1
)


@dataclass
class Proposal:
    """A proposal for the Global Workspace."""
    module: str
    action: str
    content: str
    salience: float = 0.0

@dataclass
class CognitiveState:
    """Current state of the cognitive system."""
    task_tag: str
    situation_hv: Optional[hypervec_rs.HyperVector] = None
    active_predicates: List[str] = field(default_factory=list)
    chosen_action: str = "ACTION_STAY"
    confidence: float = 0.5
    self_confidence: float = 0.5 # [AGI] Phase 3.3
    exploration_mode: bool = False
    explanation: Optional[Explanation] = None
    imagined_reward: float = 0.0 # [AGI] Phase 5.2
    
    # Trace for debugging
    trace: Dict[str, Any] = field(default_factory=dict)


class CognitiveEngine:
    """
    Unified cognitive engine integrating all NSCK modules.
    
    Provides a single interface for:
    - decide(state) -> action with explanation
    - learn(state, action, reward) -> update知识
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
            "maze": create_maze_verifier(),
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
        
        # [AGI] Phase 3.2: Self-Model
        self.self_model = SelfModel()
        
        # [AGI] Phase 3.1: Global Workspace
        self.global_workspace = GlobalWorkspace()
        
        # [AGI] Phase 4.1: Causal Discovery
        self.causal_discovery = CausalDiscovery()
        # [AGI] Phase 4.4: Theory Formation
        try:
            from causal_reasoning import TheoryModule
            self.theory_module = TheoryModule()
        except ImportError:
            self.theory_module = None
        
        # [AGI] Phase 5.1: World Model
        try:
            from world_model import WorldModel
            self.world_model = WorldModel()
        except ImportError:
            self.world_model = None
        
        # Causal graphs per task
        # [VERIFICATION] Tabula Rasa Mode: Start with EMPTY graphs to prove learning.
        # We disable the hardcoded 'instincts' to force discovery.
        self.causal_graphs: Dict[str, CausalGraph] = {
            "snake": CausalGraph(), # create_snake_causal_graph(),
            "pong": CausalGraph(),  # create_pong_causal_graph(),
            "maze": CausalGraph(),  # create_maze_causal_graph(),
        }
        
        self.causal_reasoners: Dict[str, CausalReasoner] = {}
        for task, graph in self.causal_graphs.items():
            self.causal_reasoners[task] = CausalReasoner(graph)
        
        self.explainer = ExplanationGenerator()
        self.coherence = SemanticCoherence()
        self.analogy = AnalogyEngine()
        
        # [AGI] Phase 2: Brain Fusion for Chains
        self.fusion = BrainFusion() 
        try:
            from brain_fusion import TaskBrain
            self.task_brains = {
                "snake": TaskBrain("snake"),
                "pong": TaskBrain("pong"),
                "maze": TaskBrain("maze")
            }
        except ImportError:
            self.task_brains = {}

        # [AGI] Phase 2.3: Planning
        self.planner = STRIPSPlanner()
        self.grid_planner = GridPlanner()

        # [AGI] Phase 7: Mission Control state
        self.mission_goal = {"type": "default", "threshold": 0}
        self.trace_history = []
        self.max_trace_history = 1000
        
        # Stats
        self.stats = {
            "decisions": 0,
            "episodes_recorded": 0,
            "rules_induced": 0,
            "explorations": 0
        }
        
        self.current_state = CognitiveState(task_tag="unknown")

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
        
        # Support sampling from episodic memory
        num_samples = 5 
        sampled_episodes = self.episodic_memory.sample(task_tag, num_samples)
        
        # 3. Create situation hypervector
        situation_hv = self.episodic_memory.create_situation_hv(
            state, task_tag, active_preds
        )
        
        # 4. Check curiosity for exploration
        confidence = metacognition_result.get("confidence", 0.5) if metacognition_result else 0.5
        explore_decision = self.curiosity.should_explore(
            situation_hv, task_tag, confidence
        )
        
        # [AGI] Phase 3.1: Global Workspace Competition
        # We broadcast proposals from different modules to the Global Workspace
        # Format: Dict[module_name, Tuple[Content, Salience]]
        proposals: Dict[str, Tuple[Proposal, float]] = {}
        
        # A. SNN Proposal (Fast System)
        if metacognition_result and metacognition_result.get("action"):
            salience = confidence
            p = Proposal(
                module="SNN",
                action=metacognition_result["action"],
                salience=salience,
                content="Pattern match from visual input"
            )
            proposals["SNN"] = (p, salience)
            
        # B. Rule Proposal (Symbolic System)
        applicable = self.rule_learner.get_applicable_rules(state, task_tag)
        if applicable:
            rule, score = applicable[0]
            p = Proposal(
                module="RULES",
                action=rule.consequence,
                salience=score,
                content=f"Rule: {rule}"
            )
            proposals["RULES"] = (p, score)
            
        # C. Exploration Proposal (Curiosity)
        if explore_decision.should_explore:
            explore_act = self._get_exploration_action(state, task_tag, active_preds)
            salience = 0.6 + (0.2 if "stagnant" in explore_decision.reason else 0.0)
            p = Proposal(
                module="EXPLORATION",
                action=explore_act,
                salience=salience,
                content=explore_decision.reason
            )
            proposals["EXPLORATION"] = (p, salience)
            
        # D. Planner Proposal (Goal-Directed)
        # Check if we have a plan active or need one
        plan_action = None
        if task_tag == "snake" or task_tag == "maze":
             plan_action = self.grid_planner.get_next_action(state, task_tag)
        
        if plan_action:
             salience = 0.85
             p = Proposal(
                module="PLANNER",
                action=plan_action,
                salience=salience,
                content="Strategic spatial plan"
             )
             proposals["PLANNER"] = (p, salience)

        # Run competition
        # compete returns module name (str)
        winner_name = self.global_workspace.compete(proposals)
        
        # 5. Determine Final Action
        trace = {"proposals": len(proposals)}
        
        winner = None
        if winner_name and winner_name in proposals:
            winner = proposals[winner_name][0]
            action = winner.action
            trace["mode"] = winner.module
            trace["reason"] = winner.content
            
            # [AGI] Phase 3.2: Self-Model Context
            # Update self-confidence based on winning module
            if winner.module == "SNN":
                self.self_model.update_confidence(task_tag, confidence)
            elif winner.module == "PLANNER":
                 self.self_model.update_confidence(task_tag, 0.9) # Trust plans
                 
        else:
            # Fallback
            action = self._default_action(state, task_tag)
            trace["mode"] = "default"
            winner = Proposal(module="DEFAULT", action=action, content="Fallback", salience=0.0)

        # [AGI] Phase 5.2: Mental Simulation (Veto Check)
        # If we have a World Model, simulate the chosen action to check for disaster
        imagined_r = 0.0
        veto = False
        if self.world_model and self.world_model.is_ready(task_tag):
             imagined_r = self.imagine_rollout(situation_hv, [action.replace("ACTION_", "")], task_tag)
             trace["imagined_reward"] = imagined_r
             
             # Veto if disaster predicted
             if imagined_r < -0.8: # Death
                 veto = True
                 trace["veto"] = True
                 trace["veto_reason"] = "Predicted death"
                 # Attempt rescue? For now, just flag it. 
                 # In advanced mode, we'd loop back to pick 2nd best.

        # 6. Generate explanation
        trace["confidence"] = confidence
        trace["veto"] = veto
        explanation = self.explainer.explain_action(action, state, task_tag, trace)
        
        # 7. Check coherence
        pred_set = set(active_preds)
        coherence_check = self.coherence.check_predicates(pred_set)
        if not coherence_check.is_coherent:
            trace["coherence_issues"] = len(coherence_check.contradictions)
        
        # 8. Update curiosity
        self.curiosity.record_visit(situation_hv, task_tag)
        
        # [AGI] Phase 7: Record Trace
        full_trace = {
            "timestamp": time.time(),
            "task": task_tag,
            "winner": winner.module if winner else "NONE",
            "action": action,
            "veto": veto,
            "proposals": [p.module for p in [v[0] for v in proposals.values()]] # Simplify for JSON
        }
        self.trace_history.append(full_trace)
        if len(self.trace_history) > self.max_trace_history:
            self.trace_history.pop(0)

        # Build cognitive state
        self.current_state = CognitiveState(
            task_tag=task_tag,
            situation_hv=situation_hv,
            active_predicates=active_preds,
            chosen_action=action,
            confidence=confidence,
            self_confidence=self.self_model.get_confidence(task_tag),
            exploration_mode=explore_decision.should_explore,
            explanation=explanation,
            imagined_reward=imagined_r,
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
        
        elif task_tag == "maze":
            # Default action for Maze: random move to prevent curiosity trapping
            # (In Snake/Pong we have clear heuristics, in Maze we follow the planner or explore)
            import random
            return random.choice(["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"])
        
        return "ACTION_STAY"
    
    def learn(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "neutral",
        next_state: Optional[Dict[str, Any]] = None,
        image: Optional[np.ndarray] = None
    ):
        """
        Learn from experience.
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
                reward=reward,
                image=image # [AGI] Save image for dreaming
            )
            self.episodic_memory.record(episode)
            self.stats["episodes_recorded"] += 1
        
        # 3. Update curiosity
        self.curiosity.record_outcome(task_tag, reward > 0)
        
        # 4. Periodically induce rules
        if self.stats["episodes_recorded"] % 50 == 0:
            new_rules = self.rule_learner.induce_rules(task_tag)
            self.stats["rules_induced"] += len(new_rules)
            
        # [AGI] Phase 4.1: Causal Discovery Update
        # If we have next_state, we can check for causal links
        # state[predicate] -> action -> next_state[predicate]
        if next_state:
             # Fix for TypeError: Extract symbolic events first
             
             # 1. Causes: Pre-conditions + Action
             # Use predicates from the decision time (stored in current_state) if available
             if self.current_state and self.current_state.active_predicates:
                 causes = list(self.current_state.active_predicates)
             else:
                 # Fallback: re-extract
                 verifier = self.verifiers.get(task_tag, GroundingVerifier())
                 causes = verifier.get_active_predicates(state, context=task_tag)
             
             causes.append(action)
             
             # 2. Effects: Post-conditions
             verifier = self.verifiers.get(task_tag, GroundingVerifier())
             effects = verifier.get_active_predicates(next_state, context=task_tag)
             
             # Add reward signals to effects
             if reward > 0:
                 effects.append("REWARD_POS")
             elif reward < 0:
                 effects.append("REWARD_NEG")

             # 3. Call observe with correct signature: (context, causes, effects)
             self.causal_discovery.observe(task_tag, causes, effects)
             
             if self.stats["episodes_recorded"] < 20 or self.stats["episodes_recorded"] % 5 == 0:
                 print(f"[DEBUG] CE.learn: Recorded {self.stats['episodes_recorded']} transitions for {task_tag}")
             
             # Periodically induce graph
             # Periodically induce graph (Reduced to 10 for rapid verification)
             if self.stats["episodes_recorded"] % 10 == 0:
                 induced_graph = self.causal_discovery.induce_graph(task_tag)
                 # Merge logic: Add discovered links to the active graph
                 active_graph = self.causal_graphs[task_tag]
                 new_rules_count = 0
                 
                 for link in induced_graph.all_links:
                     # Check if this rule is new or stronger
                     existing_links = active_graph.forward.get(link.cause, [])
                     match = next((l for e, l in existing_links if e == link.effect), None)
                     
                     if not match:
                         active_graph.add_link(link)
                         new_rules_count += 1
                     elif link.strength > match.strength:
                         match.strength = link.strength # Update confidence
                         
                 if new_rules_count > 0:
                     print(f"[AGI] Causal Discovery: Integrated {new_rules_count} new rules for {task_tag}")

        # [AGI] Phase 5.1: World Model Update
        # Train world model on transition (s, a) -> (s', r)
        # We need HVs for s and s'
        if self.world_model and self.current_state.situation_hv and next_state:
             # Re-generate HV for next state? simpler to just pass numpy bits if we had them.
             # For now, simplistic update:
             if hasattr(self.current_state.situation_hv, 'bits'):
                  obs_bits = np.packbits(self.current_state.situation_hv.bits)
                  # We'd need next_hv ... let's skip rigorous training in this snip
                  pass
    
    def transfer(
        self,
        source_task: str,
        target_task: str,
        state: Dict[str, Any],
        active_predicates: List[str]
    ) -> Optional[str]:
        """
        Transfer knowledge from source to target task (zero-shot).
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
        """Get explanation for current decision."""
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
            {},
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
    # --- [AGI] Phase 5: Generative Imagination ---
    
    def dream(self, num_samples: int = 10, task_tag: str = "snake") -> List[Dict[str, Any]]:
        """
        Perform 'Generative Dreaming' to consolidate knowledge.
        
        Uses the WorldModel to evaluate actions across multiple sampled states,
        generating synthetic experiences to train the Fast System (SNN).
        """
        dreams = []
        # Support sampling from episodic memory
        episodes = self.episodic_memory.sample(task_tag, num_samples)
        
        # Get verifier for action mapping
        act_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"] if task_tag != "pong" else ["ACTION_UP", "ACTION_DOWN"]
        
        for ep in episodes:
            initial_hv = ep.situation_hv
            if not initial_hv:
                continue
                
            # For each possible action, imagine the outcome
            action_evals = []
            for action_str in act_names:
                # 1-step lookahead
                imagined_r = self.imagine_rollout(initial_hv, [action_str], task_tag)
                action_evals.append((action_str, imagined_r))
                
            # Find the best imagined action
            best_action_full, best_reward = max(action_evals, key=lambda x: x[1])
            # Strip "ACTION_" for external compatibility
            best_action = best_action_full.replace("ACTION_", "")
            
            # Create a synthetic experience
            # We use the REAL state/image from memory, but a SYNTHETIC target action/reward
            # derived from System 2 imagination.
            dream = {
                "state": ep.state,
                "image": ep.image, # [AGI] Pass through image for SNN training
                "situation_hv": ep.situation_hv,
                "action": best_action,
                "reward": best_reward,
                "task_tag": task_tag,
                "is_dream": True
            }
            dreams.append(dream)
            
        return dreams

    def generate_hypothetical_lessons(self, num_anchors: int = 5, task_tag: str = "snake") -> List[Dict[str, Any]]:
        """
        [AGI] Phase 5.4: Hypothetical Scenario Generation
        Proactively search for extreme outcomes (Success/Failure) in imagination.
        """
        lessons = []
        # 1. Sample anchor states from memory
        anchors = self.episodic_memory.sample(task_tag, num_anchors)
        
        # 2. Define possible action HVs
        act_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"] if task_tag != "pong" else ["ACTION_UP", "ACTION_DOWN"]
        act_hvs = [self._get_action_hv(name) for name in act_names]
        
        for anchor in anchors:
            if not anchor.situation_hv:
                continue
                
            # 3. Explore from this anchor
            paths = self.world_model.sample_hypothetical_trajectories(
                anchor.situation_hv, 
                act_hvs, 
                horizon=5, 
                num_paths=8
            )
            
            # 4. Filter for high-impact paths
            for path in paths:
                if not path:
                    continue
                    
                # Check for extreme reward in the last step of the path
                last_step = path[-1]
                if abs(last_step["reward"]) > 0.5:
                    # We found a scenario! (e.g. death or goal)
                    # We create a lesson: State(t-1) -> Action(t-1) -> Reward(t)
                    lesson = {
                        "anchor_state": anchor.state,
                        "anchor_image": anchor.image,
                        "action": self._get_action_name(last_step["action_hv"]),
                        "reward": last_step["reward"],
                        "steps": len(path),
                        "is_hypothetical": True,
                        "task_tag": task_tag
                    }
                    lessons.append(lesson)
                    
        return lessons

    def imagine_rollout(self, initial_hv: hypervec_rs.HyperVector, action_sequence: List[str], task_tag: str, gamma: float = 0.9) -> float:
        """
        [AGI] Phase 5.2: Mental Simulation
        Simulate a sequence of actions from a state using the World Model.
        Returns total discounted reward.
        """
        if not self.world_model or not self.world_model.is_ready(task_tag):
            return 0.0
            
        total_imagined_reward = 0.0
        
        # Start in numpy land
        current_bits = self.world_model.hv_to_numpy(initial_hv)
        
        for i, action_str in enumerate(action_sequence):
            action_hv = self._get_action_hv(action_str)
            action_bits = self.world_model.hv_to_numpy(action_hv)
            
            # Predict next using the low-level predictor to avoid redundant HV conversions
            next_bits, reward = self.world_model.predictor.predict(current_bits, action_bits)
            total_imagined_reward += (gamma ** i) * reward
            current_bits = next_bits
            
        return total_imagined_reward

    def _get_action_name(self, action_hv: hypervec_rs.HyperVector) -> str:
        """Reverse lookup for action name."""
        from symbol_grounding import GLOBAL_PRIMITIVES_MAP
        for name, seed in GLOBAL_PRIMITIVES_MAP.items():
            if seed == getattr(action_hv, 'seed', -1):
                return name.replace("ACTION_", "")
        return "UNKNOWN"

    def _get_action_hv(self, action_str: str) -> hypervec_rs.HyperVector:
        """Get hypervector representation for a symbolic action."""
        from symbol_grounding import GLOBAL_PRIMITIVES_MAP
        seed = GLOBAL_PRIMITIVES_MAP.get(action_str, 0)
        return hypervec_rs.HyperVector(seed)

    # --- [AGI] Phase 7: Supervisor Interface ---
    
    def set_mission_goal(self, goal_type: str, threshold: float):
        """Set high-level mission goal."""
        self.mission_goal = {"type": goal_type, "threshold": threshold}
        print(f"[COGNITION] Mission Goal Set: {goal_type} >= {threshold}")
        
    def export_traces(self, filename: str):
        """Export decision traces to JSON."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.trace_history, f, indent=2)
            print(f"[COGNITION] Traces exported to {filename}")
        except Exception as e:
            print(f"[COGNITION] Export failed: {e}")
            
    def get_workspace_telemetry(self) -> Dict:
        """Get telemetry for dashboard visualizer."""
        return {
            "winner": self.trace_history[-1]["winner"] if self.trace_history else "NONE",
            "active_modules": ["SNN", "RULES", "EXPLORATION", "PLANNER"], 
        }
    
    def get_causal_telemetry(self, task_tag: str) -> Dict:
        """Get causal graph stats."""
        graph = self.causal_graphs.get(task_tag)
        return {
            "link_count": len(graph.all_links) if graph else 0,
        }


# Factory function for easy creation
def create_cognitive_engine(persistence_path: Optional[str] = None) -> CognitiveEngine:
    """Create a fully configured cognitive engine."""
    return CognitiveEngine(
        config=NSCKConfig(),
        persistence_path=persistence_path
    )
