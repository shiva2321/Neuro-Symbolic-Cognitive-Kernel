import numpy as np
from typing import List, Set
import hypervec_rs
from brain_fusion import FusedBrain, TaskBrain, Rule, ConceptType
from metacognition import MetacognitiveEngine, inference_to_probs

# --- CONCEPT CODEBOOK (CONSTANTS) ---
# Keeping constants for reference, though Brain handles them dynamically now.
GLOBAL_PRIMITIVES_MAP = {
    "ACTION_UP": 10,
    "ACTION_DOWN": 20,
    "ACTION_LEFT": 30,
    "ACTION_RIGHT": 40,
    "REL_ABOVE": 101,
    "REL_BELOW": 102,
    "REL_LEFT": 103,
    "REL_RIGHT": 104
}

def bootstrap_metacognitive_brain() -> MetacognitiveEngine:
    """
    Initialize FusedBrain with Ground Truth mechanics.
    """
    # 1. Create Task Brains
    snake_brain = TaskBrain("snake")
    pong_brain = TaskBrain("pong")
    
    # 2. Populate Snake Rules (Navigation)
    # Mapping old 'ReasoningEngine' XOR logic to Directional Rules
    # Note: Using full action names (ACTION_DOWN vs ACTION_DN) to match Metacognitive standards
    
    # Add Concepts (optional but good for context)
    for name, seed in GLOBAL_PRIMITIVES_MAP.items():
        snake_brain.add_concept(name, hypervec_rs.HyperVector(seed))
        
    # Rules: Relation -> Action
    snake_brain.add_rule(frozenset(["REL_ABOVE"]), "ACTION_UP", priority=1)
    snake_brain.add_rule(frozenset(["REL_BELOW"]), "ACTION_DOWN", priority=1)
    snake_brain.add_rule(frozenset(["REL_LEFT"]), "ACTION_LEFT", priority=1)
    snake_brain.add_rule(frozenset(["REL_RIGHT"]), "ACTION_RIGHT", priority=1)
    
    # 3. Populate Pong Rules
    # Reuse primitive concepts
    for name, seed in GLOBAL_PRIMITIVES_MAP.items():
        pong_brain.add_concept(name, hypervec_rs.HyperVector(seed))
        
    # Pong also uses vertical logic
    pong_brain.add_rule(frozenset(["REL_ABOVE"]), "ACTION_UP", priority=1)
    pong_brain.add_rule(frozenset(["REL_BELOW"]), "ACTION_DOWN", priority=1)
    
    # 4. Global Layer
    # Safety Veto Rules? (e.g. DANGER -> BLOCK)
    # For now, relying on Simulation Veto inside MetacognitiveEngine.
    
    # 5. Fuse
    from brain_fusion import BrainFusion
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    fusion.register_brain(pong_brain)
    
    fused_brain = fusion.fuse()
    
    # 6. Wrap
    return MetacognitiveEngine(fused_brain)


class MetacognitiveWrapper:
    """
    Shim to adapt MetacognitiveEngine to the legacy 'infer_navigation' interface.
    """
    def __init__(self):
        self.engine = bootstrap_metacognitive_brain()
        print(">> MetacognitiveEngine Bootstrapped & Grounded.")
        
    def infer_navigation(self, is_above, is_below, is_left, is_right, game_type="snake", state=None):
        # 1. Extract Facts
        facts = []
        if is_above: facts.append("REL_ABOVE")
        if is_below: facts.append("REL_BELOW")
        if is_left:  facts.append("REL_LEFT")
        if is_right: facts.append("REL_RIGHT")
        
        if not facts:
            # No input -> No output (or uniform logic?)
            # Legacy returned zeros.
            # We return uniform? Or zeros.
            return np.zeros(4, dtype=np.float32)
            
        # 2. Metacognitive Inference (Logic Channel)
        # We need a dummy state if not provided (SafetyGate check will fail or pass?)
        # For now, pass empty state if None, but SafetyGate requires valid state for veto.
        # ActionSemantics calls us... let's see where state comes from.
        if state is None: state = {}
        
        result = self.engine.infer_from_facts(
            facts=facts, 
            task_tag=game_type, 
            state=state, 
            last_action="UNKNOWN" # We don't track last action in this shim yet
        )
        
        # 3. Map to Probability Array
        # Expected order: UP, DOWN, LEFT, RIGHT
        all_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        probs = inference_to_probs(result, all_actions)
        
        # Legacy interface expects specific tensor shape?
        # python_server uses: biased_probs * (1 + VSA).
        # And it expects tensor.
        # ReasoningEngine returned np.array([sim_up, sim_dn, sim_lf, sim_rt])
        # These were similarities, not probs (could be > 1 or unnormalized).
        # Logic was: biased_probs = snn_probs * (1.0 + VSA_STRENGTH * prior_tensor)
        # So prior_tensor can be just the boost signal.
        
        # Probs sum to 1. If we return probs, it acts as a weight.
        # This is fine.
        return np.array(probs, dtype=np.float32)

# Global Instance (Lazy)
_kernel_engine = None

def get_kernel_engine():
    global _kernel_engine
    if _kernel_engine is None:
        _kernel_engine = MetacognitiveWrapper()
    return _kernel_engine


class ActionSemantics:
    """
    Defines the 'Meaning' of actions in terms of abstract predicates.
    Proxies to Metacognitive Engine.
    """
    
    @staticmethod
    def get_goal_alignment(game_type, state):
        engine = get_kernel_engine() # Lazy access
        if not state: return []
        
        if game_type == "snake":
            head = state.get("head")
            food = state.get("food")
            if head is None or food is None: return []
            hx, hy = head
            fx, fy = food
            # Pass state for Safety Veto
            return engine.infer_navigation(fy < hy, fy > hy, fx < hx, fx > hx, "snake", state)
        
        elif game_type == "pong":
            p1_y = state.get("p1_y")
            ball_y = state.get("ball_y")
            if p1_y is None or ball_y is None: return []
            paddle_center = p1_y + 3
            # Map Pong to same bitwise navigation logic as Snake
            scores_4 = engine.infer_navigation(ball_y < paddle_center - 1, ball_y > paddle_center + 1, False, False, "pong", state)
            return scores_4[:2] # Only UP/DN
            
        elif game_type == "maze":
            player = state.get("player_pos")
            exit_pos = state.get("exit_pos")
            if player is None or exit_pos is None: return []
            px, py = player
            ex, ey = exit_pos
            return engine.infer_navigation(ey < py, ey > py, ex < px, ex > px, "maze", state)
            
        return []

    @staticmethod
    def get_danger_alignment(game_type, state):
        pass
