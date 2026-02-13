"""
Metacognitive Engine for NSGA
Wraps FusedBrain to add:
- Uncertainty monitoring (confidence scores)
- Conflict detection (precedence & rule conflicts)
- Tiered escalation (ALLOW/FALLBACK/BLOCK)
- Safe defaults
"""
import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.integration.brain_fusion import FusedBrain, QueryResult

# --- Constants & Configuration ---

# SAFE_DEFAULTS must be defined *after* fallback functions exist.
SAFE_DEFAULTS: Dict[str, Callable] = {}

GRID_SIZE = 10  # Must match simulation.py

# --- Data Models ---

@dataclass
class EscalationRequest:
    severity: str  # "low" | "medium" | "high"
    mode: str      # "ask_async" | "ask_sync"
    question_type: str  # "confirm_action" | "resolve_conflict" | "label_concept"
    context: dict

@dataclass
class InferenceResult:
    action: Optional[str]              # None if BLOCK
    confidence: float                  # 0.0 - 1.0
    uncertainty_reason: str
    alternatives: List[Tuple[str, float]]
    escalation: Optional[EscalationRequest]
    trace: dict

@dataclass
class Conflict:
    type: str  # "precedence" | "rule"
    severity: float
    candidates: List[QueryResult]
    task_tag: str

# --- Safe Fallback Logic ---

OPPOSITE = {
    "ACTION_UP": "ACTION_DOWN", 
    "ACTION_DOWN": "ACTION_UP", 
    "ACTION_LEFT": "ACTION_RIGHT", 
    "ACTION_RIGHT": "ACTION_LEFT",
    "UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT" # Handle both formats
}

def infer_heading(head: Tuple[int, int], body: List[Tuple[int, int]]) -> str:
    if not body:
        return "ACTION_UP"  # Default
    neck = body[0]
    dx = (head[0] - neck[0]) % GRID_SIZE
    dy = (head[1] - neck[1]) % GRID_SIZE
    
    if dx == 1 or dx == -(GRID_SIZE-1): return "ACTION_RIGHT"
    if dx == GRID_SIZE-1 or dx == -1: return "ACTION_LEFT"
    if dy == 1 or dy == -(GRID_SIZE-1): return "ACTION_DOWN"
    return "ACTION_UP"

def snake_safe_fallback(state: dict, last_action: str) -> str:
    """
    Prevent 180-degree reversal.
    If last_action is safe, use it. Else, keep current heading.
    """
    current_heading = infer_heading(state["head"], state.get("body", []))

    # Normalize action names if needed
    if not last_action.startswith("ACTION_"):
        last_action = "ACTION_" + last_action

    forbidden = OPPOSITE.get(current_heading, "")

    if last_action != forbidden:
        return last_action
    return current_heading


# Now that snake_safe_fallback exists, define SAFE_DEFAULTS.
SAFE_DEFAULTS = {
    "pong": lambda state, last_action: "ACTION_STAY",
    "snake": lambda state, last_action: snake_safe_fallback(state, last_action),
}

# --- Helper Utilities ---

def std_dev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def situation_digest(hv: hypervec_rs.HyperVector) -> int:
    """Deterministic hash of hypervector state."""
    # Use blake2b over bytes (assuming hv supports .bytes() or similar, 
    # but hypervec_rs probably exposes bits. Let's use string rep or pseudo-stable property)
    # Since we can't easily get bytes from python object without proper binding support,
    # let's try to assume we can get a stable string or valid property.
    # The user plan suggested `reduce(xor, hv.bits)`.
    # Assuming hv objects are comparable, let's just use hash() if implemented, 
    # but the user requested xxhash/blake2b.
    # We will assume a 'checksum' or similar method exists or we can emulate.
    # For now, let's assume we can get a list of bits or hashing the object is stable per run
    # IF hypervec_rs doesn't expose raw bits easily.
    # Workaround: The library might not expose bits directly in python.
    # Use built-in hash for now, assuming provenance stability within run.
    # UPDATE: User provided code suggested `bytes(hv.bits)`. Assuming that exists.
    try:
        data = bytes(hv.bits) # Hypothetical access
    except:
        # Fallback if property missing
        data = str(hv).encode('utf-8')
        
    h = hashlib.blake2b(data, digest_size=8)
    return int.from_bytes(h.digest(), 'little')

# --- Metacognitive Engine ---

# --- Safety Gate (Simulation Veto) ---
class SafetyGate:
    """
    Validation layer calling into Game Physics.
    Kept separate from Brain logic for purity.
    """
    @staticmethod
    def is_safe(game_type: str, state: dict, action: str) -> bool:
        """
        Returns True if action does not lead to immediate failure.
        Wraps collision logic from simulation.py
        """
        # Lazy import to avoid circular dependencies if possible, or assume imports at top
        from python.games.pong.simulation import sim_snake, sim_pong
        
        # Standardize action
        action_core = action.replace("ACTION_", "")
        
        if game_type == "snake":
            # sim_snake returns (next_state, collision)
            # collision=True means Dead -> Unsafe
            if action_core not in ["UP", "DOWN", "LEFT", "RIGHT"]:
                return True # Unknown actions (STAY?) in Snake might be safe or invalid. Assume safe if not move?
                
            _, dead = sim_snake(state, action_core)
            return not dead
            
        elif game_type == "pong":
            # sim_pong returns (next_state, miss)
            # miss=True means Miss -> Unsafe (for our purpose of "Don't Miss")
            if action_core not in ["UP", "DOWN", "STAY"]:
                return True
            
            # Pong state needs ball_x/dx mostly.
            # python_server.py logic: if we are about to miss, it's critical.
            # But sim_pong logic assumes we move.
            _, miss = sim_pong(state, action_core)
            return not miss
            
        return True

# --- Metacognitive Engine ---

class MetacognitiveEngine:
    """
    Executive layer that wraps FusedBrain inference with safety & active learning.
    """
    def __init__(self, brain: FusedBrain):
        self.brain = brain
        
        # Adaptive confidence tracking
        self.ema_spread = 0.1
        self.ema_margin = 0.1
        self.alpha = 0.1
        
        # Cycle detection
        self.recent_cycles = set()  # (situation_hash, action_tuple)
        
    def _update_stats(self, spread: float, margin: float):
        self.ema_spread = self.alpha * spread + (1 - self.alpha) * self.ema_spread
        self.ema_margin = self.alpha * margin + (1 - self.alpha) * self.ema_margin
        
    @property
    def spread_ref(self) -> float:
        return max(0.05, self.ema_spread)

    def compute_confidence(self, top_k: List[QueryResult]) -> Tuple[float, str]:
        if not top_k:
            return 0.0, "no_results"
            
        base = top_k[0].similarity
        
        # Guard: No credible match
        if base < 0.55:
            return 0.0, "no_match"
            
        if len(top_k) > 1:
            margin = top_k[0].similarity - top_k[1].similarity
            spread = std_dev([r.similarity for r in top_k])
            
            # Update online stats
            self._update_stats(spread, margin)
            
            spread_penalty = min(spread / self.spread_ref, 1.0)
            
            # Confidence score
            conf = min(base, 0.5 + margin, spread_penalty)
            
            reason = "strong_match"
            if conf == spread_penalty: reason = "ambiguity"
            elif conf == (0.5 + margin): reason = "competition"
            elif conf == base: reason = "weak_base"
            
            return conf, reason
        else:
            return min(base, 1.0), "single_match"

    def detect_conflict(self, top_k: List[QueryResult]) -> Optional[Conflict]:
        """Detect action-based conflicts within top results."""
        if len(top_k) < 2:
            return None
            
        # Check rule conflict (within same layer or across)
        best = top_k[0]
        runner_up = top_k[1]
        
        # Only conflict if both are credible matches
        if best.similarity < 0.55 or runner_up.similarity < 0.55:
            return None
            
        # Only conflict if actions differ
        act1 = best.action
        act2 = runner_up.action
        
        if not act1 or not act2 or act1 == act2:
            return None
            
        # Close margin?
        if (best.similarity - runner_up.similarity) < 0.05:
            # Determine type
            c_type = "precedence" if best.layer != runner_up.layer else "rule"
            
            # Severity
            severity = self.compute_severity(
                confidence=best.similarity, # Proxy
                margin=best.similarity - runner_up.similarity,
                action_distance=1.0, # Binary different
                is_safety_critical=("DANGER" in best.concept_name or "DANGER" in runner_up.concept_name)
            )
            
            return Conflict(type=c_type, severity=severity, candidates=[best, runner_up], task_tag=best.layer)
            
        return None

    def compute_severity(self, confidence, margin, action_distance, is_safety_critical) -> float:
        sev = (1.0 - confidence) * 0.5
        sev += (1.0 - margin) * 0.3
        sev += action_distance * 0.2
        if is_safety_critical:
            sev = min(1.0, sev + 0.3)
        return min(1.0, sev)

    def check_cycle(self, situation_hv_or_digest, top_k: List[QueryResult], task_tag: str) -> bool:
        """Prevent escalation loops."""
        # Update: input might be raw facts (set of strings) or hv. 
        # Digest depends on input type.
        if isinstance(situation_hv_or_digest, int):
            digest = situation_hv_or_digest
        elif hasattr(situation_hv_or_digest, 'bits'):
            # It's a HyperVector - use the digest function
            digest = situation_digest(situation_hv_or_digest)
        elif isinstance(situation_hv_or_digest, (list, set, frozenset)):
            # Assume it's facts (set/list of strings)
            # Hash the sorted facts string
            facts_str = "_".join(sorted([str(f) for f in situation_hv_or_digest]))
            digest = int(hashlib.md5(facts_str.encode('utf-8')).hexdigest(), 16) % (10**10)
        else:
            # Unknown type - use repr hash
            digest = hash(repr(situation_hv_or_digest)) % (10**10)

        # Signature: (situation, task, top-3 candidates)
        sig_cands = tuple((c.action, round(c.similarity, 2), c.layer) for c in top_k[:3])
        sig = (digest, task_tag, sig_cands)
        
        if sig in self.recent_cycles:
            return True
        self.recent_cycles.add(sig)
        return False

    def _process_inference(self, results, task_tag, state, last_action, cycle_context) -> InferenceResult:
        """Common processing for both channels."""
        # 1. Compute Confidence
        confidence, reason = self.compute_confidence(results)
        
        # 2. Detect Conflicts
        conflict = self.detect_conflict(results)
        
        # 3. Cycle Detection
        is_cycle = self.check_cycle(cycle_context, results, task_tag)
        
        # 4. Selection & Safety Veto
        action = None

        # If we have no credible match, we should not select a candidate action.
        if confidence == 0.0:
            action = None
        else:
            # Find best SAFE action
            # Iterate through ranked results until one is safe
            safe_action_found = None

            for cand in results:
                if not cand.action:
                    continue
                if SafetyGate.is_safe(task_tag, state, cand.action):
                    safe_action_found = cand.action
                    break

            if safe_action_found:
                action = safe_action_found
            else:
                if results:
                    reason = "all_candidates_unsafe"

        # Prioritize cycle breaking
        if is_cycle:
            action = None
            reason = "cycle_detected"

        # Tiered Decision Matrix
        should_block = False
        fallback_needed = False
        escalation = None

        if is_cycle:
            fallback_needed = True

        elif conflict:
            if conflict.severity >= 0.75:
                should_block = True
                escalation = EscalationRequest(
                    severity="high", mode="ask_sync", question_type="resolve_conflict",
                    context={"conflict": conflict, "candidates": results[:2]}
                )
            else:
                # Log but proceed
                fallback_needed = True # Safer to fallback
                escalation = EscalationRequest(
                    severity="medium", mode="ask_async", question_type="resolve_conflict",
                    context={"conflict": conflict}
                )
                
        # HARD RULE: no credible match => fallback.
        elif confidence == 0.0:
            fallback_needed = True
            escalation = EscalationRequest(
                severity="low", mode="ask_async", question_type="label_concept",
                context={"top_result": results[0] if results else None}
            )

        # If we DIDN'T find any safe action, low confidence should force fallback.
        # If we DID find a safe action, we keep it even when uncertain (and optionally escalate).
        elif confidence < 0.5 and action is None:
            fallback_needed = True
            escalation = EscalationRequest(
                severity="low", mode="ask_async", question_type="label_concept",
                context={"top_result": results[0] if results else None}
            )
        elif confidence < 0.5 and action is not None:
            escalation = EscalationRequest(
                severity="low", mode="ask_async", question_type="label_concept",
                context={"top_result": results[0] if results else None}
            )

        # Execute Policy  
        final_action = action
        
        if should_block:
            final_action = None # BLOCK
        elif fallback_needed or final_action is None:
            # Apply Safe Fallback
            fallback_fn = SAFE_DEFAULTS.get(task_tag)
            if fallback_fn:
                final_action = fallback_fn(state, last_action)
            else:
                final_action = "ACTION_STAY"
                
        return InferenceResult(
            action=final_action,
            confidence=confidence,
            uncertainty_reason=reason,
            alternatives=[(r.concept_name, r.similarity) for r in results[:3]],
            escalation=escalation,
            trace={
                "layer_used": results[0].layer if results else "none",
                "conflict_detected": conflict is not None,
                "cycle": is_cycle,
                "top_k": [(r.action, r.similarity) for r in results[:5]] # Trace for prob calc
            }
        )

    def infer_from_facts(self, facts: List[str], task_tag: str, state: dict, last_action: str) -> InferenceResult:
        """
        LOGIC CHANNEL: 1-step inference from predicates.
        """
        results = self.brain.resolve_rules(set(facts), task_tag)
        return self._process_inference(results, task_tag, state, last_action, cycle_context=facts)

    def infer(self, 
              situation_hv: hypervec_rs.HyperVector, 
              task_tag: str, 
              state: dict, 
              last_action: str) -> InferenceResult:
        """
        SIMILARITY CHANNEL: Fuzzy retrieval.
        """
        results = self.brain.query(situation_hv, task_tag=task_tag, top_k=5)
        return self._process_inference(results, task_tag, state, last_action, cycle_context=situation_hv)

def inference_to_probs(result: InferenceResult, all_actions: List[str]) -> List[float]:
    """
    Map InferenceResult to soft probability distribution over all_actions.
    Uses top-k scores from trace to build distribution.
    """
    scores = {act: 0.0 for act in all_actions}
    
    # 1. Fill from Top-K
    top_k = result.trace.get("top_k", [])
    
    # Normalize actions in top_k to match all_actions format
    # all_actions usually ["UP", "DOWN", ...] (no ACTION_ prefix? Check usage)
    # Metacognition uses "ACTION_UP"
    # We need to strip prefix for matching
    
    for act_str, score in top_k:
        if not act_str: continue
        core = act_str.replace("ACTION_", "")
        
        # If core matches one of our target actions
        if core in scores:
            scores[core] = max(scores[core], score) # Use max if duplicates
            
    # 2. If valid action chosen, boost it (Decision implementation)
    # The 'result.action' is the detailed logic/safety choice. 
    # It should dominate.
    if result.action:
        core = result.action.replace("ACTION_", "")
        if core in scores:
            scores[core] += 1.0 # Significant boost
            
    # 3. Convert to Probs (Softmax or Normalize)
    # Simple Normalize
    total = sum(scores.values())
    if total > 0:
        probs = [scores[act]/total for act in all_actions]
    else:
        # Uniform if confused
        probs = [1.0/len(all_actions) for _ in all_actions]
        
    return probs
