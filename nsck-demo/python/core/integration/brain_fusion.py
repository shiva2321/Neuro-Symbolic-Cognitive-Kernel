"""
Brain Fusion Module for NSGA
Strategy: tagged_conservative (task-specific isolated, primitives merged)
"""
import python.core.vsa.hypervec_shim as hypervec_rs
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from enum import Enum
import uuid

# Global Primitives (deterministic merge - fixed seeds from symbol_grounding.py)
GLOBAL_PRIMITIVES = {
    "ACTION_UP": 10,
    "ACTION_DN": 20,
    "ACTION_LF": 30,
    "ACTION_RT": 40,
    "REL_ABOVE": 101,
    "REL_BELOW": 102,
    "REL_LEFT": 103,
    "REL_RIGHT": 104,
}


class ConceptType(Enum):
    """Concept type for type consistency gate."""
    ACTION = "action"
    RELATION = "relation"
    OBJECT = "object"
    STATE = "state"
    GOAL = "goal"
    UNKNOWN = "unknown"


def infer_type(name: str) -> ConceptType:
    """Infer concept type from naming convention."""
    name_upper = name.upper()
    if name_upper.startswith("ACTION"):
        return ConceptType.ACTION
    elif name_upper.startswith("REL"):
        return ConceptType.RELATION
    elif any(x in name_upper for x in ["GOAL", "TARGET", "WAYPOINT"]):
        return ConceptType.GOAL
    elif any(x in name_upper for x in ["FOOD", "BALL", "PADDLE", "OBJECT"]):
        return ConceptType.OBJECT
    elif any(x in name_upper for x in ["STATE", "DANGER", "SAFE"]):
        return ConceptType.STATE
    return ConceptType.UNKNOWN


@dataclass(frozen=True)
class Rule:
    """
    Directional Logic Rule.
    Immutable to safe-guard against modification during runtime.
    """
    condition: frozenset[str]
    consequence: str
    strength: float = 1.0
    priority: int = 0
    task_tag: str = "global"
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class TaskBrain:
    """Represents a single task's knowledge."""
    def __init__(self, task_tag: str):
        self.task_tag = task_tag
        self.codebook: Dict[str, hypervec_rs.HyperVector] = {}
        self.concept_types: Dict[str, ConceptType] = {}
        self.rules: List[Rule] = []
        
    def add_concept(self, name: str, hv: hypervec_rs.HyperVector, 
                    concept_type: ConceptType = None):
        self.codebook[name] = hv
        self.concept_types[name] = concept_type or infer_type(name)

    def add_rule(self, condition: frozenset[str], consequence: str, 
                 strength: float = 1.0, priority: int = 0):
        rule = Rule(
            condition=condition,
            consequence=consequence,
            strength=strength,
            priority=priority,
            task_tag=self.task_tag
        )
        self.rules.append(rule)
        return rule
    
    def get_concept_context(self, concept_name: str, top_k: int = 5) -> Optional[hypervec_rs.HyperVector]:
        """
        Compute context signature by bundling HVs of concepts that co-occur
        with this concept in rules (as conditions or consequences).
        """
        if concept_name not in self.codebook:
            return None
        
        # Gather co-occurring concepts from rules
        related_names = set()
        for rule in self.rules:
            if concept_name in rule.condition or concept_name == rule.consequence:
                related_names.update(rule.condition)
                related_names.add(rule.consequence)
        related_names.discard(concept_name)
        
        if not related_names:
            return None
        
        # Bundle HVs of related concepts (up to top_k)
        related_hvs = []
        for name in list(related_names)[:top_k]:
            if name in self.codebook:
                related_hvs.append(self.codebook[name])
        
        if not related_hvs:
            return None
        
        ctx = related_hvs[0]
        for hv in related_hvs[1:]:
            ctx = ctx.bundle(hv)
        return ctx


@dataclass
class QueryResult:
    """Structured query result with provenance."""
    concept_name: str
    action: str              # Extracted action (ACTION_UP etc)
    score: float             # Raw support score
    similarity: float        # Normalized confidence [0,1]
    layer: str  # 'global' or task_tag
    rule_ids: List[str]
    concept_type: Optional[ConceptType] = None
    task_tag: Optional[str] = None
    
    # Backwards compatibility
    @property
    def value(self): return self.similarity 


class FusedBrain:
    """Two-layer knowledge structure after fusion."""
    def __init__(self):
        # Global layer
        self.global_codebook: Dict[str, hypervec_rs.HyperVector] = {}
        self.global_types: Dict[str, ConceptType] = {}
        self.global_rules: List[Rule] = []
        
        # Context layers
        self.task_codebooks: Dict[str, Dict[str, hypervec_rs.HyperVector]] = defaultdict(dict)
        self.task_types: Dict[str, Dict[str, ConceptType]] = defaultdict(dict)
        self.task_rules: Dict[str, List[Rule]] = defaultdict(list)
        
        # Provenance
        self.provenance: Dict[str, List[str]] = {} 
        self.merge_map: Dict[Tuple[str, str], str] = {} 
        self.rollback_log: List[dict] = []
        
    def _extract_action(self, name: str) -> Optional[str]:
        """Extract action from concept name."""
        if "ACTION_" in name:
            parts = name.split("ACTION_")
            if len(parts) > 1:
                token = parts[1]
                for act in ["UP", "DN", "LF", "RT", "DOWN", "LEFT", "RIGHT", "STAY"]:
                     if token.startswith(act):
                        if act == "DN": return "ACTION_DOWN"
                        if act == "LF": return "ACTION_LEFT"
                        if act == "RT": return "ACTION_RIGHT"
                        return f"ACTION_{act}"
        return None

    def resolve_rules(self, facts: Set[str], task_tag: str, top_k: int = 5) -> List[QueryResult]:
        """
        LOGIC CHANNEL: Deterministic 1-step inference.
        Returns ranked candidates based on active rules.
        """
        evidence = defaultdict(float)
        hits = defaultdict(list)
        
        # 1. Gather Rules (Task + Global)
        # Precedence: Task rules get a boost via layer_boost calculation
        task_rules_list = self.task_rules.get(task_tag, [])
        global_rules_list = self.global_rules
        
        all_rules = task_rules_list + global_rules_list
        
        SCORE_CAP = 5.0
        
        for rule in all_rules:
            # Check condition match
            if rule.condition.issubset(facts):
                # Calculate Weight
                # Task Override: 1.2x for task-specific rules
                layer_boost = 1.2 if rule.task_tag == task_tag else 1.0
                
                # Priority Boost: +10% per priority level
                prio_boost = 1.0 + (0.1 * rule.priority)
                
                weight = rule.strength * layer_boost * prio_boost
                
                evidence[rule.consequence] += weight
                hits[rule.consequence].append(rule.rule_id)
        
        if not evidence:
            return []
            
        # Normalize and Rank
        results = []
        for name, score in evidence.items():
            # Cap evidence to prevent confidence saturation
            score_eff = min(score, SCORE_CAP)
            
            # Saturating Normalization: x / (x + 1) -> maps [0, inf] to [0, 1]
            # 1.0 -> 0.5, 5.0 -> 0.83
            similarity = score_eff / (score_eff + 1.0)
            
            # Determine layer (provenance)
            # If any rule was from task layer, we mark as task? 
            # Or based on concept definition? 
            # Let's check rule provenance.
            # If most evidence came from task, mark as task.
            # Simplify: If name exists in task codebook, use task_tag, else global.
            layer = task_tag if name in self.task_codebooks.get(task_tag, {}) else "global"
            
            results.append(QueryResult(
                concept_name=name,
                action=self._extract_action(name) or name,
                score=score,
                similarity=similarity,
                layer=layer,
                rule_ids=hits[name],
                task_tag=task_tag if layer == task_tag else None,
                concept_type=self.task_types.get(layer, {}).get(name) or self.global_types.get(name)
            ))
            
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def query(self, query_hv: hypervec_rs.HyperVector, task_tag: str = None, 
              top_k: int = 5) -> List[QueryResult]:
        """
        SIMILARITY CHANNEL: Fuzzy retrieval.
        """
        results = []
        
        # Search task-specific layer first
        if task_tag and task_tag in self.task_codebooks:
            for name, hv in self.task_codebooks[task_tag].items():
                sim = query_hv.similarity(hv)
                results.append(QueryResult(
                    concept_name=f"{task_tag}::{name}",
                    action=self._extract_action(name),
                    score=sim, # Raw sim as score
                    similarity=sim,
                    layer=task_tag,
                    rule_ids=[],
                    task_tag=task_tag,
                    concept_type=self.task_types.get(task_tag, {}).get(name)
                ))
                
        # Then search global
        for name, hv in self.global_codebook.items():
            sim = query_hv.similarity(hv)
            results.append(QueryResult(
                concept_name=name,
                action=self._extract_action(name),
                score=sim,
                similarity=sim,
                layer='global',
                rule_ids=[],
                task_tag=None,
                concept_type=self.global_types.get(name)
            ))
            
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def forward_chain_multi(self, facts: Set[str], max_steps: int = 5) -> Tuple[Set[str], List[Set[str]]]:
        """
        [AGI] Phase 2.1: Forward Chaining Engine
        Fire rules until fixed point or max steps.
        
        Allows deducing new facts from existing facts using available rules.
        """
        current_facts = set(facts)
        all_inferred = []
        
        # Combine all rules for inference (copy to avoid mutating global_rules)
        all_rules = list(self.global_rules)
        for rules in self.task_rules.values():
            all_rules.extend(rules)
            
        for step in range(max_steps):
            new_facts = set()
            
            for rule in all_rules:
                # Only check rules that produce FACTS (not ACTIONS)
                # We identify actions by "ACTION_" prefix
                if rule.consequence.startswith("ACTION_"):
                    continue
                    
                if rule.condition.issubset(current_facts):
                    if rule.consequence not in current_facts:
                        new_facts.add(rule.consequence)
            
            if not new_facts:
                break  # Fixed point
                
            current_facts.update(new_facts)
            all_inferred.append(new_facts)
            
        return current_facts, all_inferred


class BrainFusion:
    """
    Fuses multiple TaskBrains using tagged_conservative strategy.
    """
    
    # Merge thresholds (adaptive: start strict, relax with more evidence)
    HV_THRESHOLD = 0.75   # Lowered from 0.92 — use context similarity to gate
    CTX_THRESHOLD = 0.65  # Lowered from 0.85 — allow more cross-task sharing
    
    # Strict mode can be enabled for safety-critical merges
    STRICT_HV_THRESHOLD = 0.92
    STRICT_CTX_THRESHOLD = 0.85
    
    def __init__(self):
        self.registered_brains: Dict[str, TaskBrain] = {}
        
    def register_brain(self, brain: TaskBrain):
        """Register a task brain for fusion."""
        self.registered_brains[brain.task_tag] = brain
        print(f"Registered brain: {brain.task_tag} ({len(brain.codebook)} concepts, {len(brain.rules)} rules)")
        
    def _is_global_primitive(self, name: str) -> bool:
        """Check if concept is a declared global primitive."""
        return name in GLOBAL_PRIMITIVES
        
    def _compute_hub_score(self, brain: TaskBrain, concept_name: str) -> float:
        """Hub score = number of rules that use this concept."""
        if not brain.rules:
            return 0.0
        count = sum(1 for rule in brain.rules if concept_name in rule.participants)
        return count  # Raw count, not fraction
        
    def align_concepts(self, brain_a: TaskBrain, brain_b: TaskBrain) -> List[Tuple[str, str, float, float]]:
        """
        Find matching concepts across two brains.
        Applies type consistency gate: never aligns concepts of different types.
        Returns: [(name_a, name_b, hv_similarity, ctx_similarity), ...]
        """
        alignments = []
        
        for name_a, hv_a in brain_a.codebook.items():
            type_a = brain_a.concept_types.get(name_a, ConceptType.UNKNOWN)
            ctx_a = brain_a.get_concept_context(name_a)
            
            for name_b, hv_b in brain_b.codebook.items():
                type_b = brain_b.concept_types.get(name_b, ConceptType.UNKNOWN)
                
                # TYPE CONSISTENCY GATE: Never merge across types
                if type_a != type_b and type_a != ConceptType.UNKNOWN and type_b != ConceptType.UNKNOWN:
                    continue
                    
                hv_sim = hv_a.similarity(hv_b)
                
                if hv_sim < 0.7:  # Skip obviously different
                    continue
                    
                # Context similarity
                ctx_b = brain_b.get_concept_context(name_b)
                if ctx_a and ctx_b:
                    ctx_sim = ctx_a.similarity(ctx_b)
                else:
                    ctx_sim = 0.5  # No context = uncertain
                    
                alignments.append((name_a, name_b, hv_sim, ctx_sim))
                
        # Sort by HV similarity
        alignments.sort(key=lambda x: x[2], reverse=True)
        return alignments
    
    def fuse(self, strategy: str = "tagged_conservative") -> FusedBrain:
        """
        Fuse all registered brains.
        
        Strategy: tagged_conservative
        1. Global primitives → merged deterministically to global
        2. High similarity + high context → merged to global
        3. Everything else → stays in task-specific layer
        """
        result = FusedBrain()
        
        if strategy != "tagged_conservative":
            raise ValueError(f"Only 'tagged_conservative' strategy supported. Got: {strategy}")
        
        # Step 1: Add global primitives
        for name, seed in GLOBAL_PRIMITIVES.items():
            hv = hypervec_rs.HyperVector(seed)
            result.global_codebook[name] = hv
            result.global_types[name] = infer_type(name)  # Store type!
            result.provenance[name] = ["PRIMITIVE"]
        print(f"Added {len(GLOBAL_PRIMITIVES)} global primitives")
        
        # Step 2: Process each brain
        all_tasks = list(self.registered_brains.keys())
        
        for task_tag, brain in self.registered_brains.items():
            for name, hv in brain.codebook.items():
                # Skip primitives (already in global)
                if self._is_global_primitive(name):
                    continue
                    
                # Add to task-specific layer with tag and type
                result.task_codebooks[task_tag][name] = hv
                result.task_types[task_tag][name] = brain.concept_types.get(name, ConceptType.UNKNOWN)
                
            # Add rules to task layer
            result.task_rules[task_tag] = list(brain.rules)
            
        # Step 3: Check for promotion candidates (concepts in multiple tasks)
        if len(all_tasks) >= 2:
            self._try_promotions(result, all_tasks)
            
        # Log summary
        print(f"Fusion complete:")
        print(f"  Global: {len(result.global_codebook)} concepts")
        for task, cb in result.task_codebooks.items():
            print(f"  {task}: {len(cb)} concepts")
            
        return result
    
    def _try_promotions(self, result: FusedBrain, tasks: List[str]):
        """
        Try to promote task-specific concepts to global if they:
        - Exist in multiple tasks
        - Have high similarity
        - Have high context similarity
        """
        promoted = []
        
        # Find concepts that appear in multiple tasks
        task_concepts = {t: set(result.task_codebooks[t].keys()) for t in tasks}
        
        for concept_name in task_concepts[tasks[0]]:
            appearances = [t for t in tasks if concept_name in task_concepts[t]]
            
            if len(appearances) < 2:
                continue
                
            # Check similarity across all appearances
            hvs = [result.task_codebooks[t][concept_name] for t in appearances]
            
            # Pairwise similarity check
            all_similar = True
            for i in range(len(hvs)):
                for j in range(i + 1, len(hvs)):
                    if hvs[i].similarity(hvs[j]) < self.HV_THRESHOLD:
                        all_similar = False
                        break
                        
            if all_similar:
                # Promote: bundle all versions into global
                merged_hv = hvs[0]
                for hv in hvs[1:]:
                    merged_hv = merged_hv.bundle(hv)  # Equal weight
                    
                result.global_codebook[concept_name] = merged_hv
                result.provenance[concept_name] = appearances
                
                # Remove from task layers
                for t in appearances:
                    del result.task_codebooks[t][concept_name]
                    
                promoted.append(concept_name)
                
        if promoted:
            print(f"Promoted {len(promoted)} concepts to global: {promoted}")
