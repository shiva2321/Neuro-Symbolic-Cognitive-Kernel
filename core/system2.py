"""
NCGN System 2 Controller - Deliberative Processing

Implements the intervention protocol:
1. Pause System 1
2. Schema Check
3. Diagnosis (Constraint Violation vs Missing Knowledge)
4. Intervention (Modify System 1)
5. Resume

System 2 is expensive - only activated when Surprise > Threshold.
CRITICAL: System 2 cannot directly emit actions. It acts by
modifying System 1's state.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .memory import GraphMemory, EventSchema, ConceptNode, Synapse


class DiagnosisType(Enum):
    """Types of anomalies System 2 can diagnose."""
    CONSTRAINT_VIOLATION = "constraint_violation"
    MISSING_KNOWLEDGE = "missing_knowledge"
    SCHEMA_NOT_FOUND = "schema_not_found"
    ROLE_MISMATCH = "role_mismatch"
    NO_ISSUE = "no_issue"


class ResponseAction(Enum):
    """Actions System 2 can take in response to diagnosis."""
    QUERY_USER = "query_user"
    ACQUIRE_PROPERTY = "acquire_property"
    APPLY_LTD = "apply_ltd"  # Long-Term Depression (weaken synapse)
    INJECT_GOAL = "inject_goal"
    NO_ACTION = "no_action"


@dataclass
class Triple:
    """An action triple: (Agent, Action, Object)"""
    agent: str
    action: str
    object: str
    
    def __repr__(self) -> str:
        return f"Triple({self.agent} -> {self.action} -> {self.object})"


@dataclass
class DiagnosisResult:
    """Result of System 2 diagnosis."""
    diagnosis_type: DiagnosisType
    violated_constraints: List[str]
    missing_properties: List[str]
    confidence: float
    explanation: str


@dataclass
class InterventionPlan:
    """Plan for System 2 intervention on System 1."""
    action: ResponseAction
    target_synapses: List[Tuple[str, str]]  # (source, target) pairs to modify
    energy_injections: Dict[str, float]     # node_id -> energy to inject
    goals: List[str]                        # Goals to create
    query: Optional[str]                    # Question to ask user


class System2Controller:
    """
    The deliberative processing controller.
    
    Expensive, sparse activation. Only operates when surprise
    exceeds threshold. Cannot directly output actions - must
    modify System 1 state.
    """
    
    # Learning parameters
    LTD_FACTOR = 0.5  # Factor to reduce synapse weight (Long-Term Depression)
    GOAL_INJECTION_ENERGY = 0.8  # Energy to inject for goal nodes
    
    def __init__(self, memory: GraphMemory, schemas_dir: Optional[str] = None):
        self.memory = memory
        self.schemas: Dict[str, EventSchema] = {}
        
        # Property knowledge base (what we know about objects)
        self.properties: Dict[str, Dict[str, bool]] = {}
        # e.g., {"meat": {"is_edible": True}, "metal": {"is_edible": False}}
        
        # Load schemas if directory provided
        if schemas_dir and os.path.exists(schemas_dir):
            self._load_schemas(schemas_dir)
        
        # State
        self.active_diagnosis: Optional[DiagnosisResult] = None
        self.active_intervention: Optional[InterventionPlan] = None
    
    def _load_schemas(self, schemas_dir: str):
        """Load all schemas from a directory."""
        for filename in os.listdir(schemas_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(schemas_dir, filename)
                try:
                    schema = EventSchema.from_json(filepath)
                    self.schemas[schema.action] = schema
                except Exception as e:
                    print(f"Warning: Could not load schema {filename}: {e}")
    
    def add_schema(self, schema: EventSchema):
        """Add a schema programmatically."""
        self.schemas[schema.action] = schema
    
    def set_property(self, object_id: str, property_name: str, value: bool):
        """Set a property for an object."""
        if object_id not in self.properties:
            self.properties[object_id] = {}
        self.properties[object_id][property_name] = value
    
    def get_property(self, object_id: str, property_name: str) -> Optional[bool]:
        """Get a property value. Returns None if unknown."""
        return self.properties.get(object_id, {}).get(property_name)
    
    # =====================
    # The Intervention Protocol
    # =====================
    
    def process_interrupt(
        self,
        triple: Triple,
        surprise_level: float,
        firing_set: Set[str]
    ) -> Tuple[DiagnosisResult, InterventionPlan]:
        """
        Main entry point for System 2 processing.
        
        Called when surprise exceeds threshold.
        
        Args:
            triple: The action being evaluated (agent -> action -> object)
            surprise_level: How surprising the observation was
            firing_set: Nodes that fired (context)
        
        Returns:
            Tuple of (diagnosis, intervention_plan)
        """
        # Step 1: Diagnose the issue
        diagnosis = self.diagnose(triple)
        self.active_diagnosis = diagnosis
        
        # Step 2: Plan intervention
        intervention = self.plan_intervention(diagnosis, triple, firing_set)
        self.active_intervention = intervention
        
        return diagnosis, intervention
    
    def diagnose(self, triple: Triple) -> DiagnosisResult:
        """
        Diagnose why an action is surprising.
        
        Checks:
        1. Does a schema exist for this action?
        2. Does the object satisfy the schema's constraints?
        3. Are there missing properties we need to query?
        """
        action = triple.action
        obj = triple.object
        
        # Check if we have a schema
        if action not in self.schemas:
            return DiagnosisResult(
                diagnosis_type=DiagnosisType.SCHEMA_NOT_FOUND,
                violated_constraints=[],
                missing_properties=[],
                confidence=0.5,
                explanation=f"No schema found for action '{action}'"
            )
        
        schema = self.schemas[action]
        violated = []
        missing = []
        
        # Check constraints on the object (target role)
        target_constraints = schema.constraints.get('target', [])
        
        for constraint in target_constraints:
            prop_value = self.get_property(obj, constraint)
            
            if prop_value is None:
                # Unknown property
                missing.append(constraint)
            elif constraint.startswith('not_'):
                # Negative constraint (e.g., "not_toxic")
                actual_prop = constraint[4:]  # Remove "not_"
                actual_value = self.get_property(obj, actual_prop)
                if actual_value is True:
                    violated.append(constraint)
            elif not prop_value:
                # Positive constraint not met
                violated.append(constraint)
        
        # Determine diagnosis type
        if violated:
            return DiagnosisResult(
                diagnosis_type=DiagnosisType.CONSTRAINT_VIOLATION,
                violated_constraints=violated,
                missing_properties=missing,
                confidence=schema.confidence,
                explanation=f"Object '{obj}' violates constraints: {violated}"
            )
        elif missing:
            return DiagnosisResult(
                diagnosis_type=DiagnosisType.MISSING_KNOWLEDGE,
                violated_constraints=[],
                missing_properties=missing,
                confidence=0.3,  # Low confidence due to unknowns
                explanation=f"Unknown properties for '{obj}': {missing}"
            )
        else:
            return DiagnosisResult(
                diagnosis_type=DiagnosisType.NO_ISSUE,
                violated_constraints=[],
                missing_properties=[],
                confidence=schema.confidence,
                explanation="No constraint violations detected"
            )
    
    def plan_intervention(
        self,
        diagnosis: DiagnosisResult,
        triple: Triple,
        context: Set[str]
    ) -> InterventionPlan:
        """
        Plan how to modify System 1 based on diagnosis.
        
        System 2 CANNOT directly emit actions. It must:
        - Modify synapse weights (learning)
        - Inject energy into goal nodes
        - Generate queries for clarification
        """
        if diagnosis.diagnosis_type == DiagnosisType.CONSTRAINT_VIOLATION:
            return self._plan_violation_response(diagnosis, triple, context)
        
        elif diagnosis.diagnosis_type == DiagnosisType.MISSING_KNOWLEDGE:
            return self._plan_knowledge_acquisition(diagnosis, triple)
        
        else:
            return InterventionPlan(
                action=ResponseAction.NO_ACTION,
                target_synapses=[],
                energy_injections={},
                goals=[],
                query=None
            )
    
    def _plan_violation_response(
        self,
        diagnosis: DiagnosisResult,
        triple: Triple,
        context: Set[str]
    ) -> InterventionPlan:
        """
        Plan response to a constraint violation.
        
        Strategy:
        1. Apply LTD to the synapse that led to this prediction
        2. Inject energy into "Query_User" action node
        """
        # Find synapses to depress
        target_synapses = []
        
        # The synapse from agent to action should be weakened in this context
        synapse = self.memory.get_synapse(triple.agent, triple.action)
        if synapse:
            target_synapses.append((triple.agent, triple.action))
        
        # Inject goal to query user
        energy_injections = {
            "goal_clarify_inconsistency": self.GOAL_INJECTION_ENERGY,
            "action_query_user": self.GOAL_INJECTION_ENERGY * 0.9
        }
        
        # Generate query
        query = f"Expected {triple.agent} to {triple.action} something edible, " \
                f"but '{triple.object}' is not edible. Is this correct?"
        
        return InterventionPlan(
            action=ResponseAction.QUERY_USER,
            target_synapses=target_synapses,
            energy_injections=energy_injections,
            goals=["clarify_inconsistency"],
            query=query
        )
    
    def _plan_knowledge_acquisition(
        self,
        diagnosis: DiagnosisResult,
        triple: Triple
    ) -> InterventionPlan:
        """
        Plan response to missing knowledge.
        
        Strategy:
        1. Inject energy into "Ask" action nodes
        2. Generate specific property query
        """
        missing = diagnosis.missing_properties[0] if diagnosis.missing_properties else "properties"
        
        energy_injections = {
            "goal_acquire_knowledge": self.GOAL_INJECTION_ENERGY,
            "action_ask": self.GOAL_INJECTION_ENERGY * 0.9
        }
        
        query = f"Is '{triple.object}' {missing.replace('_', ' ')}?"
        
        return InterventionPlan(
            action=ResponseAction.ACQUIRE_PROPERTY,
            target_synapses=[],
            energy_injections=energy_injections,
            goals=["acquire_property"],
            query=query
        )
    
    # =====================
    # Apply Intervention
    # =====================
    
    def apply_intervention(self, plan: InterventionPlan):
        """
        Apply the intervention to the memory/graph.
        
        This modifies System 1 state so that when it resumes,
        the appropriate actions will naturally emerge.
        """
        # Apply LTD (synaptic depression)
        for source, target in plan.target_synapses:
            synapse = self.memory.get_synapse(source, target)
            if synapse:
                synapse.weight *= self.LTD_FACTOR
        
        # Inject energy into goal/action nodes
        for node_id, energy in plan.energy_injections.items():
            if not self.memory.has_node(node_id):
                self.memory.add_node(node_id)
            
            node = self.memory.get_node(node_id)
            if node:
                node.energy = min(1.0, node.energy + energy)
                self.memory.mark_active(node_id)
    
    def get_response_query(self) -> Optional[str]:
        """Get the query to present to user (if any)."""
        if self.active_intervention:
            return self.active_intervention.query
        return None
    
    def get_diagnosis_summary(self) -> Optional[str]:
        """Get human-readable diagnosis summary."""
        if self.active_diagnosis:
            return self.active_diagnosis.explanation
        return None
