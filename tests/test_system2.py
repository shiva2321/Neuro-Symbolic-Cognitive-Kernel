"""
NCGN System 2 Tests

Test the deliberative processing controller:
- Schema validation
- Constraint checking
- The "Dog Eat Metal" scenario
"""

import pytest
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import GraphMemory, EventSchema
from core.system1 import System1Engine
from core.system2 import (
    System2Controller, Triple, DiagnosisType, ResponseAction
)


class TestSchemaValidation(unittest.TestCase):
    """Test schema loading and constraint checking."""
    
    def test_schema_from_dict(self):
        """Test creating schema from dictionary."""
        data = {
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {"agent": "animate_object", "target": "edible_object"},
            "constraints": {"target": ["is_edible", "not_toxic"]}
        }
        
        schema = EventSchema.from_dict(data)
        
        assert schema.id == "schema_eat"
        assert schema.action == "eat"
        assert schema.confidence == 0.95
        assert "is_edible" in schema.constraints["target"]
    
    def test_property_management(self):
        """Test setting and getting properties."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        controller.set_property("meat", "is_edible", True)
        controller.set_property("metal", "is_edible", False)
        
        assert controller.get_property("meat", "is_edible") == True
        assert controller.get_property("metal", "is_edible") == False
        assert controller.get_property("unknown", "is_edible") is None


class TestDiagnosis(unittest.TestCase):
    """Test the diagnosis system."""
    
    def test_constraint_violation_detected(self):
        """Test detection of constraint violations."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        # Add eat schema
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {"agent": "animate_object", "target": "edible_object"},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        # Metal is NOT edible
        controller.set_property("metal", "is_edible", False)
        
        # Diagnose: Dog eat Metal
        triple = Triple(agent="dog", action="eat", object="metal")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type == DiagnosisType.CONSTRAINT_VIOLATION
        assert "is_edible" in diagnosis.violated_constraints
    
    def test_missing_knowledge_detected(self):
        """Test detection of missing knowledge."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        # We don't know if "mystery_food" is edible
        triple = Triple(agent="dog", action="eat", object="mystery_food")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type == DiagnosisType.MISSING_KNOWLEDGE
        assert "is_edible" in diagnosis.missing_properties
    
    def test_no_issue_when_constraints_met(self):
        """Test that valid actions pass."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        controller.set_property("meat", "is_edible", True)
        
        triple = Triple(agent="dog", action="eat", object="meat")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type == DiagnosisType.NO_ISSUE


class TestInterventionPlanning(unittest.TestCase):
    """Test intervention plan generation."""
    
    def test_violation_generates_query(self):
        """Test that violations lead to user queries."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        controller.set_property("metal", "is_edible", False)
        
        triple = Triple(agent="dog", action="eat", object="metal")
        diagnosis, intervention = controller.process_interrupt(
            triple, surprise_level=0.8, firing_set={"dog", "eat"}
        )
        
        assert intervention.action == ResponseAction.QUERY_USER
        assert intervention.query is not None
        assert "metal" in intervention.query.lower()
    
    def test_missing_knowledge_asks_question(self):
        """Test that missing knowledge leads to property query."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        triple = Triple(agent="dog", action="eat", object="mystery")
        diagnosis, intervention = controller.process_interrupt(
            triple, surprise_level=0.5, firing_set=set()
        )
        
        assert intervention.action == ResponseAction.ACQUIRE_PROPERTY
        assert "edible" in intervention.query.lower()


class TestDogEatMetal(unittest.TestCase):
    """
    The Definition of Done: The "Dog Eat Metal" Unit Test.
    
    1. Prediction: System expects "Meat" (High Confidence)
    2. Observation: Receives "Metal"
    3. Surprise: Triggers high-priority interrupt
    4. Diagnosis: Identifies Schema Violation (Metal ≠ Edible)
    5. Response: Queries for clarification
    """
    
    def test_dog_eat_metal_full_scenario(self):
        """
        Complete end-to-end test of the Dog Eat Metal scenario.
        """
        # ===== SETUP =====
        memory = GraphMemory()
        engine = System1Engine(
            memory, 
            k_winners=10, 
            surprise_threshold=0.3,
            decay_alpha=0.95
        )
        controller = System2Controller(memory)
        
        # Add eat schema
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {"agent": "animate_object", "target": "edible_object"},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        # Set up knowledge base
        controller.set_property("meat", "is_edible", True)
        controller.set_property("metal", "is_edible", False)
        
        # Create nodes
        memory.add_node("dog", threshold=0.5)
        memory.add_node("meat", threshold=0.5, novelty_score=0.1)  # Familiar
        memory.add_node("metal", threshold=0.5, novelty_score=0.1)  # Also familiar
        memory.add_node("eat", threshold=0.5)
        
        # Dog -> eats -> Meat (high confidence expectation)
        memory.add_synapse("dog", "eat", type="can", weight=0.9, confidence=0.9)
        memory.add_synapse("dog", "meat", type="eats", weight=0.9, confidence=0.9)
        
        # ===== STEP 1: PREDICTION =====
        # Dog is activated, expects to eat meat
        engine.inject_energy("dog", 1.0)
        engine.tick()
        
        # Verify: Meat should be activated (prediction)
        meat_node = memory.get_node("meat")
        assert meat_node.energy > 0 or meat_node.last_spike_tick >= 0, \
            "Meat should be activated as prediction"
        
        # ===== STEP 2: OBSERVATION =====
        # Instead of meat, we observe metal
        # First, suppress meat to simulate wrong observation
        meat_node.energy = 0.0
        memory.mark_inactive("meat")
        
        # Inject metal
        engine.inject_energy("metal", 1.0)
        engine.tick()
        
        # ===== STEP 3: SURPRISE =====
        # Check that surprise was elevated
        assert engine.surprise_level > 0 or engine.system2_triggered, \
            "System should register surprise or trigger System 2"
        
        # ===== STEP 4: DIAGNOSIS =====
        triple = Triple(agent="dog", action="eat", object="metal")
        diagnosis, intervention = controller.process_interrupt(
            triple,
            surprise_level=0.8,
            firing_set=engine.get_firing_set()
        )
        
        assert diagnosis.diagnosis_type == DiagnosisType.CONSTRAINT_VIOLATION, \
            f"Should detect violation, got {diagnosis.diagnosis_type}"
        assert "is_edible" in diagnosis.violated_constraints, \
            "Should identify 'is_edible' as violated constraint"
        
        # ===== STEP 5: RESPONSE =====
        assert intervention.action == ResponseAction.QUERY_USER, \
            "Should respond with query to user"
        assert intervention.query is not None, \
            "Should generate a clarification query"
        
        # Verify query is sensible
        query = intervention.query.lower()
        assert "metal" in query or "edible" in query, \
            f"Query should mention metal or edibility: {intervention.query}"
        
        print(f"\n✓ Dog Eat Metal Test PASSED")
        print(f"  Diagnosis: {diagnosis.explanation}")
        print(f"  Query: {intervention.query}")
    
    def test_dog_eat_meat_no_issue(self):
        """
        Sanity check: Dog eating meat should NOT trigger issues.
        """
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        controller.set_property("meat", "is_edible", True)
        
        triple = Triple(agent="dog", action="eat", object="meat")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type == DiagnosisType.NO_ISSUE, \
            "Dog eating meat should be fine"


class TestInterventionApplication(unittest.TestCase):
    """Test applying interventions to System 1."""
    
    def test_apply_ltd(self):
        """Test that LTD (Long-Term Depression) weakens synapses."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        memory.add_node("dog")
        memory.add_node("eat")
        synapse = memory.add_synapse("dog", "eat", weight=1.0, confidence=0.9)
        
        original_weight = synapse.weight
        
        from core.system2 import InterventionPlan
        plan = InterventionPlan(
            action=ResponseAction.APPLY_LTD,
            target_synapses=[("dog", "eat")],
            energy_injections={},
            goals=[],
            query=None
        )
        
        controller.apply_intervention(plan)
        
        assert synapse.weight < original_weight, \
            "LTD should reduce synapse weight"
    
    def test_energy_injection(self):
        """Test that interventions can inject energy."""
        memory = GraphMemory()
        controller = System2Controller(memory)
        
        from core.system2 import InterventionPlan
        plan = InterventionPlan(
            action=ResponseAction.INJECT_GOAL,
            target_synapses=[],
            energy_injections={"goal_clarify": 0.8},
            goals=["clarify"],
            query=None
        )
        
        controller.apply_intervention(plan)
        
        goal_node = memory.get_node("goal_clarify")
        assert goal_node is not None, "Goal node should be created"
        assert goal_node.energy > 0, "Goal node should have energy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
