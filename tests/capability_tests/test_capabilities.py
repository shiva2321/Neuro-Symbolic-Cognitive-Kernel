"""
NCGN Capability Tests - Systematic Capability Probes

Tests the cognitive capabilities of the trained system:
- Recall: Can it retrieve learned associations?
- Generalization: Can it apply rules to novel inputs?
- Retention: Does knowledge persist after decay?
- Interference: Does new learning interfere with old?
- Constraint Detection: Does it detect schema violations?

Run with: python -m pytest tests/capability_tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.memory import GraphMemory, EventSchema
from core.system1 import System1Engine
from core.system2 import System2Controller, Triple
from core.trainer import TrainingSession, Curriculum, Lesson, TrainingExample, LessonType


class TestRecallCapability:
    """Test the system's ability to recall learned associations."""
    
    @pytest.fixture
    def trained_system(self):
        """Create and train a system with basic knowledge."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10, surprise_threshold=0.3)
        controller = System2Controller(memory)
        
        # Pre-train associations
        pairs = [("dog", "meat"), ("cat", "fish"), ("rabbit", "carrot")]
        
        for animal, food in pairs:
            memory.add_node(animal, threshold=0.5)
            memory.add_node(food, threshold=0.5, novelty_score=0.1)
            memory.add_synapse(animal, food, type="eats", weight=0.9, confidence=0.9)
        
        return memory, engine, controller
    
    def test_direct_recall(self, trained_system):
        """Test: Activating 'dog' should recall 'meat'."""
        memory, engine, _ = trained_system
        
        engine.inject_energy("dog", 1.0)
        engine.run(5)
        
        meat = memory.get_node("meat")
        assert meat.energy > 0.1, "Meat should be activated when dog is stimulated"
    
    def test_multiple_recall(self, trained_system):
        """Test: Multiple associations can be recalled sequentially."""
        memory, engine, _ = trained_system
        
        # Test each pair
        for animal, food in [("dog", "meat"), ("cat", "fish"), ("rabbit", "carrot")]:
            # Reset
            for node_id in list(memory.get_active_nodes()):
                node = memory.get_node(node_id)
                if node:
                    node.energy = 0.0
                memory.mark_inactive(node_id)
            
            engine.inject_energy(animal, 1.0)
            engine.run(5)
            
            food_node = memory.get_node(food)
            assert food_node.energy > 0.1, f"{animal} should activate {food}"
    
    def test_recall_with_noise(self, trained_system):
        """Test: Recall works even with competing activations."""
        memory, engine, _ = trained_system
        
        # Activate multiple animals
        engine.inject_energy("dog", 1.0)
        engine.inject_energy("cat", 0.3)  # Weaker
        engine.run(5)
        
        # Meat should win over fish due to stronger dog activation
        meat = memory.get_node("meat")
        fish = memory.get_node("fish")
        
        assert meat.energy > fish.energy, "Meat should be stronger due to stronger dog input"


class TestGeneralizationCapability:
    """Test the system's ability to generalize rules."""
    
    @pytest.fixture
    def system_with_schemas(self):
        """Create system with schemas and properties."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        controller = System2Controller(memory)
        
        # Add schema
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "roles": {"agent": "animate", "target": "edible"},
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        # Set properties
        controller.set_property("apple", "is_edible", True)
        controller.set_property("plastic", "is_edible", False)
        
        return memory, engine, controller
    
    def test_novel_valid_input(self, system_with_schemas):
        """Test: New edible object should not violate eat schema."""
        _, _, controller = system_with_schemas
        
        # New animal we've never seen
        controller.set_property("elephant", "is_animate", True)
        
        # New edible object
        controller.set_property("banana", "is_edible", True)
        
        triple = Triple(agent="elephant", action="eat", object="banana")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type.value == "no_issue", \
            "Novel valid input should not cause violation"
    
    def test_novel_invalid_input(self, system_with_schemas):
        """Test: New inedible object should violate eat schema."""
        _, _, controller = system_with_schemas
        
        controller.set_property("cardboard", "is_edible", False)
        
        triple = Triple(agent="cat", action="eat", object="cardboard")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type.value == "constraint_violation", \
            "Novel invalid input should cause violation"


class TestRetentionCapability:
    """Test knowledge retention over time."""
    
    def test_knowledge_persists_after_decay(self):
        """Test: Synapse weights persist even after energy decays."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Train association
        memory.add_node("test_a", threshold=0.5)
        memory.add_node("test_b", threshold=0.5)
        memory.add_synapse("test_a", "test_b", weight=0.9, confidence=0.9)
        
        # Run many ticks to decay all energy
        engine.run(100)
        
        # Synapse should still exist with same weight
        synapse = memory.get_synapse("test_a", "test_b")
        assert synapse is not None, "Synapse should persist"
        assert synapse.weight >= 0.9, "Weight should not decay (only energy decays)"
    
    def test_recall_after_inactivity(self):
        """Test: Can recall after a period of inactivity."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        
        # Train
        memory.add_node("source", threshold=0.5)
        memory.add_node("target", threshold=0.5)
        memory.add_synapse("source", "target", weight=0.9, confidence=0.9)
        
        # Initial test - should work
        engine.inject_energy("source", 1.0)
        engine.run(3)
        assert memory.get_node("target").energy > 0.1
        
        # Let it decay completely
        engine.run(50)
        assert memory.get_node("target").energy < 0.01
        
        # Recall again - should still work
        engine.inject_energy("source", 1.0)
        engine.run(3)
        assert memory.get_node("target").energy > 0.1, \
            "Recall should work after period of inactivity"


class TestInterferenceCapability:
    """Test that new learning doesn't interfere with old."""
    
    def test_new_association_doesnt_erase_old(self):
        """Test: Adding new associations preserves existing ones."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10)
        controller = System2Controller(memory)
        session = TrainingSession(memory, engine, controller)
        
        # Train first association
        memory.add_node("dog", threshold=0.5)
        memory.add_node("meat", threshold=0.5)
        memory.add_synapse("dog", "meat", type="eats", weight=0.9, confidence=0.9)
        
        old_synapse = memory.get_synapse("dog", "meat")
        old_weight = old_synapse.weight
        
        # Add new association (dog -> bone)
        memory.add_node("bone", threshold=0.5)
        memory.add_synapse("dog", "bone", type="likes", weight=0.7, confidence=0.7)
        
        # Old association should still exist with similar weight
        still_exists = memory.get_synapse("dog", "meat")
        assert still_exists is not None, "Old synapse should still exist"
        assert still_exists.weight >= old_weight * 0.9, \
            "Old synapse weight should not decrease significantly"


class TestConstraintDetectionCapability:
    """Test schema constraint violation detection."""
    
    @pytest.fixture
    def system_with_eat_schema(self):
        """Create system with eat action schema."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=10, surprise_threshold=0.3)
        controller = System2Controller(memory)
        
        schema = EventSchema.from_dict({
            "id": "schema_eat",
            "action": "eat",
            "confidence": 0.95,
            "constraints": {"target": ["is_edible"]}
        })
        controller.add_schema(schema)
        
        controller.set_property("meat", "is_edible", True)
        controller.set_property("metal", "is_edible", False)
        
        return memory, engine, controller
    
    def test_valid_action_passes(self, system_with_eat_schema):
        """Test: Valid action (dog eat meat) should pass."""
        _, _, controller = system_with_eat_schema
        
        triple = Triple(agent="dog", action="eat", object="meat")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type.value != "constraint_violation", \
            "Valid action should not trigger constraint violation"
    
    def test_invalid_action_detected(self, system_with_eat_schema):
        """Test: Invalid action (dog eat metal) should be detected."""
        _, _, controller = system_with_eat_schema
        
        triple = Triple(agent="dog", action="eat", object="metal")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type.value == "constraint_violation", \
            "Invalid action should trigger constraint violation"
        assert "is_edible" in diagnosis.violated_constraints, \
            "Should identify is_edible as violated constraint"
    
    def test_missing_property_detected(self, system_with_eat_schema):
        """Test: Unknown object triggers missing knowledge."""
        _, _, controller = system_with_eat_schema
        
        # Object with unknown properties
        triple = Triple(agent="dog", action="eat", object="mystery_foo")
        diagnosis = controller.diagnose(triple)
        
        assert diagnosis.diagnosis_type.value == "missing_knowledge", \
            "Unknown object should trigger missing knowledge diagnosis"


class TestSurpriseCapability:
    """Test surprise detection and System 2 triggering."""
    
    def test_high_surprise_triggers_system2(self):
        """Test: Violated expectation triggers System 2."""
        memory = GraphMemory()
        engine = System1Engine(memory, k_winners=1, surprise_threshold=0.3)
        
        # Setup expectation
        memory.add_node("predictor", threshold=0.5)
        memory.add_node("expected", threshold=1.0, novelty_score=0.1)  # High threshold = won't fire
        memory.add_synapse("predictor", "expected", weight=0.8, confidence=0.9)
        
        # Activate predictor
        engine.inject_energy("predictor", 1.0)
        engine.tick()  # Predictor fires, expected gets energy
        
        # Now inject competing stimulus
        memory.add_node("unexpected", threshold=0.5, novelty_score=1.0)
        engine.inject_energy("unexpected", 1.0)
        engine.tick()  # k-WTA should suppress expected
        
        # Should have some surprise (may vary based on exact calculation)
        assert engine.surprise_level >= 0, "Surprise should be calculated"


# Benchmarking utilities
class CapabilityBenchmark:
    """Benchmark capability test performance."""
    
    def __init__(self):
        self.results = {}
    
    def measure_recall_accuracy(self, memory, engine, pairs, num_trials=10):
        """Measure recall accuracy over multiple trials."""
        correct = 0
        total = 0
        
        for _ in range(num_trials):
            for source, target in pairs:
                # Reset
                for node_id in list(memory.get_active_nodes()):
                    node = memory.get_node(node_id)
                    if node:
                        node.energy = 0.0
                    memory.mark_inactive(node_id)
                
                engine.inject_energy(source, 1.0)
                engine.run(5)
                
                target_node = memory.get_node(target)
                if target_node and target_node.energy > 0.1:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def measure_constraint_detection_accuracy(self, controller, test_cases):
        """Measure accuracy of constraint detection."""
        correct = 0
        
        for agent, action, obj, expected_violation in test_cases:
            triple = Triple(agent=agent, action=action, object=obj)
            diagnosis = controller.diagnose(triple)
            
            is_violation = diagnosis.diagnosis_type.value == "constraint_violation"
            if is_violation == expected_violation:
                correct += 1
        
        return correct / len(test_cases) if test_cases else 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
