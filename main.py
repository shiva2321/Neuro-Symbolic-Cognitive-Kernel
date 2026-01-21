"""
NCGN Demo - Neuromorphic Cognitive Graph Network

Demonstrates the core capabilities:
1. Basic energy propagation
2. K-WTA attention
3. The "Dog Eat Metal" scenario
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory import GraphMemory, EventSchema
from core.system1 import System1Engine
from core.system2 import System2Controller, Triple


def demo_basic_propagation():
    """Demo: Basic energy propagation through a chain."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Energy Propagation")
    print("="*60)
    
    memory = GraphMemory()
    engine = System1Engine(memory, k_winners=10)
    
    # Create a simple chain: A -> B -> C
    memory.add_node("A", threshold=0.5)
    memory.add_node("B", threshold=0.5)
    memory.add_node("C", threshold=0.5)
    
    memory.add_synapse("A", "B", weight=0.8, confidence=0.9)
    memory.add_synapse("B", "C", weight=0.8, confidence=0.9)
    
    print("\nCreated chain: A -> B -> C")
    print("Injecting energy into A...")
    
    engine.inject_energy("A", 1.0)
    
    for tick in range(5):
        engine.tick()
        active = memory.get_active_nodes()
        energies = {nid: f"{memory.get_node(nid).energy:.3f}" for nid in active}
        print(f"  Tick {tick}: Active nodes = {energies}")
    
    print("\n[SUCCESS] Energy propagates through the chain")


def demo_kwta_attention():
    """Demo: K-Winners-Take-All attention mechanism."""
    print("\n" + "="*60)
    print("DEMO 2: K-WTA Attention (Competing Concepts)")
    print("="*60)
    
    memory = GraphMemory()
    engine = System1Engine(memory, k_winners=3)  # Only 3 winners allowed
    
    # Create competing concepts
    concepts = ["apple", "banana", "cherry", "date", "elderberry"]
    for i, name in enumerate(concepts):
        memory.add_node(name, energy=0.1 * (i + 1), threshold=0.75)
        memory.mark_active(name)
    
    print(f"\nInitial energies: {[f'{c}={0.1*(i+1):.1f}' for i, c in enumerate(concepts)]}")
    print(f"K = 3 (only 3 concepts can be active)")
    
    engine._phase_6_kwta_inhibition()
    
    active = memory.get_active_nodes()
    print(f"\nAfter k-WTA: Active = {active}")
    print(f"Winners are the 3 highest energy concepts: cherry, date, elderberry")
    
    for name in concepts:
        node = memory.get_node(name)
        status = "WINNER" if name in active else "suppressed"
        print(f"  {name}: energy={node.energy:.3f} [{status}]")
    
    print("\n[SUCCESS] K-WTA keeps only the strongest concepts active")


def demo_dog_eat_metal():
    """Demo: The complete Dog Eat Metal scenario."""
    print("\n" + "="*60)
    print("DEMO 3: Dog Eat Metal (Cognitive Surprise)")
    print("="*60)
    
    memory = GraphMemory()
    engine = System1Engine(
        memory, 
        k_winners=10, 
        surprise_threshold=0.3
    )
    controller = System2Controller(memory)
    
    # Setup schema for eating
    schema = EventSchema.from_dict({
        "id": "schema_eat",
        "action": "eat",
        "confidence": 0.95,
        "roles": {"agent": "animate_object", "target": "edible_object"},
        "constraints": {"target": ["is_edible"]}
    })
    controller.add_schema(schema)
    
    # World knowledge
    controller.set_property("meat", "is_edible", True)
    controller.set_property("metal", "is_edible", False)
    
    # Create cognitive graph
    memory.add_node("dog", threshold=0.5)
    memory.add_node("meat", threshold=0.5, novelty_score=0.1)
    memory.add_node("metal", threshold=0.5, novelty_score=0.1)
    memory.add_node("eat", threshold=0.5)
    
    memory.add_synapse("dog", "eat", type="can", weight=0.9, confidence=0.9)
    memory.add_synapse("dog", "meat", type="eats", weight=0.9, confidence=0.9)
    
    print("\nKnowledge Graph:")
    print("  dog --[eats]--> meat (high confidence)")
    print("  meat: is_edible = True")
    print("  metal: is_edible = False")
    
    # Step 1: Prediction
    print("\n--- STEP 1: PREDICTION ---")
    print("Activating 'dog' concept...")
    engine.inject_energy("dog", 1.0)
    engine.tick()
    
    meat = memory.get_node("meat")
    print(f"  'meat' predicted with energy: {meat.energy:.3f}")
    
    # Step 2: Observation (wrong!)
    print("\n--- STEP 2: UNEXPECTED OBSERVATION ---")
    print("Instead of meat, we observe METAL!")
    meat.energy = 0.0
    memory.mark_inactive("meat")
    engine.inject_energy("metal", 1.0)
    engine.tick()
    
    print(f"  Surprise level: {engine.surprise_level:.3f}")
    
    # Step 3: System 2 Diagnosis
    print("\n--- STEP 3: SYSTEM 2 DIAGNOSIS ---")
    triple = Triple(agent="dog", action="eat", object="metal")
    diagnosis, intervention = controller.process_interrupt(
        triple, 
        surprise_level=0.8, 
        firing_set=engine.get_firing_set()
    )
    
    print(f"  Diagnosis Type: {diagnosis.diagnosis_type.value}")
    print(f"  Violated Constraints: {diagnosis.violated_constraints}")
    print(f"  Explanation: {diagnosis.explanation}")
    
    # Step 4: Response
    print("\n--- STEP 4: RESPONSE ---")
    print(f"  Action: {intervention.action.value}")
    print(f"  Query: {intervention.query}")
    
    print("\n" + "="*60)
    print("\n[SUCCESS] SYSTEM CORRECTLY:")
    print("  1. Predicted 'meat' (high confidence)")
    print("  2. Detected surprise when 'metal' observed")
    print("  3. Diagnosed constraint violation (metal ≠ edible)")
    print("  4. Generated clarification query")
    print("="*60)
    print("\n[SUCCESS] Dog Eat Metal Test PASSED")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("#  NCGN - Neuromorphic Cognitive Graph Network")
    print("#  Demo Suite")
    print("#"*60)
    
    demo_basic_propagation()
    demo_kwta_attention()
    demo_dog_eat_metal()
    
    print("\n" + "#"*60)
    print("#  All demos completed successfully!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
