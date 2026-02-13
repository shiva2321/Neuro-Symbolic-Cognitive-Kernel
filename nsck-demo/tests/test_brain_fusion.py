"""
Test Brain Fusion
Verifies: cross-task contamination prevention, promotion correctness, query isolation
"""
import python.core.vsa.hypervec_shim as hypervec_rs
from python.brain_fusion import TaskBrain, BrainFusion, GLOBAL_PRIMITIVES, ConceptType


def test_global_primitives():
    """Test that global primitives are merged deterministically."""
    print("--- Test: Global Primitives ---")
    
    # Create two brains with same primitives (via GLOBAL_PRIMITIVES seeds)
    snake_brain = TaskBrain("snake")
    snake_brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
    snake_brain.add_concept("ACTION_DN", hypervec_rs.HyperVector(20))
    
    pong_brain = TaskBrain("pong")
    pong_brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
    pong_brain.add_concept("ACTION_DN", hypervec_rs.HyperVector(20))
    
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    fusion.register_brain(pong_brain)
    
    result = fusion.fuse()
    
    # Verify primitives are in global
    assert "ACTION_UP" in result.global_codebook, "ACTION_UP should be global"
    assert "ACTION_DN" in result.global_codebook, "ACTION_DN should be global"
    
    # Verify they are identical to seed
    expected_up = hypervec_rs.HyperVector(10)
    actual_up = result.global_codebook["ACTION_UP"]
    assert actual_up.similarity(expected_up) == 1.0, "Global primitive should be exact"
    
    # Verify types are stored
    assert result.global_types["ACTION_UP"] == ConceptType.ACTION
    
    print("✅ Global Primitives Test Passed")


def test_task_isolation():
    """Test that task-specific concepts remain isolated."""
    print("\n--- Test: Task Isolation ---")
    
    # Snake has FOOD concept
    snake_brain = TaskBrain("snake")
    snake_brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
    snake_brain.add_concept("FOOD_ABOVE", hypervec_rs.HyperVector(5001))
    
    # Pong has BALL concept
    pong_brain = TaskBrain("pong")
    pong_brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
    pong_brain.add_concept("BALL_ABOVE", hypervec_rs.HyperVector(6001))
    
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    fusion.register_brain(pong_brain)
    
    result = fusion.fuse()
    
    # Verify task-specific concepts are isolated
    assert "FOOD_ABOVE" in result.task_codebooks["snake"], "FOOD_ABOVE should be in snake layer"
    assert "BALL_ABOVE" in result.task_codebooks["pong"], "BALL_ABOVE should be in pong layer"
    
    # Verify they are NOT in global
    assert "FOOD_ABOVE" not in result.global_codebook
    assert "BALL_ABOVE" not in result.global_codebook
    
    # Verify types
    assert result.task_types["snake"]["FOOD_ABOVE"] == ConceptType.OBJECT
    
    print("✅ Task Isolation Test Passed")


def test_cross_task_contamination():
    """Test that snake-only concepts don't affect pong queries."""
    print("\n--- Test: Cross-Task Contamination Prevention ---")
    
    # Snake has a dangerous concept
    snake_brain = TaskBrain("snake")
    snake_brain.add_concept("DANGER_WALL", hypervec_rs.HyperVector(9999))
    
    # Pong has no such concept
    pong_brain = TaskBrain("pong")
    pong_brain.add_concept("PADDLE_POS", hypervec_rs.HyperVector(8888))
    
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    fusion.register_brain(pong_brain)
    
    result = fusion.fuse()
    
    # Query from pong context should NOT return snake concepts
    query = hypervec_rs.HyperVector(9999)  # Same as DANGER_WALL
    pong_results = result.query(query, task_tag="pong", top_k=3)
    
    # Top result should NOT be snake::DANGER_WALL
    for qr in pong_results:
        if "snake" in qr.concept_name:
            assert qr.similarity < 0.9, f"Snake concept {qr.concept_name} should not dominate pong query"
            
    print(f"  Query results (pong context): {[(qr.concept_name, qr.similarity) for qr in pong_results[:3]]}")
    print("✅ Cross-Task Contamination Prevention Test Passed")


def test_promotion():
    """Test that shared concepts promote to global."""
    print("\n--- Test: Promotion Mechanism ---")
    
    # Both tasks have WAYPOINT concept with same seed
    shared_seed = 7777
    
    snake_brain = TaskBrain("snake")
    snake_brain.add_concept("WAYPOINT", hypervec_rs.HyperVector(shared_seed))
    
    pong_brain = TaskBrain("pong")
    pong_brain.add_concept("WAYPOINT", hypervec_rs.HyperVector(shared_seed))
    
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    fusion.register_brain(pong_brain)
    
    result = fusion.fuse()
    
    # WAYPOINT should be promoted to global (identical in both)
    assert "WAYPOINT" in result.global_codebook, "WAYPOINT should be promoted to global"
    assert "WAYPOINT" not in result.task_codebooks.get("snake", {}), "WAYPOINT should be removed from snake"
    assert "WAYPOINT" not in result.task_codebooks.get("pong", {}), "WAYPOINT should be removed from pong"
    
    # Provenance should show both tasks
    assert result.provenance["WAYPOINT"] == ["snake", "pong"], f"Provenance wrong: {result.provenance['WAYPOINT']}"
    
    print("✅ Promotion Mechanism Test Passed")


def test_query_result_structure():
    """Test that QueryResult contains proper metadata."""
    print("\n--- Test: QueryResult Structure ---")
    
    snake_brain = TaskBrain("snake")
    snake_brain.add_concept("FOOD_TARGET", hypervec_rs.HyperVector(1111))
    
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    
    result = fusion.fuse()
    
    # Query
    query = hypervec_rs.HyperVector(1111)
    results = result.query(query, task_tag="snake", top_k=3)
    
    # Check QueryResult structure
    top = results[0]
    assert hasattr(top, 'concept_name'), "QueryResult should have concept_name"
    assert hasattr(top, 'similarity'), "QueryResult should have similarity"
    assert hasattr(top, 'layer'), "QueryResult should have layer"
    assert hasattr(top, 'concept_type'), "QueryResult should have concept_type"
    
    assert top.layer == "snake"
    assert top.similarity == 1.0
    print(f"  Top result: {top}")
    
    print("✅ QueryResult Structure Test Passed")


def test_type_consistency_gate():
    """Test that concepts of different types are NOT aligned."""
    print("\n--- Test: Type Consistency Gate ---")
    
    # Create concepts of different types with similar HVs (unrealistic but tests gate)
    snake_brain = TaskBrain("snake")
    snake_brain.add_concept("ACTION_JUMP", hypervec_rs.HyperVector(100))  # ACTION type
    
    pong_brain = TaskBrain("pong")
    pong_brain.add_concept("FOOD_JUMP", hypervec_rs.HyperVector(100))  # OBJECT type (same HV!)
    
    fusion = BrainFusion()
    fusion.register_brain(snake_brain)
    fusion.register_brain(pong_brain)
    
    # Check alignment - should NOT align despite identical HVs
    alignments = fusion.align_concepts(snake_brain, pong_brain)
    
    # Filter for this specific pair
    found = [a for a in alignments if "JUMP" in a[0] and "JUMP" in a[1]]
    
    assert len(found) == 0, f"Different types should not align, but got: {found}"
    
    print("✅ Type Consistency Gate Test Passed")


if __name__ == "__main__":
    test_global_primitives()
    test_task_isolation()
    test_cross_task_contamination()
    test_promotion()
    test_query_result_structure()
    test_type_consistency_gate()
    print("\n🎉 All Brain Fusion Tests Passed!")
