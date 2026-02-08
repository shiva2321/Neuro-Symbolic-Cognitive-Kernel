"""
Test Metacognitive Engine
Verifies: confidence scoring, conflict detection, tiered escalation, safe fallback
"""
import hypervec_shim as hypervec_rs
from python.brain_fusion import TaskBrain, BrainFusion, FusedBrain, ConceptType
from python.metacognition import MetacognitiveEngine, EscalationRequest, snake_safe_fallback

def test_confidence_scoring():
    print("\n--- Test: Confidence Scoring ---")
    
    # Setup brain with ambiguity
    # Create two concepts that are extremely similar (Sim > 0.95)
    base = hypervec_rs.HyperVector(100)
    
    # Target 1: Base vector
    hv1 = base
    
    # Target 2: Base slightly perturbed (95% similar)
    noise = hypervec_rs.HyperVector(200)
    hv2 = base.weighted_bundle(noise, 0.85) 
    
    brain = TaskBrain("snake")
    brain.add_concept("TEST_UP", hv1, concept_type=ConceptType.ACTION)
    brain.add_concept("TEST_DOWN", hv2, concept_type=ConceptType.ACTION)
    
    fusion = BrainFusion()
    fusion.register_brain(brain)
    fused_brain = fusion.fuse()
    
    meta = MetacognitiveEngine(fused_brain)
    
    # Query with 'hv1' -> sim(hv1)=1.0, sim(hv2)~0.85
    # Gap ~0.15. Spread small.
    # Confidence should be impacted by low margin/spread.
    results = fused_brain.query(hv1, task_tag="snake")
    confidence, reason = meta.compute_confidence(results)
    
    print(f"Ambiguity Test: Confidence={confidence:.4f}, Reason={reason}")
    print(f"  Gap: {results[0].similarity - results[1].similarity:.4f}")
    
    # User's fix: if spread is low (flat), confidence should be penalized.
    # Here sim(hv1, hv1)=1.0, sim(hv1, hv2)=high. Gap is small.
    # Confidence should be relatively low due to margin/spread.
    
    assert reason in ["competition", "ambiguity"], f"Reason should reflect uncertain choice, got {reason}"
    assert confidence < 0.9, "Confidence should be penalized for close results"
    
    # Test No Match
    random_hv = hypervec_rs.HyperVector(999)
    results_nm = fused_brain.query(random_hv, task_tag="snake")
    conf_nm, reason_nm = meta.compute_confidence(results_nm)
    print(f"No Match Test: Confidence={conf_nm}, Reason={reason_nm}")
    
    assert conf_nm == 0.0, "Confidence should be 0 for no match"
    assert reason_nm == "no_match", "Reason should be no_match"
    
    print("✅ Confidence Scoring Test Passed")


def test_conflict_detection():
    print("\n--- Test: Conflict Detection ---")
    
    # Setup precedence conflict: Global vs Task with DIFFERENT actions
    fusion = BrainFusion()
    
    # Global says UP
    fused_brain = FusedBrain()
    hv_common = hypervec_rs.HyperVector(100)
    
    # Global primitive
    fused_brain.global_codebook["ACTION_UP"] = hv_common
    fused_brain.global_types["ACTION_UP"] = ConceptType.ACTION
    
    # Task specific override says DOWN (using same HV to force conflict)
    # Currently fusion doesn't allow overwrite like this usually, but we can manually inject in FusedBrain
    fused_brain.task_codebooks["snake"]["ACTION_DOWN"] = hv_common
    fused_brain.task_types["snake"]["ACTION_DOWN"] = ConceptType.ACTION
    
    meta = MetacognitiveEngine(fused_brain)
    
    # Query
    results = fused_brain.query(hv_common, task_tag="snake")
    # Top 2 should be snake::ACTION_DOWN (sim 1.0) and ACTION_UP (sim 1.0)
    
    conflict = meta.detect_conflict(results)
    
    assert conflict is not None, "Should detect conflict"
    assert conflict.type == "precedence", f"Should be precedence conflict, got {conflict.type}"
    assert len(conflict.candidates) == 2
    
    print(f"Conflict Detected: {conflict.type}, Severity={conflict.severity:.4f}")
    print("✅ Conflict Detection Test Passed")


def test_snake_safe_fallback():
    print("\n--- Test: Snake Safe Fallback ---")
    
    # Grid size 10. 
    # Head at (5,5). Neck at (5,6) -> Heading UP.
    state = {
        "head": (5, 5),
        "body": [(5, 6), (5, 7)] 
    }
    
    # 1. Last action VALID (LEFT)
    act = snake_safe_fallback(state, "ACTION_LEFT")
    assert act == "ACTION_LEFT", f"Should allow LEFT, got {act}"
    
    # 2. Last action INVALID (DOWN - 180 reverse)
    # Heading is UP. Opposite is DOWN.
    act = snake_safe_fallback(state, "ACTION_DOWN") 
    assert act == "ACTION_UP", f"Should fallback to current heading (UP), got {act}"
    
    # 3. Last action VALID (UP - continue)
    act = snake_safe_fallback(state, "ACTION_UP")
    assert act == "ACTION_UP"
    
    print("✅ Snake Safe Fallback Test Passed")


def test_escalation_logic():
    print("\n--- Test: Escalation Logic ---")
    
    brain = FusedBrain()
    meta = MetacognitiveEngine(brain)
    
    # Mocking low confidence scenario
    # We can inject a mock infer result by controlling the brain contents
    # But easier to just test infer() logic behavior with a constructed scenario
    
    # Let's use real components
    hv = hypervec_rs.HyperVector(555)
    brain.task_codebooks["snake"]["ACTION_LEFT"] = hv # Single concept
    
    # 1. High Confidence -> ALLOW
    res = meta.infer(hv, "snake", {"head":(0,0)}, "ACTION_UP")
    assert res.action == "ACTION_LEFT"
    assert res.escalation is None
    assert res.confidence > 0.8
    
    # 2. No Match -> ALLOW_SAFE_FALLBACK (Low conf)
    hv_nomatch = hypervec_rs.HyperVector(999)
    res = meta.infer(hv_nomatch, "snake", {"head":(5,5), "body":[(5,6)]}, "ACTION_UP")
    
    assert res.confidence == 0.0
    assert res.escalation is not None
    assert res.escalation.severity == "low"
    assert res.action == "ACTION_UP" # Fallback used (UP is current heading)
    
    print("✅ Escalation Logic Test Passed")


def test_cycle_detection():
    print("\n--- Test: Cycle Detection ---")
    
    brain = FusedBrain()
    hv = hypervec_rs.HyperVector(777)
    brain.task_codebooks["snake"]["ACTION_RIGHT"] = hv
    
    meta = MetacognitiveEngine(brain)
    state = {"head":(0,0), "body":[(0,1)]} # Heading UP
    
    # 1st call - OK
    res1 = meta.infer(hv, "snake", state, "ACTION_UP")
    assert not res1.trace["cycle"]
    
    # 2nd call - Duplicate - Should trigger cycle
    res2 = meta.infer(hv, "snake", state, "ACTION_UP")
    assert res2.trace["cycle"] == True
    assert res2.uncertainty_reason == "cycle_detected"
    assert res2.action == "ACTION_UP" # Fallback used
    
    print("✅ Cycle Detection Test Passed")


if __name__ == "__main__":
    test_confidence_scoring()
    test_conflict_detection()
    test_snake_safe_fallback()
    test_escalation_logic()
    test_cycle_detection()
    print("\n🎉 All Metacognition Tests Passed!")
