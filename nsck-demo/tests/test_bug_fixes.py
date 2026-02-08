"""
Test Bug Fixes
Verifies: fixes to existing modules (semantic_memory property binding,
brain_fusion forward_chain_multi, emotion decay, learning torch.stack,
world_model hv_to_numpy).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import hypervec_rs


def test_semantic_memory_property_binding():
    """Test that add_concept properly binds properties into HV."""
    print("--- Test: Semantic Memory Property Binding ---")

    from python.semantic_memory import SemanticMemory

    mem = SemanticMemory()

    # Add concept with properties
    mem.add_concept("apple", {"color": "red", "taste": "sweet"})
    assert "apple" in mem.concept_hvs, "apple should be in concept HVs"

    # Add concept without properties
    mem.add_concept("thing", {})
    assert "thing" in mem.concept_hvs, "thing should be in concept HVs"

    # Different properties should produce different HVs
    mem.add_concept("banana", {"color": "yellow", "taste": "sweet"})
    sim = mem.concept_hvs["apple"].similarity(mem.concept_hvs["banana"])
    assert sim < 0.99, f"Different properties should yield different HVs, sim={sim}"

    print(f"  apple-banana similarity: {sim:.4f}")
    print("✅ Semantic Memory Property Binding Test Passed")


def test_brain_fusion_forward_chain_no_mutation():
    """Test that forward_chain_multi does not mutate global_rules."""
    print("\n--- Test: Forward Chain No Mutation ---")

    from python.brain_fusion import FusedBrain, Rule

    brain = FusedBrain()

    # Add global rules
    brain.global_rules.append(Rule(
        condition=frozenset({"A"}),
        consequence="B",
        strength=1.0,
    ))
    original_count = len(brain.global_rules)

    # Add task rules
    brain.task_rules["test"] = [Rule(
        condition=frozenset({"B"}),
        consequence="C",
        strength=1.0,
    )]

    # Forward chain should not modify global_rules
    result, inferred = brain.forward_chain_multi({"A"})
    after_count = len(brain.global_rules)

    assert after_count == original_count, (
        f"global_rules mutated: before={original_count}, after={after_count}"
    )
    assert "C" in result, "Should infer C from A→B→C"
    print(f"  Inferred facts: {result}")

    print("✅ Forward Chain No Mutation Test Passed")


def test_emotion_arousal_decay():
    """Test that arousal decays smoothly instead of jumping."""
    print("\n--- Test: Emotion Arousal Decay ---")

    from python.emotion_system import EmotionSystem

    emo = EmotionSystem()

    # Set high arousal via high drives
    emo.update_from_drives({"hunger": 0.9, "pain": 0.1}, reward=0.0)
    high_arousal = emo.arousal

    # Now update with low drives — arousal should decay, not jump to 0
    emo.update_from_drives({"hunger": 0.1, "pain": 0.0}, reward=0.0)
    decayed_arousal = emo.arousal

    assert decayed_arousal > 0.0, "Arousal should not instantly drop to 0"
    assert decayed_arousal < high_arousal, "Arousal should decrease"
    print(f"  High: {high_arousal:.3f} → Decayed: {decayed_arousal:.3f}")

    print("✅ Emotion Arousal Decay Test Passed")


def test_brain_fusion_concept_context():
    """Test that get_concept_context returns a proper HV when rules exist."""
    print("\n--- Test: Concept Context ---")

    from python.brain_fusion import TaskBrain

    brain = TaskBrain("snake")
    brain.add_concept("FOOD_ABOVE", hypervec_rs.HyperVector(5001))
    brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
    brain.add_rule(
        frozenset({"FOOD_ABOVE"}),
        "ACTION_UP",
        strength=1.0,
    )

    ctx = brain.get_concept_context("FOOD_ABOVE")
    assert ctx is not None, "Should return a context HV when rules reference the concept"
    print(f"  Context HV: {ctx}")

    # Concept with no rules should return None
    brain.add_concept("UNUSED", hypervec_rs.HyperVector(9999))
    ctx_none = brain.get_concept_context("UNUSED")
    assert ctx_none is None, "Should return None for concept not in any rule"

    print("✅ Concept Context Test Passed")


def test_world_model_hv_to_numpy():
    """Test that hv_to_numpy works with both Python and Rust HVs."""
    print("\n--- Test: World Model hv_to_numpy ---")

    from python.world_model import WorldModel

    wm = WorldModel(hv_dim=10240)

    # Test with a HyperVector object
    hv = hypervec_rs.HyperVector(42)
    arr = wm.hv_to_numpy(hv)
    assert isinstance(arr, np.ndarray), "Should return ndarray"
    assert arr.dtype == np.float32, f"Should be float32, got {arr.dtype}"
    assert len(arr) > 0, "Should not be empty"

    # Test with numpy array
    np_arr = np.ones(10240, dtype=np.float64)
    result = wm.hv_to_numpy(np_arr)
    assert result.dtype == np.float32

    print(f"  HV → numpy shape: {arr.shape}, dtype: {arr.dtype}")

    print("✅ World Model hv_to_numpy Test Passed")


def test_episodic_memory_no_duplicate_import():
    """Verify the duplicate import was removed (regression test)."""
    print("\n--- Test: Episodic Memory No Duplicate Import ---")

    # Read the source file and count import lines
    ep_path = os.path.join(
        os.path.dirname(__file__), "..", "python", "episodic_memory.py"
    )
    with open(ep_path) as f:
        content = f.read()

    import_count = content.count("import hypervec_shim as hypervec_rs")
    assert import_count == 1, (
        f"Expected exactly 1 import of hypervec_shim, got {import_count}"
    )
    print(f"  hypervec_shim import count: {import_count}")

    print("✅ Episodic Memory No Duplicate Import Test Passed")


if __name__ == "__main__":
    test_semantic_memory_property_binding()
    test_brain_fusion_forward_chain_no_mutation()
    test_emotion_arousal_decay()
    test_brain_fusion_concept_context()
    test_world_model_hv_to_numpy()
    test_episodic_memory_no_duplicate_import()
    print("\n🎉 All Bug Fix Tests Passed!")
