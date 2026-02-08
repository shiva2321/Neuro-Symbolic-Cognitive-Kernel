import hypervec_shim as hypervec_rs
from python.staged_recall import StagedRecall
from python.lifecycle import LifecycleManager

def test_split():
    print("--- Test: Concept Split ---")
    
    # 1. Create Base Concept
    hv_broad = hypervec_rs.HyperVector(500)
    
    # 2. Setup Staged Recall
    codebook = {
        "fruit": hv_broad
    }
    
    sr = StagedRecall(codebook, cache_size=10)
    lm = LifecycleManager(sr)
    
    # Pre-load cache
    sr.query(hv_broad)
    
    # 3. Split Concept
    print("Splitting 'fruit' -> ['apple', 'pear']")
    new_hvs = lm.split_concept("fruit", ["apple", "pear"])
    
    assert len(new_hvs) == 2, "Should return 2 new HVs"
    
    # 4. Verify original is gone
    assert "fruit" not in sr.codebook, "Original concept should be deleted"
    assert "fruit" not in sr.working_memory, "Original concept should be removed from cache"
    
    # 5. Verify new concepts exist
    assert "apple" in sr.codebook
    assert "pear" in sr.codebook
    
    hv_apple = sr.codebook["apple"]
    hv_pear = sr.codebook["pear"]
    
    # 6. Verify Genealogy (Semantic Inheritance)
    # Apple should be similar to "fruit" (which was hv_broad)
    # Logic: Apple = Fruit + Fruit + Noise. Sim should be ~0.83 (5/6?) or high.
    sim_ancestry = hv_apple.similarity(hv_broad)
    print(f"Apple vs Fruit (Parent) Similarity: {sim_ancestry:.4f}")
    assert sim_ancestry > 0.6, "Child should resemble parent"
    
    # 7. Verify Distinctness (Sibling Differnetiation)
    # Apple vs Pear. They differ by their noise components.
    sim_sibling = hv_apple.similarity(hv_pear)
    print(f"Apple vs Pear (Sibling) Similarity: {sim_sibling:.4f}")
    assert sim_sibling < 0.95, "Siblings should not be identical"
    assert sim_sibling > 0.5, "Siblings should share family resemblance"
    
    print("✅ Split Test Passed")

if __name__ == "__main__":
    test_split()
