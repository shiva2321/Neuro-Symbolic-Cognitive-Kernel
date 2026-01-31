import hypervec_rs
from python.staged_recall import StagedRecall
from python.lifecycle import LifecycleManager

def test_merge():
    print("--- Test: Concept Merge ---")
    
    # 1. Create Base Concept
    hv_base = hypervec_rs.HyperVector(100)
    
    # 2. Create Similar Concept (A + small noise)
    # Since we can't add noise bit-wise easily from python without exposure,
    # let's try bundling with itself multiple times? No, that does nothing.
    # Bundling with a weak orthogonal vector?
    # A.bundle(B) -> 50/50 mix. Similarity ~0.75?
    
    # 2. Create Similar Concept (A + Noise)
    # A.bundle(A).bundle(A).bundle(B) -> 3 parts A, 1 part B. Sim ~0.8-0.9
    
    hv_noise = hypervec_rs.HyperVector(999)
    hv_similar = hv_base.bundle(hv_base).bundle(hv_base).bundle(hv_noise)
    
    sim = hv_base.similarity(hv_similar)
    print(f"Base vs Similar Similarity: {sim:.4f}")
    assert sim > 0.7, "Failed to create similar test vector"
    
    # 3. Setup Staged Recall
    codebook = {
        "concept_A": hv_base,
        "concept_A_dup": hv_similar,
        "concept_C": hypervec_rs.HyperVector(200) # control
    }
    
    sr = StagedRecall(codebook, cache_size=10)
    lm = LifecycleManager(sr)
    
    # Pre-load cache
    sr.query(hv_base, top_k=1)
    
    # DEBUG: Check what query returns
    print("DEBUG: Query for concept_A returns:")
    print(sr.query(hv_base, top_k=5, threshold=0.6))
    
    # 4. Detect Duplicates
    print("Scanning for duplicates...")
    # Lower threshold to catch 0.75
    dups = lm.detect_duplicates(threshold=0.70)
    print(f"Found duplicates: {dups}")
    
    assert len(dups) > 0, "Failed to detect duplicate"
    pair = dups[0]
    print(f"Merging detected pair: {pair[0]} <- {pair[1]}")
    
    # 5. Merge
    lm.merge_concepts(pair[0], pair[1])
    
    # 6. Verify
    assert pair[1] not in sr.codebook, "Duplicate B should be gone"
    assert pair[0] in sr.codebook, "Original A should remain"
    
    # Check if A is now a centroid (moved slightly?)
    new_hv = sr.codebook[pair[0]]
    sim_orig = new_hv.similarity(hv_base)
    print(f"New A vs Original A Similarity: {sim_orig:.4f}")
    
    print("✅ Merge Test Passed")

if __name__ == "__main__":
    test_merge()
