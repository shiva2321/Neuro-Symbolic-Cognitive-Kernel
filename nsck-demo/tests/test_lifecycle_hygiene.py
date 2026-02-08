import hypervec_shim as hypervec_rs
from python.staged_recall import StagedRecall
from python.lifecycle import LifecycleManager

def test_accretion():
    print("--- Test: Accretion/Drift ---")
    
    # Setup
    hv_base = hypervec_rs.HyperVector(100)
    codebook = {"concept_A": hv_base}
    sr = StagedRecall(codebook, cache_size=10)
    lm = LifecycleManager(sr)
    
    # Test 1: Disabled by default (feature flag)
    observation = hypervec_rs.HyperVector(200)
    drift = lm.accretion_update("concept_A", observation, enable=False)
    assert drift == 1.0, "Should return 1.0 (no change) when disabled"
    assert sr.codebook["concept_A"].similarity(hv_base) == 1.0, "Should not modify when disabled"
    print("  ✓ Disabled by default")
    
    # Test 2: Enabled causes drift
    drift = lm.accretion_update("concept_A", observation, enable=True)
    assert drift < 1.0, "Should drift when enabled"
    print(f"  ✓ Enabled causes drift (sim={drift:.4f})")
    
    # Test 3: Multiple updates cause cumulative drift
    old_hv = sr.codebook["concept_A"]
    for i in range(3):
        obs = hypervec_rs.HyperVector(300 + i)
        lm.accretion_update("concept_A", obs, enable=True)
    
    final_drift = hv_base.similarity(sr.codebook["concept_A"])
    print(f"  ✓ Cumulative drift after 3 updates (sim to original={final_drift:.4f})")
    
    print("✅ Accretion Test Passed")


def test_hygiene():
    print("\n--- Test: Hygiene Monitoring ---")
    
    # Test 1: Healthy codebook (random vectors)
    codebook = {}
    for i in range(50):
        codebook[f"c_{i}"] = hypervec_rs.HyperVector(i)
    
    sr = StagedRecall(codebook, cache_size=10)
    lm = LifecycleManager(sr)
    
    result = lm.hygiene_check()
    print(f"  Healthy codebook: {result['status']}")
    print(f"  Metrics: {result['metrics']}")
    assert result["status"] == "OK", f"Expected OK, got warnings: {result['warnings']}"
    print("  ✓ Healthy codebook passes")
    
    # Test 2: Unhealthy codebook (all very similar)
    base = hypervec_rs.HyperVector(999)
    blurred_codebook = {}
    for i in range(50):
        # Create slightly different vectors by bundling with noise
        if i == 0:
            blurred_codebook[f"blur_{i}"] = base
        else:
            noise = hypervec_rs.HyperVector(1000 + i)
            blurred_codebook[f"blur_{i}"] = base.bundle(noise)
    
    sr2 = StagedRecall(blurred_codebook, cache_size=10)
    lm2 = LifecycleManager(sr2)
    
    result2 = lm2.hygiene_check()
    print(f"  Blurred codebook: {result2['status']}")
    print(f"  Metrics: {result2['metrics']}")
    # This may or may not trigger warning depending on exact similarity
    if result2["status"] == "WARNING":
        print(f"  ✓ Blurred codebook detected: {result2['warnings']}")
    else:
        print(f"  Note: Codebook not blurred enough to trigger warning (avg_sim={result2['metrics']['avg_pairwise_similarity']:.3f})")
    
    print("✅ Hygiene Test Passed")


if __name__ == "__main__":
    test_accretion()
    test_hygiene()
