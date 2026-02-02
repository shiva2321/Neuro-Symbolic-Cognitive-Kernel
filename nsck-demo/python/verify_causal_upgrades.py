"""
Verify Causal Upgrades (Phase 4.1)
==================================
Tests Optimized Discovery, Dynamic Strengths, and Pruning.
"""

from causal_reasoning import CausalDiscovery, CausalGraph, CausalRelation

def main():
    print("--- Testing Causal Discovery Upgrades ---")
    cd = CausalDiscovery()
    ctx = "test_env"
    
    # 1. Test Optimized Observation
    print("\n--- Step 1: Optimized Induction ---")
    # A causes B and also C
    # B causes C
    for _ in range(20):
        cd.observe(ctx, ["A"], ["B", "C"]) # A -> B, A -> C
        cd.observe(ctx, ["B"], ["C"])      # B -> C
        # Add baseline steps so ~A and ~B don't have C
        for _ in range(4):
            cd.observe(ctx, [], [])
    
    graph = cd.induce_graph(ctx, min_confidence=0.3, min_evidence=5)
    print(f"Nodes in graph: {len(graph.forward)}")
    print(f"Total links: {len(graph.all_links)}")
    for l in graph.all_links:
        print(f"   Link: {l.cause} -> {l.effect} (Strength: {l.strength:.2f})")
    
    # Check if A->B and B->C are found
    if graph.find_path("A", "B") and graph.find_path("B", "C"):
        print("[PASS] Fundamental causal links discovered.")
    
    # 2. Test Pruning
    print("\n--- Step 2: Transitive Pruning ---")
    # Before pruning, A->C exists as a direct link too
    if any(l.cause == "A" and l.effect == "C" for l in graph.all_links):
        print("[INFO] A->C direct link exists (redundant).")
    
    graph.prune_redundant(ctx)
    
    # After pruning, A->C direct link should be gone, but path A->B->C should remain
    direct_ac = next((l for l in graph.all_links if l.cause == "A" and l.effect == "C"), None)
    if not direct_ac:
        print("[PASS] Redundant link A->C pruned.")
    else:
        print("[FAIL] Redundant link A->C still exists.")
        
    path = graph.find_path("A", "C")
    if path and len(path) == 2:
        print("[PASS] Path A->B->C still exists after pruning.")

    # 3. Test Dynamic Strength
    print("\n--- Step 3: Dynamic Updates ---")
    # Now make B lead to C only 50% of the time
    for _ in range(10):
        cd.observe(ctx, ["B"], []) # B without C
    
    cd.update_strengths(ctx, graph)
    bc_link = next(l for l in graph.all_links if l.cause == "B" and l.effect == "C")
    print(f"Updated B->C strength: {bc_link.strength:.2f}")
    if bc_link.strength < 0.8:
        print("[PASS] Strength updated dynamically.")

if __name__ == "__main__":
    main()
