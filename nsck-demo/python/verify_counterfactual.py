"""
Verify Counterfactual Reasoning (Phase 4.2)
===========================================
Tests "What If" reasoning using only the Causal Graph.
"""

from causal_reasoning import CausalGraph, CausalReasoner

def main():
    print("--- Testing Counterfactual Reasoning ---")
    
    # 1. Setup Graph with some clear outcomes
    graph = CausalGraph()
    ctx = "test_env"
    
    # Action A leads to Success
    graph.add_causes("ACTION_A", "B1", context=ctx)
    graph.add_causes("B1", "REWARD_POS", context=ctx)
    
    # Action B leads to Failure
    graph.add_causes("ACTION_B", "B2", context=ctx)
    graph.add_causes("B2", "DEATH", context=ctx)
    
    # 2. Setup Reasoner (NO simulator)
    reasoner = CausalReasoner(graph, simulator=None)
    
    # 3. Test Counterfactual query
    print("\n--- Step 1: Theoretical Counterfactual ---")
    # "What if I did ACTION_B instead of ACTION_A?"
    res = reasoner.counterfactual(
        actual_action="ACTION_A",
        alternative_action="ACTION_B",
        state={}, # No simulator, state doesn't matter much
        task_tag=ctx
    )
    
    print(f"Query: {res.query}")
    print(f"Original Outcome (A): {res.original_outcome}")
    print(f"Counterfactual Outcome (B): {res.counterfactual_outcome}")
    print(f"Confidence: {res.confidence}")
    print(f"Explanation: {res.explanation}")
    
    if res.original_outcome == "SUCCESS" and res.counterfactual_outcome == "FAILURE":
        print("[PASS] Correctly identified that B would be worse than A.")
    else:
        print("[FAIL] Unexpected counterfactual result.")

if __name__ == "__main__":
    main()
