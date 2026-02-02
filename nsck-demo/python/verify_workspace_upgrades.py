"""
Verify Global Workspace Upgrades (Phase 3.3)
===========================================
Tests Confidence-based competition and Mission focus bias.
"""

from global_workspace import GlobalWorkspace, Coalition

def main():
    print("--- Testing Global Workspace Upgrades ---")
    gw = GlobalWorkspace(attention_threshold=0.1)
    
    # 1. Confidence-based competition
    print("\n--- Step 1: Confidence Impact ---")
    # Rule with higher confidence should win over SNN even if salience is lower
    c1 = Coalition(source="SNN", content="LEFT", base_salience=0.5, sender_confidence=0.2) # Act: 0.5 + 0.1 = 0.6
    c2 = Coalition(source="RULES", content="RIGHT", base_salience=0.4, sender_confidence=0.9) # Act: 0.4 + 0.45 = 0.85
    
    winner = gw.compete([c1, c2])
    print(f"Winner: {winner.source} (Activation: {winner.activation:.2f})")
    if winner.source == "RULES":
        print("[PASS] Confident module won over higher base salience.")
        
    # 2. Mission Focus Bias
    print("\n--- Step 2: Mission Focus Bias ---")
    gw.mission_focus = "EXPLORATION"
    
    c3 = Coalition(source="PLANNER", content="UP", base_salience=0.6, sender_confidence=0.5) # Act: 0.6 + 0.25 = 0.85
    c4 = Coalition(source="EXPLORATION", content="DOWN", base_salience=0.5, sender_confidence=0.5) # Act: (0.5+0.2) + 0.25 = 0.95
    
    winner2 = gw.compete([c3, c4])
    print(f"Winner: {winner2.source} (Activation: {winner2.activation:.2f})")
    if winner2.source == "EXPLORATION":
        print("[PASS] Mission focus prioritized Exploration.")

if __name__ == "__main__":
    main()
