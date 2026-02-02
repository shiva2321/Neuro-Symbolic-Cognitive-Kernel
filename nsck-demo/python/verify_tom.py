"""
Verify Theory of Mind (Sally-Anne Test)
=======================================
Simulates the classic False Belief task to verify the TheoryOfMind module.

Scenario:
1. Sally places Ball in Basket.
2. Sally leaves.
3. Anne moves Ball to Box.
4. Sally returns.
5. Agent (Observer) must predict correct action for Sally based on her False Belief.
"""

from theory_of_mind import TheoryOfMind

def main():
    print("--- Initializing ToM ---")
    tom = TheoryOfMind()
    
    # 1. Setup Scene
    # Truth State
    reality = {
        "ball_location": "basket"
    }
    print(f"Initial Reality: {reality}")
    
    # Sally is in the room. She sees the ball in the basket.
    print("\n--- Step 1: Sally observes Ball in Basket ---")
    tom.update_agent_perspective("Sally", "room", {"ball_location": "basket"})
    
    # Anne is also here
    tom.update_agent_perspective("Anne", "room", {"ball_location": "basket"})
    
    # 2. Sally leaves
    print("\n--- Step 2: Sally leaves the room ---")
    # Sally is now 'outside'. She stops receiving updates about the 'room'.
    # We update her model with 'nothing observable' from the room.
    tom.update_agent_perspective("Sally", "outside", {})
    
    # 3. Anne moves the ball
    print("\n--- Step 3: Anne moves Ball to Box ---")
    reality["ball_location"] = "box"
    print(f"New Reality: {reality}")
    
    # Anne sees this happen
    tom.update_agent_perspective("Anne", "room", {"ball_location": "box"})
    
    # Sally does NOT see this. Her belief remains stale.
    
    # 4. Critical Question: Where does Sally believe the ball is?
    print("\n--- Step 4: False Belief Detection ---")
    
    # Check Sally
    sally_false_beliefs = tom.detect_false_belief("Sally", reality)
    print(f"False Beliefs detected for Sally: {sally_false_beliefs}")
    
    if "ball_location" in sally_false_beliefs:
        print("[PASS] Correctly identified Sally holds a False Belief about ball_location.")
    else:
        print("[FAIL] Failed to detect False Belief.")
        
    sally_belief = tom.get_or_create_model("Sally").get_belief("ball_location")
    print(f"Sally believes ball is in: {sally_belief} (Reality: {reality['ball_location']})")
    
    # Check Anne (Control)
    anne_false_beliefs = tom.detect_false_belief("Anne", reality)
    if not anne_false_beliefs:
        print("[PASS] Anne correctly holds True Belief (she saw the move).")
    else:
        print(f"[FAIL] Anne incorrectly flagged with False Belief: {anne_false_beliefs}")

    # 5. Prediction
    print("\n--- Step 5: Action Prediction ---")
    # Sally wants the ball
    tom.get_or_create_model("Sally").set_desire("find_ball")
    
    predicted_action = tom.predict_action("Sally")
    print(f"Predicted Action for Sally: {predicted_action}")
    
    expected = "search_basket"
    if predicted_action == expected:
        print(f"[PASS] Successfully predicted Sally will act on False Belief ({expected}).") 
    else:
        print(f"[FAIL] Prediction mismatch. Got {predicted_action}, expected {expected}.")

if __name__ == "__main__":
    main()
