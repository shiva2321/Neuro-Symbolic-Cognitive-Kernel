"""
Verify Empathy System (Phase 2.3)
=================================
Tests emotional contagion and social response generation.
"""

from emotion_system import EmotionSystem
from theory_of_mind import TheoryOfMind
from empathy import EmpathyModule

def main():
    print("--- Initializing Empathy Systems ---")
    es = EmotionSystem()
    tom = TheoryOfMind()
    empathy = EmpathyModule(es, tom)
    
    # --- Test 1: Sadness Contagion ---
    print("\n--- Test 1: Empathizing with Sadness (Bob) ---")
    tom.get_or_create_model("Bob")
    # Bob wants a toy but doesn't have it
    tom.agent_models["Bob"].set_desire("have_toy")
    tom.agent_models["Bob"].update_beliefs({"holding": "nothing"})
    
    # Observe Initial State
    es.valence = 0.0 # Reset self
    print(f"Self Initial Valence: {es.valence}")
    
    result = empathy.empathize("Bob")
    print(f"Interaction Result: {result}")
    
    if result["inferred_emotion"] == "sadness":
        print("[PASS] Correctly inferred Bob is sad (blocked desire).")
    else:
        print(f"[FAIL] Inferred {result['inferred_emotion']}, expected sadness.")
        
    if es.valence < 0.0:
        print(f"[PASS] Emotional Contagion successful (Valence dropped to {es.valence:.2f}).")
    else:
        print(f"[FAIL] No contagion effect (Valence {es.valence}).")
        
    if "help" in result["response"]:
        print("[PASS] Compassionate response generated.")
    else:
        print("[FAIL] Response improper.")

    # --- Test 2: Joy Contagion ---
    print("\n--- Test 2: Empathizing with Joy (Alice) ---")
    tom.get_or_create_model("Alice")
    # Alice wants food and HAS food
    tom.agent_models["Alice"].set_desire("have_food")
    tom.agent_models["Alice"].update_beliefs({"holding": "food"})
    
    # Reset self valence to 0 for clear testing
    es.valence = 0.0
    
    result = empathy.empathize("Alice")
    print(f"Interaction Result: {result}")
    
    if result["inferred_emotion"] == "joy":
        print("[PASS] Correctly inferred Alice is joyful (satisfied desire).")
    else:
        print(f"[FAIL] Inferred {result['inferred_emotion']}, expected joy.")

    if es.valence > 0.0:
        print(f"[PASS] Emotional Contagion successful (Valence rose to {es.valence:.2f}).")
    else:
        print(f"[FAIL] No contagion effect.")

if __name__ == "__main__":
    main()
