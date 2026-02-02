"""
Verify Dialogue System
======================
Tests multi-turn conversation and context resolution.
"""

from dialogue_manager import DialogueManager
from language_module import LanguageModule
from lingua_cortex import get_lingua_cortex

class MockCognitiveEngine:
    def __init__(self):
        self.state = "idle"

def main():
    print("--- Initializing Systems ---")
    cortex = get_lingua_cortex()
    cortex.learn_text_snippet("food eat digest fruit") # Teach concepts
    
    lang = LanguageModule() # MOCK mode expected
    engine = MockCognitiveEngine()
    
    dm = DialogueManager(engine, lang)
    
    print("\n--- Test 1: Direct Command ---")
    # User says something explicit
    t1_in = "Go to the food"
    print(f"User: {t1_in}")
    t1_out = dm.process_turn(t1_in)
    print(f"Agent: {t1_out}")
    
    # Assert response acknowledges food
    if "food" in t1_out or "seek" in t1_out.lower(): # Mock LLM returns 'seek' or 'move'
        print("[PASS] Command acknowledged.")
    else:
        print("[FAIL] Command failed.")
        
    print("\n--- Test 2: Anaphora (Context) ---")
    # User refers to 'it' (the food from previous turn)
    # Note: Our simple mock agent needs to have MENTIONED food in the previous turn
    # The current mock implementation of handle_command just returns "Executing..."
    # Let's verify what handle_command returns in language_module mock: 
    # "[MOCK LLM] I will {intent} the {target}."
    
    t2_in = "Eat it"
    print(f"User: {t2_in}")
    
    # Debug: Check context before processing
    # print(f"DEBUG Context: {dm.context_window}")
    
    t2_out = dm.process_turn(t2_in)
    print(f"Agent: {t2_out}")
    
    # We expect 'it' to be resolved to 'food'
    # Why? 
    # Turn 1 Agent: "... I will seek the food."
    # Turn 2 User: "Eat it" -> "Eat food"
    # Turn 2 Agent: "... I will eat the food."
    
    if "food" in t2_out:
        print("[PASS] Anaphora resolved 'it' -> 'food'.")
    else:
        print("[FAIL] Anaphora resolution failed.")

    print("\n--- Test 3: Explanation ---")
    t3_in = "Why?"
    print(f"User: {t3_in}")
    t3_out = dm.process_turn(t3_in)
    print(f"Agent: {t3_out}")
    
    if "hunger" in t3_out.lower():
         print("[PASS] Explanation provided.")
    else:
         print("[FAIL] Explanation logic failed.")

if __name__ == "__main__":
    main()
