"""
Verify LLM Isolation
====================
Checks that the LanguageModule correctly decouples "Text" from "Meaning".
"""

from language_module import LanguageModule
from lingua_cortex import get_lingua_cortex, SemanticFingerprint
import numpy as np

def main():
    print("--- Initializing Language Module ---")
    # Will default to MOCK mode if model not found, which is fine for architectural verification
    lang = LanguageModule()
    
    # Ensure Semantic Map has some data for grounding
    print("\n--- seeding Cortex ---")
    cortex = get_lingua_cortex()
    cortex.learn_text_snippet("food eat hungry energy")
    cortex.learn_text_snippet("move run walk navigate")
    
    print("\n--- Testing UNDERSTAND (Input) ---")
    user_input = "I want food"
    print(f"User Input: '{user_input}'")
    
    result = lang.understand(user_input)
    
    structured = result['structured_output']
    hv = result['grounded_hv']
    
    print(f"Structured Out: {structured}")
    print(f"Grounded HV Type: {type(hv)}")
    
    if hv is not None and isinstance(hv, SemanticFingerprint):
        print("[SUCCESS] Input grounded to VSA Hypervector.")
    else:
        print("[FAILURE] Input NOT grounded to VSA Hypervector.")
        
    print("\n--- Testing GENERATE (Output) ---")
    # Core system sends INTENT, not text
    core_intent = {"action": "move", "target": "food", "reason": "hunger high"}
    print(f"Core Intent: {core_intent}")
    
    response = lang.generate(core_intent)
    print(f"Generated Response: '{response}'")
    
    if isinstance(response, str) and len(response) > 0:
        print("[SUCCESS] Generated natural language from Intent.")
    else:
        print("[FAILURE] Did not generate text.")

if __name__ == "__main__":
    main()
