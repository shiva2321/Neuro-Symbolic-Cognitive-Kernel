import sys
import os
import time

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from text_knowledge_learner import TextKnowledgeLearner
from language_module import LanguageModule

def run_test():
    print("=== Phase 1, Test 1: Belief Revision (Unreliable Witness) ===")
    
    # 1. Setup
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_corpus")
    file_a = os.path.join(data_dir, "belief_A.txt")
    file_b = os.path.join(data_dir, "belief_B.txt")
    
    # Initialize learner
    print("\n[1] Initializing Cognitive Architecture...")
    lang_mod = LanguageModule()
    learner = TextKnowledgeLearner(language_module=lang_mod)
    
    # --- Step 1: Learn Initial Belief (Time T=1000) ---
    print("\n[2] Learning Corpus A (Time T=1000)...")
    learner.current_simulated_time = 1000.0
    learner.learn_from_text_file(file_a)
    
    # Query Check
    print("    Query: What color is the sky?")
    resp_1 = learner.query_learned_knowledge("What color is the sky?")
    ans_1 = resp_1['answer'].lower()
    print(f"    Result: {resp_1['answer']}")
    
    if "blue" in ans_1:
        print("    -> PASS: System believes sky is blue.")
    else:
        print("    -> FAIL: System failed to learn initial belief.")

    # --- Step 2: Update Belief (Time T=2000) ---
    print("\n[3] Learning Corpus B (Time T=2000) [CORRECTIVE UPDATE]...")
    learner.current_simulated_time = 2000.0
    learner.learn_from_text_file(file_b)
    
    # Query Check
    print("    Query: What color is the sky?")
    resp_2 = learner.query_learned_knowledge("What color is the sky?")
    ans_2 = resp_2['answer'].lower()
    print(f"    Result: {resp_2['answer']}")
    
    if "red" in ans_2:
        print("    -> PASS: System updated belief to red.")
    else:
        print(f"    -> FAIL: System stuck on '{ans_2}'. Expected 'red'.")

    # --- Step 3: Attempt Regression (Time T=500) ---
    print("\n[4] Attempting to inject old info (Time T=500) [SHOULD BE REJECTED]...")
    learner.current_simulated_time = 500.0
    learner.learn_from_text_file(file_a) # Re-feed "Sky is blue" but with OLD timestamp
    
    # Query Check
    print("    Query: What color is the sky?")
    resp_3 = learner.query_learned_knowledge("What color is the sky?")
    ans_3 = resp_3['answer'].lower()
    print(f"    Result: {resp_3['answer']}")
    
    # Check 1: Red must be present
    has_red = "red" in ans_3
    
    # Check 2: Red must appear BEFORE Blue (Priority)
    pos_red = ans_3.find("red")
    pos_blue = ans_3.find("blue")
    
    if has_red:
        if pos_blue == -1 or pos_red < pos_blue:
            print("    -> PASS: System prioritizes 'red' (Current Belief).")
        else:
            print("    -> FAIL: 'Red' is present but 'Blue' appears first/stronger.")
    else:
        print(f"    -> FAIL: System lost 'red'.")

if __name__ == "__main__":
    run_test()
