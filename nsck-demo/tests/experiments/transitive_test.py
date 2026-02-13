import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from text_knowledge_learner import TextKnowledgeLearner
from language_module import LanguageModule

def run_test():
    print("=== Phase 1, Test 2: Transitive Reasoning Chains (5-Step) ===")
    
    # 1. Setup
    corpus_file = os.path.join(os.path.dirname(__file__), "..", "data", "test_corpus", "long_chain.txt")
    
    # Initialize learner
    print("\n[1] Initializing Cognitive Architecture...")
    lang_mod = LanguageModule()
    learner = TextKnowledgeLearner(language_module=lang_mod)
    
    # 2. Learn
    print(f"\n[2] Learning from {os.path.basename(corpus_file)}...")
    session = learner.learn_from_text_file(corpus_file)
    print(f"    - Extracted {session.concepts_learned} concepts")
    
    # 3. Query
    query = "What is the Alpha?" # A -> B -> C -> D -> E -> Zeta
    print(f"\n[3] Query: '{query}'")
    
    response = learner.query_learned_knowledge(query)
    answer = response['answer']
    print(f"    Result: {answer}")
    
    # Check if 'Zeta' or 'Omega' was retrieved
    found_zeta = "zeta" in answer.lower()
    found_omega = "omega" in answer.lower()
    
    if found_zeta:
        print("    -> PASS: Successfully bridged Alpha -> Zeta (5 steps).")
    elif found_omega:
        print("    -> PASS: Successfully bridged Alpha -> Omega (6 steps!).") 
    else:
        print("    -> FAIL: Did not retrieve deeper links (Zeta/Omega).")
        # Print active concepts for debugging
        print(f"    Active Concepts Trace: {response['reasoning_trace']}")

if __name__ == "__main__":
    run_test()
