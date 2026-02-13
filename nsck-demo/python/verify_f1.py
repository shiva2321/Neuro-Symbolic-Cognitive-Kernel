import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from text_knowledge_learner import TextKnowledgeLearner
from language_module import LanguageModule

def run_test():
    print("=== Phase 2: verify F1 Recall Fix ===")
    
    # 1. Setup
    corpus_file = os.path.join(os.path.dirname(__file__), "..", "data", "test_corpus", "xylophone_planets.txt")
    
    # Initialize learner
    print("\n[1] Initializing Cognitive Architecture...")
    lang_mod = LanguageModule()
    learner = TextKnowledgeLearner(language_module=lang_mod)
    
    # 2. Learn
    print(f"\n[2] Learning from {os.path.basename(corpus_file)}...")
    session = learner.learn_from_text_file(corpus_file)
    print(f"    - Extracted {session.concepts_learned} concepts")
    
    # 3. Query
    query = "What are Xylophone planets made of?"
    print(f"\n[3] Query: '{query}'")
    
    response = learner.query_learned_knowledge(query)
    answer = response['answer'].lower()
    print(f"    Result: {response['answer']}")
    
    expected = ["crystal", "mineral", "glass"]
    found = [k for k in expected if k in answer]
    
    if found:
        print(f"    -> PASS: Found keywords {found}. Fix verified.")
    else:
        print("    -> FAIL: Still missing composition facts.")
        # Debug relations
        print("    Relations involving 'Planets':")
        for u, r, v in learner.semantic.concept_graph.edges(data='relation'):
            if u == 'Planets' or v == 'Planets':
                print(f"      - {u} --[{r}]--> {v}")

if __name__ == "__main__":
    run_test()
