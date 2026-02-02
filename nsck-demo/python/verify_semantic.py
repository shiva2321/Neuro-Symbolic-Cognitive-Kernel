"""
Verify Semantic Memory (Phase 3.1)
==================================
Tests Concept addition, Relations, Spreading Activation and Schema extraction.
"""

from semantic_memory import SemanticMemory
import hypervec_shim as hypervec_rs

def main():
    print("--- Testing Semantic Memory ---")
    sm = SemanticMemory()
    
    # 1. Add Concepts
    print("\n--- Step 1: Concept Addition ---")
    sm.add_concept("Apple", {"color": "red", "taste": "sweet"})
    sm.add_concept("Fruit", {"edible": True})
    sm.add_concept("Pie", {"base": "dough"})
    
    if "Apple" in sm.concept_graph:
        print("[PASS] Concepts added to graph.")
    
    # 2. Add Relations
    print("\n--- Step 2: Relation Building ---")
    sm.add_relation("Apple", "is_a", "Fruit")
    sm.add_relation("Apple", "part_of", "Pie")
    
    if sm.concept_graph.has_edge("Apple", "Fruit"):
        print("[PASS] Relations established.")
        
    # 3. Spreading Activation
    print("\n--- Step 3: Spreading Activation ---")
    # Start at Apple
    levels = sm.spread_activation(["Apple"], steps=2)
    print(f"Activation Levels: {levels}")
    
    if levels.get("Fruit", 0) > 0:
        print("[PASS] Activation spread from Apple to Fruit.")
    if levels.get("Pie", 0) > 0:
        print("[PASS] Activation spread from Apple to Pie.")

    # 4. Schema Extraction
    print("\n--- Step 4: Schema Extraction ---")
    schema = sm.extract_schema("Apple")
    print(f"Apple Schema: {schema}")
    
    if "Fruit" in schema["is_a"] and "Pie" in schema["parts"]:
        print("[PASS] Schema correctly identifies relations.")

    # 5. Similarity Query
    print("\n--- Step 5: Similarity Query ---")
    apple_hv = sm.concept_hvs["Apple"]
    matches = sm.query(apple_hv, k=1)
    print(f"Closest match to Apple HV: {matches}")
    
    if matches[0][0] == "Apple":
        print("[PASS] Self-lookup match success.")

if __name__ == "__main__":
    main()
