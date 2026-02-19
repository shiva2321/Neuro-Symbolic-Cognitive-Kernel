
import unittest
import sys
import os
import numpy as np
from typing import List, Tuple, Dict

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from python.core.memory.semantic_memory import SemanticMemory
from python.core.vsa.hypervec_shim import HyperVector

from python.core.language.dialogue_manager import DialogueManager
from python.core.reasoning.causal_interface import MockCausalService, Explanation, Prediction

class CognitiveTuringTest:
    """
    A rigorous test suite to verify 'Understanding' in the VSA model.
    Checks for:
    1. Synonymy (Semantic Similarity)
    2. Analogical Reasoning (Vector Arithmetic)
    3. Categorization (Is-A relationships)
    4. Causal Reasoning (Cause-Effect chains)
    """
    
    def __init__(self, memory_path: str = "data/vsa_memory.pkl"):
        self.memory = SemanticMemory()
        if os.path.exists(memory_path):
            print(f"[Test] Loading memory from {memory_path}...")
            self.memory.load(memory_path)
        else:
            print(f"[Test] WARNING: No memory found at {memory_path}. Tests will likely fail.")
            
        self.report = []
        
    def run_all(self):
        print("\n=== RUNNING COGNITIVE TURING TEST ===")
        self.test_synonyms()
        self.test_analogies()
        self.test_categorization()
        self.test_causal_explanation()
        self.test_counterfactual_reasoning()
        self.generate_report()
        
    def test_synonyms(self):
        print("\n[1] Testing Semantic Similarity (Synonyms)...")
        # Pairs that SHOULD be similar if learned from large corpus
        # Note: 'wikitext' might be too small for some complex ones, 
        # but 'happy'/'glad' or 'big'/'large' often works.
        pairs = [
            ("happy", "glad"),
            ("big", "large"),
            ("start", "begin"),
            ("doctor", "nurse"), # Related
            ("king", "queen"),   # Related
            ("cat", "dog"),      # Related animals
            ("apple", "banana"), # Related fruits
        ]
        
        score = 0
        for w1, w2 in pairs:
            w1 = w1.capitalize()
            w2 = w2.capitalize()
            
            if w1 in self.memory.concept_hvs and w2 in self.memory.concept_hvs:
                v1 = self.memory.get_concept(w1)
                v2 = self.memory.get_concept(w2)
                sim = v1.similarity(v2)
                
                # Check for direct relationship in graph
                rel = None
                if self.memory.concept_graph.has_edge(w1, w2):
                    rel = self.memory.concept_graph[w1][w2].get("relation")
                elif self.memory.concept_graph.has_edge(w2, w1):
                    rel = self.memory.concept_graph[w2][w1].get("relation") + " (inverse)"
                
                print(f"  {w1} <-> {w2}: Similarity={sim:.3f} | DirectRelation={rel}")
                
                # Threshold for 'relatedness' (random is ~0.0)
                if sim > 0.15 or rel is not None: 
                    score += 1
                    self.report.append(f"[PASS] Synonym/Related: {w1}-{w2} (Sim: {sim:.2f})")
                else:
                    self.report.append(f"[FAIL] Synonym/Related: {w1}-{w2} (Sim: {sim:.2f})")
            else:
                print(f"  {w1} <-> {w2}: [MISSING CONCEPTS]")
                self.report.append(f"[SKIP] Synonym: {w1}-{w2} (Missing)")
                
    def test_analogies(self):
        print("\n[2] Testing Analogical Reasoning (Vector Arithmetic)...")
        # A - B + C = D
        # King - Man + Woman = Queen
        
        triplets = [
            ("King", "Man", "Woman", "Queen"),
            ("Paris", "France", "London", "England"), # Maybe too hard for small training
        ]
        
        for a, b, c, expected in triplets:
            if all(w in self.memory.concept_hvs for w in [a, b, c]):
                va = self.memory.get_concept(a)
                vb = self.memory.get_concept(b)
                vc = self.memory.get_concept(c)
                
                # Target Vector
                target = va.xor(vb).xor(vc) # Simple XOR arithmetic for binary vectors roughly approx subtraction/addition
                # Or better: Bundle(A, C) unbind B? 
                # Standard VSA analogy: D = A * inv(B) * C? 
                # Depends on encoding. With SDR/Sparse, simple overlay often works for set logic.
                
                # Let's try simple similarity search against all concepts
                nearest = self._find_nearest(target, exclude=[a, b, c])
                
                print(f"  {a} - {b} + {c} = ? (Expected: {expected})")
                print(f"  -> Nearest: {nearest[:3]}")
                
                found_names = [n for n, s in nearest]
                if expected in found_names[:5]:
                    self.report.append(f"[PASS] Analogy: {a}-{b}+{c}={expected}")
                else:
                    self.report.append(f"[FAIL] Analogy: {a}-{b}+{c}={expected} (Got {found_names[:1]})")
            else:
                self.report.append(f"[SKIP] Analogy: {a}-{b}+{c} (Missing terms)")

    def test_categorization(self):
        print("\n[3] Testing Categorization (Is-A)...")
        # Check if 'is_a' relations exist
        
        checks = [
            ("Apple", "Fruit"),
            ("Cat", "Animal"),
            ("Car", "Vehicle"),
            ("Red", "Color")
        ]
        
        for inst, cat in checks:
            rel = None
            if self.memory.concept_graph.has_edge(inst, cat):
                rel = self.memory.concept_graph[inst][cat].get("relation")
            
            print(f"  {inst} -> {cat}: {rel}")
            if rel and rel in ["is_a", "type_of"]:
                self.report.append(f"[PASS] Categorization: {inst} is {cat}")
            else:
                # Also check similarity
                if inst in self.memory.concept_hvs and cat in self.memory.concept_hvs:
                    sim = self.memory.get_concept(inst).similarity(self.memory.get_concept(cat))
                    if sim > 0.2:
                         self.report.append(f"[PASS] Categorization: {inst} ~ {cat} (Sim: {sim:.2f})")
                    else:
                         self.report.append(f"[FAIL] Categorization: {inst} -> {cat}")
                else:
                    self.report.append(f"[SKIP] Categorization: {inst}-{cat} (Missing)")

    def test_causal_explanation(self):
        print("\n[4] Testing Causal Explanation (Why?)...")
        # Setup specific mock for valid causal chain
        class ValidCausalService(MockCausalService):
            def explain_why(self, effect, context=None):
                if effect == "wet_grass":
                    return Explanation(
                        query="Why wet_grass?",
                        cause="rain",
                        effect="wet_grass",
                        chain=["rain", "wet_grass"],
                        confidence=0.9,
                        text="" # Let NLGEngine realize it
                    )
                return super().explain_why(effect, context)

        # Instantiate DialogueManager with this service
        dm = DialogueManager(None, None, causal_service=ValidCausalService())
        
        # Test 1: Known cause
        response = dm.handle_explanation({"entities": ["wet_grass"], "intent": "why"})
        print(f"  Why wet_grass? -> {response}")
        
        if "rain causes wet_grass" in response or "rain causes wet grass" in response.lower():
             self.report.append(f"[PASS] Causal Explanation: wet_grass -> rain")
        else:
             self.report.append(f"[FAIL] Causal Explanation: wet_grass -> {response}")

        # Test 2: Unknown cause (Fallback)
        response_unknown = dm.handle_explanation({"entities": ["flying_pigs"], "intent": "why"})
        print(f"  Why flying_pigs? -> {response_unknown}")
        
        if "not sure why" in response_unknown:
             self.report.append(f"[PASS] Causal Explanation: unknown -> handled gracefully")
        else:
             self.report.append(f"[FAIL] Causal Explanation: unknown -> {response_unknown}")

    def test_counterfactual_reasoning(self):
        print("\n[5] Testing Counterfactual Reasoning (What if?)...")
        
        class ValidPredictionService(MockCausalService):
            def predict_what_if(self, action, state, context=None):
                if action == "eat_poison":
                    return Prediction(
                        query="What if eat_poison?",
                        action="eat_poison",
                        predicted_outcome="sickness",
                        confidence=0.9,
                        text=""
                    )
                return super().predict_what_if(action, state, context)
        
        dm = DialogueManager(None, None, causal_service=ValidPredictionService())
        
        # Test 1: Prediction
        response = dm.handle_prediction({"entities": ["eat_poison"], "intent": "what_if"})
        print(f"  What if eat_poison? -> {response}")
        
        if "sickness might happen" in response:
             self.report.append(f"[PASS] Counterfactual: eat_poison -> sickness")
        else:
             self.report.append(f"[FAIL] Counterfactual: eat_poison -> {response}")

    def _find_nearest(self, hv, exclude=None, top_k=5):
        exclude = exclude or []
        scores = []
        for name, v in self.memory.concept_hvs.items():
            if name in exclude: continue
            sim = hv.similarity(v)
            scores.append((name, sim))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def generate_report(self):
        print("\n=== REPORT CARD ===")
        for line in self.report:
            print(line)
        
        # Write to file
        with open("COGNITIVE_REPORT_CARD.md", "w") as f:
            f.write("# Cognitive Capabilities Report\n\n")
            f.write(f"Date: {np.datetime64('now')}\n\n")
            for line in self.report:
                f.write(f"- {line}\n")
        print("\nSaved to COGNITIVE_REPORT_CARD.md")

if __name__ == "__main__":
    test = CognitiveTuringTest()
    test.run_all()
