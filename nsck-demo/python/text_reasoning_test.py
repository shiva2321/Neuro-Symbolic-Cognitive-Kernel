
"""
NSCK Text Reasoning Benchmark
=============================
Tests the system's ability to learn from a novel text corpus and reason about it.

Corpus: "The Ecosystem of Xylophone Planets" (Fictional)
Goal: Ensure no pre-training data contamination (LLMs don't know about Space Whales eating sound).
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# Ensure we can import from local python dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_knowledge_learner import TextKnowledgeLearner, LearnedFact
from semantic_memory import SemanticMemory
from episodic_memory import EpisodicMemory

def run_test():
    print("="*60)
    print("NSCK TEXT REASONING BENCHMARK")
    print("="*60)

    # 1. Setup
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_corpus")
    corpora = [
        os.path.join(data_dir, "xylophone_planets.txt"),
        os.path.join(data_dir, "sound_physics.txt")
    ]
    
    for corpus_file in corpora:
        if not os.path.exists(corpus_file):
            print(f"ERROR: Corpus file not found at {corpus_file}")
            return

    from language_module import LanguageModule # [AGI] Phase 4: NLU Parser

    # Initialize learner (Fresh memory for this test)
    print("\n[1] Initializing Cognitive Architecture...")
    
    # Initialize parse (mock or real)
    lang_mod = LanguageModule()
    
    learner = TextKnowledgeLearner(language_module=lang_mod)
    
    # 2. Learn
    total_concepts = 0
    total_relations = 0
    
    for corpus_file in corpora:
        print(f"\n[2] Learning from {os.path.basename(corpus_file)}...")
        start_time = time.time()
        session = learner.learn_from_text_file(corpus_file)
        end_time = time.time()
        
        print(f"    - Processed {session.sentences_processed} sentences in {end_time - start_time:.2f}s")
        print(f"    - Extracted {session.concepts_learned} concepts")
        print(f"    - Discovered {session.relations_learned} relations")
        
        total_concepts += session.concepts_learned
        total_relations += session.relations_learned

    # 3. Evaluation Questions
    print("\n[3] Running Evaluation Queries...")
    
    test_cases = [
        # Type 1: Direct Fact Recall (Corpus A)
        {
            "id": "F1",
            "type": "Fact Recall (A)",
            "query": "What are Xylophone planets made of?",
            "expected_keywords": ["crystal", "singing", "mineral"],
            "difficulty": "Easy"
        },
        # Type 1b: Direct Fact Recall (Corpus B)
        {
            "id": "F2",
            "type": "Fact Recall (B)",
            "query": "What is resonance?",
            "expected_keywords": ["phenomenon", "vibration", "match", "natural"],
            "difficulty": "Easy"
        },
        # Type 2: Relation
        {
            "id": "R1",
            "type": "Relation",
            "query": "What does resonance attract?",
            "expected_keywords": ["space_whales", "whales"],
            "difficulty": "Medium"
        },
        # Type 3: Cross-Domain Generalization
        # Corpus A: Space Whales eat Sound.
        # Corpus B: Sound is Energy. Sound is Vibration.
        # Inference: Space Whales eat Energy / Vibrations.
        {
            "id": "X1",
            "type": "Cross-Domain",
            "query": "Do space_whales consume energy?",
            "expected_keywords": ["yes", "sound", "energy", "eat", "feed"],
            "difficulty": "Hard"
        },
        {
            "id": "X2",
            "type": "Cross-Domain",
            "query": "Are space_whales attracted to vibrations?",
            "expected_keywords": ["yes", "resonance", "attract", "vibration"],
            "difficulty": "Hard"
        },
        # Type 4: Conditional
        {
            "id": "N1",
            "type": "Inference",
            "query": "What happens if the surface cracks?",
            "expected_keywords": ["stop", "resonance", "silence", "leave", "fade"],
            "difficulty": "Hard"
        }
    ]

    results = []
    
    for case in test_cases:
        print(f"\n    Query {case['id']} ({case['type']}): '{case['query']}'")
        
        # Query the learner
        response = learner.query_learned_knowledge(case['query'])
        
        answer_text = response['answer'].lower()
        confidence = response['confidence']
        
        # Check against expected keywords
        found_keywords = [k for k in case['expected_keywords'] if k in answer_text]
        success = len(found_keywords) > 0
        
        result = {
            "id": case['id'],
            "type": case['type'],
            "query": case['query'],
            "success": success,
            "confidence": confidence,
            "found_keywords": found_keywords,
            "full_answer": response['answer'],
            "reasoning_trace": response['reasoning_trace']
        }
        results.append(result)
        
        status = "PASS" if success else "FAIL"
        print(f"    -> {status} (Conf: {confidence:.2f})")
        if not success:
            print(f"       Expected: {case['expected_keywords']}")
            print(f"       Got: {answer_text[:100]}...")

    # 4. Emergent Relation Discovery
    print("\n[4] Analyzing Emergent Relations...")
    emergent = learner.discover_emergent_relations(min_similarity=0.6)
    
    print(f"    Found {len(emergent)} emergent conceptual links.")
    top_emergent = emergent[:10]
    for a, b, sim in top_emergent:
        print(f"    - {a} <--> {b} (sim: {sim:.2f})")

    # 5. Generate Report
    report_file = os.path.join(data_dir, "..", "test_reports", "text_reasoning_report.md")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # Mock single session object for report compatibility
    class MockSession:
        def __init__(self, c, r, f):
            self.concepts_learned = c
            self.relations_learned = r
            self.facts_stored = f
            
    mock_session = MockSession(total_concepts, total_relations, total_relations) # Approximatiowe
    
    generate_report(report_file, mock_session, results, top_emergent)
    print(f"\n[5] Report generated at: {report_file}")


def generate_report(filepath, session, results, emergent):
    pass_count = sum(1 for r in results if r['success'])
    total = len(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# NSCK Text Reasoning Benchmark Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Score:** {pass_count}/{total} ({pass_count/total*100:.1f}%)\n\n")
        
        f.write("## 1. Learning Session\n")
        f.write(f"- **Corpus:** Xylophone Planets (Synthetic)\n")
        f.write(f"- **Concepts Learned:** {session.concepts_learned}\n")
        f.write(f"- **Relations Extracted:** {session.relations_learned}\n")
        f.write(f"- **Facts Stored:** {session.facts_stored}\n\n")
        
        f.write("## 2. Test Results\n\n")
        f.write("| ID | Type | Query | Result | Confidence | Keywords Found |\n")
        f.write("|----|------|-------|--------|------------|----------------|\n")
        
        for r in results:
            status = "✅ PASS" if r['success'] else "❌ FAIL"
            keywords = ", ".join(r['found_keywords']) or "None"
            f.write(f"| {r['id']} | {r['type']} | {r['query']} | {status} | {r['confidence']:.2f} | {keywords} |\n")
            
        f.write("\n### Detailed Responses\n")
        for r in results:
            f.write(f"#### {r['id']}: {r['query']}\n")
            f.write(f"> **System Answer:**\n{r['full_answer']}\n\n")
            f.write(f"> **Reasoning Trace:**\n")
            for trace_step in r['reasoning_trace']:
                f.write(f"- {trace_step}\n")
            f.write("\n---\n")

        f.write("\n## 3. Emergent Knowledge\n")
        f.write("Relations discovered via Semantic Folding (vector similarity) rather than explicit text patterns:\n\n")
        for a, b, sim in emergent:
            f.write(f"- **{a}** is contextually similar to **{b}** ({sim:.2f})\n")

if __name__ == "__main__":
    run_test()
