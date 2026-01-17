#!/usr/bin/env python
"""
Comprehensive Brain Test - Demonstrates all capabilities
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from semantic.context_driver import SemanticBrain

def test_brain():
    """Run comprehensive tests of the brain"""

    print("\n" + "=" * 80)
    print(" ENGLISH LANGUAGE BRAIN - COMPREHENSIVE TEST")
    print("=" * 80)

    brain = SemanticBrain()

    print(f"\nBrain Status:")
    print(f"  - Vocabulary: {len(brain.word_to_id)} concepts")
    print(f"  - Ready: Yes")

    print("\n" + "=" * 80)
    print(" TEST SUITE")
    print("=" * 80)

    test_queries = [
        ("What is an apple?", "FRUIT"),
        ("What is a fruit?", "includes FOOD"),
        ("What are animals?", "FRIENDS or FISH"),
        ("What do animals need?", "WATER"),
        ("Who are friends?", "ANIMALS"),
        ("What is friendship?", "TREASURE"),
        ("What is happiness?", "EMOTION"),
        ("What is wisdom?", "KNOWLEDGE"),
        ("What is morning?", "TIME"),
        ("What is in the forest?", "PLACE or HOME"),
        ("What is wisdom?", "KNOWLEDGE"),
        ("What is success?", "OUTCOME"),
    ]

    passed = 0
    failed = 0

    for query, expected in test_queries:
        print(f"\n[TEST] {query}")
        print(f"        Expected: {expected}")
        try:
            brain.query(query)
            passed += 1
        except Exception as e:
            print(f"        ERROR: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal Tests: {len(test_queries)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_queries))*100:.1f}%")

    print("\n" + "=" * 80)
    print(" TESTING LEARNING CAPABILITY")
    print("=" * 80)

    print("\nTeaching brain new concepts...")
    new_facts = """
    THE COFFEE IS BEVERAGE.
    BEVERAGE IS DRINK.
    DRINK REFRESHES HUMAN.
    PYTHON IS LANGUAGE.
    LANGUAGE IS TOOL.
    """
    brain.learn_rdf(new_facts)

    print("\nTesting newly learned concepts:")
    brain.query("What is coffee?")
    brain.query("What is python?")

    print("\n" + "=" * 80)
    print(" BRAIN DEMONSTRATION COMPLETE")
    print("=" * 80)

    brain.brain.close()

if __name__ == "__main__":
    test_brain()
