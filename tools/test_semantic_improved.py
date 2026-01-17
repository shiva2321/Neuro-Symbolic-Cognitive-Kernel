#!/usr/bin/env python
"""
Automated test of the improved semantic assistant.
Tests natural language query parsing and RDF graph retrieval.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.semantic import context_driver

def test_semantic_assistant():
    print("="*70)
    print("  SEMANTIC ASSISTANT - AUTOMATED TEST")
    print("="*70)

    # Use test-specific brain file
    test_brain = "test_semantic_brain.dat"

    # Clean start
    if os.path.exists(test_brain):
        try:
            os.remove(test_brain)
        except PermissionError:
            print(f"⚠️  Warning: Could not remove {test_brain} (file in use)")
            test_brain = f"test_semantic_brain_{os.getpid()}.dat"

    # Temporarily override the BRAIN_FILE constant
    import ncgn.semantic.context_driver as context_driver_module
    original_brain = context_driver_module.BRAIN_FILE
    context_driver_module.BRAIN_FILE = test_brain

    ai = context_driver.SemanticBrain()

    # Load knowledge
    print("\n📖 STEP 1: Loading Knowledge Base...")
    print("-"*70)

    knowledge = """
    GRACE HOPPER INVENTED THE COMPILER.
    ALAN TURING INVENTED THE COMPUTER.
    DENNIS RITCHIE CREATED THE C LANGUAGE.
    GUIDO VAN ROSSUM CREATED PYTHON.
    LINUS TORVALDS CREATED LINUX.
    TIM BERNERS LEE INVENTED THE WEB.
    
    THE COMPILER IS PROGRAM.
    THE COMPUTER IS MACHINE.
    PYTHON IS LANGUAGE.
    LINUX IS OPERATING SYSTEM.
    THE WEB IS NETWORK.
    
    PROGRAMMERS WRITE CODE.
    PROGRAMMERS USE COMPUTERS.
    COMPUTERS RUN PROGRAMS.
    
    THE SUN IS STAR.
    THE EARTH IS PLANET.
    BIRDS FLY.
    FISH SWIM.
    FIRE IS HOT.
    ICE IS COLD.
    """

    ai.learn_rdf(knowledge)

    # Test queries
    print("\n📝 STEP 2: Testing Natural Language Queries...")
    print("-"*70)

    test_cases = [
        "Who invented the compiler?",
        "Who created Python?",
        "What is hot?",
        "What do birds do?",
        "What is the sun?",
        "Who writes code?",
        "What invented Alan Turing?",  # Reverse query
        "What is Python?",
        "Who invented Linux?",
        "What flies?",
    ]

    results = []
    for query in test_cases:
        print(f"\n{'='*70}")
        result_found = False

        # Capture output by temporarily redirecting
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            ai.query(query)
        output = f.getvalue()

        print(output, end='')

        if "ANSWER:" in output:
            result_found = True
            results.append((query, "✓ PASSED"))
        else:
            results.append((query, "✗ FAILED"))

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, status in results if "PASSED" in status)
    total = len(results)

    for query, status in results:
        print(f"  {status}: {query}")

    print("\n" + "-"*70)
    print(f"  Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)

    ai.brain.close()

    # Restore original and cleanup
    context_driver.BRAIN_FILE = original_brain
    try:
        os.remove(test_brain)
    except:
        pass

    return passed == total

if __name__ == "__main__":
    success = test_semantic_assistant()
    sys.exit(0 if success else 1)

