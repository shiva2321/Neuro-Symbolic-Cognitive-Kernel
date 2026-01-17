#!/usr/bin/env python
"""Simple demo of the improved semantic assistant WITHOUT test framework."""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

from semantic.context_driver import SemanticBrain

# Use test brain
if os.path.exists("demo_brain.dat"):
    try:
        os.remove("demo_brain.dat")
    except:
        pass

from semantic import context_driver
context_driver.BRAIN_FILE = "demo_brain.dat"

ai = SemanticBrain()

print("="*70)
print("SEMANTIC ASSISTANT DEMO")
print("="*70)

# Load knowledge
print("\nLoading knowledge...")
knowledge = """
GRACE HOPPER INVENTED THE COMPILER.
ALAN TURING INVENTED THE COMPUTER.
DENNIS RITCHIE CREATED THE C LANGUAGE.
GUIDO VAN ROSSUM CREATED PYTHON.

THE COMPILER IS PROGRAM.
THE COMPUTER IS MACHINE.
PYTHON IS LANGUAGE.

PROGRAMMERS WRITE CODE.
BIRDS FLY.
FISH SWIM.
FIRE IS HOT.
"""

ai.learn_rdf(knowledge)

# Test queries
print("\n" + "="*70)
print("TESTING NATURAL LANGUAGE QUERIES")
print("="*70)

queries = [
    "Who invented the compiler?",
    "Who created Python?",
    "What is hot?",
    "What flies?",
    "Who writes code?",
]

for q in queries:
    ai.query(q)
    print()

ai.brain.close()

try:
    os.remove("demo_brain.dat")
except:
    pass

print("\nDemo complete!")

