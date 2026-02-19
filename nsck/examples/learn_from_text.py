#!/usr/bin/env python3
"""
Learn from Text Example
=======================
Demonstrates how to:
  1. Feed raw text into the knowledge pipeline
  2. Query the resulting semantic memory
  3. Use transitive reasoning over learned facts

Run from the nsck-demo/ directory:
    python examples/learn_from_text.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.core.memory.semantic_memory import SemanticMemory
from python.core.language.text_knowledge_learner import TextKnowledgeLearner


def main():
    # ── 1. Create semantic memory + learner ─────────────────────
    memory = SemanticMemory()
    learner = TextKnowledgeLearner(memory)

    # ── 2. Feed in a small corpus ───────────────────────────────
    corpus = """
    The sun is a star.
    Stars produce energy through nuclear fusion.
    Nuclear fusion converts hydrogen into helium.
    The sun is the closest star to Earth.
    Earth orbits the sun.
    The moon orbits Earth.
    """

    print("Learning from corpus...")
    learner.learn_from_text(corpus)

    # ── 3. Inspect what was learned ─────────────────────────────
    print(f"\nConcepts stored: {len(memory.concept_hvs)}")
    for concept in sorted(memory.concept_hvs.keys()):
        print(f"  - {concept}")

    # ── 4. Query relations (stored in concept_graph edges) ────
    print("\nRelations:")
    for u, v, data in memory.concept_graph.edges(data=True):
        rel = data.get("relation", "related_to")
        print(f"  {u} --[{rel}]--> {v}")

    # ── 5. Similarity queries ───────────────────────────────────
    if "Sun" in memory.concept_hvs and "Star" in memory.concept_hvs:
        sim = memory.concept_hvs["Sun"].similarity(memory.concept_hvs["Star"])
        print(f"\nSimilarity(Sun, Star) = {sim:.4f}")

    if "Earth" in memory.concept_hvs and "Moon" in memory.concept_hvs:
        sim = memory.concept_hvs["Earth"].similarity(memory.concept_hvs["Moon"])
        print(f"Similarity(Earth, Moon) = {sim:.4f}")

    # ── 6. Spreading activation ─────────────────────────────────
    print("\nSpreading activation from 'Sun':")
    activation = memory.spread_activation(["Sun"], steps=2, decay=0.5)
    for concept, score in sorted(activation.items(), key=lambda x: -x[1])[:8]:
        print(f"  {concept}: {score:.4f}")


if __name__ == "__main__":
    main()
