"""
examples/03_text_knowledge_learner.py
=======================================
Learn structured knowledge (subject-verb-object triples) from plain text and
then query the resulting semantic memory graph.

Run from the repository root:
    python examples/03_text_knowledge_learner.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck'))

from python.core.language.text_knowledge_learner import TextKnowledgeLearner

# ------------------------------------------------------------------
# 1. Create the learner (creates its own SemanticMemory internally)
# ------------------------------------------------------------------
learner = TextKnowledgeLearner()

# ------------------------------------------------------------------
# 2. Ingest text — learn_from_text() parses sentences into triples
# ------------------------------------------------------------------
text = (
    "Alan Turing invented the Turing machine. "
    "The Turing machine simulates computation. "
    "Turing founded theoretical computer science. "
    "Python is a programming language. "
    "Python supports functional programming."
)

session = learner.learn_from_text(text)
facts_count = len(learner.learned_facts)
print(f"Learning session: {facts_count} facts learned")
print()

# ------------------------------------------------------------------
# 3. Inspect the extracted facts
# ------------------------------------------------------------------
print("Extracted facts (first 10):")
for fact in learner.learned_facts[:10]:
    print(f"  [{fact.relation}]  {fact.subject}  →  {fact.object}")

print()

# ------------------------------------------------------------------
# 4. Inspect the semantic graph
# ------------------------------------------------------------------
mem = learner.semantic
print(f"Concepts in memory: {list(mem.concept_graph.nodes())[:15]}")
print()
print("Relations (first 10):")
for s, t, data in list(mem.concept_graph.edges(data=True))[:10]:
    rel = data.get("relation", "?")
    print(f"  {s}  --[{rel}]-->  {t}")
