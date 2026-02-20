"""
examples/02_semantic_memory.py
================================
Demonstrates storing facts in SemanticMemory and using spreading activation
to retrieve related concepts.

Run from the repository root:
    python examples/02_semantic_memory.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck'))

from python.core.memory.semantic_memory import SemanticMemory

# ------------------------------------------------------------------
# 1. Create a semantic memory store
# ------------------------------------------------------------------
mem = SemanticMemory()

# ------------------------------------------------------------------
# 2. Add concepts and relations
# ------------------------------------------------------------------
for concept in ["Paris", "France", "Europe", "London", "UK"]:
    mem.add_concept(concept, {})

mem.add_relation("Paris",  "capital_of", "France")
mem.add_relation("France", "located_in", "Europe")
mem.add_relation("London", "capital_of", "UK")
mem.add_relation("UK",     "located_in", "Europe")

# ------------------------------------------------------------------
# 3. Spreading activation from a seed concept
# ------------------------------------------------------------------
print("Spreading activation from 'Paris':")
activated = mem.spread_activation(["Paris"], steps=2, decay=0.7)

for concept, score in sorted(activated.items(), key=lambda x: -x[1])[:8]:
    print(f"  {concept:<12}  activation={score:.3f}")

print()

# ------------------------------------------------------------------
# 4. Hypervector similarity search
# ------------------------------------------------------------------
print("Most similar concepts to 'France' (by HV similarity):")
query_hv = mem.get_concept("France")
if query_hv is not None:
    results = mem.query(query_hv, k=3)
    for name, sim in results:
        print(f"  {name:<12}  similarity={sim:.3f}")
