"""NLU Accuracy Benchmark for NSCK V3."""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, Any

# Ensure project root is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_NSCK_ROOT = os.path.join(_HERE, "..")
if _NSCK_ROOT not in sys.path:
    sys.path.insert(0, _NSCK_ROOT)

TEST_SENTENCES = [
    "The cat sat on the mat.",
    "Water is essential for life.",
    "Paris is the capital of France.",
    "Scientists discovered a new planet.",
    "Dogs and cats are common pets.",
    "The Earth orbits the Sun.",
    "Gravity causes objects to fall.",
    "Plants produce oxygen from sunlight.",
    "The brain controls the nervous system.",
    "John has a red car.",
    "Mary is a scientist at MIT.",
    "The computer processes data quickly.",
    "Birds fly because they have wings.",
    "Fish live in water.",
    "The moon orbits the Earth.",
    "Energy cannot be destroyed.",
    "Atoms make up all matter.",
    "The heart pumps blood through the body.",
    "Trees convert carbon dioxide into oxygen.",
    "Knowledge is power.",
]


def run() -> Dict[str, Any]:
    from python.core.language.text_knowledge_learner import TextKnowledgeLearner
    from python.core.memory.semantic_memory import SemanticMemory
    from python.core.memory.episodic_memory import EpisodicMemory

    sem = SemanticMemory(use_rust=False)
    epi = EpisodicMemory()
    learner = TextKnowledgeLearner(sem, epi)

    results = []
    total_relations = 0
    start_all = time.perf_counter()

    for sentence in TEST_SENTENCES:
        t0 = time.perf_counter()
        try:
            learner.learn_from_text(sentence)
            elapsed = time.perf_counter() - t0
            # Count edges added
            n_relations = sem.concept_graph.number_of_edges()
        except Exception as e:
            elapsed = time.perf_counter() - t0
            n_relations = 0
        results.append({"sentence": sentence, "elapsed_s": round(elapsed, 4)})
        total_relations = sem.concept_graph.number_of_edges()

    total_elapsed = time.perf_counter() - start_all
    summary = {
        "benchmark": "nlu",
        "sentences": len(TEST_SENTENCES),
        "total_relations_learned": total_relations,
        "avg_relations_per_sentence": round(total_relations / len(TEST_SENTENCES), 2),
        "total_elapsed_s": round(total_elapsed, 4),
        "avg_elapsed_s": round(total_elapsed / len(TEST_SENTENCES), 6),
        "details": results,
    }

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    out_path = os.path.join(_HERE, "results", "bench_nlu.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
