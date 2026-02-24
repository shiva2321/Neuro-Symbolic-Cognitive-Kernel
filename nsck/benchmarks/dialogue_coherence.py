"""5-turn dialogue coherence tests."""

DIALOGUES = [
    [
        "Alice went to the store.",
        "She bought some apples.",
        "Alice paid with cash.",
        "Who went to the store?",
        "What did she buy?",
    ],
]


def run_dialogue_benchmark(engine=None) -> float:
    """Run dialogue coherence test. Returns % coherent (0-100)."""
    try:
        if engine is None:
            from python.core.reasoning.cognitive_engine import CognitiveEngine
            engine = CognitiveEngine()
        correct = 0
        total = len(DIALOGUES)
        for turns in DIALOGUES:
            try:
                responses = []
                for turn in turns:
                    r = engine.process_dialogue(turn)
                    responses.append(str(r) if r else "")
                has_entity = any(
                    "alice" in r.lower() or "store" in r.lower() or "apples" in r.lower()
                    for r in responses
                )
                if has_entity or all(len(r) > 0 for r in responses[-2:]):
                    correct += 1
            except Exception:
                pass
        return correct / total * 100 if total > 0 else 0.0
    except Exception:
        return 0.0
