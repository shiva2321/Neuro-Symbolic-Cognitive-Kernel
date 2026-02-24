"""10 NLG quality checks."""

NLG_PROMPTS = [
    "Explain what a neural network is.",
    "What is the capital of France?",
    "Describe the water cycle.",
    "What is photosynthesis?",
    "Explain gravity.",
    "What is machine learning?",
    "Describe how memory works.",
    "What is artificial intelligence?",
    "Explain the concept of reasoning.",
    "What is consciousness?",
]


def run_nlg_benchmark(engine=None) -> float:
    """Run NLG quality checks. Returns % meeting fluency criteria (0-100)."""
    try:
        if engine is None:
            from python.core.reasoning.cognitive_engine import CognitiveEngine
            engine = CognitiveEngine()
        correct = 0
        for prompt in NLG_PROMPTS:
            try:
                response = engine.process_dialogue(prompt)
                if response and len(str(response)) > 10 and ' ' in str(response):
                    correct += 1
            except Exception:
                pass
        return correct / len(NLG_PROMPTS) * 100
    except Exception:
        return 0.0
