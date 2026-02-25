"""5 cross-domain transfer tests using TransferEngine."""

TRANSFER_TASKS = [
    ("physics", "finance", "momentum → price momentum: fast-moving price continues"),
    ("physics", "finance", "resistance → support level: price resists falling below"),
    ("physics", "biology", "energy conservation → calorie balance in metabolism"),
    ("physics", "economics", "equilibrium → market equilibrium: supply equals demand"),
    ("mechanics", "social", "friction → social friction slows group momentum"),
]


def run_transfer_benchmark(engine=None) -> float:
    """Run cross-domain transfer tests. Returns % with analogies found (0-100)."""
    try:
        from python.core.reasoning.analogy import AnalogyEngine
        ae = AnalogyEngine(load_defaults=True)
        correct = 0
        for src, tgt, desc in TRANSFER_TASKS:
            try:
                results = ae.find_analogies(src, tgt, top_k=3)
                if results and len(results) > 0:
                    correct += 1
            except Exception:
                pass  # skip unavailable methods; don't inflate score
        return correct / len(TRANSFER_TASKS) * 100
    except Exception:
        return 0.0
