import sys
sys.path.insert(0, r'd:\Node_network\nsck')
try:
    from tests.benchmarks.full_architecture_benchmark import run_language_dialogue, run_snn_stress, run_sleep_consolidation, run_cross_domain_transfer, run_episodic_memory_pressure
    print("=== SNN Stress (short) ===")
    r = run_snn_stress(20)
    print(f"SNN success: {r.success_rate:.1%}")
    print("=== Language Dialogue (5 turns) ===")
    r = run_language_dialogue(5)
    print(f"Language success: {r.success_rate:.1%}")
    print("ALL PASS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED:", e)
