"""NSCK-ES Regression Gate.

Run this to check whether the current codebase meets the minimum score.

Usage:
    python -m eval.regression_gate
"""
from __future__ import annotations


NSCK_ES_BASELINE = 0.40
NSCK_ES_TOLERANCE = 0.05


def check_regression(verbose: bool = True) -> bool:
    """Run NSCK-ES and check against baseline.

    Returns True if passed (score >= baseline - tolerance).
    """
    from eval.nsck_eval_suite import NSCKEvalSuite

    suite = NSCKEvalSuite()
    results = suite.run_all()
    score = results["nsck_es"]
    passed = score >= NSCK_ES_BASELINE - NSCK_ES_TOLERANCE

    if verbose:
        print(f"NSCK-ES: {score:.3f} | {'PASS' if passed else 'FAIL'} "
              f"(baseline={NSCK_ES_BASELINE:.2f}, tol={NSCK_ES_TOLERANCE:.2f})")
        for k in ("t1", "t2", "t3", "t4", "t5"):
            print(f"  {k}: {results.get(k, 0.0):.3f}")

    return passed


if __name__ == "__main__":
    import sys
    ok = check_regression(verbose=True)
    sys.exit(0 if ok else 1)
