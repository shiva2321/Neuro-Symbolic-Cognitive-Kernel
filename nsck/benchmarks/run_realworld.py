"""CLI entry point for NSCK V5 real-world benchmarks.

Usage
-----
    python run_realworld.py
    python run_realworld.py --output-dir /tmp/results
"""
import argparse
import os
import sys

# Add the nsck root so that `from python.core...` imports resolve correctly.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    """Run the real-world benchmark harness and produce results."""
    parser = argparse.ArgumentParser(
        description="NSCK V5 Real-World Benchmark Runner"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to write JSON and Markdown results (default: benchmarks/results/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    md_path = os.path.join(args.output_dir, "benchmark_report.md")

    from realworld_harness import run_all_benchmarks
    results = run_all_benchmarks(output_json=json_path, output_md=md_path)

    print("\n=== NSCK V5 Benchmark Complete ===")
    if "iris" in results:
        print(f"Iris accuracy:    {results['iris']['accuracy']:.1%}")
    if "text" in results:
        print(f"Text accuracy:    {results['text']['accuracy']:.1%}")
    print(f"Results: {json_path}")
    print(f"Report:  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
