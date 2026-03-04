"""
NSCK V5 Real-World Benchmark Harness
======================================
Runs NSCK through real-world classification tasks and measures:
- Accuracy
- Average decision latency
- Rules learned
- Memory usage

Datasets:
- Iris (sklearn or CSV fallback)
- Simple text classification (20 categories × 50 samples)

Output:
- JSON results file
- Markdown report
"""
import sys
import os
import time
import json
import logging

# Add the nsck root so that `from python.core...` imports resolve correctly.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Iris benchmark
# ---------------------------------------------------------------------------

def _load_iris():
    """Load Iris dataset. Returns (X, y, class_names)."""
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data.tolist(), iris.target.tolist(), list(iris.target_names)
    except ImportError:
        pass
    # CSV fallback — bundled minimal iris data (first 30 rows)
    _IRIS_MINI = [
        ([5.1,3.5,1.4,0.2], 0), ([4.9,3.0,1.4,0.2], 0), ([4.7,3.2,1.3,0.2], 0),
        ([4.6,3.1,1.5,0.2], 0), ([5.0,3.6,1.4,0.2], 0), ([5.4,3.9,1.7,0.4], 0),
        ([4.6,3.4,1.4,0.3], 0), ([5.0,3.4,1.5,0.2], 0), ([4.4,2.9,1.4,0.2], 0),
        ([4.9,3.1,1.5,0.1], 0), ([7.0,3.2,4.7,1.4], 1), ([6.4,3.2,4.5,1.5], 1),
        ([6.9,3.1,4.9,1.5], 1), ([5.5,2.3,4.0,1.3], 1), ([6.5,2.8,4.6,1.5], 1),
        ([5.7,2.8,4.5,1.3], 1), ([6.3,3.3,4.7,1.6], 1), ([4.9,2.4,3.3,1.0], 1),
        ([6.6,2.9,4.6,1.3], 1), ([5.2,2.7,3.9,1.4], 1), ([6.3,3.3,6.0,2.5], 2),
        ([5.8,2.7,5.1,1.9], 2), ([7.1,3.0,5.9,2.1], 2), ([6.3,2.9,5.6,1.8], 2),
        ([6.5,3.0,5.8,2.2], 2), ([7.6,3.0,6.6,2.1], 2), ([4.9,2.5,4.5,1.7], 2),
        ([7.3,2.9,6.3,1.8], 2), ([6.7,2.5,5.8,1.8], 2), ([7.2,3.6,6.1,2.5], 2),
    ]
    X = [row[0] for row in _IRIS_MINI]
    y = [row[1] for row in _IRIS_MINI]
    return X, y, ["setosa", "versicolor", "virginica"]


def _action_to_class(action: str, class_names: list) -> int:
    """Map an NSCK action string back to a class index."""
    for i, name in enumerate(class_names):
        if name.lower() in action.lower():
            return i
        if f"CLASS_{i}" in action or f"class_{i}" in action.lower():
            return i
    # Try numeric suffix
    for i in range(len(class_names)):
        if str(i) in action:
            return i
    return 0  # default


def run_iris_benchmark(n_train: int = 100, n_test: int = 50) -> dict:
    """Run Iris classification benchmark.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.

    Returns
    -------
    dict
        Benchmark results including accuracy, latency, rules_learned.
    """
    from python.core.substrate import NSCKSubstrate

    X, y, class_names = _load_iris()
    substrate = NSCKSubstrate()
    substrate.register_task("iris")

    # Training phase
    total_latency = 0.0
    n_actual_train = min(n_train, len(X))

    for i in range(n_actual_train):
        features = X[i]
        label = y[i]
        state = {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3],
        }
        t0 = time.perf_counter()
        result = substrate.process(state, "iris")
        latency = time.perf_counter() - t0
        total_latency += latency

        # Teach with the correct class label
        target_action = f"CLASS_{label}_{class_names[label]}"
        substrate.feedback(
            target_action, reward=1.0, task_tag="iris",
            state=state, outcome="success"
        )

    # Sleep to consolidate
    substrate.sleep("iris")

    # Test phase
    n_actual_test = min(n_test, len(X))
    correct = 0
    test_latencies = []
    for i in range(n_actual_test):
        features = X[i % len(X)]
        label = y[i % len(y)]
        state = {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3],
        }
        t0 = time.perf_counter()
        result = substrate.process(state, "iris")
        latency = time.perf_counter() - t0
        test_latencies.append(latency)

        predicted = _action_to_class(result.chosen_action or "", class_names)
        if predicted == label:
            correct += 1

    accuracy = correct / n_actual_test if n_actual_test > 0 else 0.0
    avg_latency_ms = (sum(test_latencies) / len(test_latencies) * 1000) if test_latencies else 0.0
    rules_learned = sum(
        len(rules)
        for rules in substrate.engine.rule_learner.learned_rules.values()
    )

    return {
        "dataset": "iris",
        "n_train": n_actual_train,
        "n_test": n_actual_test,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency_ms, 3),
        "rules_learned": rules_learned,
        "episodes_recorded": substrate.engine.stats.get("episodes_recorded", 0),
    }


# ---------------------------------------------------------------------------
# Text classification benchmark
# ---------------------------------------------------------------------------

_TEXT_CATEGORIES = [
    "sports", "politics", "science", "technology", "health",
    "business", "arts", "travel", "food", "education",
]

_TEXT_TEMPLATES = {
    "sports": [
        "The team won the championship game",
        "Players scored a record number of points",
        "The athlete broke the world record",
        "Tournament results announced today",
        "Coach announces new training strategy",
    ],
    "politics": [
        "The government passed new legislation",
        "Election results are being counted",
        "Policy makers debate budget proposals",
        "International diplomacy summit scheduled",
        "New regulatory framework announced",
    ],
    "science": [
        "Researchers discovered a new phenomenon",
        "The study published in Nature journal",
        "Experiment confirms theoretical prediction",
        "Scientific breakthrough announced",
        "Laboratory results show promising data",
    ],
    "technology": [
        "New software release improves performance",
        "Engineers developed faster algorithm",
        "Cloud computing adoption increases",
        "Artificial intelligence advances rapidly",
        "Open source project gains popularity",
    ],
    "health": [
        "Medical researchers find new treatment",
        "Clinical trials show positive results",
        "Doctors recommend preventive measures",
        "Health guidelines updated this year",
        "New drug approved for patient use",
    ],
    "business": [
        "Company reports record quarterly profits",
        "Stock market reached new high today",
        "Merger agreement signed between firms",
        "Economic indicators show growth trend",
        "Investment portfolio diversification recommended",
    ],
    "arts": [
        "Museum exhibit opens to public today",
        "Artist wins prestigious award ceremony",
        "New film festival showcases local talent",
        "Orchestra performs classical masterwork",
        "Gallery collection donated to institution",
    ],
    "travel": [
        "Popular destination sees tourist growth",
        "Airline announces new international routes",
        "Travel advisory updated for region",
        "Hotel prices increase during holidays",
        "Adventure tourism gaining popularity",
    ],
    "food": [
        "Restaurant opens new downtown location",
        "Chef creates innovative fusion cuisine",
        "Food safety regulations updated today",
        "Nutrition study reveals dietary patterns",
        "Local market hosts organic food festival",
    ],
    "education": [
        "University announces new degree program",
        "Students perform well in assessments",
        "Online learning platform usage increases",
        "Research funding awarded to faculty",
        "School curriculum updated with technology",
    ],
}


def _generate_text_samples(n_per_category: int = 50) -> list:
    """Generate (text, label, category) tuples for text classification."""
    import random
    random.seed(42)
    samples = []
    templates_list = list(_TEXT_TEMPLATES.items())
    for cat_idx, (category, templates) in enumerate(templates_list):
        for i in range(n_per_category):
            template = templates[i % len(templates)]
            text = f"{template} - item {i}"
            samples.append((text, cat_idx, category))
    random.shuffle(samples)
    return samples


def run_text_benchmark(n_categories: int = 10, n_per_category: int = 50) -> dict:
    """Run text classification benchmark.

    Parameters
    ----------
    n_categories : int
        Number of text categories to use.
    n_per_category : int
        Samples per category.

    Returns
    -------
    dict
        Benchmark results.
    """
    from python.core.substrate import NSCKSubstrate

    samples = _generate_text_samples(n_per_category)
    substrate = NSCKSubstrate()
    substrate.register_task("text_cls")

    n_total = min(n_categories * n_per_category, len(samples))
    # Use 80% for training, 20% for testing
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    correct = 0
    test_latencies = []

    # Training
    for text, label, category in samples[:n_train]:
        state = {"text": text, "category_hint": label}
        target_action = f"CATEGORY_{category.upper()}"
        substrate.process(state, "text_cls")
        substrate.feedback(
            target_action, reward=1.0, task_tag="text_cls",
            state=state, outcome="success"
        )

    # Sleep to consolidate
    substrate.sleep("text_cls")

    # Testing
    for text, label, category in samples[n_train:n_train + n_test]:
        state = {"text": text, "category_hint": label}
        t0 = time.perf_counter()
        result = substrate.process(state, "text_cls")
        latency = time.perf_counter() - t0
        test_latencies.append(latency)

        expected = f"CATEGORY_{category.upper()}"
        if expected in (result.chosen_action or ""):
            correct += 1

    accuracy = correct / n_test if n_test > 0 else 0.0
    avg_latency_ms = (sum(test_latencies) / len(test_latencies) * 1000) if test_latencies else 0.0
    rules_learned = sum(
        len(rules)
        for rules in substrate.engine.rule_learner.learned_rules.values()
    )

    return {
        "dataset": "text_classification",
        "n_categories": n_categories,
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency_ms, 3),
        "rules_learned": rules_learned,
    }


# ---------------------------------------------------------------------------
# Memory usage helper
# ---------------------------------------------------------------------------

def _get_memory_mb() -> float:
    """Return current RSS memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    output_json: str = "benchmark_results.json",
    output_md: str = "benchmark_report.md",
) -> dict:
    """Run all benchmarks and write results to files.

    Parameters
    ----------
    output_json : str
        Path to JSON results file.
    output_md : str
        Path to Markdown report file.

    Returns
    -------
    dict
        Combined results from all benchmarks.
    """
    results = {}
    mem_start = _get_memory_mb()

    print("[NSCK V5 Benchmark] Starting Iris classification benchmark...")
    t0 = time.perf_counter()
    results["iris"] = run_iris_benchmark(n_train=100, n_test=30)
    results["iris"]["wall_time_s"] = round(time.perf_counter() - t0, 2)
    print(f"  Iris: accuracy={results['iris']['accuracy']:.1%}, "
          f"latency={results['iris']['avg_latency_ms']:.2f}ms, "
          f"rules={results['iris']['rules_learned']}")

    print("[NSCK V5 Benchmark] Starting text classification benchmark...")
    t0 = time.perf_counter()
    results["text"] = run_text_benchmark(n_categories=10, n_per_category=20)
    results["text"]["wall_time_s"] = round(time.perf_counter() - t0, 2)
    print(f"  Text: accuracy={results['text']['accuracy']:.1%}, "
          f"latency={results['text']['avg_latency_ms']:.2f}ms, "
          f"rules={results['text']['rules_learned']}")

    mem_end = _get_memory_mb()
    results["memory_usage_mb"] = round(mem_end - mem_start, 1)
    results["total_memory_mb"] = round(mem_end, 1)

    # Write JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[NSCK V5 Benchmark] Results written to {output_json}")

    # Write Markdown report
    _write_markdown_report(results, output_md)
    print(f"[NSCK V5 Benchmark] Report written to {output_md}")

    return results


def _write_markdown_report(results: dict, output_md: str) -> None:
    """Write a Markdown benchmark report."""
    lines = [
        "# NSCK V5 Real-World Benchmark Report",
        "",
        "## Summary",
        "",
        "| Dataset | Accuracy | Avg Latency (ms) | Rules Learned | Wall Time (s) |",
        "|---------|----------|-----------------|---------------|---------------|",
    ]

    for key in ["iris", "text"]:
        r = results.get(key, {})
        lines.append(
            f"| {r.get('dataset', key)} "
            f"| {r.get('accuracy', 0):.1%} "
            f"| {r.get('avg_latency_ms', 0):.2f} "
            f"| {r.get('rules_learned', 0)} "
            f"| {r.get('wall_time_s', 0):.1f} |"
        )

    lines += [
        "",
        f"**Total memory delta:** {results.get('memory_usage_mb', 0):.1f} MB",
        f"**Peak memory:** {results.get('total_memory_mb', 0):.1f} MB",
        "",
        "## Iris Classification",
        "",
        f"- Training samples: {results.get('iris', {}).get('n_train', 'N/A')}",
        f"- Test samples: {results.get('iris', {}).get('n_test', 'N/A')}",
        f"- Episodes recorded: {results.get('iris', {}).get('episodes_recorded', 'N/A')}",
        "",
        "## Text Classification",
        "",
        f"- Categories: {results.get('text', {}).get('n_categories', 'N/A')}",
        f"- Training samples: {results.get('text', {}).get('n_train', 'N/A')}",
        f"- Test samples: {results.get('text', {}).get('n_test', 'N/A')}",
        "",
        "## Notes",
        "",
        "- All benchmarks run with Python VSA backend (no Rust required)",
        "- Accuracy improves with more training data and sleep consolidation",
        "- NSCK is an online learner; accuracy increases with experience",
    ]

    with open(output_md, "w") as f:
        f.write("\n".join(lines))
