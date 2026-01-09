"""
NCGN Agents Demonstration
Showcases the five specialized agents working together in a complete pipeline.

Pipeline:
1. Data Harvester: Acquires and filters graph data
2. Topological Converter: Transforms into MC-TEG and Graph Words
3. Bottleneck Optimizer: Rewires graph to alleviate bottlenecks
4. Analytic Learner: Trains using backpropagation-free methods
5. Analytics Suite: Evaluates and visualizes results
"""

import torch
import numpy as np
import logging
from pathlib import Path

from agents.data_harvester import DataHarvester, DataHarvesterConfig
from agents.topological_converter import TopologicalConverter, MCTEGConfig, Graph2SeqConfig
from agents.bottleneck_optimizer import BottleneckOptimizer, RewiringConfig
from agents.analytic_learner import AnalyticLearner, AnalyticLearningConfig
from agents.analytics_suite import AnalyticsSuite, CGLBConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_pipeline(command: str = "Train for English language", run_full_pipeline: bool = True):
    """
    Run the complete NCGN agents pipeline.

    Args:
        command: Natural language command for data harvesting
        run_full_pipeline: Whether to run all agents or just data harvesting
    """
    print("=" * 80)
    print("NCGN SPECIALIZED AGENTS DEMONSTRATION")
    print("=" * 80)
    print()

    # ============================================================================
    # Agent 1: Data Harvester
    # ============================================================================
    print("\n" + "=" * 80)
    print("AGENT 1: AUTOMATED DATA HARVESTER")
    print("=" * 80)

    harvester_config = DataHarvesterConfig(
        cache_dir=Path("./data_cache"),
        max_nodes=100_000,  # Limit for demo
        use_semantic_filter=True,
        filter_threshold=0.3
    )

    harvester = DataHarvester(harvester_config)

    # List available datasets
    print("\nAvailable datasets:")
    datasets = harvester.list_available_datasets()
    for i, dataset in enumerate(datasets[:5]):  # Show first 5
        print(f"  {i+1}. {dataset.name} ({dataset.domain}) - {dataset.num_nodes:,} nodes")

    # Harvest data based on natural language command
    print(f"\nExecuting command: '{command}'")
    harvest_result = harvester.harvest(command)

    if not harvest_result['success']:
        print(f"Error: {harvest_result['error']}")
        return

    print(f"✓ Harvested dataset: {harvest_result['dataset_name']}")
    print(f"  Nodes: {harvest_result['num_nodes']:,}")
    print(f"  Edges: {harvest_result['num_edges']:,}")
    print(f"  Domain: {harvest_result['metadata'].domain}")

    graph = harvest_result['graph']

    if not run_full_pipeline:
        print("\nDemo complete (data harvesting only)")
        return

    # ============================================================================
    # Agent 2: Topological Converter
    # ============================================================================
    print("\n" + "=" * 80)
    print("AGENT 2: LINGUISTIC-TOPOLOGICAL CONVERTER")
    print("=" * 80)

    mcteg_config = MCTEGConfig(
        use_laplacian_pe=True,
        use_rwpe=True,
        num_laplacian_eigenvectors=8,
        rwpe_walk_length=20
    )

    graph2seq_config = Graph2SeqConfig(
        walk_length=30,
        num_walks_per_node=5,  # Reduced for demo
        graph_word_dim=128,
        vocab_size=10000
    )

    converter = TopologicalConverter(mcteg_config, graph2seq_config)

    # Generate sample node descriptions (in practice, these come from data)
    node_descriptions = {i: f"node_{i}" for i in range(min(100, graph.num_nodes()))}

    print("\nConverting graph to MC-TEG and Graph Words...")
    conversion_result = converter.convert(
        graph,
        node_descriptions=node_descriptions,
        generate_sequences=True
    )

    mcteg = conversion_result['mcteg']

    print(f"✓ MC-TEG constructed:")
    print(f"  Nodes: {conversion_result['num_nodes']:,}")
    print(f"  Edges: {conversion_result['num_edges']:,}")
    print(f"  Laplacian PE: {'✓' if 'laplacian_pe' in mcteg.ndata else '✗'}")
    print(f"  RWPE: {'✓' if 'rwpe' in mcteg.ndata else '✗'}")
    print(f"\n✓ Graph Words generated:")
    print(f"  Unique tokens: {conversion_result['num_unique_tokens']:,}")
    print(f"  Token sequences: {conversion_result['num_sequences']:,}")

    # ============================================================================
    # Agent 3: Bottleneck Optimizer
    # ============================================================================
    print("\n" + "=" * 80)
    print("AGENT 3: BOTTLENECK OPTIMIZER")
    print("=" * 80)

    rewiring_config = RewiringConfig(
        use_ricci_curvature=True,
        ricci_threshold=-0.5,
        max_edges_to_add=500,  # Reduced for demo
        max_edges_to_remove=200,
        add_virtual_node=True,
        rewiring_iterations=2  # Reduced for demo
    )

    optimizer = BottleneckOptimizer(rewiring_config)

    print("\nAnalyzing graph structure...")
    bottleneck_stats = optimizer.get_bottleneck_statistics(mcteg)

    print(f"✓ Bottleneck analysis:")
    print(f"  Bottleneck edges: {bottleneck_stats['num_bottleneck_edges']}")
    print(f"  High-betweenness nodes: {bottleneck_stats['num_high_betweenness_nodes']}")
    if 'mean_curvature' in bottleneck_stats:
        print(f"  Mean curvature: {bottleneck_stats['mean_curvature']:.4f}")
        print(f"  Negative curvature ratio: {bottleneck_stats['negative_curvature_ratio']:.2%}")

    print("\nOptimizing graph structure...")
    optimization_result = optimizer.optimize(mcteg, num_iterations=2)

    optimized_graph = optimization_result['optimized_graph']

    print(f"✓ Optimization complete:")
    print(f"  Original edges: {optimization_result['original_edges']:,}")
    print(f"  Final edges: {optimization_result['final_edges']:,}")
    print(f"  Edges added: {optimization_result['edges_added_total']}")
    print(f"  Edges removed: {optimization_result['edges_removed_total']}")
    print(f"  Virtual node: {'✓' if optimization_result['virtual_node_id'] is not None else '✗'}")

    # ============================================================================
    # Agent 4: Analytic Learner
    # ============================================================================
    print("\n" + "=" * 80)
    print("AGENT 4: ONE-PASS ANALYTIC LEARNER")
    print("=" * 80)

    analytic_config = AnalyticLearningConfig(
        use_rls=True,
        forgetting_factor=0.99,
        use_dynamic_threshold=True,
        use_implicit_diff=False,  # Disabled for demo speed
        memory_efficient=True
    )

    learner = AnalyticLearner(analytic_config)

    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    num_nodes = optimized_graph.num_nodes()
    feature_dim = 128
    num_classes = 5

    # Use positional encodings as features
    if 'laplacian_pe' in optimized_graph.ndata:
        features = optimized_graph.ndata['laplacian_pe']
        # Pad to feature_dim
        if features.shape[1] < feature_dim:
            padding = torch.zeros(num_nodes, feature_dim - features.shape[1])
            features = torch.cat([features, padding], dim=1)
        else:
            features = features[:, :feature_dim]
    else:
        features = torch.randn(num_nodes, feature_dim)

    # Random labels for demo
    labels = torch.randint(0, num_classes, (num_nodes,))

    print("\nTraining with Recursive Least Squares...")
    train_metrics = learner.train_step(optimized_graph, features, labels, layer_name='demo_classifier')

    print(f"✓ Training complete:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Samples processed: {train_metrics['samples_seen']:,}")
    print(f"  RLS updates: {train_metrics['num_updates']}")

    # Demonstrate dynamic threshold learning
    print("\nTraining with Dynamic Threshold Neurons...")
    threshold_metrics = learner.train_with_dynamic_threshold(
        optimized_graph, features, labels, layer_name='threshold_demo'
    )

    print(f"✓ Dynamic threshold training:")
    print(f"  Accuracy: {threshold_metrics['accuracy']:.4f}")
    print(f"  Average threshold: {threshold_metrics['avg_threshold']:.4f}")

    # Demonstrate continual learning
    print("\nDemonstrating continual learning...")

    # Simulate new data stream
    new_features = torch.randn(min(1000, num_nodes), feature_dim)
    new_labels = torch.randint(0, num_classes, (len(new_features),))

    continual_metrics = learner.continual_learn(
        optimized_graph, new_features, new_labels, layer_name='demo_classifier'
    )

    print(f"✓ Continual learning:")
    print(f"  New accuracy: {continual_metrics['accuracy']:.4f}")
    print(f"  Total samples: {continual_metrics['samples_seen']:,}")

    # ============================================================================
    # Agent 5: Analytics Suite
    # ============================================================================
    print("\n" + "=" * 80)
    print("AGENT 5: CGLB ANALYTICS SUITE")
    print("=" * 80)

    cglb_config = CGLBConfig(
        num_tasks=3,  # Reduced for demo
        track_forgetting=True,
        run_mia=False,  # Disabled for demo speed
        visualize_hebbian=True,
        output_dir=Path("./cglb_results")
    )

    analytics = AnalyticsSuite(cglb_config)

    # Simulate multi-task evaluation
    print("\nEvaluating on multiple tasks...")

    tasks = [
        ("task_1", "English Language", features[:len(features)//3], labels[:len(labels)//3]),
        ("task_2", "Citation Network", features[len(features)//3:2*len(features)//3],
         labels[len(labels)//3:2*len(labels)//3]),
        ("task_3", "Social Network", features[2*len(features)//3:], labels[2*len(labels)//3:])
    ]

    for task_id, (task_name, task_desc, task_features, task_labels) in enumerate(tasks):
        print(f"\n  Task {task_id + 1}: {task_desc}")

        # Get RLS estimator
        rls_estimator = learner.rls_estimators.get('demo_classifier')

        result = analytics.evaluate_task(
            rls_estimator, optimized_graph, task_features, task_labels,
            task_id=task_id, task_name=task_name
        )

        # Visualize Hebbian traces
        weight_changes = torch.randn(optimized_graph.num_edges())  # Simulated
        analytics.visualize_hebbian_traces(optimized_graph, weight_changes, task_name)

    # Generate comprehensive report
    print("\nGenerating CGLB benchmark report...")
    cglb_metrics = analytics.generate_cglb_report()

    print(f"\n✓ CGLB Benchmark Results:")
    print(f"  Average Performance (AP): {cglb_metrics.average_performance:.4f}")
    print(f"  Average Forgetting (AF): {cglb_metrics.average_forgetting:.4f}")
    print(f"  Forward Transfer: {cglb_metrics.forward_transfer:.4f}")
    print(f"  Backward Transfer: {cglb_metrics.backward_transfer:.4f}")
    print(f"  Final Performance: {cglb_metrics.final_performance:.4f}")
    print(f"  Total Time: {cglb_metrics.total_time:.2f}s")

    # Plot learning curves
    analytics.plot_learning_curves()

    print(f"\n✓ Results saved to: {cglb_config.output_dir}")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 80)
    print("\n✓ All 5 agents executed successfully:")
    print("  1. Data Harvester: Acquired and filtered graph data")
    print("  2. Topological Converter: Generated MC-TEG and Graph Words")
    print("  3. Bottleneck Optimizer: Rewired graph structure")
    print("  4. Analytic Learner: Trained without backpropagation")
    print("  5. Analytics Suite: Evaluated and visualized results")
    print("\n" + "=" * 80)


def quick_demo():
    """
    Quick demonstration with minimal computation.
    """
    print("\n🚀 QUICK DEMO - Data Harvesting Only\n")
    demo_pipeline(command="Train for English language", run_full_pipeline=False)


def full_demo():
    """
    Full demonstration with all agents.
    """
    print("\n🚀 FULL DEMO - Complete Pipeline\n")
    demo_pipeline(command="Train for English language", run_full_pipeline=True)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("NCGN SPECIALIZED AGENTS - DEMONSTRATION")
    print("=" * 80)
    print("\nSelect demo mode:")
    print("  1. Quick Demo (Data Harvesting only)")
    print("  2. Full Demo (All 5 agents)")
    print()

    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        quick_demo()
    elif choice == "2":
        full_demo()
    else:
        print("Invalid choice. Running quick demo.")
        quick_demo()

    print("\n✨ Demo complete!\n")

