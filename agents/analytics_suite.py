"""
Agent 5: The CGLB Analytics Suite (Tester)
Role: Performance Auditor & Interpretability Specialist

Evaluates the model using the Continual Graph Learning Benchmark (CGLB) and visualizes
engram formation with Hebbian trace maps.

Features:
- Average Performance (AP) tracking for skill measurement
- Average Forgetting (AF) tracking for stability measurement
- Membership Inference Attacks (MIA) for privacy verification
- Hebbian Trace visualization for cognitive path analysis
- Comprehensive benchmarking and reporting
"""

import torch
import torch.nn as nn
import numpy as np
import dgl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CGLBConfig:
    """Configuration for CGLB benchmarking"""
    num_tasks: int = 5
    eval_frequency: int = 1  # Evaluate every N tasks
    track_forgetting: bool = True
    track_transfer: bool = True
    run_mia: bool = True
    mia_attack_samples: int = 1000
    visualize_hebbian: bool = True
    output_dir: Path = Path("./cglb_results")


@dataclass
class TaskResult:
    """Results for a single task"""
    task_id: int
    task_name: str
    accuracy: float
    loss: float
    training_time: float
    num_samples: int
    timestamp: float


@dataclass
class CGLBMetrics:
    """CGLB benchmark metrics"""
    average_performance: float  # AP - average accuracy across all tasks
    average_forgetting: float  # AF - average forgetting across tasks
    forward_transfer: float  # How much learning task i helps with task j > i
    backward_transfer: float  # How much learning new tasks hurts old tasks
    final_performance: float  # Performance on final task
    total_time: float
    task_results: List[TaskResult] = field(default_factory=list)


class MembershipInferenceAttack:
    """
    Performs Membership Inference Attacks to verify information erasure.
    """

    def __init__(self):
        """Initialize MIA attacker"""
        self.attack_model = None

    def train_attack_model(self, member_scores: torch.Tensor,
                          non_member_scores: torch.Tensor):
        """
        Train a simple attack model to distinguish members from non-members.

        Args:
            member_scores: Scores (e.g., confidence) for member samples
            non_member_scores: Scores for non-member samples
        """
        # Simple threshold-based attack
        # In practice, use a more sophisticated ML model

        # Combine data
        X = torch.cat([member_scores, non_member_scores], dim=0)
        y = torch.cat([
            torch.ones(len(member_scores)),
            torch.zeros(len(non_member_scores))
        ], dim=0)

        # Train logistic regression
        from sklearn.linear_model import LogisticRegression

        self.attack_model = LogisticRegression()
        self.attack_model.fit(X.numpy().reshape(-1, 1), y.numpy())

        logger.info("Trained MIA attack model")

    def evaluate_attack(self, test_member_scores: torch.Tensor,
                       test_non_member_scores: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the attack success rate.

        Args:
            test_member_scores: Test member scores
            test_non_member_scores: Test non-member scores

        Returns:
            Dictionary with attack metrics
        """
        if self.attack_model is None:
            return {'error': 'Attack model not trained'}

        # Predict
        X_test = torch.cat([test_member_scores, test_non_member_scores], dim=0)
        y_test = torch.cat([
            torch.ones(len(test_member_scores)),
            torch.zeros(len(test_non_member_scores))
        ], dim=0)

        predictions = self.attack_model.predict(X_test.numpy().reshape(-1, 1))
        accuracy = (predictions == y_test.numpy()).mean()

        # Calculate precision and recall
        true_positives = ((predictions == 1) & (y_test.numpy() == 1)).sum()
        false_positives = ((predictions == 1) & (y_test.numpy() == 0)).sum()
        false_negatives = ((predictions == 0) & (y_test.numpy() == 1)).sum()

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        return {
            'attack_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'privacy_leakage': accuracy - 0.5  # Excess over random guessing
        }


class HebbianTraceVisualizer:
    """
    Visualizes Hebbian traces showing which connections were strengthened.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trace_history = []

    def record_trace(self, graph: dgl.DGLGraph, weight_changes: torch.Tensor,
                    task_name: str):
        """
        Record Hebbian trace for a task.

        Args:
            graph: DGL graph
            weight_changes: Changes in edge weights
            task_name: Name of the task
        """
        self.trace_history.append({
            'task_name': task_name,
            'graph': graph,
            'weight_changes': weight_changes.clone(),
            'timestamp': time.time()
        })

    def visualize_trace(self, task_name: str, top_k: int = 50):
        """
        Visualize the Hebbian trace for a specific task.

        Args:
            task_name: Task name
            top_k: Number of top edges to show
        """
        # Find task in history
        task_traces = [t for t in self.trace_history if t['task_name'] == task_name]

        if not task_traces:
            logger.warning(f"No traces found for task: {task_name}")
            return

        trace = task_traces[-1]  # Most recent
        graph = trace['graph']
        weight_changes = trace['weight_changes']

        # Get top-k edges by weight change
        top_indices = torch.topk(weight_changes.abs(), k=min(top_k, len(weight_changes)))[1]

        # Create subgraph with top edges
        src, dst = graph.edges()
        top_src = src[top_indices]
        top_dst = dst[top_indices]
        top_weights = weight_changes[top_indices]

        # Convert to NetworkX for visualization
        nx_graph = nx.DiGraph()

        for s, d, w in zip(top_src.numpy(), top_dst.numpy(), top_weights.numpy()):
            nx_graph.add_edge(int(s), int(d), weight=float(w))

        # Visualize
        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)

        # Color edges by weight change (red=strengthened, blue=weakened)
        edge_colors = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]

        # Draw
        nx.draw_networkx_nodes(nx_graph, pos, node_size=100, node_color='lightblue')
        nx.draw_networkx_edges(nx_graph, pos, edge_color=edge_colors,
                              edge_cmap=plt.cm.RdBu, width=2,
                              edge_vmin=-max(abs(w) for w in edge_colors),
                              edge_vmax=max(abs(w) for w in edge_colors))
        nx.draw_networkx_labels(nx_graph, pos, font_size=8)

        plt.title(f"Hebbian Trace: {task_name}\n(Top {len(nx_graph.edges())} strengthened connections)")
        plt.axis('off')
        plt.tight_layout()

        # Save
        save_path = self.output_dir / f"hebbian_trace_{task_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Hebbian trace visualization to {save_path}")

    def generate_trace_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive trace report.

        Returns:
            Dictionary with trace statistics
        """
        if not self.trace_history:
            return {'error': 'No traces recorded'}

        report = {
            'num_tasks': len(self.trace_history),
            'tasks': []
        }

        for trace in self.trace_history:
            weight_changes = trace['weight_changes']

            task_stats = {
                'task_name': trace['task_name'],
                'mean_weight_change': weight_changes.mean().item(),
                'max_weight_change': weight_changes.max().item(),
                'min_weight_change': weight_changes.min().item(),
                'num_strengthened': (weight_changes > 0).sum().item(),
                'num_weakened': (weight_changes < 0).sum().item(),
                'total_change': weight_changes.abs().sum().item()
            }

            report['tasks'].append(task_stats)

        return report


class AnalyticsSuite:
    """
    Main analytics suite for comprehensive evaluation.
    """

    def __init__(self, config: Optional[CGLBConfig] = None):
        """
        Initialize the Analytics Suite.

        Args:
            config: Configuration object
        """
        self.config = config or CGLBConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.mia_attacker = MembershipInferenceAttack()
        self.hebbian_visualizer = HebbianTraceVisualizer(self.config.output_dir)

        # Tracking
        self.task_accuracies: Dict[int, List[float]] = defaultdict(list)
        self.task_results: List[TaskResult] = []
        self.initial_accuracies: Dict[int, float] = {}

        logger.info(f"AnalyticsSuite initialized. Output: {self.config.output_dir}")

    def evaluate_task(self, model: Any, graph: dgl.DGLGraph,
                     features: torch.Tensor, labels: torch.Tensor,
                     task_id: int, task_name: str) -> TaskResult:
        """
        Evaluate model on a specific task.

        Args:
            model: Model to evaluate
            graph: Test graph
            features: Node features
            labels: Ground truth labels
            task_id: Task identifier
            task_name: Task name

        Returns:
            TaskResult object
        """
        start_time = time.time()

        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'predict'):
                predictions = model.predict(features)
            elif hasattr(model, 'forward'):
                predictions = model(features)
            else:
                predictions = features  # Fallback

            # Calculate metrics
            if labels.dim() == 1:
                pred_labels = predictions.argmax(dim=1) if predictions.dim() > 1 else predictions
                accuracy = (pred_labels == labels).float().mean().item()
                loss = 0.0  # Simplified
            else:
                accuracy = ((predictions > 0.5) == (labels > 0.5)).float().mean().item()
                loss = torch.nn.functional.mse_loss(predictions, labels).item()

        elapsed_time = time.time() - start_time

        result = TaskResult(
            task_id=task_id,
            task_name=task_name,
            accuracy=accuracy,
            loss=loss,
            training_time=elapsed_time,
            num_samples=len(labels),
            timestamp=time.time()
        )

        # Track accuracy
        self.task_accuracies[task_id].append(accuracy)

        # Store initial accuracy for forgetting calculation
        if task_id not in self.initial_accuracies:
            self.initial_accuracies[task_id] = accuracy

        self.task_results.append(result)

        logger.info(f"Task {task_id} ({task_name}): Accuracy={accuracy:.4f}, Loss={loss:.4f}")

        return result

    def compute_average_performance(self) -> float:
        """
        Compute Average Performance (AP) across all tasks.

        Returns:
            Average performance score
        """
        if not self.task_results:
            return 0.0

        # Get latest accuracy for each task
        latest_accuracies = {}
        for result in self.task_results:
            latest_accuracies[result.task_id] = result.accuracy

        ap = np.mean(list(latest_accuracies.values()))

        logger.info(f"Average Performance (AP): {ap:.4f}")
        return ap

    def compute_average_forgetting(self) -> float:
        """
        Compute Average Forgetting (AF) across all tasks.

        Returns:
            Average forgetting score
        """
        if not self.initial_accuracies:
            return 0.0

        forgetting_scores = []

        for task_id, initial_acc in self.initial_accuracies.items():
            if task_id in self.task_accuracies and len(self.task_accuracies[task_id]) > 1:
                current_acc = self.task_accuracies[task_id][-1]
                forgetting = max(0, initial_acc - current_acc)
                forgetting_scores.append(forgetting)

        if not forgetting_scores:
            return 0.0

        af = np.mean(forgetting_scores)

        logger.info(f"Average Forgetting (AF): {af:.4f}")
        return af

    def compute_forward_transfer(self) -> float:
        """
        Compute forward transfer (how learning helps future tasks).

        Returns:
            Forward transfer score
        """
        if len(self.initial_accuracies) < 2:
            return 0.0

        # Compare initial accuracy on task i with zero-shot baseline
        # Positive transfer = better than random

        baseline = 0.5  # Assume binary classification baseline
        transfers = []

        for task_id, initial_acc in self.initial_accuracies.items():
            if task_id > 0:  # Skip first task
                transfer = initial_acc - baseline
                transfers.append(transfer)

        return np.mean(transfers) if transfers else 0.0

    def run_membership_inference_attack(self, model: Any,
                                       train_data: Tuple[torch.Tensor, torch.Tensor],
                                       test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Run Membership Inference Attack to verify information erasure.

        Args:
            model: Trained model
            train_data: (features, labels) for training set
            test_data: (features, labels) for test set

        Returns:
            Dictionary with MIA results
        """
        logger.info("Running Membership Inference Attack...")

        train_features, train_labels = train_data
        test_features, test_labels = test_data

        # Get model confidence scores
        with torch.no_grad():
            if hasattr(model, 'predict'):
                train_scores = model.predict(train_features)
                test_scores = model.predict(test_features)
            else:
                train_scores = train_features
                test_scores = test_features

            # Use max probability as confidence
            if train_scores.dim() > 1:
                train_confidence = train_scores.max(dim=1)[0]
                test_confidence = test_scores.max(dim=1)[0]
            else:
                train_confidence = train_scores.abs()
                test_confidence = test_scores.abs()

        # Sample for efficiency
        num_samples = min(self.config.mia_attack_samples, len(train_confidence), len(test_confidence))

        train_sample = train_confidence[torch.randperm(len(train_confidence))[:num_samples]]
        test_sample = test_confidence[torch.randperm(len(test_confidence))[:num_samples]]

        # Train attack model
        self.mia_attacker.train_attack_model(train_sample, test_sample)

        # Evaluate attack
        results = self.mia_attacker.evaluate_attack(train_sample, test_sample)

        logger.info(f"MIA Results: Attack Accuracy={results['attack_accuracy']:.4f}, "
                   f"Privacy Leakage={results['privacy_leakage']:.4f}")

        return results

    def visualize_hebbian_traces(self, graph: dgl.DGLGraph,
                                 weight_changes: torch.Tensor,
                                 task_name: str):
        """
        Visualize Hebbian traces for a task.

        Args:
            graph: DGL graph
            weight_changes: Weight change tensor
            task_name: Task name
        """
        if self.config.visualize_hebbian:
            self.hebbian_visualizer.record_trace(graph, weight_changes, task_name)
            self.hebbian_visualizer.visualize_trace(task_name)

    def generate_cglb_report(self) -> CGLBMetrics:
        """
        Generate comprehensive CGLB benchmark report.

        Returns:
            CGLBMetrics object
        """
        logger.info("Generating CGLB benchmark report...")

        ap = self.compute_average_performance()
        af = self.compute_average_forgetting()
        ft = self.compute_forward_transfer()
        bt = -af  # Backward transfer is negative forgetting

        final_performance = self.task_results[-1].accuracy if self.task_results else 0.0
        total_time = sum(r.training_time for r in self.task_results)

        metrics = CGLBMetrics(
            average_performance=ap,
            average_forgetting=af,
            forward_transfer=ft,
            backward_transfer=bt,
            final_performance=final_performance,
            total_time=total_time,
            task_results=self.task_results
        )

        # Save report to JSON
        report_path = self.config.output_dir / "cglb_report.json"

        report_dict = {
            'average_performance': ap,
            'average_forgetting': af,
            'forward_transfer': ft,
            'backward_transfer': bt,
            'final_performance': final_performance,
            'total_time': total_time,
            'task_results': [
                {
                    'task_id': r.task_id,
                    'task_name': r.task_name,
                    'accuracy': r.accuracy,
                    'loss': r.loss,
                    'training_time': r.training_time,
                    'num_samples': r.num_samples
                }
                for r in self.task_results
            ]
        }

        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"CGLB report saved to {report_path}")

        # Generate Hebbian trace report
        trace_report = self.hebbian_visualizer.generate_trace_report()
        trace_report_path = self.config.output_dir / "hebbian_trace_report.json"

        with open(trace_report_path, 'w') as f:
            json.dump(trace_report, f, indent=2)

        logger.info(f"Hebbian trace report saved to {trace_report_path}")

        return metrics

    def plot_learning_curves(self):
        """
        Plot learning curves showing performance over time.
        """
        plt.figure(figsize=(12, 6))

        # Plot accuracy for each task over time
        for task_id, accuracies in self.task_accuracies.items():
            plt.plot(accuracies, marker='o', label=f'Task {task_id}')

        plt.xlabel('Evaluation Step')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves: Task Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        save_path = self.config.output_dir / "learning_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Learning curves saved to {save_path}")

    def export_results(self) -> Dict[str, Any]:
        """
        Export all results for external analysis.

        Returns:
            Dictionary with all results
        """
        return {
            'task_accuracies': {k: v for k, v in self.task_accuracies.items()},
            'task_results': [
                {
                    'task_id': r.task_id,
                    'task_name': r.task_name,
                    'accuracy': r.accuracy,
                    'loss': r.loss,
                    'training_time': r.training_time,
                    'num_samples': r.num_samples,
                    'timestamp': r.timestamp
                }
                for r in self.task_results
            ],
            'initial_accuracies': self.initial_accuracies,
            'metrics': {
                'average_performance': self.compute_average_performance(),
                'average_forgetting': self.compute_average_forgetting(),
                'forward_transfer': self.compute_forward_transfer()
            }
        }

