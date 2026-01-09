"""
Metrics Monitor for NCGN Dashboard
Tracks training metrics, continual learning stats, and convergence.
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict, deque
import time


class MetricsMonitor:
    """
    Monitors and tracks all training metrics for the dashboard.
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics monitor.

        Args:
            history_size: Number of historical points to keep
        """
        self.history_size = history_size

        # Metric histories
        self.metrics = defaultdict(lambda: deque(maxlen=history_size))

        # Continual learning metrics
        self.task_performance = {}  # {task_id: [accuracies]}
        self.forgetting_matrix = {}  # {(task_i, task_j): forgetting}

        # Convergence tracking
        self.loss_history = deque(maxlen=history_size)
        self.convergence_threshold = 1e-4

        # Timestamps
        self.timestamps = deque(maxlen=history_size)

    def log_metric(self, name: str, value: float, step: int = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        self.metrics[name].append({
            'value': value,
            'step': step if step is not None else len(self.metrics[name]),
            'timestamp': time.time()
        })

        self.timestamps.append(time.time())

    def log_batch_metrics(self, metrics: Dict[str, float], step: int):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def compute_average_performance(self, task_id: int) -> float:
        """
        Compute Average Performance (AP) for continual learning.

        AP = (1/T) * Σ R_{T,i}
        where R_{T,i} is accuracy on task i after learning all T tasks.

        Args:
            task_id: Current task ID

        Returns:
            Average performance across all tasks
        """
        if not self.task_performance:
            return 0.0

        total_acc = 0.0
        count = 0

        for tid, accs in self.task_performance.items():
            if tid <= task_id and accs:
                total_acc += accs[-1]  # Latest accuracy
                count += 1

        return total_acc / max(count, 1)

    def compute_average_forgetting(self, task_id: int) -> float:
        """
        Compute Average Forgetting (AF) for continual learning.

        AF = (1/(T-1)) * Σ_{i=1}^{T-1} (max_j∈{i,...,T-1} R_{j,i} - R_{T,i})

        Args:
            task_id: Current task ID

        Returns:
            Average forgetting across previous tasks
        """
        if task_id <= 1:
            return 0.0

        total_forgetting = 0.0
        count = 0

        for tid in range(task_id):
            if tid in self.task_performance:
                accs = self.task_performance[tid]
                if len(accs) >= 2:
                    max_acc = max(accs)
                    current_acc = accs[-1]
                    forgetting = max(0, max_acc - current_acc)
                    total_forgetting += forgetting
                    count += 1

        return total_forgetting / max(count, 1)

    def update_task_performance(self, task_id: int, accuracy: float):
        """Update performance for a specific task"""
        if task_id not in self.task_performance:
            self.task_performance[task_id] = []

        self.task_performance[task_id].append(accuracy)

    def check_convergence(self, window_size: int = 20) -> bool:
        """
        Check if training has converged based on loss stability.

        Args:
            window_size: Number of recent steps to check

        Returns:
            True if converged
        """
        if len(self.loss_history) < window_size:
            return False

        recent_losses = list(self.loss_history)[-window_size:]
        loss_std = np.std(recent_losses)

        return loss_std < self.convergence_threshold

    def get_edge_of_chaos_metric(self) -> float:
        """
        Compute 'edge of chaos' metric for SNN.

        Measures balance between order and chaos in spiking dynamics.
        Returns value in [0, 1] where 0.5 is ideal.
        """
        if 'spike_rate' not in self.metrics or len(self.metrics['spike_rate']) == 0:
            return 0.5

        # Get recent spike rates
        recent_rates = [m['value'] for m in list(self.metrics['spike_rate'])[-50:]]

        if not recent_rates:
            return 0.5

        # Calculate coefficient of variation (CV)
        mean_rate = np.mean(recent_rates)
        std_rate = np.std(recent_rates)

        if mean_rate == 0:
            return 0.0

        cv = std_rate / mean_rate

        # Map CV to [0, 1] where CV=1 (edge of chaos) maps to 0.5
        edge_metric = 1.0 / (1.0 + np.exp(-5 * (cv - 1.0)))

        return edge_metric

    def get_latest(self) -> Dict[str, Any]:
        """Get latest metrics"""
        latest = {}

        for name, history in self.metrics.items():
            if history:
                latest[name] = history[-1]['value']

        return latest

    def get_history(self, metric_name: str = None) -> Dict[str, List]:
        """
        Get metric history.

        Args:
            metric_name: Specific metric name, or None for all

        Returns:
            Dictionary of metric histories
        """
        if metric_name:
            return {
                metric_name: [
                    {'step': m['step'], 'value': m['value'], 'timestamp': m['timestamp']}
                    for m in self.metrics[metric_name]
                ]
            }

        # Return all metrics
        result = {}
        for name, history in self.metrics.items():
            result[name] = [
                {'step': m['step'], 'value': m['value'], 'timestamp': m['timestamp']}
                for m in history
            ]

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        summary = {
            'total_steps': len(self.timestamps),
            'latest_metrics': self.get_latest(),
            'convergence': self.check_convergence(),
            'edge_of_chaos': self.get_edge_of_chaos_metric(),
        }

        # Add continual learning metrics
        if self.task_performance:
            current_task = max(self.task_performance.keys())
            summary['average_performance'] = self.compute_average_performance(current_task)
            summary['average_forgetting'] = self.compute_average_forgetting(current_task)

        return summary

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.task_performance.clear()
        self.forgetting_matrix.clear()
        self.loss_history.clear()
        self.timestamps.clear()

    def export_to_wandb(self, wandb_run):
        """Export metrics to Weights & Biases"""
        try:
            import wandb

            for name, history in self.metrics.items():
                for entry in history:
                    wandb_run.log({
                        name: entry['value'],
                        'step': entry['step']
                    })
        except ImportError:
            print("Warning: wandb not installed")

    def export_to_mlflow(self, mlflow_run):
        """Export metrics to MLflow"""
        try:
            import mlflow

            for name, history in self.metrics.items():
                for entry in history:
                    mlflow.log_metric(
                        name,
                        entry['value'],
                        step=entry['step']
                    )
        except ImportError:
            print("Warning: mlflow not installed")

