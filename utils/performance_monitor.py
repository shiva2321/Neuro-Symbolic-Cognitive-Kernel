"""
Performance Monitor
Tools for monitoring and optimizing system performance.
Implements Phase 3: Scalability and Performance
"""

import time
import psutil
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"PerformanceMetrics(op='{self.operation}', "
                f"duration={self.duration:.3f}s, "
                f"memory_delta={self.memory_delta:.2f}MB, "
                f"cpu={self.cpu_percent:.1f}%)")


class PerformanceMonitor:
    """
    Monitors and tracks system performance metrics.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.current_operation: Optional[Dict[str, Any]] = None

        # Get process
        self.process = psutil.Process(os.getpid())

    def start_operation(self, operation_name: str, metadata: Optional[Dict] = None):
        """
        Start monitoring an operation.

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata about the operation
        """
        if not self.enabled:
            return

        self.current_operation = {
            'name': operation_name,
            'start_time': time.time(),
            'memory_before': self._get_memory_usage(),
            'cpu_before': self.process.cpu_percent(),
            'metadata': metadata or {}
        }

    def end_operation(self) -> Optional[PerformanceMetrics]:
        """
        End monitoring the current operation.

        Returns:
            PerformanceMetrics for the operation
        """
        if not self.enabled or not self.current_operation:
            return None

        end_time = time.time()
        memory_after = self._get_memory_usage()
        cpu_after = self.process.cpu_percent()

        metrics = PerformanceMetrics(
            operation=self.current_operation['name'],
            start_time=self.current_operation['start_time'],
            end_time=end_time,
            duration=end_time - self.current_operation['start_time'],
            memory_before=self.current_operation['memory_before'],
            memory_after=memory_after,
            memory_delta=memory_after - self.current_operation['memory_before'],
            cpu_percent=(cpu_after + self.current_operation['cpu_before']) / 2,
            metadata=self.current_operation['metadata']
        )

        self.metrics.append(metrics)
        self.operation_stats[metrics.operation].append(metrics.duration)

        self.current_operation = None
        return metrics

    def monitor_operation(self, operation_name: str, metadata: Optional[Dict] = None):
        """
        Context manager for monitoring operations.

        Usage:
            with monitor.monitor_operation("training"):
                # code to monitor
        """
        return OperationContext(self, operation_name, metadata)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all performance metrics.

        Returns:
            Dictionary with performance summary
        """
        if not self.metrics:
            return {'message': 'No metrics collected'}

        total_time = sum(m.duration for m in self.metrics)
        total_memory = sum(m.memory_delta for m in self.metrics)
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)

        # Operation-specific stats
        operation_summary = {}
        for op_name, durations in self.operation_stats.items():
            operation_summary[op_name] = {
                'count': len(durations),
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations)
            }

        return {
            'total_operations': len(self.metrics),
            'total_time': total_time,
            'total_memory_delta': total_memory,
            'average_cpu': avg_cpu,
            'current_memory': self._get_memory_usage(),
            'operations': operation_summary,
            'slowest_operations': self._get_slowest_operations(5),
            'memory_intensive_operations': self._get_memory_intensive_operations(5)
        }

    def _get_slowest_operations(self, n: int = 5) -> List[Dict]:
        """Get the n slowest operations"""
        sorted_metrics = sorted(self.metrics, key=lambda m: m.duration, reverse=True)
        return [
            {
                'operation': m.operation,
                'duration': m.duration,
                'metadata': m.metadata
            }
            for m in sorted_metrics[:n]
        ]

    def _get_memory_intensive_operations(self, n: int = 5) -> List[Dict]:
        """Get the n most memory-intensive operations"""
        sorted_metrics = sorted(self.metrics, key=lambda m: m.memory_delta, reverse=True)
        return [
            {
                'operation': m.operation,
                'memory_delta': m.memory_delta,
                'metadata': m.metadata
            }
            for m in sorted_metrics[:n]
        ]

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify performance bottlenecks.

        Returns:
            Dictionary with bottleneck analysis
        """
        summary = self.get_summary()
        bottlenecks = {
            'time_bottlenecks': [],
            'memory_bottlenecks': [],
            'recommendations': []
        }

        # Identify time bottlenecks (operations taking > 10% of total time)
        total_time = summary['total_time']
        for op_name, stats in summary['operations'].items():
            if stats['total_time'] > total_time * 0.1:
                bottlenecks['time_bottlenecks'].append({
                    'operation': op_name,
                    'percentage': (stats['total_time'] / total_time) * 100,
                    'avg_time': stats['avg_time']
                })

        # Identify memory bottlenecks (operations using > 100MB)
        for metric in self.metrics:
            if metric.memory_delta > 100:
                bottlenecks['memory_bottlenecks'].append({
                    'operation': metric.operation,
                    'memory_delta': metric.memory_delta,
                    'metadata': metric.metadata
                })

        # Generate recommendations
        if bottlenecks['time_bottlenecks']:
            bottlenecks['recommendations'].append(
                "Consider optimizing the following time-intensive operations: " +
                ", ".join([b['operation'] for b in bottlenecks['time_bottlenecks']])
            )

        if bottlenecks['memory_bottlenecks']:
            bottlenecks['recommendations'].append(
                "Consider reducing memory usage in operations or implementing batch processing"
            )

        if summary['average_cpu'] > 80:
            bottlenecks['recommendations'].append(
                "High CPU usage detected. Consider parallelization or optimization."
            )

        return bottlenecks

    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file.

        Args:
            filepath: Output file path
        """
        data = {
            'metrics': [
                {
                    'operation': m.operation,
                    'duration': m.duration,
                    'memory_delta': m.memory_delta,
                    'cpu_percent': m.cpu_percent,
                    'metadata': m.metadata
                }
                for m in self.metrics
            ],
            'summary': self.get_summary(),
            'bottlenecks': self.identify_bottlenecks()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Performance metrics exported to: {filepath}")

    def print_report(self):
        """Print a formatted performance report"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"\nTotal Operations: {summary['total_operations']}")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print(f"Total Memory Delta: {summary['total_memory_delta']:.2f}MB")
        print(f"Average CPU: {summary['average_cpu']:.1f}%")
        print(f"Current Memory: {summary['current_memory']:.2f}MB")

        print("\nOperation Summary:")
        for op_name, stats in summary['operations'].items():
            print(f"  {op_name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total Time: {stats['total_time']:.2f}s")
            print(f"    Avg Time: {stats['avg_time']:.3f}s")
            print(f"    Range: {stats['min_time']:.3f}s - {stats['max_time']:.3f}s")

        print("\nSlowest Operations:")
        for i, op in enumerate(summary['slowest_operations'], 1):
            print(f"  {i}. {op['operation']}: {op['duration']:.3f}s")

        print("\nMemory Intensive Operations:")
        for i, op in enumerate(summary['memory_intensive_operations'], 1):
            print(f"  {i}. {op['operation']}: {op['memory_delta']:.2f}MB")

        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks['recommendations']:
            print("\nRecommendations:")
            for rec in bottlenecks['recommendations']:
                print(f"  • {rec}")

        print("\n" + "="*60 + "\n")

    def clear(self):
        """Clear all collected metrics"""
        self.metrics.clear()
        self.operation_stats.clear()
        self.current_operation = None


class OperationContext:
    """Context manager for monitoring operations"""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str, metadata: Optional[Dict] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata

    def __enter__(self):
        self.monitor.start_operation(self.operation_name, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics = self.monitor.end_operation()
        if metrics and exc_type is None:
            print(f"  ✓ {self.operation_name}: {metrics.duration:.3f}s")
        return False

