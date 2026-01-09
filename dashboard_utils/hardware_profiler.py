"""
Hardware Profiler for NCGN Dashboard
Real-time monitoring of GPU, CPU, RAM, and energy consumption.
"""

import torch
import psutil
import time
from typing import Dict, Any, Optional
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardwareProfiler:
    """
    Profiles hardware usage for dashboard display.
    """

    def __init__(self, history_size: int = 300):
        """
        Initialize hardware profiler.

        Args:
            history_size: Number of historical samples to keep (300 = 5 mins at 1Hz)
        """
        self.history_size = history_size

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()

        # Initialize NVML for detailed GPU stats
        self.nvml_available = False
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            self.nvml = nvml
            self.nvml_available = True
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except:
            logger.warning("NVML not available - limited GPU monitoring")

        # History buffers
        self.gpu_memory_history = deque(maxlen=history_size)
        self.gpu_utilization_history = deque(maxlen=history_size)
        self.cpu_utilization_history = deque(maxlen=history_size)
        self.ram_usage_history = deque(maxlen=history_size)
        self.energy_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

        # Energy tracking
        self.last_energy_reading = 0
        self.total_energy_mj = 0  # milliJoules

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        stats = {
            'available': self.cuda_available,
            'memory_allocated': 0,
            'memory_reserved': 0,
            'memory_total': 0,
            'utilization': 0,
            'temperature': 0,
            'power_usage': 0,
            'name': 'N/A'
        }

        if not self.cuda_available:
            return stats

        # Basic PyTorch stats
        stats['memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9  # GB
        stats['memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9  # GB
        stats['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        stats['name'] = torch.cuda.get_device_name(0)

        # NVML stats if available
        if self.nvml_available:
            try:
                # GPU utilization
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                stats['utilization'] = util.gpu

                # Temperature
                temp = self.nvml.nvmlDeviceGetTemperature(
                    self.gpu_handle,
                    self.nvml.NVML_TEMPERATURE_GPU
                )
                stats['temperature'] = temp

                # Power usage
                power = self.nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Watts
                stats['power_usage'] = power

            except Exception as e:
                logger.debug(f"Error reading NVML stats: {e}")

        return stats

    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics"""
        return {
            'utilization': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get system memory statistics"""
        mem = psutil.virtual_memory()

        return {
            'total': mem.total / 1e9,  # GB
            'available': mem.available / 1e9,  # GB
            'used': mem.used / 1e9,  # GB
            'percent': mem.percent
        }

    def get_disk_stats(self) -> Dict[str, Any]:
        """Get disk I/O statistics"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total': disk.total / 1e9,  # GB
                'used': disk.used / 1e9,  # GB
                'free': disk.free / 1e9,  # GB
                'percent': disk.percent
            }
        except:
            return {'total': 0, 'used': 0, 'free': 0, 'percent': 0}

    def compute_energy(self, power_watts: float, duration_seconds: float) -> float:
        """
        Compute energy consumption.

        Args:
            power_watts: Power usage in watts
            duration_seconds: Time duration

        Returns:
            Energy in milliJoules
        """
        # Energy (J) = Power (W) * Time (s)
        energy_joules = power_watts * duration_seconds
        return energy_joules * 1000  # Convert to milliJoules

    def update(self):
        """Update all statistics and history"""
        timestamp = time.time()

        # Get current stats
        gpu_stats = self.get_gpu_stats()
        cpu_stats = self.get_cpu_stats()
        mem_stats = self.get_memory_stats()

        # Update histories
        self.gpu_memory_history.append(gpu_stats['memory_allocated'])
        self.gpu_utilization_history.append(gpu_stats['utilization'])
        self.cpu_utilization_history.append(cpu_stats['utilization'])
        self.ram_usage_history.append(mem_stats['percent'])
        self.timestamps.append(timestamp)

        # Energy tracking
        if gpu_stats['power_usage'] > 0:
            duration = timestamp - self.last_energy_reading if self.last_energy_reading > 0 else 1.0
            energy_mj = self.compute_energy(gpu_stats['power_usage'], duration)
            self.total_energy_mj += energy_mj
            self.energy_history.append(energy_mj)
            self.last_energy_reading = timestamp

    def get_stats(self) -> Dict[str, Any]:
        """Get complete hardware statistics"""
        # Update current stats
        self.update()

        gpu_stats = self.get_gpu_stats()
        cpu_stats = self.get_cpu_stats()
        mem_stats = self.get_memory_stats()
        disk_stats = self.get_disk_stats()

        return {
            'gpu': gpu_stats,
            'cpu': cpu_stats,
            'memory': mem_stats,
            'disk': disk_stats,
            'energy': {
                'total_mj': self.total_energy_mj,
                'recent_mj': list(self.energy_history)[-10:] if self.energy_history else [],
                'average_power_w': sum(self.energy_history) / max(len(self.energy_history), 1) if self.energy_history else 0
            },
            'timestamp': time.time()
        }

    def get_history(self, metric: str = 'all') -> Dict[str, Any]:
        """Get historical data"""
        if metric == 'all':
            return {
                'gpu_memory': list(self.gpu_memory_history),
                'gpu_utilization': list(self.gpu_utilization_history),
                'cpu_utilization': list(self.cpu_utilization_history),
                'ram_usage': list(self.ram_usage_history),
                'energy': list(self.energy_history),
                'timestamps': list(self.timestamps)
            }

        history_map = {
            'gpu_memory': self.gpu_memory_history,
            'gpu_utilization': self.gpu_utilization_history,
            'cpu_utilization': self.cpu_utilization_history,
            'ram_usage': self.ram_usage_history,
            'energy': self.energy_history
        }

        return {metric: list(history_map.get(metric, []))}

    def get_memory_wall_analysis(self) -> Dict[str, Any]:
        """
        Analyze memory usage patterns for 'memory wall' detection.

        Returns:
            Analysis of memory allocation patterns
        """
        if not self.gpu_memory_history:
            return {'status': 'no_data'}

        recent_memory = list(self.gpu_memory_history)[-50:]

        # Calculate statistics
        avg_memory = sum(recent_memory) / len(recent_memory)
        max_memory = max(recent_memory)
        gpu_stats = self.get_gpu_stats()
        total_memory = gpu_stats['memory_total']

        # Memory pressure
        memory_pressure = max_memory / total_memory if total_memory > 0 else 0

        # Volatility (how much memory usage fluctuates)
        if len(recent_memory) > 1:
            import numpy as np
            volatility = np.std(recent_memory) / (avg_memory + 1e-6)
        else:
            volatility = 0

        # Determine status
        if memory_pressure > 0.95:
            status = 'critical'
            recommendation = 'Reduce batch size or enable CPU offloading'
        elif memory_pressure > 0.85:
            status = 'warning'
            recommendation = 'Consider enabling gradient checkpointing'
        elif memory_pressure > 0.70:
            status = 'caution'
            recommendation = 'Monitor closely, approaching memory limit'
        else:
            status = 'healthy'
            recommendation = 'Memory usage is optimal'

        return {
            'status': status,
            'memory_pressure': memory_pressure,
            'avg_memory_gb': avg_memory,
            'max_memory_gb': max_memory,
            'total_memory_gb': total_memory,
            'volatility': volatility,
            'recommendation': recommendation
        }

    def get_gpu_name(self) -> str:
        """Get GPU name"""
        if self.cuda_available:
            return torch.cuda.get_device_name(0)
        return "No GPU"

    def reset_energy(self):
        """Reset energy counter"""
        self.total_energy_mj = 0
        self.last_energy_reading = 0

    def __del__(self):
        """Cleanup"""
        if self.nvml_available:
            try:
                self.nvml.nvmlShutdown()
            except:
                pass

