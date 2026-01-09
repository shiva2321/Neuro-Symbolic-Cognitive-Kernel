"""
Hebbian Trace Monitor for NCGN Dashboard
Tracks synaptic weight changes and 'engrams' over time.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HebbianTraceMonitor:
    """
    Monitors Hebbian learning traces and synaptic weight evolution.
    Provides insights into how the network forms persistent memories.
    """

    def __init__(self, trace_length: int = 100):
        """
        Initialize Hebbian trace monitor.

        Args:
            trace_length: Number of timesteps to track
        """
        self.trace_length = trace_length

        # Weight histories for tracked layers
        self.weight_traces = {}  # {layer_name: deque of weight snapshots}
        self.weight_changes = {}  # {layer_name: deque of delta weights}

        # Attention histories
        self.attention_traces = deque(maxlen=trace_length)

        # Synaptic strength statistics
        self.synapse_stats = defaultdict(list)

        # Hooks for model monitoring
        self.hooks = []
        self.model = None

        # Latest captures
        self.latest_attention = None
        self.latest_weight_changes = {}

    def attach(self, model: torch.nn.Module):
        """
        Attach monitor to a model.

        Args:
            model: PyTorch model to monitor
        """
        self.model = model
        self.clear_hooks()

        # Register hooks for weight tracking
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Store initial weights
                self.weight_traces[name] = deque(maxlen=self.trace_length)
                self.weight_changes[name] = deque(maxlen=self.trace_length)

                # Add hook to track weight updates
                hook = self._create_weight_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)

        logger.info(f"Attached Hebbian monitor to {len(self.hooks)} layers")

    def _create_weight_hook(self, layer_name: str):
        """Create a forward hook for a layer"""
        def hook(module, input, output):
            if hasattr(module, 'weight') and module.weight is not None:
                # Capture weight snapshot
                weight_snapshot = module.weight.data.clone().cpu()

                # Store snapshot
                self.weight_traces[layer_name].append(weight_snapshot)

                # Compute weight change if we have history
                if len(self.weight_traces[layer_name]) >= 2:
                    prev_weights = self.weight_traces[layer_name][-2]
                    delta_weights = weight_snapshot - prev_weights
                    self.weight_changes[layer_name].append(delta_weights)
                    self.latest_weight_changes[layer_name] = delta_weights

        return hook

    def capture_attention(self, attention_weights: torch.Tensor):
        """
        Capture attention weights.

        Args:
            attention_weights: Attention matrix (batch, heads, seq, seq)
        """
        # Average over batch and heads
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=(0, 1))  # (seq, seq)
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=0)  # (seq, seq)
        else:
            attn = attention_weights

        # Store
        self.attention_traces.append(attn.detach().cpu().numpy())
        self.latest_attention = attn.detach().cpu().numpy()

    def get_latest_attention(self) -> Optional[np.ndarray]:
        """Get latest attention weights"""
        return self.latest_attention

    def compute_synaptic_strength(self, layer_name: str) -> Dict[str, float]:
        """
        Compute synaptic strength statistics for a layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Statistics dict with mean, std, max, min
        """
        if layer_name not in self.weight_traces or not self.weight_traces[layer_name]:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}

        latest_weights = self.weight_traces[layer_name][-1].numpy()

        return {
            'mean': float(np.mean(np.abs(latest_weights))),
            'std': float(np.std(latest_weights)),
            'max': float(np.max(np.abs(latest_weights))),
            'min': float(np.min(np.abs(latest_weights)))
        }

    def compute_weight_stability(self, layer_name: str, window: int = 10) -> float:
        """
        Compute weight stability over recent timesteps.

        Stability = 1 / (1 + mean_absolute_change)

        Args:
            layer_name: Name of the layer
            window: Number of recent steps to analyze

        Returns:
            Stability score in [0, 1]
        """
        if layer_name not in self.weight_changes or not self.weight_changes[layer_name]:
            return 1.0

        recent_changes = list(self.weight_changes[layer_name])[-window:]

        if not recent_changes:
            return 1.0

        # Compute mean absolute change
        mean_change = np.mean([np.abs(delta.numpy()).mean() for delta in recent_changes])

        # Map to stability score
        stability = 1.0 / (1.0 + mean_change * 100)

        return float(stability)

    def identify_engrams(self, layer_name: str, threshold: float = 0.5) -> List[tuple]:
        """
        Identify persistent synaptic patterns (engrams).

        An engram is a synapse that has maintained high strength over time.

        Args:
            layer_name: Name of the layer
            threshold: Minimum strength to consider

        Returns:
            List of (row, col, strength) tuples
        """
        if layer_name not in self.weight_traces or len(self.weight_traces[layer_name]) < 5:
            return []

        # Get recent weight snapshots
        recent_weights = [w.numpy() for w in list(self.weight_traces[layer_name])[-10:]]

        # Average weights over time
        avg_weights = np.mean(recent_weights, axis=0)

        # Find strong, persistent connections
        engrams = []
        rows, cols = np.where(np.abs(avg_weights) > threshold)

        for i, j in zip(rows, cols):
            strength = float(avg_weights[i, j])
            engrams.append((int(i), int(j), strength))

        # Sort by strength
        engrams.sort(key=lambda x: abs(x[2]), reverse=True)

        return engrams[:100]  # Return top 100

    def compute_plasticity_index(self, layer_name: str) -> float:
        """
        Compute plasticity index: how much weights are changing.

        High plasticity = learning actively
        Low plasticity = weights stabilized

        Args:
            layer_name: Name of the layer

        Returns:
            Plasticity index in [0, 1]
        """
        if layer_name not in self.weight_changes or not self.weight_changes[layer_name]:
            return 0.0

        recent_changes = list(self.weight_changes[layer_name])[-20:]

        if not recent_changes:
            return 0.0

        # Compute variance of weight changes
        change_magnitudes = [np.abs(delta.numpy()).mean() for delta in recent_changes]
        plasticity = np.std(change_magnitudes)

        # Normalize to [0, 1]
        plasticity_normalized = min(1.0, plasticity * 1000)

        return float(plasticity_normalized)

    def get_trace_summary(self) -> Dict[str, Any]:
        """
        Get summary of all traces.

        Returns:
            Summary dictionary
        """
        summary = {
            'layers': {},
            'attention': {
                'available': len(self.attention_traces) > 0,
                'latest_shape': self.latest_attention.shape if self.latest_attention is not None else None
            }
        }

        for layer_name in self.weight_traces.keys():
            summary['layers'][layer_name] = {
                'synaptic_strength': self.compute_synaptic_strength(layer_name),
                'stability': self.compute_weight_stability(layer_name),
                'plasticity': self.compute_plasticity_index(layer_name),
                'num_snapshots': len(self.weight_traces[layer_name])
            }

        return summary

    def get_attention_heatmap(self, top_k: int = 50) -> Optional[Dict[str, Any]]:
        """
        Get attention heatmap data for visualization.

        Args:
            top_k: Number of top nodes to include

        Returns:
            Heatmap data
        """
        if not self.attention_traces:
            return None

        # Get latest attention
        attn = self.attention_traces[-1]

        # Limit to top_k x top_k for visualization
        if attn.shape[0] > top_k:
            attn = attn[:top_k, :top_k]

        return {
            'matrix': attn.tolist(),
            'shape': attn.shape,
            'max_value': float(np.max(attn)),
            'min_value': float(np.min(attn))
        }

    def get_engram_visualization(self, layer_name: str) -> Dict[str, Any]:
        """
        Get engram data for visualization.

        Args:
            layer_name: Name of the layer

        Returns:
            Visualization data
        """
        engrams = self.identify_engrams(layer_name, threshold=0.3)

        if not engrams:
            return {'nodes': [], 'edges': []}

        # Convert to graph format
        nodes = set()
        edges = []

        for source, target, strength in engrams:
            nodes.add(source)
            nodes.add(target)
            edges.append({
                'source': source,
                'target': target,
                'strength': strength,
                'type': 'excitatory' if strength > 0 else 'inhibitory'
            })

        return {
            'nodes': [{'id': n} for n in sorted(nodes)],
            'edges': edges,
            'layer': layer_name
        }

    def export_traces(self, layer_name: str, format: str = 'numpy') -> Any:
        """
        Export weight traces for a layer.

        Args:
            layer_name: Name of the layer
            format: Export format ('numpy', 'list', 'tensor')

        Returns:
            Exported data
        """
        if layer_name not in self.weight_traces:
            return None

        traces = list(self.weight_traces[layer_name])

        if format == 'numpy':
            return [w.numpy() for w in traces]
        elif format == 'list':
            return [w.tolist() for w in traces]
        elif format == 'tensor':
            return traces

        return None

    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def reset(self):
        """Reset all traces"""
        self.weight_traces.clear()
        self.weight_changes.clear()
        self.attention_traces.clear()
        self.synapse_stats.clear()
        self.latest_attention = None
        self.latest_weight_changes.clear()

    def __del__(self):
        """Cleanup"""
        self.clear_hooks()

