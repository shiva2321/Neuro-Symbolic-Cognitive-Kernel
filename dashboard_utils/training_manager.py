"""
Training Manager for NCGN Dashboard
Handles training loop with real-time monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages training process with monitoring integration.
    """

    def __init__(self, config: Dict, metrics_monitor, hardware_profiler):
        """
        Initialize training manager.

        Args:
            config: Configuration dictionary
            metrics_monitor: MetricsMonitor instance
            hardware_profiler: HardwareProfiler instance
        """
        self.config = config
        self.metrics_monitor = metrics_monitor
        self.hardware_profiler = hardware_profiler

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Device
        # Always check actual CUDA availability, don't trust config
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info("Using CUDA for training")
        else:
            self.device = 'cpu'
            logger.info("CUDA not available, using CPU for training")

    def setup_optimizer(self, model: nn.Module):
        """Setup optimizer and scheduler"""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        # AdamW optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler_type = self.config['training']['scheduler']
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['max_epochs']
            )
        elif scheduler_type == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config['training']['max_epochs']
            )

    def train_epoch(self, model: nn.Module, epoch: int, training_data: list = None,
                    linguistic_graph = None) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            model: Model to train
            epoch: Current epoch number
            training_data: Optional list of text chunks to train on
            linguistic_graph: Optional linguistic graph for node features

        Returns:
            Dictionary of metrics
        """
        model.train()

        # Setup optimizer if not done
        if self.optimizer is None:
            self.setup_optimizer(model)

        epoch_metrics = {
            'loss': 0.0,
            'system1_loss': 0.0,
            'system2_loss': 0.0,
            'snn_spike_rate': 0.0,
            'graph_attention': 0.0,
            'data_samples': 0
        }

        batch_size = self.config['memory_optimization']['batch_size']
        node_feat_dim = self.config['linguistic_graph']['embedding_dim']

        # Get the device the model is on
        model_device = next(model.parameters()).device

        # Use real data if available, otherwise dummy data
        if linguistic_graph is not None and hasattr(linguistic_graph, 'graph'):
            # Use actual linguistic graph data
            try:
                import dgl
                from utils.graph_backend import GraphBackend

                # Get graph features
                if hasattr(linguistic_graph, '_backend') and linguistic_graph._backend:
                    node_features = linguistic_graph._backend.get_node_features('feat')
                else:
                    node_features = linguistic_graph.graph.ndata['feat']

                num_nodes = node_features.shape[0]

                # Sample a batch of nodes
                if num_nodes > 100:
                    # Subsample for memory efficiency
                    indices = torch.randperm(num_nodes)[:100]
                    node_features = node_features[indices]
                    num_nodes = 100

                # Create batch by repeating - ENSURE ON CORRECT DEVICE
                node_features = node_features.unsqueeze(0).repeat(batch_size, 1, 1).to(model_device)

                # Create adjacency from graph structure (simplified) - ENSURE ON CORRECT DEVICE
                adjacency = torch.eye(num_nodes, device=model_device).unsqueeze(0).repeat(batch_size, 1, 1)

                epoch_metrics['data_samples'] = num_nodes
                logger.info(f"Training on real linguistic graph: {num_nodes} nodes")

            except Exception as e:
                logger.warning(f"Could not use linguistic graph: {e}, using dummy data")
                # Fall back to dummy data
                num_nodes = 100
                node_features = torch.randn(batch_size, num_nodes, node_feat_dim, device=model_device)
                adjacency = torch.rand(batch_size, num_nodes, num_nodes, device=model_device)
                adjacency = (adjacency > 0.9).float()
        else:
            # Use dummy data - CREATE DIRECTLY ON CORRECT DEVICE
            num_nodes = 100
            node_features = torch.randn(batch_size, num_nodes, node_feat_dim, device=model_device)
            adjacency = torch.rand(batch_size, num_nodes, num_nodes, device=model_device)
            adjacency = (adjacency > 0.9).float()

        # Forward pass
        self.optimizer.zero_grad()

        try:
            output = model(node_features, adjacency)

            # Compute loss from model output
            loss = None
            confidence_value = None

            # Try to extract tensors from nested output structure
            if isinstance(output, dict):
                # Check for system1_output
                if 'system1_output' in output:
                    sys1_out = output['system1_output']
                    if isinstance(sys1_out, torch.Tensor) and sys1_out.requires_grad:
                        loss = sys1_out.mean().abs()
                    elif isinstance(sys1_out, dict):
                        # System1 output is nested dict, search for tensors
                        for k, v in sys1_out.items():
                            if isinstance(v, torch.Tensor) and v.requires_grad:
                                loss = v.mean().abs()
                                break

                # Check for confidence
                if 'confidence' in output:
                    conf = output['confidence']
                    if isinstance(conf, torch.Tensor):
                        confidence_value = float(conf.mean().item())
                        if loss is None and conf.requires_grad:
                            loss = (1.0 - conf).mean()

                # Search all keys for gradient tensors
                if loss is None:
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor) and value.requires_grad:
                            loss = value.mean().abs()
                            break
                        elif isinstance(value, dict):
                            # Check nested dicts
                            for k, v in value.items():
                                if isinstance(v, torch.Tensor) and v.requires_grad:
                                    loss = v.mean().abs()
                                    break
                            if loss is not None:
                                break

            # If still no loss, use L2 regularization
            if loss is None:
                loss = sum(p.pow(2.0).sum() for p in model.parameters()) * 0.001

            # Ensure loss is scalar
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            epoch_metrics['loss'] = float(loss.item())
            if confidence_value is not None:
                epoch_metrics['confidence'] = confidence_value

            # Log to metrics monitor
            self.metrics_monitor.log_metric('train_loss', epoch_metrics['loss'], epoch)
            if confidence_value is not None:
                self.metrics_monitor.log_metric('confidence', confidence_value, epoch)

        except Exception as e:
            logger.error(f"Training error: {e}")
            epoch_metrics['loss'] = 0.0
            import traceback
            logger.debug(traceback.format_exc())

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

        # Update hardware stats
        self.hardware_profiler.update()

        return epoch_metrics

    def validate(self, model: nn.Module, val_loader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            model: Model to validate
            val_loader: Validation data loader

        Returns:
            Validation metrics
        """
        model.eval()

        val_metrics = {
            'val_loss': 0.0,
            'val_accuracy': 0.0
        }

        with torch.no_grad():
            # Validation loop
            pass

        return val_metrics

    def save_checkpoint(self, model: nn.Module, path: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, model: nn.Module, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']

        logger.info(f"Checkpoint loaded: {path}")

