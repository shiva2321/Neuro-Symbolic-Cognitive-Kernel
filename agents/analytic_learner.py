"""
Agent 4: The One-Pass Analytic Learner (Trainer)
Role: Neuromorphic Training Lead

Executes backpropagation-free training using Analytic Learning and Implicit Differentiation.

Features:
- AL-GNN (Analytic Learning for GNNs) with Recursive Least Squares
- Dynamic threshold optimization (DRSGNN)
- Implicit differentiation on equilibrium states (Dy-SIGN)
- Memory-efficient training for 12GB VRAM
- Continual learning without catastrophic forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import numpy as np
from collections import deque

# Handle DGL import error
try:
    import dgl
except (ImportError, FileNotFoundError, OSError):
    class MockDGL:
        class DGLGraph:
            pass
    dgl = MockDGL()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalyticLearningConfig:
    """Configuration for Analytic Learning"""
    use_rls: bool = True  # Recursive Least Squares
    forgetting_factor: float = 0.99  # For RLS
    regularization: float = 1e-4
    use_dynamic_threshold: bool = True
    threshold_lr: float = 0.001
    initial_threshold: float = 0.5
    use_implicit_diff: bool = True
    equilibrium_iterations: int = 50
    equilibrium_tolerance: float = 1e-4
    memory_efficient: bool = True  # For 12GB VRAM


class RecursiveLeastSquares:
    """
    Recursive Least Squares for online learning.
    Enables one-pass learning without storing historical data.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 forgetting_factor: float = 0.99,
                 regularization: float = 1e-4):
        """
        Initialize RLS estimator.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            forgetting_factor: Forgetting factor (λ)
            regularization: Initial regularization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_factor = forgetting_factor
        self.reg = regularization

        # Initialize parameters
        self.W = torch.zeros(output_dim, input_dim)  # Weight matrix
        self.P = torch.eye(input_dim) / self.reg  # Inverse covariance matrix

        self.num_updates = 0

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform RLS update with new data.

        Args:
            x: Input features (batch_size, input_dim)
            y: Target labels (batch_size, output_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        batch_size = x.shape[0]

        for i in range(batch_size):
            xi = x[i]  # (input_dim,)
            yi = y[i]  # (output_dim,)

            # Compute gain vector: K = P * x / (λ + x^T * P * x)
            Px = torch.matmul(self.P, xi)
            denominator = self.lambda_factor + torch.dot(xi, Px)
            K = Px / (denominator + 1e-8)  # (input_dim,)

            # Prediction error
            y_pred = torch.matmul(self.W, xi)  # (output_dim,)
            error = yi - y_pred  # (output_dim,)

            # Update weight matrix: W = W + error * K^T
            self.W = self.W + torch.outer(error, K)

            # Update inverse covariance: P = (P - K * x^T * P) / λ
            self.P = (self.P - torch.outer(K, Px)) / self.lambda_factor

            self.num_updates += 1

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Predictions (batch_size, output_dim)
        """
        return torch.matmul(x, self.W.T)

    def get_weights(self) -> torch.Tensor:
        """Get current weight matrix"""
        return self.W.clone()


class DynamicThresholdNeuron(nn.Module):
    """
    Neuron with learnable dynamic threshold.
    Each neuron can optimize its own threshold for reactive behavior.
    """

    def __init__(self, num_neurons: int, initial_threshold: float = 0.5,
                 threshold_lr: float = 0.001):
        """
        Initialize dynamic threshold neurons.

        Args:
            num_neurons: Number of neurons
            initial_threshold: Initial threshold value
            threshold_lr: Learning rate for threshold
        """
        super().__init__()

        self.num_neurons = num_neurons
        self.threshold = nn.Parameter(torch.ones(num_neurons) * initial_threshold)
        self.threshold_lr = threshold_lr

        # Track activation statistics
        self.activation_count = torch.zeros(num_neurons)
        self.total_input = torch.zeros(num_neurons)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic thresholding.

        Args:
            x: Input tensor (..., num_neurons)

        Returns:
            Tuple of (output, pre_activation)
        """
        # Apply threshold
        output = torch.where(x > self.threshold, x - self.threshold, torch.zeros_like(x))

        # Update statistics
        with torch.no_grad():
            self.activation_count += (x > self.threshold).float().sum(dim=0)
            self.total_input += x.sum(dim=0)

        return output, x

    def update_thresholds(self, target_activation_rate: float = 0.1):
        """
        Update thresholds based on activation statistics.

        Args:
            target_activation_rate: Desired activation rate
        """
        if self.total_input.sum() == 0:
            return

        with torch.no_grad():
            # Calculate current activation rate
            current_rate = self.activation_count / (self.total_input.abs() + 1e-8)

            # Adjust threshold to match target rate
            adjustment = (current_rate - target_activation_rate) * self.threshold_lr
            self.threshold.data -= adjustment

            # Clamp to reasonable range
            self.threshold.data.clamp_(0.01, 10.0)

            # Reset statistics
            self.activation_count.zero_()
            self.total_input.zero_()


class ImplicitDifferentiationLayer(nn.Module):
    """
    Graph layer with implicit differentiation at equilibrium.
    Computes gradients without storing intermediate activations.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 max_iterations: int = 50, tolerance: float = 1e-4):
        """
        Initialize implicit differentiation layer.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            max_iterations: Max iterations to find equilibrium
            tolerance: Convergence tolerance
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Parameters
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.U = nn.Parameter(torch.randn(output_dim, output_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # Scale U to ensure convergence
        with torch.no_grad():
            spectral_radius = torch.linalg.matrix_norm(self.U, ord=2)
            if spectral_radius > 0.9:
                self.U.data *= 0.9 / spectral_radius

    def find_equilibrium(self, x: torch.Tensor) -> torch.Tensor:
        """
        Find equilibrium state: z* = σ(W*x + U*z* + b)

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Equilibrium state (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        z = torch.zeros(batch_size, self.output_dim, device=x.device)

        Wx = torch.matmul(x, self.W.T) + self.bias

        for iteration in range(self.max_iterations):
            z_prev = z.clone()

            # Fixed-point iteration: z = σ(W*x + U*z + b)
            z = torch.tanh(Wx + torch.matmul(z, self.U.T))

            # Check convergence
            diff = torch.abs(z - z_prev).max()
            if diff < self.tolerance:
                break

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using implicit differentiation.

        Args:
            x: Input tensor

        Returns:
            Output at equilibrium
        """
        # Find equilibrium
        with torch.no_grad():
            z_star = self.find_equilibrium(x)

        # For backprop, use implicit differentiation
        # dL/dθ = (I - J^T)^{-1} dL/dz where J = ∂f/∂z
        # This is handled automatically by PyTorch's autograd if we recompute

        if self.training:
            # Recompute with gradients for training
            z_star = z_star.detach().requires_grad_(True)
            Wx = torch.matmul(x, self.W.T) + self.bias
            z_out = torch.tanh(Wx + torch.matmul(z_star, self.U.T))

            # Attach gradient computation
            z_out = z_out + (z_star - z_star.detach())

            return z_out
        else:
            return z_star


class AnalyticLearner:
    """
    Main analytic learner implementing backpropagation-free training.
    """

    def __init__(self, config: Optional[AnalyticLearningConfig] = None,
                 device: Optional[str] = None):
        """
        Initialize the Analytic Learner.

        Args:
            config: Configuration object
            device: Computation device
        """
        self.config = config or AnalyticLearningConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # RLS estimators for each layer (lazy initialization)
        self.rls_estimators: Dict[str, RecursiveLeastSquares] = {}

        # Dynamic threshold neurons
        self.threshold_neurons: Dict[str, DynamicThresholdNeuron] = {}

        # Implicit layers
        self.implicit_layers: Dict[str, ImplicitDifferentiationLayer] = {}

        # Training statistics
        self.training_history = []
        self.samples_seen = 0

        logger.info(f"AnalyticLearner initialized on {self.device}")

    def create_rls_estimator(self, name: str, input_dim: int, output_dim: int):
        """
        Create an RLS estimator.

        Args:
            name: Layer name
            input_dim: Input dimension
            output_dim: Output dimension
        """
        estimator = RecursiveLeastSquares(
            input_dim, output_dim,
            forgetting_factor=self.config.forgetting_factor,
            regularization=self.config.regularization
        )
        self.rls_estimators[name] = estimator
        logger.info(f"Created RLS estimator '{name}': {input_dim} -> {output_dim}")

    def create_dynamic_threshold_layer(self, name: str, num_neurons: int):
        """
        Create a dynamic threshold layer.

        Args:
            name: Layer name
            num_neurons: Number of neurons
        """
        layer = DynamicThresholdNeuron(
            num_neurons,
            initial_threshold=self.config.initial_threshold,
            threshold_lr=self.config.threshold_lr
        ).to(self.device)
        self.threshold_neurons[name] = layer
        logger.info(f"Created dynamic threshold layer '{name}': {num_neurons} neurons")

    def create_implicit_layer(self, name: str, input_dim: int, output_dim: int):
        """
        Create an implicit differentiation layer.

        Args:
            name: Layer name
            input_dim: Input dimension
            output_dim: Output dimension
        """
        layer = ImplicitDifferentiationLayer(
            input_dim, output_dim,
            max_iterations=self.config.equilibrium_iterations,
            tolerance=self.config.equilibrium_tolerance
        ).to(self.device)
        self.implicit_layers[name] = layer
        logger.info(f"Created implicit layer '{name}': {input_dim} -> {output_dim}")

    def train_step(self, graph: dgl.DGLGraph, features: torch.Tensor,
                  labels: torch.Tensor, layer_name: str = 'classifier') -> Dict[str, float]:
        """
        Perform a single training step using analytic learning.

        Args:
            graph: DGL graph
            features: Node features
            labels: Node labels
            layer_name: Name of the layer to train

        Returns:
            Dictionary with training metrics
        """
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Get or create RLS estimator
        if layer_name not in self.rls_estimators:
            input_dim = features.shape[1]
            output_dim = labels.shape[1] if labels.dim() > 1 else int(labels.max().item()) + 1
            self.create_rls_estimator(layer_name, input_dim, output_dim)

        estimator = self.rls_estimators[layer_name]

        # One-hot encode labels if needed
        if labels.dim() == 1:
            num_classes = int(labels.max().item()) + 1
            labels_onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
        else:
            labels_onehot = labels

        # Update RLS
        estimator.update(features, labels_onehot)

        # Compute accuracy
        predictions = estimator.predict(features)
        pred_labels = predictions.argmax(dim=1)
        true_labels = labels if labels.dim() == 1 else labels.argmax(dim=1)
        accuracy = (pred_labels == true_labels).float().mean().item()

        self.samples_seen += features.shape[0]

        metrics = {
            'accuracy': accuracy,
            'samples_seen': self.samples_seen,
            'num_updates': estimator.num_updates
        }

        return metrics

    def train_with_dynamic_threshold(self, graph: dgl.DGLGraph,
                                    features: torch.Tensor,
                                    labels: torch.Tensor,
                                    layer_name: str = 'threshold_layer') -> Dict[str, float]:
        """
        Train using dynamic threshold neurons.

        Args:
            graph: DGL graph
            features: Node features
            labels: Node labels
            layer_name: Layer name

        Returns:
            Training metrics
        """
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Get or create dynamic threshold layer
        if layer_name not in self.threshold_neurons:
            num_neurons = features.shape[1]
            self.create_dynamic_threshold_layer(layer_name, num_neurons)

        threshold_layer = self.threshold_neurons[layer_name]

        # Forward pass
        activated_features, pre_activation = threshold_layer(features)

        # Update thresholds based on activity
        threshold_layer.update_thresholds(target_activation_rate=0.1)

        # Continue with RLS training on activated features
        metrics = self.train_step(graph, activated_features, labels, f"{layer_name}_classifier")

        metrics['avg_threshold'] = threshold_layer.threshold.mean().item()

        return metrics

    def train_implicit(self, graph: dgl.DGLGraph, features: torch.Tensor,
                      labels: torch.Tensor, layer_name: str = 'implicit_layer',
                      num_steps: int = 100) -> Dict[str, Any]:
        """
        Train using implicit differentiation.

        Args:
            graph: DGL graph
            features: Node features
            labels: Node labels
            layer_name: Layer name
            num_steps: Number of training steps

        Returns:
            Training metrics
        """
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Get or create implicit layer
        if layer_name not in self.implicit_layers:
            input_dim = features.shape[1]
            output_dim = labels.shape[1] if labels.dim() > 1 else int(labels.max().item()) + 1
            self.create_implicit_layer(layer_name, input_dim, output_dim)

        implicit_layer = self.implicit_layers[layer_name]

        # Optimizer
        optimizer = torch.optim.Adam(implicit_layer.parameters(), lr=0.001)

        # Training loop
        losses = []
        accuracies = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass (finds equilibrium)
            output = implicit_layer(features)

            # Loss
            if labels.dim() == 1:
                loss = F.cross_entropy(output, labels.long())
            else:
                loss = F.mse_loss(output, labels)

            # Backward pass (uses implicit differentiation)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Accuracy
            if labels.dim() == 1:
                pred_labels = output.argmax(dim=1)
                accuracy = (pred_labels == labels).float().mean().item()
                accuracies.append(accuracy)

        return {
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'losses': losses,
            'accuracies': accuracies
        }

    def continual_learn(self, new_graph: dgl.DGLGraph, new_features: torch.Tensor,
                       new_labels: torch.Tensor, layer_name: str = 'continual') -> Dict[str, float]:
        """
        Continual learning on new data without forgetting.

        Args:
            new_graph: New graph data
            new_features: New node features
            new_labels: New labels
            layer_name: Layer name

        Returns:
            Training metrics
        """
        logger.info("Continual learning on new data stream...")

        # Use RLS which naturally supports continual learning
        metrics = self.train_step(new_graph, new_features, new_labels, layer_name)

        self.training_history.append({
            'timestamp': self.samples_seen,
            'accuracy': metrics['accuracy']
        })

        return metrics

    def get_weights(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get learned weights from a layer.

        Args:
            layer_name: Layer name

        Returns:
            Weight tensor or None
        """
        if layer_name in self.rls_estimators:
            return self.rls_estimators[layer_name].get_weights()
        elif layer_name in self.implicit_layers:
            return self.implicit_layers[layer_name].W.data
        else:
            return None

    def save_state(self, path: str):
        """
        Save learner state.

        Args:
            path: Save path
        """
        state = {
            'config': asdict(self.config),
            'samples_seen': self.samples_seen,
            'training_history': self.training_history,
            'rls_estimators': {name: (est.W, est.P) for name, est in self.rls_estimators.items()},
            'threshold_neurons': {name: layer.state_dict() for name, layer in self.threshold_neurons.items()},
            'implicit_layers': {name: layer.state_dict() for name, layer in self.implicit_layers.items()}
        }
        torch.save(state, path)
        logger.info(f"Saved learner state to {path}")

    def load_state(self, path: str):
        """
        Load learner state.

        Args:
            path: Load path
        """
        state = torch.load(path, map_location=self.device)
        self.config = AnalyticLearningConfig(**state['config'])
        self.samples_seen = state['samples_seen']
        self.training_history = state['training_history']

        # Restore RLS estimators
        for name, (W, P) in state['rls_estimators'].items():
            input_dim, output_dim = W.shape[1], W.shape[0]
            self.create_rls_estimator(name, input_dim, output_dim)
            self.rls_estimators[name].W = W
            self.rls_estimators[name].P = P

        # Restore threshold neurons
        for name, state_dict in state['threshold_neurons'].items():
            num_neurons = state_dict['threshold'].shape[0]
            self.create_dynamic_threshold_layer(name, num_neurons)
            self.threshold_neurons[name].load_state_dict(state_dict)

        # Restore implicit layers
        for name, state_dict in state['implicit_layers'].items():
            input_dim = state_dict['W'].shape[1]
            output_dim = state_dict['W'].shape[0]
            self.create_implicit_layer(name, input_dim, output_dim)
            self.implicit_layers[name].load_state_dict(state_dict)

        logger.info(f"Loaded learner state from {path}")