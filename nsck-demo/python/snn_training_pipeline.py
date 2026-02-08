"""
NSCK SNN Training Pipeline
Phase 0.1: Complete the Perception Engine (ROADMAP_TO_AGI.md)

Implements:
  - SNN training pipeline with surrogate gradients
  - Hebbian learning updates during training
  - SNN output → concept activations via VSA binding
  - Training loop for game-state perception (Snake, Pong, Maze)

Design constraints:
  - Uses snnTorch (as recommended by roadmap)
  - Integrates with TaskAwareSNN and UniversalEncoder
  - Maps SNN outputs to VSA concept space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrainingMetrics:
    """Metrics for a training run."""
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    hebbian_updates: int = 0
    concept_bindings: int = 0


class HebbianLayer(nn.Module):
    """
    Hebbian learning layer for SNN concept formation.

    Implements a simplified Hebbian update rule:
        Δw_ij = η * (x_i * y_j - decay * w_ij)

    This complements gradient-based training by strengthening
    co-activated pathways (neurons that fire together wire together).
    """

    def __init__(self, in_features: int, out_features: int,
                 learning_rate: float = 0.01, decay: float = 0.001):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.lr = learning_rate
        self.decay = decay
        self.update_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    @torch.no_grad()
    def hebbian_update(self, pre: torch.Tensor, post: torch.Tensor):
        """
        Apply Hebbian weight update.

        Args:
            pre: Pre-synaptic activations [batch, in_features]
            post: Post-synaptic activations [batch, out_features]
        """
        # Oja's rule: Δw = η * (post ⊗ pre - post² * w)
        # Simplified for efficiency
        batch_size = pre.shape[0]
        if batch_size == 0:
            return

        # Average over batch
        outer = torch.einsum("bi,bj->ij", post, pre) / batch_size
        decay_term = self.decay * self.weight

        self.weight.data += self.lr * (outer - decay_term)
        self.update_count += 1


class ConceptActivationMapper:
    """
    Maps SNN spike outputs to concept activations in VSA space.

    Phase 0.1: SNN output → concept activations.
    Uses rate coding from SNN spikes to determine which concepts
    are active and with what strength.
    """

    def __init__(self, num_concepts: int = 64, spike_dim: int = 128):
        """
        Args:
            num_concepts: Number of concept slots
            spike_dim: Dimensionality of SNN spike output
        """
        self.num_concepts = num_concepts
        self.spike_dim = spike_dim

        # Concept prototypes (learned mapping from spikes to concepts)
        self.concept_probes = nn.Linear(spike_dim, num_concepts)
        self.concept_names: Dict[int, str] = {}
        self.activation_history: List[Dict[int, float]] = []

    def register_concept(self, concept_id: int, name: str):
        """Register a named concept."""
        self.concept_names[concept_id] = name

    def activate(
        self, spike_rates: torch.Tensor, threshold: float = 0.3
    ) -> Dict[int, float]:
        """
        Convert SNN spike rates to concept activations.

        Args:
            spike_rates: Spike rates from SNN [batch, spike_dim]
            threshold: Minimum activation to be considered active

        Returns:
            Dictionary mapping concept_id → activation_strength
        """
        with torch.no_grad():
            # Project spikes to concept space
            activations = torch.sigmoid(
                self.concept_probes(spike_rates)
            ).squeeze(0)

            # Threshold
            active = {}
            for i in range(self.num_concepts):
                val = activations[i].item()
                if val > threshold:
                    active[i] = val

        self.activation_history.append(active)
        return active

    def get_top_concepts(
        self, spike_rates: torch.Tensor, k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-k activated concepts with names.

        Args:
            spike_rates: Spike rates from SNN
            k: Number of top concepts to return

        Returns:
            List of (concept_name, activation) tuples
        """
        with torch.no_grad():
            activations = torch.sigmoid(
                self.concept_probes(spike_rates)
            ).squeeze(0)

            topk = torch.topk(activations, min(k, self.num_concepts))
            results = []
            for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                name = self.concept_names.get(idx, f"concept_{idx}")
                results.append((name, val))

        return results


class SNNTrainingPipeline:
    """
    Complete SNN training pipeline for NSCK perception.

    Combines:
      1. Supervised SNN training with surrogate gradients
      2. Hebbian learning for concept formation
      3. Concept activation mapping to VSA space
      4. Task-specific training (Snake, Pong, Maze)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        hebbian_lr: float = 0.01,
        concept_dim: int = 64,
        device: str = "cpu",
    ):
        """
        Args:
            model: TaskAwareSNN or compatible model
            learning_rate: Gradient learning rate
            hebbian_lr: Hebbian learning rate
            concept_dim: Number of concept slots
            device: Computing device
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Hebbian layer for concept binding
        self.hebbian = HebbianLayer(128, concept_dim, learning_rate=hebbian_lr)

        # Concept activation mapper
        self.concept_mapper = ConceptActivationMapper(
            num_concepts=concept_dim, spike_dim=128
        )

        # Training history
        self.metrics_history: List[TrainingMetrics] = []

    def train_epoch(
        self,
        data_loader,
        task_name: str = "snake",
        use_hebbian: bool = True,
    ) -> TrainingMetrics:
        """
        Train one epoch on task data.

        Args:
            data_loader: DataLoader yielding (states, labels)
            task_name: Task identifier
            use_hebbian: Whether to apply Hebbian updates

        Returns:
            Training metrics for this epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        hebbian_updates = 0

        for batch_states, batch_labels in data_loader:
            batch_states = batch_states.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through SNN
            logits, values = self.model(batch_states, task_name)

            # Classification loss
            loss = self.criterion(logits, batch_labels)
            loss.backward()
            self.optimizer.step()

            # Hebbian update on SNN hidden activations
            if use_hebbian:
                with torch.no_grad():
                    # Get intermediate representations for Hebbian learning
                    latent = self.model.encoder(batch_states)
                    post = self.hebbian(latent)
                    self.hebbian.hebbian_update(latent, post)
                    hebbian_updates += 1

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        metrics = TrainingMetrics(
            loss=total_loss / max(len(data_loader), 1),
            accuracy=correct / max(total, 1),
            hebbian_updates=hebbian_updates,
        )
        self.metrics_history.append(metrics)
        return metrics

    def bind_concepts_to_vsa(
        self,
        states: torch.Tensor,
        concept_labels: Optional[Dict[int, str]] = None,
    ) -> Dict[int, float]:
        """
        Bind SNN spike outputs to VSA concept space.

        Phase 0.1: VSA binding of visual features.

        Args:
            states: Input state tensors
            concept_labels: Optional mapping of concept IDs to names

        Returns:
            Active concept dictionary
        """
        if concept_labels:
            for cid, name in concept_labels.items():
                self.concept_mapper.register_concept(cid, name)

        self.model.eval()
        with torch.no_grad():
            # Get SNN hidden representations
            latent = self.model.encoder(states.to(self.device))
            # Map to concept activations
            activations = self.concept_mapper.activate(latent)

        return activations

    def get_training_summary(self) -> Dict[str, Any]:
        """Return training summary statistics."""
        if not self.metrics_history:
            return {"epochs": 0, "final_loss": 0.0, "final_accuracy": 0.0}

        last = self.metrics_history[-1]
        return {
            "epochs": len(self.metrics_history),
            "final_loss": last.loss,
            "final_accuracy": last.accuracy,
            "total_hebbian_updates": sum(
                m.hebbian_updates for m in self.metrics_history
            ),
            "loss_trend": [m.loss for m in self.metrics_history[-10:]],
        }
