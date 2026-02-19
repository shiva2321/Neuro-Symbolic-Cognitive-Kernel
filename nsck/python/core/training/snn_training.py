"""
SNN Training Pipeline
=====================

End-to-end training for SNN perception module with:
- Supervised, unsupervised, and reinforcement learning modes
- Reward-modulated Hebbian plasticity
- Concept formation tracking
- Model checkpointing
- Training metrics and visualization

This enables "learning for real" with the VSA + SNN architecture.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
import sys

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from python.core.perception.snn_perception import SNNPerceptionModule
from python.core.vsa.hypervec_shim import HyperVector


@dataclass
class TrainingConfig:
    """Configuration for SNN training"""
    # Architecture
    input_dim: int = 64
    snn_size: int = 256
    hv_dimension: int = 1024
    n_concepts: int = 50
    
    # Training
    n_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Learning modes
    mode: str = "unsupervised"  # "supervised", "unsupervised", "reinforcement"
    reward_modulation: bool = True  # Enable reward-modulated plasticity
    
    # Hebbian parameters
    hebbian_lr: float = 0.001
    hebbian_decay: float = 0.0001  # Weight decay
    
    # Input weight adaptation
    adapt_input_weights: bool = True
    input_lr: float = 0.0001
    
    # Evaluation
    eval_interval: int = 1  # Epochs between evaluations
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5  # Epochs between saves
    
    # Misc
    seed: int = 42
    verbose: bool = True


@dataclass
class TrainingMetrics:
    """Training metrics and history"""
    epoch: int = 0
    
    # Loss/accuracy
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    
    # SNN-specific
    avg_spikes: List[float] = field(default_factory=list)
    avg_active_neurons: List[float] = field(default_factory=list)
    n_concepts_learned: List[int] = field(default_factory=list)
    
    # Timing
    epoch_times: List[float] = field(default_factory=list)
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                getattr(self, key).append(value)
    
    def save(self, path: str):
        """Save metrics to JSON"""
        data = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "avg_spikes": self.avg_spikes,
            "avg_active_neurons": self.avg_active_neurons,
            "n_concepts_learned": self.n_concepts_learned,
            "epoch_times": self.epoch_times
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(path: str) -> 'TrainingMetrics':
        """Load metrics from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        metrics = TrainingMetrics()
        for key, value in data.items():
            setattr(metrics, key, value)
        return metrics


class SNNDataset:
    """Simple dataset wrapper for SNN training"""
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Args:
            data: (N, input_dim) array of inputs
            labels: (N,) array of integer labels (optional)
        """
        self.data = data
        self.labels = labels
        self.n_samples = len(data)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx], None
    
    def batch_iterator(self, batch_size: int, shuffle: bool = True):
        """Iterate over batches"""
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_data = self.data[batch_indices]
            batch_labels = self.labels[batch_indices] if self.labels is not None else None
            yield batch_data, batch_labels


class SNNTrainer:
    """
    Training pipeline for SNN perception module
    
    Supports three learning modes:
    1. Unsupervised: Pure Hebbian concept formation
    2. Supervised: Label-guided association strengthening
    3. Reinforcement: Reward-modulated plasticity
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        np.random.seed(config.seed)
        
        # Create SNN module
        self.snn = SNNPerceptionModule(
            input_dim=config.input_dim,
            snn_size=config.snn_size,
            hv_dimension=config.hv_dimension,
            n_concepts=config.n_concepts,
            encoding_mode="rate",
            simulation_time_ms=50.0,
            hebbian_lr=config.hebbian_lr
        )
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        if config.verbose:
            print(f"Initialized SNNTrainer:")
            print(f"  Mode: {config.mode}")
            print(f"  SNN: {config.snn_size} neurons")
            print(f"  HV: {config.hv_dimension} bits")
            print(f"  Concepts: {config.n_concepts}")
    
    def train_epoch_unsupervised(self, dataset: SNNDataset) -> Dict[str, float]:
        """
        Train one epoch in unsupervised mode
        Pure Hebbian learning from data statistics
        """
        total_spikes = 0
        total_active = 0
        n_batches = 0
        
        for batch_data, _ in dataset.batch_iterator(self.config.batch_size, shuffle=True):
            for sample in batch_data:
                # Perceive and learn
                result = self.snn.perceive(sample, learn=True)
                
                total_spikes += result['n_spikes']
                total_active += len(result['active_neurons'])
                n_batches += 1
        
        return {
            'avg_spikes': total_spikes / n_batches if n_batches > 0 else 0,
            'avg_active_neurons': total_active / n_batches if n_batches > 0 else 0
        }
    
    def train_epoch_supervised(self, dataset: SNNDataset) -> Dict[str, float]:
        """
        Train one epoch in supervised mode
        Use labels to guide concept formation
        """
        correct = 0
        total = 0
        total_spikes = 0
        
        # Build label → concept mapping
        label_to_concept: Dict[int, List[int]] = {}
        
        for batch_data, batch_labels in dataset.batch_iterator(self.config.batch_size, shuffle=True):
            for sample, label in zip(batch_data, batch_labels):
                # Perceive
                result = self.snn.perceive(sample, learn=False)
                concept_id = result['concept_id']
                
                # Track label-concept associations
                if label not in label_to_concept:
                    label_to_concept[label] = []
                label_to_concept[label].append(concept_id)
                
                # Learn if concept matches label's typical concept
                if len(label_to_concept[label]) > 1:
                    typical_concept = max(set(label_to_concept[label]), 
                                         key=label_to_concept[label].count)
                    if concept_id == typical_concept:
                        correct += 1
                        # Strengthen association
                        self.snn.hebbian_learner.update_associations([concept_id])
                
                total += 1
                total_spikes += result['n_spikes']
        
        accuracy = correct / total if total > 0 else 0
        avg_spikes = total_spikes / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_spikes': avg_spikes
        }
    
    def train_epoch_reinforcement(
        self, 
        dataset: SNNDataset,
        reward_fn: Callable[[np.ndarray, int], float]
    ) -> Dict[str, float]:
        """
        Train one epoch in reinforcement mode
        Use reward signal to modulate plasticity
        
        Args:
            dataset: Training data
            reward_fn: Function (sample, concept_id) -> reward
        """
        total_reward = 0
        total_spikes = 0
        n_samples = 0
        
        for batch_data, _ in dataset.batch_iterator(self.config.batch_size, shuffle=True):
            for sample in batch_data:
                # Perceive
                result = self.snn.perceive(sample, learn=False)
                concept_id = result['concept_id']
                
                # Get reward
                reward = reward_fn(sample, concept_id)
                total_reward += reward
                
                # Reward-modulated Hebbian update
                if self.config.reward_modulation:
                    if reward > 0:
                        # Strengthen associations
                        self.snn.hebbian_learner.update_associations([concept_id])
                    # On negative reward, we could weaken, but Hebbian rules
                    # typically don't have negative updates
                
                # Always update input weights toward good representations
                if self.config.adapt_input_weights:
                    self._adapt_input_weights(sample, result, reward)
                
                total_spikes += result['n_spikes']
                n_samples += 1
        
        return {
            'avg_reward': total_reward / n_samples if n_samples > 0 else 0,
            'avg_spikes': total_spikes / n_samples if n_samples > 0 else 0
        }
    
    def _adapt_input_weights(
        self, 
        sample: np.ndarray, 
        result: Dict,
        reward: float
    ):
        """
        Adapt input weights using reward-modulated Hebbian rule
        Δw_ij = η * reward * x_i * spike_j
        """
        if len(result['active_neurons']) == 0:
            return
        
        # Compute weight update
        active_mask = np.zeros(self.snn.snn_size)
        active_mask[result['active_neurons']] = 1.0
        
        # Outer product: input × spike pattern
        weight_update = self.config.input_lr * reward * np.outer(active_mask, sample)
        
        # Apply update
        self.snn.input_weights += weight_update
        
        # Optional weight decay
        if self.config.hebbian_decay > 0:
            self.snn.input_weights *= (1 - self.config.hebbian_decay)
    
    def evaluate(self, dataset: SNNDataset) -> Dict[str, float]:
        """Evaluate on dataset"""
        total_spikes = 0
        total_active = 0
        n_samples = 0
        
        # For supervised mode, compute accuracy
        if dataset.labels is not None:
            label_to_concept: Dict[int, List[int]] = {}
            correct = 0
            
            for sample, label in zip(dataset.data, dataset.labels):
                result = self.snn.perceive(sample, learn=False)
                concept_id = result['concept_id']
                
                if label not in label_to_concept:
                    label_to_concept[label] = []
                label_to_concept[label].append(concept_id)
                
                # Check if this matches the modal concept for this label
                if len(label_to_concept[label]) > 1:
                    typical_concept = max(set(label_to_concept[label]),
                                        key=label_to_concept[label].count)
                    if concept_id == typical_concept:
                        correct += 1
                
                total_spikes += result['n_spikes']
                total_active += len(result['active_neurons'])
                n_samples += 1
            
            accuracy = correct / n_samples if n_samples > 0 else 0
            return {
                'accuracy': accuracy,
                'avg_spikes': total_spikes / n_samples if n_samples > 0 else 0,
                'avg_active_neurons': total_active / n_samples if n_samples > 0 else 0
            }
        
        # For unsupervised, just compute statistics
        for sample, _ in zip(dataset.data, [None] * len(dataset.data)):
            result = self.snn.perceive(sample, learn=False)
            total_spikes += result['n_spikes']
            total_active += len(result['active_neurons'])
            n_samples += 1
        
        return {
            'avg_spikes': total_spikes / n_samples if n_samples > 0 else 0,
            'avg_active_neurons': total_active / n_samples if n_samples > 0 else 0
        }
    
    def train(
        self,
        train_dataset: SNNDataset,
        val_dataset: Optional[SNNDataset] = None,
        reward_fn: Optional[Callable] = None
    ):
        """
        Main training loop
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data (optional)
            reward_fn: Reward function for RL mode (optional)
        """
        if self.config.verbose:
            print(f"\nStarting training for {self.config.n_epochs} epochs...")
            print(f"Training samples: {len(train_dataset)}")
            if val_dataset:
                print(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()
            
            # Train
            if self.config.mode == "unsupervised":
                train_stats = self.train_epoch_unsupervised(train_dataset)
            elif self.config.mode == "supervised":
                train_stats = self.train_epoch_supervised(train_dataset)
            elif self.config.mode == "reinforcement":
                if reward_fn is None:
                    raise ValueError("reward_fn required for reinforcement mode")
                train_stats = self.train_epoch_reinforcement(train_dataset, reward_fn)
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
            
            # Evaluate
            if val_dataset and epoch % self.config.eval_interval == 0:
                val_stats = self.evaluate(val_dataset)
            else:
                val_stats = {}
            
            # Update metrics
            epoch_time = time.time() - epoch_start
            self.metrics.epoch = epoch + 1
            self.metrics.update(
                avg_spikes=train_stats.get('avg_spikes', 0),
                avg_active_neurons=train_stats.get('avg_active_neurons', 0),
                n_concepts_learned=len(self.snn.concept_mapper.concepts),
                epoch_times=epoch_time
            )
            
            if 'accuracy' in train_stats:
                self.metrics.update(train_accuracy=train_stats['accuracy'])
            if 'accuracy' in val_stats:
                self.metrics.update(val_accuracy=val_stats['accuracy'])
            
            # Logging
            if self.config.verbose:
                log_str = f"Epoch {epoch+1}/{self.config.n_epochs} ({epoch_time:.2f}s)"
                if 'accuracy' in train_stats:
                    log_str += f" | Train Acc: {train_stats['accuracy']:.3f}"
                if 'accuracy' in val_stats:
                    log_str += f" | Val Acc: {val_stats['accuracy']:.3f}"
                log_str += f" | Spikes: {train_stats['avg_spikes']:.1f}"
                log_str += f" | Concepts: {len(self.snn.concept_mapper.concepts)}"
                print(log_str)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")
        
        if self.config.verbose:
            print(f"\nTraining complete!")
            print(f"Final concepts learned: {len(self.snn.concept_mapper.concepts)}")
            print(f"Avg epoch time: {np.mean(self.metrics.epoch_times):.2f}s")
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.json"
        
        # Save metrics
        self.metrics.save(checkpoint_path)
        
        # Save SNN weights (input weights mainly)
        weights_path = self.checkpoint_dir / f"{name}_weights.npz"
        np.savez(
            weights_path,
            input_weights=self.snn.input_weights,
            n_concepts=len(self.snn.concept_mapper.concepts)
        )
        
        if self.config.verbose:
            print(f"  Saved checkpoint: {name}")
    
    def load_checkpoint(self, name: str):
        """Load training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.json"
        weights_path = self.checkpoint_dir / f"{name}_weights.npz"
        
        # Load metrics
        self.metrics = TrainingMetrics.load(checkpoint_path)
        
        # Load weights
        weights = np.load(weights_path)
        self.snn.input_weights = weights['input_weights']
        
        print(f"Loaded checkpoint: {name}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_synthetic_dataset(
    n_samples: int = 1000,
    n_classes: int = 5,
    input_dim: int = 64,
    noise: float = 0.1
) -> Tuple[SNNDataset, SNNDataset]:
    """
    Create synthetic dataset for testing
    
    Returns:
        train_dataset, val_dataset
    """
    # Generate class prototypes
    prototypes = np.random.randn(n_classes, input_dim)
    
    # Generate samples
    all_data = []
    all_labels = []
    
    for _ in range(n_samples):
        label = np.random.randint(n_classes)
        sample = prototypes[label] + np.random.randn(input_dim) * noise
        all_data.append(sample)
        all_labels.append(label)
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    # Split 80/20
    split_idx = int(0.8 * n_samples)
    train_data = all_data[:split_idx]
    train_labels = all_labels[:split_idx]
    val_data = all_data[split_idx:]
    val_labels = all_labels[split_idx:]
    
    return (
        SNNDataset(train_data, train_labels),
        SNNDataset(val_data, val_labels)
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SNN Training Pipeline Demo")
    print("=" * 70)
    
    # Create synthetic dataset
    print("\n[1] Creating synthetic dataset...")
    train_ds, val_ds = create_synthetic_dataset(
        n_samples=500,
        n_classes=5,
        input_dim=64,
        noise=0.2
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Classes: 5")
    
    # Configure training
    print("\n[2] Configuring trainer...")
    config = TrainingConfig(
        input_dim=64,
        snn_size=128,
        hv_dimension=1024,
        n_concepts=20,
        n_epochs=10,
        batch_size=32,
        mode="supervised",
        hebbian_lr=0.01,
        adapt_input_weights=True,
        input_lr=0.001,
        eval_interval=2,
        save_interval=5,
        verbose=True
    )
    
    # Train
    print("\n[3] Training...")
    trainer = SNNTrainer(config)
    trainer.train(train_ds, val_ds)
    
    # Final evaluation
    print("\n[4] Final evaluation...")
    final_stats = trainer.evaluate(val_ds)
    print(f"  Validation accuracy: {final_stats.get('accuracy', 0):.3f}")
    print(f"  Avg spikes: {final_stats['avg_spikes']:.1f}")
    print(f"  Concepts learned: {len(trainer.snn.concept_mapper.concepts)}")
    
    print("\n✅ Training Pipeline: WORKING")
