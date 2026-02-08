"""
Phase 1 Training Demonstration
===============================
Demonstrates the Phase 1 neural learning capabilities:
1. Multi-task learning across Snake, Pong, and Maze
2. Rule extraction from trained neural policies
3. Dual inference with neural + symbolic reasoning

Usage:
    python train_phase1_demo.py [--epochs 10] [--batch-size 32]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import argparse

from multi_task_learning import (
    create_multitask_network, MultiTaskTrainer, MultiTaskMetrics
)
from rule_extraction import (
    create_dual_inference_system, NeuralRuleExtractor
)


def generate_synthetic_game_data(task_name: str, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic training data for a game task.
    
    In a real scenario, this would come from actual game episodes.
    """
    # State dimension is 512
    states = torch.randn(num_samples, 512)
    
    # Action space varies by task
    if task_name == "snake":
        num_actions = 4  # UP, DOWN, LEFT, RIGHT
    elif task_name == "pong":
        num_actions = 3  # UP, DOWN, STAY
    else:  # maze
        num_actions = 4  # UP, DOWN, LEFT, RIGHT
    
    # Random actions (in real training, these would be from policy)
    actions = torch.randint(0, num_actions, (num_samples,))
    
    # Returns (in real training, these would be computed from rewards)
    # Positive bias to ensure reasonable learning
    returns = torch.randn(num_samples) + 0.5
    
    return states, actions, returns


def train_multitask_policy(epochs: int = 10, batch_size: int = 32) -> nn.Module:
    """
    Train a multi-task policy on Snake, Pong, and Maze.
    
    Returns:
        Trained multi-task network
    """
    print("=" * 60)
    print("Phase 1.2: Multi-Task Learning Training")
    print("=" * 60)
    
    # Create multi-task network
    model = create_multitask_network()
    trainer = MultiTaskTrainer(model, lr=0.001)
    
    print(f"\nModel architecture:")
    print(f"  Shared encoder: 512 → 128 → 64")
    print(f"  Task heads: Snake (4 actions), Pong (3 actions), Maze (4 actions)")
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    
    # Training loop
    for epoch in range(epochs):
        # Generate batch data for each task
        batch_data = {}
        for task_name in ["snake", "pong", "maze"]:
            states, actions, returns = generate_synthetic_game_data(task_name, batch_size)
            batch_data[task_name] = (states, actions, returns)
        
        # Train step with gradient surgery
        metrics = trainer.train_step(batch_data, use_gradient_surgery=True)
        
        # Print progress
        if (epoch + 1) % 2 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {metrics.total_loss:.4f}")
            for task_name, loss in metrics.task_losses.items():
                print(f"  {task_name.capitalize()} Loss: {loss:.4f}")
    
    print("\n✓ Multi-task training complete!")
    return model


def extract_rules_from_policy(model: nn.Module, task_name: str = "snake") -> None:
    """
    Extract symbolic rules from trained neural policy.
    
    Demonstrates Phase 1.1: Neural-Symbolic Integration
    """
    print("\n" + "=" * 60)
    print("Phase 1.1: Rule Extraction from Neural Policy")
    print("=" * 60)
    
    # Generate sample states for rule extraction
    print(f"\nExtracting rules for {task_name.upper()}...")
    states = torch.randn(200, 512)
    
    # Create task-specific wrapper for rule extraction
    class TaskWrapper(nn.Module):
        def __init__(self, model, task_name):
            super().__init__()
            self.model = model
            self.task_name = task_name
        
        def forward(self, x):
            logits, _ = self.model(x, self.task_name)
            return logits
    
    task_model = TaskWrapper(model, task_name)
    
    # Define features and actions
    feature_names = [f"feat_{i}" for i in range(512)]
    if task_name == "snake":
        action_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
    elif task_name == "pong":
        action_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_STAY"]
    else:  # maze
        action_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
    
    # Extract rules
    extractor = NeuralRuleExtractor(task_model, feature_names, action_names)
    rule_set = extractor.extract_rules(states, max_rules=10, min_support=5)
    
    print(f"\n✓ Extracted {len(rule_set)} rules!")
    print("\nSample rules:")
    for i, rule in enumerate(rule_set.rules[:5]):  # Show first 5 rules
        print(f"  {i+1}. {rule}")


def demonstrate_dual_inference(model: nn.Module, task_name: str = "snake") -> None:
    """
    Demonstrate dual inference: neural + symbolic.
    
    Shows how the system can use both fast neural inference and
    interpretable symbolic rules.
    """
    print("\n" + "=" * 60)
    print("Phase 1.1: Dual Inference (Neural + Symbolic)")
    print("=" * 60)
    
    # Create task-specific wrapper
    class TaskWrapper(nn.Module):
        def __init__(self, model, task_name):
            super().__init__()
            self.model = model
            self.task_name = task_name
        
        def forward(self, x):
            logits, _ = self.model(x, self.task_name)
            return logits
    
    task_model = TaskWrapper(model, task_name)
    
    # Generate sample states
    sample_states = torch.randn(100, 512)
    
    # Define features and actions
    feature_names = [f"feat_{i}" for i in range(512)]
    if task_name == "snake":
        action_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
    elif task_name == "pong":
        action_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_STAY"]
    else:
        action_names = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
    
    # Create dual inference engine
    print(f"\nCreating dual inference engine for {task_name.upper()}...")
    engine = create_dual_inference_system(
        task_model, sample_states, feature_names, action_names
    )
    
    # Test decisions
    print("\nMaking decisions on test states...")
    num_tests = 10
    for i in range(num_tests):
        state = torch.randn(1, 512)
        state_features = {f"feat_{j}": np.random.randn() for j in range(10)}
        
        action, metadata = engine.decide(state, state_features, require_safe=True)
        
        if i < 3:  # Show first 3 decisions
            print(f"\n  Test {i+1}:")
            print(f"    Action: {action}")
            print(f"    Mode: {metadata['mode']}")
            print(f"    Neural Confidence: {metadata['neural_confidence']:.3f}")
            print(f"    Symbolic Confidence: {metadata['symbolic_confidence']:.3f}")
            if metadata.get('override'):
                print(f"    ⚠ Symbolic override!")
    
    # Print statistics
    stats = engine.get_stats()
    print(f"\n✓ Dual inference statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Neural decisions: {stats['neural_decisions']}")
    print(f"  Symbolic overrides: {stats['symbolic_overrides']}")
    print(f"  Safety overrides: {stats['safety_overrides']}")
    if stats['total_decisions'] > 0:
        print(f"  Neural rate: {stats['neural_rate']:.2%}")
        print(f"  Override rate: {stats['override_rate']:.2%}")


def main():
    """Main demonstration script."""
    parser = argparse.ArgumentParser(description="Phase 1 Training Demonstration")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--task", type=str, default="snake", 
                       choices=["snake", "pong", "maze"],
                       help="Task for rule extraction demo")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("NSCK Phase 1: Neural Learning Engine Demonstration")
    print("=" * 60)
    print("\nThis demo showcases:")
    print("  1. Multi-task learning (Phase 1.2)")
    print("  2. Rule extraction (Phase 1.1)")
    print("  3. Dual inference (Phase 1.1)")
    print("\nNote: Using synthetic data for demonstration purposes.")
    print("In production, this would use real game episodes.")
    
    # Step 1: Train multi-task policy
    model = train_multitask_policy(epochs=args.epochs, batch_size=args.batch_size)
    
    # Step 2: Extract rules from trained policy
    extract_rules_from_policy(model, task_name=args.task)
    
    # Step 3: Demonstrate dual inference
    demonstrate_dual_inference(model, task_name=args.task)
    
    print("\n" + "=" * 60)
    print("Phase 1 Demonstration Complete!")
    print("=" * 60)
    print("\nKey Achievements:")
    print("  ✓ Multi-task learning with shared encoder")
    print("  ✓ Gradient surgery for conflict resolution")
    print("  ✓ Rule extraction from neural policies")
    print("  ✓ Dual inference (neural + symbolic)")
    print("  ✓ Safety override mechanism")
    print("\nThe system successfully combines neural learning (fast, generalizable)")
    print("with symbolic reasoning (interpretable, safe).")
    print("\nNext: Phase 2 - Perception Systems (vision, audio, language)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
