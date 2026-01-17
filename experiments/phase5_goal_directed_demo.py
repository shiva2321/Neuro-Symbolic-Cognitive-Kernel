"""
Phase 5 Demo: Goal-Directed Behavior & Intrinsic Motivation

Demonstrates:
1. Goal selection based on utility and curiosity
2. Intrinsic motivation (curiosity-driven exploration)
3. Self-modeling (outcome prediction and learning)
4. Full integration with gridworld environment
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system2 import System2, Goal
from core.metrics import SignalComputer
from core.control_layer import ControlLayer
from core.self_model import SelfModel
from experiments.gridworld_env import SimpleGridworld


def demo_goal_selection():
    """Demo 1: Goal selection based on utility and curiosity."""
    print("=" * 60)
    print("DEMO 1: Goal Selection")
    print("=" * 60)

    system2 = System2()

    # Define multiple goals
    goals = [
        Goal(goal_id=1, description="Explore corner", utility=0.3, progress=0.0),
        Goal(goal_id=2, description="Find reward", utility=0.8, progress=0.0),
        Goal(goal_id=3, description="Learn pattern", utility=0.5, progress=0.0),
    ]

    system2.set_goals(goals)

    print("\nAvailable Goals:")
    for goal in system2.get_all_goals():
        print(f"  {goal}")

    # Select goal by utility alone
    print("\n1. Selection by utility (β_curiosity=0.0):")
    selected = system2.select_goal(beta_curiosity=0.0)
    print(f"   Selected: {selected}")

    # Reset goals
    for g in goals:
        g.state = "idle"
    system2._current_goal_id = None

    # Select goal with curiosity weighting
    novelty_scores = {1: 0.9, 2: 0.1, 3: 0.5}  # High novelty for goal 1
    print("\n2. Selection with curiosity (β_curiosity=0.5):")
    print(f"   Novelty scores: {novelty_scores}")
    selected = system2.select_goal(beta_curiosity=0.5, novelty_scores=novelty_scores)
    print(f"   Selected: {selected}")
    print(f"   (Goal 1 wins due to high novelty: 0.3 + 0.5*0.9 = 0.75)")

    print("\n✓ Goal selection demo complete\n")


def demo_intrinsic_motivation():
    """Demo 2: Intrinsic motivation drives exploration."""
    print("=" * 60)
    print("DEMO 2: Intrinsic Motivation")
    print("=" * 60)

    signal_computer = SignalComputer()
    control_layer = ControlLayer(beta_blend=0.3)

    print("\nScenario: Agent explores environment with curiosity")
    print("Extrinsic reward: 0.1 (small external reward)")
    print("Novelty: 0.8 (novel pattern)")
    print("Pred error delta: 0.3 (learning happening)")

    # Compute intrinsic reward
    intrinsic = signal_computer.compute_intrinsic_reward(
        novelty=0.8,
        pred_error_delta=0.3,
        beta_intrinsic=0.5
    )
    print(f"\nIntrinsic reward: {intrinsic:.3f}")

    # Blend rewards
    extrinsic = 0.1
    blended = control_layer.blend_rewards(extrinsic, intrinsic)

    print(f"\nReward blending (β={control_layer.beta_blend}):")
    print(f"  Extrinsic only: {extrinsic:.3f}")
    print(f"  Intrinsic only: {intrinsic:.3f}")
    print(f"  Blended reward: {blended:.3f}")
    print(f"  Boost from curiosity: {(blended - extrinsic):.3f}")

    print("\n✓ Intrinsic motivation demo complete\n")


def demo_self_modeling():
    """Demo 3: Self-model learns to predict outcomes."""
    print("=" * 60)
    print("DEMO 3: Self-Modeling")
    print("=" * 60)

    model = SelfModel(input_dim=5, num_actions=4)

    print("\nScenario: Agent learns action-outcome relationships")
    print("State: [0.2, 0.2, 0.8, 0.8, 0.6] (agent far from goal)")

    state = [0.2, 0.2, 0.8, 0.8, 0.6]
    action = 1  # Down

    # Initial prediction (no experience)
    pred_1, unc_1 = model.predict_outcome(state, action)
    print(f"\nInitial prediction:")
    print(f"  Predicted next state: {[f'{v:.2f}' for v in pred_1]}")
    print(f"  Uncertainty: {unc_1:.3f} (high - no experience)")

    # Learn from experience
    print("\nLearning from 5 experiences...")
    for i in range(5):
        actual_next = [0.2, 0.4, 0.8, 0.8, 0.5]  # Row increases (down)
        residual = model.update(state, action, actual_next)
        print(f"  Update {i+1}: residual = {residual:.4f}")

    # Later prediction (with experience)
    pred_2, unc_2 = model.predict_outcome(state, action)
    print(f"\nAfter learning:")
    print(f"  Predicted next state: {[f'{v:.2f}' for v in pred_2]}")
    print(f"  Uncertainty: {unc_2:.3f} (lower - learned pattern)")

    # Statistics
    stats = model.get_statistics()
    print(f"\nModel statistics:")
    print(f"  Mean residual: {stats['mean_residual']:.4f}")
    print(f"  Residual variance: {stats['residual_variance']:.4f}")
    print(f"  Transitions learned: {stats['num_transitions_learned']}")

    print("\n✓ Self-modeling demo complete\n")


def demo_full_integration():
    """Demo 4: Full integration with gridworld."""
    print("=" * 60)
    print("DEMO 4: Full Integration")
    print("=" * 60)

    # Initialize all components
    system2 = System2()
    signal_computer = SignalComputer()
    control_layer = ControlLayer(beta_blend=0.4)
    self_model = SelfModel(input_dim=5, num_actions=4)
    env = SimpleGridworld(grid_size=5, max_steps=20)

    # Set goal
    goal = Goal(
        goal_id=1,
        description="Reach target location",
        utility=1.0,
        progress=0.0,
        reward_model=lambda state: 1.0 - state[4] if len(state) > 4 else 0.0  # Reward = 1 - distance
    )
    system2.set_goals([goal])
    selected_goal = system2.select_goal()

    print(f"\nGoal: {selected_goal.description}")
    print(f"Environment: {env.grid_size}x{env.grid_size} gridworld")
    print(f"Starting position: {env.agent_pos}")
    print(f"Goal position: {env.goal_pos}\n")

    # Run episode
    obs = env.reset()
    episode_data = []

    for step in range(10):
        # Simple policy: move toward goal
        if obs[0] < obs[2]:  # Agent row < goal row
            action = 1  # Down
        elif obs[1] < obs[3]:  # Agent col < goal col
            action = 3  # Right
        else:
            action = 0  # Default

        # Predict outcome
        pred_next, uncertainty = self_model.predict_outcome(obs, action)

        # Take action
        next_obs, extrinsic_reward, done = env.step(action)

        # Compute intrinsic reward
        novelty = 0.5 if step < 3 else 0.1  # Higher novelty early
        pred_error = self_model.compute_residual_error(pred_next, next_obs)
        intrinsic_reward = signal_computer.compute_intrinsic_reward(
            novelty=novelty,
            pred_error_delta=pred_error,
            beta_intrinsic=0.3
        )

        # Blend rewards
        total_reward = control_layer.blend_rewards(extrinsic_reward, intrinsic_reward)

        # Update self-model
        self_model.update(obs, action, next_obs)

        # Update goal progress
        system2.evaluate_goal_progress(1, {"distance": obs[4]})

        # Record
        episode_data.append({
            "step": step,
            "pos": env.agent_pos,
            "action": ["Up", "Down", "Left", "Right"][action],
            "extrinsic": extrinsic_reward,
            "intrinsic": intrinsic_reward,
            "total": total_reward,
            "pred_error": pred_error,
            "progress": selected_goal.progress
        })

        obs = next_obs

        if done:
            break

    # Display results
    print("Episode trace:")
    print("Step | Position | Action | Ext | Int | Total | Pred Err | Progress")
    print("-" * 75)
    for data in episode_data:
        print(f"{data['step']:4d} | {str(data['pos']):8s} | {data['action']:6s} | "
              f"{data['extrinsic']:+.2f} | {data['intrinsic']:.3f} | "
              f"{data['total']:+.3f} | {data['pred_error']:.4f} | {data['progress']:.2f}")

    print(f"\nFinal statistics:")
    print(f"  Goal progress: {selected_goal.progress:.2f}")
    print(f"  Goal state: {selected_goal.state}")
    print(f"  Total reward: {env.total_reward:.2f}")
    print(f"  Steps taken: {env.steps}")
    print(f"  Self-model mean residual: {self_model.get_mean_residual():.4f}")

    print("\n✓ Full integration demo complete\n")


def main():
    """Run all Phase 5 demos."""
    print("\n" + "=" * 60)
    print("PHASE 5: GOAL-DIRECTED BEHAVIOR & INTRINSIC MOTIVATION")
    print("=" * 60 + "\n")

    demo_goal_selection()
    demo_intrinsic_motivation()
    demo_self_modeling()
    demo_full_integration()

    print("=" * 60)
    print("ALL PHASE 5 DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nPhase 5 Features Demonstrated:")
    print("  ✓ Goal API and selection policies")
    print("  ✓ Intrinsic reward computation")
    print("  ✓ Reward blending (extrinsic + intrinsic)")
    print("  ✓ Self-model prediction and learning")
    print("  ✓ Full integration with environment")
    print("\nNext steps: Run tests with `python -m unittest discover tests`")
    print()


if __name__ == '__main__':
    main()
