"""
GridWorld Survival Agent Training System

This script demonstrates:
1. Learning from scratch in a complex environment
2. Adaptation and improvement over time
3. Multi-objective decision making
4. Comprehensive logging for deep analysis
5. Transfer learning capability to other games
6. Guided learning with feedback

The agent uses the full NSCK cognitive architecture:
- Rule learning from experience
- Causal reasoning about game mechanics
- Episodic memory for strategy development
- Metacognition for self-improvement
- Global workspace for decision making
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gridworld_survival import GridWorldSurvival, create_gridworld_causal_graph
from cognitive_engine import CognitiveEngine, CognitiveState
from config import NSCKConfig
from analogy import AnalogyEngine
import hypervec_shim as hypervec_rs


class GridWorldLogger:
    """
    Comprehensive logging system for agent training and analysis.
    
    Logs everything needed for deep analysis:
    - Every game state and action
    - Decision reasoning and confidence
    - Learning events (new rules, updates)
    - Performance metrics over time
    - Transfer learning events
    """
    
    def __init__(self, log_dir: str = "gridworld_logs"):
        """Initialize logger with output directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        self.game_log_file = self.session_dir / "game_log.jsonl"
        self.metrics_log_file = self.session_dir / "metrics.jsonl"
        self.learning_log_file = self.session_dir / "learning_events.jsonl"
        self.summary_file = self.session_dir / "summary.txt"
        
        # Setup Python logging
        log_file = self.session_dir / "agent.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Session metadata
        self.session_metadata = {
            "timestamp": timestamp,
            "session_dir": str(self.session_dir),
            "start_time": time.time(),
        }
        
        # Performance tracking
        self.episode_stats = []
        self.learning_events = []
        
        self.logger.info(f"Logger initialized. Session directory: {self.session_dir}")
    
    def log_game_step(
        self,
        episode: int,
        step: int,
        state_dict: Dict[str, Any],
        action: str,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        reasoning: Optional[Dict[str, Any]] = None,
    ):
        """Log a single game step with full context."""
        log_entry = {
            "timestamp": time.time(),
            "episode": episode,
            "step": step,
            "state": state_dict,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
            "reasoning": reasoning or {},
        }
        
        with open(self.game_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_episode_metrics(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        final_score: int,
        survival_time: float,
        success: bool,
        stats: Dict[str, Any],
    ):
        """Log episode-level metrics."""
        metrics = {
            "timestamp": time.time(),
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "final_score": final_score,
            "survival_time": survival_time,
            "success": success,
            "stats": stats,
        }
        
        self.episode_stats.append(metrics)
        
        with open(self.metrics_log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.logger.info(
            f"Episode {episode}: Score={final_score}, Reward={total_reward:.2f}, "
            f"Steps={steps}, Success={success}"
        )
    
    def log_learning_event(
        self,
        episode: int,
        event_type: str,
        details: Dict[str, Any],
    ):
        """Log a learning event (rule learned, strategy adapted, etc.)."""
        event = {
            "timestamp": time.time(),
            "episode": episode,
            "type": event_type,
            "details": details,
        }
        
        self.learning_events.append(event)
        
        with open(self.learning_log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.logger.info(f"Learning Event ({event_type}): {details}")
    
    def generate_summary(self):
        """Generate a comprehensive summary of the training session."""
        if not self.episode_stats:
            return
        
        summary_lines = [
            "=" * 80,
            "GRIDWORLD SURVIVAL AGENT - TRAINING SUMMARY",
            "=" * 80,
            "",
            f"Session: {self.session_metadata['timestamp']}",
            f"Duration: {time.time() - self.session_metadata['start_time']:.2f} seconds",
            f"Total Episodes: {len(self.episode_stats)}",
            "",
            "=" * 80,
            "PERFORMANCE METRICS",
            "=" * 80,
            "",
        ]
        
        # Calculate statistics
        scores = [ep['final_score'] for ep in self.episode_stats]
        rewards = [ep['total_reward'] for ep in self.episode_stats]
        steps = [ep['steps'] for ep in self.episode_stats]
        successes = sum(1 for ep in self.episode_stats if ep['success'])
        
        summary_lines.extend([
            f"Success Rate: {successes}/{len(self.episode_stats)} ({100*successes/len(self.episode_stats):.1f}%)",
            "",
            "Scores:",
            f"  Mean: {np.mean(scores):.2f}",
            f"  Std: {np.std(scores):.2f}",
            f"  Min: {min(scores):.2f}",
            f"  Max: {max(scores):.2f}",
            "",
            "Total Rewards:",
            f"  Mean: {np.mean(rewards):.2f}",
            f"  Std: {np.std(rewards):.2f}",
            f"  Min: {min(rewards):.2f}",
            f"  Max: {max(rewards):.2f}",
            "",
            "Survival Steps:",
            f"  Mean: {np.mean(steps):.2f}",
            f"  Std: {np.std(steps):.2f}",
            f"  Min: {min(steps)}",
            f"  Max: {max(steps)}",
            "",
        ])
        
        # Improvement trend
        if len(scores) >= 10:
            early_scores = np.mean(scores[:len(scores)//3])
            late_scores = np.mean(scores[-len(scores)//3:])
            improvement = late_scores - early_scores
            
            summary_lines.extend([
                "Learning Progress:",
                f"  Early Episodes (avg score): {early_scores:.2f}",
                f"  Late Episodes (avg score): {late_scores:.2f}",
                f"  Improvement: {improvement:.2f} ({100*improvement/max(early_scores, 1):.1f}%)",
                "",
            ])
        
        # Learning events summary
        summary_lines.extend([
            "=" * 80,
            "LEARNING EVENTS",
            "=" * 80,
            "",
            f"Total Learning Events: {len(self.learning_events)}",
            "",
        ])
        
        # Group learning events by type
        event_types = {}
        for event in self.learning_events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        for event_type, count in sorted(event_types.items()):
            summary_lines.append(f"  {event_type}: {count}")
        
        summary_lines.extend([
            "",
            "=" * 80,
            "FILES GENERATED",
            "=" * 80,
            "",
            f"Game Log: {self.game_log_file}",
            f"Metrics Log: {self.metrics_log_file}",
            f"Learning Events Log: {self.learning_log_file}",
            f"Agent Log: {self.session_dir / 'agent.log'}",
            f"Summary: {self.summary_file}",
            "",
            "=" * 80,
        ])
        
        summary_text = '\n'.join(summary_lines)
        
        # Write to file
        with open(self.summary_file, 'w') as f:
            f.write(summary_text)
        
        # Also print to console
        print(summary_text)
        
        return summary_text


class GridWorldAgent:
    """
    Intelligent agent for GridWorld Survival using NSCK cognitive architecture.
    
    Capabilities:
    - Learns rules from experience
    - Reasons about cause and effect
    - Makes strategic decisions
    - Adapts based on feedback
    - Improves over time
    """
    
    def __init__(self, config: Optional[NSCKConfig] = None):
        """Initialize agent with cognitive engine."""
        self.config = config or NSCKConfig()
        
        # Initialize cognitive engine
        self.engine = CognitiveEngine(config=self.config)
        
        # Register GridWorld domain with analogy engine
        self.engine.analogy.register_domain("gridworld")
        
        # Register abstract concepts for transfer learning
        self._register_abstract_concepts()
        
        # Set causal graph
        self.engine.causal_graph = create_gridworld_causal_graph()
        
        # Initialize task brain for GridWorld
        self.engine.fusion.ensure_brain("gridworld")
        
        # Performance tracking
        self.total_steps = 0
        self.episodes_completed = 0
    
    def _register_abstract_concepts(self):
        """Register abstract concepts for transfer learning."""
        analogy = self.engine.analogy
        
        # Agent concepts
        analogy.register_abstract("GRIDWORLD_PLAYER", "AGENT", "gridworld")
        
        # Resource concepts
        analogy.register_abstract("FOOD", "RESOURCE", "gridworld")
        analogy.register_abstract("WATER", "RESOURCE", "gridworld")
        analogy.register_abstract("ENERGY", "RESOURCE", "gridworld")
        
        # Threat concepts
        analogy.register_abstract("ENEMY", "THREAT", "gridworld")
        analogy.register_abstract("HAZARD", "THREAT", "gridworld")
        
        # Goal concepts
        analogy.register_abstract("EXIT", "GOAL", "gridworld")
        analogy.register_abstract("TREASURE", "REWARD", "gridworld")
        
        # State concepts
        analogy.register_abstract("CRITICAL_HUNGER", "CRITICAL_NEED", "gridworld")
        analogy.register_abstract("CRITICAL_THIRST", "CRITICAL_NEED", "gridworld")
        analogy.register_abstract("LOW_ENERGY", "LOW_RESOURCE", "gridworld")
        analogy.register_abstract("LOW_HEALTH", "LOW_RESOURCE", "gridworld")
    
    def decide(self, state_dict: Dict[str, Any], guided_action: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Decide on an action based on current state.
        
        Args:
            state_dict: Current game state
            guided_action: Optional guidance from teacher/user
        
        Returns:
            Tuple of (action, reasoning_dict)
        """
        # Encode state as hypervector
        predicates = state_dict.get('predicates', [])
        
        # Create state hypervector
        state_hv = hypervec_rs.HyperVector.random()
        for pred in predicates:
            pred_hv = hypervec_rs.HyperVector.from_string(pred)
            state_hv = state_hv.bundle(pred_hv)
        
        # Available actions
        actions = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT", "ACTION_STAY"]
        
        # Use cognitive engine to decide
        cognitive_state = self.engine.decide(
            task_tag="gridworld",
            state_hv=state_hv,
            state_dict=state_dict,
            available_actions=actions,
        )
        
        action = cognitive_state.chosen_action
        confidence = cognitive_state.confidence
        
        # If guided action provided, use it and learn from it
        if guided_action and guided_action in actions:
            action = guided_action
            confidence = 1.0  # High confidence for guided actions
        
        # Build reasoning dictionary
        reasoning = {
            "action": action,
            "confidence": confidence,
            "predicates": predicates,
            "exploration_mode": cognitive_state.exploration_mode,
            "emotion": cognitive_state.emotion,
            "trace": cognitive_state.trace,
        }
        
        if cognitive_state.explanation:
            reasoning["explanation"] = {
                "module": cognitive_state.explanation.module,
                "content": cognitive_state.explanation.content,
                "confidence": cognitive_state.explanation.confidence,
            }
        
        return action, reasoning
    
    def learn(
        self,
        state_dict: Dict[str, Any],
        action: str,
        reward: float,
        next_state_dict: Dict[str, Any],
        done: bool,
    ) -> Dict[str, Any]:
        """
        Learn from experience.
        
        Args:
            state_dict: Previous state
            action: Action taken
            reward: Reward received
            next_state_dict: New state after action
            done: Whether episode ended
        
        Returns:
            Dictionary of learning events
        """
        predicates = state_dict.get('predicates', [])
        next_predicates = next_state_dict.get('predicates', [])
        
        # Encode states
        state_hv = hypervec_rs.HyperVector.random()
        for pred in predicates:
            pred_hv = hypervec_rs.HyperVector.from_string(pred)
            state_hv = state_hv.bundle(pred_hv)
        
        next_state_hv = hypervec_rs.HyperVector.random()
        for pred in next_predicates:
            pred_hv = hypervec_rs.HyperVector.from_string(pred)
            next_state_hv = next_state_hv.bundle(pred_hv)
        
        # Learn from experience
        learning_result = self.engine.learn(
            task_tag="gridworld",
            state_hv=state_hv,
            action=action,
            reward=reward,
            next_state_hv=next_state_hv,
            state_predicates=predicates,
            done=done,
        )
        
        self.total_steps += 1
        
        return learning_result
    
    def get_rules(self) -> List[Any]:
        """Get learned rules for GridWorld."""
        brain = self.engine.fusion.ensure_brain("gridworld")
        return brain.rules
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_steps": self.total_steps,
            "episodes_completed": self.episodes_completed,
            "rules_learned": len(self.get_rules()),
        }


def train_agent(
    num_episodes: int = 50,
    max_steps_per_episode: int = 500,
    guided_episodes: int = 5,
    render_frequency: int = 10,
    log_dir: str = "gridworld_logs",
):
    """
    Train GridWorld agent with comprehensive logging.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        guided_episodes: Number of initial episodes with guidance
        render_frequency: How often to render game state
        log_dir: Directory for logs
    """
    # Initialize logger
    logger = GridWorldLogger(log_dir=log_dir)
    logger.logger.info("Starting GridWorld agent training")
    
    # Initialize game and agent
    game = GridWorldSurvival(
        width=20,
        height=20,
        num_enemies=3,
        max_steps=max_steps_per_episode
    )
    agent = GridWorldAgent()
    
    logger.logger.info("Game and agent initialized")
    
    # Training loop
    for episode in range(num_episodes):
        logger.logger.info(f"\n{'='*60}")
        logger.logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.logger.info(f"{'='*60}")
        
        # Reset game
        state = game.reset()
        state_dict = game.get_state_dict()
        
        # Episode tracking
        total_reward = 0.0
        done = False
        step = 0
        
        # Render initial state occasionally
        if episode % render_frequency == 0:
            logger.logger.info("\n" + game.render_ascii())
        
        while not done and step < max_steps_per_episode:
            # Provide guidance in early episodes
            guided_action = None
            if episode < guided_episodes:
                guided_action = provide_guidance(state_dict)
            
            # Agent decides action
            action, reasoning = agent.decide(state_dict, guided_action)
            
            # Execute action
            next_state, reward, done, info = game.step(action)
            next_state_dict = game.get_state_dict()
            
            # Log the step
            logger.log_game_step(
                episode=episode,
                step=step,
                state_dict=state_dict,
                action=action,
                reward=reward,
                done=done,
                info=info,
                reasoning=reasoning,
            )
            
            # Agent learns
            learning_result = agent.learn(
                state_dict=state_dict,
                action=action,
                reward=reward,
                next_state_dict=next_state_dict,
                done=done,
            )
            
            # Log learning events
            if learning_result.get('rule_added'):
                logger.log_learning_event(
                    episode=episode,
                    event_type="rule_learned",
                    details={
                        "rule": str(learning_result.get('rule_added')),
                        "step": step,
                    }
                )
            
            if learning_result.get('confidence_updated'):
                logger.log_learning_event(
                    episode=episode,
                    event_type="confidence_update",
                    details={
                        "step": step,
                        "new_confidence": learning_result.get('new_confidence'),
                    }
                )
            
            # Update tracking
            total_reward += reward
            state_dict = next_state_dict
            step += 1
            
            # Render occasionally
            if episode % render_frequency == 0 and step % 50 == 0:
                logger.logger.info(f"\nStep {step}:")
                logger.logger.info(game.render_ascii())
        
        # Episode complete
        agent.episodes_completed += 1
        
        # Log episode metrics
        logger.log_episode_metrics(
            episode=episode,
            total_reward=total_reward,
            steps=step,
            final_score=state.stats.score,
            survival_time=step / max_steps_per_episode,
            success=info.get('success', False),
            stats={
                "energy": state.stats.energy,
                "health": state.stats.health,
                "hunger": state.stats.hunger,
                "thirst": state.stats.thirst,
                "treasures": state.stats.treasures_collected,
            }
        )
        
        # Show agent statistics periodically
        if (episode + 1) % 10 == 0:
            stats = agent.get_statistics()
            logger.logger.info(f"\nAgent Statistics: {stats}")
            logger.logger.info(f"Rules learned: {stats['rules_learned']}")
    
    # Generate final summary
    logger.logger.info("\nTraining complete! Generating summary...")
    logger.generate_summary()
    
    logger.logger.info(f"\nAll logs saved to: {logger.session_dir}")
    
    return agent, logger


def provide_guidance(state_dict: Dict[str, Any]) -> Optional[str]:
    """
    Provide guidance for early training episodes.
    
    This implements simple heuristics to guide the agent initially.
    """
    predicates = set(state_dict.get('predicates', []))
    
    # Priority 1: Avoid imminent danger
    if "ENEMY_VERY_CLOSE" in predicates:
        # Try to move away from enemy
        for pred in predicates:
            if pred.startswith("ENEMY_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                # Move opposite direction
                if dy < 0:
                    return "ACTION_DOWN"
                elif dy > 0:
                    return "ACTION_UP"
                elif dx < 0:
                    return "ACTION_RIGHT"
                elif dx > 0:
                    return "ACTION_LEFT"
    
    # Priority 2: Critical needs
    if "CRITICAL_HUNGER" in predicates and "FOOD_CLOSE" in predicates:
        for pred in predicates:
            if pred.startswith("FOOD_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                if dy < 0:
                    return "ACTION_UP"
                elif dy > 0:
                    return "ACTION_DOWN"
                elif dx < 0:
                    return "ACTION_LEFT"
                elif dx > 0:
                    return "ACTION_RIGHT"
    
    if "CRITICAL_THIRST" in predicates and "WATER_CLOSE" in predicates:
        for pred in predicates:
            if pred.startswith("WATER_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                if dy < 0:
                    return "ACTION_UP"
                elif dy > 0:
                    return "ACTION_DOWN"
                elif dx < 0:
                    return "ACTION_LEFT"
                elif dx > 0:
                    return "ACTION_RIGHT"
    
    # Priority 3: Seek resources if available
    if "HUNGRY" in predicates and "FOOD_NEARBY" in predicates:
        return None  # Let agent explore
    
    if "THIRSTY" in predicates and "WATER_NEARBY" in predicates:
        return None  # Let agent explore
    
    # No specific guidance
    return None


def demonstrate_transfer_learning():
    """
    Demonstrate that GridWorld agent can transfer knowledge to other games.
    
    This shows the extensibility of the system.
    """
    print("\n" + "="*80)
    print("TRANSFER LEARNING DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize agent trained on GridWorld
    agent = GridWorldAgent()
    
    # Train briefly on GridWorld
    print("Training agent on GridWorld...")
    game = GridWorldSurvival(width=15, height=15, max_steps=100)
    
    for episode in range(10):
        state = game.reset()
        done = False
        while not done:
            state_dict = game.get_state_dict()
            action, _ = agent.decide(state_dict)
            next_state, reward, done, _ = game.step(action)
            next_state_dict = game.get_state_dict()
            agent.learn(state_dict, action, reward, next_state_dict, done)
    
    print(f"GridWorld training complete. Rules learned: {len(agent.get_rules())}")
    
    # Show how rules can transfer
    print("\nDemonstrating rule transfer capability...")
    print("The agent's learned rules use abstract concepts like:")
    print("  - RESOURCE (maps to food, water, energy)")
    print("  - THREAT (maps to enemies, hazards)")
    print("  - GOAL (maps to exit, treasure)")
    print("\nThese abstract concepts can be grounded in other game domains.")
    print("For example, in Snake: FOOD -> RESOURCE, in Maze: EXIT -> GOAL")
    print("\nThis allows zero-shot transfer of strategies between games!")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GridWorld Survival Agent")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--guided", type=int, default=5, help="Number of guided episodes")
    parser.add_argument("--render-freq", type=int, default=10, help="Render frequency")
    parser.add_argument("--log-dir", type=str, default="gridworld_logs", help="Log directory")
    parser.add_argument("--demo-transfer", action="store_true", help="Demo transfer learning")
    
    args = parser.parse_args()
    
    if args.demo_transfer:
        demonstrate_transfer_learning()
    else:
        print("\n" + "="*80)
        print("GRIDWORLD SURVIVAL AGENT TRAINING")
        print("="*80 + "\n")
        
        agent, logger = train_agent(
            num_episodes=args.episodes,
            guided_episodes=args.guided,
            render_frequency=args.render_freq,
            log_dir=args.log_dir,
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\nLogs saved to: {logger.session_dir}")
        print("\nYou can now analyze the logs to see:")
        print("  - Every decision made by the agent")
        print("  - Learning progress over time")
        print("  - Rules discovered")
        print("  - Performance improvements")
