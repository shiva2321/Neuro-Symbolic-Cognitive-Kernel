"""
NCGN v6.0 Runner - Main Entry Point

This module provides:
1. Training loop for games (Corridor, Snake)
2. Dashboard integration with v6.0 System 1 and learning
3. Real-time brain activity visualization

Usage:
    python -m ncgn_v6.runner
    # Opens browser to http://localhost:5000
"""

import time
import random
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .memory import GraphMemory, ClusterType
from .system1 import System1Engine
from .learning import DopamineModulator, ThreeFactorLearner, EpisodicBuffer
from .planner import EpisodicMonteCarlo, System2Controller
from .games.interface import EgocentricEncoder, SensoryInput, ValenceEstimator
from .games.corridor import CorridorGame


@dataclass
class TrainingConfig:
    """Configuration for training runs."""
    episodes: int = 100
    max_steps_per_episode: int = 100
    ticks_per_step: int = 3
    learning_rate: float = 0.1
    trace_decay: float = 0.95
    gamma: float = 0.99
    use_episodic: bool = True  # Use episodic (Monte Carlo) learning
    

class NCGNAgent:
    """
    The complete NCGN v6.0 agent integrating all components.
    
    This is the main interface for:
    - Playing games (Corridor, Snake)
    - Learning from rewards
    - Visualizing brain activity
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        
        # Core components
        self.memory = GraphMemory()
        self.engine = System1Engine(
            self.memory,
            trace_decay=self.config.trace_decay
        )
        
        # Learning components
        self.modulator = DopamineModulator()
        self.learner = ThreeFactorLearner(
            self.memory,
            learning_rate=self.config.learning_rate
        )
        self.episodic_buffer = EpisodicBuffer()
        
        # Planning components
        self.planner = EpisodicMonteCarlo()
        self.system2 = System2Controller(self.planner, [])
        
        # Game interface
        self.encoder = EgocentricEncoder(self.memory)
        self.valence = ValenceEstimator()
        
        # Current game
        self.game: Optional[CorridorGame] = None
        
        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Callbacks for dashboard
        self.on_step: Optional[Callable[[Dict], None]] = None
        self.on_episode_end: Optional[Callable[[Dict], None]] = None
    
    def setup_corridor_game(self, length: int = 7):
        """Set up the Corridor game environment."""
        self.game = CorridorGame(length=length)
        self.game_type = "corridor"
        
        # Create motor nodes for actions
        actions = self.game.get_possible_actions()
        for action in actions:
            self.memory.add_node(action, cluster=ClusterType.MOTOR)
        
        # Create initial sensory-motor connections
        for action in actions:
            # Hidden layer
            hidden = f"hidden_{action}"
            self.memory.add_node(hidden, cluster=ClusterType.HIDDEN)
            
            # Motor output
            self.memory.add_synapse(hidden, action, weight=0.5)
        
        # Update System 2 with possible actions
        self.system2.update_actions(actions)
        
        return self.game
    
    def setup_snake_game(self, width: int = 8, height: int = 8):
        """Set up the Snake game environment."""
        from .games.snake import SnakeGame
        self.game = SnakeGame(width=width, height=height)
        self.game_type = "snake"
        
        # Create motor nodes for 4-directional actions
        actions = self.game.get_possible_actions()
        for action in actions:
            self.memory.add_node(action, cluster=ClusterType.MOTOR)
            
            # Hidden layer
            hidden = f"hidden_{action}"
            self.memory.add_node(hidden, cluster=ClusterType.HIDDEN)
            
            # Motor output
            self.memory.add_synapse(hidden, action, weight=0.5)
        
        # Update System 2 with possible actions
        self.system2.update_actions(actions)
        
        return self.game
    
    def create_snake_network(self):
        """Create a sensory-motor network for Snake game."""
        actions = self.game.get_possible_actions() if self.game else []
        
        # Create bias node
        self.memory.add_node("bias", cluster=ClusterType.SENSORY)
        
        # Sensory nodes for Snake (egocentric vision)
        sensory_nodes = [
            "vision_food_ahead", "vision_empty_ahead", "vision_wall_ahead", "vision_danger_ahead",
            "vision_food_left", "vision_empty_left", "vision_wall_left", "vision_danger_left",
            "vision_food_right", "vision_empty_right", "vision_wall_right", "vision_danger_right",
            "vision_empty_behind", "vision_wall_behind",
            "proprio_food_direction_x", "proprio_food_direction_y"
        ]
        
        for sensory in sensory_nodes:
            self.memory.add_node(sensory, cluster=ClusterType.SENSORY)
        
        # Connect food sensors to relevant actions
        self.memory.add_synapse("vision_food_ahead", "hidden_action_move_up", weight=0.5)
        self.memory.add_synapse("vision_food_left", "hidden_action_move_left", weight=0.5)
        self.memory.add_synapse("vision_food_right", "hidden_action_move_right", weight=0.5)
        
        # Connect danger sensors to avoid those directions
        self.memory.add_synapse("vision_wall_ahead", "hidden_action_move_left", weight=0.3)
        self.memory.add_synapse("vision_wall_ahead", "hidden_action_move_right", weight=0.3)
        self.memory.add_synapse("vision_danger_ahead", "hidden_action_move_left", weight=0.4)
        self.memory.add_synapse("vision_danger_ahead", "hidden_action_move_right", weight=0.4)
        
        # Bias for exploration
        for action in actions:
            hidden = f"hidden_{action}"
            self.memory.add_synapse("bias", hidden, weight=0.2)
    
    def create_sensory_motor_network(self):
        """
        Create a proper sensory-motor network for learning.
        
        Network architecture:
        - Sensory nodes (created dynamically by encoder)
        - Hidden nodes connecting to motor outputs
        - Pre-wired connections from common sensory patterns
        """
        actions = self.game.get_possible_actions() if self.game else []
        
        # Create bias node for baseline activity
        self.memory.add_node("bias", cluster=ClusterType.SENSORY)
        
        # Pre-create expected sensory nodes and connect to hidden layer
        sensory_nodes = [
            "vision_goal_ahead",
            "vision_empty_ahead", 
            "vision_wall_ahead",
            "vision_empty_behind",
            "vision_wall_behind",
            "proprio_distance_to_goal"
        ]
        
        for sensory in sensory_nodes:
            self.memory.add_node(sensory, cluster=ClusterType.SENSORY)
        
        # Connect sensory to hidden with initial weights
        # Goal ahead should favor right action
        hidden_right = "hidden_action_move_right"
        hidden_left = "hidden_action_move_left"
        
        # Sensory → Hidden connections (these will be learned)
        self.memory.add_synapse("vision_goal_ahead", hidden_right, weight=0.6)
        self.memory.add_synapse("vision_empty_ahead", hidden_right, weight=0.4)
        self.memory.add_synapse("vision_wall_ahead", hidden_left, weight=0.3)
        self.memory.add_synapse("vision_wall_behind", hidden_right, weight=0.4)
        self.memory.add_synapse("proprio_distance_to_goal", hidden_right, weight=0.5)
        
        # Bias to both actions (exploration)
        self.memory.add_synapse("bias", hidden_right, weight=0.3)
        self.memory.add_synapse("bias", hidden_left, weight=0.3)
    
    def step(self, render: bool = False) -> tuple:
        """
        Execute one step in the current game.
        
        Uses epsilon-greedy action selection with learning based on
        sensory-motor associations.
        
        Returns:
            (observation, reward, done, info)
        """
        if not self.game:
            raise RuntimeError("No game set up. Call setup_corridor_game() first.")
        
        # 1. Get sensory encoding
        obs = self.game.get_sensory()
        sensory_activations = self.encoder.encode(obs)
        
        # 2. Inject sensory energy
        for node_id, energy in sensory_activations.items():
            self.engine.inject_energy(node_id, energy, ClusterType.SENSORY)
            
            # Also directly connect sensory to relevant hidden nodes
            # This creates traces for learning
            for hidden_id in ["hidden_action_move_right", "hidden_action_move_left"]:
                hidden = self.memory.get_node(hidden_id)
                if hidden:
                    hidden.energy = min(1.0, hidden.energy + energy * 0.3)
                    self.memory.mark_active(hidden_id)
        
        # 3. Run physics ticks  
        for _ in range(self.config.ticks_per_step):
            self.engine.tick()
        
        # 4. Epsilon-greedy action selection
        actions = self.game.get_possible_actions()
        epsilon = 0.3  # 30% random exploration
        
        if random.random() < epsilon:
            # Random exploration
            action = random.choice(actions)
        else:
            # Greedy: choose based on hidden→motor weights
            action_values = {}
            for action in actions:
                hidden_id = f"hidden_{action}"
                # Sum of incoming weights times sensory activations
                total = 0.0
                for src, syn in self.memory.get_incoming(hidden_id):
                    src_node = self.memory.get_node(src)
                    if src_node:
                        total += syn.weight * src_node.energy
                action_values[action] = total
            
            # Select best action
            action = max(action_values, key=action_values.get)
        
        # 5. Mark traces for the chosen action pathway
        hidden_id = f"hidden_{action}"
        for src, syn in self.memory.get_incoming(hidden_id):
            if self.memory.get_node(src) and self.memory.get_node(src).energy > 0.05:
                syn.set_eligible()
                self.memory.mark_synapse_traced(src, hidden_id)
        
        # Also trace hidden→motor
        for syn in self.memory.get_outgoing(hidden_id):
            if syn.target_id == action:
                syn.set_eligible()
                self.memory.mark_synapse_traced(hidden_id, action)
        
        # 6. Execute action in game
        new_obs, reward, done = self.game.step(action)
        
        # 7. Calculate valence
        valence = self.valence.estimate(reward, done, survived=not done or reward > 0)
        
        # 8. Record for episodic learning
        if self.config.use_episodic:
            self.episodic_buffer.record_step(self.memory)
            self.episodic_buffer.record_reward(valence)
        else:
            # Online learning
            dopamine = self.modulator.calculate_signal(valence)
            self.learner.apply_reward(dopamine)
        
        # 9. Render if requested
        if render:
            print(f"{self.game.render()} | Action: {action.split('_')[-1]} | R: {reward:.2f}")
        
        # 10. Notify callback
        if self.on_step:
            self.on_step({
                "tick": self.engine.current_tick,
                "action": action,
                "reward": reward,
                "valence": valence,
                "done": done,
                "active_nodes": len(self.memory.get_active_nodes()),
                "temperature": self.engine.temperature
            })
        
        return new_obs, reward, done, {"action": action, "valence": valence}
    
    def run_episode(self, render: bool = False) -> float:
        """
        Run one complete episode.
        
        Returns:
            Total reward for the episode
        """
        if not self.game:
            raise RuntimeError("No game set up.")
        
        self.game.reset()
        self.episodic_buffer.clear()
        
        total_reward = 0.0
        steps = 0
        
        for step in range(self.config.max_steps_per_episode):
            _, reward, done, _ = self.step(render=render)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Apply episodic learning
        if self.config.use_episodic and self.episodic_buffer.get_episode_length() > 0:
            from .learning import apply_episodic_learning
            apply_episodic_learning(
                self.memory,
                self.episodic_buffer,
                self.learner,
                self.modulator,
                gamma=self.config.gamma
            )
        
        # Record stats
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        # Notify callback
        if self.on_episode_end:
            self.on_episode_end({
                "episode": len(self.episode_rewards),
                "reward": total_reward,
                "steps": steps,
                "success": self.game.get_stats()["success_count"] > 0
            })
        
        return total_reward
    
    def train(self, episodes: int = None, render: bool = False) -> List[float]:
        """
        Train the agent for multiple episodes.
        
        Returns:
            List of rewards per episode
        """
        episodes = episodes or self.config.episodes
        
        for ep in range(episodes):
            reward = self.run_episode(render=render)
            
            if render and (ep + 1) % 10 == 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                print(f"\nEpisode {ep + 1}: Avg Reward (last 10): {avg_reward:.2f}")
        
        return self.episode_rewards
    
    def get_state_for_dashboard(self) -> dict:
        """Get current state formatted for dashboard visualization."""
        nodes = []
        for node_id, node in self.memory.nodes.items():
            nodes.append({
                "id": node_id,
                "energy": node.energy,
                "cluster": node.cluster.value if hasattr(node, 'cluster') else "hidden",
                "fired": node_id in self.engine.firing_set,
                "threshold": node.threshold
            })
        
        edges = []
        for source_id in self.memory.nodes:
            for synapse in self.memory.get_outgoing(source_id):
                edges.append({
                    "source": source_id,
                    "target": synapse.target_id,
                    "weight": synapse.weight,
                    "trace": synapse.trace,
                    "stability": synapse.stability
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "tick": self.engine.current_tick,
            "temperature": self.engine.temperature,
            "total_energy": self.memory.get_total_energy(),
            "episode": len(self.episode_rewards),
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "modulator": self.modulator.get_stats(),
            "learner": self.learner.get_stats()
        }


def demo_corridor_training():
    """Demo: Train agent on Corridor game."""
    print("=" * 50)
    print("NCGN v6.0 Corridor Training Demo")
    print("=" * 50)
    
    agent = NCGNAgent(TrainingConfig(
        episodes=50,
        learning_rate=0.2,
        use_episodic=True
    ))
    
    agent.setup_corridor_game(length=7)
    agent.create_sensory_motor_network()
    
    print("\nTraining...")
    rewards = agent.train(render=False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Episodes: {len(rewards)}")
    print(f"Final 10 avg reward: {sum(rewards[-10:])/10:.2f}")
    print(f"Success rate: {agent.game.get_stats()['success_rate']*100:.1f}%")
    
    # Show final learned weights
    print("\nLearned Weights:")
    for synapse in agent.memory.get_outgoing("hidden_action_move_right"):
        print(f"  hidden_action_move_right → {synapse.target_id}: {synapse.weight:.3f}")
    for synapse in agent.memory.get_outgoing("hidden_action_move_left"):
        print(f"  hidden_action_move_left → {synapse.target_id}: {synapse.weight:.3f}")


if __name__ == "__main__":
    demo_corridor_training()
