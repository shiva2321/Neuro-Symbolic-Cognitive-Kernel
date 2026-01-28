"""
NCGN v2.0 Game-Brain Bridge

Connects the Brain (System 1) to Games via Sensory Inputs and Action Outputs.
Handles:
1. Sensory Injection (Game State -> Brain Activation)
2. Action Selection (Brain Activation -> Game Action)
3. Reinforcement (Game Reward -> Brain Learning)
"""

from typing import Dict, Any, List, Optional
from ncgn.brain import Brain
from ncgn.games.interface import GameInterface, Direction

class GameBrainBridge:
    def __init__(self, brain: Brain, game: GameInterface, game_type: str = "custom"):
        self.brain = brain
        self.game = game
        self.game_type = game_type
        
        # Stats
        self.total_reward = 0.0
        self.steps = 0
        self.episode = 0
    
    def setup_concepts(self):
        """Create necessary concepts in the brain."""
        # Actions
        actions = self.game.get_possible_actions()
        for action in actions:
            if not self.brain.has_concept(action):
                self.brain.add_concept(action, initial_energy=0.1)
                
        # Senses (Custom per game type mostly)
        if self.game_type == 'corridor':
            self.brain.add_concept("see_goal", initial_energy=0.1)
            self.brain.add_concept("see_wall", initial_energy=0.1)
            self.brain.add_concept("see_empty", initial_energy=0.1)
            
            # Initial Wiring (Reflexes)
            # Goal -> Right
            self.brain.connect("see_goal", "action_move_right", weight=0.5)
            # Wall -> Left/Right (avoid)
            self.brain.connect("see_wall", "action_move_left", weight=0.3)
            
        elif self.game_type == 'snake':
            self.brain.add_concept("food_ahead", initial_energy=0.1)
            self.brain.add_concept("food_left", initial_energy=0.1)
            self.brain.add_concept("food_right", initial_energy=0.1)
            self.brain.add_concept("danger_ahead", initial_energy=0.1)
            self.brain.add_concept("danger_left", initial_energy=0.1)
            self.brain.add_concept("danger_right", initial_energy=0.1)

    def inject_senses(self):
        """Translate game state to brain energy."""
        obs = self.game.get_sensory()
        
        # Reset sensory nodes? Or just boost?
        # Ideally we decay others? The Brain does decay in think().
        
        if self.game_type == 'corridor':
            ahead = obs.vision.get(Direction.AHEAD, "empty")
            if ahead == "goal":
                self.brain.inject("see_goal", 0.9)
            elif ahead == "wall":
                self.brain.inject("see_wall", 0.9)
            else:
                self.brain.inject("see_empty", 0.5)
                
        elif self.game_type == 'snake':
            # Simplified mappings
            if obs.vision.get(Direction.AHEAD) == "food":
                self.brain.inject("food_ahead", 0.9)
            if obs.vision.get(Direction.LEFT) == "food":
                self.brain.inject("food_left", 0.9)
            if obs.vision.get(Direction.RIGHT) == "food":
                self.brain.inject("food_right", 0.9)
                
            if obs.vision.get(Direction.AHEAD) == "wall" or obs.vision.get(Direction.AHEAD) == "body":
                self.brain.inject("danger_ahead", 0.9)
            if obs.vision.get(Direction.LEFT) == "wall":
                self.brain.inject("danger_left", 0.9)
            if obs.vision.get(Direction.RIGHT) == "wall":
                self.brain.inject("danger_right", 0.9)

    def decide_action(self) -> str:
        """Propagate and select action."""
        self.brain.think(steps=3)
        
        actions = self.game.get_possible_actions()
        
        # Winner-Take-All on action nodes
        best_action = actions[0] # Default
        best_energy = -1.0
        
        # Add some random noise/exploration?
        import random
        if random.random() < 0.1:
            return random.choice(actions)
        
        for action in actions:
            energy = self.brain.get_concept_energy(action) or 0.0
            if energy > best_energy:
                best_energy = energy
                best_action = action
                
        return best_action

    def step(self) -> Dict[str, Any]:
        """Perform one step: Sense -> Think -> Act -> Learn."""
        self.inject_senses()
        
        action = self.decide_action()
        
        _, reward, done = self.game.step(action)
        self.total_reward += reward
        self.steps += 1
        
        # Learn
        self.brain.learn(reward)
        
        return {
            "action": action,
            "reward": reward,
            "done": done,
            "render": self.game.render(),
            "stats": self.game.get_stats()
        }
        
    def reset_stats(self):
        self.total_reward = 0.0
        self.steps = 0
        self.episode += 1
