"""
NCGN v6.0 Learning Module - 3-Factor Hebbian Plasticity

Implements valence-driven learning with:
- DopamineModulator: Calculates Reward Prediction Error (RPE)
- ThreeFactorLearner: 3-Factor Hebbian learning (Pre × Post × Dopamine)

Key mechanisms:
- Eligibility traces bridge temporal gap between action and reward
- Stability factor prevents catastrophic forgetting
- RPE baseline (avg_reward) prevents dopamine washout

References:
- 3-Factor Hebbian: ΔW = η × (1 - stability) × dopamine × trace
- RPE: δ = R_actual - R_expected
"""

from typing import Dict, List, Set, Optional, Tuple
from .memory import GraphMemory, Synapse, ClusterType


class DopamineModulator:
    """
    Calculates the global neuromodulator signal (Dopamine).
    
    In biological systems, dopamine signals Reward Prediction Error (RPE):
    - Positive RPE: Better than expected → Strengthen behavior
    - Negative RPE: Worse than expected → Weaken behavior
    - Zero RPE: As expected → No change (stability)
    
    DOPAMINE WASHOUT PREVENTION (explicit comment as required):
    --------------------------------------------------------
    The `avg_reward` tracks a moving average of recent rewards.
    When calculating RPE, we subtract this baseline:
    
        effective_reward = current_reward - avg_reward
        
    This ensures that in constant-reward scenarios (e.g., surviving
    in Snake), the RPE approaches zero, preventing runaway weight
    growth. The system only learns when outcomes DIFFER from the
    baseline, not when they are constantly "good" or "bad."
    
    Without this mechanism, an agent receiving +0.1 every tick would
    continuously strengthen all traced synapses, leading to weight
    saturation and loss of discriminative ability (washout).
    """
    
    # Moving average decay rate
    # Higher = slower adaptation to reward changes
    BASELINE_ALPHA = 0.01
    
    def __init__(self, baseline_alpha: float = BASELINE_ALPHA):
        self.baseline_alpha = baseline_alpha
        
        # Moving average of rewards (the baseline expectation)
        self.avg_reward: float = 0.0
        
        # Most recent RPE signal
        self.rpe: float = 0.0
        
        # History for debugging
        self.reward_history: List[float] = []
        self.rpe_history: List[float] = []
        self.max_history = 100
    
    def calculate_signal(
        self, 
        current_reward: float, 
        predicted_value: float = 0.0
    ) -> float:
        """
        Calculate the dopamine signal (Reward Prediction Error).
        
        Args:
            current_reward: Actual reward received this tick
            predicted_value: Value estimate of current state (for TD learning)
        
        Returns:
            RPE signal that modulates learning
        
        DOPAMINE WASHOUT PREVENTION:
        The baseline subtraction ensures that constant rewards
        eventually produce zero RPE, preventing saturation.
        """
        # 1. Update the baseline (slow exponential moving average)
        self.avg_reward = (
            (1 - self.baseline_alpha) * self.avg_reward + 
            self.baseline_alpha * current_reward
        )
        
        # 2. Calculate effective reward (baseline-corrected)
        # If reward is constant, effective_reward → 0 over time
        effective_reward = current_reward - self.avg_reward
        
        # 3. Calculate RPE (difference from prediction)
        # For simple tasks, predicted_value can be 0
        # For TD learning, predicted_value = V(s_t)
        self.rpe = effective_reward - predicted_value
        
        # 4. Record history
        self.reward_history.append(current_reward)
        self.rpe_history.append(self.rpe)
        if len(self.reward_history) > self.max_history:
            self.reward_history = self.reward_history[-self.max_history:]
            self.rpe_history = self.rpe_history[-self.max_history:]
        
        return self.rpe
    
    def reset_baseline(self):
        """Reset the reward baseline (e.g., when switching tasks)."""
        self.avg_reward = 0.0
    
    def get_stats(self) -> Dict:
        """Get modulator statistics."""
        return {
            "avg_reward": self.avg_reward,
            "last_rpe": self.rpe,
            "recent_rewards": self.reward_history[-10:] if self.reward_history else [],
            "recent_rpe": self.rpe_history[-10:] if self.rpe_history else []
        }


class ThreeFactorLearner:
    """
    3-Factor Hebbian Learning with eligibility traces.
    
    Learning Rule:
        ΔW = η × (1 - stability) × dopamine × trace
    
    Where:
    - η: Base learning rate
    - stability: Consolidation factor (protects old knowledge)
    - dopamine: Neuromodulator signal (RPE)
    - trace: Eligibility trace (Pre × Post coincidence memory)
    
    CATASTROPHIC FORGETTING PREVENTION (explicit comment as required):
    ---------------------------------------------------------------
    The `stability` attribute on each synapse gates the effective
    learning rate. New synapses have stability ≈ 0 (fully plastic),
    while repeatedly reinforced synapses accumulate stability → 1.
    
    Effective learning rate = η × (1 - stability)
    
    When stability = 0.8, effective_lr = η × 0.2 (80% protection)
    When stability = 1.0, effective_lr = 0 (fully consolidated)
    
    This ensures that:
    1. Core game knowledge (e.g., "walls hurt") becomes rigid
    2. New tasks can learn without overwriting core knowledge
    3. Very old unused knowledge slowly decays (stability_decay)
    
    Without this mechanism, learning Ping Pong would corrupt
    the Snake-playing weights, requiring full retraining.
    """
    
    # Learning parameters
    DEFAULT_LR = 0.1               # Base learning rate
    DEFAULT_STABILITY_INC = 0.01   # Stability increase on positive RPE
    DEFAULT_STABILITY_DECAY = 0.9999  # Very slow stability decay
    
    # Weight bounds
    W_MIN = 0.01
    W_MAX = 1.0
    
    def __init__(
        self,
        memory: GraphMemory,
        learning_rate: float = DEFAULT_LR,
        stability_inc: float = DEFAULT_STABILITY_INC,
        stability_decay: float = DEFAULT_STABILITY_DECAY
    ):
        self.memory = memory
        self.lr = learning_rate
        self.stability_inc = stability_inc
        self.stability_decay = stability_decay
        
        # Statistics
        self.total_updates = 0
        self.positive_updates = 0
        self.negative_updates = 0
    
    def apply_reward(self, dopamine: float):
        """
        Apply reward signal to all eligible synapses.
        
        This is the core 3-Factor learning step. Called after
        the DopamineModulator produces an RPE signal.
        
        Args:
            dopamine: The RPE signal from DopamineModulator
        
        CATASTROPHIC FORGETTING PREVENTION:
        The (1 - stability) term ensures that consolidated
        synapses are protected from large weight changes.
        """
        if abs(dopamine) < 0.001:
            return  # No signal, no learning
        
        # Get all synapses with active eligibility traces
        traced_synapses = self.memory.get_traced_synapses()
        
        for source_id, synapse in traced_synapses:
            if synapse.trace < 0.01:
                continue
            
            # 3-Factor Hebbian Rule:
            # ΔW = η × (1 - stability) × dopamine × trace
            plasticity = 1.0 - synapse.stability
            delta_w = self.lr * plasticity * dopamine * synapse.trace
            
            # Apply weight change
            old_weight = synapse.weight
            synapse.weight = max(self.W_MIN, min(self.W_MAX, synapse.weight + delta_w))
            
            # Update stability based on reward direction
            if dopamine > 0:
                # Positive RPE: This synapse contributed to success
                # Increase stability (consolidate)
                synapse.stability = min(1.0, synapse.stability + self.stability_inc * synapse.trace)
                self.positive_updates += 1
            else:
                # Negative RPE: This synapse contributed to failure
                # Don't change stability (allow correction)
                self.negative_updates += 1
            
            # Update confidence based on usage
            synapse.confidence = min(1.0, synapse.confidence + abs(delta_w) * 0.1)
            
            self.total_updates += 1
    
    def decay_stability(self):
        """
        Very slowly decay all stability values.
        
        This allows old, unused knowledge to eventually become
        plastic again (neuroplasticity over long timescales).
        Should be called occasionally, not every tick.
        """
        for node_id in self.memory.nodes:
            for synapse in self.memory.get_outgoing(node_id):
                synapse.stability *= self.stability_decay
    
    def get_stats(self) -> Dict:
        """Get learner statistics."""
        return {
            "total_updates": self.total_updates,
            "positive_updates": self.positive_updates,
            "negative_updates": self.negative_updates,
            "traced_synapses": self.memory.traced_count
        }


class EpisodicBuffer:
    """
    Buffer for episodic (Monte Carlo) reward distribution.
    
    For games like Snake where rewards are sparse and delayed,
    we accumulate state-action pairs during an episode and
    apply rewards retroactively at episode end.
    
    This is simpler than TD(λ) and works well for short episodes.
    """
    
    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        
        # Episode storage
        self.episode_traces: List[List[Tuple[str, str]]] = []  # Per-step traced synapses
        self.episode_rewards: List[float] = []
        
        # Current step
        self.current_traces: List[Tuple[str, str]] = []
    
    def record_step(self, memory: GraphMemory):
        """Record which synapses are traced at this step."""
        self.current_traces = [
            (src, syn.target_id) 
            for src, syn in memory.get_traced_synapses()
        ]
        self.episode_traces.append(self.current_traces)
    
    def record_reward(self, reward: float):
        """Record reward for this step."""
        self.episode_rewards.append(reward)
    
    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns for each step.
        
        G_t = R_t + γR_{t+1} + γ²R_{t+2} + ...
        """
        returns = []
        G = 0.0
        
        # Work backwards
        for r in reversed(self.episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def clear(self):
        """Clear episode buffer for new episode."""
        self.episode_traces.clear()
        self.episode_rewards.clear()
        self.current_traces.clear()
    
    def get_episode_length(self) -> int:
        """Get number of steps in current episode."""
        return len(self.episode_traces)


def apply_episodic_learning(
    memory: GraphMemory,
    buffer: EpisodicBuffer,
    learner: ThreeFactorLearner,
    modulator: DopamineModulator,
    gamma: float = 0.99
):
    """
    Apply learning at end of episode using Monte Carlo returns.
    
    For each step in the episode:
    1. Set synapse traces based on recorded activity
    2. Compute RPE from return
    3. Apply 3-Factor learning
    
    Args:
        memory: The graph memory
        buffer: Episodic buffer with recorded traces
        learner: The 3-Factor learner
        modulator: Dopamine modulator
        gamma: Discount factor for returns
    """
    returns = buffer.compute_returns(gamma)
    
    if not returns:
        return
    
    # Average return for normalization
    avg_return = sum(returns) / len(returns)
    
    for step_idx, (traces, G) in enumerate(zip(buffer.episode_traces, returns)):
        # Set traces for this step's synapses
        for src_id, tgt_id in traces:
            synapse = memory.get_synapse(src_id, tgt_id)
            if synapse:
                synapse.trace = 1.0
                memory.mark_synapse_traced(src_id, tgt_id)
        
        # Compute RPE (reward relative to episode average)
        dopamine = modulator.calculate_signal(G, avg_return)
        
        # Apply learning
        learner.apply_reward(dopamine)
        
        # Clear traces for next step
        memory.decay_all_traces(0.0)  # Immediate clear
    
    buffer.clear()
