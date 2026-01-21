"""
NCGN v6.0 Planner Module - System 2 Time-Travel Simulator

Implements deliberative planning via forward simulation:
- SimulationSandbox: Isolated copy of System 1 for "what-if" scenarios
- EpisodicMonteCarlo: Plan by simulating futures and evaluating valence

This is the "slow" thinking system that activates on high surprise.
It cannot directly produce actions - it influences System 1 by
modifying energy/weights before resuming reactive processing.

Key design decisions:
1. Sandbox uses COPY of memory to prevent contamination
2. Forward simulation uses same physics engine (consistency)
3. Action selection based on predicted cumulative valence
"""

import copy
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class PlanResult:
    """Result of a planning episode."""
    best_action: Optional[str]
    action_values: Dict[str, float]
    simulation_ticks: int
    surprise_trigger: float


class SimulationSandbox:
    """
    Isolated copy of System 1 for forward simulation.
    
    The sandbox allows System 2 to "imagine" future scenarios
    without affecting the real graph memory. This is critical
    for model-based planning.
    
    Usage:
        sandbox = SimulationSandbox.from_system1(real_system1)
        sandbox.inject_action("action_move_right")
        sandbox.simulate(steps=10)
        predicted_valence = sandbox.evaluate()
    """
    
    def __init__(self, memory_copy, system1_class, system1_params: dict):
        """
        Create a sandbox from copied components.
        
        Args:
            memory_copy: Deep copy of GraphMemory
            system1_class: The System1Engine class
            system1_params: Parameters for System 1 initialization
        """
        self.memory = memory_copy
        self.engine = system1_class(self.memory, **system1_params)
        
        # Valence accumulator
        self.cumulative_valence: float = 0.0
        self.step_count: int = 0
        
        # Valence estimator function (set externally)
        self.valence_estimator: Optional[Callable[[], float]] = None
    
    @classmethod
    def from_system1(cls, system1_engine) -> 'SimulationSandbox':
        """
        Create a sandbox by deep-copying an existing System 1.
        
        This is the primary factory method for creating sandboxes.
        """
        # Deep copy the memory
        memory_copy = copy.deepcopy(system1_engine.memory)
        
        # Extract parameters from the source engine
        params = {
            'decay_alpha': system1_engine.decay_alpha,
            'refractory_period': system1_engine.refractory_period,
            'trace_decay': system1_engine.trace_decay,
            'novelty_decay': system1_engine.novelty_decay,
            'base_temperature': system1_engine.base_temperature,
            'energy_cap': system1_engine.energy_cap
        }
        
        return cls(memory_copy, type(system1_engine), params)
    
    def inject_action(self, action_node_id: str, energy: float = 1.0):
        """Inject an action into the sandbox for simulation."""
        from .memory import ClusterType
        self.engine.inject_energy(action_node_id, energy, ClusterType.MOTOR)
    
    def inject_sensory(self, sensory_data: Dict[str, float]):
        """Inject sensory state into the sandbox."""
        from .memory import ClusterType
        for node_id, energy in sensory_data.items():
            self.engine.inject_energy(node_id, energy, ClusterType.SENSORY)
    
    def simulate(self, steps: int) -> float:
        """
        Run forward simulation for N steps.
        
        Args:
            steps: Number of ticks to simulate
        
        Returns:
            Cumulative valence over simulated steps
        """
        self.cumulative_valence = 0.0
        
        for _ in range(steps):
            self.engine.tick()
            self.step_count += 1
            
            # Estimate valence at this step
            if self.valence_estimator:
                step_valence = self.valence_estimator()
                self.cumulative_valence += step_valence
        
        return self.cumulative_valence
    
    def get_predicted_state(self) -> Dict[str, float]:
        """Get the predicted energy state after simulation."""
        return {
            node_id: node.energy
            for node_id, node in self.memory.nodes.items()
            if node.energy > 0.01
        }


class EpisodicMonteCarlo:
    """
    Episodic Monte Carlo Planner for deliberative action selection.
    
    When surprise exceeds threshold, this planner:
    1. Pauses System 1
    2. For each possible action:
       a. Creates a simulation sandbox
       b. Injects the action
       c. Runs forward simulation for N ticks
       d. Evaluates predicted cumulative valence
    3. Selects the action with highest predicted valence
    4. Injects that action into real System 1
    5. Resumes System 1
    
    COMPUTATIONAL COST NOTE (explicit comment as required):
    ------------------------------------------------------
    Forward simulation is expensive: O(actions × sim_steps × tick_cost).
    To mitigate this:
    - Only invoke on surprise > threshold (sparse activation)
    - Use small sim_steps (5-10 typically sufficient)
    - Cache sandbox where possible
    - Consider hierarchical planning for complex domains
    """
    
    # Planning parameters
    DEFAULT_SIM_STEPS = 8
    DEFAULT_SURPRISE_THRESHOLD = 0.7
    
    def __init__(
        self,
        sim_steps: int = DEFAULT_SIM_STEPS,
        surprise_threshold: float = DEFAULT_SURPRISE_THRESHOLD
    ):
        self.sim_steps = sim_steps
        self.surprise_threshold = surprise_threshold
        
        # Valence estimator (must be set by the game interface)
        self.valence_estimator: Optional[Callable[[], float]] = None
        
        # Statistics
        self.plan_count: int = 0
        self.total_sim_steps: int = 0
    
    def should_plan(self, surprise_level: float) -> bool:
        """Check if surprise triggers deliberative planning."""
        return surprise_level > self.surprise_threshold
    
    def plan(
        self,
        system1_engine,
        possible_actions: List[str],
        current_sensory: Dict[str, float]
    ) -> PlanResult:
        """
        Perform deliberative planning via forward simulation.
        
        Args:
            system1_engine: The real System 1 engine
            possible_actions: List of action node IDs to evaluate
            current_sensory: Current sensory state to maintain
        
        Returns:
            PlanResult with best action and value estimates
        """
        if not possible_actions:
            return PlanResult(
                best_action=None,
                action_values={},
                simulation_ticks=0,
                surprise_trigger=0.0
            )
        
        action_values: Dict[str, float] = {}
        total_sim_ticks = 0
        
        for action in possible_actions:
            # Create fresh sandbox for each action
            sandbox = SimulationSandbox.from_system1(system1_engine)
            sandbox.valence_estimator = self.valence_estimator
            
            # Inject current sensory state
            sandbox.inject_sensory(current_sensory)
            
            # Inject the action to evaluate
            sandbox.inject_action(action)
            
            # Run forward simulation
            predicted_valence = sandbox.simulate(self.sim_steps)
            action_values[action] = predicted_valence
            total_sim_ticks += self.sim_steps
        
        # Select best action
        best_action = max(action_values, key=action_values.get)
        
        # Update statistics
        self.plan_count += 1
        self.total_sim_steps += total_sim_ticks
        
        return PlanResult(
            best_action=best_action,
            action_values=action_values,
            simulation_ticks=total_sim_ticks,
            surprise_trigger=system1_engine.surprise_level
        )
    
    def get_stats(self) -> Dict:
        """Get planner statistics."""
        return {
            "plan_count": self.plan_count,
            "total_sim_steps": self.total_sim_steps,
            "avg_sims_per_plan": (
                self.total_sim_steps / self.plan_count 
                if self.plan_count > 0 else 0
            )
        }


class System2Controller:
    """
    The deliberative control system integrating planning with System 1.
    
    This is the high-level interface that:
    1. Monitors surprise from System 1
    2. Triggers planning when threshold exceeded
    3. Applies planning results to System 1
    4. Manages the pause/resume cycle
    
    Unlike v5.0's JSON schema checker, this System 2 is a genuine
    model-based planner that simulates futures to select actions.
    """
    
    def __init__(
        self,
        planner: EpisodicMonteCarlo,
        possible_actions: List[str]
    ):
        self.planner = planner
        self.possible_actions = possible_actions
        
        # Current sensory state (set by game interface)
        self.current_sensory: Dict[str, float] = {}
        
        # Intervention history
        self.interventions: List[PlanResult] = []
    
    def check_and_intervene(self, system1_engine) -> Optional[PlanResult]:
        """
        Check if intervention needed and perform planning if so.
        
        Returns:
            PlanResult if planning occurred, None otherwise
        """
        if not self.planner.should_plan(system1_engine.surprise_level):
            return None
        
        # Pause System 1
        system1_engine.pause()
        
        # Perform planning
        result = self.planner.plan(
            system1_engine,
            self.possible_actions,
            self.current_sensory
        )
        
        # Apply best action by injecting energy
        if result.best_action:
            from .memory import ClusterType
            system1_engine.inject_energy(
                result.best_action, 
                1.0, 
                ClusterType.MOTOR
            )
        
        # Resume System 1
        system1_engine.resume()
        
        # Record intervention
        self.interventions.append(result)
        
        return result
    
    def update_sensory(self, sensory_data: Dict[str, float]):
        """Update current sensory state for planning."""
        self.current_sensory = sensory_data
    
    def update_actions(self, actions: List[str]):
        """Update the set of possible actions."""
        self.possible_actions = actions
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            "interventions": len(self.interventions),
            "last_result": (
                self.interventions[-1] if self.interventions else None
            ),
            "planner_stats": self.planner.get_stats()
        }
