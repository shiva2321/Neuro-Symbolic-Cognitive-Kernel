"""
NCGN Training Framework - Learning and Retention

Implements structured training for the cognitive network:
- Curriculum: Ordered sequence of lessons with difficulty progression
- TrainingSession: Iterative learning with STDP and retention testing
- Hebbian Learning: Spike-Timing-Dependent Plasticity (STDP)

The trainer rigorously trains the system until knowledge is retained.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import json
import os
import time

from .memory import GraphMemory, ConceptNode, Synapse, EventSchema
from .system1 import System1Engine
from .system2 import System2Controller


class LessonType(Enum):
    """Types of training lessons."""
    ASSOCIATION = "association"      # Learn A -> B associations
    PROPERTY = "property"            # Learn object properties
    SCHEMA = "schema"                # Learn action constraints
    REINFORCEMENT = "reinforcement"  # Strengthen existing knowledge


@dataclass
class TrainingExample:
    """A single training example."""
    input_nodes: List[Tuple[str, float]]  # (node_id, energy) pairs to inject
    expected_outputs: List[str]            # Node IDs that should activate
    constraints: Dict[str, bool] = field(default_factory=dict)  # Property constraints
    description: str = ""


@dataclass
class Lesson:
    """A collection of related training examples."""
    id: str
    name: str
    type: LessonType
    examples: List[TrainingExample]
    difficulty: int = 1  # 1-10 scale
    prerequisites: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Lesson':
        """Create lesson from dictionary."""
        examples = [
            TrainingExample(
                input_nodes=[(n['id'], n.get('energy', 1.0)) for n in ex['inputs']],
                expected_outputs=ex['expected'],
                constraints=ex.get('constraints', {}),
                description=ex.get('description', '')
            )
            for ex in data.get('examples', [])
        ]
        return cls(
            id=data['id'],
            name=data['name'],
            type=LessonType(data.get('type', 'association')),
            examples=examples,
            difficulty=data.get('difficulty', 1),
            prerequisites=data.get('prerequisites', [])
        )


@dataclass
class TrainingMetrics:
    """Metrics from a training session."""
    accuracy: float              # Correct outputs / total examples
    avg_surprise: float          # Average surprise level
    convergence_epoch: int       # Epoch where 95%+ accuracy reached
    total_epochs: int
    examples_trained: int
    synapse_updates: int
    training_time_seconds: float


@dataclass
class RetentionReport:
    """Report on knowledge retention after training."""
    lessons_tested: int
    examples_tested: int
    recall_accuracy: float       # % of learned associations still retrievable
    degraded_synapses: List[Tuple[str, str, float]]  # (source, target, weight_loss)
    stable_knowledge: List[str]  # Lesson IDs with 100% retention


class Curriculum:
    """
    Structured sequence of training lessons.
    
    Manages lesson ordering based on prerequisites and difficulty.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.lessons: Dict[str, Lesson] = {}
        self.completed_lessons: Set[str] = set()
    
    def add_lesson(self, lesson: Lesson):
        """Add a lesson to the curriculum."""
        self.lessons[lesson.id] = lesson
    
    def get_next_lesson(self) -> Optional[Lesson]:
        """Get the next lesson to train (respects prerequisites)."""
        available = []
        for lesson_id, lesson in self.lessons.items():
            if lesson_id in self.completed_lessons:
                continue
            # Check prerequisites
            if all(prereq in self.completed_lessons for prereq in lesson.prerequisites):
                available.append(lesson)
        
        if not available:
            return None
        
        # Return lowest difficulty available
        return min(available, key=lambda l: l.difficulty)
    
    def mark_completed(self, lesson_id: str):
        """Mark a lesson as completed."""
        self.completed_lessons.add(lesson_id)
    
    def get_progress(self) -> Tuple[int, int]:
        """Return (completed, total) lesson counts."""
        return len(self.completed_lessons), len(self.lessons)
    
    @classmethod
    def load_from_directory(cls, dir_path: str, name: str = "default") -> 'Curriculum':
        """Load curriculum from a directory of JSON files."""
        curriculum = cls(name)
        
        if not os.path.exists(dir_path):
            return curriculum
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Handle both single lesson and lesson list formats
                    if isinstance(data, list):
                        for lesson_data in data:
                            lesson = Lesson.from_dict(lesson_data)
                            curriculum.add_lesson(lesson)
                    else:
                        lesson = Lesson.from_dict(data)
                        curriculum.add_lesson(lesson)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
        
        return curriculum


class LearningRule(Enum):
    """Available learning algorithms."""
    STDP = "stdp"                    # Spike-Timing Dependent Plasticity
    BCM = "bcm"                      # Bienenstock-Cooper-Munro (sliding threshold)
    OJA = "oja"                      # Oja's rule (normalized Hebbian)
    ERROR_DRIVEN = "error_driven"    # Supervised error correction
    COMBINED = "combined"            # All rules together


class EligibilityTrace:
    """
    Eligibility trace for credit assignment across time.
    
    Maintains a decaying record of synaptic activity, allowing
    delayed rewards/errors to update the correct synapses.
    """
    
    def __init__(self, decay_rate: float = 0.9, max_history: int = 50):
        self.decay_rate = decay_rate
        self.max_history = max_history
        # (source_id, target_id) -> eligibility value
        self.traces: Dict[Tuple[str, str], float] = {}
        self.history: List[Tuple[int, str, str]] = []  # (tick, source, target)
    
    def record_activity(self, tick: int, source_id: str, target_id: str):
        """Record synaptic activity for eligibility."""
        key = (source_id, target_id)
        self.traces[key] = 1.0  # Full eligibility
        self.history.append((tick, source_id, target_id))
        
        # Trim old history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_eligibility(self, source_id: str, target_id: str) -> float:
        """Get current eligibility for a synapse."""
        return self.traces.get((source_id, target_id), 0.0)
    
    def decay_all(self):
        """Decay all eligibility traces."""
        expired = []
        for key, value in self.traces.items():
            new_value = value * self.decay_rate
            if new_value < 0.01:
                expired.append(key)
            else:
                self.traces[key] = new_value
        
        for key in expired:
            del self.traces[key]
    
    def get_eligible_synapses(self, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
        """Get all synapses with eligibility above threshold."""
        return [
            (src, tgt, elig) 
            for (src, tgt), elig in self.traces.items() 
            if elig >= threshold
        ]


class AdvancedLearner:
    """
    Comprehensive learning system with multiple algorithms.
    
    Implements:
    1. STDP: Spike-timing dependent plasticity (timing-based)
    2. BCM: Bienenstock-Cooper-Munro (activity-dependent threshold)
    3. Oja's Rule: Normalized Hebbian (prevents weight explosion)
    4. Error-Driven: Supervised learning with eligibility traces
    
    The combined mode uses all rules weighted by their applicability.
    """
    
    # STDP Parameters
    A_PLUS = 0.1        # LTP magnitude
    A_MINUS = 0.05      # LTD magnitude
    TAU_PLUS = 20       # LTP time constant (ticks)
    TAU_MINUS = 20      # LTD time constant (ticks)
    
    # BCM Parameters (sliding threshold)
    BCM_TAU_M = 100     # Activity averaging time constant
    BCM_TARGET_RATE = 0.3  # Target average activity
    
    # Oja's Rule Parameters
    OJA_ALPHA = 0.01    # Learning rate for Oja's rule
    
    # Error-Driven Parameters
    ERROR_LR = 0.15     # Learning rate for error correction
    REWARD_LR = 0.1     # Learning rate for reward signals
    
    # General Bounds
    W_MAX = 1.0         # Maximum weight
    W_MIN = 0.01        # Minimum weight
    
    def __init__(
        self, 
        memory: GraphMemory, 
        rule: LearningRule = LearningRule.COMBINED
    ):
        self.memory = memory
        self.rule = rule
        self.updates_count = 0
        
        # BCM sliding threshold per node
        self.activity_avg: Dict[str, float] = {}
        self.bcm_threshold: Dict[str, float] = {}
        
        # Eligibility traces for credit assignment
        self.eligibility = EligibilityTrace()
        
        # Learning rate modulation based on surprise/novelty
        self.current_learning_rate = 1.0
        
    def set_learning_rate(self, rate: float):
        """Modulate learning rate based on context (e.g., high surprise = faster learning)."""
        self.current_learning_rate = max(0.1, min(2.0, rate))
    
    def apply_stdp(
        self,
        pre_node_id: str,
        post_node_id: str,
        delta_t: int
    ) -> float:
        """
        Apply STDP learning rule.
        
        Pre-before-post: strengthen (LTP)
        Post-before-pre: weaken (LTD)
        """
        synapse = self.memory.get_synapse(pre_node_id, post_node_id)
        if not synapse:
            return 0.0
        
        import math
        
        if delta_t > 0:
            dw = self.A_PLUS * math.exp(-delta_t / self.TAU_PLUS)
        else:
            dw = -self.A_MINUS * math.exp(delta_t / self.TAU_MINUS)
        
        dw *= self.current_learning_rate
        old_weight = synapse.weight
        synapse.weight = max(self.W_MIN, min(self.W_MAX, synapse.weight + dw))
        
        if dw > 0:
            synapse.confidence = min(1.0, synapse.confidence + dw * 0.5)
        
        self.updates_count += 1
        return synapse.weight - old_weight
    
    def apply_bcm(
        self,
        pre_node_id: str,
        post_node_id: str,
        pre_activity: float,
        post_activity: float
    ) -> float:
        """
        Apply BCM learning rule with sliding threshold.
        
        The threshold θ slides based on postsynaptic activity history:
        - If post_activity > θ: strengthen (LTP)
        - If post_activity < θ: weaken (LTD)
        
        This prevents runaway excitation and ensures competition.
        """
        synapse = self.memory.get_synapse(pre_node_id, post_node_id)
        if not synapse:
            return 0.0
        
        # Update activity average for BCM threshold
        old_avg = self.activity_avg.get(post_node_id, self.BCM_TARGET_RATE)
        new_avg = old_avg + (post_activity - old_avg) / self.BCM_TAU_M
        self.activity_avg[post_node_id] = new_avg
        
        # BCM threshold is quadratic function of average activity
        theta = new_avg ** 2 / self.BCM_TARGET_RATE
        self.bcm_threshold[post_node_id] = theta
        
        # BCM learning rule: Δw = η * pre * post * (post - θ)
        dw = 0.01 * pre_activity * post_activity * (post_activity - theta)
        dw *= self.current_learning_rate
        
        old_weight = synapse.weight
        synapse.weight = max(self.W_MIN, min(self.W_MAX, synapse.weight + dw))
        
        self.updates_count += 1
        return synapse.weight - old_weight
    
    def apply_oja(
        self,
        pre_node_id: str,
        post_node_id: str,
        pre_activity: float,
        post_activity: float
    ) -> float:
        """
        Apply Oja's normalized Hebbian rule.
        
        Δw = α * y * (x - y*w)
        
        This keeps weights bounded and normalized, preventing explosion.
        Useful for feature extraction and competition.
        """
        synapse = self.memory.get_synapse(pre_node_id, post_node_id)
        if not synapse:
            return 0.0
        
        # Oja's rule: decays weight by output squared
        dw = self.OJA_ALPHA * post_activity * (
            pre_activity - post_activity * synapse.weight
        )
        dw *= self.current_learning_rate
        
        old_weight = synapse.weight
        synapse.weight = max(self.W_MIN, min(self.W_MAX, synapse.weight + dw))
        
        self.updates_count += 1
        return synapse.weight - old_weight
    
    def apply_error_driven(
        self,
        error: float,
        expected_outputs: List[str],
        actual_outputs: Set[str]
    ) -> int:
        """
        Apply error-driven learning with eligibility traces.
        
        Uses the eligibility trace to identify which synapses
        contributed to the error and should be updated.
        """
        updates = 0
        
        # Get eligible synapses
        eligible = self.eligibility.get_eligible_synapses(threshold=0.1)
        
        for source_id, target_id, eligibility in eligible:
            synapse = self.memory.get_synapse(source_id, target_id)
            if not synapse:
                continue
            
            # Determine if this synapse contributed to error
            if target_id in expected_outputs and target_id not in actual_outputs:
                # False negative: should have activated but didn't
                dw = self.ERROR_LR * eligibility * abs(error)
            elif target_id in actual_outputs and target_id not in expected_outputs:
                # False positive: activated but shouldn't have
                dw = -self.ERROR_LR * eligibility * abs(error)
            else:
                # Correct prediction: small positive reinforcement
                dw = self.REWARD_LR * eligibility * (1.0 - abs(error))
            
            dw *= self.current_learning_rate
            synapse.weight = max(self.W_MIN, min(self.W_MAX, synapse.weight + dw))
            synapse.confidence = min(1.0, synapse.confidence + abs(dw) * 0.3)
            updates += 1
        
        self.updates_count += updates
        return updates
    
    def record_activity(self, tick: int, source_id: str, target_id: str):
        """Record synaptic activity for eligibility traces."""
        self.eligibility.record_activity(tick, source_id, target_id)
    
    def decay_traces(self):
        """Decay eligibility traces (call each tick)."""
        self.eligibility.decay_all()
    
    def apply_combined_learning(
        self,
        pre_node_id: str,
        post_node_id: str,
        pre_activity: float,
        post_activity: float,
        delta_t: Optional[int] = None
    ) -> float:
        """
        Apply all applicable learning rules and combine their effects.
        
        The rules are weighted based on their inputs:
        - STDP: requires timing info (delta_t)
        - BCM: requires activity levels
        - Oja's: requires activity levels
        """
        total_dw = 0.0
        contributions = 0
        
        # STDP (if timing info available)
        if delta_t is not None:
            dw = self.apply_stdp(pre_node_id, post_node_id, delta_t)
            total_dw += dw * 0.4  # 40% weight
            contributions += 1
        
        # BCM (sliding threshold)
        dw = self.apply_bcm(pre_node_id, post_node_id, pre_activity, post_activity)
        total_dw += dw * 0.3  # 30% weight
        contributions += 1
        
        # Oja's (normalization)
        dw = self.apply_oja(pre_node_id, post_node_id, pre_activity, post_activity)
        total_dw += dw * 0.3  # 30% weight
        contributions += 1
        
        return total_dw
    
    def strengthen_association(
        self,
        source_id: str,
        target_id: str,
        amount: float = 0.1
    ):
        """Directly strengthen an association (supervised learning signal)."""
        synapse = self.memory.get_synapse(source_id, target_id)
        if synapse:
            amount *= self.current_learning_rate
            synapse.weight = min(self.W_MAX, synapse.weight + amount)
            synapse.confidence = min(1.0, synapse.confidence + amount * 0.5)
            self.updates_count += 1
    
    def weaken_association(
        self,
        source_id: str,
        target_id: str,
        amount: float = 0.05
    ):
        """Directly weaken an association (error correction)."""
        synapse = self.memory.get_synapse(source_id, target_id)
        if synapse:
            amount *= self.current_learning_rate
            synapse.weight = max(self.W_MIN, synapse.weight - amount)
            self.updates_count += 1
    
    def get_stats(self) -> Dict[str, any]:
        """Get learning statistics."""
        return {
            "total_updates": self.updates_count,
            "active_traces": len(self.eligibility.traces),
            "learning_rate": self.current_learning_rate,
            "bcm_thresholds": len(self.bcm_threshold)
        }


# Backwards compatible alias
STDPLearner = AdvancedLearner


class TrainingSession:
    """
    Manages iterative training with learning and retention testing.
    
    Trains the system rigorously until knowledge is retained.
    """
    
    # Training parameters
    DEFAULT_EPOCHS = 50
    DEFAULT_TICKS_PER_EXAMPLE = 5
    RETENTION_THRESHOLD = 0.95  # 95% accuracy for "learned"
    DECAY_TICKS_BETWEEN_TESTS = 100  # Simulate forgetting
    
    def __init__(
        self,
        memory: GraphMemory,
        engine: System1Engine,
        controller: System2Controller
    ):
        self.memory = memory
        self.engine = engine
        self.controller = controller
        self.stdp = STDPLearner(memory)
        
        # State tracking
        self.current_epoch = 0
        self.metrics_history: List[TrainingMetrics] = []
        self.on_progress: Optional[Callable[[int, float], None]] = None
    
    def setup_knowledge_base(self):
        """Initialize the knowledge base with basic concepts."""
        # Create standard goal/action nodes
        goal_nodes = [
            "goal_clarify_inconsistency",
            "goal_acquire_knowledge", 
            "action_query_user",
            "action_ask"
        ]
        for node_id in goal_nodes:
            if not self.memory.has_node(node_id):
                self.memory.add_node(node_id, threshold=0.5)
    
    def train_lesson(
        self,
        lesson: Lesson,
        epochs: int = DEFAULT_EPOCHS,
        verbose: bool = False
    ) -> TrainingMetrics:
        """
        Train a single lesson until convergence or max epochs.
        
        Returns training metrics.
        """
        start_time = time.time()
        correct_count = 0
        total_count = 0
        surprise_sum = 0.0
        convergence_epoch = epochs
        converged = False
        
        for epoch in range(epochs):
            epoch_correct = 0
            epoch_total = 0
            
            for example in lesson.examples:
                success = self._train_example(example, verbose and epoch == 0)
                if success:
                    epoch_correct += 1
                epoch_total += 1
                surprise_sum += self.engine.surprise_level
            
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            
            if verbose:
                print(f"  Epoch {epoch + 1}: Accuracy = {accuracy:.2%}")
            
            if self.on_progress:
                self.on_progress(epoch + 1, accuracy)
            
            # Check for convergence
            if accuracy >= self.RETENTION_THRESHOLD and not converged:
                convergence_epoch = epoch + 1
                converged = True
            
            correct_count += epoch_correct
            total_count += epoch_total
            
            # Early stopping if perfect for 3 epochs
            if converged and epoch >= convergence_epoch + 2:
                break
        
        training_time = time.time() - start_time
        
        return TrainingMetrics(
            accuracy=correct_count / total_count if total_count > 0 else 0.0,
            avg_surprise=surprise_sum / total_count if total_count > 0 else 0.0,
            convergence_epoch=convergence_epoch,
            total_epochs=epochs,
            examples_trained=total_count,
            synapse_updates=self.stdp.updates_count,
            training_time_seconds=training_time
        )
    
    def _train_example(
        self,
        example: TrainingExample,
        verbose: bool = False
    ) -> bool:
        """Train a single example. Returns True if output was correct."""
        # Reset active state
        for node_id in list(self.memory.get_active_nodes()):
            node = self.memory.get_node(node_id)
            if node:
                node.energy = 0.0
            self.memory.mark_inactive(node_id)
        
        # Ensure all nodes exist
        for node_id, _ in example.input_nodes:
            if not self.memory.has_node(node_id):
                self.memory.add_node(node_id, novelty_score=1.0)
        
        for node_id in example.expected_outputs:
            if not self.memory.has_node(node_id):
                self.memory.add_node(node_id, novelty_score=1.0)
        
        # Inject input energy
        for node_id, energy in example.input_nodes:
            self.engine.inject_energy(node_id, energy)
        
        # Run ticks
        for _ in range(self.DEFAULT_TICKS_PER_EXAMPLE):
            self.engine.tick()
        
        # Check outputs
        active_nodes = self.memory.get_active_nodes()
        correct = all(
            node_id in active_nodes or 
            (self.memory.get_node(node_id) and 
             self.memory.get_node(node_id).energy > 0.1)
            for node_id in example.expected_outputs
        )
        
        # Apply learning
        for input_node, _ in example.input_nodes:
            for expected in example.expected_outputs:
                # Ensure synapse exists
                if not self.memory.get_synapse(input_node, expected):
                    self.memory.add_synapse(
                        input_node, expected,
                        type="learned",
                        weight=0.3,
                        confidence=0.3
                    )
                
                if correct:
                    # Reinforce correct associations
                    self.stdp.strengthen_association(input_node, expected, 0.05)
                else:
                    # Still strengthen (supervised signal)
                    self.stdp.strengthen_association(input_node, expected, 0.1)
        
        # Learn properties
        for obj_id, value in example.constraints.items():
            if '.' in obj_id:
                parts = obj_id.split('.')
                self.controller.set_property(parts[0], parts[1], value)
        
        if verbose:
            status = "✓" if correct else "✗"
            print(f"    {status} {example.description or 'Example'}")
        
        return correct
    
    def test_retention(
        self,
        lessons: List[Lesson],
        decay_ticks: int = DECAY_TICKS_BETWEEN_TESTS
    ) -> RetentionReport:
        """
        Test how well knowledge is retained after decay.
        
        Runs decay_ticks to simulate time passing, then tests recall.
        """
        # Simulate time passing
        for _ in range(decay_ticks):
            self.engine.tick()
        
        total_examples = 0
        correct_examples = 0
        degraded = []
        stable_lessons = []
        
        for lesson in lessons:
            lesson_correct = 0
            
            for example in lesson.examples:
                # Reset state
                for node_id in list(self.memory.get_active_nodes()):
                    node = self.memory.get_node(node_id)
                    if node:
                        node.energy = 0.0
                    self.memory.mark_inactive(node_id)
                
                # Inject inputs
                for node_id, energy in example.input_nodes:
                    self.engine.inject_energy(node_id, energy)
                
                # Run ticks
                for _ in range(self.DEFAULT_TICKS_PER_EXAMPLE):
                    self.engine.tick()
                
                # Check outputs
                active_nodes = self.memory.get_active_nodes()
                if all(node_id in active_nodes for node_id in example.expected_outputs):
                    lesson_correct += 1
                    correct_examples += 1
                else:
                    # Track degraded synapses
                    for input_node, _ in example.input_nodes:
                        for expected in example.expected_outputs:
                            synapse = self.memory.get_synapse(input_node, expected)
                            if synapse and synapse.weight < 0.5:
                                degraded.append((input_node, expected, synapse.weight))
                
                total_examples += 1
            
            if lesson_correct == len(lesson.examples):
                stable_lessons.append(lesson.id)
        
        return RetentionReport(
            lessons_tested=len(lessons),
            examples_tested=total_examples,
            recall_accuracy=correct_examples / total_examples if total_examples > 0 else 0.0,
            degraded_synapses=degraded,
            stable_knowledge=stable_lessons
        )
    
    def train_until_mastery(
        self,
        curriculum: Curriculum,
        max_total_epochs: int = 500,
        verbose: bool = True
    ) -> List[TrainingMetrics]:
        """
        Train all lessons in curriculum until mastery.
        
        Continues training each lesson until retention threshold is met,
        then periodically re-tests earlier lessons.
        """
        all_metrics = []
        total_epochs = 0
        completed_lessons: List[Lesson] = []
        
        while total_epochs < max_total_epochs:
            lesson = curriculum.get_next_lesson()
            
            if lesson is None:
                # All lessons complete - test retention
                if completed_lessons:
                    if verbose:
                        print("\n--- Testing Knowledge Retention ---")
                    report = self.test_retention(completed_lessons)
                    if verbose:
                        print(f"Retention: {report.recall_accuracy:.1%}")
                    
                    if report.recall_accuracy >= self.RETENTION_THRESHOLD:
                        print("\n✓ Knowledge fully retained!")
                        break
                    else:
                        # Retrain degraded lessons
                        for lesson in completed_lessons:
                            if lesson.id not in report.stable_knowledge:
                                curriculum.completed_lessons.discard(lesson.id)
                else:
                    break
            else:
                if verbose:
                    print(f"\n--- Training: {lesson.name} ---")
                
                metrics = self.train_lesson(lesson, epochs=self.DEFAULT_EPOCHS, verbose=verbose)
                all_metrics.append(metrics)
                total_epochs += metrics.total_epochs
                
                if metrics.accuracy >= self.RETENTION_THRESHOLD:
                    curriculum.mark_completed(lesson.id)
                    completed_lessons.append(lesson)
                    if verbose:
                        print(f"✓ Lesson mastered in {metrics.convergence_epoch} epochs")
                else:
                    if verbose:
                        print(f"✗ Lesson needs more training (accuracy: {metrics.accuracy:.1%})")
        
        return all_metrics


def create_default_curriculum() -> Curriculum:
    """Create a default curriculum for testing."""
    curriculum = Curriculum("default")
    
    # Lesson 1: Basic animal-food associations
    lesson1 = Lesson(
        id="basic_associations",
        name="Basic Animal-Food Associations",
        type=LessonType.ASSOCIATION,
        difficulty=1,
        examples=[
            TrainingExample(
                input_nodes=[("dog", 1.0)],
                expected_outputs=["meat"],
                description="Dogs eat meat"
            ),
            TrainingExample(
                input_nodes=[("cat", 1.0)],
                expected_outputs=["fish"],
                description="Cats eat fish"
            ),
            TrainingExample(
                input_nodes=[("rabbit", 1.0)],
                expected_outputs=["carrot"],
                description="Rabbits eat carrots"
            ),
            TrainingExample(
                input_nodes=[("bird", 1.0)],
                expected_outputs=["seed"],
                description="Birds eat seeds"
            )
        ]
    )
    curriculum.add_lesson(lesson1)
    
    # Lesson 2: Edibility properties
    lesson2 = Lesson(
        id="edibility",
        name="Edibility Properties",
        type=LessonType.PROPERTY,
        difficulty=2,
        prerequisites=["basic_associations"],
        examples=[
            TrainingExample(
                input_nodes=[("meat", 1.0)],
                expected_outputs=["edible"],
                constraints={"meat.is_edible": True},
                description="Meat is edible"
            ),
            TrainingExample(
                input_nodes=[("fish", 1.0)],
                expected_outputs=["edible"],
                constraints={"fish.is_edible": True},
                description="Fish is edible"
            ),
            TrainingExample(
                input_nodes=[("metal", 1.0)],
                expected_outputs=["inedible"],
                constraints={"metal.is_edible": False},
                description="Metal is NOT edible"
            ),
            TrainingExample(
                input_nodes=[("plastic", 1.0)],
                expected_outputs=["inedible"],
                constraints={"plastic.is_edible": False},
                description="Plastic is NOT edible"
            )
        ]
    )
    curriculum.add_lesson(lesson2)
    
    return curriculum
