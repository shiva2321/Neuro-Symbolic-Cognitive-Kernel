
"""
Learning Progress Tracker (Phase 1.4 / 1.5)
===========================================

Implements Learning Progress based on the change in accumulated reward 
or prediction error over time. This is key for:
1. Intelligent Adaptive Curiosity (IAC)
2. Self-Curriculum (Task Switching)
3. Measuring "Competence"

Metric:
    LP(task, t) = MeanReward(task, t) - MeanReward(task, t - window)

Usage:
    tracker = LearningProgressTracker()
    tracker.update("snake", reward=1.0)
    if tracker.is_plateaued("snake"):
        # Switch task
"""

from collections import deque
import numpy as np

class LearningProgressTracker:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.history = {} # {task_name: deque(maxlen=window_size)}
        self.long_history = {} # {task_name: deque(maxlen=window_size*4)}
        
    def update(self, task: str, reward: float):
        if task not in self.history:
            self.history[task] = deque(maxlen=self.window_size)
            self.long_history[task] = deque(maxlen=self.window_size * 4)
            
        self.history[task].append(reward)
        self.long_history[task].append(reward)
        
    def get_progress(self, task: str) -> float:
        """
        Calculate learning progress (derivative of reward).
        Positive = Improving.
        Zero = Plateaued (or mastered/failed).
        """
        if task not in self.history or len(self.history[task]) < self.window_size:
            return 0.0
            
        current_perf = np.mean(self.history[task])
        # Compare to older history
        # (This is a simplification; ideally we split the window)
        # Let's say improvement over the last N steps vs previous N steps
        
        all_hist = list(self.long_history[task])
        if len(all_hist) < self.window_size * 2:
            return 0.0
            
        recent = np.mean(all_hist[-self.window_size:])
        older = np.mean(all_hist[-self.window_size*2 : -self.window_size])
        
        return recent - older

    def is_plateaued(self, task: str, threshold: float = 0.05) -> bool:
        """Returns True if learning progress is minimal."""
        progress = abs(self.get_progress(task))
        return progress < threshold

    def get_competence(self, task: str) -> float:
        """Returns raw competence level (0 to 1 approx, depending on reward scale)."""
        if task not in self.history or len(self.history[task]) == 0:
            return 0.0
        return float(np.mean(self.history[task]))
