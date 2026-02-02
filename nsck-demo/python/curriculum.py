"""
NSCK Curriculum Learning Module (Phase 4.3)
===========================================
Automatically designs learning paths based on agent competence.
Uses Vygotsky's Zone of Proximal Development (ZPD).
"""

import math
import random
import numpy as np
from typing import List, Dict, Any, Optional

class CurriculumDesigner:
    """
    Manages the task library and suggests next steps based on mastery.
    """
    def __init__(self):
        self.tasks = {} # task_id -> {difficulty: 0..1, prereqs: [ids]}
        self.mastery = {} # task_id -> confidence 0..1
        
    def add_task(self, task_id: str, difficulty: float, prerequisites: List[str] = []):
        self.tasks[task_id] = {
            "difficulty": difficulty,
            "prereqs": prerequisites
        }
        self.mastery[task_id] = 0.0

    def update_mastery(self, task_id: str, score: float):
        """Update agent's perceived competence for a task."""
        if task_id in self.mastery:
            # Alpha-blending for smooth updates
            self.mastery[task_id] = 0.8 * self.mastery[task_id] + 0.2 * score

    def select_next_task(self) -> str:
        """
        Identify the optimal next task.
        Criteria:
        1. Prerequisites met (> 0.7 mastery)
        2. Difficulty matches 'current overall competence + offset'
        """
        overall_comp = self._get_overall_competence()
        
        candidates = []
        for tid, info in self.tasks.items():
            # Skip if already mastered
            if self.mastery[tid] > 0.9:
                continue
                
            # Check prereqs
            if all(self.mastery.get(p, 0.0) > 0.7 for p in info['prereqs']):
                # Score based on ZPD (Zone of Proximal Development)
                # Ideal difficulty = competence + 0.1
                dist = abs(info['difficulty'] - (overall_comp + 0.1))
                score = math.exp(- (dist**2) / 0.05) # Gaussian
                candidates.append((tid, score))
                
        if not candidates:
            # Fallback to random unmastered or lowest difficulty
            return random.choice(list(self.tasks.keys()))
            
        # Return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _get_overall_competence(self) -> float:
        if not self.mastery: return 0.0
        return sum(self.mastery.values()) / len(self.mastery)

class SelfGuidedLearner:
    """
    Combines structured curriculum with curiosity-driven exploration.
    """
    def __init__(self, curriculum: CurriculumDesigner, curiosity_module: Any):
        self.curriculum = curriculum
        self.curiosity = curiosity_module # Needs suggest_exploration_target()
        
    def get_next_goal(self, exploration_prob: float = 0.3) -> str:
        """
        Decide what to work on next.
        """
        if random.random() < exploration_prob and self.curiosity:
            print("[CURRICULUM] Curiosity-driven goal selected.")
            return self.curiosity.suggest_exploration_target()
        else:
            print("[CURRICULUM] Curriculum-suggested goal selected.")
            return self.curriculum.select_next_task()

    def report_progress(self, task_id: str, success: bool, reward: float):
        score = 1.0 if success else (0.5 if reward > 0 else 0.0)
        self.curriculum.update_mastery(task_id, score)
        
        # Also notify curiosity about engagement
        if hasattr(self.curiosity, "update_engagement"):
            self.curiosity.update_engagement(task_id, score)
