"""
NSCK Proto-Self (Homeostasis)
=============================
Implements the biological substrate of the agent.
Monitors critical internal variables (Energy, Integrity, Temperature).
Generates 'Drives' (Urgency signals) that bias the Global Workspace.

Theory: Damasio's Proto-Self.
"""

import time
import math
from typing import Dict

class HomeostaticMonitor:
    def __init__(self):
        # --- Internal Variables (Setpoints = 1.0) ---
        self.energy = 1.0      # 1.0 = Full Battery, 0.0 = Dead
        self.integrity = 1.0   # 1.0 = Perfect Health
        self.latency = 0.0     # 0.0 = Fast, 1.0 = Laggy
        
        # --- Parameters ---
        self.decay_rate = 0.001  # Energy loss per tick
        self.recovery_rate = 0.05 # Recharging speed
        
        # --- Drives (Output) ---
        # Values 0.0 (Satisfied) to 1.0 (Critical Urgency)
        self.drives: Dict[str, float] = {
            "hunger": 0.0,    # Need for Energy
            "pain": 0.0,      # Need for Integrity repair
            "anxiety": 0.0    # Need for Latency reduction (simplify task)
        }

    def update(self, tick_duration: float = 1.0):
        """
        Simulate the passage of time.
        Energy decays.
        Drives increase non-linearly as variables deviate from setpoints.
        """
        # 1. Decay Physics
        self.energy = max(0.0, self.energy - (self.decay_rate * tick_duration))
        
        # 2. Calculate Drives (Sigmoid / Exponential activation)
        # Hunger: Low energy -> High Drive
        # f(x) = (1 - x)^2  (Simple quadratic urgency)
        self.drives["hunger"] = (1.0 - self.energy) ** 2
        
        # Pain: Low integrity -> High Drive
        self.drives["pain"] = (1.0 - self.integrity) ** 3
        
        # Anxiety: High Latency -> High Drive
        self.drives["anxiety"] = min(1.0, self.latency * 2)

    def consume(self, resource_type: str, amount: float):
        """Action effect: System inputs"""
        if resource_type == "energy":
            self.energy = min(1.0, self.energy + amount)
        elif resource_type == "integrity":
            self.integrity = min(1.0, self.integrity + amount)
            
    def set_latency(self, latency_ms: float):
        """External sensor for system load"""
        # Normalize: > 100ms is "High Stress"
        self.latency = min(1.0, latency_ms / 100.0)

    def get_dominant_drive(self) -> str:
        """Returns the name of the most urgent drive."""
        return max(self.drives, key=self.drives.get)

    def __repr__(self):
        return f"<ProtoSelf Energy={self.energy:.2f} Drives={self.drives}>"
