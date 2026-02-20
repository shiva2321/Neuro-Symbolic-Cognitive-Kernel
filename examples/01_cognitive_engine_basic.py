"""
examples/01_cognitive_engine_basic.py
======================================
Minimal example: register a domain, feed an observation, and get a decision
with a human-readable explanation from the CognitiveEngine.

Run from the repository root:
    python examples/01_cognitive_engine_basic.py
"""

import sys
import os

# Add the nsck package to the path when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nsck'))

from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.integration.config import NSCKConfig

# ------------------------------------------------------------------
# 1. Create the engine
# ------------------------------------------------------------------
engine = CognitiveEngine(NSCKConfig())

# ------------------------------------------------------------------
# 2. Register a domain (task_tag identifies the domain)
# ------------------------------------------------------------------
engine.register_task(task_tag="navigation")

# ------------------------------------------------------------------
# 3. Make decisions for different world states
#    decide(state_dict, task_tag) -> CognitiveState
# ------------------------------------------------------------------
states = [
    {"obstacle_ahead": True,  "position_x": 3, "position_y": 5},
    {"obstacle_ahead": False, "position_x": 3, "position_y": 5},
]

for state in states:
    result = engine.decide(state, "navigation")
    print(f"State     : {state}")
    print(f"Action    : {result.chosen_action}")
    print(f"Confidence: {result.confidence:.2f}")
    if result.explanation:
        print(f"Reason    : {result.explanation.summary}")
    print()
