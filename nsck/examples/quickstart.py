#!/usr/bin/env python3
"""
Quickstart Example
==================
Demonstrates the full cognitive loop:
  1. Create an engine
  2. Register a domain
  3. Decide → learn → explain

Run from the nsck-demo/ directory:
    python examples/quickstart.py
"""
import sys
from pathlib import Path

# Ensure nsck-demo is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.core.reasoning.cognitive_engine import create_cognitive_engine
from python.core.perception.grounding_verifier import GroundingVerifier
from python.core.reasoning.causal_reasoning import CausalGraph, CausalLink, CausalRelation


# ── 1. Custom verifier ──────────────────────────────────────────
class WeatherVerifier(GroundingVerifier):
    """Extract symbolic predicates from a weather-station reading."""

    def get_active_predicates(self, state, context=""):
        preds = []
        temp = state.get("temperature", 20)
        if temp > 35:
            preds.append("VERY_HOT")
        elif temp > 25:
            preds.append("WARM")
        elif temp < 5:
            preds.append("FREEZING")

        if state.get("rain", False):
            preds.append("RAINING")
        if state.get("wind_speed", 0) > 50:
            preds.append("HIGH_WIND")
        return preds


# ── 2. Bootstrap causal knowledge ───────────────────────────────
weather_graph = CausalGraph()
weather_graph.add_link(CausalLink(
    cause="FREEZING", effect="ICE_RISK",
    relation=CausalRelation.CAUSES, strength=0.9,
))
weather_graph.add_link(CausalLink(
    cause="HIGH_WIND", effect="DAMAGE_RISK",
    relation=CausalRelation.CAUSES, strength=0.8,
))
weather_graph.add_link(CausalLink(
    cause="RAINING", effect="FLOOD_RISK",
    relation=CausalRelation.CAUSES, strength=0.6,
))


def main():
    # ── 3. Create engine & register domain ──────────────────────
    engine = create_cognitive_engine()
    engine.register_task(
        "weather",
        verifier=WeatherVerifier(),
        causal_graph=weather_graph,
    )

    # ── 4. Simulate a few cycles ────────────────────────────────
    readings = [
        {"temperature": 38, "rain": False, "wind_speed": 10},
        {"temperature": 2, "rain": True, "wind_speed": 60},
        {"temperature": 22, "rain": False, "wind_speed": 5},
    ]

    for i, state in enumerate(readings, 1):
        print(f"\n{'='*50}")
        print(f"Cycle {i}: state = {state}")

        result = engine.decide(state, task_tag="weather")
        print(f"  Action  : {result.chosen_action}")
        print(f"  Mode    : {result.trace.get('mode', '?')}")
        print(f"  Conf    : {result.confidence:.2f}")
        print(f"  Explore : {result.exploration_mode}")

        # Simulate reward
        reward = 1.0 if result.chosen_action != "ACTION_STAY" else -0.1
        engine.learn(state, result.chosen_action, reward, task_tag="weather")

    # ── 5. Explain the last decision ────────────────────────────
    print(f"\n{'='*50}")
    print("Explanation:", engine.explain())

    # ── 6. Stats ────────────────────────────────────────────────
    print(f"\nStats: {engine.get_stats()}")


if __name__ == "__main__":
    main()
