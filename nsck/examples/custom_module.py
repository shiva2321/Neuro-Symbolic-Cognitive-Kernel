#!/usr/bin/env python3
"""
Custom Module Example
=====================
Shows how to create a new cognitive module that participates in the
Global Workspace Theory (GWT) competition.

The example adds a "SafetyMonitor" module that proposes EMERGENCY_STOP
when it detects dangerous conditions.

Run from the nsck-demo/ directory:
    python examples/custom_module.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.core.reasoning.global_workspace import (
    GlobalWorkspace,
    Coalition,
    WorkspaceModule,
)


# ── 1. Define a custom module ───────────────────────────────────

class SafetyMonitor(WorkspaceModule):
    """
    A simple safety module that listens to GWT broadcasts and
    proposes EMERGENCY_STOP when it detects danger.
    """

    def __init__(self):
        self.danger_level = 0.0
        self.last_broadcast = None

    def receive_broadcast(self, content):
        """Called by GWT when a winner is broadcast."""
        self.last_broadcast = content
        # If the broadcast mentions danger, increase danger level
        if isinstance(content, str) and "danger" in content.lower():
            self.danger_level = min(1.0, self.danger_level + 0.3)
        else:
            self.danger_level = max(0.0, self.danger_level - 0.1)

    def propose(self, state: dict) -> Coalition:
        """Generate a proposal for the GWT competition."""
        # Check for dangerous conditions
        temperature = state.get("temperature", 20)
        pressure = state.get("pressure", 1.0)

        danger = self.danger_level
        if temperature > 100:
            danger += 0.5
        if pressure > 10:
            danger += 0.4

        return Coalition(
            source="SAFETY",
            content="EMERGENCY_STOP" if danger > 0.5 else "MONITOR",
            base_salience=danger,
            relevance=0.0,
            sender_confidence=0.9,
        )


def main():
    # ── 2. Create GWT and register the module ─────────────────
    gws = GlobalWorkspace()
    safety = SafetyMonitor()
    gws.register_module("safety", safety)

    # ── 3. Simulate scenarios ─────────────────────────────────
    scenarios = [
        {"temperature": 25, "pressure": 1.0},   # Normal
        {"temperature": 120, "pressure": 1.0},   # Hot!
        {"temperature": 50, "pressure": 15.0},   # High pressure!
        {"temperature": 22, "pressure": 1.0},    # Back to normal
    ]

    for i, state in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {state} ---")

        # Generate proposals from our module + a default "do nothing"
        proposals = [
            safety.propose(state),
            Coalition(
                source="DEFAULT",
                content="CONTINUE",
                base_salience=0.3,
                relevance=0.0,
                sender_confidence=0.5,
            ),
        ]

        # Run GWT competition
        winner = gws.compete(proposals)

        if winner:
            print(f"  Winner: {winner.source} → {winner.content}")
            print(f"  Activation: {winner.activation:.2f}")
        else:
            print("  No winner (below threshold)")

        # The safety module receives the broadcast via GWT
        print(f"  Safety danger level: {safety.danger_level:.2f}")


if __name__ == "__main__":
    main()
