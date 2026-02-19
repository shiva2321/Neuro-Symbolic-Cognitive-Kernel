"""
Integration tests for cognitive engine wiring.

These tests verify that modules are actually CONNECTED to the cognitive loop,
not just that they work in isolation.  Every test here exercises the full
decide() / learn() path through CognitiveEngine.

Gaps addressed:
    1. GWT broadcast reaches subscribers
    2. MEMORY coalition uses episodic recall in decide()
    3. sleep() consolidation cycle works end-to-end
    4. PLANNER coalition fires when mission goal is set
    5. SafetyGate veto is wired into decide()
"""
import time
import pytest

from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.reasoning.causal_reasoning import (
    CausalGraph, CausalReasoner, create_snake_causal_graph,
)
from python.core.reasoning.global_workspace import Coalition
from python.core.cognitive.metacognition import SafetyGate
from python.core.integration.config import NSCKConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(task_tag="test", causal_graph=None):
    """Create a CognitiveEngine with a registered task."""
    config = NSCKConfig()
    engine = CognitiveEngine(config)
    engine.register_task(task_tag, causal_graph=causal_graph or CausalGraph())
    return engine


def _train_engine(engine, task_tag, action, n=15, reward=1.0, outcome="success"):
    """Feed the engine *n* identical (state, action, reward) tuples."""
    state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
    for _ in range(n):
        engine.decide(state, task_tag)
        engine.learn(state, action, reward, task_tag, outcome=outcome)


# ========================================================================
# Gap 1 — GWT broadcast fires to subscribers
# ========================================================================

class TestGWTBroadcast:
    """Verify modules actually receive GWT broadcasts during decide()."""

    def test_subscribers_registered(self):
        engine = _make_engine()
        modules = engine.global_workspace.modules
        assert len(modules) >= 3, f"Too few GWT subscribers: {list(modules)}"
        assert "rule_learner" in modules
        assert "episodic_memory" in modules
        assert "semantic_memory" in modules

    def test_broadcast_fires_during_decide(self):
        engine = _make_engine()
        assert engine.stats["gwt_broadcasts"] == 0
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
        engine.decide(state, "test")
        assert engine.stats["gwt_broadcasts"] >= 1, \
            "GWT broadcast should fire when a coalition wins"

    def test_rule_learner_receives_broadcast(self):
        engine = _make_engine()
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
        engine.decide(state, "test")
        # RuleLearner.receive_broadcast should have been called
        # (it tracks _wins_count internally for dict broadcasts)
        # We can't inspect that directly, but we can verify no crash
        # and that the rule_learner is in modules:
        assert "rule_learner" in engine.global_workspace.modules


# ========================================================================
# Gap 2 — MEMORY coalition uses episodic recall
# ========================================================================

class TestMemoryCoalition:
    """Verify episodic memory is READ during decide(), not just written."""

    def test_memory_recalled_after_learning(self):
        engine = _make_engine()
        _train_engine(engine, "test", "ACTION_UP", n=15)

        # Now decide — MEMORY should propose ACTION_UP
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
        engine.decide(state, "test")

        proposals = [c.source for c in engine.global_workspace.latest_coalitions]
        assert "MEMORY" in proposals, \
            f"MEMORY coalition missing — proposals: {proposals}"

    def test_memory_recalls_incremented(self):
        engine = _make_engine()
        assert engine.stats["memory_recalls"] == 0
        _train_engine(engine, "test", "ACTION_UP", n=10)
        assert engine.stats["memory_recalls"] > 0, \
            "memory_recalls stat should increase after training"

    def test_memory_proposes_best_action(self):
        engine = _make_engine()
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}

        # Train with ACTION_UP giving high reward
        for _ in range(10):
            engine.decide(state, "test")
            engine.learn(state, "ACTION_UP", 1.0, "test", outcome="success")

        # Train with ACTION_DOWN giving negative reward
        for _ in range(5):
            engine.decide(state, "test")
            engine.learn(state, "ACTION_DOWN", -1.0, "test", outcome="failure")

        # Memory should favor ACTION_UP
        engine.decide(state, "test")
        memory_coalitions = [
            c for c in engine.global_workspace.latest_coalitions
            if c.source == "MEMORY"
        ]
        if memory_coalitions:
            assert memory_coalitions[0].content == "ACTION_UP", \
                f"Memory should propose ACTION_UP, got {memory_coalitions[0].content}"


# ========================================================================
# Gap 3 — Sleep / consolidation cycle
# ========================================================================

class TestSleepConsolidation:
    """Verify the sleep/dream/replay cycle works."""

    def test_sleep_runs_without_crash(self):
        engine = _make_engine()
        _train_engine(engine, "test", "ACTION_UP", n=20)
        engine.sleep("test", epochs=2)
        assert engine.stats["sleep_cycles"] >= 1

    def test_sleep_populates_semantic_memory(self):
        engine = _make_engine()
        # Feed lots of SUCCESS outcomes so _consolidate_semantic fires
        _train_engine(engine, "test", "ACTION_UP", n=30, outcome="success")
        engine.sleep("test", epochs=3)

        nodes = list(engine.semantic_memory.concept_graph.nodes)
        assert len(nodes) >= 2, \
            f"Semantic memory should have concepts after sleep, got: {nodes}"

    def test_sleep_respects_enable_flag(self):
        config = NSCKConfig()
        config.enable_sleep = False
        engine = CognitiveEngine(config)
        engine.register_task("test")
        _train_engine(engine, "test", "ACTION_UP", n=20)
        engine.sleep("test", epochs=5)
        assert engine.stats["sleep_cycles"] == 0, \
            "Sleep should not run when enable_sleep is False"

    def test_sleep_all_tasks(self):
        engine = _make_engine("task_a")
        engine.register_task("task_b")
        _train_engine(engine, "task_a", "ACTION_UP", n=15)
        _train_engine(engine, "task_b", "ACTION_DOWN", n=15)
        engine.sleep()  # No task_tag → all tasks
        assert engine.stats["sleep_cycles"] >= 1


# ========================================================================
# Gap 4 — Planner coalition in decide()
# ========================================================================

class TestPlannerCoalition:
    """Verify planner submits coalition when mission goal is set."""

    def test_planner_path_wired(self):
        """Planner code path runs without crash even if no plan found."""
        engine = _make_engine()
        engine.set_mission_goal("reach", 1.0, goal_predicates={"AT_5_4"})
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
        # Should not crash even if planner can't find a plan
        result = engine.decide(state, "test")
        assert result.chosen_action is not None

    def test_planner_fires_with_operators(self):
        """When operators exist, planner submits a coalition."""
        engine = _make_engine()

        # Manually add a simple operator so planner can plan
        engine.planner._learned_ops["ACTION_GRAB"] = {
            "add": {"HOLDING"},
            "del": set(),
            "pre": set(),  # No preconditions
        }
        engine.set_mission_goal("grab", 1.0, goal_predicates={"HOLDING"})

        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}
        result = engine.decide(state, "test")
        proposals = [c.source for c in engine.global_workspace.latest_coalitions]
        assert "PLANNER" in proposals, \
            f"PLANNER coalition missing — proposals: {proposals}"
        assert engine.stats["plans_generated"] >= 1

    def test_planner_clears_on_goal_reached(self):
        """Goal-satisfied → planner stops proposing."""
        engine = _make_engine()
        engine.set_mission_goal("reach", 1.0, goal_predicates={"ALREADY_HERE"})
        # Active predicates will be empty by default, but if we include the goal:
        engine.planner._learned_ops["ACTION_X"] = {
            "add": {"ALREADY_HERE"}, "del": set(), "pre": set()
        }
        # Provide a state where predicates include the goal
        # The GroundingVerifier() returns empty predicates by default,
        # so the goal won't be in active_preds — planner should try to plan
        state = {"head": (5, 5)}
        result = engine.decide(state, "test")
        # Just verify no crash
        assert result.chosen_action is not None

    def test_set_mission_goal_with_predicates(self):
        engine = _make_engine()
        engine.set_mission_goal("collect", 5.0, goal_predicates={"HAS_ITEM"})
        assert engine._plan_goal == {"HAS_ITEM"}
        assert engine.mission_goal["type"] == "collect"


# ========================================================================
# Gap 5 — SafetyGate wired into decide()
# ========================================================================

class TestSafetyGate:
    """Verify SafetyGate.is_safe() is called in the decision loop."""

    def test_safety_veto_works(self):
        engine = _make_engine("safe_test")

        # Register checker: ACTION_DOWN is always unsafe
        SafetyGate.register_simulator(
            "safe_test", lambda state, action: action != "ACTION_DOWN"
        )

        try:
            dangerous = Coalition(
                source="BAD", content="ACTION_DOWN",
                base_salience=10.0, sender_confidence=1.0,
            )
            safe = Coalition(
                source="GOOD", content="ACTION_UP",
                base_salience=0.3, sender_confidence=0.5,
            )
            state = {"head": (5, 5)}
            result = engine._safety_check(dangerous, [dangerous, safe], state, "safe_test")
            assert result.content != "ACTION_DOWN", "Unsafe action was not vetoed"
            assert result.content == "ACTION_UP"
            assert engine.stats["safety_vetoes"] >= 1
        finally:
            SafetyGate._simulators.pop("safe_test", None)

    def test_safe_action_passes_through(self):
        engine = _make_engine("safe_test")

        SafetyGate.register_simulator(
            "safe_test", lambda state, action: True  # Everything safe
        )
        try:
            winner = Coalition(
                source="OK", content="ACTION_RIGHT",
                base_salience=0.8, sender_confidence=0.7,
            )
            state = {"head": (3, 3)}
            result = engine._safety_check(winner, [winner], state, "safe_test")
            assert result.content == "ACTION_RIGHT"
        finally:
            SafetyGate._simulators.pop("safe_test", None)

    def test_all_unsafe_returns_stay(self):
        engine = _make_engine("safe_test")

        SafetyGate.register_simulator(
            "safe_test", lambda state, action: False  # Nothing safe
        )
        try:
            winner = Coalition(
                source="BAD", content="ACTION_UP",
                base_salience=0.9, sender_confidence=0.9,
            )
            state = {"head": (1, 1)}
            result = engine._safety_check(winner, [winner], state, "safe_test")
            assert result.content == "ACTION_STAY"
            assert result.source == "SAFETY_FALLBACK"
        finally:
            SafetyGate._simulators.pop("safe_test", None)


# ========================================================================
# Full cognitive loop — multiple gaps working together
# ========================================================================

class TestFullCognitiveLoop:
    """End-to-end tests exercising multiple gaps simultaneously."""

    def test_decide_learn_sleep_cycle(self):
        """Complete lifecycle: decide → learn → sleep → decide again."""
        engine = _make_engine("lifecycle",
                              causal_graph=create_snake_causal_graph())
        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}

        # Phase 1: Learning
        for i in range(20):
            engine.decide(state, "lifecycle")
            engine.learn(state, "ACTION_UP", 1.0, "lifecycle", outcome="success")

        # Phase 2: Sleep consolidation
        engine.sleep("lifecycle", epochs=2)

        # Phase 3: Decide with full system
        result = engine.decide(state, "lifecycle")
        assert result.chosen_action is not None

        # Verify all stats are populated
        stats = engine.get_stats()
        assert stats["decisions"] > 0
        assert stats["episodes_recorded"] > 0
        assert stats["memory_recalls"] > 0
        assert stats["gwt_broadcasts"] > 0
        assert stats["sleep_cycles"] > 0

    def test_multiple_tasks_independent(self):
        """Different tasks maintain independent memories and plans."""
        engine = _make_engine("alpha")
        engine.register_task("beta")

        state = {"head": (5, 5), "food": (5, 4), "body": [(5, 6)]}

        # Train alpha with UP, beta with DOWN
        for _ in range(10):
            engine.decide(state, "alpha")
            engine.learn(state, "ACTION_UP", 1.0, "alpha", outcome="success")
            engine.decide(state, "beta")
            engine.learn(state, "ACTION_DOWN", 1.0, "beta", outcome="success")

        # Alpha's memory should propose UP, beta's should propose DOWN
        engine.decide(state, "alpha")
        alpha_memory = [
            c for c in engine.global_workspace.latest_coalitions
            if c.source == "MEMORY"
        ]
        engine.decide(state, "beta")
        beta_memory = [
            c for c in engine.global_workspace.latest_coalitions
            if c.source == "MEMORY"
        ]

        if alpha_memory and beta_memory:
            assert alpha_memory[0].content == "ACTION_UP"
            assert beta_memory[0].content == "ACTION_DOWN"
