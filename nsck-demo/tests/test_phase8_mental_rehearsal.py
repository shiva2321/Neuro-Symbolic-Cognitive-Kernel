"""
Phase 8 Verification: Mental Rehearsal & Veto
==============================================
Tests the compete_with_rehearsal() mechanism in GlobalWorkspace, including
danger vector registration, veto logic, and deadlock fallback.
"""
import sys, os

import unittest
import numpy as np
import python.core.vsa.hypervec_shim as hv
from python.core.vsa.hypervec_py import HyperVectorPy
from python.core.reasoning.global_workspace import GlobalWorkspace, Coalition


class FakeWorldModel:
    """Mock WorldModel that returns a controllable prediction."""

    def __init__(self, predicted_bits=None, predicted_reward=0.0):
        self._predicted_bits = predicted_bits
        self._predicted_reward = predicted_reward

    def set_prediction(self, bits, reward):
        self._predicted_bits = bits
        self._predicted_reward = reward

    def imagine(self, state_hv, action_hv):
        return self._predicted_bits, self._predicted_reward

    def is_ready(self, task_tag):
        return True


def make_action_hv(action_str):
    """Simple deterministic HV from action string (mirrors CognitiveEngine.get_concept_hv)."""
    return hv.HyperVector(hash(action_str) % (2**32))


class TestVetoMechanism(unittest.TestCase):
    """Verify danger vector veto logic."""

    def test_veto_prevents_dangerous_action(self):
        """A proposal whose predicted state matches a danger vector should be vetoed."""
        gw = GlobalWorkspace(attention_threshold=0.3)

        # Create a known danger state
        danger_hv = HyperVectorPy(seed=999)
        gw.register_danger(danger_hv)

        # FakeWorldModel predicts exactly the danger state bits
        wm = FakeWorldModel(
            predicted_bits=danger_hv.bits.astype(np.float64),
            predicted_reward=-1.0,
        )

        current_hv = hv.HyperVector(1)

        proposals = [
            Coalition(source="RULES", content="ACTION_UP", base_salience=0.9),
        ]

        winner = gw.compete_with_rehearsal(
            proposals, current_hv, wm, make_action_hv
        )

        # Should fall through to emergency (only proposal was vetoed)
        self.assertEqual(winner.source, "EMERGENCY")
        self.assertEqual(winner.content, "ACTION_STAY")
        self.assertGreater(len(gw.rehearsal_log), 0)

    def test_safe_action_passes(self):
        """A proposal whose predicted state is safe should pass through."""
        gw = GlobalWorkspace(attention_threshold=0.3)

        # Danger state
        danger_hv = HyperVectorPy(seed=999)
        gw.register_danger(danger_hv)

        # Predicted state is completely different (safe)
        safe_bits = HyperVectorPy(seed=1).bits.astype(np.float64)
        wm = FakeWorldModel(predicted_bits=safe_bits, predicted_reward=0.5)

        current_hv = hv.HyperVector(2)

        proposals = [
            Coalition(source="PLANNER", content="ACTION_RIGHT", base_salience=0.8),
        ]

        winner = gw.compete_with_rehearsal(
            proposals, current_hv, wm, make_action_hv
        )

        self.assertEqual(winner.source, "PLANNER")
        self.assertEqual(winner.content, "ACTION_RIGHT")

    def test_deadlock_fallback(self):
        """When ALL proposals are dangerous, EMERGENCY fallback should trigger."""
        gw = GlobalWorkspace(attention_threshold=0.3)

        danger_hv = HyperVectorPy(seed=500)
        gw.register_danger(danger_hv)

        wm = FakeWorldModel(
            predicted_bits=danger_hv.bits.astype(np.float64),
            predicted_reward=-1.0,
        )

        current_hv = hv.HyperVector(3)

        proposals = [
            Coalition(source="RULES", content="ACTION_UP", base_salience=0.9),
            Coalition(source="SNN", content="ACTION_DOWN", base_salience=0.7),
            Coalition(source="EXPLORATION", content="ACTION_LEFT", base_salience=0.6),
        ]

        winner = gw.compete_with_rehearsal(
            proposals, current_hv, wm, make_action_hv
        )

        self.assertEqual(winner.source, "EMERGENCY")

    def test_second_proposal_selected_after_veto(self):
        """If the top proposal is dangerous, the next-best safe one should win."""
        gw = GlobalWorkspace(attention_threshold=0.3)

        danger_hv = HyperVectorPy(seed=700)
        gw.register_danger(danger_hv)

        danger_bits = danger_hv.bits.astype(np.float64)
        safe_bits = HyperVectorPy(seed=1).bits.astype(np.float64)

        call_count = [0]
        original_imagine = None

        class AlternatingWorldModel:
            """First call returns danger, second call returns safe."""
            def imagine(self, state_hv, action_hv):
                call_count[0] += 1
                if call_count[0] == 1:
                    return danger_bits, -1.0
                return safe_bits, 0.5

            def is_ready(self, task_tag):
                return True

        wm = AlternatingWorldModel()
        current_hv = hv.HyperVector(4)

        proposals = [
            Coalition(source="RULES", content="BAD_ACTION", base_salience=0.9),
            Coalition(source="PLANNER", content="GOOD_ACTION", base_salience=0.7),
        ]

        winner = gw.compete_with_rehearsal(
            proposals, current_hv, wm, make_action_hv
        )

        self.assertEqual(winner.source, "PLANNER")
        self.assertEqual(winner.content, "GOOD_ACTION")
        # First proposal should have been vetoed
        self.assertEqual(len(gw.rehearsal_log), 1)
        self.assertEqual(gw.rehearsal_log[0].vetoed_source, "RULES")


class TestDangerVectorRegistry(unittest.TestCase):
    """Verify danger vector management."""

    def test_register_and_query(self):
        gw = GlobalWorkspace()
        self.assertEqual(len(gw._danger_vectors), 0)

        dv = HyperVectorPy(seed=100)
        gw.register_danger(dv)
        self.assertEqual(len(gw._danger_vectors), 1)

    def test_lru_eviction(self):
        gw = GlobalWorkspace()
        gw.max_danger_vectors = 5
        for i in range(10):
            gw.register_danger(HyperVectorPy(seed=i))
        self.assertEqual(len(gw._danger_vectors), 5)

    def test_is_dangerous_positive(self):
        gw = GlobalWorkspace()
        dv = HyperVectorPy(seed=50)
        gw.register_danger(dv)

        # Same vector should be dangerous
        is_d, sim = gw._is_dangerous(dv)
        self.assertTrue(is_d)
        self.assertGreater(sim, 0.99)

    def test_is_dangerous_negative(self):
        gw = GlobalWorkspace()
        dv = HyperVectorPy(seed=50)
        gw.register_danger(dv)

        # Random vector should not be dangerous
        other = HyperVectorPy(seed=9999)
        is_d, sim = gw._is_dangerous(other)
        self.assertFalse(is_d)
        self.assertAlmostEqual(sim, 0.5, delta=0.05)


class TestTelemetry(unittest.TestCase):
    """Verify telemetry methods."""

    def test_get_status_includes_phase8(self):
        gw = GlobalWorkspace()
        status = gw.get_status()
        self.assertIn("danger_vectors", status)
        self.assertIn("rehearsal_vetoes", status)

    def test_get_recent_vetoes_empty(self):
        gw = GlobalWorkspace()
        self.assertEqual(gw.get_recent_vetoes(), [])


if __name__ == "__main__":
    unittest.main()
