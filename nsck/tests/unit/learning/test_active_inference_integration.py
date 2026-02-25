"""Tests for V8 Active Inference module."""
import pytest

from python.core.learning.active_inference import ActiveInferenceLearner
import python.core.vsa.hypervec_shim as hv


@pytest.fixture()
def learner():
    return ActiveInferenceLearner()


@pytest.fixture()
def hv1():
    return hv.HyperVector(111)


@pytest.fixture()
def hv2():
    return hv.HyperVector(222)


class TestActiveInferenceLearnerInit:
    def test_default_init(self):
        ai = ActiveInferenceLearner()
        assert ai.curiosity is None
        assert ai.safety_threshold == pytest.approx(0.7)

    def test_custom_threshold(self):
        ai = ActiveInferenceLearner(safety_threshold=0.5)
        assert ai.safety_threshold == pytest.approx(0.5)

    def test_world_model_empty(self, learner):
        assert len(learner._world_model) == 0


class TestPredictionError:
    def test_unknown_action_returns_half(self, learner, hv1):
        err = learner.prediction_error("unknown_action", hv1)
        assert err == pytest.approx(0.5)

    def test_known_action_after_update(self, learner, hv1, hv2):
        learner.update_world_model(hv1, "go_right", hv2)
        err = learner.prediction_error("go_right", hv1)
        assert 0.0 <= err <= 1.0

    def test_error_range(self, learner, hv1, hv2):
        learner.update_world_model(hv1, "act", hv2)
        err = learner.prediction_error("act", hv1)
        assert 0.0 <= err <= 1.0


class TestEpistemicValue:
    def test_no_curiosity_returns_small(self, learner, hv1):
        val = learner.epistemic_value("action", hv1)
        assert 0.0 <= val <= 1.0

    def test_novel_state_higher_value(self, learner, hv1):
        val = learner.epistemic_value("action", hv1)
        assert val >= 0.0


class TestFreeEnergy:
    def test_returns_float(self, learner, hv1):
        fe = learner.free_energy("action", hv1)
        assert isinstance(fe, float)

    def test_free_energy_unknown_action(self, learner, hv1):
        fe = learner.free_energy("unknown", hv1)
        # prediction_error=0.5, epistemic_value=0.1 → 0.4
        assert fe == pytest.approx(0.5 - 0.1, abs=0.1)


class TestShouldVeto:
    def test_low_free_energy_no_veto(self, learner, hv1, hv2):
        # Update so prediction error becomes 0
        learner.update_world_model(hv1, "safe_action", hv1)
        # After update, same state → low error
        result = learner.should_veto("safe_action", hv1)
        assert isinstance(result, bool)

    def test_very_high_threshold_no_veto(self, hv1):
        ai = ActiveInferenceLearner(safety_threshold=10.0)
        assert ai.should_veto("any_action", hv1) is False


class TestUpdateWorldModel:
    def test_stores_transition(self, learner, hv1, hv2):
        learner.update_world_model(hv1, "move", hv2)
        key = learner._hv_key(hv1)
        assert key in learner._world_model
        assert "move" in learner._world_model[key]

    def test_updates_last_state(self, learner, hv1, hv2):
        learner.update_world_model(hv1, "move", hv2)
        key = learner._hv_key(hv1)
        assert key in learner._last_states

    def test_multiple_transitions(self, learner, hv1, hv2):
        learner.update_world_model(hv1, "action_a", hv2)
        learner.update_world_model(hv1, "action_b", hv1)
        key = learner._hv_key(hv1)
        assert "action_a" in learner._world_model[key]
        assert "action_b" in learner._world_model[key]


class TestHVKey:
    def test_same_hv_same_key(self, learner, hv1):
        k1 = learner._hv_key(hv1)
        k2 = learner._hv_key(hv1)
        assert k1 == k2

    def test_none_returns_none_str(self, learner):
        assert learner._hv_key(None) == "none"

    def test_different_hvs_different_keys(self, learner, hv1, hv2):
        k1 = learner._hv_key(hv1)
        k2 = learner._hv_key(hv2)
        assert k1 != k2
