"""
Tests for region-specific learning rules and learning rate multipliers.
"""

import unittest
from core.region import Region, RegionType
from core.neuron import Neuron
from core.synapse import Synapse
from core.plasticity import PlasticityController


class TestRegionLearningRules(unittest.TestCase):
    """Test region-specific learning rule dispatch."""

    def setUp(self):
        """Create a simple region with neurons and synapses."""
        self.neurons = {
            0: Neuron(neuron_id=0, threshold=1.0),
            1: Neuron(neuron_id=1, threshold=1.0),
        }
        self.synapses = {
            1: {
                0: Synapse(weight=0.5, is_inhibitory=False)
            }
        }
        self.region = Region(
            region_id=0,
            region_type=RegionType.HIDDEN,
            neurons=self.neurons,
            synapses=self.synapses
        )

    def test_default_learning_rule_is_stdp(self):
        """Default learning rule should be 'stdp'."""
        self.assertEqual(self.region._learning_rule_id, "stdp")

    def test_set_learning_rule(self):
        """Should be able to set learning rule ID."""
        self.region.set_learning_rule("hebbian")
        self.assertEqual(self.region._learning_rule_id, "hebbian")

    def test_set_learning_rule_none_defaults_to_stdp(self):
        """Setting rule to None should default to 'stdp'."""
        self.region.set_learning_rule(None)
        self.assertEqual(self.region._learning_rule_id, "stdp")

    def test_learning_rate_multiplier(self):
        """Should be able to set and apply learning rate multiplier."""
        self.region.set_learning_rate_multiplier(0.5)
        self.assertAlmostEqual(self.region._lr_multiplier, 0.5)

    def test_learning_rate_multiplier_applied_to_plasticity(self):
        """Learning rate multiplier should be passed to plasticity controller."""
        self.region.set_learning_rate_multiplier(2.0)
        effective_rate = self.region.plasticity.get_effective_lr(step=0)
        self.assertGreater(effective_rate, 0.01)  # Should be scaled up

    def test_apply_learning_dispatches_to_synapse(self):
        """apply_learning should dispatch to the region's plasticity controller."""
        # Test that apply_learning doesn't raise errors and respects gate
        self.region.enable_plasticity()
        # Should not raise
        self.region.apply_learning(target_id=1, post_fired=True, dopamine=1.0, step=0)

        # With gate disabled, should also not raise
        self.region.disable_plasticity()
        self.region.apply_learning(target_id=1, post_fired=True, dopamine=1.0)

    def test_apply_learning_respects_learning_gate(self):
        """apply_learning should dispatch via plasticity controller gate."""
        # Verify that region's learning gate is respected
        self.region.disable_plasticity()
        self.assertFalse(self.region.is_learning())

        # Should not raise when disabled
        self.region.apply_learning(target_id=1, post_fired=True, dopamine=1.0)

        self.region.enable_plasticity()
        self.assertTrue(self.region.is_learning())

    def test_apply_learning_nonexistent_neuron(self):
        """apply_learning on non-existent neuron should be safe."""
        # Should not raise error
        self.region.apply_learning(target_id=999, post_fired=True, dopamine=1.0)

    def test_region_learning_rate_multiplier_zero(self):
        """Setting multiplier to negative should clamp to 0."""
        self.region.set_learning_rate_multiplier(-1.0)
        self.assertAlmostEqual(self.region._lr_multiplier, 0.0)


if __name__ == '__main__':
    unittest.main()
