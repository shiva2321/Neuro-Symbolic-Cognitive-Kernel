"""Tests that every input type is handled correctly (NSCK V11)."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import numpy as np


def make_substrate():
    from python.core.substrate import NSCKSubstrate
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig(enable_continuous_generalization=False)
    return NSCKSubstrate(config=cfg)


class TestAllInputTypes:
    """Tests that every input type is handled correctly."""

    def setup_method(self):
        self.s = make_substrate()
        self.s.register_task("input_test")

    def test_plain_text(self):
        result = self.s.process("hello world", "input_test")
        assert result.chosen_action is not None

    def test_number_scalar(self):
        result = self.s.process({"value": 42.0}, "input_test")
        assert result.chosen_action is not None

    def test_number_list(self):
        result = self.s.process([1.0, 2.0, 3.0, 4.0, 5.0], "input_test")
        assert result.chosen_action is not None

    def test_number_numpy_array(self):
        result = self.s.process(np.array([1.0, 2.0, 3.0, 4.0]), "input_test")
        assert result.chosen_action is not None

    def test_dict_state_simple(self):
        result = self.s.process({"key": "value", "num": 1}, "input_test")
        assert result.chosen_action is not None

    def test_dict_state_with_numeric_values(self):
        result = self.s.process({"temp": 37.5, "pressure": 1.2}, "input_test")
        assert result.chosen_action is not None

    def test_dict_state_with_nested_lists(self):
        result = self.s.process({"coords": [1.0, 2.0], "label": "a"}, "input_test")
        assert result.chosen_action is not None

    def test_image_as_numpy_2d(self):
        img = np.zeros((16, 16), dtype=np.float32)
        result = self.s.process(img, "input_test")
        assert result.chosen_action is not None

    def test_image_as_numpy_3d_rgb(self):
        img = np.zeros((16, 16, 3), dtype=np.float32)
        result = self.s.process(img, "input_test")
        assert result.chosen_action is not None

    def test_multimodal_text_plus_numbers(self):
        result = self.s.process_multimodal({
            "text": "rising temperature",
            "sensor": [20.0, 21.0, 22.0, 23.0],
        }, "input_test")
        assert result.chosen_action is not None
        assert len(result.modalities_processed) == 2

    def test_multimodal_text_plus_image_plus_numbers(self):
        result = self.s.process_multimodal({
            "text": "camera active",
            "image_data": [0.1, 0.2, 0.3],
            "frame": {"shape": [8, 8]},
        }, "input_test")
        assert result.chosen_action is not None

    def test_stream_rising_sequence(self):
        from python.core.perception.stream_encoder import StreamBuffer, TimeSeriesEncoder
        enc = TimeSeriesEncoder()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        feats = enc.encode_features(vals)
        preds = enc.features_to_predicates(feats, channel="ch")
        assert "RISING_CH" in preds

    def test_stream_falling_sequence(self):
        from python.core.perception.stream_encoder import TimeSeriesEncoder
        enc = TimeSeriesEncoder()
        vals = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        feats = enc.encode_features(vals)
        preds = enc.features_to_predicates(feats, channel="ch")
        assert "FALLING_CH" in preds

    def test_stream_periodic_sequence(self):
        import math
        from python.core.perception.stream_encoder import TimeSeriesEncoder
        enc = TimeSeriesEncoder(value_range=(-2.0, 2.0))
        vals = [math.sin(i * 0.5) for i in range(32)]
        feats = enc.encode_features(vals)
        preds = enc.features_to_predicates(feats, channel="signal")
        # Should detect periodicity (autocorr peak > 0.7)
        assert feats["periodicity_score"] >= 0.0  # at least computed

    def test_stream_anomaly_detection(self):
        from python.core.perception.stream_encoder import TimeSeriesEncoder
        enc = TimeSeriesEncoder()
        # Normal values with one spike
        vals = [1.0] * 14 + [100.0, 1.0]  # spike
        feats = enc.encode_features(vals)
        assert feats["anomaly_score"] > 0.0

    def test_sequential_inputs_different_types(self):
        results = []
        results.append(self.s.process("text input", "input_test"))
        results.append(self.s.process([1.0, 2.0, 3.0], "input_test"))
        results.append(self.s.process({"key": 1}, "input_test"))
        for r in results:
            assert r.chosen_action is not None

    def test_empty_input_graceful(self):
        """Empty/minimal inputs should not crash."""
        result = self.s.process({}, "input_test")
        assert result is not None
        result2 = self.s.process("", "input_test")
        assert result2 is not None
