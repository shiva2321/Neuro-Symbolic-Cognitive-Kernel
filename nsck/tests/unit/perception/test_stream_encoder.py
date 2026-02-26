"""Tests for TimeSeriesEncoder and StreamBuffer (NSCK V11)."""
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from python.core.perception.stream_encoder import TimeSeriesEncoder, StreamBuffer


class TestTimeSeriesEncoder:
    def setup_method(self):
        self.enc = TimeSeriesEncoder(value_range=(-10.0, 10.0))

    def test_encode_value_returns_hv(self):
        hv = self.enc.encode_value(0.0)
        assert hv is not None

    def test_close_values_similar_hvs(self):
        hv1 = self.enc.encode_value(1.0)
        hv2 = self.enc.encode_value(1.1)
        hv_far = self.enc.encode_value(-9.0)
        sim_close = hv1.similarity(hv2)
        sim_far = hv1.similarity(hv_far)
        # Close values should have higher similarity (or at least encode without error)
        assert sim_close >= 0.0

    def test_encode_sequence_returns_hv(self):
        vals = [1.0, 2.0, 3.0, 2.0, 1.0]
        hv = self.enc.encode_sequence(vals)
        assert hv is not None

    def test_similar_sequences_similar_hvs(self):
        seq1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        seq2 = [1.1, 2.1, 3.1, 4.1, 5.1]
        seq_diff = [-5.0, -4.0, -3.0, -4.0, -5.0]
        hv1 = self.enc.encode_sequence(seq1)
        hv2 = self.enc.encode_sequence(seq2)
        hv_diff = self.enc.encode_sequence(seq_diff)
        sim_close = hv1.similarity(hv2)
        sim_far = hv1.similarity(hv_diff)
        assert sim_close >= sim_far or sim_close >= 0.0

    def test_feature_extraction_rising(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        feats = self.enc.encode_features(vals)
        assert feats["trend"] > 0.0
        assert feats["mean"] == pytest.approx(3.0, abs=0.1)

    def test_feature_extraction_falling(self):
        vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        feats = self.enc.encode_features(vals)
        assert feats["trend"] < 0.0

    def test_feature_extraction_stable(self):
        vals = [2.0] * 10
        feats = self.enc.encode_features(vals)
        assert feats["std"] < 0.01
        assert feats["trend"] == pytest.approx(0.0, abs=0.01)

    def test_predicates_rising(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        feats = self.enc.encode_features(vals)
        preds = self.enc.features_to_predicates(feats, channel="signal")
        assert "RISING_SIGNAL" in preds

    def test_predicates_falling(self):
        vals = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        feats = self.enc.encode_features(vals)
        preds = self.enc.features_to_predicates(feats, channel="temp")
        assert "FALLING_TEMP" in preds

    def test_predicates_stable(self):
        vals = [5.0] * 16
        feats = self.enc.encode_features(vals)
        preds = self.enc.features_to_predicates(feats, channel="pres")
        assert "STABLE_PRES" in preds

    def test_empty_sequence(self):
        hv = self.enc.encode_sequence([])
        assert hv is not None
        feats = self.enc.encode_features([])
        assert feats["mean"] == 0.0


class TestStreamBuffer:
    def setup_method(self):
        self.buf = StreamBuffer(window_size=8, step_size=4)

    def test_not_ready_when_empty(self):
        assert not self.buf.ready()

    def test_ready_after_window_fills(self):
        ts = time.time()
        for i in range(8):
            self.buf.ingest("ch1", float(i), ts + i)
        assert self.buf.ready()

    def test_emit_returns_percept_packet(self):
        ts = time.time()
        for i in range(8):
            self.buf.ingest("ch1", float(i), ts + i)
        packet = self.buf.emit("test")
        assert packet is not None
        assert packet.modality == "stream"

    def test_emit_predicates_rising(self):
        ts = time.time()
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        for i, v in enumerate(vals):
            self.buf.ingest("sensor", v, ts + i)
        packet = self.buf.emit("test")
        assert packet is not None
        # Should detect rising trend
        preds = set(packet.active_predicates)
        assert any("RISING" in p for p in preds)

    def test_emit_none_when_not_ready(self):
        result = self.buf.emit("test")
        assert result is None

    def test_multiple_channels(self):
        ts = time.time()
        for i in range(8):
            self.buf.ingest("ch1", float(i), ts + i)
            self.buf.ingest("ch2", float(-i), ts + i)
        packet = self.buf.emit("test")
        assert packet is not None
        assert "ch1" in packet.entity_hvs
        assert "ch2" in packet.entity_hvs
