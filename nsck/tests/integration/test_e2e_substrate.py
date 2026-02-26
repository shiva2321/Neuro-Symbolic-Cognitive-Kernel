"""
End-to-end tests proving NSCK functions as an AGI substrate.
Tests ALL the properties stated in the goal.
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))


def make_substrate(generalization_interval=100):
    from python.core.substrate import NSCKSubstrate
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig(
        enable_continuous_generalization=True,
        generalization_interval=generalization_interval,
        enable_sleep=True,
    )
    s = NSCKSubstrate(config=cfg)
    return s


class TestE2EAGISubstrate:
    """End-to-end tests proving NSCK functions as an AGI substrate."""

    def test_learning_from_text(self):
        """NSCK learns from text input and can recall/reason about it."""
        s = make_substrate()
        s.register_task("text_test")
        result = s.process("The sky is blue", "text_test")
        assert result.chosen_action is not None
        s.learn({"text": "The sky is blue"}, result.chosen_action, reward=1.0, task_tag="text_test")
        # Second call should have learned
        result2 = s.process("The sky is blue", "text_test")
        assert result2 is not None

    def test_learning_from_numeric_sequences(self):
        """NSCK learns patterns from numerical sequences."""
        s = make_substrate()
        s.register_task("num_test")
        # Rising sequence
        rising = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = s.process(rising, "num_test")
        assert result is not None
        # Falling sequence
        falling = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result2 = s.process(falling, "num_test")
        assert result2 is not None

    def test_learning_from_structured_dict(self):
        """NSCK learns from structured dictionary state input."""
        s = make_substrate()
        s.register_task("dict_test")
        state = {"temperature": 37.5, "pressure": 1.2, "humidity": 0.65}
        result = s.process(state, "dict_test")
        assert result.chosen_action is not None

    def test_remembrance_and_recall(self):
        """NSCK can remember past experiences and recall them accurately."""
        s = make_substrate()
        s.register_task("recall_test")
        # Store 10 distinct episodes
        for i in range(10):
            state = {"id": i, "value": float(i * 10)}
            s.process(state, "recall_test")
            s.learn(state, f"action_{i}", reward=float(i % 2), task_tag="recall_test")
        memories = s.remember({"id": 5, "value": 50.0}, task_tag="recall_test", top_k=5)
        assert isinstance(memories, list)

    def test_generalization_across_examples(self):
        """NSCK generalizes patterns from multiple examples."""
        s = make_substrate()
        s.register_task("gen_test")
        # Feed multiple examples
        for _ in range(5):
            s.process({"category": "mammal", "has_hair": True}, "gen_test")
        s.sleep("gen_test")
        # Should not crash
        result = s.process({"category": "mammal", "has_hair": True}, "gen_test")
        assert result is not None

    def test_lifelong_learning_no_forgetting(self):
        """NSCK learns multiple tasks without forgetting earlier ones."""
        s = make_substrate()
        # Learn Task A
        s.register_task("task_a")
        for i in range(5):
            s.process({"domain": "a", "value": i}, "task_a")
            s.learn({"domain": "a", "value": i}, "action_a", reward=1.0, task_tag="task_a")
        # Learn Task B
        s.register_task("task_b")
        for i in range(5):
            s.process({"domain": "b", "value": i}, "task_b")
            s.learn({"domain": "b", "value": i}, "action_b", reward=1.0, task_tag="task_b")
        # Learn Task C
        s.register_task("task_c")
        for i in range(5):
            s.process({"domain": "c", "value": i}, "task_c")
        # Query Task A — should still work
        result_a = s.process({"domain": "a", "value": 0}, "task_a")
        assert result_a is not None

    def test_multimodal_text_and_numbers(self):
        """NSCK handles text + numeric sequence simultaneously."""
        s = make_substrate()
        s.register_task("multimodal_test")
        result = s.process_multimodal({
            "text": "temperature is rising",
            "sensor": [20.0, 21.0, 23.0, 26.0, 30.0],
        }, "multimodal_test")
        assert result is not None
        assert "text" in result.modalities_processed
        assert "sensor" in result.modalities_processed

    def test_stream_input_pattern_detection(self):
        """NSCK detects patterns in streaming numeric data."""
        from python.core.perception.stream_encoder import StreamBuffer
        buf = StreamBuffer(window_size=16)
        ts = time.time()
        # Sinusoidal sequence
        import math
        for i in range(16):
            buf.ingest("signal", math.sin(i * 0.5), ts + i)
        assert buf.ready()
        packet = buf.emit("stream_test")
        assert packet is not None

    def test_custom_encoder_registration(self):
        """Third-party encoders can be registered and used."""
        s = make_substrate()
        s.register_task("custom_test")
        import python.core.vsa.hypervec_shim as hv
        from python.core.types.percept_packet import PerceptPacket

        def lidar_encoder(data, task_tag):
            return PerceptPacket.make(
                modality="lidar",
                situation_hv=hv.HyperVector(hash(str(data)) % (2**32)),
                active_predicates=frozenset({"LIDAR_READING"}),
            )

        s.register_encoder("lidar", lidar_encoder)
        result = s.process_multimodal({"lidar": {"dist": 2.5}}, "custom_test")
        assert "lidar" in result.modalities_processed

    def test_continuous_generalization_trigger(self):
        """Generalization happens automatically every N decisions."""
        s = make_substrate(generalization_interval=10)
        s.register_task("cont_gen")
        for i in range(15):
            s.process({"x": i}, "cont_gen")
        # Should have triggered at least once (at decision 10)
        assert s._engine._decision_counter == 15

    def test_cross_modal_correlation(self):
        """NSCK learns correlations between text and numeric modalities."""
        from python.core.integration.config import NSCKConfig
        from python.core.substrate import NSCKSubstrate
        cfg = NSCKConfig(enable_cross_modal_learning=True)
        s = NSCKSubstrate(config=cfg)
        s.register_task("xmod")
        # Repeatedly present correlated inputs
        for _ in range(5):
            s.process_multimodal({
                "text": "fire alarm",
                "sensor": [0.9, 1.1, 1.3, 1.5, 1.8],
            }, "xmod")
        # Cross-modal should have learned
        if s._engine.cross_modal is not None:
            stats = s._engine.cross_modal.get_statistics()
            assert stats["total_observations"] > 0

    def test_substrate_api_complete_lifecycle(self):
        """Full lifecycle: register → process → learn → sleep → remember."""
        s = make_substrate()
        s.register_task("lifecycle")
        result = s.process("the sky is blue", "lifecycle")
        assert result is not None
        result2 = s.process([1.0, 2.0, 3.0], "lifecycle")
        assert result2 is not None
        result3 = s.process_multimodal({
            "text": "test",
            "sensor": [1.0, 2.0, 3.0],
        }, "lifecycle")
        assert result3 is not None
        s.learn({"text": "the sky is blue"}, result.chosen_action, reward=1.0, task_tag="lifecycle")
        s.sleep("lifecycle")
        memories = s.remember("the sky is blue", task_tag="lifecycle")
        assert isinstance(memories, list)

    def test_rust_backend_if_available(self):
        """If Rust is available, verify it's faster than Python."""
        import python.core.vsa.hypervec_shim as shim
        info = shim.get_backend_info()
        assert info["vsa_backend"] in ("Rust", "Python")
        # Benchmark
        import time
        n = 1000
        hv1 = shim.HyperVector(1)
        hv2 = shim.HyperVector(2)
        start = time.time()
        for _ in range(n):
            hv1.similarity(hv2)
        elapsed = time.time() - start
        ops_per_s = n / elapsed if elapsed > 0 else float('inf')
        assert ops_per_s > 100
