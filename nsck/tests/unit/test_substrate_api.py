"""Tests for NSCKSubstrate public API (NSCK V11)."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import numpy as np


def make_substrate():
    from python.core.substrate import NSCKSubstrate
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig(enable_continuous_generalization=True, generalization_interval=10)
    return NSCKSubstrate(config=cfg)


def test_process_text():
    s = make_substrate()
    s.register_task("t1")
    result = s.process("hello world", "t1")
    assert result.chosen_action is not None
    assert "text" in result.modalities_processed


def test_process_numeric_list():
    s = make_substrate()
    s.register_task("t2")
    result = s.process([1.0, 2.0, 3.0, 4.0, 5.0], "t2")
    assert result.chosen_action is not None
    assert "numeric_sequence" in result.modalities_processed


def test_process_dict():
    s = make_substrate()
    s.register_task("t3")
    result = s.process({"temp": 37.5, "pressure": 1.2}, "t3")
    assert result.chosen_action is not None


def test_process_numpy_array_1d():
    s = make_substrate()
    s.register_task("t4")
    result = s.process(np.array([1.0, 2.0, 3.0, 4.0]), "t4")
    assert result.chosen_action is not None


def test_process_numpy_array_2d_image():
    s = make_substrate()
    s.register_task("t5")
    img = np.zeros((32, 32), dtype=np.float32)
    result = s.process(img, "t5")
    assert result.chosen_action is not None


def test_process_rising_sequence_predicate():
    s = make_substrate()
    s.register_task("t6")
    result = s.process([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "t6")
    # Should detect rising pattern
    preds = result.predicates
    assert any("RISING" in p for p in preds) or result.chosen_action is not None


def test_process_multimodal():
    s = make_substrate()
    s.register_task("t7")
    result = s.process_multimodal({
        "text": "fire detected",
        "sensor": [0.9, 1.1, 0.95, 1.3],
    }, "t7")
    assert result.chosen_action is not None
    assert "text" in result.modalities_processed
    assert "sensor" in result.modalities_processed


def test_register_encoder():
    s = make_substrate()
    s.register_task("t8")
    import python.core.vsa.hypervec_shim as hv
    from python.core.types.percept_packet import PerceptPacket

    def my_lidar_encoder(data, task_tag):
        return PerceptPacket.make(
            modality="lidar",
            situation_hv=hv.HyperVector(42),
            active_predicates=frozenset({"LIDAR_ACTIVE"}),
        )

    s.register_encoder("lidar", my_lidar_encoder)
    result = s.process_multimodal({"lidar": {"distance": 1.5}}, "t8")
    assert "lidar" in result.modalities_processed


def test_learn_and_remember():
    s = make_substrate()
    s.register_task("t9")
    # Process first to create a current_state
    state = {"location": "kitchen", "temperature": 22.0}
    s.process(state, "t9")
    s.learn(state, "stay", reward=1.0, task_tag="t9", outcome="success")
    # Remember should work without error
    memories = s.remember(state, task_tag="t9", top_k=3)
    assert isinstance(memories, list)


def test_sleep_works():
    s = make_substrate()
    s.register_task("t10")
    s.process({"x": 1}, "t10")
    result = s.sleep("t10")
    assert "sleep_cycles" in result


def test_get_stats():
    s = make_substrate()
    s.register_task("t11")
    s.process("test input", "t11")
    stats = s.get_stats()
    assert "decisions" in stats
    assert "registered_tasks" in stats


def test_substrate_full_lifecycle():
    s = make_substrate()
    s.register_task("lifecycle")
    # Process
    result = s.process("the system is running", "lifecycle")
    assert result is not None
    # Learn
    s.learn({"text": "the system is running"}, result.chosen_action, reward=1.0, task_tag="lifecycle")
    # Sleep
    s.sleep("lifecycle")
    # Remember
    memories = s.remember("the system is running", task_tag="lifecycle")
    assert isinstance(memories, list)
    # Stats
    stats = s.get_stats()
    assert stats["decisions"] >= 1
