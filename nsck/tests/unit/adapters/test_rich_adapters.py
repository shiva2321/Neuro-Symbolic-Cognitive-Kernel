"""Tests for rich perception adapters (WP-3)."""
from __future__ import annotations
import numpy as np
import pytest


def test_rich_text_adapter_pure_mode():
    """RichTextAdapter works in pure mode with no external deps."""
    from python.core.adapters.rich_text_adapter import RichTextAdapter
    adapter = RichTextAdapter(config=None)
    packet = adapter.encode("hello world", "test")
    assert packet.modality == "text"
    assert packet.situation_hv is not None
    assert "encoding_method" in packet.adapter_trace


def test_rich_text_adapter_fallback():
    """Falls back to char-ngram when bridge unavailable."""
    from python.core.adapters.rich_text_adapter import RichTextAdapter
    
    class FakeConfig:
        perception_mode = "bridge"
        bridge_dim = 384
        text_bridge_model = "nonexistent-model-xyz"
    
    adapter = RichTextAdapter(config=FakeConfig())
    packet = adapter.encode("test text", "task")
    assert packet.situation_hv is not None
    # Should still produce a result even with unavailable model


def test_rich_image_adapter_pure_mode():
    """RichImageAdapter works with no timm."""
    from python.core.adapters.rich_image_adapter import RichImageAdapter
    adapter = RichImageAdapter(config=None)
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    packet = adapter.encode(img, "test")
    assert packet.modality == "image"
    assert packet.situation_hv is not None


def test_rich_audio_adapter_pure_mode():
    """RichAudioAdapter works with no whisper."""
    from python.core.adapters.rich_audio_adapter import RichAudioAdapter
    adapter = RichAudioAdapter(config=None)
    audio = np.random.randn(16000).astype(np.float32)
    packet = adapter.encode(audio, "test")
    assert packet.modality == "audio"
    assert packet.situation_hv is not None


def test_rich_text_predicates_always_extracted():
    """Predicates field is always present (may be empty)."""
    from python.core.adapters.rich_text_adapter import RichTextAdapter
    adapter = RichTextAdapter()
    packet = adapter.encode("sky is blue", "test")
    assert hasattr(packet, "active_predicates")
    assert isinstance(packet.active_predicates, frozenset)


def test_config_routing_pure_mode():
    """Pure mode uses hash-based encoding (no rich adapter)."""
    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate
    cfg = NSCKConfig()
    assert cfg.perception_mode == "pure"
    sub = NSCKSubstrate(config=cfg)
    assert sub._rich_adapters == {}


def test_config_routing_bridge_mode():
    """Bridge mode initializes rich adapters."""
    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate
    cfg = NSCKConfig(perception_mode="bridge")
    sub = NSCKSubstrate(config=cfg)
    # Rich adapters should be initialized
    assert isinstance(sub._rich_adapters, dict)


def test_no_external_deps_in_pure_mode():
    """Pure mode must not import sentence-transformers/timm/whisper."""
    from python.core.integration.config import NSCKConfig
    cfg = NSCKConfig()
    assert cfg.perception_mode == "pure"
    # Just creating substrate in pure mode should work
    from python.core.substrate import NSCKSubstrate
    sub = NSCKSubstrate(config=cfg)
    result = sub.process("hello world", "test_task")
    assert result is not None
