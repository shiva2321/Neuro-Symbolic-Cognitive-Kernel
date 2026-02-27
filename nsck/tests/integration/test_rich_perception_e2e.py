"""End-to-end tests for rich perception pipeline (WP-3)."""
from __future__ import annotations
import numpy as np
import pytest


def test_rich_perception_text_e2e():
    """Rich perception text → decide → explain."""
    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate
    
    cfg = NSCKConfig(perception_mode="bridge")
    sub = NSCKSubstrate(config=cfg)
    sub.register_task("test_rich")
    
    result = sub.ingest("The cat sat on the mat", "test_rich")
    assert result is not None
    assert result.chosen_action is not None
    assert isinstance(result.explanation, str)


def test_pure_mode_e2e():
    """Pure mode works end-to-end with no extra deps."""
    from python.core.integration.config import NSCKConfig
    from python.core.substrate import NSCKSubstrate
    
    cfg = NSCKConfig()  # pure mode default
    sub = NSCKSubstrate(config=cfg)
    sub.register_task("test_pure")
    
    result = sub.ingest("hello world", "test_pure")
    assert result is not None


def test_for_scale_factory():
    """for_scale() produces valid config."""
    from python.core.integration.config import NSCKConfig
    
    cfg_small = NSCKConfig.for_scale(100)
    assert cfg_small.memory_capacity >= 200
    
    cfg_large = NSCKConfig.for_scale(10_000)
    assert cfg_large.enable_hnsw_index is True
