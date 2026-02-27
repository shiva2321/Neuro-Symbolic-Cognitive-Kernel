"""Tests for KnowledgePack JSON serialization (Initiative 1 — V16)."""
import base64
import gzip
import json
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from python.core.integration.knowledge_pack import KnowledgePack, SCHEMA_VERSION, _hv_to_json, _json_to_hv
import python.core.vsa.hypervec_shim as hv_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pack() -> KnowledgePack:
    pack = KnowledgePack(name="test_pack")
    pack.add_concept("dog", {"type": "animal", "legs": 4})
    pack.add_concept("cat", {"type": "animal", "legs": 4})
    pack.add_relation("dog", "is_a", "animal")
    pack.add_causal_link("rain", "wet_ground", strength=0.9)
    return pack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_produces_json_not_pickle(tmp_path):
    pack = _make_pack()
    p = tmp_path / "test.kp"
    pack.save(str(p))
    # Must be readable as JSON after decompression
    with gzip.open(str(p), "rt", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["name"] == "test_pack"


def test_load_json_roundtrip_preserves_all_fields(tmp_path):
    pack = _make_pack()
    p = tmp_path / "test.kp"
    pack.save(str(p))
    loaded = KnowledgePack.load(str(p))
    assert loaded.name == "test_pack"
    assert len(loaded._concepts) == 2
    assert len(loaded._relations) == 1
    assert len(loaded._causal_links) == 1
    concept_names = [c[0] for c in loaded._concepts]
    assert "dog" in concept_names
    assert "cat" in concept_names


def test_hv_roundtrip_via_base64():
    hv = hv_mod.HyperVector(42)
    encoded = _hv_to_json(hv)
    assert encoded is not None
    assert isinstance(encoded, str)
    restored = _json_to_hv(encoded)
    assert restored is not None
    # Check bits match
    orig_bits = np.asarray(hv.bits, dtype=np.int8)
    rest_bits = np.asarray(restored.bits, dtype=np.int8)
    assert np.array_equal(orig_bits, rest_bits)


def test_hv_none_roundtrip():
    assert _hv_to_json(None) is None
    assert _json_to_hv(None) is None


def test_legacy_pickle_still_loads(tmp_path):
    """V14/V15 pickle files must still load with a DeprecationWarning."""
    pack = _make_pack()
    # Manually create a pickle file (old format)
    data = {
        "name": pack.name,
        "concepts": pack._concepts,
        "relations": pack._relations,
        "causal_links": pack._causal_links,
    }
    p = tmp_path / "legacy.kp"
    with gzip.open(str(p), "wb") as f:
        pickle.dump(data, f)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded = KnowledgePack.load(str(p))
        assert any(issubclass(x.category, DeprecationWarning) for x in w), \
            "Expected DeprecationWarning for pickle load"
    
    assert loaded.name == "test_pack"
    assert len(loaded._concepts) == 2


def test_schema_version_2_in_output(tmp_path):
    pack = _make_pack()
    p = tmp_path / "v2.kp"
    pack.save(str(p))
    with gzip.open(str(p), "rt", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == 2


def test_substrate_loads_json_packs_on_init(tmp_path):
    """NSCKSubstrate should load JSON packs without error."""
    from python.core.integration.knowledge_pack import KnowledgePack
    from python.core.integration.config import NSCKConfig
    pack = KnowledgePack(name="init_test")
    pack.add_concept("tree", {"type": "plant"})
    p = tmp_path / "init_test.kp"
    pack.save(str(p))
    
    cfg = NSCKConfig()
    cfg.knowledge_packs = [str(p)]
    # Import substrate here to avoid heavy import at module level
    from python.core.substrate import NSCKSubstrate
    sub = NSCKSubstrate(config=cfg)
    # Should have loaded the concept
    assert "tree" in sub._engine.semantic_memory.concept_graph


def test_no_arbitrary_code_on_malformed(tmp_path):
    """Invalid JSON should raise a clean exception, not execute code."""
    p = tmp_path / "bad.kp"
    with gzip.open(str(p), "wt", encoding="utf-8") as f:
        f.write("not valid json {{{")
    # Should fall through to pickle, which should also fail on non-pickle data
    with pytest.raises(Exception):
        KnowledgePack.load(str(p))
