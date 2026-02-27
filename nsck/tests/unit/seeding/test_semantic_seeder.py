"""Tests for SemanticSeeder (V16 Initiative 4)."""
import json
import tempfile
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from python.core.seeding.semantic_seeder import SemanticSeeder
from python.core.seeding.conceptnet_loader import ConceptNetLoader
from python.core.integration.config import NSCKConfig


_SMALL_CSV = "\n".join([
    "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]"
        "\t/r/IsA\t/c/en/dog\t/c/en/animal\t" + json.dumps({"weight": 3.0}),
    "/a/[/r/IsA/,/c/en/cat/,/c/en/animal/]"
        "\t/r/IsA\t/c/en/cat\t/c/en/animal\t" + json.dumps({"weight": 2.5}),
])


def _make_substrate():
    from python.core.substrate import NSCKSubstrate
    cfg = NSCKConfig()
    return NSCKSubstrate(config=cfg)


def test_post_seed_enrich_runs():
    """post_seed_enrich should not raise even if optional methods are absent."""
    substrate = _make_substrate()
    seeder = SemanticSeeder()
    seeder.post_seed_enrich(substrate)  # Should not raise


def test_seed_from_pack_injects_concepts(tmp_path):
    """seed_from_conceptnet_pack should inject concepts into substrate."""
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_SMALL_CSV, min_weight=2.0, from_string=True)
    pack_path = str(tmp_path / "test.kp")
    pack.save(pack_path)

    substrate = _make_substrate()
    seeder = SemanticSeeder()
    counts = seeder.seed_from_conceptnet_pack(substrate, pack_path)

    assert counts["concepts"] >= 2
    sem = substrate.engine.semantic_memory
    assert "dog" in sem.concept_hvs
    assert "animal" in sem.concept_hvs


def test_is_a_transitive_closure_after_enrich(tmp_path):
    """After enrich, is_a transitive relations should be computed if supported."""
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_SMALL_CSV, min_weight=2.0, from_string=True)
    pack_path = str(tmp_path / "test.kp")
    pack.save(pack_path)

    substrate = _make_substrate()
    seeder = SemanticSeeder()
    seeder.seed_from_conceptnet_pack(substrate, pack_path)
    # Should not raise
    seeder.post_seed_enrich(substrate)


def test_seeded_substrate_ingest_still_works(tmp_path):
    """Seeded substrate should still process new percepts correctly."""
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_SMALL_CSV, min_weight=2.0, from_string=True)
    pack_path = str(tmp_path / "test.kp")
    pack.save(pack_path)

    substrate = _make_substrate()
    seeder = SemanticSeeder()
    seeder.seed_from_conceptnet_pack(substrate, pack_path)

    # Try ingesting a new percept
    result = substrate.ingest({"type": "test", "content": "hello world"}, "test_task")
    assert result is not None


def test_nsck_config_seeded_preset_fields():
    cfg = NSCKConfig.seeded()
    assert getattr(cfg, 'enable_seeding', False) is True
    assert getattr(cfg, 'seed_conceptnet_pack', '') != ''
