"""Integration test: seeded substrate has more concepts than unseeded (V16)."""
import json
import tempfile
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from python.core.integration.config import NSCKConfig
from python.core.substrate import NSCKSubstrate
from python.core.seeding.conceptnet_loader import ConceptNetLoader
from python.core.seeding.semantic_seeder import SemanticSeeder


_CSV = "\n".join([
    "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]"
        "\t/r/IsA\t/c/en/dog\t/c/en/animal\t" + json.dumps({"weight": 3.0}),
    "/a/[/r/IsA/,/c/en/cat/,/c/en/animal/]"
        "\t/r/IsA\t/c/en/cat\t/c/en/animal\t" + json.dumps({"weight": 2.5}),
    "/a/[/r/IsA/,/c/en/fish/,/c/en/animal/]"
        "\t/r/IsA\t/c/en/fish\t/c/en/animal\t" + json.dumps({"weight": 2.0}),
    "/a/[/r/Causes/,/c/en/fire/,/c/en/heat/]"
        "\t/r/Causes\t/c/en/fire\t/c/en/heat\t" + json.dumps({"weight": 3.0}),
    "/a/[/r/UsedFor/,/c/en/knife/,/c/en/cutting/]"
        "\t/r/UsedFor\t/c/en/knife\t/c/en/cutting\t" + json.dumps({"weight": 2.0}),
])


def test_seeded_substrate_has_more_concepts_than_unseeded(tmp_path):
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_CSV, min_weight=2.0, from_string=True)
    pack_path = str(tmp_path / "cn.kp")
    pack.save(pack_path)

    unseeded = NSCKSubstrate(config=NSCKConfig())
    baseline = len(unseeded.engine.semantic_memory.concept_hvs)

    seeded = NSCKSubstrate(config=NSCKConfig())
    seeder = SemanticSeeder()
    seeder.seed_from_conceptnet_pack(seeded, pack_path)

    seeded_count = len(seeded.engine.semantic_memory.concept_hvs)
    assert seeded_count > baseline
