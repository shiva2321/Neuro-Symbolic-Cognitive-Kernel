"""Tests for ConceptNetLoader (V16 Initiative 4)."""
import json
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from python.core.seeding.conceptnet_loader import ConceptNetLoader, RELATION_MAP, _en_concept


# Minimal CSV string (TSV format as ConceptNet uses)
_MINIMAL_CSV = "\n".join([
    "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]\t/r/IsA\t/c/en/dog\t/c/en/animal\t" +
        json.dumps({"weight": 3.0}),
    "/a/[/r/Causes/,/c/en/fire/,/c/en/heat/]"  +
        "\t/r/Causes\t/c/en/fire\t/c/en/heat\t" + json.dumps({"weight": 4.0}),
    "/a/[/r/UsedFor/,/c/en/knife/,/c/en/cutting/]" +
        "\t/r/UsedFor\t/c/en/knife\t/c/en/cutting\t" + json.dumps({"weight": 2.5}),
    # Non-English — should be filtered
    "/a/[/r/IsA/,/c/fr/chien/,/c/fr/animal/]" +
        "\t/r/IsA\t/c/fr/chien\t/c/fr/animal\t" + json.dumps({"weight": 3.0}),
    # Below min_weight — should be filtered
    "/a/[/r/IsA/,/c/en/cat/,/c/en/animal/]" +
        "\t/r/IsA\t/c/en/cat\t/c/en/animal\t" + json.dumps({"weight": 0.5}),
])


def test_conceptnet_loader_instantiates():
    loader = ConceptNetLoader()
    assert loader is not None


def test_relation_mapping_correct():
    assert RELATION_MAP["IsA"] == "is_a"
    assert RELATION_MAP["Causes"] == "causes"
    assert RELATION_MAP["CapableOf"] == "capable_of"
    assert RELATION_MAP["UsedFor"] == "used_for"
    assert RELATION_MAP["HasPart"] == "has_part"
    assert RELATION_MAP["AtLocation"] == "at_location"


def test_causal_link_strength_from_weight():
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_MINIMAL_CSV, min_weight=2.0, from_string=True)
    # Should have causal link for fire→heat (weight=4.0, strength=min(1.0, 4.0/5.0)=0.8)
    causal = {(c, e): s for c, e, s in pack._causal_links}
    assert ("fire", "heat") in causal
    assert abs(causal[("fire", "heat")] - 0.8) < 1e-6


def test_load_from_minimal_csv_string():
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_MINIMAL_CSV, min_weight=2.0, from_string=True)
    concept_names = {c[0] for c in pack._concepts}
    assert "dog" in concept_names
    assert "animal" in concept_names
    assert "fire" in concept_names
    assert "heat" in concept_names


def test_english_only_filter():
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_MINIMAL_CSV, min_weight=2.0, from_string=True)
    concept_names = {c[0] for c in pack._concepts}
    # French concepts should NOT appear
    assert "chien" not in concept_names


def test_min_weight_filter():
    loader = ConceptNetLoader()
    pack = loader.load_from_csv(_MINIMAL_CSV, min_weight=2.0, from_string=True)
    concept_names = {c[0] for c in pack._concepts}
    # cat/animal from low-weight row should NOT appear
    # (cat only appears in the 0.5-weight row, not in any other row here)
    # Note: "animal" does appear in the dog/animal row (weight=3.0), so it's ok
    assert "cat" not in concept_names


def test_en_concept_extraction():
    assert _en_concept("/c/en/dog") == "dog"
    assert _en_concept("/c/en/happy_person") == "happy person"
    assert _en_concept("/c/fr/chien") is None
    assert _en_concept("/r/IsA") is None
