"""
Test Suite: Brain Versioning & Export System
=============================================

Validates checkpointing, export/import functionality for NSCK brain state.
Tests .nsck file format, merge capabilities, and data integrity.

Part of NSCK V2 substrate transformation (Task 8).
"""

import sys
import os
import pytest
import time
import json
import tempfile
import zipfile
from pathlib import Path

# Add nsck-demo/python to path

from python.core.integration.persistence import BrainStore, Rule, Concept, Episode, BrainVersion


@pytest.fixture
def temp_brain():
    """Create a temporary brain for testing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "test_brain.db")
        brain = BrainStore(db_path)
        yield brain
        brain.close()


@pytest.fixture
def populated_brain(temp_brain):
    """Create a brain with sample data."""
    # Add some rules
    for i in range(5):
        rule = Rule(
            id=None,
            condition=frozenset([f"pred_{i}"]),
            consequence=f"action_{i}",
            priority=i+1,
            support_count=i*2,
            confidence=0.5 + i*0.1
        )
        temp_brain.save_rule(rule)
    
    # Add some concepts
    for i in range(3):
        concept = Concept(
            id=None,
            name=f"concept_{i}",
            hv_bytes=bytes([i]*10),
            concept_type="TEST",
            created_at=time.time()
        )
        temp_brain.save_concept(concept)
    
    # Add some episodes
    for i in range(10):
        episode = Episode(
            id=None,
            timestamp=time.time(),
            task_tag="test_task",
            situation_hv_bytes=bytes([i]*10),
            state_sketch={"key": f"value_{i}"},
            action=f"action_{i}",
            outcome="success",
            reward=float(i),
            impact_score=0.5
        )
        temp_brain.record_episode(episode)
    
    temp_brain.flush_episodes()
    return temp_brain


def test_create_checkpoint(populated_brain):
    """Test creating version checkpoints."""
    print("\n📌 TEST 1: Create Checkpoint")
    print("=" * 60)
    
    version_id = populated_brain.create_checkpoint(
        version_tag="v1.0",
        description="Initial test checkpoint"
    )
    
    print(f"✅ Checkpoint created with ID: {version_id}")
    assert version_id > 0
    
    # Verify version was recorded
    version = populated_brain.get_version("v1.0")
    assert version is not None
    assert version.version_tag == "v1.0"
    assert version.description == "Initial test checkpoint"
    assert version.rules_count == 5
    assert version.concepts_count == 3
    assert version.episodes_count == 10
    
    print(f"✅ Version metadata:")
    print(f"   Rules: {version.rules_count}")
    print(f"   Concepts: {version.concepts_count}")
    print(f"   Episodes: {version.episodes_count}")


def test_list_versions(populated_brain):
    """Test listing all checkpoints."""
    print("\n📋 TEST 2: List Versions")
    print("=" * 60)
    
    # Create multiple checkpoints
    populated_brain.create_checkpoint("v1.0", "First version")
    time.sleep(0.1)  # Ensure different timestamps
    populated_brain.create_checkpoint("v1.1", "Second version")
    time.sleep(0.1)
    populated_brain.create_checkpoint("v2.0", "Major update")
    
    versions = populated_brain.list_versions()
    
    print(f"✅ Found {len(versions)} versions")
    assert len(versions) == 3
    
    # Verify ordering (newest first)
    assert versions[0].version_tag == "v2.0"
    assert versions[1].version_tag == "v1.1"
    assert versions[2].version_tag == "v1.0"
    
    for v in versions:
        print(f"   📌 {v.version_tag}: {v.description}")


def test_export_brain_basic(populated_brain):
    """Test basic brain export to .nsck file."""
    print("\n📦 TEST 3: Export Brain (Basic)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "exported_brain.nsck"
        
        # Create checkpoint and export
        populated_brain.create_checkpoint("export_test", "Test export")
        populated_brain.export_brain(str(export_path), version_tag="export_test")
        
        assert export_path.exists()
        print(f"✅ Brain exported to: {export_path.name}")
        print(f"   File size: {export_path.stat().st_size} bytes")
        
        # Verify .nsck structure (it's a ZIP file)
        with zipfile.ZipFile(export_path, 'r') as zf:
            files = zf.namelist()
            assert "brain.db" in files
            assert "metadata.json" in files
            print(f"✅ Archive contains: {', '.join(files)}")
            
            # Read and verify metadata
            metadata = json.loads(zf.read("metadata.json"))
            assert metadata["version_tag"] == "export_test"
            assert metadata["description"] == "Test export"
            assert metadata["rules_count"] == 5
            assert metadata["concepts_count"] == 3
            assert metadata["episodes_count"] == 10
            print(f"✅ Metadata valid:")
            print(f"   Version: {metadata['version_tag']}")
            print(f"   Description: {metadata['description']}")


def test_export_without_version(populated_brain):
    """Test exporting brain without explicit version tag."""
    print("\n📦 TEST 4: Export Brain (Auto-version)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "auto_export"  # No .nsck extension
        
        populated_brain.export_brain(str(export_path))  # No version_tag
        
        # Should add .nsck extension automatically
        final_path = export_path.with_suffix('.nsck')
        assert final_path.exists()
        print(f"✅ Brain exported with auto-extension: {final_path.name}")
        
        # Verify auto-generated version tag
        with zipfile.ZipFile(final_path, 'r') as zf:
            metadata = json.loads(zf.read("metadata.json"))
            assert metadata["version_tag"].startswith("exported_")
            assert metadata["description"] == "Exported brain snapshot"
            print(f"✅ Auto-generated version: {metadata['version_tag']}")


def test_import_brain_replace(populated_brain):
    """Test importing brain with replace mode."""
    print("\n📥 TEST 5: Import Brain (Replace Mode)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export populated brain
        export_path = Path(tmpdir) / "source.nsck"
        populated_brain.export_brain(str(export_path))
        print(f"✅ Exported source brain")
        
        # Create new empty brain
        target_db = Path(tmpdir) / "target_brain.db"
        target_brain = BrainStore(str(target_db))
        
        # Verify target is empty
        assert len(target_brain.load_rules()) == 0
        assert len(target_brain.load_concepts()) == 0
        print(f"✅ Target brain initially empty")
        
        # Import (replace mode)
        target_brain.import_brain(str(export_path), merge=False)
        
        # Verify data was imported
        rules = target_brain.load_rules()
        concepts = target_brain.load_concepts()
        episode_count = target_brain.count_episodes()
        
        assert len(rules) == 5
        assert len(concepts) == 3
        assert episode_count == 10
        
        print(f"✅ Data imported successfully:")
        print(f"   Rules: {len(rules)}")
        print(f"   Concepts: {len(concepts)}")
        print(f"   Episodes: {episode_count}")
        
        target_brain.close()


def test_import_brain_merge(temp_brain):
    """Test importing brain with merge mode."""
    print("\n🔀 TEST 6: Import Brain (Merge Mode)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create brain A with some data
        brain_a_db = Path(tmpdir) / "brain_a.db"
        brain_a = BrainStore(str(brain_a_db))
        
        brain_a.save_rule(Rule(
            id=None,
            condition=frozenset(["pred_a"]),
            consequence="action_a",
            priority=1
        ))
        brain_a.save_concept(Concept(
            id=None,
            name="concept_a",
            hv_bytes=bytes([1]*10),
            concept_type="TEST"
        ))
        
        export_a = Path(tmpdir) / "brain_a.nsck"
        brain_a.create_checkpoint("brain_a_v1", "Brain A data")
        brain_a.export_brain(str(export_a), version_tag="brain_a_v1")
        brain_a.close()
        print(f"✅ Created Brain A (1 rule, 1 concept)")
        
        # Create brain B with different data
        brain_b_db = Path(tmpdir) / "brain_b.db"
        brain_b = BrainStore(str(brain_b_db))
        
        brain_b.save_rule(Rule(
            id=None,
            condition=frozenset(["pred_b"]),
            consequence="action_b",
            priority=1
        ))
        brain_b.save_concept(Concept(
            id=None,
            name="concept_b",
            hv_bytes=bytes([2]*10),
            concept_type="TEST"
        ))
        
        print(f"✅ Created Brain B (1 rule, 1 concept)")
        
        # Merge Brain A into Brain B
        brain_b.import_brain(str(export_a), merge=True)
        
        # Verify Brain B now has data from both
        rules = brain_b.load_rules()
        concepts = brain_b.load_concepts()
        
        assert len(rules) == 2
        assert len(concepts) == 2
        
        rule_names = [r.consequence for r in rules]
        concept_names = [c.name for c in concepts]
        
        assert "action_a" in rule_names
        assert "action_b" in rule_names
        assert "concept_a" in concept_names
        assert "concept_b" in concept_names
        
        print(f"✅ Merge successful:")
        print(f"   Total rules: {len(rules)} (action_a, action_b)")
        print(f"   Total concepts: {len(concepts)} (concept_a, concept_b)")
        
        # Verify merge was recorded in versions
        versions = brain_b.list_versions()
        assert len(versions) > 0
        assert versions[0].version_tag.startswith("merged_")
        print(f"✅ Merge recorded as: {versions[0].version_tag}")
        
        brain_b.close()


def test_import_nonexistent_file(temp_brain):
    """Test importing from nonexistent file raises error."""
    print("\n❌ TEST 7: Import Nonexistent File")
    print("=" * 60)
    
    with pytest.raises(FileNotFoundError):
        temp_brain.import_brain("nonexistent.nsck")
    
    print("✅ Correctly raises FileNotFoundError")


def test_duplicate_checkpoint_tag(populated_brain):
    """Test creating checkpoint with duplicate tag fails."""
    print("\n🔁 TEST 8: Duplicate Checkpoint Tag")
    print("=" * 60)
    
    # Create first checkpoint
    populated_brain.create_checkpoint("v1.0", "First")
    print("✅ Created first checkpoint: v1.0")
    
    # Try to create duplicate (should raise IntegrityError)
    raised = False
    try:
        populated_brain.create_checkpoint("v1.0", "Duplicate")
    except Exception:
        raised = True
    
    assert raised, "Should have raised an exception for duplicate tag"
    print("✅ Correctly rejects duplicate version tag")


def test_data_integrity_after_export_import(populated_brain):
    """Test that data remains identical after export/import cycle."""
    print("\n✅ TEST 9: Data Integrity (Export/Import Cycle)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get original data
        original_rules = populated_brain.load_rules()
        original_concepts = populated_brain.load_concepts()
        original_episode_count = populated_brain.count_episodes()
        
        print(f"Original state:")
        print(f"   Rules: {len(original_rules)}")
        print(f"   Concepts: {len(original_concepts)}")
        print(f"   Episodes: {original_episode_count}")
        
        # Export
        export_path = Path(tmpdir) / "integrity_test.nsck"
        populated_brain.export_brain(str(export_path))
        
        # Import into new brain
        target_db = Path(tmpdir) / "restored.db"
        restored_brain = BrainStore(str(target_db))
        restored_brain.import_brain(str(export_path), merge=False)
        
        # Compare data
        restored_rules = restored_brain.load_rules()
        restored_concepts = restored_brain.load_concepts()
        restored_episode_count = restored_brain.count_episodes()
        
        assert len(restored_rules) == len(original_rules)
        assert len(restored_concepts) == len(original_concepts)
        assert restored_episode_count == original_episode_count
        
        # Verify rule content
        for orig, rest in zip(original_rules, restored_rules):
            assert orig.condition == rest.condition
            assert orig.consequence == rest.consequence
            assert orig.confidence == rest.confidence
        
        # Verify concept content
        for orig, rest in zip(original_concepts, restored_concepts):
            assert orig.name == rest.name
            assert orig.hv_bytes == rest.hv_bytes
        
        print(f"✅ Data integrity preserved:")
        print(f"   Rules match: {len(restored_rules)}/{len(original_rules)}")
        print(f"   Concepts match: {len(restored_concepts)}/{len(original_concepts)}")
        print(f"   Episodes match: {restored_episode_count}/{original_episode_count}")
        
        restored_brain.close()


if __name__ == "__main__":
    print("=" * 60)
    print("BRAIN VERSIONING & EXPORT TEST SUITE")
    print("NSCK V2 Substrate - Task 8")
    print("=" * 60)
    
    pytest.main([__file__, "-v", "-s"])
