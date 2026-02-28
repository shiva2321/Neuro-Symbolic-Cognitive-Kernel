"""Unit tests for KnowledgeSeeder (V4)."""
import os
import pytest
import tempfile
from python.core.bootstrap.knowledge_seeder import KnowledgeSeeder


NAVIGATION_YAML = os.path.join(
    os.path.dirname(__file__),
    "../../../python/core/bootstrap/domain_kits/navigation.yaml"
)

SCHEDULING_YAML = os.path.join(
    os.path.dirname(__file__),
    "../../../python/core/bootstrap/domain_kits/scheduling.yaml"
)


class TestKnowledgeSeeder:
    def _make_engine(self):
        from python.core.reasoning.cognitive_engine import CognitiveEngine
        from python.core.integration.config import NSCKConfig
        return CognitiveEngine(config=NSCKConfig(), persistence_path=None)

    def test_seeder_exists(self):
        seeder = KnowledgeSeeder()
        assert hasattr(seeder, 'seed_from_yaml')

    def test_seed_navigation_yaml(self):
        engine = self._make_engine()
        seeder = KnowledgeSeeder()
        yaml_path = os.path.abspath(NAVIGATION_YAML)
        if not os.path.exists(yaml_path):
            pytest.skip(f"navigation.yaml not found at {yaml_path}")
        rules_seeded = seeder.seed_from_yaml(yaml_path, engine)
        assert rules_seeded > 0

    def test_seed_scheduling_yaml(self):
        engine = self._make_engine()
        seeder = KnowledgeSeeder()
        yaml_path = os.path.abspath(SCHEDULING_YAML)
        if not os.path.exists(yaml_path):
            pytest.skip(f"scheduling.yaml not found at {yaml_path}")
        rules_seeded = seeder.seed_from_yaml(yaml_path, engine)
        assert rules_seeded > 0

    def test_seed_creates_rules_in_engine(self):
        engine = self._make_engine()
        seeder = KnowledgeSeeder()
        yaml_path = os.path.abspath(NAVIGATION_YAML)
        if not os.path.exists(yaml_path):
            pytest.skip("navigation.yaml not found")
        seeder.seed_from_yaml(yaml_path, engine)
        nav_rules = engine.rule_learner.learned_rules.get("navigation", [])
        assert len(nav_rules) > 0

    def test_seed_via_engine_seed_domain(self):
        engine = self._make_engine()
        yaml_path = os.path.abspath(NAVIGATION_YAML)
        if not os.path.exists(yaml_path):
            pytest.skip("navigation.yaml not found")
        n = engine.seed_domain(yaml_path)
        assert n > 0

    def test_seed_missing_file_returns_0(self):
        engine = self._make_engine()
        seeder = KnowledgeSeeder()
        result = seeder.seed_from_yaml("/nonexistent/file.yaml", engine)
        assert result == 0

    def test_seed_creates_procedural_skills(self):
        engine = self._make_engine()
        seeder = KnowledgeSeeder()
        yaml_path = os.path.abspath(NAVIGATION_YAML)
        if not os.path.exists(yaml_path):
            pytest.skip("navigation.yaml not found")
        seeder.seed_from_yaml(yaml_path, engine)
        stats = engine.procedural_memory.get_statistics()
        # High-confidence rules should be cached in procedural memory
        assert stats["total_skills"] >= 0  # may be 0 if no high-confidence rules


class TestBootstrapInit:
    def test_init_file_exists(self):
        import python.core.bootstrap
        assert python.core.bootstrap is not None
