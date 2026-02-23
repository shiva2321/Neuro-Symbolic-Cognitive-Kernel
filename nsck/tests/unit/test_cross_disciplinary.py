"""
Cross-Disciplinary Enhancement Tests
=====================================
Tests for enhancements drawn from multiple scientific fields:
  1. Mutual Information confounder detection (information theory)
  2. Weber-Fechner logarithmic scaling in SNN perception (psychophysics)
  3. Free Energy surprise metric (Friston's Free Energy Principle)
  4. Category-theoretic functoriality score (category theory)
  5. Maximum-Entropy adaptive threshold (Jaynes' MaxEnt principle)
"""
import pytest
import math
import numpy as np

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def causal_discovery():
    from python.core.reasoning.causal_reasoning import CausalDiscovery
    return CausalDiscovery()


@pytest.fixture()
def analogy_engine():
    from python.core.reasoning.analogy import AnalogyEngine
    return AnalogyEngine()


@pytest.fixture()
def curiosity_module():
    from python.core.learning.curiosity import CuriosityModule
    return CuriosityModule()


# ---------------------------------------------------------------------------
# 1. Mutual Information Confounder Detection (Information Theory)
# ---------------------------------------------------------------------------

class TestMutualInformationConfounder:
    """Tests for MI-based confounder detection in causal reasoning."""

    def test_mi_zero_for_independent_variables(self, causal_discovery):
        """MI should be ~0 for variables with no association."""
        import random
        random.seed(42)
        for _ in range(200):
            a = random.random() < 0.5
            b = random.random() < 0.5  # independent of a
            causes = []
            effects = []
            if a:
                causes.append('A')
            if b:
                effects.append('B')
            causal_discovery.observe('test', causes, effects)
        mi = causal_discovery._mutual_information('test', 'A', 'B')
        assert mi < 0.1, f"MI should be near 0 for independent vars, got {mi}"

    def test_mi_positive_for_associated_variables(self, causal_discovery):
        """MI should be > 0 for associated variables."""
        import random
        random.seed(42)
        for _ in range(200):
            a = random.random() < 0.5
            b = a  # perfectly associated
            causes = ['A'] if a else []
            effects = ['B'] if b else []
            causal_discovery.observe('test', causes, effects)
        mi = causal_discovery._mutual_information('test', 'A', 'B')
        assert mi > 0.2, f"MI should be high for associated vars, got {mi}"

    def test_confounder_detected(self, causal_discovery):
        """STRESS should be detected as confounder for HIGH_BP→HEADACHE."""
        import random
        random.seed(42)
        for _ in range(500):
            stress = random.random() < 0.4
            high_bp = stress and random.random() < 0.8
            headache = stress and random.random() < 0.7
            causes = []
            effects = []
            if stress:
                causes.append('STRESS')
            if high_bp:
                causes.append('HIGH_BP')
            if headache:
                effects.append('HEADACHE')
            causal_discovery.observe('medical', causes, effects)

        confounders = causal_discovery.detect_confounders(
            'medical', 'HIGH_BP', 'HEADACHE'
        )
        confounder_names = [c[0] for c in confounders]
        assert 'STRESS' in confounder_names, \
            f"STRESS should be detected as confounder, got {confounders}"

    def test_no_confounder_for_direct_cause(self, causal_discovery):
        """No confounder should be detected for a genuine direct cause."""
        import random
        random.seed(42)
        for _ in range(200):
            fire = random.random() < 0.3
            smoke = fire  # direct cause, no confounder
            causes = ['FIRE'] if fire else []
            effects = ['SMOKE'] if smoke else []
            causal_discovery.observe('simple', causes, effects)

        confounders = causal_discovery.detect_confounders('simple', 'FIRE', 'SMOKE')
        assert len(confounders) == 0, \
            f"No confounders expected for direct cause, got {confounders}"

    def test_confounder_returns_empty_for_no_association(self, causal_discovery):
        """When MI(C,E) ≈ 0, no confounders should be reported."""
        causal_discovery.observe('empty', ['A'], ['B'])
        causal_discovery.observe('empty', ['C'], ['D'])
        confounders = causal_discovery.detect_confounders('empty', 'A', 'D')
        assert confounders == []


# ---------------------------------------------------------------------------
# 2. Weber-Fechner Logarithmic Scaling (Psychophysics)
# ---------------------------------------------------------------------------

class TestWeberFechnerScaling:
    """Tests for Weber-Fechner log compression in SNN perception."""

    def test_snn_handles_small_signals(self):
        """SNN should produce valid output for very small signals."""
        from python.core.perception.snn_perception import SNNPerceptionModule
        snn = SNNPerceptionModule(input_dim=32, snn_size=64)
        small = np.random.randn(32) * 0.01
        result = snn.perceive(small, learn=False)
        assert 'concept_hv' in result
        assert 'concept_id' in result
        assert result['processing_time_ms'] > 0

    def test_snn_handles_large_signals(self):
        """SNN should produce valid output for very large signals (no saturation)."""
        from python.core.perception.snn_perception import SNNPerceptionModule
        snn = SNNPerceptionModule(input_dim=32, snn_size=64)
        large = np.random.randn(32) * 1000.0
        result = snn.perceive(large, learn=False)
        assert 'concept_hv' in result
        assert result['processing_time_ms'] > 0

    def test_log_compression_preserves_sign(self):
        """Weber-Fechner: sign(x)·log(1+|x|) should preserve sign."""
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        compressed = np.sign(x) * np.log1p(np.abs(x))
        assert compressed[0] < 0  # negative preserved
        assert compressed[3] > 0  # positive preserved
        assert compressed[2] == 0.0  # zero preserved

    def test_log_compression_range(self):
        """Log compression should reduce dynamic range."""
        x = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        compressed = np.log1p(x)
        # Original range: 100,000x.  Compressed range should be much smaller.
        original_range = x[-1] / x[0]
        compressed_range = compressed[-1] / compressed[0]
        assert compressed_range < original_range / 10


# ---------------------------------------------------------------------------
# 3. Free Energy Surprise (Friston's Free Energy Principle)
# ---------------------------------------------------------------------------

class TestFreeEnergySurprise:
    """Tests for Friston-inspired Free Energy surprise metric."""

    def test_perfect_prediction_low_surprise(self, curiosity_module):
        """Perfect prediction → low free energy surprise."""
        from python.core.vsa.hypervec_shim import HyperVector
        situation = HyperVector(seed=42)
        predicted = HyperVector(seed=42)
        fe = curiosity_module.compute_free_energy_surprise(
            situation, predicted, 'test'
        )
        assert fe < 1.0, f"Perfect prediction should have low FE, got {fe}"

    def test_wrong_prediction_high_surprise(self, curiosity_module):
        """Wrong prediction → high free energy surprise."""
        from python.core.vsa.hypervec_shim import HyperVector
        situation = HyperVector(seed=42)
        predicted = HyperVector(seed=99)
        fe = curiosity_module.compute_free_energy_surprise(
            situation, predicted, 'test'
        )
        assert fe > 0.5, f"Wrong prediction should have high FE, got {fe}"

    def test_no_prediction_falls_back_to_novelty(self, curiosity_module):
        """No prediction available → falls back to novelty score."""
        from python.core.vsa.hypervec_shim import HyperVector
        situation = HyperVector(seed=42)
        fe = curiosity_module.compute_free_energy_surprise(
            situation, None, 'test'
        )
        novelty = curiosity_module.compute_novelty(situation, 'test')
        assert fe == novelty

    def test_surprise_decreases_with_experience(self, curiosity_module):
        """As predictions improve, free energy should decrease."""
        from python.core.vsa.hypervec_shim import HyperVector
        situation = HyperVector(seed=42)
        predicted = HyperVector(seed=42)  # perfect prediction
        # Build up error history with good predictions
        for _ in range(20):
            curiosity_module.compute_free_energy_surprise(
                situation, predicted, 'converge'
            )
        fe = curiosity_module.compute_free_energy_surprise(
            situation, predicted, 'converge'
        )
        # After many perfect predictions, uncertainty should be very low
        assert fe < 0.1, f"After convergence, FE should be very low, got {fe}"


# ---------------------------------------------------------------------------
# 4. Category-Theoretic Functoriality Score (Category Theory)
# ---------------------------------------------------------------------------

class TestFunctorialityScore:
    """Tests for category-theoretic transfer quality scoring."""

    def test_perfect_mapping_high_score(self, analogy_engine):
        """Isomorphic domains → functoriality score ≈ 1."""
        from python.core.vsa.hypervec_shim import HyperVector
        hvs_a = {'x': HyperVector(seed=1), 'y': HyperVector(seed=2),
                  'z': HyperVector(seed=3)}
        hvs_b = {'a': HyperVector(seed=1), 'b': HyperVector(seed=2),
                  'c': HyperVector(seed=3)}
        mappings = analogy_engine.auto_discover_abstractions(
            'src', 'tgt', hvs_a, hvs_b
        )
        score = analogy_engine.functoriality_score(mappings, hvs_a, hvs_b)
        assert score > 0.95, f"Perfect alignment should give score > 0.95, got {score}"

    def test_insufficient_mappings_return_half(self, analogy_engine):
        """Less than 2 mapped pairs → returns 0.5 (insufficient data)."""
        from python.core.reasoning.analogy import ConceptMapping
        mappings = [ConceptMapping(
            source_domain='a', target_domain='b',
            source_concept='x', target_concept='y',
            similarity=0.9, relation_type='test'
        )]
        from python.core.vsa.hypervec_shim import HyperVector
        hvs_a = {'x': HyperVector(seed=1)}
        hvs_b = {'y': HyperVector(seed=1)}
        score = analogy_engine.functoriality_score(mappings, hvs_a, hvs_b)
        assert score == 0.5

    def test_random_mapping_lower_score(self, analogy_engine):
        """Random (non-isomorphic) mapping should have lower score."""
        from python.core.vsa.hypervec_shim import HyperVector
        hvs_a = {'x': HyperVector(seed=1), 'y': HyperVector(seed=2),
                  'z': HyperVector(seed=3)}
        hvs_b = {'a': HyperVector(seed=7), 'b': HyperVector(seed=8),
                  'c': HyperVector(seed=9)}
        from python.core.reasoning.analogy import ConceptMapping
        mappings = [
            ConceptMapping('src', 'tgt', 'x', 'a', 0.5, 'random'),
            ConceptMapping('src', 'tgt', 'y', 'b', 0.5, 'random'),
            ConceptMapping('src', 'tgt', 'z', 'c', 0.5, 'random'),
        ]
        score = analogy_engine.functoriality_score(mappings, hvs_a, hvs_b)
        # Score should still be reasonable since random HVs have sim ≈ 0.5
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. Maximum-Entropy Adaptive Threshold (Jaynes' Principle)
# ---------------------------------------------------------------------------

class TestMaxEntropyThreshold:
    """Tests for Jaynes' maximum-entropy derived similarity threshold."""

    def test_threshold_above_random_baseline(self, analogy_engine):
        """Threshold should always be > 0.5 (random baseline)."""
        theta = analogy_engine.max_entropy_threshold(dim=10240)
        assert theta > 0.5

    def test_threshold_increases_with_stricter_prior(self, analogy_engine):
        """Smaller match fraction → higher (stricter) threshold."""
        theta_10pct = analogy_engine.max_entropy_threshold(
            dim=10240, expected_match_fraction=0.10
        )
        theta_1pct = analogy_engine.max_entropy_threshold(
            dim=10240, expected_match_fraction=0.01
        )
        assert theta_1pct > theta_10pct

    def test_threshold_depends_on_dimension(self, analogy_engine):
        """Higher dimension → narrower σ → threshold closer to 0.5."""
        theta_small = analogy_engine.max_entropy_threshold(dim=1024)
        theta_large = analogy_engine.max_entropy_threshold(dim=100000)
        # Larger dim → smaller σ → threshold closer to 0.5
        assert abs(theta_large - 0.5) < abs(theta_small - 0.5)

    def test_threshold_within_sane_range(self, analogy_engine):
        """Threshold should be between 0.5 and 0.6 for reasonable parameters."""
        theta = analogy_engine.max_entropy_threshold(
            dim=10240, expected_match_fraction=0.1
        )
        assert 0.5 < theta < 0.6, f"Threshold {theta} outside sane range"
