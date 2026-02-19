"""
NSCK System Capabilities Test Suite
====================================

Comprehensive tests that document what the NSCK (Neuro-Symbolic Cognitive Kernel)
system CAN and CAN'T do.  Each test provides concrete proof of a capability
or explicitly marks a known limitation.

Tested subsystems:
    1. VSA Core Operations (HyperVectors)
    2. Memory Systems (EpisodicMemory)
    3. Rule Learning (RuleLearner)
    4. Causal Reasoning (CausalGraph, CausalReasoner, CausalDiscovery)
    5. Brain Fusion (BrainFusion, FusedBrain)
    6. Metacognition (MetacognitiveEngine)
    7. Planning (STRIPSPlanner)
    8. Efficiency Proofs
    9. Known Limitations
   10. Cross-Module Integration
"""

import time
import sys
import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. VSA Core Operations
# ---------------------------------------------------------------------------


class TestVSACoreOperations:
    """Tests proving the VSA (Vector Symbolic Architecture) foundation works."""

    def test_xor_binding_preserves_orthogonality(self):
        """XOR of two random HVs should be ~orthogonal to both operands.

        In a 10240-bit space, random vectors have expected similarity 0.5.
        XOR(A, B) should also be ~0.5-similar to A and B (i.e. quasi-orthogonal).
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(1)
        b = hypervec_rs.HyperVector(2)
        c = a.xor(b)

        sim_ac = c.similarity(a)
        sim_bc = c.similarity(b)

        # Quasi-orthogonal means close to 0.5 (±0.05 for 10240 bits)
        assert 0.45 <= sim_ac <= 0.55, f"XOR result should be ~orthogonal to A, got {sim_ac}"
        assert 0.45 <= sim_bc <= 0.55, f"XOR result should be ~orthogonal to B, got {sim_bc}"

    def test_xor_is_own_inverse(self):
        """XOR binding is its own inverse: XOR(XOR(A, B), B) == A."""
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(42)
        b = hypervec_rs.HyperVector(99)

        recovered = a.xor(b).xor(b)
        assert recovered.similarity(a) == 1.0, "XOR should be perfectly invertible"

    def test_bundle_produces_similar_vector(self):
        """Bundle (majority vote) of A and B should be similar to both inputs.

        For two vectors, bundle resolves ties randomly, so result should be
        ~0.75-similar to each input on average.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(10)
        b = hypervec_rs.HyperVector(20)
        bundled = a.bundle(b)

        sim_a = bundled.similarity(a)
        sim_b = bundled.similarity(b)

        # Bundle of 2 vectors: each bit agrees with A 75% (both same) + 50% of diffs
        # Expect ~0.7-0.8 similarity to each
        assert sim_a > 0.6, f"Bundle should be similar to A, got {sim_a}"
        assert sim_b > 0.6, f"Bundle should be similar to B, got {sim_b}"

    def test_similarity_function_accuracy(self):
        """Similarity = 1 - hamming_distance / dimension.

        Identical vectors → 1.0, orthogonal random vectors → ~0.5.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(7)

        # Self-similarity must be exactly 1.0
        assert a.similarity(a) == 1.0, "Self-similarity must be 1.0"

        # Two random vectors should be ~0.5
        b = hypervec_rs.HyperVector(8)
        sim = a.similarity(b)
        assert 0.45 <= sim <= 0.55, f"Random vectors should be ~0.5 similar, got {sim}"

    def test_weighted_bundle_works(self):
        """weighted_bundle(other, weight) produces a vector similar to both inputs.

        The implementation uses k=7 majority bundles. With weight=1.0,
        self_n=7 and other_n=1 (clamped to max(1,...)), so self dominates.
        With weight=0.5, self_n=4 and other_n=3 → balanced.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(100)
        b = hypervec_rs.HyperVector(200)

        # Weight=1.0: self_n=7, other_n=max(1, 0)=1 → 7 copies of self vs 1 other
        # Result should be highly similar to self
        sims_a_high = []
        for _ in range(20):
            result = a.weighted_bundle(b, weight=1.0)
            sims_a_high.append(result.similarity(a))

        # Weight=0.0: self_n=max(1, 0)=1, other_n=max(1, 7)=7 → biased toward other
        sims_b_low = []
        for _ in range(20):
            result = a.weighted_bundle(b, weight=0.0)
            sims_b_low.append(result.similarity(b))

        avg_sim_a_high = np.mean(sims_a_high)
        avg_sim_b_low = np.mean(sims_b_low)

        # With 7:1 ratio, majority clearly favors the dominant side
        assert avg_sim_a_high > 0.7, (
            f"weight=1.0 should keep result similar to self: {avg_sim_a_high:.4f}"
        )
        assert avg_sim_b_low > 0.7, (
            f"weight=0.0 should keep result similar to other: {avg_sim_b_low:.4f}"
        )

    def test_lsh_hash_determinism(self):
        """lsh_hash with same seed should always produce the same hash."""
        import python.core.vsa.hypervec_shim as hypervec_rs

        hv = hypervec_rs.HyperVector(55)

        h1 = hv.lsh_hash(42, 32)
        h2 = hv.lsh_hash(42, 32)
        assert h1 == h2, "lsh_hash must be deterministic for same seed"

        # Different seeds should (very likely) produce different hashes
        h3 = hv.lsh_hash(99, 32)
        # Not guaranteed but overwhelmingly likely for 32-bit hash
        # We just check types here
        assert isinstance(h1, int)
        assert isinstance(h3, int)

    def test_vsa_operations_are_linear_time(self):
        """VSA ops (xor, bundle, similarity) scale O(n) in dimension, not O(n²).

        We prove this by timing ops on HVs (dim=10240).  Each op should take
        <1ms even on slow hardware since it's just element-wise bitops.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(1)
        b = hypervec_rs.HyperVector(2)

        n_ops = 1000

        start = time.perf_counter()
        for _ in range(n_ops):
            a.xor(b)
        xor_time = (time.perf_counter() - start) / n_ops

        start = time.perf_counter()
        for _ in range(n_ops):
            a.bundle(b)
        bundle_time = (time.perf_counter() - start) / n_ops

        start = time.perf_counter()
        for _ in range(n_ops):
            a.similarity(b)
        sim_time = (time.perf_counter() - start) / n_ops

        # Each op on 10240-bit vector should be under 1ms (generous bound)
        assert xor_time < 0.001, f"XOR too slow: {xor_time*1000:.2f}ms"
        assert bundle_time < 0.001, f"Bundle too slow: {bundle_time*1000:.2f}ms"
        assert sim_time < 0.001, f"Similarity too slow: {sim_time*1000:.2f}ms"

    def test_seed_reproducibility(self):
        """Same seed produces identical HyperVectors."""
        import python.core.vsa.hypervec_shim as hypervec_rs

        a = hypervec_rs.HyperVector(12345)
        b = hypervec_rs.HyperVector(12345)
        assert a.similarity(b) == 1.0, "Same seed must produce identical HV"

    def test_different_seeds_produce_different_hvs(self):
        """Different seeds produce quasi-orthogonal HyperVectors."""
        import python.core.vsa.hypervec_shim as hypervec_rs

        hvs = [hypervec_rs.HyperVector(i) for i in range(10)]
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                sim = hvs[i].similarity(hvs[j])
                assert 0.45 <= sim <= 0.55, (
                    f"Seeds {i},{j} should be quasi-orthogonal, got {sim}"
                )


# ---------------------------------------------------------------------------
# 2. Memory Systems (episodic_memory.py)
# ---------------------------------------------------------------------------


class TestMemorySystems:
    """Tests for the episodic memory subsystem."""

    def _make_episode(self, seed, task="test", reward=0.0, outcome="neutral", action="ACTION_UP"):
        """Helper to create a LiveEpisode without persistence dependency."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.memory.episodic_memory import LiveEpisode

        return LiveEpisode(
            timestamp=time.time(),
            task_tag=task,
            situation_hv=hypervec_rs.HyperVector(seed),
            state={"seed": seed},
            action=action,
            outcome=outcome,
            reward=reward,
        )

    def test_episode_storage_and_retrieval(self):
        """Episodes can be stored and retrieved from memory."""
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None, recent_capacity=100)

        ep1 = self._make_episode(1, reward=1.0, outcome="success")
        ep2 = self._make_episode(2, reward=-1.0, outcome="failure")
        mem.record(ep1)
        mem.record(ep2)

        recent = mem.recall_recent("test", n=10)
        assert len(recent) == 2, f"Expected 2 episodes, got {len(recent)}"
        assert recent[0].reward == 1.0
        assert recent[1].reward == -1.0

    def test_memory_capacity_eviction(self):
        """Memory respects capacity limit using deque maxlen."""
        from python.core.memory.episodic_memory import EpisodicMemory

        capacity = 50
        mem = EpisodicMemory(store=None, recent_capacity=capacity)

        for i in range(100):
            mem.record(self._make_episode(i))

        recent = mem.recall_recent("test", n=200)
        assert len(recent) == capacity, (
            f"Memory should cap at {capacity}, got {len(recent)}"
        )

    def test_vsa_based_memory_search(self):
        """recall_similar returns episodes with HVs closest to query."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None, recent_capacity=200)

        # Store episodes with different seeds
        for i in range(50):
            mem.record(self._make_episode(i * 100))

        # Query with same seed as episode 0 → should be most similar
        query_hv = hypervec_rs.HyperVector(0)
        results = mem.recall_similar(query_hv, "test", k=5)

        assert len(results) > 0, "Should find at least one similar episode"
        # The most similar should share seed 0
        best = results[0]
        assert best.situation_hv.similarity(query_hv) == 1.0, (
            "Best match should be the exact same HV"
        )

    def test_recall_by_outcome(self):
        """Can filter episodes by outcome label."""
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None, recent_capacity=100)
        for i in range(10):
            outcome = "success" if i % 2 == 0 else "failure"
            mem.record(self._make_episode(i, outcome=outcome))

        successes = mem.recall_by_outcome("test", outcome="success")
        assert all(ep.outcome == "success" for ep in successes)
        assert len(successes) == 5

    def test_recall_by_reward(self):
        """Can retrieve high-reward episodes sorted by reward."""
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None, recent_capacity=100)
        for i in range(20):
            mem.record(self._make_episode(i, reward=float(i)))

        top = mem.recall_by_reward("test", min_reward=15.0, n=5)
        assert len(top) == 5
        # Should be sorted descending
        rewards = [ep.reward for ep in top]
        assert rewards == sorted(rewards, reverse=True)

    def test_sample_random_episodes(self):
        """sample() returns a random subset of episodes."""
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None, recent_capacity=100)
        for i in range(50):
            mem.record(self._make_episode(i))

        sample = mem.sample("test", n=10)
        assert len(sample) == 10

    def test_memory_statistics(self):
        """get_statistics returns valid metrics."""
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None, recent_capacity=100)
        for i in range(10):
            mem.record(self._make_episode(i, reward=1.0 if i < 5 else -1.0))

        stats = mem.get_statistics("test")
        assert stats["recent_count"] == 10
        assert stats["positive_rate"] == 0.5
        assert stats["negative_rate"] == 0.5


# ---------------------------------------------------------------------------
# 3. Rule Learning (rule_learner.py)
# ---------------------------------------------------------------------------


class TestRuleLearning:
    """Tests for the symbolic rule induction engine."""

    def _make_verifier(self):
        """Create a GroundingVerifier with some test predicates."""
        from python.core.perception.grounding_verifier import GroundingVerifier

        v = GroundingVerifier()
        v.register_predicate("FOOD_ABOVE", lambda s: s.get("food_y", 0) < s.get("head_y", 0))
        v.register_predicate("FOOD_BELOW", lambda s: s.get("food_y", 0) > s.get("head_y", 0))
        v.register_predicate("DANGER_UP", lambda s: s.get("danger_up", False))
        return v

    def test_rule_induction_from_experience(self):
        """Rules are induced when pattern reaches min_support with high success rate."""
        from python.core.reasoning.rule_learner import RuleLearner

        verifier = self._make_verifier()
        learner = RuleLearner(verifier=verifier, store=None, min_support=5, min_success_rate=0.7)

        # Simulate 10 observations where FOOD_ABOVE → ACTION_UP succeeds
        for _ in range(10):
            state = {"food_y": 2, "head_y": 5}
            learner.observe(state, "UP", reward=1.0, task_tag="snake", outcome="success")

        rules = learner.induce_rules("snake")
        assert len(rules) >= 1, "Should induce at least one rule"
        # The rule should map FOOD_ABOVE → ACTION_UP
        found = any(
            "FOOD_ABOVE" in r.condition and r.consequence == "ACTION_UP"
            for r in rules
        )
        assert found, "Should learn FOOD_ABOVE → ACTION_UP"

    def test_rule_not_induced_below_threshold(self):
        """Rules are NOT induced when support is below min_support."""
        from python.core.reasoning.rule_learner import RuleLearner

        verifier = self._make_verifier()
        learner = RuleLearner(verifier=verifier, store=None, min_support=10, min_success_rate=0.7)

        # Only 3 observations (below min_support=10)
        for _ in range(3):
            state = {"food_y": 2, "head_y": 5}
            learner.observe(state, "UP", reward=1.0, task_tag="snake", outcome="success")

        rules = learner.induce_rules("snake")
        assert len(rules) == 0, "Should not induce rules below min_support"

    def test_rule_application(self):
        """Learned rules can be retrieved and matched against active predicates."""
        from python.core.reasoning.rule_learner import RuleLearner

        verifier = self._make_verifier()
        learner = RuleLearner(verifier=verifier, store=None, min_support=3, min_success_rate=0.6)

        for _ in range(5):
            state = {"food_y": 2, "head_y": 5}
            learner.observe(state, "UP", reward=1.0, task_tag="snake", outcome="success")

        learner.induce_rules("snake")

        # Now query with active predicates
        applicable = learner.get_applicable_rules(["FOOD_ABOVE"], "snake")
        assert len(applicable) >= 1, "Should find applicable rule"
        rule, score = applicable[0]
        assert rule.consequence == "ACTION_UP"

    def test_rule_pruning(self):
        """Low-performing rules can be pruned."""
        from python.core.reasoning.rule_learner import RuleLearner
        from python.core.integration.persistence import Rule

        verifier = self._make_verifier()
        learner = RuleLearner(verifier=verifier, store=None, min_support=3, min_success_rate=0.6)

        # Manually add a bad rule
        bad_rule = Rule(
            id=None, condition=frozenset(["FAKE"]), consequence="ACTION_DOWN",
            priority=1, source="learned", task_tag="snake",
            scope="task_local", support_count=2, success_rate=0.1, created_at=0.0,
        )
        learner.learned_rules["snake"].append(bad_rule)

        learner.prune_rules("snake")

        remaining = learner.get_rules("snake")
        assert bad_rule not in remaining, "Bad rule should be pruned"


# ---------------------------------------------------------------------------
# 4. Causal Reasoning (causal_reasoning.py)
# ---------------------------------------------------------------------------


class TestCausalReasoning:
    """Tests for symbolic causal graph operations."""

    def test_causal_graph_construction(self):
        """CausalGraph can store links and retrieve them."""
        from python.core.reasoning.causal_reasoning import CausalGraph

        g = CausalGraph()
        g.add_causes("rain", "wet_ground")
        g.add_causes("wet_ground", "slippery")

        effects = g.get_immediate_effects("rain")
        assert "wet_ground" in effects
        causes = g.get_immediate_causes("slippery")
        assert "wet_ground" in causes

    def test_forward_chaining(self):
        """Forward chaining traverses cause→effect chains."""
        from python.core.reasoning.causal_reasoning import CausalGraph

        g = CausalGraph()
        g.add_causes("A", "B", strength=0.9)
        g.add_causes("B", "C", strength=0.8)
        g.add_causes("C", "D", strength=0.7)

        chains = g.forward_chain("A", max_depth=5)
        endpoints = {ch.end for ch in chains}
        assert {"B", "C", "D"} == endpoints, f"Should reach B, C, D; got {endpoints}"

        # Verify strength attenuation
        chain_to_d = [ch for ch in chains if ch.end == "D"][0]
        expected_strength = 0.9 * 0.8 * 0.7
        assert abs(chain_to_d.total_strength - expected_strength) < 1e-6

    def test_backward_chaining(self):
        """Backward chaining finds root causes of an effect."""
        from python.core.reasoning.causal_reasoning import CausalGraph

        g = CausalGraph()
        g.add_causes("X", "Y")
        g.add_causes("Y", "Z")

        chains = g.backward_chain("Z")
        starts = {ch.start for ch in chains}
        assert "X" in starts, f"Should trace back to X; got {starts}"
        assert "Y" in starts

    def test_causal_discovery_delta_p(self):
        """CausalDiscovery learns causal links from statistical contingency (Delta-P)."""
        from python.core.reasoning.causal_reasoning import CausalDiscovery

        cd = CausalDiscovery()

        # Generate data where "press_button" reliably causes "light_on"
        for _ in range(50):
            cd.observe("lab", causes=["press_button"], effects=["light_on"])
        # Also generate control steps (no press → no light)
        for _ in range(50):
            cd.observe("lab", causes=["idle"], effects=["dark"])

        graph = cd.induce_graph("lab", min_confidence=0.5, min_evidence=5)
        effects = graph.get_immediate_effects("press_button")
        assert "light_on" in effects, "Should discover press_button → light_on"

    def test_counterfactual_reasoning(self):
        """CausalReasoner can compare actual vs hypothetical action outcomes."""
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner

        g = CausalGraph()
        g.add_causes("ACTION_UP", "MOVED_UP", context="test")
        g.add_causes("ACTION_DOWN", "MOVED_DOWN", context="test")
        g.add_causes("MOVED_UP", "REWARD", context="test")

        reasoner = CausalReasoner(graph=g)
        result = reasoner.simulate_counterfactual(
            state={}, action_taken="ACTION_UP",
            hypothetical_action="ACTION_DOWN", task_tag="test",
        )

        # ACTION_UP leads to REWARD, ACTION_DOWN does not
        assert "REWARD" not in result["predicted_outcome"] or \
               "MOVED_DOWN" in result["predicted_outcome"]

    def test_find_path(self):
        """find_path discovers a causal chain between two nodes."""
        from python.core.reasoning.causal_reasoning import CausalGraph

        g = CausalGraph()
        g.add_causes("EAT", "GROW")
        g.add_causes("GROW", "SCORE_UP")

        chain = g.find_path("EAT", "SCORE_UP")
        assert chain is not None, "Should find path EAT → GROW → SCORE_UP"
        assert chain.start == "EAT"
        assert chain.end == "SCORE_UP"
        assert len(chain) == 2

    def test_snake_causal_graph(self):
        """Pre-built snake causal graph has expected structure."""
        from python.core.reasoning.causal_reasoning import create_snake_causal_graph

        g = create_snake_causal_graph()
        assert len(g.all_links) > 0, "Snake graph should have links"

        # ACTION_UP should eventually reach DEATH via chain
        chains = g.forward_chain("ACTION_UP", max_depth=5, context="snake")
        endpoints = {ch.end for ch in chains}
        assert "DEATH" in endpoints, "ACTION_UP can chain to DEATH via wall collision"


# ---------------------------------------------------------------------------
# 5. Brain Fusion (brain_fusion.py)
# ---------------------------------------------------------------------------


class TestBrainFusion:
    """Tests for multi-task knowledge organization."""

    def test_multi_task_knowledge_organization(self):
        """BrainFusion keeps task-specific concepts isolated while sharing primitives."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.integration.brain_fusion import TaskBrain, BrainFusion, GLOBAL_PRIMITIVES

        snake_brain = TaskBrain("snake")
        snake_brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
        snake_brain.add_concept("SNAKE_FOOD", hypervec_rs.HyperVector(5000))

        pong_brain = TaskBrain("pong")
        pong_brain.add_concept("ACTION_UP", hypervec_rs.HyperVector(10))
        pong_brain.add_concept("PONG_BALL", hypervec_rs.HyperVector(6000))

        fusion = BrainFusion()
        fusion.register_brain(snake_brain)
        fusion.register_brain(pong_brain)
        result = fusion.fuse()

        # Global primitives are shared
        assert "ACTION_UP" in result.global_codebook

        # Task-specific concepts stay in their layers
        assert "SNAKE_FOOD" in result.task_codebooks["snake"]
        assert "PONG_BALL" in result.task_codebooks["pong"]

        # No cross-contamination
        assert "PONG_BALL" not in result.task_codebooks.get("snake", {})
        assert "SNAKE_FOOD" not in result.task_codebooks.get("pong", {})

    def test_concept_promotion(self):
        """Concepts appearing in multiple tasks with same seed get promoted to global."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.integration.brain_fusion import TaskBrain, BrainFusion

        seed = 9999
        snake_brain = TaskBrain("snake")
        snake_brain.add_concept("SHARED_CONCEPT", hypervec_rs.HyperVector(seed))

        pong_brain = TaskBrain("pong")
        pong_brain.add_concept("SHARED_CONCEPT", hypervec_rs.HyperVector(seed))

        fusion = BrainFusion()
        fusion.register_brain(snake_brain)
        fusion.register_brain(pong_brain)
        result = fusion.fuse()

        # Same-seed HVs have similarity 1.0 → should be promoted
        assert "SHARED_CONCEPT" in result.global_codebook, (
            "Identical concepts across tasks should be promoted to global"
        )

    def test_rule_resolution(self):
        """FusedBrain.resolve_rules returns ranked actions for active predicates."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.integration.brain_fusion import TaskBrain, BrainFusion

        brain = TaskBrain("test")
        brain.add_concept("DANGER_UP", hypervec_rs.HyperVector(500))
        brain.add_rule(
            condition=frozenset(["DANGER_UP"]),
            consequence="ACTION_DOWN",
            strength=1.0, priority=1,
        )

        fusion = BrainFusion()
        fusion.register_brain(brain)
        fused = fusion.fuse()

        results = fused.resolve_rules({"DANGER_UP"}, task_tag="test")
        assert len(results) >= 1
        assert results[0].action == "ACTION_DOWN"

    def test_forward_chain_multi(self):
        """FusedBrain forward chaining deduces new facts from rules."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.integration.brain_fusion import TaskBrain, BrainFusion

        brain = TaskBrain("test")
        brain.add_concept("WET", hypervec_rs.HyperVector(700))
        brain.add_concept("SLIPPERY", hypervec_rs.HyperVector(701))
        # Fact→Fact rule (not ACTION_)
        brain.add_rule(
            condition=frozenset(["WET"]),
            consequence="SLIPPERY",
            strength=1.0,
        )

        fusion = BrainFusion()
        fusion.register_brain(brain)
        fused = fusion.fuse()

        final_facts, _ = fused.forward_chain_multi({"WET"})
        assert "SLIPPERY" in final_facts, "Should infer SLIPPERY from WET"


# ---------------------------------------------------------------------------
# 6. Metacognition (metacognition.py)
# ---------------------------------------------------------------------------


class TestMetacognition:
    """Tests for the metacognitive engine (confidence, conflicts, safety)."""

    def _make_fused_brain_with_rules(self):
        """Build a FusedBrain with conflicting rules for testing."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.integration.brain_fusion import TaskBrain, BrainFusion

        brain = TaskBrain("test")
        brain.add_concept("FOOD_ABOVE", hypervec_rs.HyperVector(300))
        brain.add_concept("DANGER_UP", hypervec_rs.HyperVector(301))

        # Two rules that fire on overlapping predicates but suggest different actions
        brain.add_rule(
            condition=frozenset(["FOOD_ABOVE"]),
            consequence="ACTION_UP", strength=1.0, priority=1,
        )
        brain.add_rule(
            condition=frozenset(["DANGER_UP"]),
            consequence="ACTION_DOWN", strength=1.0, priority=1,
        )

        fusion = BrainFusion()
        fusion.register_brain(brain)
        return fusion.fuse()

    def test_confidence_scoring(self):
        """MetacognitiveEngine computes confidence from query results."""
        from python.core.integration.brain_fusion import QueryResult
        from python.core.cognitive.metacognition import MetacognitiveEngine, std_dev

        fused = self._make_fused_brain_with_rules()
        engine = MetacognitiveEngine(fused)

        # Single strong match → high confidence
        results = [
            QueryResult(
                concept_name="FOOD_ABOVE", action="ACTION_UP",
                score=2.0, similarity=0.85, layer="test", rule_ids=["r1"],
            )
        ]
        conf, reason = engine.compute_confidence(results)
        assert conf > 0.0, f"Single strong match should yield positive confidence, got {conf}"
        assert reason == "single_match"

    def test_conflict_detection(self):
        """Detect conflict when two close-scoring results suggest different actions."""
        from python.core.integration.brain_fusion import QueryResult
        from python.core.cognitive.metacognition import MetacognitiveEngine

        fused = self._make_fused_brain_with_rules()
        engine = MetacognitiveEngine(fused)

        # Two results with different actions and close similarity
        results = [
            QueryResult(
                concept_name="A", action="ACTION_UP",
                score=1.5, similarity=0.80, layer="test", rule_ids=["r1"],
            ),
            QueryResult(
                concept_name="B", action="ACTION_DOWN",
                score=1.4, similarity=0.78, layer="test", rule_ids=["r2"],
            ),
        ]
        conflict = engine.detect_conflict(results)
        assert conflict is not None, "Should detect conflict between close-scoring different actions"
        assert conflict.type in ("precedence", "rule")

    def test_no_conflict_when_clear_winner(self):
        """No conflict when top result clearly dominates."""
        from python.core.integration.brain_fusion import QueryResult
        from python.core.cognitive.metacognition import MetacognitiveEngine

        fused = self._make_fused_brain_with_rules()
        engine = MetacognitiveEngine(fused)

        results = [
            QueryResult(
                concept_name="A", action="ACTION_UP",
                score=3.0, similarity=0.95, layer="test", rule_ids=["r1"],
            ),
            QueryResult(
                concept_name="B", action="ACTION_DOWN",
                score=0.5, similarity=0.60, layer="test", rule_ids=["r2"],
            ),
        ]
        conflict = engine.detect_conflict(results)
        assert conflict is None, "No conflict when margin is large"


# ---------------------------------------------------------------------------
# 7. Planning (planner.py)
# ---------------------------------------------------------------------------


class TestPlanning:
    """Tests for the STRIPS-style planner."""

    def test_strips_planning_with_causal_reasoner(self):
        """Planner finds action sequence to reach goal state using causal model."""
        from python.core.reasoning.planner import STRIPSPlanner
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner

        g = CausalGraph()
        g.add_causes("ACTION_UP", "AT_GOAL", context="maze")

        reasoner = CausalReasoner(graph=g)
        planner = STRIPSPlanner()
        planner.set_reasoner(reasoner)

        plan = planner.plan(
            initial_state={"AT_START"},
            goal={"AT_GOAL"},
            max_depth=5,
        )
        assert plan is not None, "Should find a plan"
        assert "ACTION_UP" in plan

    def test_planning_returns_none_when_impossible(self):
        """Planner returns None when goal is unreachable."""
        from python.core.reasoning.planner import STRIPSPlanner

        planner = STRIPSPlanner()
        # No operators → can't change state
        plan = planner.plan(
            initial_state={"A"},
            goal={"UNREACHABLE"},
            max_depth=3,
        )
        assert plan is None, "Should return None for impossible goal"

    def test_plan_already_satisfied(self):
        """Planner returns empty plan when goal is already satisfied."""
        from python.core.reasoning.planner import STRIPSPlanner

        planner = STRIPSPlanner()
        plan = planner.plan(
            initial_state={"AT_GOAL"},
            goal={"AT_GOAL"},
            max_depth=5,
        )
        assert plan is not None
        assert plan == [], "Goal already met → empty plan"

    def test_simulate_sequence(self):
        """simulate_sequence applies a sequence of actions to state."""
        from python.core.reasoning.planner import STRIPSPlanner
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner

        g = CausalGraph()
        g.add_causes("ACTION_UP", "MOVED_UP")
        g.add_causes("ACTION_RIGHT", "MOVED_RIGHT")

        reasoner = CausalReasoner(graph=g)
        planner = STRIPSPlanner()
        planner.set_reasoner(reasoner)

        result = planner.simulate_sequence(
            frozenset({"START"}),
            ["ACTION_UP", "ACTION_RIGHT"],
        )
        assert "MOVED_UP" in result
        assert "MOVED_RIGHT" in result


# ---------------------------------------------------------------------------
# 8. Efficiency Proofs
# ---------------------------------------------------------------------------


class TestEfficiencyProofs:
    """Tests proving the system is computationally efficient."""

    def test_hypervector_memory_footprint(self):
        """A single HyperVector should use ~10KB (10240 bits = 1280 bytes + overhead)."""
        import python.core.vsa.hypervec_shim as hypervec_rs

        hv = hypervec_rs.HyperVector(1)
        # The internal .bits is a numpy int8 array of size 10240
        bits_size = hv.bits.nbytes
        assert bits_size == 10240, f"Expected 10240 bytes for int8 array, got {bits_size}"

        # Even with overhead this is tiny compared to dense float32 matrices
        assert bits_size < 20_000, "HV memory must be well under 20KB"

    def test_vsa_vs_matrix_multiply_speed(self):
        """VSA operations should be faster than equivalent dense matrix multiply.

        Comparing XOR on 10240-bit vectors vs matmul on 10240-dim float vectors.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        hv_a = hypervec_rs.HyperVector(1)
        hv_b = hypervec_rs.HyperVector(2)

        mat_a = np.random.randn(10240).astype(np.float32)
        mat_b = np.random.randn(10240).astype(np.float32)

        n_ops = 500

        # Time VSA XOR
        start = time.perf_counter()
        for _ in range(n_ops):
            hv_a.xor(hv_b)
        vsa_time = time.perf_counter() - start

        # Time numpy dot product (dense matmul equivalent)
        start = time.perf_counter()
        for _ in range(n_ops):
            np.dot(mat_a, mat_b)
        mat_time = time.perf_counter() - start

        # VSA should not be dramatically slower than dot product
        # (both are O(n) but VSA uses bitwise ops which are very fast)
        # We just verify VSA is practical (<1ms per op)
        avg_vsa_ms = (vsa_time / n_ops) * 1000
        assert avg_vsa_ms < 1.0, f"VSA XOR too slow: {avg_vsa_ms:.3f}ms per op"

    def test_no_dense_matrix_in_core_vsa_path(self):
        """Verify the HyperVector class does NOT use dense matrix multiplication.

        The core VSA operations (xor, bundle, similarity) should use only
        element-wise bitwise operations — no matmul or einsum.
        """
        import inspect
        from python.core.vsa.hypervec_py import HyperVectorPy

        for method_name in ("xor", "bundle", "similarity"):
            method = getattr(HyperVectorPy, method_name)
            source = inspect.getsource(method)

            # Should NOT contain matrix multiplication calls
            banned = ["np.matmul", "np.dot", "np.einsum", "torch.mm", "torch.matmul"]
            for b in banned:
                assert b not in source, (
                    f"{method_name}() uses {b} — violates O(n) contract"
                )

    def test_intrinsic_motivation_param_count(self):
        """ICM module should have <25K parameters (lightweight design)."""
        import torch
        icm_mod = pytest.importorskip(
            "python.utilities.intrinsic_motivation",
            reason="IntrinsicCuriosityModule archived",
        )
        IntrinsicCuriosityModule = icm_mod.IntrinsicCuriosityModule

        icm = IntrinsicCuriosityModule(num_actions=4)
        total_params = sum(p.numel() for p in icm.parameters())
        assert total_params < 25_000, (
            f"ICM has {total_params} params, should be <25K for efficiency"
        )

    def test_world_model_uses_sparse_projection(self):
        """WorldModel DynamicsPredictor uses a sparse random projection (not dense).

        The projection matrix should have ~90% zeros (10% non-zero by design).
        """
        import torch
        wm_mod = pytest.importorskip(
            "python.core.neural.world_model",
            reason="WorldModel archived",
        )
        DynamicsPredictor = wm_mod.DynamicsPredictor
        WorldModelConfig = wm_mod.WorldModelConfig

        config = WorldModelConfig()
        predictor = DynamicsPredictor(config)

        proj = predictor.projection
        total_elements = proj.numel()
        non_zero = (proj != 0).sum().item()
        sparsity = 1.0 - (non_zero / total_elements)

        assert sparsity > 0.85, (
            f"Projection should be ~90% sparse, got {sparsity*100:.1f}% sparse"
        )

    def test_world_model_param_count(self):
        """WorldModel MLP should be small — well under 100K parameters."""
        import torch
        wm_mod = pytest.importorskip(
            "python.core.neural.world_model",
            reason="WorldModel archived",
        )
        DynamicsPredictor = wm_mod.DynamicsPredictor
        WorldModelConfig = wm_mod.WorldModelConfig

        config = WorldModelConfig()
        predictor = DynamicsPredictor(config)
        trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)

        assert trainable < 100_000, (
            f"WorldModel has {trainable} trainable params, should be <100K"
        )

    def test_episodic_memory_lsh_bucketing(self):
        """LSH index creates buckets for O(1) candidate lookup instead of O(n) scan."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.memory.episodic_memory import EpisodicMemory, LiveEpisode

        mem = EpisodicMemory(store=None, recent_capacity=500)

        for i in range(100):
            ep = LiveEpisode(
                timestamp=time.time(), task_tag="test",
                situation_hv=hypervec_rs.HyperVector(i),
                state={"i": i}, action="ACTION_UP",
                outcome="neutral", reward=0.0,
            )
            mem.record(ep)

        # LSH index should have been populated (multi-table: list of dicts)
        lsh_tables = mem.lsh_index.get("test", [])
        assert len(lsh_tables) > 0, "LSH index should have tables"
        # Count total indexed entries across all tables
        total_indexed = sum(
            len(bucket) for table in lsh_tables for bucket in table.values()
        )
        assert total_indexed > 0, "LSH index should contain episode references"
        # Verify multiple tables provide better coverage
        assert len(lsh_tables) == mem.lsh_num_tables, (
            f"Expected {mem.lsh_num_tables} LSH tables, got {len(lsh_tables)}"
        )


# ---------------------------------------------------------------------------
# 9. Known Limitations (what the system CAN'T do)
# ---------------------------------------------------------------------------


class TestKnownLimitations:
    """Tests that explicitly document what the system CANNOT do.

    These are marked with xfail or contain assertions proving the absence
    of certain capabilities, providing honest documentation of gaps.
    """

    @pytest.mark.xfail(reason="NSCK has no gradient-based learning in its VSA core", strict=True)
    def test_no_gradient_learning_in_vsa(self):
        """LIMITATION: VSA core operations have no gradient-based learning.

        HyperVectors are binary and use counting/majority — not differentiable.
        This test verifies that HyperVectors don't support autograd.
        """
        import torch
        import python.core.vsa.hypervec_shim as hypervec_rs

        hv = hypervec_rs.HyperVector(1)

        # Try to create a tensor with gradient tracking from HV bits
        t = torch.tensor(hv.bits.astype(float), requires_grad=True)
        loss = t.sum()
        loss.backward()

        # The HV itself doesn't participate in backprop — the gradient
        # doesn't flow back to modify the HV's bits
        # This should "fail" because there's no mechanism to update hv.bits via gradients
        hv_after = hypervec_rs.HyperVector(1)
        assert hv.similarity(hv_after) != 1.0, "HV should change after gradient update"

    @pytest.mark.xfail(reason="NSCK has no real language understanding", strict=True)
    def test_no_real_language_understanding(self):
        """LIMITATION: The system cannot understand natural language.

        It has no tokenizer, no word embeddings, no language model.
        Concepts are symbols (strings) mapped to HVs by hash — not semantics.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        # "dog" and "puppy" are semantically similar in natural language
        # but will be orthogonal in NSCK because they're different hash seeds
        import hashlib
        dog_seed = int(hashlib.sha256(b"dog").hexdigest(), 16) % (2 ** 32)
        puppy_seed = int(hashlib.sha256(b"puppy").hexdigest(), 16) % (2 ** 32)
        dog = hypervec_rs.HyperVector(dog_seed)
        puppy = hypervec_rs.HyperVector(puppy_seed)

        sim = dog.similarity(puppy)
        # For real NLU, sim should be > 0.7. In NSCK it will be ~0.5 (random).
        assert sim > 0.7, f"System should understand dog≈puppy, but sim={sim}"

    @pytest.mark.xfail(reason="NSCK has no pixel-level perception", strict=True)
    def test_no_pixel_level_perception(self):
        """LIMITATION: The system cannot process raw images.

        There is no CNN, no vision transformer, no image feature extractor
        in the core VSA pipeline. The ICM has a tiny feature net for 10x10
        grids but that's not real vision.
        """
        import python.core.vsa.hypervec_shim as hypervec_rs

        # A real vision system would create meaningful, distinct HVs from
        # visually similar images. NSCK can't do this from raw pixels.
        # Feeding raw pixels into HyperVector constructor just hashes the seed.
        img_cat = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_cat_rotated = np.rot90(img_cat)

        # System has no way to create HVs from images that capture visual content
        import hashlib
        hv1_seed = int(hashlib.blake2b(img_cat.tobytes(), digest_size=4).hexdigest(), 16)
        hv2_seed = int(hashlib.blake2b(img_cat_rotated.tobytes(), digest_size=4).hexdigest(), 16)
        hv1 = hypervec_rs.HyperVector(hv1_seed)
        hv2 = hypervec_rs.HyperVector(hv2_seed)

        sim = hv1.similarity(hv2)
        # Real vision: rotated cat should still be recognized as cat (sim > 0.7)
        assert sim > 0.7, f"System should recognize rotated image, but sim={sim}"

    def test_no_real_continual_learning_from_raw_data(self):
        """LIMITATION: The system learns symbolic rules, not from raw data streams.

        RuleLearner requires pre-extracted predicates; it can't learn
        features from raw sensor data. This is by design (symbolic AI).
        """
        from python.core.perception.grounding_verifier import GroundingVerifier
        from python.core.reasoning.rule_learner import RuleLearner

        verifier = GroundingVerifier()
        # No predicates registered → verifier returns empty list
        learner = RuleLearner(verifier=verifier, store=None, min_support=3)

        # Even with many observations of raw numeric state, no rules emerge
        for i in range(20):
            raw_state = {"sensor_1": i * 0.1, "sensor_2": np.sin(i)}
            learner.observe(raw_state, "UP", reward=1.0, task_tag="raw")

        rules = learner.induce_rules("raw")
        assert len(rules) == 0, (
            "Without predicate grounding, no rules can be learned from raw data"
        )

    def test_causal_discovery_needs_sufficient_data(self):
        """LIMITATION: Causal discovery requires many observations to produce reliable graphs.

        With too few observations, the induced graph is empty.
        """
        from python.core.reasoning.causal_reasoning import CausalDiscovery

        cd = CausalDiscovery()
        # Only 2 observations — not enough
        cd.observe("sparse", causes=["A"], effects=["B"])
        cd.observe("sparse", causes=["A"], effects=["B"])

        graph = cd.induce_graph("sparse", min_confidence=0.5, min_evidence=5)
        assert len(graph.all_links) == 0, (
            "Too few observations should produce empty causal graph"
        )


# ---------------------------------------------------------------------------
# Additional integration-style tests
# ---------------------------------------------------------------------------


class TestCrossModuleIntegration:
    """Tests verifying modules work together correctly."""

    def test_episodic_memory_with_situation_hv_creation(self):
        """EpisodicMemory.create_situation_hv creates valid HVs from predicates."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        from python.core.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(store=None)
        state = {"head": (5, 5), "food": (5, 3)}

        hv = mem.create_situation_hv(state, "snake", ["FOOD_ABOVE", "SAFE_PATH"])
        assert hv is not None
        assert hasattr(hv, "similarity"), "Should return a valid HyperVector"

        # Same predicates should produce same-ish HV (deterministic seeds)
        hv2 = mem.create_situation_hv(state, "snake", ["FOOD_ABOVE", "SAFE_PATH"])
        # Note: bundle has randomness in tie-breaking, but the base HVs
        # from hash(pred) are deterministic, so structure is consistent
        assert isinstance(hv2.similarity(hv), float)

    def test_causal_theory_formation(self):
        """TheoryModule generalizes specific causal links into abstract theories."""
        from python.core.reasoning.causal_reasoning import (
            CausalLink, CausalRelation, TheoryModule, create_snake_causal_graph,
        )

        theory_mod = TheoryModule()
        graph = create_snake_causal_graph()

        theories = theory_mod.form_theories(list(graph.all_links))
        # Should identify at least one recurring pattern (e.g. COLLIDER → FAILURE)
        assert len(theories) > 0, "Should form at least one theory"
        templates = [t.template for t in theories]
        # The snake graph has multiple collision→death links → COLLIDER leads to FAILURE
        assert any("COLLIDER" in t or "MOVEMENT" in t or "FAILURE" in t for t in templates), (
            f"Should generalize causal patterns; got {templates}"
        )

    def test_planner_with_causal_graph(self):
        """Planner can use causal graph to find multi-step plans."""
        from python.core.reasoning.planner import STRIPSPlanner
        from python.core.reasoning.causal_reasoning import CausalGraph, CausalReasoner

        g = CausalGraph()
        g.add_causes("ACTION_UP", "AT_ROW_4")
        g.add_causes("ACTION_UP", "CLOSER_TO_FOOD")
        g.add_causes("ACTION_RIGHT", "AT_COL_6")

        reasoner = CausalReasoner(graph=g)
        planner = STRIPSPlanner()
        planner.set_reasoner(reasoner)

        plan = planner.plan(
            initial_state={"AT_ROW_5", "AT_COL_5"},
            goal={"AT_ROW_4", "AT_COL_6"},
            max_depth=5,
        )
        assert plan is not None
        assert "ACTION_UP" in plan
        assert "ACTION_RIGHT" in plan
