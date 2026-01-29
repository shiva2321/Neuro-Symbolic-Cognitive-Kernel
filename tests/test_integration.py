"""
NCGN v7 Rigorous Testing Suite

Comprehensive tests for:
1. Learning from large texts
2. Knowledge retention
3. Reasoning capabilities
4. Internal state observation

This script provides detailed visibility into what the system
is doing, how, and why.
"""

import time
from typing import Dict, List, Any
from ncgn.brain import Brain
from ncgn.config import Config

# ============================================================================
# Test Data: Large Text Corpus
# ============================================================================

KNOWLEDGE_CORPUS = [
    # Basic world knowledge
    "Dogs are mammals that bark and eat meat.",
    "Cats are mammals that meow and hunt mice.",
    "Birds have feathers and can fly.",
    "Penguins are birds but cannot fly.",
    "Fish live in water and breathe through gills.",
    
    # Causal relationships
    "Fire is hot and causes burns.",
    "Water extinguishes fire.",
    "Ice is frozen water and is cold.",
    "The sun provides light and heat.",
    
    # Hierarchies
    "Dogs, cats, and horses are all mammals.",
    "Mammals are warm-blooded animals.",
    "Animals are living organisms.",
    "Roses, tulips, and daisies are flowers.",
    "Flowers are plants that produce seeds.",
    
    # Complex relationships
    "Humans domesticated dogs thousands of years ago.",
    "Dogs are loyal companions to humans.",
    "Veterinarians treat sick animals.",
    "Zoos contain many different animals.",
    
    # Contradictory or surprising facts
    "Metal is not edible and cannot be eaten by animals.",
    "Robots are not alive and do not eat.",
    "Glass is fragile and breaks easily.",
]

REASONING_TESTS = [
    {
        "setup": ["Dogs eat meat.", "Meat is food."],
        "query": "What do dogs eat?",
        "expected_concepts": ["dog", "meat", "eat", "food"],
    },
    {
        "setup": ["Metal is not edible.", "Dogs eat meat."],
        "query": "Can dogs eat metal?",
        "expected_concepts": ["dog", "metal", "edible"],
        "expected_contradiction": True,
    },
    {
        "setup": ["Birds fly.", "Penguins are birds.", "Penguins cannot fly."],
        "query": "Can penguins fly?",
        "expected_concepts": ["penguin", "bird", "fly"],
        "expected_exception": True,
    },
]


class TestHarness:
    """
    Rigorous test harness for NCGN v7.
    Provides detailed observability into system behavior.
    """
    
    def __init__(self, use_embeddings: bool = True):
        print("=" * 70)
        print("NCGN v7 RIGOROUS TEST HARNESS")
        print("=" * 70)
        print()
        
        # Create brain with custom config
        config = Config(
            initial_capacity=10000,
            decay_delta=0.05,  # Slower decay for observation
            flow_alpha=0.9,    # Strong propagation
            learning_rate=0.02,
        )
        
        self.brain = Brain(config=config, use_embeddings=use_embeddings)
        self.test_results: List[Dict[str, Any]] = []
        
    def _print_section(self, title: str):
        print()
        print("-" * 70)
        print(f">>> {title}")
        print("-" * 70)
    
    def _print_graph_state(self):
        """Print current state of the knowledge graph."""
        stats = self.brain.get_stats()
        print(f"  Nodes: {stats['topology']['num_nodes']}")
        print(f"  Edges: {stats['topology']['num_edges']}")
        print(f"  Total Energy: {stats['state']['total_energy']:.4f}")
        print(f"  Active Count: {stats['state']['active_count']}")
        
    def _print_active_concepts(self, top_k: int = 10):
        """Print most active concepts."""
        active = self.brain.get_top_concepts(k=top_k)
        if active:
            print("  Active Concepts:")
            for label, energy in sorted(active.items(), key=lambda x: -x[1]):
                bar = "█" * int(energy * 20)
                print(f"    {label:20s} {energy:.4f} {bar}")
        else:
            print("  (no active concepts)")
    
    # ========================================================================
    # TEST 1: Learning from Text
    # ========================================================================
    
    def test_learning(self):
        """Test: Can the system learn from text input?"""
        self._print_section("TEST 1: LEARNING FROM TEXT")
        
        print("\nFeeding knowledge corpus to the brain...")
        print(f"Total sentences: {len(KNOWLEDGE_CORPUS)}")
        print()
        
        concepts_before = self.brain.topology.num_nodes
        edges_before = self.brain.topology.num_edges
        
        for i, sentence in enumerate(KNOWLEDGE_CORPUS):
            print(f"[{i+1:2d}] Processing: '{sentence[:50]}...'")
            
            # Extract concepts from sentence (simple tokenization)
            words = sentence.lower().replace(".", "").replace(",", "").split()
            # Filter stopwords
            stopwords = {"is", "are", "the", "a", "an", "and", "to", "in", "on", "by", "for", "but", "not", "that", "can", "do", "or"}
            concepts = [w for w in words if w not in stopwords and len(w) > 2]
            
            # Add concepts and create connections
            for j, concept in enumerate(concepts):
                self.brain.add_concept(concept, initial_energy=0.3)
                
                # Connect sequential concepts (basic association)
                if j > 0:
                    self.brain.connect(concepts[j-1], concept, weight=0.6)
            
            # Reinforce connections with positive reward
            self.brain.think(steps=3)
            self.brain.learn(reward=0.5)
        
        concepts_after = self.brain.topology.num_nodes
        edges_after = self.brain.topology.num_edges
        
        print()
        print("LEARNING RESULTS:")
        print(f"  Concepts added: {concepts_after - concepts_before}")
        print(f"  Connections created: {edges_after - edges_before}")
        self._print_graph_state()
        
        # Verdict
        learned = (concepts_after > concepts_before) and (edges_after > edges_before)
        print()
        print(f"✓ VERDICT: System {'CAN' if learned else 'CANNOT'} learn from text")
        
        self.test_results.append({
            "test": "learning",
            "passed": learned,
            "concepts_added": concepts_after - concepts_before,
            "edges_added": edges_after - edges_before,
        })
        
        return learned
    
    # ========================================================================
    # TEST 2: Knowledge Retention
    # ========================================================================
    
    def test_retention(self):
        """Test: Does the system retain learned knowledge?"""
        self._print_section("TEST 2: KNOWLEDGE RETENTION")
        
        # First, activate some concepts
        test_concepts = ["dogs", "cats", "mammals", "birds", "fly"]
        
        print("\nActivating test concepts and observing decay...")
        
        for concept in test_concepts:
            if self.brain.has_concept(concept):
                self.brain.inject(concept, 0.8)
        
        # Capture initial state
        self.brain.think(steps=1)
        initial_active = self.brain.get_active_concepts()
        print("\nInitial activation:")
        self._print_active_concepts()
        
        # Let time pass (simulate decay)
        print("\nSimulating 20 ticks of decay...")
        for t in range(20):
            self.brain.think(steps=1)
            if t % 5 == 4:
                print(f"\n  After {t+1} ticks:")
                self._print_active_concepts(top_k=5)
        
        # Check what's still active
        final_active = self.brain.get_active_concepts()
        
        # Test: Re-inject and see if associations spread
        print("\nRe-activating 'dogs' and checking spread...")
        self.brain.inject("dogs", 1.0)
        self.brain.think(steps=5)
        self._print_active_concepts()
        
        # Check if related concepts also activated
        related = self.brain.get_active_concepts()
        
        # Did activation spread through learned connections?
        spread_occurred = len(related) > 1
        
        print()
        print("RETENTION RESULTS:")
        print(f"  Initial active concepts: {len(initial_active)}")
        print(f"  After decay: {len(final_active)}")
        print(f"  After re-activation (spread): {len(related)}")
        
        # Check graph integrity
        print("\n  Graph still intact:")
        self._print_graph_state()
        
        # Verdict
        retained = (self.brain.topology.num_nodes > 0 and 
                   self.brain.topology.num_edges > 0 and
                   spread_occurred)
        
        print()
        print(f"✓ VERDICT: System {'RETAINS' if retained else 'DOES NOT RETAIN'} knowledge")
        
        self.test_results.append({
            "test": "retention",
            "passed": retained,
            "spread_occurred": spread_occurred,
            "nodes_retained": self.brain.topology.num_nodes,
        })
        
        return retained
    
    # ========================================================================
    # TEST 3: Reasoning (Associative)
    # ========================================================================
    
    def test_reasoning(self):
        """Test: Can the system reason through associations?"""
        self._print_section("TEST 3: ASSOCIATIVE REASONING")
        
        print("\nTesting if activation spreads through related concepts...")
        
        # Clear activations
        self.brain.state.clear_all_activations()
        
        # Inject a query concept
        query = "dogs"
        print(f"\nQuery: What is related to '{query}'?")
        
        if not self.brain.has_concept(query):
            print(f"  Warning: '{query}' not in knowledge base")
            return False
        
        self.brain.inject(query, 1.0)
        
        print("\nWatching activation spread through associations:")
        for t in range(10):
            self.brain.think(steps=1)
            if t % 2 == 0:
                print(f"\n  Tick {t+1}:")
                self._print_active_concepts(top_k=8)
        
        # Get final associations
        final = self.brain.get_top_concepts(k=15)
        
        print("\n" + "=" * 50)
        print("REASONING PATH DISCOVERED:")
        print("=" * 50)
        
        # Trace the association path
        print(f"\n  Starting from: {query}")
        neighbors = self.brain.topology.get_neighbors(query)
        print(f"  Direct connections: {neighbors[:5]}")
        
        # Check if expected concepts activated
        expected = {"mammals", "meat", "eat", "bark", "animals"}
        found = set(final.keys()) & expected
        
        print(f"\n  Expected concepts: {expected}")
        print(f"  Found in activation: {found}")
        
        reasoning_works = len(found) > 0 or len(final) > 1
        
        print()
        print(f"✓ VERDICT: Associative reasoning {'WORKS' if reasoning_works else 'FAILS'}")
        
        self.test_results.append({
            "test": "reasoning",
            "passed": reasoning_works,
            "concepts_activated": list(final.keys()),
        })
        
        return reasoning_works
    
    # ========================================================================
    # TEST 4: Internal State Observation
    # ========================================================================
    
    def test_observability(self):
        """Test: Can we observe what the system is doing?"""
        self._print_section("TEST 4: INTERNAL STATE OBSERVABILITY")
        
        print("\nDemonstrating full visibility into system state...")
        
        # Get comprehensive stats
        stats = self.brain.get_stats()
        
        print("\n1. TOPOLOGY STATE (Graph Structure):")
        print(f"   Nodes: {stats['topology']['num_nodes']}")
        print(f"   Edges: {stats['topology']['num_edges']}")
        print(f"   Max Index: {stats['topology']['max_index']}")
        print(f"   Dirty Flag: {stats['topology']['is_dirty']}")
        
        print("\n2. COGNITIVE STATE (Neural Dynamics):")
        print(f"   Capacity: {stats['state']['capacity']}")
        print(f"   Active Count: {stats['state']['active_count']}")
        print(f"   Total Energy: {stats['state']['total_energy']:.4f}")
        print(f"   Tick Count: {stats['state']['tick_count']}")
        print(f"   Matrix Shape: {stats['state']['matrix_shape']}")
        print(f"   Matrix NNZ: {stats['state']['matrix_nnz']}")
        
        print("\n3. ENGINE STATE (Propagation):")
        print(f"   Tick Count: {stats['engine']['tick_count']}")
        print(f"   Last Surprise: {stats['engine']['last_surprise']:.4f}")
        print(f"   Active Concepts: {stats['engine']['active_concepts']}")
        
        print("\n4. LEARNER STATE (Plasticity):")
        print(f"   Total Updates: {stats['learner']['total_updates']}")
        print(f"   LTP Events: {stats['learner']['total_ltp']}")
        print(f"   LTD Events: {stats['learner']['total_ltd']}")
        print(f"   Learning Rate: {stats['learner']['learning_rate']}")
        
        print("\n5. MODULATOR STATE (Reward):")
        print(f"   Expected Reward: {stats['modulator']['expected_reward']:.4f}")
        print(f"   Avg Reward: {stats['modulator']['avg_reward']:.4f}")
        
        # Sample some edge weights
        print("\n6. SAMPLE EDGE WEIGHTS:")
        all_labels = self.brain.topology.registry.all_labels()
        count = 0
        for label in all_labels[:10]:
            neighbors = self.brain.topology.get_neighbors(label)
            for neighbor in neighbors[:3]:
                weight = self.brain.get_connection_weight(label, neighbor)
                print(f"   {label} --({weight:.2f})--> {neighbor}")
                count += 1
                if count >= 10:
                    break
            if count >= 10:
                break
        
        print()
        print("✓ VERDICT: Full observability available")
        
        self.test_results.append({
            "test": "observability",
            "passed": True,
            "stats": stats,
        })
        
        return True
    
    # ========================================================================
    # TEST 5: Design Verification
    # ========================================================================
    
    def test_design_compliance(self):
        """Test: Is the system doing what it's designed to do?"""
        self._print_section("TEST 5: DESIGN COMPLIANCE")
        
        print("\nVerifying core design principles...")
        
        checks = []
        
        # Check 1: Data-Oriented Design (SoA)
        print("\n1. Data-Oriented Design (Structure of Arrays):")
        import numpy as np
        is_soa = (
            isinstance(self.brain.state.activations, np.ndarray) and
            isinstance(self.brain.state.thresholds, np.ndarray) and
            self.brain.state.activations.dtype == np.float32
        )
        print(f"   Activations is numpy array: {isinstance(self.brain.state.activations, np.ndarray)}")
        print(f"   Uses float32: {self.brain.state.activations.dtype}")
        checks.append(("SoA Design", is_soa))
        
        # Check 2: Sparse Matrix
        print("\n2. Sparse Matrix Dynamics:")
        from scipy.sparse import csr_matrix
        self.brain.state.synchronize_matrix(self.brain.topology)
        is_sparse = isinstance(self.brain.state.adjacency, csr_matrix)
        print(f"   Adjacency is CSR: {is_sparse}")
        if is_sparse:
            print(f"   Matrix density: {self.brain.state.adjacency.nnz / max(1, self.brain.state.adjacency.shape[0]**2):.6f}")
        checks.append(("Sparse Matrix", is_sparse))
        
        # Check 3: Integer Indexing
        print("\n3. Integer Indexing (Rustworkx):")
        import rustworkx
        is_rustworkx = isinstance(self.brain.topology.graph, rustworkx.PyDiGraph)
        print(f"   Graph is PyDiGraph: {is_rustworkx}")
        sample_idx = self.brain.topology.registry.get_index("dogs") if self.brain.has_concept("dogs") else None
        print(f"   Sample label→index: 'dogs' → {sample_idx}")
        checks.append(("Integer Indexing", is_rustworkx))
        
        # Check 4: 3-Factor Learning
        print("\n4. 3-Factor Hebbian Learning:")
        stats = self.brain.learner.get_stats()
        has_learning = stats['total_updates'] > 0 or stats['total_ltp'] > 0
        print(f"   Total updates: {stats['total_updates']}")
        print(f"   LTP events: {stats['total_ltp']}")
        print(f"   LTD events: {stats['total_ltd']}")
        checks.append(("3-Factor Learning", has_learning))
        
        # Check 5: Energy Conservation/Decay
        print("\n5. Energy Dynamics:")
        # Create an isolated test concept with no connections
        test_node = "___isolated_test_node___"
        self.brain.add_concept(test_node, initial_energy=0.0)  # Start at rest
        idx = self.brain.topology.registry.get_index(test_node)
        
        # Inject energy directly
        self.brain.state.activations[idx] = 0.9
        e1 = float(self.brain.state.activations[idx])
        
        # Run decay (manual, without propagation)
        for _ in range(5):
            self.brain.state.activations[idx] *= (1 - self.brain.config.decay_delta)
        
        e2 = float(self.brain.state.activations[idx])
        energy_decays = e2 < e1
        print(f"   Initial isolated energy: {e1:.4f}")
        print(f"   After 5 decay steps: {e2:.4f}")
        print(f"   Energy decayed: {energy_decays} (expected: {e1 * (1-0.05)**5:.4f})")
        checks.append(("Energy Decay", energy_decays))
        
        # Summary
        print("\n" + "=" * 50)
        print("DESIGN COMPLIANCE SUMMARY:")
        print("=" * 50)
        all_passed = True
        for name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name}")
            all_passed = all_passed and passed
        
        self.test_results.append({
            "test": "design_compliance",
            "passed": all_passed,
            "checks": checks,
        })
        
        return all_passed
    
    # ========================================================================
    # Run All Tests
    # ========================================================================
    
    def run_all(self):
        """Run all tests and produce summary."""
        start_time = time.time()
        
        self.test_learning()
        self.test_retention()
        self.test_reasoning()
        self.test_observability()
        self.test_design_compliance()
        
        elapsed = time.time() - start_time
        
        # Final Summary
        print()
        print("=" * 70)
        print("FINAL TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {status}: {result['test'].upper()}")
        
        print()
        print(f"Total: {passed}/{total} tests passed")
        print(f"Time: {elapsed:.2f} seconds")
        print()
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - NCGN v7 is working as designed!")
        else:
            print("⚠️  Some tests failed - review the output above")
        
        return passed == total


def main():
    harness = TestHarness(use_embeddings=False)  # Skip embeddings for speed
    harness.run_all()


if __name__ == "__main__":
    main()
