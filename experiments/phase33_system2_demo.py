"""
Phase 3.3 Demonstration: System 2 Reasoning

This experiment demonstrates System 2 (slow, deliberate reasoning):

1. System 1 (spikes): Fast, reactive, automatic
2. System 1.5 (control): Monitors, detects conflicts/uncertainty
3. System 2 (reasoning): Explicit, rule-based, slow

When System 1.5 detects:
- High conflict → System 2 resolves with reasoning
- Low confidence → System 2 reasons about uncertainty
- Learnable patterns → System 2 consolidates to rules

This shows how a brain transitions from reactive to deliberate reasoning.
"""

from core.system2 import System2, RuleType


def run_system2_demonstration():
    """Demonstrate System 2 reasoning capabilities."""
    print("=" * 80)
    print("PHASE 3.3 DEMONSTRATION: System 2 Slow Reasoning")
    print("=" * 80)
    print()

    # Initialize System 2
    print("Initializing System 2 (slow reasoning layer)...")
    system2 = System2()
    print("✓ System 2 initialized")
    print()

    # Phase 1: Detect and resolve conflicts
    print("-" * 80)
    print("PHASE 1: Conflict Resolution")
    print("-" * 80)
    print()

    print("Scenario 1: Ambiguous pattern detected")
    print("  System 1.5 reports: Conflict=0.8, Confidence=0.3")
    print("  Available actions: [freeze_learning, continue_learning]")
    print()

    patterns = ["unusual_pattern_1", "unusual_pattern_2"]
    actions = ["freeze_learning", "continue_learning"]

    chosen = system2.handle_conflict(
        patterns=patterns,
        available_actions=actions
    )

    print(f"  System 2 chooses: {chosen}")
    print()

    print("Scenario 2: Second conflict")
    print("  System 1.5 reports: Conflict=0.7, Confidence=0.4")
    print("  Available actions: [slow_mode, normal_mode]")
    print()

    patterns2 = ["pattern_a", "pattern_b"]
    actions2 = ["slow_mode", "normal_mode"]

    chosen2 = system2.handle_conflict(
        patterns=patterns2,
        available_actions=actions2
    )

    print(f"  System 2 chooses: {chosen2}")
    print()

    # Phase 2: Learn explicit rules
    print("-" * 80)
    print("PHASE 2: Explicit Rule Learning")
    print("-" * 80)
    print()

    print("Learning rules from experience...")
    print()

    # Learn rule 1
    print("Rule 1: IF high-novelty pattern THEN activate learning")
    rid1 = system2.learn_rule(
        condition="High novelty detected",
        action="Activate online learning",
        confidence=0.9
    )
    print(f"  ✓ Learned (Rule ID: {rid1})")
    print()

    # Learn rule 2
    print("Rule 2: IF low-confidence output THEN invoke reasoning")
    rid2 = system2.learn_rule(
        condition="Output confidence < 0.5",
        action="Invoke System 2 reasoning",
        confidence=0.85
    )
    print(f"  ✓ Learned (Rule ID: {rid2})")
    print()

    # Learn rule 3
    print("Rule 3: IF conflict detected THEN freeze plasticity")
    rid3 = system2.learn_rule(
        condition="High conflict + Low confidence",
        action="Freeze region plasticity",
        confidence=0.75
    )
    print(f"  ✓ Learned (Rule ID: {rid3})")
    print()

    # Phase 3: Store facts
    print("-" * 80)
    print("PHASE 3: Explicit Fact Storage")
    print("-" * 80)
    print()

    print("Storing facts from observations...")
    print()

    facts_to_store = [
        "Novel patterns appear in first 10% of training",
        "Learning rate should be 0.01 for stable convergence",
        "Region freezing prevents catastrophic forgetting",
        "Novelty threshold 0.5 works well for this domain",
    ]

    for i, fact_text in enumerate(facts_to_store, 1):
        fid = system2.store_fact(fact_text, confidence=0.8)
        print(f"Fact {i}: {fact_text}")
        print(f"  ✓ Stored (Fact ID: {fid})")

    print()

    # Phase 4: Memory consolidation
    print("-" * 80)
    print("PHASE 4: Memory Consolidation (Sleep/Replay)")
    print("-" * 80)
    print()

    print("Consolidating memories from training episodes...")
    print()

    memories = [
        {
            "pattern": "Pattern A (high novelty)",
            "outcome": "Learning gate opened, weights updated",
            "confidence": 0.9
        },
        {
            "pattern": "Pattern B (familiar)",
            "outcome": "Learning gate closed, weights unchanged",
            "confidence": 0.85
        },
        {
            "pattern": "Pattern C (conflicting signals)",
            "outcome": "System 2 invoked for reasoning",
            "confidence": 0.7
        },
    ]

    for i, memory in enumerate(memories, 1):
        success = system2.consolidate_memory(memory)
        status = "✓ Consolidated" if success else "✗ Rejected"
        print(f"Memory {i}: {memory['pattern']}")
        print(f"  Outcome: {memory['outcome']}")
        print(f"  Confidence: {memory['confidence']:.2f}")
        print(f"  {status}")
        print()

    # Phase 5: Consolidate batch (simulated sleep)
    print("-" * 80)
    print("PHASE 5: Batch Consolidation (Sleep Phase)")
    print("-" * 80)
    print()

    print("Adding remaining memories to consolidation queue...")

    additional_memories = [
        {"pattern": "Pattern D", "outcome": "Stable behavior", "confidence": 0.88},
        {"pattern": "Pattern E", "outcome": "Learning adapted", "confidence": 0.82},
        {"pattern": "Pattern F", "outcome": "Consolidation complete", "confidence": 0.91},
    ]

    for mem in additional_memories:
        system2.consolidator.add_to_consolidation_queue(mem)

    print(f"  Added {len(additional_memories)} memories to queue")
    print()

    print("Running batch consolidation (simulating sleep/offline learning)...")
    consolidated_count = system2.consolidate_batch()
    print(f"  ✓ Consolidated {consolidated_count} memories in batch mode")
    print()

    # Final statistics
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print()

    stats = system2.get_full_statistics()

    print("System 2 Activity:")
    print(f"  Reasoning invocations: {stats['reasoning_invocations']}")
    print(f"  Consolidation invocations: {stats['consolidation_invocations']}")
    print()

    kb_stats = stats["knowledge_base"]
    print("Knowledge Base:")
    print(f"  Rules learned: {kb_stats['num_rules']}")
    print(f"  Facts stored: {kb_stats['num_facts']}")
    print(f"  Total rule frequency: {kb_stats['total_rule_frequency']}")
    print(f"  Mean rule confidence: {kb_stats['mean_rule_confidence']:.3f}")
    print(f"  Mean fact confidence: {kb_stats['mean_fact_confidence']:.3f}")
    print()

    conflict_stats = stats["conflicts"]
    print("Conflict Resolution:")
    print(f"  Total conflicts detected: {conflict_stats['total_conflicts']}")
    print(f"  Resolutions: {conflict_stats['resolutions']}")
    print()

    consol_stats = stats["consolidation"]
    print("Memory Consolidation:")
    print(f"  Consolidations completed: {consol_stats['consolidations']}")
    print(f"  Memories in queue: {consol_stats['in_queue']}")
    print(f"  Success rate: {consol_stats['success_rate']:.1%}")
    print()

    # Display learned knowledge
    print("=" * 80)
    print("LEARNED KNOWLEDGE")
    print("=" * 80)
    print()

    rules = system2.knowledge_base.get_rules()
    print(f"Rules ({len(rules)} total):")
    for rule in rules:
        print(f"  {rule}")
    print()

    facts = system2.knowledge_base.get_facts()
    print(f"Facts ({len(facts)} total):")
    for fact in facts:
        print(f"  {fact}")
    print()

    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    print("1. SYSTEM 1 (Spikes): Fast, automatic, reactive")
    print("   - Processes sensory input in milliseconds")
    print("   - Spreads activity through synaptic connections")
    print("   - No explicit reasoning needed")
    print()

    print("2. SYSTEM 1.5 (Control): Monitoring & gating")
    print("   - Detects novelty, confidence, conflict")
    print("   - Gates plasticity based on context")
    print("   - Decides when System 2 is needed")
    print()

    print("3. SYSTEM 2 (Reasoning): Slow, deliberate, explicit")
    print("   - Resolves conflicts using knowledge base")
    print("   - Learns explicit rules and facts")
    print("   - Consolidates experiences to long-term knowledge")
    print()

    print("4. INTEGRATION: System 1 ← System 1.5 → System 2")
    print("   - System 1 performs inference")
    print("   - System 1.5 monitors and detects issues")
    print("   - System 2 reasons when needed")
    print("   - Knowledge feeds back to System 1")
    print()

    print("=" * 80)
    print("Phase 3.3: System 2 Reasoning Implemented")
    print("=" * 80)
    print()


if __name__ == "__main__":
    run_system2_demonstration()
