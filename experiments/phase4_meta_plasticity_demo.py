"""
Phase 4 Demo: Meta-Plasticity - Learning to Learn

Demonstrates:
1. Adaptive learning rate scheduling based on performance/stability
2. Region-specific learning rates
3. Signal computation: confidence, novelty, error, stability
4. MetaLearner.update() for adaptive hyperparameters
"""

import sys
sys.path.insert(0, '/d/Node_network')

from core.neuron import Neuron
from core.synapse import Synapse
from core.region import Region, RegionType
from core.meta_learner import MetaLearner, AdaptationSignal
from core.metrics import SignalComputer
from core.plasticity import PlasticityController


def create_simple_region() -> Region:
    """Create a small region with 3 neurons and some connections."""
    neurons = {
        0: Neuron(neuron_id=0, threshold=1.0, neuron_type="input"),
        1: Neuron(neuron_id=1, threshold=1.0, neuron_type="hidden"),
        2: Neuron(neuron_id=2, threshold=1.0, neuron_type="output"),
    }

    synapses = {
        1: {0: Synapse(weight=0.5, is_inhibitory=False)},
        2: {1: Synapse(weight=0.6, is_inhibitory=False)},
    }

    region = Region(
        region_id=0,
        region_type=RegionType.HIDDEN,
        neurons=neurons,
        synapses=synapses,
    )
    return region


def demo_adaptive_plasticity():
    """Demonstrate adaptive learning rate scheduling."""
    print("\n" + "="*70)
    print("PHASE 4 DEMO: ADAPTIVE PLASTICITY")
    print("="*70)

    region = create_simple_region()
    signal_computer = SignalComputer(window_size=50)

    print("\n1. ADAPTIVE LEARNING RATE SCHEDULING")
    print("-" * 70)

    # Simulate performance improving then degrading
    scenarios = [
        ("Good performance (improving)", 0.5, 0.2),   # perf_delta, stability
        ("Stable but not improving", 0.0, 0.8),
        ("Poor performance (unstable)", -0.3, 0.2),
    ]

    for scenario_name, perf_delta, stability in scenarios:
        rate = region.plasticity.get_effective_lr(step=0, perf_delta=perf_delta, stability=stability)
        print(f"  {scenario_name:35} → LR = {rate:.5f}")

    print("\n2. REGION-SPECIFIC LEARNING RATE MULTIPLIERS")
    print("-" * 70)

    multipliers = [0.2, 0.5, 1.0, 2.0]
    base_rate = region.plasticity.get_effective_lr(step=0)

    for mult in multipliers:
        region.set_learning_rate_multiplier(mult)
        effective = region.plasticity.get_effective_lr(step=0)
        print(f"  Multiplier: {mult:3.1f}x  →  Effective LR: {effective:.5f}")

    region.set_learning_rate_multiplier(1.0)  # Reset

    print("\n3. SIGNAL COMPUTATION (Confidence, Novelty, Error, Stability)")
    print("-" * 70)

    # Simulate stable learning episode
    print("  Scenario A: Stable learning (same weights)")
    signal_computer.clear()
    for step in range(20):
        signal_computer.update(
            current_weights=[0.5, 0.6, 0.7],
            current_firing_rate=0.25,
            prediction_error=0.1,
            observed_pattern_id="pattern_stable"
        )

    signals_a = signal_computer.get_signals("pattern_stable")
    print(f"    Confidence:  {signals_a['confidence']:.3f} (high = stable)")
    print(f"    Novelty:     {signals_a['novelty']:.3f} (0 = seen before)")
    print(f"    Error:       {signals_a['error']:.3f}")
    print(f"    Stability:   {signals_a['stability']:.3f} (high = consistent)")

    # Simulate unstable/novel learning
    print("\n  Scenario B: Novel/unstable learning (variable weights)")
    signal_computer.clear()
    for step in range(20):
        signal_computer.update(
            current_weights=[0.2 + step*0.02, 0.4 + step*0.01, 0.8 - step*0.015],
            current_firing_rate=0.1 + (step % 3) * 0.15,
            prediction_error=0.3 + (step % 5) * 0.1,
            observed_pattern_id="pattern_novel"
        )

    signals_b = signal_computer.get_signals("pattern_novel")
    print(f"    Confidence:  {signals_b['confidence']:.3f} (low = variable)")
    print(f"    Novelty:     {signals_b['novelty']:.3f} (1.0 = new)")
    print(f"    Error:       {signals_b['error']:.3f}")
    print(f"    Stability:   {signals_b['stability']:.3f} (low = inconsistent)")

    print("\n4. META-LEARNER HYPERPARAMETER ADAPTATION")
    print("-" * 70)

    meta_learner = MetaLearner()
    meta_learner.add_region(region_id=0, specialization="hidden", initial_learning_rate=0.01)

    # Simulate successful learning
    print("  Scenario: High confidence, low error → SUCCESS")
    hyperparams = {'region_id': 0}
    signals_success = {
        'confidence': 0.85,
        'novelty': 0.2,
        'error': 0.15,
        'stability': 0.7,
        'performance_delta': 0.2,
    }

    updated = meta_learner.update(hyperparams, signals_success)
    print(f"    Suggested region_lr_multiplier: {updated['region_lr_multiplier']:.3f}")
    print(f"    Novelty threshold:              {updated['novelty_threshold']:.3f}")
    print(f"    Reward threshold:               {updated['reward_threshold']:.3f}")
    print(f"    Error threshold:                {updated['error_threshold']:.3f}")

    # Simulate struggling learning
    print("\n  Scenario: Low confidence, high error → FAILURE")
    signals_failure = {
        'confidence': 0.3,
        'novelty': 0.7,
        'error': 0.8,
        'stability': 0.2,
        'performance_delta': -0.3,
    }

    updated2 = meta_learner.update(hyperparams, signals_failure)
    print(f"    Suggested region_lr_multiplier: {updated2['region_lr_multiplier']:.3f}")
    print(f"    Novelty threshold:              {updated2['novelty_threshold']:.3f}")

    print("\n5. REGION LEARNING WITH ADAPTIVE RATES")
    print("-" * 70)

    # Apply learning with adaptive rate
    synapse = list(list(region.synapses.values())[0].values())[0]
    original_w = synapse.weight

    region.enable_plasticity()
    for step in range(5):
        perf = 0.5 + (step * 0.1)  # Improving performance
        region.apply_learning(
            target_id=1,
            post_fired=True,
            dopamine=0.5,
            step=step,
            perf_delta=perf,
            stability=0.6
        )
        print(f"  Step {step}: Weight {original_w:.3f} → {synapse.weight:.3f} (perf_delta={perf:.2f})")

    print("\n6. META-LEARNER STATISTICS")
    print("-" * 70)
    stats = meta_learner.get_full_statistics()
    print(f"  Total decisions:  {stats['total_decisions']}")
    print(f"  Successful:       {stats['decisions_successful']}")
    print(f"  Current thresholds:")
    print(f"    Novelty:  {stats['thresholds']['novelty']['value']:.3f}")
    print(f"    Reward:   {stats['thresholds']['reward']['value']:.3f}")
    print(f"    Error:    {stats['thresholds']['error']['value']:.3f}")

    print("\n" + "="*70)
    print("PHASE 4 DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demo_adaptive_plasticity()
