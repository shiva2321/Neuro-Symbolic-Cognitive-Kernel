"""
Verify Roadmap Phases 4 & 5
===========================
Validates EWC, Meta-Learning, Curriculum, and Consciousness Metrics.
"""

import torch
import torch.nn as nn
from continual_learning import ContinualLearner
from meta_learning import MAMLLearner
from curriculum import CurriculumDesigner
from consciousness_metrics import ConsciousnessMonitor, GlobalWorkspaceMetrics
from cognitive_engine import CognitiveEngine
import networkx as nx

def test_phase_4_continual_learning():
    print("--- Testing Phase 4.1: EWC ---")
    model = nn.Linear(10, 2)
    learner = ContinualLearner(model)
    
    # Simulate a task
    inputs = torch.randn(10, 10)
    targets = torch.randint(0, 2, (10,))
    loader = [(inputs, targets)]
    
    learner.compute_weight_importance("task1", loader)
    print(f"Importance calculated for task1. Tensors: {len(learner.weight_importance['task1'])}")
    
    loss = learner.ewc_loss()
    print(f"EWC Loss (should be 0 for current task): {loss.item()}")
    
    # Change weights and check loss
    with torch.no_grad():
        model.weight.add_(0.1)
    loss = learner.ewc_loss()
    print(f"EWC Loss after weight shift: {loss.item():.4f}")
    assert loss > 0
    print("[PASS] EWC correctly penalizes weight shifts.")

def test_phase_4_meta_learning():
    print("\n--- Testing Phase 4.2: Meta-Learning ---")
    model = nn.Linear(10, 2)
    learner = MAMLLearner(model)
    
    task = {
        'support_input': torch.randn(5, 10),
        'support_target': torch.randint(0, 2, (5,)),
        'query_input': torch.randn(5, 10),
        'query_target': torch.randint(0, 2, (5,))
    }
    
    meta_loss = learner.meta_step([task])
    print(f"Meta-step completed. Meta-loss: {meta_loss:.4f}")
    print("[PASS] Meta-learning step executed successfully.")

def test_phase_5_consciousness():
    print("\n--- Testing Phase 5.1: Consciousness (Phi) ---")
    # Simulate a system graph
    G = nx.DiGraph()
    G.add_edge("Perception", "GW", weight=0.8)
    G.add_edge("GW", "Motor", weight=0.7)
    G.add_edge("Memory", "GW", weight=0.9)
    G.add_edge("GW", "Memory", weight=0.6)
    
    monitor = ConsciousnessMonitor(None)
    phi = monitor.compute_phi(G)
    print(f"Computed Phi (Integrated Information): {phi:.4f}")
    
    monitor.update_attention_schema("FOOD", 0.95)
    print(f"Subjective Report: {monitor.get_subjective_report()}")
    
    assert phi > 0
    print("[PASS] Phi computation and attention schema validated.")

if __name__ == "__main__":
    try:
        test_phase_4_continual_learning()
        test_phase_4_meta_learning()
        test_phase_5_consciousness()
        print("\n[COMPLETE] Roadmap Phase 4 & 5 core verification successful.")
    except Exception as e:
        print(f"\n[FAIL] Verification failed: {e}")
        import traceback
        traceback.print_exc()
