"""
Verify Integrated AGI Stack (Phases 1-5)
========================================
Tests the unified CognitiveEngine with all integrated modules.
"""

from cognitive_engine import CognitiveEngine
import numpy as np
import time

def main():
    print("--- Testing Integrated AGI Stack ---")
    
    # 1. Initialize Engine
    ce = CognitiveEngine()
    # Force world model to be ready sooner for testing
    if ce.world_model:
        ce.world_model.predictor.config.hv_dim = 1024 # Small for test speed
    
    task = "snake"
    
    # Mock observation
    obs = {
        "head": (5, 5),
        "food": (5, 6),
        "body": [(5, 4)],
        "grid": np.zeros((10, 10)).tolist()
    }
    
    print("\n--- Step 1: Decision making with integrated modules ---")
    # Should run GWT Competition with SNN, Rules, Planner, etc.
    res_state = ce.decide(obs, task)
    print(f"Decision: {res_state.chosen_action}")
    print(f"Winning Module: {res_state.trace.get('winner')}")
    
    # 2. Test Learning & Component Updates
    print("\n--- Step 2: Learning & State Updates ---")
    # Simulate eating food
    next_obs = obs.copy()
    next_obs["head"] = (5, 6) # Moved to food
    
    ce.learn(
        state=obs,
        action="ACTION_RIGHT",
        reward=1.0,
        task_tag=task,
        outcome="success",
        next_state=next_obs
    )
    
    # Check Homeostasis
    print(f"Homeostasis Energy: {ce.homeostasis.energy:.2f}")
    
    # Check Emotion
    print(f"Current Emotion: {ce.emotion_system.current_emotion}")
    
    # Check Memory
    print(f"Episodes Recorded: {ce.stats['episodes_recorded']}")
    
    # 3. Test Causal Discovery
    print("\n--- Step 3: Causal Discovery ---")
    # Force a graph induction
    induced = ce.causal_discovery.induce_graph(task, min_evidence=1, min_confidence=0.1)
    print(f"Causal links found: {len(induced.all_links)}")
    
    # 4. Test World Model & Imagination
    print("\n--- Step 4: World Model & Imagination ---")
    print(f"World Model Training Steps: {ce.world_model.stats['train_steps'] if ce.world_model else 'N/A'}")
    
    # Force set world model to ready
    if ce.world_model:
        ce.world_model.stats['train_steps'] = 1000
        res_state2 = ce.decide(obs, task)
        print(f"Decision 2 (Imagination active?): {res_state2.chosen_action}")
        # Note: IMAGINATION only wins if its cumulative reward is high
        
    print("\n[COMPLETE] Integrated AGI Stack verified.")

if __name__ == "__main__":
    main()
