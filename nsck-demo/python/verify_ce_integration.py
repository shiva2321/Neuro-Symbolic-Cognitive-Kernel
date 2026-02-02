"""
Verify Cognitive Engine Integration (Phase 2 Wrap-up)
=====================================================
Tests that CognitiveEngine correctly integrates Homeostasis, Emotions, and ToM.
"""

from cognitive_engine import CognitiveEngine
import time

def main():
    print("--- Testing Cognitive Engine Integration ---")
    ce = CognitiveEngine()
    
    # 1. Step the engine with a mock state
    state = {"head": (5,5), "food": (6,5), "body": [(5,5)]}
    task = "snake"
    
    # Mock metacognition result
    meta = {"action": "ACTION_RIGHT", "confidence": 0.8}
    
    print("\n--- Step 1: Decision making ---")
    c_state = ce.decide(state, task, meta)
    print(f"Chosen Action: {c_state.chosen_action}")
    print(f"Current Emotion: {c_state.emotion}")
    
    # 2. Learn from outcome
    print("\n--- Step 2: Learning and Emotion Update ---")
    # Simulate finding food
    ce.learn(state, "ACTION_RIGHT", 1.0, task, outcome="success")
    
    print(f"Drives: {ce.homeostasis.drives}")
    print(f"Emotion after reward: {ce.emotion_system.current_emotion}")
    
    if ce.emotion_system.current_emotion == "joy" or ce.emotion_system.current_emotion == "trust":
         print("[PASS] Emotion system responded to reward.")
    else:
         print(f"[FAIL] Emotion system state: {ce.emotion_system.current_emotion}")

    # 3. Test ToM integration
    print("\n--- Step 3: Theory of Mind integration ---")
    # Sally is in the room
    ce.theory_of_mind.update_agent_perspective("Sally", "room", {"ball": "basket"})
    
    # Learn an episode
    ce.learn(state, "ACTION_STAY", 0.0, task, outcome="neutral")
    
    # Retrieve last episode from memory
    episodes = ce.episodic_memory.recall_recent(task, 1)
    if episodes:
        ep = episodes[0]
        # Check if ToM snapshot was saved
        if ep.tom_beliefs and ep.tom_beliefs.get("ball") == "basket":
            print("[PASS] Theory of Mind state captured in episodic memory.")
        else:
            print(f"[FAIL] ToM state in memory: {ep.tom_beliefs}")
    else:
        print("[FAIL] No episodes recorded.")

if __name__ == "__main__":
    main()
