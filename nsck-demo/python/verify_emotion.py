"""
Verify Emotion System (Phase 2.1)
=================================
Tests the transformation of Drives/Rewards into Emotional States.
"""

from emotion_system import EmotionSystem

def main():
    print("--- Initializing Emotion System ---")
    es = EmotionSystem()
    
    print("\n--- Test 1: Baseline (Neutral) ---")
    drives = {"hunger": 0.0, "pain": 0.0}
    reward = 0.0
    es.update_from_drives(drives, reward)
    info = es.get_emotion_info()
    print(f"State: {info}")
    
    if info["name"] == "neutral" or info["name"] == "trust": # Trust is low arousal pos
        print("[PASS] Baseline is neutral/calm.")
    else:
        print("[FAIL] Baseline incorrect.")

    print("\n--- Test 2: Reward (Joy) ---")
    # Give positive reward + High Arousal (Excitement)
    # Joy = High Valence + High Arousal
    drives = {"hunger": 0.6, "pain": 0.0} 
    reward = 1.0 # Food found!
    es.update_from_drives(drives, reward)
    es.update_from_drives(drives, reward) # Boost it
    
    info = es.get_emotion_info()
    print(f"State: {info}")
    
    if info["name"] == "joy":
        print("[PASS] Positive reward triggers Joy.")
    else:
        print(f"[FAIL] Expected Joy, got {info['name']}.")

    print("\n--- Test 3: High Drive/Pain (Anger/Fear) ---")
    # Reset valence roughly
    es.valence = 0.0
    
    # High Pain + High Arousal
    drives = {"hunger": 0.2, "pain": 0.9} 
    reward = -0.5 # Punishment
    
    es.update_from_drives(drives, reward)
    info = es.get_emotion_info()
    print(f"State: {info}")
    
    # High arousal + Negative Valence -> Fear or Anger depending on map
    possible = ["fear", "anger", "sadness"]
    if info["name"] in possible:
        print(f"[PASS] Negative state detected: {info['name']}")
    else:
        print(f"[FAIL] Expected Negative state, got {info['name']}")

    print("\n--- Test 4: Text Recognition ---")
    txt = "I am so angry right now"
    rec = es.recognize_emotion_from_text(txt)
    print(f"Text: '{txt}' -> {rec}")
    
    if rec == "anger":
        print("[PASS] Text emotion recognized.")
    else:
        print("[FAIL] Text emotion failed.")

    print("\n--- Test 5: VSA Generation ---")
    hv = es.get_emotion_hypervector()
    print(f"HV Type: {type(hv)}")
    if hv is not None:
        print("[PASS] Hypervector generated.")
    else:
        print("[FAIL] Hypervector missing.")

if __name__ == "__main__":
    main()
