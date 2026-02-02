"""
Verify Self-Model (Phase 3.2)
============================
Tests Identity, Capability mapping, and Calibration tracking.
"""

from self_model import SelfModel
import hypervec_shim as hypervec_rs

def main():
    print("--- Testing Self-Model ---")
    sm = SelfModel()
    
    # 1. Identity Check
    print("\n--- Step 1: Identity ---")
    identity = sm.get_identity()
    print(f"Identity HV type: {type(identity)}")
    if isinstance(identity, hypervec_rs.HyperVector):
        print("[PASS] Identity Hypervector generated.")
    
    # 2. Capability Scaling
    print("\n--- Step 2: Capability Mapping ---")
    action = "ACTION_UP"
    print(f"Initial capability ({action}): {sm.get_capability(action)}")
    
    # Simulate 5 successes
    for _ in range(5):
        sm.update("snake", action, 0.8, True, 1.0)
        
    cap = sm.get_capability(action)
    print(f"Capability after 5 successes: {cap:.2f}")
    if cap > 0.4:
        print("[PASS] Capability increased with success.")
        
    # Simulate 2 failures
    for _ in range(2):
        sm.update("snake", action, 0.8, False, -1.0)
    
    cap2 = sm.get_capability(action)
    print(f"Capability after 2 failures: {cap2:.2f}")
    if cap2 < cap:
        print("[PASS] Capability decreased with failure.")

    # 3. Calibration Tracking
    print("\n--- Step 3: Calibration ---")
    # We had 0.8 confidence but failed 2/7 times.
    ece = sm.get_calibration_error("snake")
    print(f"Expected Calibration Error: {ece:.4f}")
    if ece > 0:
        print("[PASS] Calibration error tracked.")

    # 4. Success Prediction
    print("\n--- Step 4: Success Prediction ---")
    # Force more attempts to bypass cold start
    for _ in range(10):
        sm.update("snake", "ACTION_RIGHT", 0.5, True, 1.0)
    
    prob = sm.predict_success("snake")
    print(f"Probability of success for snake: {prob:.2f}")
    if prob > 0.5:
        print("[PASS] Success prediction functioning.")

if __name__ == "__main__":
    main()
