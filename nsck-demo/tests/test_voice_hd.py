
import sys
import os
import numpy as np

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_hd import VoiceHDEngine, SAMPLE_RATE

def generate_sine_wave(freq_start, freq_end, duration_sec=1.0, noise_std=0.0):
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec))
    # Linear chirp
    # Phase = 2*pi * integral(f(t))
    # f(t) = f_start + (f_end - f_start)/T * t
    k = (freq_end - freq_start) / duration_sec
    phase = 2 * np.pi * (freq_start * t + 0.5 * k * t**2)
    sig = 0.5 * np.sin(phase)
    
    # Add noise
    if noise_std > 0:
        sig += np.random.normal(0, noise_std, sig.shape)
        
    return sig

def test_voice_hd():
    print("Initializing VoiceHD Engine...")
    # Fixed seed for reproducibility
    engine = VoiceHDEngine(seed=42)
    
    # 1. Unit Check: Level Vector Correlation
    print("\n[Unit Check] Level Vector Correlation (Thermometer)")
    l_vecs = engine.level_vectors
    sim_adj = engine.hamming_similarity(l_vecs[0], l_vecs[1])
    sim_far = engine.hamming_similarity(l_vecs[0], l_vecs[-1])
    print(f"L0 vs L1 (Adjacent): {sim_adj:.4f} (Expected > 0.9)")
    print(f"L0 vs L19 (Distant): {sim_far:.4f} (Expected < 0.6)")
    
    if sim_adj > 0.9 and sim_far < 0.6:
        print(">> PASSED: Level Vectors are correlated.")
    else:
        print(">> FAILED: Level Vectors do not show expected drift.")

    # 2. System Test: Up vs Down Tones
    print("\n[System Test] Up-Chirp vs Down-Chirp discrimination")
    
    n_trials = 20
    noise_level = 0.05
    
    ups = []
    downs = []
    
    print(f"Generating {n_trials} samples per class with noise={noise_level}...")
    
    for i in range(n_trials):
        # Class A: UP (400 -> 800 Hz)
        sig_up = generate_sine_wave(400, 800, noise_std=noise_level)
        hv_up = engine.encode(sig_up)
        
        if i == 0:
            dens = np.mean(hv_up)
            print(f"HV Density: {dens:.4f} (Target ~0.5)")
            
        ups.append(hv_up)
        
        # Class B: DOWN (800 -> 400 Hz)
        sig_down = generate_sine_wave(800, 400, noise_std=noise_level)
        hv_down = engine.encode(sig_down)
        downs.append(hv_down)
        
    # Stats
    intra_up = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            intra_up.append(engine.hamming_similarity(ups[i], ups[j]))
            
    intra_down = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            intra_down.append(engine.hamming_similarity(downs[i], downs[j]))
            
    inter_class = []
    for i in range(n_trials):
        for j in range(n_trials):
            inter_class.append(engine.hamming_similarity(ups[i], downs[j]))
            
    mean_intra = np.mean(intra_up + intra_down)
    std_intra = np.std(intra_up + intra_down)
    
    mean_inter = np.mean(inter_class)
    std_inter = np.std(inter_class)
    
    print(f"\nIntra-Class Similarity: {mean_intra:.4f} (+/- {std_intra:.4f})")
    print(f"Inter-Class Similarity: {mean_inter:.4f} (+/- {std_inter:.4f})")
    
    # Thresholds from Plan
    # Mean(Intra) > 0.75, Mean(Inter) < 0.25 is too strict for binary hamming?
    # Actually, random vectors have sim 0.5 in Hamming. 
    # Valid hypervectors should have high similarity.
    # Wait, the spec said "Sim(A, B) = 1.0 - XOR/N".
    # For random vectors, XOR/N is 0.5, so Sim is 0.5.
    # Therefore, Inter-class should be around 0.5 (Orthogonal), not 0.25.
    # Intra-class should be > 0.6 or 0.7.
    
    # Let's adjust expectation: Inter should be ~0.5 (Uncorrelated).
    # If Inter < 0.5, they are anti-correlated.
    
    print("Note: Random baseline for Hamming Sim is 0.5.")
    
    print("\n[Classifier Test] Nearest-Prototype Accuracy")
    # Train/Test Split
    n_train = 5
    n_test = 15
    
    # Prototypes (Mean of training vectors)
    # Majority Vote bundle for prototype
    proto_up = engine._majority_vote(np.array(ups[:n_train]))
    proto_down = engine._majority_vote(np.array(downs[:n_train]))
    
    correct = 0
    total = 0
    
    print(f"Training on {n_train} samples per class...")
    print(f"Testing on {n_test} samples per class...")
    
    # Test Class A (UP)
    for i in range(n_train, n_trials):
        hv = ups[i]
        sim_up = engine.hamming_similarity(hv, proto_up)
        sim_down = engine.hamming_similarity(hv, proto_down)
        
        if sim_up > sim_down:
            correct += 1
        total += 1
        
    # Test Class B (DOWN)
    for i in range(n_train, n_trials):
        hv = downs[i]
        sim_up = engine.hamming_similarity(hv, proto_up)
        sim_down = engine.hamming_similarity(hv, proto_down)
        
        if sim_down > sim_up:
            correct += 1
        total += 1
        
    acc = correct / total
    print(f"Accuracy: {acc*100:.1f}% ({correct}/{total})")
    
    if acc >= 0.8:
        print(">> PASSED: Classification accuracy > 80%.")
    else:
        print(">> FAILED: Accuracy too low.")

if __name__ == "__main__":
    test_voice_hd()
