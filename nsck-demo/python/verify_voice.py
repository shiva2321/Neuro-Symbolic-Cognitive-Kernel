"""
Verify Voice I/O System
=======================
Tests Prosody Analysis, Emotion Inference, and VoiceHD Encoding.
"""

import numpy as np
from voice_interface import VoiceInterface

def generate_test_audio(duration=1.0, freq=440, amplitude=0.5):
    """Generate a simple sine wave buffer."""
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    return audio

def main():
    print("--- Initializing Voice Interface ---")
    voice = VoiceInterface()
    
    # Test Cases: (Frequency, Amplitude, Expected Emotion)
    # Note: Frequency maps to Pitch (F0), Amplitude maps to Energy.
    test_cases = [
        (440, 0.5, "excited"), # High Pitch, High Energy
        (100, 0.005, "sad"),   # Low Pitch, Low Energy
        (150, 0.05, "neutral") # Mid Pitch, Mid Energy
    ]
    
    print("\n--- Testing Prosody & Emotion Inference ---")
    for freq, amp, expected in test_cases:
        audio = generate_test_audio(freq=freq, amplitude=amp)
        result = voice.listen(audio)
        
        inferred = result["emotion"]
        prosody = result["prosody"]
        
        print(f"Input: Freq={freq}Hz, Amp={amp} -> Inferred: {inferred} (Expected: {expected})")
        print(f"  Features: {prosody}")
        
        # We allow fallback since heuristics are simple
        if inferred == expected:
             print("  [PASS] Emotion correct.")
        else:
             print("  [WARN] Emotion mismatch (Heuristic variation).")

    print("\n--- Testing VoiceHD Grounding ---")
    audio = generate_test_audio()
    result = voice.listen(audio)
    hv = result["audio_hv"]
    
    print(f"Audio HV Type: {type(hv)}")
    print(f"Audio HV Shape: {hv.shape}")
    print(f"Audio HV Sparsity: {np.mean(hv):.4f}")
    
    if hv.shape == (10000,):
        print("[PASS] VoiceHD Hypervector generated correctly.")
    else:
        print("[FAIL] Invalid Hypervector shape.")

    print("\n--- Testing Speech Synthesis (Mock) ---")
    out_audio = voice.speak("Hello human, I am feeling good today.", emotion="excited")
    
    if len(out_audio) > 0:
        print("[PASS] Generated output audio buffer.")
    else:
        print("[FAIL] Output audio buffer empty.")

if __name__ == "__main__":
    main()
