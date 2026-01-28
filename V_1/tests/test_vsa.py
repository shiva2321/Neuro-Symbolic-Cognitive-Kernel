import pytest
import torch
import torchhd
from ncgn.memory.vsa_core import VSAEngine
from ncgn.memory.associative import AssociativeMemory

@pytest.fixture
def vsa():
    return VSAEngine(dimensions=10000)

@pytest.fixture
def memory(vsa):
    mem = AssociativeMemory(vsa)
    # Populate with some standard concepts
    for i in range(10):
        mem.add_concept(f"Concept_{i}")
    return mem

def test_orthogonality(vsa):
    """Verify that random vectors are approximately orthogonal."""
    v1 = vsa.create_vector()
    v2 = vsa.create_vector()
    sim = vsa.similarity(v1, v2)
    print(f"\nOrthogonality check: Sim(v1, v2) = {sim}")
    # In D=10k, similarity should be very close to 0 (typically < 0.05)
    assert abs(sim) < 0.1

def test_binding_property(vsa):
    """
    Verify binding properties:
    1. Result is orthogonal to inputs.
    2. Reversibility (Unbinding).
    """
    role = vsa.create_vector("Role")
    filler = vsa.create_vector("Filler")
    
    bound = vsa.bind(role, filler)
    
    # 1. Result orthogonal to inputs
    sim_role = vsa.similarity(bound, role)
    sim_filler = vsa.similarity(bound, filler)
    assert abs(sim_role) < 0.1
    assert abs(sim_filler) < 0.1
    
    # 2. Reversibility: Bound * Role = Filler (approx or exact depending on model)
    # In MAP/bipolar, XOR is its own inverse.
    unbound = vsa.bind(bound, role)
    sim_recover = vsa.similarity(unbound, filler)
    print(f"Unbinding recovery: {sim_recover}")
    assert sim_recover > 0.99  # Should be perfect for integer XOR

def test_bundling_property(vsa):
    """
    Verify bundling properties:
    1. Result is similar to all constituents.
    """
    v1 = vsa.create_vector("Apple")
    v2 = vsa.create_vector("Banana")
    v3 = vsa.create_vector("Cherry")
    
    fruit_basket = vsa.bundle([v1, v2, v3])
    
    sim1 = vsa.similarity(fruit_basket, v1)
    sim2 = vsa.similarity(fruit_basket, v2)
    sim3 = vsa.similarity(fruit_basket, v3)
    
    print(f"Bundle Similarities: {sim1:.3f}, {sim2:.3f}, {sim3:.3f}")
    
    # Ideally sim should be approx 1/sqrt(3) ~= 0.57
    assert sim1 > 0.4
    assert sim2 > 0.4
    assert sim3 > 0.4
    
    # Result should be orthogonal to a random vector
    v_rand = vsa.create_vector("Dog")
    assert abs(vsa.similarity(fruit_basket, v_rand)) < 0.1

def test_z_score_confidence(vsa, memory):
    """
    Verify the R5 Z-Score logic using a 'Noisy' vector.
    """
    # 1. Create a "Noisy" version of Concept_5
    target_label = "Concept_5"
    original_vec = memory.get_vector(target_label)
    
    # Add noise by flipping bits (or binding with random noise with small weight if continuous)
    # For binary hypervectors, we can simulate noise by flipping, say, 30% of bits.
    # However, torchhd vectors might be float or integer depending on model.
    # Let's use the bundle method to mix in noise: 70% Original + 30% Noise? 
    # Actually, simpler to just bundle target + random + random.
    
    # Let's create a vector that is somewhat similar to Concept_5
    # Since operations are usually integer/boolean in torchhd by default, strict weighted sum isn't always direct.
    # But usually bundle([A, A, B]) makes result closer to A.
    
    # Let's explicitly create a 'noisy' probe
    noise = vsa.create_vector()
    # Concept_5 + Concept_5 + Noise -> Result is ~0.8 similar to Concept_5
    noisy_probe = vsa.bundle([original_vec, original_vec, noise]) 
    
    label, score, z_score = memory.query(noisy_probe)
    
    print(f"\nQuerying Noisy {target_label}: Found={label}, Sim={score:.3f}, Z={z_score:.3f}")
    
    assert label == target_label
    assert score > 0.6  # Similarity should be high
    assert z_score > 3.0 # Z-score should be very confident (statistically significant outlier)

def test_unknown_vector(vsa, memory):
    """
    Verify that a completely random vector returns low Z-score.
    """
    random_probe = vsa.create_vector("Alien_Concept")
    
    label, score, z_score = memory.query(random_probe)
    
    print(f"Querying Random Vector: Found={label} (Best Guess), Sim={score:.3f}, Z={z_score:.3f}")
    
    # Calculating stats manually to explain
    # Mean of 10 random vectors is ~0.
    # Best match of 10 random vectors will be small (maybe 0.05).
    # Z score might be (0.05 - 0.0) / 0.02 = 2.5?
    # Actually, for D=10k, sigma is 0.01. Max of 10 samples from N(0, 0.01) is likely < 0.03.
    # So valid matches (Sim > 0.4) will have massive Z-scores ( > 40).
    # Random matches will have Z-scores < 3 typically.
    
    assert z_score < 4.0 # Should be much lower than a real match
    assert score < 0.2  # Should be low similarity

if __name__ == "__main__":
    # Manual setup consistent with fixtures
    v = VSAEngine(dimensions=10000)
    m = AssociativeMemory(v)
    for i in range(10):
        m.add_concept(f"Concept_{i}")
    
    print("Running test_orthogonality...")
    test_orthogonality(v)
    
    print("Running test_binding_property...")
    test_binding_property(v)
    
    print("Running test_bundling_property...")
    test_bundling_property(v)
    
    print("Running test_z_score_confidence...")
    test_z_score_confidence(v, m)
    
    print("Running test_unknown_vector...")
    test_unknown_vector(v, m)
    
    print("\nAll tests passed!")

