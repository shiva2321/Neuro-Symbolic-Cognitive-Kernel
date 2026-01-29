use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DIMENSION: usize = 10240;

#[pyclass(module = "hypervec_rs")]
#[derive(Clone, Debug)]
struct HyperVector {
    bits: Vec<u64>, // Using u64 blocks for 10240 bits. 10240 / 64 = 160 blocks.
}

#[pymethods]
impl HyperVector {
    #[new]
    fn new(seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        let num_u64 = DIMENSION / 64;
        let mut bits = Vec::with_capacity(num_u64);
        
        for _ in 0..num_u64 {
            bits.push(rng.gen());
        }

        HyperVector { bits }
    }

    #[staticmethod]
    fn zero() -> Self {
        let num_u64 = DIMENSION / 64;
        HyperVector { bits: vec![0; num_u64] }
    }

    fn xor(&self, other: &HyperVector) -> HyperVector {
        let fused: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        HyperVector { bits: fused }
    }

    // Simple majority bundling: if used mostly for superposition, usually we'd track counts. 
    // For binary VSA, "bundling" usually means majority vote. Without counters, we can approximate 
    // or just use XOR if it's MAP. But standard bundling requires integers or random tie breaking.
    // For this prototype, let's implement a deterministic bitwise majority if we had 3 vectors, 
    // but for 2 vectors, bundling is often just XOR or OR. 
    // However, the prompt implies specific VSA operations.
    // Let's implement a 'bundle' that takes a list of vectors? 
    // Or implementing simple OR for now effectively creates a bloom filter like property, 
    // but strict VSA bundling (addition) usually needs bipolar or integer vectors.
    // Given the constraints (binary hypervectors), bundling 2 vectors usually requires a random tie-break 
    // for every bit where they differ.
    fn bundle(&self, other: &HyperVector) -> HyperVector {
        // Implementation: For each bit, if they are same, keep it. If differ, random choice (or fixed deterministic per dimension).
        // A common trick for binary bundling of A and B is to use a permutation or just random selection.
        // Let's use a seeded random choice based on the index to be deterministic but "fair".
        
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF); // Fixed seed for reproducibility of operation
        
        let fused: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| {
                let mask: u64 = rng.gen(); 
                // Effectively choosing bits from A or B randomly where they differ
                // (a & b) | (a & mask) | (b & !mask) ?? 
                // Actually, standard way is majority. With 2, it is random.
                let diff = a ^ b;
                let same = a & b;
                // If diff is 1, we need to choose. 
                // if mask bit is 1, take a, else take b.
                // (a & mask) | (b & !mask) covers the choice.
                // The same bits are preserved automatically?
                // if a=1, b=1 -> (1&m)|(1&!m) = m|!m = 1. Correct.
                // if a=0, b=0 -> 0. Correct.
                // if a=1, b=0 -> (1&m) = m.
                // if a=0, b=1 -> (1&!m) = !m.
                
                (a & mask) | (b & !mask)
            })
            .collect();
        HyperVector { bits: fused }
    }

    /// Weighted bundle: Creates a vector that is `weight` similar to self and `(1-weight)` to other.
    /// weight=1.0 -> returns self (similarity 1.0)
    /// weight=0.5 -> standard bundle (similarity ~0.75)
    /// weight=0.9 -> 90% of bits from self, 10% from other (similarity ~0.95)
    #[pyo3(signature = (other, weight, seed = None))]
    fn weighted_bundle(&self, other: &HyperVector, weight: f64, seed: Option<u64>) -> HyperVector {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(0xCAFEBABE),
        };
        
        // weight determines probability of picking from self vs other
        // For each bit position, pick from self with probability `weight`
        let fused: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| {
                let mut result: u64 = 0;
                for bit_pos in 0..64 {
                    let self_bit = (a >> bit_pos) & 1;
                    let other_bit = (b >> bit_pos) & 1;
                    
                    // Pick bit based on weight probability
                    let chosen_bit = if rng.gen::<f64>() < weight {
                        self_bit
                    } else {
                        other_bit
                    };
                    result |= chosen_bit << bit_pos;
                }
                result
            })
            .collect();
        HyperVector { bits: fused }
    }

    fn similarity(&self, other: &HyperVector) -> f64 {
        let mut hamming_dist: u32 = 0;
        for (a, b) in self.bits.iter().zip(other.bits.iter()) {
            hamming_dist += (a ^ b).count_ones();
        }
        
        // Similarity = 1.0 - (Hamming Distance / Total Dimensions)
        // Normalized to [0, 1]
        1.0 - (hamming_dist as f64 / DIMENSION as f64)
    }

    // --- LSH Support ---
    
    // Project hypervector onto a set of random hypervectors to get a signature
    // We can't pass a list of HVs easily from Python without overhead, 
    // so let's allow generating the projection vector from a seed internally.
    fn lsh_hash(&self, seed: u64, n_bits: usize) -> u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let num_u64 = DIMENSION / 64;
        
        let mut signature: u64 = 0;
        
        for i in 0..n_bits {
            if i >= 64 { break; } // Limit to 64-bit signature for efficient integer storage
            
            // Generate random projection vector
            let mut proj_bits = Vec::with_capacity(num_u64);
            for _ in 0..num_u64 {
                proj_bits.push(rng.gen());
            }
            
            // Dot product (XOR count)
            let mut hamming_dist: u32 = 0;
            for (a, b) in self.bits.iter().zip(proj_bits.iter()) {
                hamming_dist += (a ^ b).count_ones();
            }
            
            // If similarity > 0.5 (Hamming < DIM/2), set bit to 1
            // DIM=10240, DIM/2 = 5120
            if hamming_dist < (DIMENSION as u32 / 2) {
                signature |= 1 << i;
            }
        }
        
        signature
    }

    fn __repr__(&self) -> String {
        format!("<HyperVector dim={}>", DIMENSION)
    }

    // Pickle Support
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Return bits as a byte array or list of ints. 
        // List of u64 is simplest for now.
        Ok(self.bits.to_object(py))
    }

    fn __setstate__(&mut self, state: PyObject, py: Python) -> PyResult<()> {
        let bits: Vec<u64> = state.extract(py)?;
        self.bits = bits;
        Ok(())
    }
}

#[pymodule]
fn hypervec_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HyperVector>()?;
    Ok(())
}
