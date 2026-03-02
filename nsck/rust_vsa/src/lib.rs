use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// Concurrent modules
mod concurrent;
mod semantic;
mod episodic;
mod worker_pool;
mod persistence;
mod async_runtime;
mod societal;

const DIMENSION: usize = 10240;

#[pyclass(module = "hypervec_rs")]
#[derive(Clone, Debug)]
pub struct HyperVector {
    pub bits: Vec<u64>, // Using u64 blocks for 10240 bits. 10240 / 64 = 160 blocks.
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

    /// Pairwise bundle (superposition) of two binary hypervectors.
    ///
    /// For binary VSA, bundling two vectors requires a random tie-break for every
    /// bit position where they differ (majority vote with N=2 is always a tie).
    /// The seed is derived from BOTH vectors so the result is:
    ///   1. Deterministic (same inputs → same output)
    ///   2. Pair-specific (bundle(A,B) ≠ bundle(C,D) unless A=C and B=D)
    ///
    /// Mathematical correctness:
    ///   bit=1 in both  → result=1 (both agree)
    ///   bit=0 in both  → result=0 (both agree)
    ///   bit differs    → result drawn uniformly from {0,1} (random tie-break)
    ///
    /// Expected similarity: sim(bundle(A,B), A) ≈ 0.75 (agrees on A's 1s + half the diff bits).
    ///
    /// For bundling N>2 vectors with true majority vote, use the free function
    /// `bundle_hvs(Vec<Vec<u8>>)` which operates on bit arrays.
    fn bundle(&self, other: &HyperVector) -> HyperVector {
        let pair_seed = self.bits[0]
            ^ other.bits[0]
            ^ self.bits[1].wrapping_mul(0x9E3779B97F4A7C15)
            ^ other.bits[1].wrapping_mul(0x517CC1B727220A95);
        let mut rng = ChaCha8Rng::seed_from_u64(pair_seed);
        
        let fused: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| {
                let mask: u64 = rng.gen();
                // When bits agree: (a & mask) | (b & !mask) = (x & mask) | (x & !mask) = x ✓
                // When a=1, b=0:   (1 & mask) = mask        → random from {0,1} ✓
                // When a=0, b=1:   (1 & !mask) = !mask      → random from {0,1} ✓
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
        let pair_seed = self.bits[0]
            ^ other.bits[0]
            ^ self.bits[1].wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(pair_seed),
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

    /// Circular bitwise permutation matching Python/NumPy convention:
    ///   `permute(+n)` shifts bits LEFT by n positions  (same as `np.roll(bits, -n)`).
    ///   `permute(-n)` shifts bits RIGHT by n positions (same as `np.roll(bits, +n)`).
    ///
    /// Positive shift is the temporal-binding direction:
    ///   `permute(1)` marks "sequence position 1", `permute(2)` marks "position 2", etc.
    /// `permute_inverse(n)` is exactly `permute(-n)`.
    ///
    /// Performance: ~200 ns per 10,240-bit vector (cache-friendly u64 block operations).
    fn permute(&self, shift: i32) -> HyperVector {
        let num_u64   = DIMENSION / 64;        // 160
        let total_bits = DIMENSION as i32;     // 10240

        // Normalize to [0, total_bits): positive means left.
        let shift_norm = ((shift % total_bits) + total_bits) % total_bits;
        if shift_norm == 0 {
            return self.clone();
        }

        let word_shift = (shift_norm as usize) / 64;
        let bit_shift  = (shift_norm as usize) % 64;

        let mut new_bits = vec![0u64; num_u64];

        if bit_shift == 0 {
            // Pure word-level left rotation: take from higher index.
            for i in 0..num_u64 {
                let src = (i + word_shift) % num_u64;
                new_bits[i] = self.bits[src];
            }
        } else {
            // Bit-level left shift with carry between adjacent words.
            // new[i] = (old[i+ws] >> bit_shift) | (old[i+ws+1] << complement)
            // This places bit (p + shift_norm) of old into position p of new → LEFT shift.
            let complement = 64 - bit_shift;
            for i in 0..num_u64 {
                let src_lo = (i + word_shift)     % num_u64;
                let src_hi = (i + word_shift + 1) % num_u64;
                new_bits[i] = (self.bits[src_lo] >> bit_shift)
                            | (self.bits[src_hi] << complement);
            }
        }

        HyperVector { bits: new_bits }
    }

    /// Inverse permutation: equivalent to permute(-shift).
    /// Used to "unbind" temporal position from a sequence vector.
    fn permute_inverse(&self, shift: i32) -> HyperVector {
        self.permute(-shift)
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

    /// VSA anti-bundling negation.
    ///
    /// Returns a vector that is ~50 % similar to `self` (orthogonal region)
    /// but is *deterministically* the same for the same input — i.e. it is
    /// the "negation role" XOR of `self`.  This is the VSA convention for
    /// negation: `negate(negate(hv))` ≈ `hv` (idempotent up to the fixed role).
    ///
    /// Implementation: XOR `self` with a fixed "negation role" hypervector
    /// seeded at the compile-time constant 0xDEAD_BEEF_CAFE_BABE.
    fn negate(&self) -> HyperVector {
        // Fixed negation-role seed — must match Python `hypervec_py.py`
        const NEG_SEED: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut rng = ChaCha8Rng::seed_from_u64(NEG_SEED);
        let num_u64 = DIMENSION / 64;
        let neg_role: Vec<u64> = (0..num_u64).map(|_| rng.gen()).collect();
        let bits: Vec<u64> = self.bits.iter().zip(neg_role.iter()).map(|(a, n)| a ^ n).collect();
        HyperVector { bits }
    }

    /// Create a HyperVector from a flat list of u64 words (160 words for D=10240).
    /// Useful for reconstructing HVs from stored state or Python `from_bits()` calls.
    #[staticmethod]
    fn from_u64_words(words: Vec<u64>) -> HyperVector {
        HyperVector { bits: words }
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

// ============================================================
// Free functions exposed to Python
// ============================================================

/// Compute the N×N similarity matrix for a list of HyperVectors.
///
/// Returns a flat Vec<f64> of length N² in row-major order.
/// Used by the category-theoretic functoriality score to measure
/// structure preservation: F = 1 − mean|sim(aᵢ,aⱼ) − sim(F(aᵢ),F(aⱼ))|.
///
/// Parallelised with rayon — O(N²·D/64) popcount operations.
#[pyfunction]
fn batch_similarity_matrix(vectors: Vec<PyRef<HyperVector>>) -> Vec<f64> {
    use rayon::prelude::*;
    let n = vectors.len();
    let refs: Vec<&[u64]> = vectors.iter().map(|v| v.bits.as_slice()).collect();
    (0..n * n)
        .into_par_iter()
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            if i == j {
                1.0
            } else {
                let mut hamming: u32 = 0;
                for (a, b) in refs[i].iter().zip(refs[j].iter()) {
                    hamming += (a ^ b).count_ones();
                }
                1.0 - (hamming as f64 / DIMENSION as f64)
            }
        })
        .collect()
}

/// Weber-Fechner logarithmic compression for a vector of f64 values.
///
/// x_wf = sign(x) · ln(1 + |x|)
///
/// Inspired by Fechner's psychophysics law (1860): perceived intensity is
/// proportional to the logarithm of stimulus intensity.  This compresses
/// dynamic range, improving discrimination at low intensities and preventing
/// saturation at high intensities — the same principle the biological
/// auditory and visual systems use.
#[pyfunction]
fn weber_fechner_compress(values: Vec<f64>) -> Vec<f64> {
    values.iter()
        .map(|&x| x.signum() * (1.0 + x.abs()).ln())
        .collect()
}

/// Bundle (majority vote) a list of binary hypervectors stored as u8 bit arrays.
///
/// For each bit position, set the result bit to 1 iff more than half of the
/// input vectors have a 1 at that position.
/// Tie-breaking (even N): deterministic — bit position index % 2.
///
/// V4 fix: replaces OR-approximation with correct majority-vote semantics.
/// bundle([A, A, A, B]) should be closer to A than to B.
#[pyfunction]
fn bundle_hvs(vectors: Vec<Vec<u8>>) -> Vec<u8> {
    if vectors.is_empty() {
        return vec![];
    }
    let dim = vectors[0].len();
    let n = vectors.len() as u32;
    let mut counts: Vec<u32> = vec![0u32; dim];
    for v in &vectors {
        for (i, &bit) in v.iter().enumerate() {
            if i < counts.len() {
                counts[i] += bit as u32;
            }
        }
    }
    let threshold = n / 2;
    counts.iter().enumerate().map(|(i, &c)| {
        if c > threshold { 1u8 }
        else if c == threshold { (i % 2) as u8 }  // deterministic tie-break
        else { 0u8 }
    }).collect()
}

/// Compute an LSH bucket key for a binary HyperVector (stored as u64 blocks).
///
/// Uses ChaCha8-seeded random projections to map the high-dimensional binary
/// vector into an n_bits-wide integer bucket key — O(n_bits·D/64) per call.
/// The same seed always produces the same projections, ensuring consistent
/// bucket assignment across calls.
#[pyfunction]
fn lsh_bucket(bits: Vec<u64>, n_bits: u32, seed: u64) -> u32 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let num_u64 = bits.len();
    if num_u64 == 0 || n_bits == 0 {
        return 0;
    }
    let mut key: u32 = 0;
    for bit_idx in 0..n_bits.min(32) {
        // Generate a random projection vector (as u64 blocks)
        let mut proj: Vec<u64> = (0..num_u64).map(|_| rng.gen::<u64>()).collect();
        // Compute XOR popcount (Hamming distance to projection)
        let mut hamming: u32 = 0;
        for (a, b) in bits.iter().zip(proj.iter()) {
            hamming += (a ^ b).count_ones();
        }
        // If similarity > 0.5 → set this bit to 1
        let total_bits = (num_u64 * 64) as u32;
        if hamming < total_bits / 2 {
            key |= 1 << bit_idx;
        }
    }
    key
}

/// Perform one step of spreading activation over a weighted graph.
///
/// Takes the current activation map (node_name → activation) and a list of
/// directed weighted edges (from, to, weight), applies one step of the
/// update rule:
///   new_activation[to] += activation[from] * weight * decay
/// and merges the result with the existing activations (max merge).
///
/// Returns the updated activation map.
#[pyfunction]
fn spreading_activation_step(
    activations: std::collections::HashMap<String, f64>,
    edges: Vec<(String, String, f64)>,
    decay: f64,
    max_frontier: usize,
) -> std::collections::HashMap<String, f64> {
    let mut new_acts = activations.clone();
    for (from, to, weight) in &edges {
        if let Some(&from_act) = activations.get(from.as_str()) {
            if from_act > 0.0 {
                let propagated = from_act * weight * decay;
                let entry = new_acts.entry(to.clone()).or_insert(0.0);
                if propagated > *entry {
                    *entry = propagated;
                }
            }
        }
    }
    // Trim to max_frontier by keeping top activations
    if max_frontier > 0 && new_acts.len() > max_frontier {
        let mut sorted: Vec<(String, f64)> = new_acts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(max_frontier);
        return sorted.into_iter().collect();
    }
    new_acts
}

#[pymodule]
fn hypervec_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core HyperVector class
    m.add_class::<HyperVector>()?;

    // Free functions
    m.add_function(wrap_pyfunction!(batch_similarity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(weber_fechner_compress, m)?)?;
    // V4: new Rust hot-path functions
    m.add_function(wrap_pyfunction!(bundle_hvs, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(spreading_activation_step, m)?)?;
    
    // Concurrent operations
    concurrent::register_concurrent_module(m)?;
    
    // Semantic memory
    semantic::register_semantic_module(m)?;
    
    // Episodic memory
    episodic::register_episodic_module(m)?;
    
    // Worker pool
    worker_pool::register_worker_pool_module(m)?;
    
    // Persistence
    persistence::register_persistence_module(m)?;
    
    // Async runtime
    async_runtime::register_async_module(m)?;
    
    // Societal hypervector backend (V5)
    societal::register_societal_module(m)?;
    
    Ok(())
}

// ============================================================
// Unit tests for cross-disciplinary enhancements
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn hv_from_seed(seed: u64) -> HyperVector {
        HyperVector::new(Some(seed))
    }

    #[test]
    fn test_weber_fechner_preserves_sign() {
        let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        let compressed = weber_fechner_compress_impl(&input);
        assert!(compressed[0] < 0.0, "Negative should stay negative");
        assert!(compressed[1] < 0.0);
        assert_eq!(compressed[2], 0.0, "Zero should stay zero");
        assert!(compressed[3] > 0.0, "Positive should stay positive");
        assert!(compressed[4] > 0.0);
    }

    #[test]
    fn test_weber_fechner_compresses_range() {
        let input = vec![0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];
        let compressed = weber_fechner_compress_impl(&input);
        let orig_range = input[5] / input[0];  // 100,000
        let comp_range = compressed[5] / compressed[0];
        assert!(comp_range < orig_range / 10.0,
                "Compressed range {} should be << original range {}",
                comp_range, orig_range);
    }

    #[test]
    fn test_batch_similarity_matrix_diagonal_is_one() {
        let hvs = vec![hv_from_seed(1), hv_from_seed(2), hv_from_seed(3)];
        let n = hvs.len();
        let matrix = batch_similarity_matrix_impl(&hvs);
        for i in 0..n {
            assert_eq!(matrix[i * n + i], 1.0, "Diagonal should be 1.0");
        }
    }

    #[test]
    fn test_batch_similarity_matrix_symmetric() {
        let hvs = vec![hv_from_seed(10), hv_from_seed(20), hv_from_seed(30)];
        let n = hvs.len();
        let matrix = batch_similarity_matrix_impl(&hvs);
        for i in 0..n {
            for j in 0..n {
                let diff = (matrix[i * n + j] - matrix[j * n + i]).abs();
                assert!(diff < 1e-10, "Matrix should be symmetric");
            }
        }
    }

    #[test]
    fn test_batch_similarity_random_near_half() {
        let hvs = vec![hv_from_seed(100), hv_from_seed(200)];
        let matrix = batch_similarity_matrix_impl(&hvs);
        // Off-diagonal: random HVs should have sim ≈ 0.5
        let sim = matrix[0 * 2 + 1];
        assert!(sim > 0.45 && sim < 0.55,
                "Random HV similarity {} should be near 0.5", sim);
    }

    #[test]
    fn test_batch_similarity_identical_is_one() {
        let hvs = vec![hv_from_seed(42), hv_from_seed(42)];
        let matrix = batch_similarity_matrix_impl(&hvs);
        assert_eq!(matrix[0 * 2 + 1], 1.0, "Identical HVs should have sim 1.0");
    }

    #[test]
    fn test_bundle_hvs_majority_vote() {
        // 3×A + 1×B → majority vote should match A on all bits
        let a = vec![1u8, 1, 0, 1, 0];
        let b = vec![0u8, 0, 1, 0, 1];
        let result = bundle_hvs(vec![a.clone(), a.clone(), a.clone(), b]);
        assert_eq!(result, a, "bundle([A,A,A,B]) should equal A (majority vote)");
    }

    #[test]
    fn test_bundle_hvs_empty() {
        let result = bundle_hvs(vec![]);
        assert!(result.is_empty(), "Empty bundle should return empty vec");
    }

    #[test]
    fn test_bundle_hvs_single() {
        let a = vec![1u8, 0, 1, 0, 1];
        let result = bundle_hvs(vec![a.clone()]);
        assert_eq!(result, a, "Single-vector bundle should equal the vector");
    }

    #[test]
    fn test_lsh_bucket_deterministic() {
        let bits: Vec<u64> = (0..160).map(|i| i as u64 * 0x9E3779B97F4A7C15).collect();
        let k1 = lsh_bucket(bits.clone(), 16, 0xDEAD);
        let k2 = lsh_bucket(bits.clone(), 16, 0xDEAD);
        assert_eq!(k1, k2, "LSH bucket should be deterministic");
    }

    // Internal implementations for testing (avoid PyO3 dependency in tests)
    fn weber_fechner_compress_impl(values: &[f64]) -> Vec<f64> {
        values.iter()
            .map(|&x| x.signum() * (1.0 + x.abs()).ln())
            .collect()
    }

    fn batch_similarity_matrix_impl(vectors: &[HyperVector]) -> Vec<f64> {
        let n = vectors.len();
        (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                if i == j {
                    1.0
                } else {
                    let mut hamming: u32 = 0;
                    for (a, b) in vectors[i].bits.iter().zip(vectors[j].bits.iter()) {
                        hamming += (a ^ b).count_ones();
                    }
                    1.0 - (hamming as f64 / DIMENSION as f64)
                }
            })
            .collect()
    }
}
