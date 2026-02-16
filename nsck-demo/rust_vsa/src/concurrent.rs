/// Concurrent VSA operations module
/// 
/// This module provides thread-safe, parallel implementations of:
/// - Parallel similarity search
/// - Concurrent hypervector registry
/// - Lock-free data structures for cognitive memory

use pyo3::prelude::*;
use rayon::prelude::*;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use crate::HyperVector;

const TOP_K_DEFAULT: usize = 10;

/// Thread-safe hypervector registry with concurrent access
/// Uses DashMap for lock-free reads and minimal write contention
#[pyclass(module = "hypervec_rs")]
#[derive(Clone)]
pub struct HyperVectorRegistry {
    vectors: Arc<DashMap<String, HyperVector>>,
}

#[pymethods]
impl HyperVectorRegistry {
    #[new]
    fn new() -> Self {
        HyperVectorRegistry {
            vectors: Arc::new(DashMap::new()),
        }
    }

    /// Register a hypervector with a name (thread-safe)
    fn register(&self, name: String, hv: HyperVector) {
        self.vectors.insert(name, hv);
    }

    /// Get a hypervector by name (thread-safe)
    fn get(&self, name: &str) -> Option<HyperVector> {
        self.vectors.get(name).map(|entry| entry.value().clone())
    }

    /// Remove a hypervector by name (thread-safe)
    fn remove(&self, name: &str) -> bool {
        self.vectors.remove(name).is_some()
    }

    /// Get all registered names
    fn keys(&self) -> Vec<String> {
        self.vectors.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get registry size
    fn size(&self) -> usize {
        self.vectors.len()
    }

    /// Clear all entries
    fn clear(&self) {
        self.vectors.clear();
    }

    /// Parallel nearest neighbor search across all registered vectors
    /// Returns list of (name, similarity) tuples sorted by similarity (descending)
    #[pyo3(signature = (query, k = TOP_K_DEFAULT))]
    fn nearest_neighbors(&self, query: &HyperVector, k: usize) -> Vec<(String, f64)> {
        // Collect into vec for parallel processing
        let entries: Vec<_> = self.vectors.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        // Parallel similarity computation
        let mut results: Vec<(String, f64)> = entries
            .par_iter()
            .map(|(name, hv)| (name.clone(), query.similarity(hv)))
            .collect();

        // Sort by similarity (descending) and take top-k
        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Batch similarity search for multiple queries (parallelized)
    fn batch_nearest_neighbors(&self, queries: Vec<HyperVector>, k: usize) -> Vec<Vec<(String, f64)>> {
        queries
            .par_iter()
            .map(|query| self.nearest_neighbors(query, k))
            .collect()
    }
}

/// Parallel similarity search across a collection of hypervectors
/// Returns indices and similarities of top-k matches
#[pyfunction]
#[pyo3(signature = (query, candidates, k = TOP_K_DEFAULT))]
pub fn parallel_similarity_search(
    query: &HyperVector,
    candidates: Vec<HyperVector>,
    k: usize,
) -> Vec<(usize, f64)> {
    // Parallel computation of all similarities
    let mut results: Vec<(usize, f64)> = candidates
        .par_iter()
        .enumerate()
        .map(|(idx, hv)| (idx, query.similarity(hv)))
        .collect();

    // Sort by similarity (descending) and take top-k
    results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Batch parallel similarity search for multiple queries
/// Returns Vec<Vec<(idx, similarity)>> where outer vec is per-query
#[pyfunction]
pub fn batch_parallel_similarity_search(
    queries: Vec<HyperVector>,
    candidates: Vec<HyperVector>,
    k: usize,
) -> Vec<Vec<(usize, f64)>> {
    queries
        .par_iter()
        .map(|query| parallel_similarity_search(query, candidates.clone(), k))
        .collect()
}

/// Parallel bundle operation: majority vote across multiple vectors
/// Uses Rayon for parallel bit counting
#[pyfunction]
pub fn parallel_bundle(vectors: Vec<HyperVector>) -> PyResult<HyperVector> {
    if vectors.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot bundle empty vector list"
        ));
    }

    if vectors.len() == 1 {
        return Ok(vectors[0].clone());
    }

    let num_u64 = vectors[0].bits.len();
    let num_vectors = vectors.len();
    let threshold = (num_vectors / 2) as u32;

    // Parallel computation of majority vote for each u64 block
    let result_bits: Vec<u64> = (0..num_u64)
        .into_par_iter()
        .map(|block_idx| {
            let mut result_block = 0u64;
            
            // For each bit position in this block
            for bit_pos in 0..64 {
                let mut count = 0u32;
                
                // Count how many vectors have this bit set
                for hv in &vectors {
                    if (hv.bits[block_idx] >> bit_pos) & 1 == 1 {
                        count += 1;
                    }
                }
                
                // Majority vote
                if count > threshold {
                    result_block |= 1u64 << bit_pos;
                }
            }
            
            result_block
        })
        .collect();

    Ok(HyperVector { bits: result_bits })
}

/// Thread-safe spreading activation accumulator
/// Used for concurrent accumulation of activation values
#[pyclass(module = "hypervec_rs")]
pub struct ActivationAccumulator {
    activations: Arc<DashMap<String, f64>>,
}

#[pymethods]
impl ActivationAccumulator {
    #[new]
    fn new() -> Self {
        ActivationAccumulator {
            activations: Arc::new(DashMap::new()),
        }
    }

    /// Add activation to a concept (thread-safe accumulation)
    fn add_activation(&self, concept: String, value: f64) {
        self.activations
            .entry(concept)
            .and_modify(|v| *v += value)
            .or_insert(value);
    }

    /// Get activation value for a concept
    fn get_activation(&self, concept: &str) -> f64 {
        self.activations
            .get(concept)
            .map(|entry| *entry.value())
            .unwrap_or(0.0)
    }

    /// Get all activations as a dictionary
    fn get_all(&self) -> Vec<(String, f64)> {
        self.activations
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Get top-k activated concepts
    fn get_top_k(&self, k: usize) -> Vec<(String, f64)> {
        let mut results: Vec<(String, f64)> = self.get_all();
        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Clear all activations
    fn clear(&self) {
        self.activations.clear();
    }

    /// Get number of activated concepts
    fn size(&self) -> usize {
        self.activations.len()
    }
}

// Register concurrent types with PyO3 module
pub fn register_concurrent_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<HyperVectorRegistry>()?;
    m.add_class::<ActivationAccumulator>()?;
    m.add_function(wrap_pyfunction!(parallel_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(batch_parallel_similarity_search, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_bundle, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_concurrent_insert() {
        let registry = HyperVectorRegistry::new();
        
        // Simulate concurrent inserts
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let reg = registry.clone();
                std::thread::spawn(move || {
                    for j in 0..100 {
                        let name = format!("vec_{}_{}", i, j);
                        let hv = HyperVector::new(Some((i * 100 + j) as u64));
                        reg.register(name, hv);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.size(), 1000);
    }

    #[test]
    fn test_parallel_similarity_search() {
        let query = HyperVector::new(Some(42));
        let candidates: Vec<_> = (0..1000)
            .map(|i| HyperVector::new(Some(i)))
            .collect();

        let results = parallel_similarity_search(&query, candidates, 10);
        
        assert_eq!(results.len(), 10);
        // Check that results are sorted by similarity (descending)
        for i in 1..results.len() {
            assert!(results[i-1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_activation_accumulator_concurrent() {
        let acc = ActivationAccumulator::new();
        
        // Concurrent accumulation
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let acc_clone = acc.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        acc_clone.add_activation(format!("concept_{}", i % 5), 0.1);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Each of 5 concepts should have accumulated 10*100*0.1 = 100.0
        for i in 0..5 {
            let val = acc.get_activation(&format!("concept_{}", i));
            assert!((val - 200.0).abs() < 1e-6); // 10 threads * 100 iterations * 0.1
        }
    }
}
