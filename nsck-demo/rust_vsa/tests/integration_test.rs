/// Property-based tests for concurrent VSA operations
///
/// These tests use proptest to verify correctness under randomized inputs
/// and concurrent stress conditions

use proptest::prelude::*;

// Import from the library
// Note: We need to declare the module path correctly
extern crate rust_vsa;
use rust_vsa::hypervec_rs::*;

#[cfg(test)]
mod property_tests {
    use super::*;

    // Helper to create HyperVector from seed
    fn hv_from_seed(seed: u64) -> HyperVector {
        HyperVector::new(Some(seed))
    }

    proptest! {
        /// Test that HyperVector operations are deterministic
        #[test]
        fn test_hypervec_deterministic(seed in 0u64..10000) {
            let hv1 = hv_from_seed(seed);
            let hv2 = hv_from_seed(seed);
            
            // Same seed should produce identical vectors
            prop_assert_eq!(hv1.similarity(&hv2), 1.0);
        }

        /// Test XOR operation is reversible
        #[test]
        fn test_xor_reversible(seed1 in 0u64..10000, seed2 in 0u64..10000) {
            let a = hv_from_seed(seed1);
            let b = hv_from_seed(seed2);
            
            // XOR binding and unbinding
            let bound = a.xor(&b);
            let unbound = bound.xor(&b);
            
            // Should recover original (or very close)
            let similarity = a.similarity(&unbound);
            prop_assert!(similarity > 0.99, "Similarity {} should be > 0.99", similarity);
        }

        /// Test permutation is reversible
        #[test]
        fn test_permute_reversible(seed in 0u64..10000, shift in -100i32..100) {
            let hv = hv_from_seed(seed);
            
            let permuted = hv.permute(shift);
            let restored = permuted.permute_inverse(shift);
            
            prop_assert_eq!(hv.similarity(&restored), 1.0);
        }

        /// Test similarity is symmetric
        #[test]
        fn test_similarity_symmetric(seed1 in 0u64..10000, seed2 in 0u64..10000) {
            let a = hv_from_seed(seed1);
            let b = hv_from_seed(seed2);
            
            let sim_ab = a.similarity(&b);
            let sim_ba = b.similarity(&a);
            
            prop_assert_eq!(sim_ab, sim_ba);
        }

        /// Test similarity is in valid range [0, 1]
        #[test]
        fn test_similarity_range(seed1 in 0u64..10000, seed2 in 0u64..10000) {
            let a = hv_from_seed(seed1);
            let b = hv_from_seed(seed2);
            
            let sim = a.similarity(&b);
            prop_assert!(sim >= 0.0 && sim <= 1.0);
        }

        /// Test bundle operation maintains valid vectors
        #[test]
        fn test_bundle_validity(seed1 in 0u64..10000, seed2 in 0u64..10000) {
            let a = hv_from_seed(seed1);
            let b = hv_from_seed(seed2);
            
            let bundled = a.bundle(&b);
            
            // Bundled vector should be similar to both inputs
            let sim_a = bundled.similarity(&a);
            let sim_b = bundled.similarity(&b);
            
            prop_assert!(sim_a > 0.4 && sim_a < 1.0);
            prop_assert!(sim_b > 0.4 && sim_b < 1.0);
        }

        /// Test weighted bundle respects weights
        #[test]
        fn test_weighted_bundle(seed1 in 0u64..10000, seed2 in 0u64..10000, weight in 0.0f64..1.0) {
            let a = hv_from_seed(seed1);
            let b = hv_from_seed(seed2);
            
            let bundled = a.weighted_bundle(&b, weight, Some(42));
            
            let sim_a = bundled.similarity(&a);
            let sim_b = bundled.similarity(&b);
            
            // Higher weight should mean more similar to a
            if weight > 0.7 {
                prop_assert!(sim_a > sim_b, "weight={}, sim_a={}, sim_b={}", weight, sim_a, sim_b);
            } else if weight < 0.3 {
                prop_assert!(sim_b > sim_a, "weight={}, sim_a={}, sim_b={}", weight, sim_a, sim_b);
            }
        }

        /// Test LSH hash stability
        #[test]
        fn test_lsh_hash_stable(seed in 0u64..10000, hash_seed in 0u64..10000) {
            let hv = hv_from_seed(seed);
            
            let hash1 = hv.lsh_hash(hash_seed, 32);
            let hash2 = hv.lsh_hash(hash_seed, 32);
            
            prop_assert_eq!(hash1, hash2);
        }

        /// Test parallel similarity search consistency
        #[test]
        fn test_parallel_search_consistency(
            query_seed in 0u64..1000,
            candidate_seeds in prop::collection::vec(0u64..1000, 10..100)
        ) {
            let query = hv_from_seed(query_seed);
            let candidates: Vec<_> = candidate_seeds.iter().map(|s| hv_from_seed(*s)).collect();
            
            let results = parallel_similarity_search(&query, candidates.clone(), 10);
            
            // Results should be sorted by similarity (descending)
            for i in 1..results.len() {
                prop_assert!(results[i-1].1 >= results[i].1);
            }
            
            // Top result should have highest similarity
            if !results.is_empty() {
                let (idx, sim) = results[0];
                prop_assert!(idx < candidates.len());
                
                // Verify similarity is correct
                let computed_sim = query.similarity(&candidates[idx]);
                prop_assert!((sim - computed_sim).abs() < 1e-10);
            }
        }

        /// Test parallel bundle correctness
        #[test]
        fn test_parallel_bundle_majority(
            seeds in prop::collection::vec(0u64..1000, 3..20)
        ) {
            let vectors: Vec<_> = seeds.iter().map(|s| hv_from_seed(*s)).collect();
            
            let bundled = parallel_bundle(vectors.clone()).expect("Bundle failed");
            
            // Bundled should be similar to all inputs
            for hv in &vectors {
                let sim = bundled.similarity(hv);
                prop_assert!(sim > 0.3, "Similarity {} should be > 0.3", sim);
            }
        }

        /// Test registry concurrent operations
        #[test]
        fn test_registry_concurrent(
            entries in prop::collection::vec((0u64..1000, 0u64..1000), 10..100)
        ) {
            let registry = HyperVectorRegistry::new();
            
            // Add all entries
            for (i, seed) in entries.iter().enumerate() {
                let name = format!("vec_{}", i);
                let hv = hv_from_seed(*seed);
                registry.register(name, hv);
            }
            
            prop_assert_eq!(registry.size(), entries.len());
            
            // Check all entries are retrievable
            for (i, _) in entries.iter().enumerate() {
                let name = format!("vec_{}", i);
                prop_assert!(registry.get(&name).is_some());
            }
        }

        /// Test activation accumulator correctness
        #[test]
        fn test_activation_accumulator(
            activations in prop::collection::vec((0u64..100, 0.0f64..1.0), 10..50)
        ) {
            let acc = ActivationAccumulator::new();
            
            // Add activations
            for (concept_id, value) in &activations {
                let concept = format!("concept_{}", concept_id);
                acc.add_activation(concept, *value);
            }
            
            // Verify accumulated values
            let mut expected: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
            for (concept_id, value) in &activations {
                let concept = format!("concept_{}", concept_id);
                *expected.entry(concept).or_insert(0.0) += value;
            }
            
            for (concept, expected_value) in expected {
                let actual_value = acc.get_activation(&concept);
                prop_assert!((actual_value - expected_value).abs() < 1e-10);
            }
        }
    }

    #[cfg(test)]
    mod stress_tests {
        use super::*;
        use std::sync::Arc;
        use std::thread;

        #[test]
        fn test_concurrent_registry_inserts() {
            let registry = Arc::new(HyperVectorRegistry::new());
            let num_threads = 10;
            let inserts_per_thread = 100;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|thread_id| {
                    let reg = registry.clone();
                    thread::spawn(move || {
                        for i in 0..inserts_per_thread {
                            let name = format!("vec_{}_{}", thread_id, i);
                            let hv = hv_from_seed((thread_id * 1000 + i) as u64);
                            reg.register(name, hv);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            assert_eq!(registry.size(), num_threads * inserts_per_thread);
        }

        #[test]
        fn test_concurrent_similarity_searches() {
            let registry = HyperVectorRegistry::new();
            
            // Populate registry
            for i in 0..1000 {
                let name = format!("vec_{}", i);
                let hv = hv_from_seed(i);
                registry.register(name, hv);
            }
            
            let registry = Arc::new(registry);
            let num_threads = 10;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|thread_id| {
                    let reg = registry.clone();
                    thread::spawn(move || {
                        let query = hv_from_seed((thread_id * 100) as u64);
                        let results = reg.nearest_neighbors(&query, 10);
                        
                        // Verify results are valid
                        assert_eq!(results.len(), 10);
                        for i in 1..results.len() {
                            assert!(results[i-1].1 >= results[i].1);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn test_concurrent_activation_accumulation() {
            let acc = Arc::new(ActivationAccumulator::new());
            let num_threads = 10;
            let accumulations_per_thread = 100;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|thread_id| {
                    let acc_clone = acc.clone();
                    thread::spawn(move || {
                        for i in 0..accumulations_per_thread {
                            let concept = format!("concept_{}", i % 10);
                            acc_clone.add_activation(concept, 0.1);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            // Each of 10 concepts should have accumulated:
            // num_threads * (accumulations_per_thread / 10) * 0.1
            for i in 0..10 {
                let concept = format!("concept_{}", i);
                let value = acc.get_activation(&concept);
                let expected = (num_threads * accumulations_per_thread / 10) as f64 * 0.1;
                assert!((value - expected).abs() < 1e-10);
            }
        }
    }
}
