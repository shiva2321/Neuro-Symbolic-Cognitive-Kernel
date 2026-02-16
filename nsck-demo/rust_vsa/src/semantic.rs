/// Concurrent semantic memory with parallel spreading activation
///
/// This module provides thread-safe semantic memory operations:
/// - Parallel spreading activation
/// - Concurrent concept graph operations
/// - Thread-safe relation management

use pyo3::prelude::*;
use rayon::prelude::*;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use crate::HyperVector;

/// Thread-safe semantic memory with concurrent access
#[pyclass(module = "hypervec_rs")]
#[derive(Clone)]
pub struct SemanticMemoryConcurrent {
    /// Concept name -> HyperVector mapping (concurrent access)
    concepts: Arc<DashMap<String, HyperVector>>,
    
    /// Graph structure: concept -> set of neighbors (concurrent access)
    /// Each entry represents outgoing edges from a concept
    graph: Arc<DashMap<String, HashSet<String>>>,
    
    /// Reverse graph for incoming edges
    reverse_graph: Arc<DashMap<String, HashSet<String>>>,
}

#[pymethods]
impl SemanticMemoryConcurrent {
    #[new]
    pub fn new() -> Self {
        SemanticMemoryConcurrent {
            concepts: Arc::new(DashMap::new()),
            graph: Arc::new(DashMap::new()),
            reverse_graph: Arc::new(DashMap::new()),
        }
    }

    /// Add a concept with its hypervector (thread-safe)
    pub fn add_concept(&self, name: String, hv: HyperVector) {
        self.concepts.insert(name.clone(), hv);
        // Initialize empty neighbor sets if not present
        self.graph.entry(name.clone()).or_insert_with(HashSet::new);
        self.reverse_graph.entry(name).or_insert_with(HashSet::new);
    }

    /// Get a concept's hypervector
    fn get_concept(&self, name: &str) -> Option<HyperVector> {
        self.concepts.get(name).map(|entry| entry.value().clone())
    }

    /// Add a directed edge from source to target (thread-safe)
    fn add_relation(&self, source: String, target: String) {
        // Add to forward graph
        self.graph
            .entry(source.clone())
            .and_modify(|neighbors| {
                neighbors.insert(target.clone());
            })
            .or_insert_with(|| {
                let mut set = HashSet::new();
                set.insert(target.clone());
                set
            });

        // Add to reverse graph
        self.reverse_graph
            .entry(target.clone())
            .and_modify(|neighbors| {
                neighbors.insert(source.clone());
            })
            .or_insert_with(|| {
                let mut set = HashSet::new();
                set.insert(source);
                set
            });
    }

    /// Get neighbors of a concept (outgoing edges)
    fn get_neighbors(&self, concept: &str) -> Vec<String> {
        self.graph
            .get(concept)
            .map(|entry| entry.value().iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get incoming neighbors (reverse edges)
    fn get_incoming_neighbors(&self, concept: &str) -> Vec<String> {
        self.reverse_graph
            .get(concept)
            .map(|entry| entry.value().iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get number of concepts
    fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Get number of relations (edges)
    fn relation_count(&self) -> usize {
        self.graph.iter().map(|entry| entry.value().len()).sum()
    }

    /// Parallel spreading activation algorithm
    /// 
    /// Args:
    ///     start_concepts: Initial concepts to activate (with value 1.0)
    ///     steps: Number of spreading steps
    ///     decay: Decay factor per step (0.0 to 1.0)
    ///     min_activation: Minimum activation threshold (prune below this)
    ///     bidirectional: Whether to spread in both directions
    ///
    /// Returns:
    ///     Dictionary mapping concept names to final activation values
    #[pyo3(signature = (start_concepts, steps = 3, decay = 0.7, min_activation = 0.01, bidirectional = false))]
    pub fn parallel_spread_activation(
        &self,
        start_concepts: Vec<String>,
        steps: usize,
        decay: f64,
        min_activation: f64,
        bidirectional: bool,
    ) -> HashMap<String, f64> {
        // Initialize activation map
        let mut activation: DashMap<String, f64> = DashMap::new();
        for concept in start_concepts {
            activation.insert(concept, 1.0);
        }

        // Spreading activation loop
        for step in 0..steps {
            let current_activation: Vec<_> = activation
                .iter()
                .map(|entry| (entry.key().clone(), *entry.value()))
                .collect();

            // Parallel spreading for each activated concept
            let new_activations: Vec<_> = current_activation
                .par_iter()
                .filter(|(_, act)| *act >= min_activation)
                .flat_map(|(concept, act)| {
                    let mut spreads = Vec::new();
                    
                    // Get neighbors (outgoing edges)
                    if let Some(neighbors_entry) = self.graph.get(concept) {
                        let neighbors = neighbors_entry.value();
                        let num_neighbors = neighbors.len();
                        
                        if num_neighbors > 0 {
                            let spread_val = (act * decay) / num_neighbors as f64;
                            
                            for neighbor in neighbors.iter() {
                                spreads.push((neighbor.clone(), spread_val));
                            }
                        }
                    }

                    // Optionally spread backwards (incoming edges)
                    if bidirectional {
                        if let Some(incoming_entry) = self.reverse_graph.get(concept) {
                            let incoming = incoming_entry.value();
                            let num_incoming = incoming.len();
                            
                            if num_incoming > 0 {
                                let spread_val = (act * decay * 0.5) / num_incoming as f64;
                                
                                for neighbor in incoming.iter() {
                                    spreads.push((neighbor.clone(), spread_val));
                                }
                            }
                        }
                    }

                    spreads
                })
                .collect();

            // Accumulate new activations
            for (concept, spread_val) in new_activations {
                activation
                    .entry(concept)
                    .and_modify(|v| *v += spread_val)
                    .or_insert(spread_val);
            }

            // Prune low activations for efficiency (optional, can be done after all steps)
            if step < steps - 1 {
                activation.retain(|_, v| *v >= min_activation);
            }
        }

        // Convert to regular HashMap for return
        activation
            .into_iter()
            .map(|(k, v)| (k, v))
            .collect()
    }

    /// Get top-k most activated concepts after spreading
    #[pyo3(signature = (start_concepts, k = 10, steps = 3, decay = 0.7, bidirectional = false))]
    fn get_activated_concepts(
        &self,
        start_concepts: Vec<String>,
        k: usize,
        steps: usize,
        decay: f64,
        bidirectional: bool,
    ) -> Vec<(String, f64)> {
        let activation = self.parallel_spread_activation(
            start_concepts,
            steps,
            decay,
            0.01,
            bidirectional,
        );

        let mut results: Vec<_> = activation.into_iter().collect();
        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Parallel semantic search: find concepts most similar to query
    #[pyo3(signature = (query_hv, k = 10))]
    pub fn parallel_semantic_search(&self, query_hv: &HyperVector, k: usize) -> Vec<(String, f64)> {
        // Collect all concepts for parallel processing
        let concepts: Vec<_> = self.concepts
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        // Parallel similarity computation
        let mut results: Vec<(String, f64)> = concepts
            .par_iter()
            .map(|(name, hv)| (name.clone(), query_hv.similarity(hv)))
            .collect();

        // Sort and return top-k
        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Combined search: semantic similarity + spreading activation
    /// 
    /// First finds semantically similar concepts, then spreads activation from them
    #[pyo3(signature = (query_hv, k = 10, spread_steps = 2, spread_decay = 0.7))]
    fn hybrid_search(
        &self,
        query_hv: &HyperVector,
        k: usize,
        spread_steps: usize,
        spread_decay: f64,
    ) -> Vec<(String, f64)> {
        // Step 1: Find top-k semantically similar concepts
        let similar = self.parallel_semantic_search(query_hv, k);
        let start_concepts: Vec<String> = similar.iter().map(|(name, _)| name.clone()).collect();

        // Step 2: Spread activation from similar concepts
        let activation = self.parallel_spread_activation(
            start_concepts,
            spread_steps,
            spread_decay,
            0.01,
            false,
        );

        // Step 3: Combine similarity scores with activation
        let mut results: Vec<(String, f64)> = activation
            .into_iter()
            .map(|(concept, act)| {
                // Combine semantic similarity with activation
                let sim = if let Some(hv) = self.get_concept(&concept) {
                    query_hv.similarity(&hv)
                } else {
                    0.0
                };
                // Weighted combination: 70% similarity, 30% activation
                (concept, 0.7 * sim + 0.3 * act)
            })
            .collect();

        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Clear all concepts and relations
    fn clear(&self) {
        self.concepts.clear();
        self.graph.clear();
        self.reverse_graph.clear();
    }

    /// Get all concept names
    fn get_all_concepts(&self) -> Vec<String> {
        self.concepts.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get statistics about the semantic memory
    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("concepts".to_string(), self.concept_count());
        stats.insert("relations".to_string(), self.relation_count());
        
        // Calculate average degree
        let total_degree: usize = self.graph.iter().map(|e| e.value().len()).sum();
        let avg_degree = if self.concept_count() > 0 {
            total_degree / self.concept_count()
        } else {
            0
        };
        stats.insert("avg_degree".to_string(), avg_degree);
        
        stats
    }
}

// Register semantic memory module
pub fn register_semantic_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<SemanticMemoryConcurrent>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_concept_addition() {
        let mem = SemanticMemoryConcurrent::new();
        
        // Concurrent concept additions
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let mem_clone = mem.clone();
                std::thread::spawn(move || {
                    for j in 0..100 {
                        let name = format!("concept_{}_{}", i, j);
                        let hv = HyperVector::new(Some((i * 100 + j) as u64));
                        mem_clone.add_concept(name, hv);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(mem.concept_count(), 1000);
    }

    #[test]
    fn test_parallel_spreading_activation() {
        let mem = SemanticMemoryConcurrent::new();
        
        // Create a simple graph: A -> B -> C -> D
        for i in 0..4 {
            let name = format!("concept_{}", i);
            let hv = HyperVector::new(Some(i));
            mem.add_concept(name, hv);
        }
        
        mem.add_relation("concept_0".to_string(), "concept_1".to_string());
        mem.add_relation("concept_1".to_string(), "concept_2".to_string());
        mem.add_relation("concept_2".to_string(), "concept_3".to_string());

        // Spread from concept_0
        let activation = mem.parallel_spread_activation(
            vec!["concept_0".to_string()],
            3,
            0.7,
            0.01,
            false,
        );

        // Check that activation reaches all concepts
        assert!(activation.contains_key("concept_0"));
        assert!(activation.contains_key("concept_1"));
        assert!(activation.contains_key("concept_2"));
        assert!(activation.contains_key("concept_3"));
        
        // Check that concept_0 has activation (starting point)
        assert!(activation["concept_0"] > 0.9);
        
        // Check that later concepts have progressively lower activation
        // (due to decay and distance from source)
        assert!(activation["concept_1"] > activation["concept_2"]);
        assert!(activation["concept_2"] > activation["concept_3"]);
    }

    #[test]
    fn test_parallel_semantic_search() {
        let mem = SemanticMemoryConcurrent::new();
        
        // Add concepts
        for i in 0..100 {
            let name = format!("concept_{}", i);
            let hv = HyperVector::new(Some(i));
            mem.add_concept(name, hv);
        }

        // Search with a query
        let query = HyperVector::new(Some(42));
        let results = mem.parallel_semantic_search(&query, 10);

        assert_eq!(results.len(), 10);
        // Check sorted by similarity
        for i in 1..results.len() {
            assert!(results[i-1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_bidirectional_spreading() {
        let mem = SemanticMemoryConcurrent::new();
        
        // Create a graph with bidirectional potential: A -> B, C -> B
        mem.add_concept("A".to_string(), HyperVector::new(Some(1)));
        mem.add_concept("B".to_string(), HyperVector::new(Some(2)));
        mem.add_concept("C".to_string(), HyperVector::new(Some(3)));
        
        mem.add_relation("A".to_string(), "B".to_string());
        mem.add_relation("C".to_string(), "B".to_string());

        // Spread from B with bidirectional
        let activation = mem.parallel_spread_activation(
            vec!["B".to_string()],
            1,
            0.7,
            0.01,
            true,
        );

        // Should reach both A and C through reverse edges
        assert!(activation.contains_key("A"));
        assert!(activation.contains_key("C"));
    }
}
