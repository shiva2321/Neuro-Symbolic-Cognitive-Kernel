/// Concurrent episodic memory with lock-free operations
///
/// This module provides thread-safe episodic memory:
/// - Lock-free episode storage with RwLock
/// - Parallel k-NN search
/// - Thread-safe episode retrieval

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::VecDeque;
use crate::HyperVector;

/// Episode data structure (immutable after creation)
#[pyclass(module = "hypervec_rs")]
#[derive(Clone, Debug)]
pub struct Episode {
    #[pyo3(get)]
    pub timestamp: f64,
    
    #[pyo3(get)]
    pub task_tag: String,
    
    /// HyperVector encoding of the situation
    pub situation_hv: HyperVector,
    
    #[pyo3(get)]
    pub action: String,
    
    #[pyo3(get)]
    pub outcome: String,
    
    #[pyo3(get)]
    pub reward: f64,
    
    #[pyo3(get)]
    pub impact_score: f64,
}

#[pymethods]
impl Episode {
    #[new]
    #[pyo3(signature = (timestamp, task_tag, situation_hv, action, outcome, reward, impact_score = 0.0))]
    pub fn new(
        timestamp: f64,
        task_tag: String,
        situation_hv: HyperVector,
        action: String,
        outcome: String,
        reward: f64,
        impact_score: f64,
    ) -> Self {
        Episode {
            timestamp,
            task_tag,
            situation_hv,
            action,
            outcome,
            reward,
            impact_score,
        }
    }

    /// Get the situation hypervector
    fn get_situation_hv(&self) -> HyperVector {
        self.situation_hv.clone()
    }

    /// Calculate similarity to another episode
    fn similarity(&self, other: &Episode) -> f64 {
        self.situation_hv.similarity(&other.situation_hv)
    }

    fn __repr__(&self) -> String {
        format!(
            "Episode(task={}, action={}, reward={:.2}, impact={:.2})",
            self.task_tag, self.action, self.reward, self.impact_score
        )
    }
}

/// Thread-safe episodic memory with concurrent access
#[pyclass(module = "hypervec_rs")]
#[derive(Clone)]
pub struct EpisodicMemoryConcurrent {
    /// Hot tier: recent and high-priority episodes in RAM
    /// Using RwLock for concurrent reads, exclusive writes
    hot_tier: Arc<RwLock<VecDeque<Episode>>>,
    
    /// Maximum size of hot tier
    max_hot_size: usize,
}

#[pymethods]
impl EpisodicMemoryConcurrent {
    #[new]
    #[pyo3(signature = (max_hot_size = 10000))]
    pub fn new(max_hot_size: usize) -> Self {
        EpisodicMemoryConcurrent {
            hot_tier: Arc::new(RwLock::new(VecDeque::with_capacity(max_hot_size))),
            max_hot_size,
        }
    }

    /// Add an episode to memory (thread-safe)
    /// Returns evicted episode if hot tier is full
    fn add_episode(&self, episode: Episode) -> Option<Episode> {
        let mut hot = self.hot_tier.write();
        
        hot.push_back(episode);
        
        // Evict oldest if exceeds capacity
        if hot.len() > self.max_hot_size {
            hot.pop_front()
        } else {
            None
        }
    }

    /// Get number of episodes in hot tier
    pub fn size(&self) -> usize {
        self.hot_tier.read().len()
    }

    /// Get all episodes (thread-safe read)
    fn get_all_episodes(&self) -> Vec<Episode> {
        self.hot_tier.read().iter().cloned().collect()
    }

    /// Parallel k-NN search: find k most similar episodes to query
    #[pyo3(signature = (query_hv, k = 10, task_filter = None))]
    pub fn parallel_knn_search(
        &self,
        query_hv: &HyperVector,
        k: usize,
        task_filter: Option<String>,
    ) -> Vec<(usize, f64, Episode)> {
        let hot = self.hot_tier.read();
        
        // Collect episodes (with optional filtering)
        let episodes: Vec<_> = hot.iter().enumerate().collect();

        // Parallel similarity computation
        let mut results: Vec<(usize, f64, Episode)> = episodes
            .par_iter()
            .filter(|(_, ep)| {
                if let Some(ref task) = task_filter {
                    &ep.task_tag == task
                } else {
                    true
                }
            })
            .map(|(idx, ep)| {
                let sim = query_hv.similarity(&ep.situation_hv);
                (*idx, sim, (*ep).clone())
            })
            .collect();

        // Sort by similarity (descending) and take top-k
        results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search episodes by task tag
    fn search_by_task(&self, task_tag: &str) -> Vec<Episode> {
        let hot = self.hot_tier.read();
        hot.iter()
            .filter(|ep| ep.task_tag == task_tag)
            .cloned()
            .collect()
    }

    /// Search episodes by reward range
    fn search_by_reward(&self, min_reward: f64, max_reward: f64) -> Vec<Episode> {
        let hot = self.hot_tier.read();
        hot.iter()
            .filter(|ep| ep.reward >= min_reward && ep.reward <= max_reward)
            .cloned()
            .collect()
    }

    /// Get episodes with highest impact scores
    fn get_high_impact_episodes(&self, k: usize) -> Vec<Episode> {
        let hot = self.hot_tier.read();
        let mut episodes: Vec<_> = hot.iter().cloned().collect();
        
        episodes.par_sort_by(|a, b| {
            b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        episodes.truncate(k);
        episodes
    }

    /// Get recent episodes
    fn get_recent_episodes(&self, n: usize) -> Vec<Episode> {
        let hot = self.hot_tier.read();
        let len = hot.len();
        let start = if len > n { len - n } else { 0 };
        hot.iter().skip(start).cloned().collect()
    }

    /// Clear all episodes
    fn clear(&self) {
        self.hot_tier.write().clear();
    }

    /// Get statistics about episodic memory
    fn get_stats(&self, py: Python) -> PyResult<Py<PyDict>> {
        let hot = self.hot_tier.read();
        let dict = PyDict::new(py);
        
        dict.set_item("total_episodes", hot.len())?;
        dict.set_item("max_capacity", self.max_hot_size)?;
        
        if !hot.is_empty() {
            let avg_reward: f64 = hot.iter().map(|ep| ep.reward).sum::<f64>() / hot.len() as f64;
            let avg_impact: f64 = hot.iter().map(|ep| ep.impact_score).sum::<f64>() / hot.len() as f64;
            
            dict.set_item("avg_reward", avg_reward)?;
            dict.set_item("avg_impact", avg_impact)?;
            
            // Get task distribution
            let mut task_counts = std::collections::HashMap::new();
            for ep in hot.iter() {
                *task_counts.entry(ep.task_tag.clone()).or_insert(0) += 1;
            }
            let task_dict = PyDict::new(py);
            for (task, count) in task_counts {
                task_dict.set_item(task, count)?;
            }
            dict.set_item("task_distribution", task_dict)?;
        }
        
        Ok(dict.into())
    }

    /// Batch add episodes (more efficient than individual adds)
    fn batch_add_episodes(&self, episodes: Vec<Episode>) -> Vec<Episode> {
        let mut hot = self.hot_tier.write();
        let mut evicted = Vec::new();
        
        for episode in episodes {
            hot.push_back(episode);
            
            if hot.len() > self.max_hot_size {
                if let Some(evicted_ep) = hot.pop_front() {
                    evicted.push(evicted_ep);
                }
            }
        }
        
        evicted
    }

    /// Parallel batch k-NN search for multiple queries
    fn batch_knn_search(
        &self,
        query_hvs: Vec<HyperVector>,
        k: usize,
    ) -> Vec<Vec<(usize, f64, Episode)>> {
        // Acquire read lock once for all queries
        let hot = self.hot_tier.read();
        let episodes: Vec<_> = hot.iter().cloned().collect();
        drop(hot); // Release lock

        // Process all queries in parallel
        query_hvs
            .par_iter()
            .map(|query_hv| {
                let mut results: Vec<(usize, f64, Episode)> = episodes
                    .iter()
                    .enumerate()
                    .map(|(idx, ep)| {
                        let sim = query_hv.similarity(&ep.situation_hv);
                        (idx, sim, ep.clone())
                    })
                    .collect();

                results.par_sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(k);
                results
            })
            .collect()
    }
}

// Register episodic memory module
pub fn register_episodic_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<Episode>()?;
    m.add_class::<EpisodicMemoryConcurrent>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_episode_addition() {
        let mem = EpisodicMemoryConcurrent::new(1000);
        
        // Concurrent episode additions
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let mem_clone = mem.clone();
                std::thread::spawn(move || {
                    for j in 0..100 {
                        let ep = Episode::new(
                            (i * 100 + j) as f64,
                            format!("task_{}", i),
                            HyperVector::new(Some((i * 100 + j) as u64)),
                            format!("action_{}", j),
                            "success".to_string(),
                            0.5,
                            0.3,
                        );
                        mem_clone.add_episode(ep);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(mem.size(), 1000); // Capped at max size
    }

    #[test]
    fn test_parallel_knn_search() {
        let mem = EpisodicMemoryConcurrent::new(1000);
        
        // Add episodes
        for i in 0..100 {
            let ep = Episode::new(
                i as f64,
                format!("task_{}", i % 5),
                HyperVector::new(Some(i)),
                format!("action_{}", i),
                "success".to_string(),
                0.5,
                0.3,
            );
            mem.add_episode(ep);
        }

        // Search with query
        let query = HyperVector::new(Some(42));
        let results = mem.parallel_knn_search(&query, 10, None);

        assert_eq!(results.len(), 10);
        // Check sorted by similarity
        for i in 1..results.len() {
            assert!(results[i-1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_task_filtering() {
        let mem = EpisodicMemoryConcurrent::new(1000);
        
        // Add episodes with different tasks
        for i in 0..50 {
            let ep = Episode::new(
                i as f64,
                "task_A".to_string(),
                HyperVector::new(Some(i)),
                format!("action_{}", i),
                "success".to_string(),
                0.5,
                0.3,
            );
            mem.add_episode(ep);
        }
        
        for i in 50..100 {
            let ep = Episode::new(
                i as f64,
                "task_B".to_string(),
                HyperVector::new(Some(i)),
                format!("action_{}", i),
                "success".to_string(),
                0.5,
                0.3,
            );
            mem.add_episode(ep);
        }

        // Search only task_A
        let query = HyperVector::new(Some(25));
        let results = mem.parallel_knn_search(&query, 10, Some("task_A".to_string()));

        assert_eq!(results.len(), 10);
        for (_, _, ep) in results {
            assert_eq!(ep.task_tag, "task_A");
        }
    }

    #[test]
    fn test_batch_operations() {
        let mem = EpisodicMemoryConcurrent::new(100);
        
        // Create batch of episodes
        let episodes: Vec<_> = (0..150)
            .map(|i| Episode::new(
                i as f64,
                format!("task_{}", i % 3),
                HyperVector::new(Some(i)),
                format!("action_{}", i),
                "success".to_string(),
                0.5,
                0.3,
            ))
            .collect();

        let evicted = mem.batch_add_episodes(episodes);

        assert_eq!(mem.size(), 100);
        assert_eq!(evicted.len(), 50); // 150 - 100
    }
}
