/// Cognitive worker pool with thread-local working memory
///
/// This module provides a thread pool where each worker maintains
/// its own working memory while sharing semantic/episodic memory

use pyo3::prelude::*;
use crossbeam::channel::{bounded, unbounded, Sender, Receiver};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::collections::VecDeque;
use crate::{HyperVector, semantic::SemanticMemoryConcurrent, episodic::EpisodicMemoryConcurrent};

/// Task types that workers can process
#[derive(Clone, Debug)]
pub enum CognitiveTask {
    /// Query processing task
    Query {
        query_id: String,
        query_hv: HyperVector,
        params: QueryParams,
    },
    /// Training task
    Training {
        task_id: String,
        text: String,
        concepts: Vec<(String, HyperVector)>,
    },
    /// Batch processing task
    Batch {
        batch_id: String,
        queries: Vec<(String, HyperVector)>,
    },
    /// Shutdown signal
    Shutdown,
}

/// Query parameters
#[derive(Clone, Debug)]
pub struct QueryParams {
    pub k: usize,              // Top-k results
    pub spread_steps: usize,   // Spreading activation steps
    pub spread_decay: f64,     // Decay factor
    pub timeout_ms: u64,       // Query timeout
}

impl Default for QueryParams {
    fn default() -> Self {
        QueryParams {
            k: 10,
            spread_steps: 3,
            spread_decay: 0.7,
            timeout_ms: 5000,
        }
    }
}

/// Result from cognitive processing
#[derive(Clone, Debug)]
pub enum CognitiveResult {
    /// Query result
    QueryResult {
        query_id: String,
        similar_concepts: Vec<(String, f64)>,
        activated_concepts: Vec<(String, f64)>,
        episodes: Vec<(usize, f64)>,
        latency_ms: u64,
    },
    /// Training result
    TrainingResult {
        task_id: String,
        concepts_added: usize,
        relations_added: usize,
        episodes_added: usize,
    },
    /// Batch result
    BatchResult {
        batch_id: String,
        results: Vec<CognitiveResult>,
        total_latency_ms: u64,
    },
    /// Error result
    Error {
        task_id: String,
        error: String,
    },
}

/// Thread-local working memory for each worker
pub struct WorkingMemory {
    /// Recently activated concepts (LRU cache)
    activated_cache: VecDeque<(String, f64)>,
    /// Recent queries (for context)
    query_history: VecDeque<String>,
    /// Temporary computation buffers
    temp_hvs: Vec<HyperVector>,
    /// Max cache size
    max_cache_size: usize,
}

impl WorkingMemory {
    pub fn new(max_cache_size: usize) -> Self {
        WorkingMemory {
            activated_cache: VecDeque::with_capacity(max_cache_size),
            query_history: VecDeque::with_capacity(100),
            temp_hvs: Vec::with_capacity(10),
            max_cache_size,
        }
    }

    /// Add activated concept to cache
    pub fn cache_activation(&mut self, concept: String, activation: f64) {
        if self.activated_cache.len() >= self.max_cache_size {
            self.activated_cache.pop_front();
        }
        self.activated_cache.push_back((concept, activation));
    }

    /// Check if concept is in cache
    pub fn get_cached_activation(&self, concept: &str) -> Option<f64> {
        self.activated_cache
            .iter()
            .rev()
            .find(|(c, _)| c == concept)
            .map(|(_, a)| *a)
    }

    /// Add query to history
    pub fn add_query(&mut self, query_id: String) {
        if self.query_history.len() >= 100 {
            self.query_history.pop_front();
        }
        self.query_history.push_back(query_id);
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.activated_cache.clear();
        self.query_history.clear();
        self.temp_hvs.clear();
    }
}

/// Cognitive worker that processes tasks
struct CognitiveWorker {
    worker_id: usize,
    task_rx: Receiver<CognitiveTask>,
    result_tx: Sender<CognitiveResult>,
    semantic_memory: Arc<SemanticMemoryConcurrent>,
    episodic_memory: Arc<EpisodicMemoryConcurrent>,
    working_memory: WorkingMemory,
}

impl CognitiveWorker {
    fn new(
        worker_id: usize,
        task_rx: Receiver<CognitiveTask>,
        result_tx: Sender<CognitiveResult>,
        semantic_memory: Arc<SemanticMemoryConcurrent>,
        episodic_memory: Arc<EpisodicMemoryConcurrent>,
    ) -> Self {
        CognitiveWorker {
            worker_id,
            task_rx,
            result_tx,
            semantic_memory,
            episodic_memory,
            working_memory: WorkingMemory::new(1000),
        }
    }

    /// Main worker loop
    fn run(&mut self) {
        loop {
            match self.task_rx.recv() {
                Ok(CognitiveTask::Shutdown) => {
                    println!("Worker {} shutting down", self.worker_id);
                    break;
                }
                Ok(task) => {
                    let result = self.process_task(task);
                    if let Err(e) = self.result_tx.send(result) {
                        eprintln!("Worker {} failed to send result: {}", self.worker_id, e);
                    }
                }
                Err(_) => {
                    // Channel closed
                    break;
                }
            }
        }
    }

    /// Process a cognitive task
    fn process_task(&mut self, task: CognitiveTask) -> CognitiveResult {
        match task {
            CognitiveTask::Query { query_id, query_hv, params } => {
                self.process_query(query_id, query_hv, params)
            }
            CognitiveTask::Training { task_id, text, concepts } => {
                self.process_training(task_id, text, concepts)
            }
            CognitiveTask::Batch { batch_id, queries } => {
                self.process_batch(batch_id, queries)
            }
            _ => CognitiveResult::Error {
                task_id: "unknown".to_string(),
                error: "Invalid task".to_string(),
            },
        }
    }

    /// Process query task
    fn process_query(
        &mut self,
        query_id: String,
        query_hv: HyperVector,
        params: QueryParams,
    ) -> CognitiveResult {
        let start = std::time::Instant::now();

        // 1. Semantic search
        let similar_concepts = self.semantic_memory.parallel_semantic_search(&query_hv, params.k);

        // 2. Spreading activation from top concepts
        let start_concepts: Vec<String> = similar_concepts
            .iter()
            .take(5)
            .map(|(name, _)| name.clone())
            .collect();

        let activation_map = self.semantic_memory.parallel_spread_activation(
            start_concepts,
            params.spread_steps,
            params.spread_decay,
            0.01,
            false,
        );

        // Cache activations in working memory
        for (concept, activation) in activation_map.iter() {
            self.working_memory.cache_activation(concept.clone(), *activation);
        }

        let mut activated_concepts: Vec<(String, f64)> = activation_map
            .into_iter()
            .collect();

        // 3. Episode retrieval
        let episodes = self.episodic_memory.parallel_knn_search(&query_hv, params.k, None);
        let episode_results: Vec<(usize, f64)> = episodes
            .into_iter()
            .map(|(idx, sim, _)| (idx, sim))
            .collect();

        // Add to query history
        self.working_memory.add_query(query_id.clone());

        let latency_ms = start.elapsed().as_millis() as u64;

        CognitiveResult::QueryResult {
            query_id,
            similar_concepts,
            activated_concepts,
            episodes: episode_results,
            latency_ms,
        }
    }

    /// Process training task
    fn process_training(
        &mut self,
        task_id: String,
        _text: String,
        concepts: Vec<(String, HyperVector)>,
    ) -> CognitiveResult {
        let mut concepts_added = 0;
        let relations_added = 0;

        // Add concepts to semantic memory
        for (name, hv) in concepts {
            self.semantic_memory.add_concept(name, hv);
            concepts_added += 1;
        }

        // Note: Relations and episodes would be added here
        // This is a simplified implementation

        CognitiveResult::TrainingResult {
            task_id,
            concepts_added,
            relations_added,
            episodes_added: 0,
        }
    }

    /// Process batch task
    fn process_batch(
        &mut self,
        batch_id: String,
        queries: Vec<(String, HyperVector)>,
    ) -> CognitiveResult {
        let start = std::time::Instant::now();
        let mut results = Vec::new();

        for (query_id, query_hv) in queries {
            let result = self.process_query(
                query_id,
                query_hv,
                QueryParams::default(),
            );
            results.push(result);
        }

        let total_latency_ms = start.elapsed().as_millis() as u64;

        CognitiveResult::BatchResult {
            batch_id,
            results,
            total_latency_ms,
        }
    }
}

/// Cognitive worker pool
#[pyclass(module = "hypervec_rs")]
pub struct CognitiveWorkerPool {
    num_workers: usize,
    task_tx: Sender<CognitiveTask>,
    result_rx: Receiver<CognitiveResult>,
    worker_handles: Vec<JoinHandle<()>>,
    semantic_memory: Arc<SemanticMemoryConcurrent>,
    episodic_memory: Arc<EpisodicMemoryConcurrent>,
}

#[pymethods]
impl CognitiveWorkerPool {
    #[new]
    #[pyo3(signature = (semantic_memory, episodic_memory, num_workers = 4))]
    fn new(
        semantic_memory: SemanticMemoryConcurrent,
        episodic_memory: EpisodicMemoryConcurrent,
        num_workers: usize,
    ) -> Self {
        let (task_tx, task_rx) = unbounded();
        let (result_tx, result_rx) = unbounded();

        let semantic_memory = Arc::new(semantic_memory);
        let episodic_memory = Arc::new(episodic_memory);

        let mut worker_handles = Vec::new();

        for worker_id in 0..num_workers {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let semantic_memory = semantic_memory.clone();
            let episodic_memory = episodic_memory.clone();

            let handle = thread::spawn(move || {
                let mut worker = CognitiveWorker::new(
                    worker_id,
                    task_rx,
                    result_tx,
                    semantic_memory,
                    episodic_memory,
                );
                worker.run();
            });

            worker_handles.push(handle);
        }

        CognitiveWorkerPool {
            num_workers,
            task_tx,
            result_rx,
            worker_handles,
            semantic_memory,
            episodic_memory,
        }
    }

    /// Submit a query task
    fn submit_query(
        &self,
        query_id: String,
        query_hv: HyperVector,
        k: usize,
        spread_steps: usize,
        spread_decay: f64,
    ) -> PyResult<()> {
        let task = CognitiveTask::Query {
            query_id,
            query_hv,
            params: QueryParams {
                k,
                spread_steps,
                spread_decay,
                timeout_ms: 5000,
            },
        };

        self.task_tx.send(task)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to submit task: {}", e)
            ))
    }

    /// Get result (blocking)
    fn get_result(&self, timeout_ms: u64) -> PyResult<String> {
        let timeout = std::time::Duration::from_millis(timeout_ms);
        
        match self.result_rx.recv_timeout(timeout) {
            Ok(result) => {
                // Convert result to JSON string for Python
                Ok(format!("{:?}", result))
            }
            Err(_) => Err(PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                "Timeout waiting for result"
            )),
        }
    }

    /// Get number of workers
    fn worker_count(&self) -> usize {
        self.num_workers
    }

    /// Shutdown all workers
    fn shutdown(&mut self) {
        println!("Shutting down worker pool...");
        
        // Send shutdown signal to all workers
        for _ in 0..self.num_workers {
            let _ = self.task_tx.send(CognitiveTask::Shutdown);
        }

        // Wait for all workers to finish
        while let Some(handle) = self.worker_handles.pop() {
            let _ = handle.join();
        }

        println!("Worker pool shut down");
    }
}

// Ensure proper cleanup
impl Drop for CognitiveWorkerPool {
    fn drop(&mut self) {
        if !self.worker_handles.is_empty() {
            self.shutdown();
        }
    }
}

// Register worker pool module
pub fn register_worker_pool_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<CognitiveWorkerPool>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_working_memory() {
        let mut wm = WorkingMemory::new(10);
        
        // Cache activations
        wm.cache_activation("concept_1".to_string(), 0.9);
        wm.cache_activation("concept_2".to_string(), 0.7);
        
        // Retrieve cached
        assert_eq!(wm.get_cached_activation("concept_1"), Some(0.9));
        assert_eq!(wm.get_cached_activation("concept_2"), Some(0.7));
        assert_eq!(wm.get_cached_activation("concept_3"), None);
    }

    #[test]
    fn test_worker_pool_creation() {
        let semantic = SemanticMemoryConcurrent::new();
        let episodic = EpisodicMemoryConcurrent::new(1000);
        
        let pool = CognitiveWorkerPool::new(semantic, episodic, 4);
        assert_eq!(pool.worker_count(), 4);
    }
}
