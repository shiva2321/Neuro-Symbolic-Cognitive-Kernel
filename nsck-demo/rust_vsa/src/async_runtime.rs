/// Async runtime integration with Tokio
///
/// Provides async/await support for concurrent cognitive operations

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::{Runtime, Handle};
use tokio::sync::{mpsc, oneshot};
use std::sync::Arc;
use parking_lot::Mutex;
use crate::{HyperVector, semantic::SemanticMemoryConcurrent, episodic::EpisodicMemoryConcurrent};

/// Async task types
#[derive(Clone, Debug)]
pub enum AsyncTask {
    /// Semantic search task
    SemanticSearch {
        query_hv: HyperVector,
        k: usize,
        response_tx: Arc<Mutex<Option<oneshot::Sender<Vec<(String, f64)>>>>>,
    },
    /// Spreading activation task
    SpreadingActivation {
        start_concepts: Vec<String>,
        steps: usize,
        decay: f64,
        response_tx: Arc<Mutex<Option<oneshot::Sender<std::collections::HashMap<String, f64>>>>>,
    },
    /// Episode search task
    EpisodeSearch {
        query_hv: HyperVector,
        k: usize,
        task_filter: Option<String>,
        response_tx: Arc<Mutex<Option<oneshot::Sender<Vec<(usize, f64)>>>>>,
    },
}

/// Async runtime wrapper for cognitive operations
#[pyclass(module = "hypervec_rs")]
pub struct AsyncCognitiveRuntime {
    runtime: Arc<Runtime>,
    task_tx: mpsc::UnboundedSender<AsyncTask>,
    semantic_memory: Arc<SemanticMemoryConcurrent>,
    episodic_memory: Arc<EpisodicMemoryConcurrent>,
}

#[pymethods]
impl AsyncCognitiveRuntime {
    #[new]
    fn new(
        semantic_memory: SemanticMemoryConcurrent,
        episodic_memory: EpisodicMemoryConcurrent,
    ) -> PyResult<Self> {
        // Create Tokio runtime
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("cognitive-async")
            .enable_all()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create async runtime: {}", e)
            ))?;

        let (task_tx, mut task_rx) = mpsc::unbounded_channel::<AsyncTask>();

        let semantic_memory = Arc::new(semantic_memory);
        let episodic_memory = Arc::new(episodic_memory);

        // Spawn task processor
        let sem_mem = semantic_memory.clone();
        let ep_mem = episodic_memory.clone();
        
        runtime.spawn(async move {
            while let Some(task) = task_rx.recv().await {
                match task {
                    AsyncTask::SemanticSearch { query_hv, k, response_tx } => {
                        let results = sem_mem.parallel_semantic_search(&query_hv, k);
                        if let Some(tx) = response_tx.lock().take() {
                            let _ = tx.send(results);
                        }
                    }
                    AsyncTask::SpreadingActivation { start_concepts, steps, decay, response_tx } => {
                        let results = sem_mem.parallel_spread_activation(
                            start_concepts, steps, decay, 0.01, false
                        );
                        if let Some(tx) = response_tx.lock().take() {
                            let _ = tx.send(results);
                        }
                    }
                    AsyncTask::EpisodeSearch { query_hv, k, task_filter, response_tx } => {
                        let results = ep_mem.parallel_knn_search(&query_hv, k, task_filter);
                        let simplified: Vec<(usize, f64)> = results
                            .into_iter()
                            .map(|(idx, sim, _)| (idx, sim))
                            .collect();
                        if let Some(tx) = response_tx.lock().take() {
                            let _ = tx.send(simplified);
                        }
                    }
                }
            }
        });

        Ok(AsyncCognitiveRuntime {
            runtime: Arc::new(runtime),
            task_tx,
            semantic_memory,
            episodic_memory,
        })
    }

    /// Submit semantic search task (non-blocking)
    fn submit_semantic_search(&self, query_hv: HyperVector, k: usize) -> PyResult<u64> {
        let (response_tx, _response_rx) = oneshot::channel();
        let response_tx = Arc::new(Mutex::new(Some(response_tx)));
        
        let task = AsyncTask::SemanticSearch {
            query_hv,
            k,
            response_tx,
        };

        self.task_tx.send(task)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to submit task: {}", e)
            ))?;

        // Return a task ID (simplified - in production would track tasks)
        Ok(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64)
    }

    /// Submit spreading activation task (non-blocking)
    fn submit_spreading_activation(
        &self,
        start_concepts: Vec<String>,
        steps: usize,
        decay: f64,
    ) -> PyResult<u64> {
        let (response_tx, _response_rx) = oneshot::channel();
        let response_tx = Arc::new(Mutex::new(Some(response_tx)));
        
        let task = AsyncTask::SpreadingActivation {
            start_concepts,
            steps,
            decay,
            response_tx,
        };

        self.task_tx.send(task)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to submit task: {}", e)
            ))?;

        Ok(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64)
    }

    /// Submit episode search task (non-blocking)
    fn submit_episode_search(
        &self,
        query_hv: HyperVector,
        k: usize,
        task_filter: Option<String>,
    ) -> PyResult<u64> {
        let (response_tx, _response_rx) = oneshot::channel();
        let response_tx = Arc::new(Mutex::new(Some(response_tx)));
        
        let task = AsyncTask::EpisodeSearch {
            query_hv,
            k,
            task_filter,
            response_tx,
        };

        self.task_tx.send(task)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to submit task: {}", e)
            ))?;

        Ok(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64)
    }

    /// Execute semantic search synchronously (blocking)
    fn semantic_search_sync(&self, query_hv: HyperVector, k: usize) -> PyResult<Vec<(String, f64)>> {
        let (response_tx, response_rx) = oneshot::channel();
        let response_tx = Arc::new(Mutex::new(Some(response_tx)));
        
        let task = AsyncTask::SemanticSearch {
            query_hv,
            k,
            response_tx,
        };

        self.task_tx.send(task)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to submit task: {}", e)
            ))?;

        // Block on result
        let result = self.runtime.block_on(response_rx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to receive result: {}", e)
            ))?;

        Ok(result)
    }

    /// Get runtime statistics
    fn get_stats(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        
        dict.set_item("worker_threads", 4)?;
        dict.set_item("runtime_type", "tokio-multi-thread")?;
        dict.set_item("semantic_concepts", self.semantic_memory.concept_count())?;
        dict.set_item("episodic_episodes", self.episodic_memory.size())?;

        Ok(dict.into())
    }
}

// Simpler synchronous-style async helpers for Python
#[pyfunction]
pub fn run_semantic_search_async(
    semantic_memory: SemanticMemoryConcurrent,
    query_hv: HyperVector,
    k: usize,
) -> PyResult<Vec<(String, f64)>> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create runtime: {}", e)
        ))?;

    let result = runtime.block_on(async move {
        // Run in async context
        tokio::task::spawn_blocking(move || {
            semantic_memory.parallel_semantic_search(&query_hv, k)
        }).await
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Task failed: {}", e)
    ))?;

    Ok(result)
}

// Register async module
pub fn register_async_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<AsyncCognitiveRuntime>()?;
    m.add_function(wrap_pyfunction!(run_semantic_search_async, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_runtime_creation() {
        let semantic = SemanticMemoryConcurrent::new();
        let episodic = EpisodicMemoryConcurrent::new(1000);
        
        let runtime = AsyncCognitiveRuntime::new(semantic, episodic);
        assert!(runtime.is_ok());
    }

    #[tokio::test]
    async fn test_async_semantic_search() {
        let semantic = SemanticMemoryConcurrent::new();
        let episodic = EpisodicMemoryConcurrent::new(1000);

        // Add some concepts
        for i in 0..10 {
            let hv = HyperVector::new(Some(i));
            semantic.add_concept(format!("concept_{}", i), hv);
        }

        let runtime = AsyncCognitiveRuntime::new(semantic, episodic).unwrap();
        
        let query = HyperVector::new(Some(5));
        let results = runtime.semantic_search_sync(query, 5).unwrap();
        
        assert_eq!(results.len(), 5);
    }
}
