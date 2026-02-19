/// Batch-atomic persistence layer
///
/// Provides thread-safe, batched writes to persistent storage with ACID guarantees

use pyo3::prelude::*;
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::VecDeque;
use std::path::PathBuf;
use rusqlite::{Connection, params, Result as SqlResult};
use serde::{Serialize, Deserialize};
use crate::{HyperVector, episodic::Episode};

/// Batch write buffer configuration
const DEFAULT_BATCH_SIZE: usize = 100;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 1000;

/// Serializable episode for storage
#[derive(Serialize, Deserialize, Clone, Debug)]
struct StoredEpisode {
    timestamp: f64,
    task_tag: String,
    situation_hv_bits: Vec<u8>,  // Serialized HV bits
    action: String,
    outcome: String,
    reward: f64,
    impact_score: f64,
}

impl StoredEpisode {
    fn from_episode(episode: &Episode) -> Self {
        // Serialize HV bits as little-endian bytes: 160 u64 × 8 bytes = 1280 bytes exactly.
        let situation_hv_bits: Vec<u8> = episode.situation_hv.bits
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();

        StoredEpisode {
            timestamp: episode.timestamp,
            task_tag: episode.task_tag.clone(),
            situation_hv_bits,
            action: episode.action.clone(),
            outcome: episode.outcome.clone(),
            reward: episode.reward,
            impact_score: episode.impact_score,
        }
    }

    /// Deserialize back into a full Episode, reconstructing the HyperVector from stored bytes.
    fn to_episode(&self) -> Episode {
        const NUM_U64: usize = 160; // 10240 / 64
        let mut bits: Vec<u64> = self.situation_hv_bits
            .chunks_exact(8)
            .map(|chunk| {
                let arr: [u8; 8] = chunk.try_into().unwrap_or([0u8; 8]);
                u64::from_le_bytes(arr)
            })
            .collect();
        // Pad/truncate to exactly 160 u64s to guard against corruption.
        bits.resize(NUM_U64, 0u64);

        let mut hv = HyperVector::zero();
        hv.bits = bits;

        Episode {
            timestamp:    self.timestamp,
            task_tag:     self.task_tag.clone(),
            situation_hv: hv,
            action:       self.action.clone(),
            outcome:      self.outcome.clone(),
            reward:       self.reward,
            impact_score: self.impact_score,
        }
    }
}

/// Batch write buffer
struct WriteBatch {
    episodes: VecDeque<StoredEpisode>,
    max_size: usize,
}

impl WriteBatch {
    fn new(max_size: usize) -> Self {
        WriteBatch {
            episodes: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn add(&mut self, episode: StoredEpisode) -> bool {
        self.episodes.push_back(episode);
        self.episodes.len() >= self.max_size
    }

    fn drain(&mut self) -> Vec<StoredEpisode> {
        self.episodes.drain(..).collect()
    }

    fn len(&self) -> usize {
        self.episodes.len()
    }

    fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

/// Persistent storage backend using SQLite
#[pyclass(module = "hypervec_rs")]
pub struct PersistentStorage {
    db_path: PathBuf,
    connection: Arc<Mutex<Connection>>,
    write_batch: Arc<Mutex<WriteBatch>>,
    batch_size: usize,
}

#[pymethods]
impl PersistentStorage {
    #[new]
    #[pyo3(signature = (db_path, batch_size = DEFAULT_BATCH_SIZE))]
    fn new(db_path: String, batch_size: usize) -> PyResult<Self> {
        let path = PathBuf::from(db_path);
        
        // Create connection
        let conn = Connection::open(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to open database: {}", e)
            ))?;

        // Create tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                task_tag TEXT NOT NULL,
                situation_hv BLOB NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reward REAL NOT NULL,
                impact_score REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create table: {}", e)
        ))?;

        // Create indexes for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)",
            [],
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create index: {}", e)
        ))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_tag ON episodes(task_tag)",
            [],
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to create index: {}", e)
        ))?;

        Ok(PersistentStorage {
            db_path: path,
            connection: Arc::new(Mutex::new(conn)),
            write_batch: Arc::new(Mutex::new(WriteBatch::new(batch_size))),
            batch_size,
        })
    }

    /// Add episode to write buffer (batched, non-blocking)
    fn buffer_episode(&self, episode: Episode) -> PyResult<bool> {
        let stored = StoredEpisode::from_episode(&episode);
        let mut batch = self.write_batch.lock();
        let should_flush = batch.add(stored);
        Ok(should_flush)
    }

    /// Flush pending writes to database (atomic transaction)
    fn flush(&self) -> PyResult<usize> {
        let episodes = {
            let mut batch = self.write_batch.lock();
            if batch.is_empty() {
                return Ok(0);
            }
            batch.drain()
        };

        let count = episodes.len();
        let conn = self.connection.lock();

        // Begin transaction
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to begin transaction: {}", e)
            ))?;

        // Insert all episodes in batch
        for episode in episodes {
            conn.execute(
                "INSERT INTO episodes (timestamp, task_tag, situation_hv, action, outcome, reward, impact_score)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    episode.timestamp,
                    episode.task_tag,
                    episode.situation_hv_bits,
                    episode.action,
                    episode.outcome,
                    episode.reward,
                    episode.impact_score,
                ],
            ).map_err(|e| {
                // Rollback on error
                let _ = conn.execute("ROLLBACK", []);
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to insert episode: {}", e)
                )
            })?;
        }

        // Commit transaction
        conn.execute("COMMIT", [])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to commit transaction: {}", e)
            ))?;

        Ok(count)
    }

    /// Get total episode count
    fn episode_count(&self) -> PyResult<usize> {
        let conn = self.connection.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM episodes",
            [],
            |row| row.get(0),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to count episodes: {}", e)
        ))?;
        Ok(count as usize)
    }

    /// Get buffered episode count (not yet flushed)
    fn buffered_count(&self) -> usize {
        self.write_batch.lock().len()
    }

    /// Query episodes by time range
    fn query_by_time_range(&self, start_time: f64, end_time: f64, limit: usize) -> PyResult<Vec<String>> {
        let conn = self.connection.lock();
        let mut stmt = conn.prepare(
            "SELECT timestamp, task_tag, action, outcome, reward, impact_score 
             FROM episodes 
             WHERE timestamp BETWEEN ?1 AND ?2 
             ORDER BY timestamp DESC 
             LIMIT ?3"
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to prepare query: {}", e)
        ))?;

        let results = stmt.query_map(
            params![start_time, end_time, limit as i64],
            |row| {
                Ok(format!(
                    "Episode(ts={}, task={}, action={}, reward={})",
                    row.get::<_, f64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(4)?,
                ))
            },
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to execute query: {}", e)
        ))?;

        let mut episodes = Vec::new();
        for result in results {
            episodes.push(result.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to read result: {}", e)
                )
            })?);
        }

        Ok(episodes)
    }

    /// Query episodes by task tag
    fn query_by_task(&self, task_tag: &str, limit: usize) -> PyResult<Vec<String>> {
        let conn = self.connection.lock();
        let mut stmt = conn.prepare(
            "SELECT timestamp, task_tag, action, outcome, reward, impact_score 
             FROM episodes 
             WHERE task_tag = ?1 
             ORDER BY timestamp DESC 
             LIMIT ?2"
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to prepare query: {}", e)
        ))?;

        let results = stmt.query_map(
            params![task_tag, limit as i64],
            |row| {
                Ok(format!(
                    "Episode(ts={}, task={}, action={}, reward={})",
                    row.get::<_, f64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(4)?,
                ))
            },
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to execute query: {}", e)
        ))?;

        let mut episodes = Vec::new();
        for result in results {
            episodes.push(result.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to read result: {}", e)
                )
            })?);
        }

        Ok(episodes)
    }

    /// Load full Episode objects (with reconstructed HyperVectors) from the database.
    ///
    /// Pass `task_tag = Some("my_task")` to filter by task, or `None` for all tasks.
    /// Results are ordered newest-first, capped at `limit`.
    #[pyo3(signature = (task_tag = None, limit = 1000))]
    fn load_episodes_full(
        &self,
        task_tag: Option<String>,
        limit: usize,
    ) -> PyResult<Vec<Episode>> {
        let conn = self.connection.lock();

        let stored: Vec<StoredEpisode> = if let Some(ref tag) = task_tag {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, task_tag, situation_hv, action, outcome, reward, impact_score \
                     FROM episodes WHERE task_tag = ?1 ORDER BY timestamp DESC LIMIT ?2",
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prepare failed: {}", e))
                })?;

            let rows: Vec<StoredEpisode> = stmt.query_map(params![tag, limit as i64], |row| {
                Ok(StoredEpisode {
                    timestamp:          row.get(0)?,
                    task_tag:           row.get(1)?,
                    situation_hv_bits:  row.get(2)?,
                    action:             row.get(3)?,
                    outcome:            row.get(4)?,
                    reward:             row.get(5)?,
                    impact_score:       row.get(6)?,
                })
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Query failed: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();
            rows
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, task_tag, situation_hv, action, outcome, reward, impact_score \
                     FROM episodes ORDER BY timestamp DESC LIMIT ?1",
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prepare failed: {}", e))
                })?;

            let rows: Vec<StoredEpisode> = stmt.query_map(params![limit as i64], |row| {
                Ok(StoredEpisode {
                    timestamp:          row.get(0)?,
                    task_tag:           row.get(1)?,
                    situation_hv_bits:  row.get(2)?,
                    action:             row.get(3)?,
                    outcome:            row.get(4)?,
                    reward:             row.get(5)?,
                    impact_score:       row.get(6)?,
                })
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Query failed: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();
            rows
        };

        Ok(stored.into_iter().map(|se| se.to_episode()).collect())
    }

    /// Get database statistics
    fn get_stats(&self, py: Python) -> PyResult<Py<pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        
        let dict = PyDict::new(py);
        
        dict.set_item("total_episodes", self.episode_count()?)?;
        dict.set_item("buffered_episodes", self.buffered_count())?;
        dict.set_item("batch_size", self.batch_size)?;
        dict.set_item("db_path", self.db_path.to_string_lossy().to_string())?;

        // Get database size
        if let Ok(metadata) = std::fs::metadata(&self.db_path) {
            dict.set_item("db_size_bytes", metadata.len())?;
        }

        Ok(dict.into())
    }

    /// Vacuum database to reclaim space
    fn vacuum(&self) -> PyResult<()> {
        let conn = self.connection.lock();
        conn.execute("VACUUM", [])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to vacuum database: {}", e)
            ))?;
        Ok(())
    }
}

// Register persistence module
pub fn register_persistence_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<PersistentStorage>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_storage_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let storage = PersistentStorage::new(
            db_path.to_string_lossy().to_string(),
            10,
        ).unwrap();

        assert_eq!(storage.buffered_count(), 0);
    }

    #[test]
    fn test_batch_buffering() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let storage = PersistentStorage::new(
            db_path.to_string_lossy().to_string(),
            5,  // Small batch size
        ).unwrap();

        // Add episodes
        for i in 0..4 {
            let episode = Episode::new(
                i as f64,
                format!("task_{}", i),
                HyperVector::new(Some(i)),
                format!("action_{}", i),
                "success".to_string(),
                0.5,
                0.3,
            );
            let should_flush = storage.buffer_episode(episode).unwrap();
            assert_eq!(should_flush, i == 4); // Flush on 5th episode (batch_size=5)
        }

        assert_eq!(storage.buffered_count(), 4);
    }

    #[test]
    fn test_flush_and_query() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let storage = PersistentStorage::new(
            db_path.to_string_lossy().to_string(),
            100,
        ).unwrap();

        // Add and flush episodes
        for i in 0..10 {
            let episode = Episode::new(
                i as f64,
                "test_task".to_string(),
                HyperVector::new(Some(i)),
                format!("action_{}", i),
                "success".to_string(),
                0.5,
                0.3,
            );
            storage.buffer_episode(episode).unwrap();
        }

        let flushed = storage.flush().unwrap();
        assert_eq!(flushed, 10);
        assert_eq!(storage.buffered_count(), 0);

        // Query
        let count = storage.episode_count().unwrap();
        assert_eq!(count, 10);

        let results = storage.query_by_task("test_task", 5).unwrap();
        assert_eq!(results.len(), 5);
    }
}
