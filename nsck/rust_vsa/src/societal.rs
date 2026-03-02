/// Societal hypervector backend for NSCK V5.
///
/// Provides:
/// - `LivingHvStore`   — concurrent property store (age/stability/valence/activation)
/// - `SocietalHnswRs`  — greedy HNSW-like ANN index over f32 vectors
/// - `SpectralRgRs`    — spectral RG helpers (cosine-similarity matrix, degree vector)

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// LivingHvStore
// ---------------------------------------------------------------------------

/// Thread-safe property store for LivingHyperVector metadata.
///
/// Stores `(age, stability, valence, activation)` per concept.
#[pyclass(module = "hypervec_rs")]
#[derive(Clone)]
pub struct LivingHvStore {
    inner: Arc<RwLock<HashMap<String, (u64, f64, f64, f64)>>>,
}

#[pymethods]
impl LivingHvStore {
    #[new]
    pub fn new() -> Self {
        LivingHvStore {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store properties for a concept.
    pub fn set_properties(
        &self,
        concept_id: String,
        age: u64,
        stability: f64,
        valence: f64,
        activation: f64,
    ) {
        let mut store = self.inner.write().unwrap();
        store.insert(concept_id, (age, stability, valence, activation));
    }

    /// Retrieve (age, stability, valence, activation) or None.
    pub fn get_properties(&self, concept_id: &str) -> Option<(u64, f64, f64, f64)> {
        let store = self.inner.read().unwrap();
        store.get(concept_id).copied()
    }

    /// Update only the activation value.
    pub fn update_activation(&self, concept_id: &str, value: f64) {
        let mut store = self.inner.write().unwrap();
        if let Some(entry) = store.get_mut(concept_id) {
            entry.3 = value.max(0.0).min(1.0);
        }
    }

    /// Decay all activations by `decay` fraction (clamped to 0.0).
    pub fn tick_all(&self, decay: f64) {
        let mut store = self.inner.write().unwrap();
        for entry in store.values_mut() {
            entry.0 += 1; // age++
            entry.3 = (entry.3 - decay).max(0.0);
        }
    }

    /// Return concept IDs with activation above threshold.
    pub fn get_active_concepts(&self, threshold: f64) -> Vec<String> {
        let store = self.inner.read().unwrap();
        store
            .iter()
            .filter(|(_, v)| v.3 > threshold)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Number of stored concepts.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    /// Clear all stored concepts.
    pub fn clear(&self) {
        self.inner.write().unwrap().clear();
    }
}

// ---------------------------------------------------------------------------
// SocietalHnswRs
// ---------------------------------------------------------------------------

/// Simple greedy HNSW-like ANN index for f32 hypervectors.
#[pyclass(module = "hypervec_rs")]
pub struct SocietalHnswRs {
    dim: usize,
    m: usize,
    _ef_construction: usize,
    ids: Vec<String>,
    vecs: Vec<Vec<f32>>,
    graph: HashMap<usize, Vec<usize>>,
}

fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot  // assumes pre-normalised vectors
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalise(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

#[pymethods]
impl SocietalHnswRs {
    #[new]
    #[pyo3(signature = (dim, m=16, ef_construction=200))]
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        SocietalHnswRs {
            dim,
            m,
            _ef_construction: ef_construction,
            ids: Vec::new(),
            vecs: Vec::new(),
            graph: HashMap::new(),
        }
    }

    /// Add a concept with its vector.
    pub fn add_item(&mut self, concept_id: String, vector: Vec<f32>) -> PyResult<()> {
        let idx = self.ids.len();
        let norm_vec = normalise(&vector);
        self.ids.push(concept_id);
        self.vecs.push(norm_vec.clone());
        self.graph.insert(idx, Vec::new());

        if idx == 0 {
            return Ok(());
        }

        // Find M nearest neighbours
        let mut sims: Vec<(f32, usize)> = (0..idx)
            .map(|j| (cosine_sim_f32(&norm_vec, &self.vecs[j]), j))
            .collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let neighbours: Vec<usize> = sims.iter().take(self.m).map(|(_, j)| *j).collect();
        for &nb in &neighbours {
            self.graph.get_mut(&idx).unwrap().push(nb);
            let nb_edges = self.graph.get_mut(&nb).unwrap();
            if nb_edges.len() < self.m * 2 {
                nb_edges.push(idx);
            }
        }

        Ok(())
    }

    /// Query for top-k nearest neighbours. Returns (concept_id, similarity) pairs.
    pub fn query(&self, query_vec: Vec<f32>, top_k: usize) -> Vec<(String, f32)> {
        let n = self.ids.len();
        if n == 0 {
            return Vec::new();
        }

        let qvec = normalise(&query_vec);
        let mut visited = vec![false; n];
        let mut results: Vec<(f32, usize)> = Vec::new();

        // Greedy search from entry point 0
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        let entry_sim = cosine_sim_f32(&qvec, &self.vecs[0]);
        candidates.push((entry_sim, 0));
        visited[0] = true;
        results.push((entry_sim, 0));

        while let Some((sim, curr)) = candidates.pop() {
            for &nb in self.graph.get(&curr).unwrap_or(&vec![]) {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let nb_sim = cosine_sim_f32(&qvec, &self.vecs[nb]);
                results.push((nb_sim, nb));
                if nb_sim > sim - 0.1 {
                    candidates.push((nb_sim, nb));
                }
            }
            if results.len() >= top_k * 5 {
                break;
            }
        }

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results
            .iter()
            .take(top_k)
            .map(|(s, idx)| (self.ids[*idx].clone(), *s))
            .collect()
    }

    /// Number of indexed concepts.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

// ---------------------------------------------------------------------------
// SpectralRgRs
// ---------------------------------------------------------------------------

/// Static spectral RG helpers.
#[pyclass(module = "hypervec_rs")]
pub struct SpectralRgRs {}

#[pymethods]
impl SpectralRgRs {
    #[new]
    pub fn new() -> Self {
        SpectralRgRs {}
    }

    /// Compute NxN cosine similarity matrix from a list of f32 vectors.
    ///
    /// Each vector is L2-normalised before computing dot products.
    #[staticmethod]
    pub fn cosine_similarity_matrix(vectors: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let n = vectors.len();
        let normed: Vec<Vec<f32>> = vectors.iter().map(|v| normalise(v)).collect();
        let mut result = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            result[i][i] = 1.0;
            for j in (i + 1)..n {
                let s = cosine_sim_f32(&normed[i], &normed[j]);
                result[i][j] = s;
                result[j][i] = s;
            }
        }
        result
    }

    /// Compute degree vector (row sums) from an adjacency matrix.
    #[staticmethod]
    pub fn degree_vector(adj: Vec<Vec<f32>>) -> Vec<f32> {
        adj.iter().map(|row| row.iter().sum()).collect()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_societal_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<LivingHvStore>()?;
    m.add_class::<SocietalHnswRs>()?;
    m.add_class::<SpectralRgRs>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_living_hv_store_set_get() {
        let store = LivingHvStore::new();
        store.set_properties("cat".to_string(), 0, 0.5, 0.1, 0.8);
        let props = store.get_properties("cat").unwrap();
        assert_eq!(props.0, 0);
        assert!((props.2 - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_living_hv_store_tick_all() {
        let store = LivingHvStore::new();
        store.set_properties("a".to_string(), 0, 0.5, 0.0, 0.6);
        store.tick_all(0.1);
        let props = store.get_properties("a").unwrap();
        assert_eq!(props.0, 1);
        assert!((props.3 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_societal_hnsw_add_query() {
        let mut index = SocietalHnswRs::new(4, 4, 100);
        index.add_item("a".to_string(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add_item("b".to_string(), vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        index.add_item("c".to_string(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        let results = index.query(vec![1.0, 0.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_spectral_rg_cosine_matrix() {
        let vecs = vec![
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![1.0f32, 0.0],
        ];
        let mat = SpectralRgRs::cosine_similarity_matrix(vecs);
        assert_eq!(mat.len(), 3);
        assert!((mat[0][0] - 1.0).abs() < 1e-6);
        assert!((mat[0][1]).abs() < 1e-6);   // orthogonal
        assert!((mat[0][2] - 1.0).abs() < 1e-6);  // identical
    }
}
