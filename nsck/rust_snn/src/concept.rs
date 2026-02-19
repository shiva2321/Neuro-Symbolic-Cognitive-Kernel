/// ConceptMapper — Jaccard-similarity concept recognition
/// RateCoder    — spike train → (active_neurons, spike_rates)
///
/// These replace SimpleConceptMapper and the active-neuron extraction
/// in Python's vsa_snn_bridge.py.  The HV objects themselves remain
/// in Python (using the existing hypervec_shim) — only the numeric
/// pattern-matching hot paths live here.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ============================================================
// ConceptMapper
// ============================================================

/// Maps spike-pattern activity sets to integer concept IDs via Jaccard similarity.
///
/// Drop-in replacement for Python's `SimpleConceptMapper`.
/// Only the IDs and neuron-set membership are stored in Rust.
/// The associated HyperVector is looked up in Python by seed = concept_id + 1000.
#[pyclass(module = "snn_rs")]
pub struct ConceptMapper {
    /// concept_id → set of active neuron indices
    concepts: HashMap<usize, HashSet<usize>>,
    next_id:  usize,
}

#[pymethods]
impl ConceptMapper {
    #[new]
    fn new() -> Self {
        ConceptMapper {
            concepts: HashMap::new(),
            next_id:  0,
        }
    }

    /// Find the best matching concept for `active_neurons` using Jaccard similarity.
    ///
    /// Returns (concept_id, similarity) or (-1, 0.0) for novel patterns.
    /// Parallel search across all concepts with rayon.
    fn recognize_pattern(
        &self,
        active_neurons: Vec<usize>,
        threshold: f64,
    ) -> (i64, f64) {
        if self.concepts.is_empty() || active_neurons.is_empty() {
            return (-1, 0.0);
        }

        let active_set: HashSet<usize> = active_neurons.into_iter().collect();

        // Collect concept entries for parallel iteration
        let concepts_vec: Vec<(usize, &HashSet<usize>)> = self.concepts
            .iter()
            .map(|(&id, set)| (id, set))
            .collect();

        // Find best match in parallel
        let best = concepts_vec
            .par_iter()
            .map(|(id, neuron_set)| {
                let intersection = active_set.intersection(neuron_set).count();
                let union        = active_set.union(neuron_set).count();
                let sim = if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                };
                (*id, sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .reduce_with(|a, b| if a.1 >= b.1 { a } else { b });

        match best {
            Some((id, sim)) => (id as i64, sim),
            None            => (-1, 0.0),
        }
    }

    /// Register a new concept from `active_neurons` and return its ID.
    fn register_concept(&mut self, active_neurons: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.concepts.insert(id, active_neurons.into_iter().collect());
        id
    }

    /// Number of concepts registered.
    fn n_concepts(&self) -> usize { self.concepts.len() }

    /// Concept seed used by Python to recreate the HyperVector:
    ///   HyperVector(seed = concept_id + 1000)
    #[staticmethod]
    fn concept_hv_seed(concept_id: usize) -> u64 {
        (concept_id + 1000) as u64
    }

    fn __repr__(&self) -> String {
        format!("<ConceptMapper n_concepts={}>", self.concepts.len())
    }
}

// ============================================================
// RateCoder
// ============================================================

/// Extracts active neurons and per-neuron firing rates from a spike train.
///
/// Replaces the hot inner loop of Python's `RateCoder.encode()`.
///
/// Input:  spike_train — flat Vec<f64> of length (n_steps × n_neurons) row-major
///         n_neurons   — number of neurons (columns)
///         time_window_ms, rate_threshold — same semantics as Python
///
/// Output: (active_neurons: Vec<usize>, spike_rates: Vec<f64>)
///   active_neurons — indices of neurons firing above rate_threshold
///   spike_rates    — firing rate per neuron (spikes / window)  length n_neurons
///
/// The caller uses active_neurons to build the HyperVector in Python using
/// the existing hypervec_shim (avoiding a dependency on hypervec_rs here).
#[pyclass(module = "snn_rs")]
pub struct RateCoder {
    n_neurons:       usize,
    rate_threshold:  f64,
}

#[pymethods]
impl RateCoder {
    #[new]
    fn new(n_neurons: usize, rate_threshold: f64) -> Self {
        RateCoder { n_neurons, rate_threshold }
    }

    /// Compute firing rates and return (active_neuron_ids, spike_rates).
    ///
    /// spike_train_flat: flattened (n_steps, n_neurons) array from LIFLayer
    fn encode(
        &self,
        spike_train_flat: Vec<f64>,
        time_window_ms:   f64,
    ) -> PyResult<(Vec<usize>, Vec<f64>)> {
        let n = self.n_neurons;
        if spike_train_flat.len() % n != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "spike_train length {} not divisible by n_neurons {}", spike_train_flat.len(), n
            )));
        }
        let n_steps = spike_train_flat.len() / n;

        // Sum spikes per neuron across all time steps (parallel)
        let spike_counts: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|neuron| {
                (0..n_steps).map(|t| spike_train_flat[t * n + neuron]).sum()
            })
            .collect();

        // Firing rate = spikes / time_window_ms · 1000 (spikes per second as fraction)
        let scale = if time_window_ms > 0.0 { 1000.0 / time_window_ms } else { 1.0 };
        let spike_rates: Vec<f64> = spike_counts
            .iter()
            .map(|&c| c * scale / n_steps.max(1) as f64)
            .collect();

        let threshold = self.rate_threshold;
        let active_neurons: Vec<usize> = spike_rates
            .iter()
            .enumerate()
            .filter(|(_, &r)| r >= threshold)
            .map(|(i, _)| i)
            .collect();

        Ok((active_neurons, spike_rates))
    }

    #[getter]
    fn n_neurons(&self) -> usize { self.n_neurons }

    fn __repr__(&self) -> String {
        format!("<RateCoder n_neurons={} threshold={}>",
                self.n_neurons, self.rate_threshold)
    }
}

// Called from lib.rs
pub fn register_concept_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<ConceptMapper>()?;
    m.add_class::<RateCoder>()?;
    Ok(())
}
