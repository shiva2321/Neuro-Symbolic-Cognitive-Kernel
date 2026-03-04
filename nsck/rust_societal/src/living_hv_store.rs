use pyo3::prelude::*;
use dashmap::DashMap;
use std::collections::VecDeque;

#[pyclass]
pub struct LivingHVStore {
    // Concurrent map from concept_id to domain_affinities string (JSON for simplicity here, or structured later)
    affinities: DashMap<String, String>,
    activation_counts: DashMap<String, u32>,
}

#[pymethods]
impl LivingHVStore {
    #[new]
    pub fn new() -> Self {
        LivingHVStore {
            affinities: DashMap::new(),
            activation_counts: DashMap::new(),
        }
    }

    pub fn set_affinity(&self, concept_id: String, affinities_json: String) {
        self.affinities.insert(concept_id, affinities_json);
    }

    pub fn get_affinity(&self, concept_id: String) -> Option<String> {
        self.affinities.get(&concept_id).map(|r| r.value().clone())
    }

    pub fn increment_activation(&self, concept_id: String) -> u32 {
        let mut count = self.activation_counts.entry(concept_id).or_insert(0);
        *count += 1;
        *count
    }
}
