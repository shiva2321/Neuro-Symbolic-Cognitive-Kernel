use pyo3::prelude::*;
use dashmap::DashMap;

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

/// Batch decay bonds for all non-diamond LivingHyperVectors.
///
/// This is a pure Rayon-vectorised free function.  The Python caller assembles
/// the input arrays once (O(N) loop) and passes them here; Rust then runs the
/// per-bond decay calculation in parallel across all bonds.
///
/// Parameters
/// ----------
/// is_stale : Vec<bool>   — per-bond: True when the owning concept's
///                          ``(current_epoch - last_activated_epoch) > decay_epochs``
/// is_diamond : Vec<bool> — per-bond: True when the owning concept's
///                          stability_class == "diamond" (bonds never decay)
/// strengths : Vec<f32>   — current bond strength for each bond
/// decay_factor : f32     — multiplicative decay (default 0.9)
/// min_strength : f32     — strength below which a bond is removed (default 0.35)
///
/// Returns
/// -------
/// Vec<f32> — new bond strength per entry; 0.0 means the bond should be removed
#[pyfunction]
pub fn decay_bonds_batch(
    is_stale: Vec<bool>,
    is_diamond: Vec<bool>,
    strengths: Vec<f32>,
    decay_factor: f32,
    min_strength: f32,
) -> Vec<f32> {
    use rayon::prelude::*;
    // Zip the three parallel arrays and compute new strength for each bond.
    is_stale
        .par_iter()
        .zip(is_diamond.par_iter())
        .zip(strengths.par_iter())
        .map(|((stale, diamond), &s)| {
            if *diamond || !stale {
                // Diamond bonds never decay; non-stale concepts' bonds are unchanged.
                s
            } else {
                let ns = s * decay_factor;
                if ns < min_strength { 0.0 } else { ns }
            }
        })
        .collect()
}
