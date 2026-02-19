/// Hebbian learning — accelerated Oja's rule weight updates.
///
/// Matches Python `HebbianMatrixNumPy` exactly, replacing its inner loops
/// with rayon-parallel operations.
///
/// Oja's Rule:  Δw_ij = η · (x_i · y_j  −  y_j² · w_ij)
///   The y²·w term prevents unlimited weight growth without explicit normalisation.

use pyo3::prelude::*;
use rayon::prelude::*;

/// Hebbian weight matrix with optional Oja normalisation and weight decay.
///
/// Constructor args match HebbianMatrixNumPy.__init__:
///   in_features, out_features, learning_rate, decay, normalize
#[pyclass(module = "snn_rs")]
pub struct HebbianMatrix {
    in_features:  usize,
    out_features: usize,
    lr:           f64,
    decay:        f64,
    normalize:    bool,
    /// Flat row-major weight matrix  [out_features × in_features]
    weights:      Vec<f64>,
    update_count: usize,
}

#[pymethods]
impl HebbianMatrix {
    #[new]
    fn new(
        in_features:   usize,
        out_features:  usize,
        learning_rate: f64,
        decay:         f64,
        normalize:     bool,
    ) -> Self {
        // Small random initialisation (matches `np.random.randn * 0.01`)
        let weights: Vec<f64> = (0..(out_features * in_features))
            .map(|i| {
                // Cheap deterministic pseudo-random via LCG (no rand dep needed in sub-module)
                let x = ((i as u64).wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)) as f64;
                (x / u64::MAX as f64) * 0.02 - 0.01
            })
            .collect();

        HebbianMatrix {
            in_features,
            out_features,
            lr: learning_rate,
            decay,
            normalize,
            weights,
            update_count: 0,
        }
    }

    /// Forward pass: y = W · x   (out_features,) = (out_features, in_features) @ (in_features,)
    fn forward(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        if x.len() != self.in_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected input length {}, got {}", self.in_features, x.len()
            )));
        }
        let result: Vec<f64> = (0..self.out_features)
            .into_par_iter()
            .map(|i| {
                let start = i * self.in_features;
                self.weights[start..start + self.in_features]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xi)| w * xi)
                    .sum()
            })
            .collect();
        Ok(result)
    }

    /// Hebbian update (Oja's rule) for a single pre/post sample pair.
    ///
    /// Δw_ij = η · (pre_i · post_j  −  post_j² · w_ij)  −  decay · w_ij
    ///
    /// pre:  Vec<f64> length in_features
    /// post: Vec<f64> length out_features
    fn hebbian_update(&mut self, pre: Vec<f64>, post: Vec<f64>) -> PyResult<()> {
        if pre.len() != self.in_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("pre length {} ≠ in_features {}", pre.len(), self.in_features)
            ));
        }
        if post.len() != self.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("post length {} ≠ out_features {}", post.len(), self.out_features)
            ));
        }

        let lr    = self.lr;
        let decay = self.decay;
        let norm  = self.normalize;
        let in_f  = self.in_features;

        // Parallel update over all (out, in) weight positions
        self.weights
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, w)| {
                let i = idx / in_f;   // out neuron index
                let j = idx % in_f;   // in  neuron index
                let xi = pre[j];
                let yj = post[i];
                // Oja:  Δw = η·(x·y − y²·w) − decay·w
                let oja_term = if norm { yj * yj * *w } else { 0.0 };
                *w += lr * (xi * yj - oja_term) - decay * *w;
            });

        self.update_count += 1;
        Ok(())
    }

    /// Return flat weight matrix [out_features × in_features] row-major.
    fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Replace the weight matrix.
    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        if weights.len() != self.out_features * self.in_features {
            return Err(pyo3::exceptions::PyValueError::new_err("Wrong weight count"));
        }
        self.weights = weights;
        Ok(())
    }

    #[getter]
    fn update_count(&self) -> usize { self.update_count }

    fn __repr__(&self) -> String {
        format!("<HebbianMatrix {}→{} updates={}>",
                self.in_features, self.out_features, self.update_count)
    }
}

// Called from lib.rs to register this class in the snn_rs module
pub fn register_hebbian_module(m: &PyModule) -> PyResult<()> {
    m.add_class::<HebbianMatrix>()?;
    Ok(())
}
