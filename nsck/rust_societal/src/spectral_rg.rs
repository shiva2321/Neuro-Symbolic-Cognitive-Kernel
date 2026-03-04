use pyo3::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

#[pyclass]
pub struct SpectralRG {}

#[pymethods]
impl SpectralRG {
    #[new]
    pub fn new() -> Self {
        SpectralRG {}
    }

    /// Computes Laplacian eigendecomposition for coarse-graining
    pub fn compute_spectral_gap(&self, adjacency_matrix: Vec<Vec<f64>>) -> PyResult<(Vec<f64>, usize)> {
        // We'll compute normalized Laplacian and its eigenvalues
        let n = adjacency_matrix.len();
        if n == 0 {
            return Ok((vec![], 0));
        }

        let mut d_vec = vec![0.0; n];
        let mut adj = DMatrix::zeros(n, n);
        
        for i in 0..n {
            let row = &adjacency_matrix[i];
            for j in 0..n {
                let val = row[j];
                adj[(i, j)] = val;
                d_vec[i] += val;
            }
        }

        let mut laplacian = DMatrix::zeros(n, n);
        for i in 0..n {
            let d_i = d_vec[i].sqrt();
            for j in 0..n {
                let d_j = d_vec[j].sqrt();
                if i == j {
                    laplacian[(i, j)] = 1.0;
                } else if d_i > 1e-10 && d_j > 1e-10 {
                    laplacian[(i, j)] = -adj[(i, j)] / (d_i * d_j);
                }
            }
        }

        // Compute eigendecomposition
        let eig = laplacian.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find spectral gap
        let mut max_gap = 0.0;
        let mut k_natural = 1;
        let limit = usize::min(50, n - 1);
        for k in 1..limit {
            let gap = eigenvalues[k] - eigenvalues[k - 1];
            if gap > max_gap {
                max_gap = gap;
                k_natural = k;
            }
        }

        Ok((eigenvalues, k_natural))
    }
}
