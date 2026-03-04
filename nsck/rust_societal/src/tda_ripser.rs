use pyo3::prelude::*;
use std::collections::{HashSet, HashMap};

#[pyclass]
pub struct TDARipser {}

#[pymethods]
impl TDARipser {
    #[new]
    pub fn new() -> Self {
        TDARipser {}
    }

    /// Computes Betti-0 and Betti-1 for the 1-skeleton (Vietoris-Rips graph approach)
    /// landmarks: list of high dimensional points vector 
    /// max_radius: epsilon for connecting nodes
    pub fn compute_betti_numbers(&self, landmarks: Vec<Vec<f32>>, max_radius: f64) -> PyResult<(usize, usize, usize)> {
        let n = landmarks.len();
        if n == 0 {
            return Ok((0, 0, 0));
        }

        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        
        for i in 0..n {
            for j in (i+1)..n {
                let dist = hamming_or_cosine_sim(&landmarks[i], &landmarks[j]);
                // using sim > max_radius as connection threshold (radius equivalent)
                if dist > max_radius as f32 {
                    edges.push((i, j, dist));
                }
            }
        }
        
        // Compute Betti-0 (connected components) and Betti-1 (cycles)
        // using a spanning tree count via union-find
        let mut parent: Vec<usize> = (0..n).collect();
        let mut size: Vec<usize> = vec![1; n];
        let mut components = n;
        
        let mut find = |mut i: usize, p: &mut Vec<usize>| -> usize {
            while i != p[i] {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        };

        // Betti 1 = Edges - Vertices + Components
        let mut b1_loops = 0;
        
        for (u, v, _) in &edges {
            let root_u = find(*u, &mut parent);
            let root_v = find(*v, &mut parent);
            if root_u != root_v {
                // union
                if size[root_u] < size[root_v] {
                    parent[root_u] = root_v;
                    size[root_v] += size[root_u];
                } else {
                    parent[root_v] = root_u;
                    size[root_u] += size[root_v];
                }
                components -= 1;
            } else {
                b1_loops += 1;
            }
        }
        
        let b0 = components;
        let b1 = b1_loops;
        
        // Betti 2 (Voids) is computationally expensive without a full boundary matrix. 
        // For semantic graphs, B0 and B1 are the critical health markers.
        // We set b2 to 0 as an approximation for now.
        let b2 = 0;

        Ok((b0, b1, b2))
    }
}

// Helper: Cosine similarity
fn hamming_or_cosine_sim(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}
