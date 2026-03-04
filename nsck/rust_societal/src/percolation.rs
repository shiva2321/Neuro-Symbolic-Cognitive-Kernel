use pyo3::prelude::*;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

// A simple Disjoint Set (Union-Find) struct to find connected components
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    sets: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
            sets: n,
        }
    }

    fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            self.parent[root] = self.parent[self.parent[root]];
            root = self.parent[root];
        }
        root
    }

    fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if self.size[root_i] < self.size[root_j] {
                self.parent[root_i] = root_j;
                self.size[root_j] += self.size[root_i];
            } else {
                self.parent[root_j] = root_i;
                self.size[root_i] += self.size[root_j];
            }
            self.sets -= 1;
            true
        } else {
            false
        }
    }

    fn max_component_size(&self) -> usize {
        *self.size.iter().max().unwrap_or(&0)
    }
}

#[pyclass]
pub struct PercolationDetector {
    threshold: f64,
}

#[pymethods]
impl PercolationDetector {
    #[new]
    pub fn new(threshold: f64) -> Self {
        PercolationDetector { threshold }
    }

    /// Check if the similarity graph has percolated (formed a giant component representing a dominant Knowledge City)
    /// Returns a tuple of (bool_percolated, max_component_size, total_isolated_components)
    pub fn check_transition(&self, num_nodes: usize, edges: Vec<(usize, usize, f64)>) -> PyResult<(bool, usize, usize)> {
        let mut uf = UnionFind::new(num_nodes);
        
        let mut edges_added = 0;
        for (u, v, weight) in edges {
            if weight > self.threshold {
                if uf.union(u, v) {
                    edges_added += 1;
                }
            }
        }

        let max_size = uf.max_component_size();
        let total_components = uf.sets;

        // percolation threshold is traditionally tested as O(ln N) or when half of the graph is connected.
        // For semantic networks we assume percolation when max size > sqrt(num_nodes) * ln(num_nodes) or arbitrary tuned ratio bounds.
        // Here we'll consider it percolated if max_size >= num_nodes / 2 for city emergence.
        let percolated = max_size >= (num_nodes / 2) && num_nodes > 1;

        Ok((percolated, max_size, total_components))
    }
}
