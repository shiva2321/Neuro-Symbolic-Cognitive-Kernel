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
    /// Returns a tuple of (bool_percolated, max_component_size, comp_sizes, node_to_root)
    /// - comp_sizes: dict mapping root_index → component_size
    /// - node_to_root: list mapping each node index → its root component index
    pub fn check_transition(&self, num_nodes: usize, edges: Vec<(usize, usize, f64)>) -> PyResult<(bool, usize, std::collections::HashMap<usize, usize>, Vec<usize>)> {
        let mut uf = UnionFind::new(num_nodes);

        for (u, v, weight) in edges {
            if weight > self.threshold && u < num_nodes && v < num_nodes {
                uf.union(u, v);
            }
        }

        let max_size = uf.max_component_size();

        // Build node_to_root: for each node, find its canonical root
        let mut node_to_root = vec![0usize; num_nodes];
        for i in 0..num_nodes {
            node_to_root[i] = uf.find(i);
        }

        // Build comp_sizes: root_index → count of nodes in that component
        let mut comp_sizes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &root in &node_to_root {
            *comp_sizes.entry(root).or_insert(0) += 1;
        }

        let percolated = max_size >= (num_nodes / 2) && num_nodes > 1;

        Ok((percolated, max_size, comp_sizes, node_to_root))
    }
}
