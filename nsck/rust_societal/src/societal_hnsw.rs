use pyo3::prelude::*;
use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Ordering;
use dashmap::DashMap;

#[derive(Debug, Clone)]
struct Node {
    id: String,
    vector: Vec<f32>,
    layer: u8,
    neighbors: HashMap<u8, Vec<String>>,
}

#[derive(Clone, PartialEq, PartialOrd)]
struct OrderedFloat(f32);
impl Eq for OrderedFloat {}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(PartialEq, Eq)]
struct HeapElement {
    distance: OrderedFloat,
    node_id: String,
}

impl PartialOrd for HeapElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior based on distance
        other.distance.cmp(&self.distance)
    }
}

#[pyclass]
pub struct SocietalHNSW {
    nodes: DashMap<String, Node>,
    entry_point: Option<String>,
    max_layer: u8,
    m: usize,
    m_max: usize,
    ef_construction: usize,
}

#[pymethods]
impl SocietalHNSW {
    #[new]
    pub fn new() -> Self {
        SocietalHNSW {
            nodes: DashMap::new(),
            entry_point: None,
            max_layer: 0,
            m: 16,
            m_max: 32,
            ef_construction: 100,
        }
    }

    /// Insert a new concept into the HNSW graph. 
    /// Layer is deterministically assigned from central place theory instead of randomly.
    pub fn insert_node(&mut self, concept_id: String, layer: u8, vector: Vec<f32>) {
        let mut new_node = Node {
            id: concept_id.clone(),
            vector: vector.clone(),
            layer,
            neighbors: HashMap::new(),
        };

        if self.nodes.is_empty() {
            self.entry_point = Some(concept_id.clone());
            self.max_layer = layer;
            self.nodes.insert(concept_id, new_node);
            return;
        }

        let mut curr_entry = self.entry_point.clone().unwrap();
        let mut curr_dist = self.cosine_distance(&vector, &self.nodes.get(&curr_entry).unwrap().vector);

        // Find entry point in the current layer by traversing down from top
        for lc in (layer + 1..=self.max_layer).rev() {
            loop {
                let mut changed = false;
                if let Some(node) = self.nodes.get(&curr_entry) {
                    if let Some(neighbors) = node.neighbors.get(&lc) {
                        for neighbor_id in neighbors {
                            if let Some(neighbor_node) = self.nodes.get(neighbor_id) {
                                let dist = self.cosine_distance(&vector, &neighbor_node.vector);
                                if dist < curr_dist {
                                    curr_dist = dist;
                                    curr_entry = neighbor_id.clone();
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Search for neighbors at each relevant layer and connect
        for lc in (0..=layer).rev() {
            let top_candidates = self.search_layer(&vector, &curr_entry, self.ef_construction, lc);
            let selected_neighbors = self.select_neighbors(top_candidates, self.m);
            
            new_node.neighbors.insert(lc, selected_neighbors.clone());
            
            for neighbor_id in &selected_neighbors {
                // Must clone vector outside mut borrow
                let neighbor_vector = {
                    if let Some(n) = self.nodes.get(neighbor_id) {
                        n.vector.clone()
                    } else {
                        continue; // Should not happen
                    }
                };

                if let Some(mut neighbor) = self.nodes.get_mut(neighbor_id) {
                    let layer_neighbors = neighbor.neighbors.entry(lc).or_insert_with(Vec::new);
                    layer_neighbors.push(concept_id.clone());
                    
                    let m_max_current = if lc == 0 { self.m_max * 2 } else { self.m_max };
                    if layer_neighbors.len() > m_max_current {
                        let mut neighbor_candidates = Vec::new();
                        for id in layer_neighbors.iter() {
                            let dist = self.cosine_distance(&self.nodes.get(id).unwrap().vector, &neighbor_vector);
                            neighbor_candidates.push(HeapElement { distance: OrderedFloat(dist), node_id: id.clone() });
                        }
                        let new_selection = self.select_neighbors(neighbor_candidates, m_max_current);
                        *layer_neighbors = new_selection;
                    }
                }
            }
            // Update entry for next lower layer
            curr_entry = selected_neighbors.first().unwrap_or(&curr_entry).clone();
        }

        if layer > self.max_layer {
            self.max_layer = layer;
            self.entry_point = Some(concept_id.clone());
        }

        self.nodes.insert(concept_id, new_node);
    }

    /// HNSW search to return top K nearest nodes
    pub fn search(&self, query: Vec<f32>, k: usize, ef: usize) -> Vec<(String, f32)> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let mut curr_entry = self.entry_point.clone().unwrap();
        let mut curr_dist = self.cosine_distance(&query, &self.nodes.get(&curr_entry).unwrap().vector);

        for lc in (1..=self.max_layer).rev() {
            loop {
                let mut changed = false;
                if let Some(node) = self.nodes.get(&curr_entry) {
                    if let Some(neighbors) = node.neighbors.get(&lc) {
                        for neighbor_id in neighbors {
                            if let Some(neighbor_node) = self.nodes.get(neighbor_id) {
                                let dist = self.cosine_distance(&query, &neighbor_node.vector);
                                if dist < curr_dist {
                                    curr_dist = dist;
                                    curr_entry = neighbor_id.clone();
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        let top_candidates = self.search_layer(&query, &curr_entry, ef, 0);
        
        let mut results = Vec::new();
        let mut count = 0;
        let mut sorted_candidates: Vec<_> = top_candidates.into_iter().collect();
        sorted_candidates.sort_by(|a, b| a.distance.cmp(&b.distance));

        for element in sorted_candidates {
            if count >= k { break; }
            results.push((element.node_id, element.distance.0));
            count += 1;
        }

        results
    }
}

impl SocietalHNSW {
    fn search_layer(&self, query: &Vec<f32>, entry_point: &String, ef: usize, layer: u8) -> Vec<HeapElement> {
        let mut visited = HashSet::new();
        visited.insert(entry_point.clone());
        
        let mut candidates = BinaryHeap::new(); // Max-heap mimicking Min-heap due to custom Ord
        let mut results = BinaryHeap::new(); // Max-heap properties using reverse ord
        
        let init_dist = self.cosine_distance(query, &self.nodes.get(entry_point).unwrap().vector);
        
        candidates.push(HeapElement { distance: OrderedFloat(init_dist), node_id: entry_point.clone() });
        results.push(HeapElement { distance: OrderedFloat(init_dist), node_id: entry_point.clone() });

        while let Some(c) = candidates.pop() {
            if let Some(farthest) = results.peek() {
                if c.distance > farthest.distance {
                    break;
                }
            }
            
            if let Some(node) = self.nodes.get(&c.node_id) {
                if let Some(neighbors) = node.neighbors.get(&layer) {
                    for e in neighbors {
                        if !visited.contains(e) {
                            visited.insert(e.clone());
                            let f_dist = self.cosine_distance(query, &self.nodes.get(e).unwrap().vector);
                            
                            if results.len() < ef || OrderedFloat(f_dist) < results.peek().unwrap().distance {
                                candidates.push(HeapElement { distance: OrderedFloat(f_dist), node_id: e.clone() });
                                results.push(HeapElement { distance: OrderedFloat(f_dist), node_id: e.clone() });
                                
                                if results.len() > ef {
                                    // Remove the farthest element. BinaryHeap in Rust is max-heap. 
                                    // Our Ord is reversed, so the largest element is popped.
                                    let mut temp: Vec<_> = results.into_iter().collect();
                                    temp.sort_by(|a, b| a.distance.cmp(&b.distance));
                                    temp.pop();
                                    results = temp.into_iter().collect();
                                }
                            }
                        }
                    }
                }
            }
        }
        
        results.into_iter().collect()
    }

    fn select_neighbors(&self, mut candidates: Vec<HeapElement>, m: usize) -> Vec<String> {
        candidates.sort_by(|a, b| a.distance.cmp(&b.distance));
        candidates.into_iter().take(m).map(|e| e.node_id).collect()
    }

    fn cosine_distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            dot_product += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0; // max distance since similarity is 0
        }
        
        let sim = dot_product / (norm_a.sqrt() * norm_b.sqrt());
        1.0 - sim // cosine distance
    }
}
