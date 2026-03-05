import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set

from python.core.vsa.hypervec_shim import HyperVector
from python.core.vsa.living_hv import LivingHyperVector, ValenceEngine
from python.core.vsa.hodge_laplacian import HodgeLaplacian
from python.core.memory.knowledge_neighborhood import KnowledgeNeighborhood, KnowledgeDomain
from python.core.memory.societal_hnsw import SocietalHNSW
from python.core.learning.spectral_laplacian_rg import SpectralLaplacianRG
from python.core.learning.percolation_monitor import PercolationMonitor
from python.core.learning.zipf_validator import ZipfValidator

class SocietalKnowledgeWorld:
    """
    The orchestrator for the V5 Associative Memory logic.
    Governs the birth of LivingHyperVectors, Domain Percolation,
    Valence Engine logic, Spectral Coarse Graining, and HNSW routing.
    """
    
    def __init__(self, dim: int = 10240):
        self.dim = dim
        self.registry: Dict[str, LivingHyperVector] = {}
        self.domains: Dict[str, KnowledgeDomain] = {}
        
        # Engines and Monitors
        self.valence_engine = ValenceEngine(min_co_activations=2)
        self.hnsw = SocietalHNSW()
        self.spectral_rg = SpectralLaplacianRG(tau=0.5)
        self.percolator = PercolationMonitor(threshold=0.60)
        self.zipf_validator = ZipfValidator()
        self.laplacian = None # Rebuilt periodically
        self.last_zipf_alpha = 1.0 # Initial healthy assumption
        
        self.epoch_ticker = 0
        self._pending_co_activations = []
        
    def ingest_concept(self, concept_id: str, vector: HyperVector, source: str = "native") -> LivingHyperVector:
        """Add or retrieve a concept.

        The HNSW index insertion is throttled: only every 20th concept triggers
        an immediate HNSW insert (O(log N) to O(N) search).  All concepts are
        always registered in ``self.registry`` so the LivingHyperVector metadata
        is complete.  A full HNSW rebuild happens during ``tick_world`` (every
        100 ticks).  This avoids O(N²) cost during bulk loads (e.g. adding 1K+
        concepts in a loop).
        """
        if concept_id in self.registry:
            lhv = self.registry[concept_id]
            # Drift existing towards newer occurrence 10%
            lhv.hv = lhv.hv.weighted_bundle(vector, weight=0.9)
            lhv.activate(self.epoch_ticker)
            return lhv

        lhv = LivingHyperVector(hv=vector, concept_id=concept_id, source=source)
        lhv.activate(self.epoch_ticker)
        self.registry[concept_id] = lhv

        # Throttle HNSW insertions: every 20th new concept gets indexed immediately;
        # the rest are picked up during the next tick_world rebuild.
        self._hnsw_insert_counter = getattr(self, '_hnsw_insert_counter', 0) + 1
        if self._hnsw_insert_counter % 20 == 1:
            self.hnsw.insert_node(concept_id, layer=0, vector=vector)

        return lhv

    def record_co_activation(self, concepts: List[str]):
        """Record episodic co-occurrence for valence bonding."""
        valid = [c for c in concepts if c in self.registry]
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                u, v = valid[i], valid[j]
                self.valence_engine.record_co_activation(u, v)
                # Queue for processing in tick_world
                self._pending_co_activations.append((u, v))

    def tick_world(self):
        """
        Advance global epoch. Resolve bonds, decay stale relationships, 
        and occasionally trigger macroscopic coarse-graining (city formation).
        """
        self.epoch_ticker += 1
        
        # 1. Decay and Bonding
        for lhv in self.registry.values():
            self.valence_engine.remove_stale_bonds(lhv, self.epoch_ticker)
            
        while self._pending_co_activations:
            u, v = self._pending_co_activations.pop(0)
            self.valence_engine.try_bind(self.registry[u], self.registry[v])
            
        # 2. Rebuild Adjacency Graph Periodically
        if self.epoch_ticker % 100 == 0:
            self._rebuild_global_topology()
            self._detect_domain_percolations()

    def _rebuild_global_topology(self):
        """Construct Hodge Laplacian to analyze the entire graph structure."""
        nodes = list(self.registry.keys())
        self.laplacian = HodgeLaplacian(nodes)
        
        for u in nodes:
            for v, weight in self.registry[u].current_bonds.items():
                self.laplacian.add_edge(u, v, weight)
                
    def _detect_domain_percolations(self):
        """
        Use Renormalization Group / Percolation logic to identify emergent semantic domains.
        Supports hierarchical promotion: Town -> City -> State -> Country -> Continent.
        """
        if not self.laplacian or len(self.registry) < 10:
            return
            
        nodes = self.laplacian.node_list
        W = self.laplacian.W
        
        # Check components
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if W[i, j] > 0:
                    edges.append((i, j, float(W[i, j])))
                    
        is_percolated, max_size, comp_sizes, node_to_root = self.percolator.check_transition(len(nodes), edges)
        
        # Mapping component sizes to hierarchy levels
        # 10-50: Town
        # 51-200: City
        # 201-1000: State
        # 1001-5000: Country
        # >5000: Continent
        
        significant_components = [root for root, size in comp_sizes.items() if size >= 10]
        
        for root_idx in significant_components:
            size = comp_sizes[root_idx]
            level = "Town"
            if size > 5000: level = "Continent"
            elif size > 1000: level = "Country"
            elif size > 200: level = "State"
            elif size > 50: level = "City"
            
            comp_node_indices = [i for i, r in enumerate(node_to_root) if r == root_idx]
            comp_nodes = [nodes[i] for i in comp_node_indices]
            sub_W = W[np.ix_(comp_node_indices, comp_node_indices)]
            
            # Use Spectral RG to find internal neighborhood structure
            n_macrostates = min(5, max(1, len(comp_nodes) // 20))
            supernodes = self.spectral_rg.create_supernodes(comp_nodes, sub_W, num_macrostates=n_macrostates)
            
            # Find or Create Domain
            unique_domain_id = f"dom_{root_idx}"
            if unique_domain_id not in self.domains:
                domain_name = f"{level}_{self.epoch_ticker}_{root_idx}"
                self.domains[unique_domain_id] = KnowledgeDomain(domain_name=domain_name, hierarchy_level=level)
                self.domains[unique_domain_id].emergence_epoch = self.epoch_ticker
            
            domain = self.domains[unique_domain_id]
            domain.hierarchy_level = level # Update if grown
            
            for k, (sub_id, members) in enumerate(supernodes.items()):
                n_id = f"nh_{unique_domain_id}_{k}"
                
                # Bundle members for anchor
                hvs = [self.registry[m].hv for m in members]
                if not hvs: continue
                
                anchor = hvs[0]
                for idx in range(1, len(hvs)):
                    anchor = anchor.bundle(hvs[idx])
                
                nh = KnowledgeNeighborhood(
                    neighborhood_id=n_id, 
                    domain_name=domain.domain_name, 
                    anchor_hv=anchor,
                    parent_id=unique_domain_id
                )
                for m in members:
                    nh.add_member(m)
                    self.registry[m].update_domain_affinity(domain.domain_name, 1.0)
                    self.registry[m].neighborhood_id = n_id
                    # Insert into HNSW Layer 1 (Neighborhoods)
                    self.hnsw.insert_node(m, layer=1, vector=self.registry[m].hv)
                
                domain.add_neighborhood(nh)
            
            domain.update_global_centroid()
            # Insert Domain into HNSW Layer 2 (Cities/States)
            self.hnsw.insert_node(unique_domain_id, layer=2, vector=domain.global_centroid)

        # Check Zipf health Law
        degrees = [len(lhv.current_bonds) for lhv in self.registry.values()]
        healthy, alpha = self.zipf_validator.validate_health(degrees)
        self.last_zipf_alpha = alpha
        if not healthy:
            print(f"[Societal World] Warning: Network Zipf Alpha {alpha:.2f} deviates from healthy parameters.")
            
    def semantic_search(self, query: HyperVector, k: int = 10) -> List[Tuple[str, float]]:
        """Search utilizing the HNSW spatial routing."""
        return self.hnsw.search(query, k=k, ef=50)
