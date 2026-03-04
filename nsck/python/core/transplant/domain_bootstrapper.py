import numpy as np
import logging
import time
from typing import Dict, List, Tuple
from python.core.vsa.hypervec_shim import HyperVector
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld

logger = logging.getLogger("nsck.transplant.domain_bootstrapper")

class DomainBootstrapper:
    """
    Bootstraps Societal Knowledge Domains from dense neural codebooks (e.g. VQ-VAE, ViT).
    Instead of waiting for percolation, it pre-computes clusters on the dense embeddings,
    projects them to HyperVectors, and seeds them into the Societal World.
    
    Optimized for large scales (O(N) bonding instead of O(N^2)).
    """
    
    def __init__(self, world: SocietalKnowledgeWorld):
        self.world = world
        
    def bootstrap_from_codebook(self, codebook: np.ndarray, labels: List[str], n_domains: int = 5):
        """
        Takes a dense neural codebook of shape (N, D), clusters it,
        projects to the VSA space, and seeds the Societal World.
        """
        start_time = time.time()
        N, D = codebook.shape
        if len(labels) != N:
            raise ValueError("Labels and codebook length mismatch.")
            
        logger.info(f"Bootstrapping {N} neural embeddings into {n_domains} societal domains.")
        
        # 1. Clustering in dense space
        if N < n_domains:
            n_domains = max(1, N // 2)
            
        try:
            from sklearn.cluster import MiniBatchKMeans, KMeans
            if N > 1000:
                # Use MiniBatchKMeans for speed on large vocabularies (e.g. BERT 30k)
                logger.debug("Using MiniBatchKMeans for high-speed clustering.")
                kmeans = MiniBatchKMeans(n_clusters=n_domains, n_init=3, batch_size=1024)
            else:
                kmeans = KMeans(n_clusters=n_domains, n_init=10)
            clusters = kmeans.fit_predict(codebook)
        except ImportError:
            # Fallback random clustering
            logger.warning("scikit-learn not found, using random clustering assignment.")
            clusters = np.random.randint(0, n_domains, size=N)
            
        # 2. Project to VSA space and Ingest
        np.random.seed(42)
        # Using a stable random projection matrix
        proj_matrix = np.random.randn(self.world.dim, D).astype(np.float32)
        
        # Create a dictionary to hold nodes per cluster for bonding
        domain_nodes: Dict[int, List[str]] = {i: [] for i in range(n_domains)}
        
        logger.debug("Projecting and ingesting concepts...")
        for i in range(N):
            dense_vec = codebook[i]
            label = labels[i]
            
            # Random projection to sign-quantized HyperVector
            proj = np.dot(proj_matrix, dense_vec)
            
            # Use from_bits for correct initialization from bit array
            bits = (proj > 0).astype(np.int8)
            hv = HyperVector.from_bits(bits)
            
            self.world.ingest_concept(label, hv, source="transplant_bootstrapper")
            domain_nodes[clusters[i]].append(label)
            
        # 3. Optimized Intra-Domain Bonding (Hub-and-Spoke)
        # Instead of O(N^2) cliques, we use O(N) connections to establish connectivity.
        logger.debug("Establishing Hub-and-Spoke bonds for domain emergence...")
        
        for d_id, nodes in domain_nodes.items():
            if len(nodes) < 2:
                continue
            
            # Pick a small set of hubs to represent the domain "core"
            n_hubs = min(len(nodes), 8)
            hubs = nodes[:n_hubs]
            others = nodes[n_hubs:]
            
            # Bond hubs together (small clique: O(H^2))
            self.world.record_co_activation(hubs)
            
            # Bond every other node to 2 random hubs (O(N))
            for node in others:
                # Use a small subset of hubs to keep it sparse but connected
                selected_hubs = [hubs[idx] for idx in np.random.choice(n_hubs, size=min(2, n_hubs), replace=False)]
                for hub in selected_hubs:
                    # Repeating co-activation 5 times ensures high valence stability (Diamond class)
                    for _ in range(5):
                        self.world.record_co_activation([node, hub])
                        
        duration = time.time() - start_time
        logger.info(f"Bootstrapping complete in {duration:.2f}s. Emergence pending next world tick.")
