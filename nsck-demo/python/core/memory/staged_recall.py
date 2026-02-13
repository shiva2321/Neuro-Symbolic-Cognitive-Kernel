import numpy as np
from collections import OrderedDict, defaultdict
import python.core.vsa.hypervec_shim as hypervec_rs

class StagedRecall:
    """
    Four-level retrieval hierarchy for NSGA:
    L0: Working Memory Cache (Active Set) - O(1)
    L1: Graph Neighborhood (k-hop) - O(k*d)
    L2: LSH Index (Approximate Retrieval) - O(log N)
    L3: Brute-Force Scan (Fallback) - O(N)
    """
    def __init__(self, codebook=None, cache_size=128):
        self.codebook = codebook if codebook else {}
        
        # L0: Working Memory (LRU Cache)
        # Using OrderedDict as LRU: move_to_end(key) on access, popitem(last=False) on evict
        self.working_memory = OrderedDict()
        self.cache_size = cache_size
        
        # L2: LSH Index - Tuned for recall/precision balance
        # 8 bits = 256 buckets, 32 tables for high recall
        # At 10k concepts: ~40 items per bucket on average
        self.n_lsh_bits = 8
        self.lsh_tables = [defaultdict(list) for _ in range(32)]

        # NOTE: On some platforms NumPy's legacy RandomState.randint uses int32 bounds.
        # 2**32 can overflow and raise ValueError. We generate uint32 seeds instead.
        self.lsh_seeds = [int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32)) for _ in range(32)]

        # Only enable LSH if the HyperVector implementation supports it.
        # Older builds of the Rust extension expose only xor/bundle/similarity.
        self.lsh_enabled = False
        if self.codebook:
            any_hv = next(iter(self.codebook.values()))
            self.lsh_enabled = hasattr(any_hv, "lsh_hash")

        # Populate LSH if codebook exists and LSH is supported
        if self.lsh_enabled:
            for name, hv in self.codebook.items():
                self._lsh_insert(name, hv)

        # Stats
        self.stats = {"l0_hits": 0, "l1_hits": 0, "l2_hits": 0, "l3_hits": 0, "queries": 0}

    def _lsh_insert(self, name, hv):
        """Insert concept into all LSH tables"""
        if not getattr(self, "lsh_enabled", False):
            return
        for i, seed in enumerate(self.lsh_seeds):
            sig = hv.lsh_hash(int(seed), self.n_lsh_bits)
            self.lsh_tables[i][sig].append(name)

    def _lsh_remove(self, name):
        """Remove concept from all LSH tables"""
        if not getattr(self, "lsh_enabled", False):
            return
        for table in self.lsh_tables:
            for sig, names in table.items():
                if name in names:
                    names.remove(name)

    def update_cache(self, concept_name, hv):
        """Update L0 cache with new concept usage"""
        if concept_name in self.working_memory:
            self.working_memory.move_to_end(concept_name)
        else:
            self.working_memory[concept_name] = hv
            if len(self.working_memory) > self.cache_size:
                self.working_memory.popitem(last=False)

    def query(self, query_hv, top_k=1, threshold=0.0):
        """
        Multi-stage retrieval.
        Returns: list of (name, similarity) tuples
        """
        self.stats["queries"] += 1
        
        # 1. L0: Check Working Memory
        best_l0 = []
        for name, hv in reversed(self.working_memory.items()):
            sim = query_hv.similarity(hv)
            if sim > 0.8: # High confidence hit
                best_l0.append((name, sim))
        
        if best_l0:
            best_l0.sort(key=lambda x: x[1], reverse=True)
            # Only early exit if we found enough high-quality matches
            if best_l0[0][1] > 0.9 and len(best_l0) >= top_k:
                self.stats["l0_hits"] += 1
                self.update_cache(best_l0[0][0], self.working_memory[best_l0[0][0]])
                return best_l0[:top_k]

        # 2. L1: Graph Neighborhood (Not implemented yet - requires topology access)
        
        # 3. L2: LSH Index (Approximate Retrieval)
        if getattr(self, "lsh_enabled", False) and hasattr(query_hv, "lsh_hash"):
            candidates = set()
            for i, seed in enumerate(self.lsh_seeds):
                sig = query_hv.lsh_hash(int(seed), self.n_lsh_bits)
                candidates.update(self.lsh_tables[i].get(sig, []))
                if len(candidates) > 200: # Max candidates cap
                    break

            if candidates:
                l2_results = []
                for name in candidates:
                    hv = self.codebook[name]
                    sim = query_hv.similarity(hv)
                    if sim > threshold:
                        l2_results.append((name, sim))

                l2_results.sort(key=lambda x: x[1], reverse=True)
                if l2_results:
                    self.stats["l2_hits"] += 1
                    self.update_cache(l2_results[0][0], self.codebook[l2_results[0][0]])
                    return l2_results[:top_k]

        # 4. L3: Brute-Force Fallback (Current Baseline)
        self.stats["l3_hits"] += 1
        results = []
        for name, hv in self.codebook.items():
            sim = query_hv.similarity(hv)
            if sim > threshold:
                results.append((name, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Update cache with winner
        if top_results:
            self.update_cache(top_results[0][0], self.codebook[top_results[0][0]])
            
        return top_results
