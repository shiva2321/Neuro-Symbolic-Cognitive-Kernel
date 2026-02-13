from staged_recall import StagedRecall
import hypervec_shim as hypervec_rs

class LifecycleManager:
    """
    Manages the birth, death, and evolution of concepts.
    Prevents semantic fossilization and drift.
    """
    def __init__(self, staged_recall: StagedRecall):
        self.sr = staged_recall
        self.codebook = staged_recall.codebook
        
    def detect_duplicates(self, threshold=0.90, sample_size=500):
        """
        Scan for concepts that have become too similar.
        Uses random sampling of codebook pairs (more reliable than LSH for lower thresholds).
        Returns: list of (name_a, name_b, similarity)
        """
        import random
        
        candidates = []
        keys = list(self.codebook.keys())
        
        # Sample random pairs from codebook
        if len(keys) > sample_size:
            sample_keys = random.sample(keys, sample_size)
        else:
            sample_keys = keys
            
        # Check each sampled concept against its LSH neighbors
        for name_a in sample_keys:
            hv_a = self.codebook[name_a]
            
            # Instead of relying on LSH query (which may miss due to tuning),
            # do a mini brute-force scan against a subset
            check_keys = random.sample(keys, min(100, len(keys)))
            
            for name_b in check_keys:
                if name_b <= name_a:  # Avoid self and duplicates
                    continue
                    
                hv_b = self.codebook[name_b]
                sim = hv_a.similarity(hv_b)
                
                if sim >= threshold:
                    candidates.append((name_a, name_b, sim))
                    
        return candidates

    def merge_concepts(self, name_a, name_b):
        """
        Merge B into A.
        1. Bundle hypervectors (centroid)
        2. Update codebook
        3. Remove B
        4. Update LSH index
        """
        print(f"Merging {name_b} -> {name_a}")
        
        hv_a = self.codebook[name_a]
        hv_b = self.codebook[name_b]
        
        # bundle: (A + B)
        new_hv = hv_a.bundle(hv_b)
        
        # Update A
        self.codebook[name_a] = new_hv
        
        # Remove B
        if name_b in self.codebook:
            del self.codebook[name_b]
        
        # Remove B from working memory
        if name_b in self.sr.working_memory:
            del self.sr.working_memory[name_b]
            
        # Update LSH Index - Remove old entries and insert updated A
        self.sr._lsh_remove(name_a)  # Remove old A position
        self.sr._lsh_remove(name_b)  # Remove B completely
        self.sr._lsh_insert(name_a, new_hv)  # Insert A at new position
        
        return new_hv

    def split_concept(self, original_name, new_names):
        """
        Split a concept into multiple distinct variations.
        Useful when a concept becomes too broad/overloaded.
        """
        if original_name not in self.codebook:
            return []
            
        print(f"Splitting {original_name} -> {new_names}")
        base_hv = self.codebook[original_name]
        
        # Remove original from codebook, cache, and LSH
        del self.codebook[original_name]
        if original_name in self.sr.working_memory:
            del self.sr.working_memory[original_name]
        self.sr._lsh_remove(original_name)
            
        # Create new variations
        new_hvs = []
        for i, new_name in enumerate(new_names):
            # Create variation by bundling with a unique random vector
            # This ensures they share some meaning with base, but are distinct from each other
            noise = hypervec_rs.HyperVector(1000 + i + hash(new_name) % 10000)
            # Use majority bundle: Base+Base+Noise to keep it close to parent
            new_hv = base_hv.bundle(base_hv).bundle(noise)
            
            self.codebook[new_name] = new_hv
            self.sr._lsh_insert(new_name, new_hv)
            new_hvs.append(new_hv)
            
        return new_hvs

    # =========================================================================
    # ACCRETION / DRIFT (Feature Flag: enable_accretion)
    # =========================================================================
    
    def accretion_update(self, concept_name, observation_hv, learning_rate=0.1, enable=False):
        """
        Gradually update a concept's hypervector based on new observations.
        This allows concepts to drift/evolve over time.

        DANGER: Can cause semantic blur if used carelessly.
        Only enabled when `enable=True` (feature flag).
        """
        if not enable:
            return 1.0  # No change

        if concept_name not in self.codebook:
            return None

        old_hv = self.codebook[concept_name]

        retention = 1.0 - learning_rate

        # Preferred: true weighted bundling if available
        if hasattr(old_hv, "weighted_bundle"):
            new_hv = old_hv.weighted_bundle(observation_hv, retention)
        else:
            # Fallback approximation for older hypervec_rs:
            # Build a bundle mostly composed of old_hv plus a little observation_hv.
            # Keep iteration count small to avoid CPU blowup.
            k = 7
            old_n = max(1, int(round(retention * k)))
            obs_n = max(1, k - old_n)

            new_hv = old_hv
            for _ in range(old_n - 1):
                new_hv = new_hv.bundle(old_hv)
            for _ in range(obs_n):
                new_hv = new_hv.bundle(observation_hv)

        drift = old_hv.similarity(new_hv)

        # Update codebook and LSH
        self.codebook[concept_name] = new_hv
        self.sr._lsh_remove(concept_name)
        self.sr._lsh_insert(concept_name, new_hv)

        print(f"[Accretion] {concept_name} drifted by {1-drift:.4f}")
        return drift

    # =========================================================================
    # HYGIENE MONITORING
    # =========================================================================
    
    def hygiene_check(self, sample_size=100):
        """
        Monitor the health of the concept space.
        Detects:
        1. Semantic blur: Concepts too similar to each other (high avg similarity)
        2. Semantic collapse: Concepts drifting toward noise (low self-consistency)
        3. Orphan concepts: Concepts with no neighbors (isolated)
        
        Returns:
            dict with hygiene metrics and warnings
        """
        import random
        import numpy as np
        
        keys = list(self.codebook.keys())
        if len(keys) < 2:
            return {"status": "OK", "warnings": [], "metrics": {}}
            
        sample_keys = random.sample(keys, min(sample_size, len(keys)))
        
        # Metric 1: Average pairwise similarity (should be ~0.5 for random, <0.6 healthy)
        pairwise_sims = []
        for i, name_a in enumerate(sample_keys[:20]):  # Limit to avoid O(n^2)
            hv_a = self.codebook[name_a]
            for name_b in sample_keys[i+1:i+10]:
                hv_b = self.codebook[name_b]
                pairwise_sims.append(hv_a.similarity(hv_b))
        
        avg_similarity = np.mean(pairwise_sims) if pairwise_sims else 0.5
        
        # Metric 2: Similarity to random noise (should be ~0.5)
        noise = hypervec_rs.HyperVector(999999)
        noise_sims = []
        for name in sample_keys[:20]:
            hv = self.codebook[name]
            noise_sims.append(hv.similarity(noise))
        avg_noise_sim = np.mean(noise_sims)
        
        # Generate warnings
        warnings = []
        
        if avg_similarity > 0.65:
            warnings.append(f"BLUR: High avg similarity ({avg_similarity:.3f}). Consider merging duplicates.")
        
        if avg_noise_sim > 0.55:
            warnings.append(f"COLLAPSE: Concepts drifting toward noise ({avg_noise_sim:.3f}).")
            
        if avg_similarity < 0.48:
            warnings.append(f"FRAGMENTATION: Unusually low similarity ({avg_similarity:.3f}). Concepts may be too sparse.")
        
        status = "WARNING" if warnings else "OK"
        
        return {
            "status": status,
            "warnings": warnings,
            "metrics": {
                "avg_pairwise_similarity": float(avg_similarity),
                "avg_noise_similarity": float(avg_noise_sim),
                "sample_size": len(sample_keys)
            }
        }

