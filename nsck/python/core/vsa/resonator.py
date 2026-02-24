"""
Resonator Network (Factorization)
=================================

Implements Iterative Resonator Networks for factorizing composite HyperVectors.
Reference: "Resonator Networks for Factorizing High-Dimensional Vectors", Kent et al.

Problem: Given S = A * B * C, finding A, B, C is hard (inverse problem).
Solution: Iteratively estimate each factor using the others as a query.
"""

from typing import List, Dict, Tuple, Optional
import python.core.vsa.hypervec_shim as hv
from python.core.memory.semantic_memory import SemanticMemory

class ResonatorNetwork:
    """ An iterative network to factorize a composite vector S into known components. """
    
    def __init__(self, codebooks: Dict[str, SemanticMemory], verbose: bool = False):
        """
        Args:
            codebooks: Map of factor name -> Semantic Memory containing valid values.
                       e.g. {'agent': agent_mem, 'action': action_mem}
            verbose: If True, prints convergence steps.
        """
        self.codebooks = codebooks
        self.factor_names = list(codebooks.keys())
        self.verbose = verbose
        
        # Cache for superposition initializers per codebook.
        # Key: factor_name, Value: (size_when_cached, superposition_hv)
        # Invalidated automatically when codebook size changes.
        self._super_cache: Dict[str, Tuple[int, hv.HyperVector]] = {}
        # Pre-calculate superposition of all items for initialization (optional optimization)
        self.initial_guesses: Dict[str, hv.HyperVector] = {}
        # NOTE: In a real large system, we might use a random vector or "Identity".
        # For now, we don't pre-calculate to avoid startup cost.

    def factorize(self, target_s: hv.HyperVector, max_iter: int = 3, convergence_threshold: float = 0.65) -> Dict[str, Tuple[str, float]]:
        """
        Factorize target_s into components.
        
        Returns:
             Dict {factor_name: (best_match_name, confidence)}
        """
        current_estimates: Dict[str, hv.HyperVector] = {}
        
        # Initialization Phase — use cached superposition per codebook.
        # Cache key is (factor_name, codebook size) so we rebuild when new
        # concepts are added but avoid the O(N) bundle loop on every call.
        for name in self.factor_names:
            mem = self.codebooks[name]
            mem_size = len(mem.concept_hvs) if hasattr(mem, "concept_hvs") else 0
            
            cached = self._super_cache.get(name)
            if cached is not None and cached[0] == mem_size:
                current_estimates[name] = cached[1]
            else:
                vecs = list(mem.concept_hvs.values())[:50] if hasattr(mem, "concept_hvs") else []
                if vecs:
                    superposition = vecs[0]
                    for v in vecs[1:]:
                        superposition = superposition.bundle(v)
                    self._super_cache[name] = (mem_size, superposition)
                    current_estimates[name] = superposition
                else:
                    fallback = hv.HyperVector(seed=123)
                    self._super_cache[name] = (0, fallback)
                    current_estimates[name] = fallback

        
        history = []
        
        # 2. Iteration Loop
        for i in range(max_iter):
            stable = True
            step_results = {}
            
            for f_target in self.factor_names:
                # To find A, we compute S * inv(B) * inv(C)...
                # Ideally, we bind S with the INVERSES of all other factors.
                
                # Start with S
                proposal = target_s
                
                # Unbind (Bind with Inverse) all other factors
                for f_other in self.factor_names:
                    if f_other == f_target:
                        continue
                    
                    est = current_estimates[f_other]
                    # Inverse of XOR is itself? Yes for binary.
                    # For HRR, it's Involution.
                    # hypervec_shim implements binary VSA (BSC).
                    # So Unbind == Bind (XOR).
                    # Check shim: permute_inverse exists, but standard bind (xor) is self-inverse.
                    proposal = proposal.xor(est) # Using XOR for binding/unbinding
                
                # Cleanup: Find nearest neighbor in codebook
                mem = self.codebooks[f_target]
                matches = mem.query(proposal, k=1)
                
                if matches:
                    best_name, confidence = matches[0]
                    # Get vector (Safe access)
                    best_vec = mem.concept_hvs.get(best_name)
                    if best_vec is None and hasattr(mem, "get_concept_vector"):
                        best_vec = mem.get_concept_vector(best_name) # Potential future API?
                        
                    if best_vec is None: 
                        # Should not happen if query found it in concept_hvs
                        continue
                    
                    # Store result
                    step_results[f_target] = (best_name, confidence)
                    
                    # Update estimate?
                    # "Hard" update: Set to the clean best_vec.
                    # "Soft" update: Bundle valid vector into current estimate.
                    # Standard Resonator uses "Hard" update (lock to prediction).
                    
                    old_est = current_estimates[f_target]
                    # Use fast Rust Hamming similarity (returns [0,1]) instead of
                    # cosine_similarity which requires expensive .bits extraction.
                    sim_change = best_vec.similarity(old_est)
                    
                    if sim_change < 0.99: # If changed significantly
                        stable = False
                        
                    current_estimates[f_target] = best_vec
                else:
                    # No match found? Keep old or random?
                    pass

            if self.verbose:
                print(f"Iter {i}: {step_results}")
                
            # Convergence Check
            # If all confidences are high enough?
            # Or if abstract "stable" flag is true.
            # Let's check min confidence.
            min_conf = min([res[1] for res in step_results.values()]) if step_results else 0.0
            
            if stable and min_conf > convergence_threshold:
                 if self.verbose: print(f"Converged at iter {i}")
                 return step_results
                 
        return step_results

    def reset_estimates(self):
        self.initial_guesses = {}


class HierarchicalResonatorNetwork:
    """
    2-level hierarchical resonator for nested structures like relative clauses.
    
    Level 1: sentence-level roles (AGENT, VERB, PATIENT)
    Level 2: clause-level roles (MODIFIER_AGENT, MODIFIER_VERB, MODIFIER_PATIENT)
    """
    
    L1_ROLES = ["AGENT", "VERB", "PATIENT", "THEME", "INSTRUMENT"]
    L2_ROLES = ["MODIFIER_AGENT", "MODIFIER_VERB", "MODIFIER_PATIENT"]
    
    def __init__(self, codebooks: Dict[str, SemanticMemory], verbose: bool = False):
        self.codebooks = codebooks
        self.verbose = verbose
        self._l1 = ResonatorNetwork(codebooks, verbose=verbose)
        self._l2 = ResonatorNetwork(codebooks, verbose=verbose)
    
    def factorize_hierarchical(self, composite_hv, depth: int = 2) -> Dict:
        """
        Factorize composite_hv at two levels.
        Returns dict with "L1" and (if depth>=2) "L2" entries.
        """
        result = {}
        result["L1"] = self._l1.factorize(composite_hv, max_iter=20)
        if depth >= 2:
            result["L2"] = self._l2.factorize(composite_hv, max_iter=20)
        return result
    
    def factorize(self, target_s, max_iter: int = 20, convergence_threshold: float = 0.65) -> Dict:
        """Compatibility method: factorize at L1."""
        return self._l1.factorize(target_s, max_iter=max_iter, convergence_threshold=convergence_threshold)
