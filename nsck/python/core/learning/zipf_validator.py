import math
from typing import Dict, List, Tuple
import numpy as np

class ZipfValidator:
    """
    Monitors the semantic graph to ensure its degree distribution 
    roughly follows a Zipfian/Power-Law distribution, which is characteristic
    of healthy, natural language and knowledge networks.
    """
    
    def __init__(self, expected_alpha: float = 1.1, tolerance: float = 0.3):
        """
        expected_alpha: Ideal power-law scaling parameter (often ~1.0 for Zipf)
        tolerance: Allowable deviation before triggering health warnings
        """
        self.expected_alpha = expected_alpha
        self.tolerance = tolerance
        
    def estimate_power_law_alpha(self, degrees: List[int]) -> float:
        """
        Estimate the power law parameter alpha using maximum likelihood.
        alpha = 1 + n / sum( ln( x_i / x_min ) )
        """
        if not degrees:
            return 0.0
            
        degrees = np.array(degrees, dtype=np.float64)
        degrees = degrees[degrees > 0] # Filter isolated nodes
        
        if len(degrees) < 3:
            return 0.0 # Not enough data
            
        x_min = np.min(degrees)
        if x_min == 0:
            x_min = 1.0 # Protect against log(0) if filter missed
            
        n = len(degrees)
        
        # MLE estimator for continuous power law approximation
        sum_ln = np.sum(np.log(degrees / x_min))
        if sum_ln == 0: 
            return 0.0 # Degenerate case where all degrees are equal
            
        alpha = 1.0 + n / sum_ln
        return float(alpha)
        
    def validate_health(self, degrees: List[int]) -> Tuple[bool, float]:
        """
        Validate if the graph's degree distribution is healthy.
        Returns: (is_healthy, current_alpha)
        """
        alpha = self.estimate_power_law_alpha(degrees)
        if alpha == 0.0:
            # If graph is too small, assume it's forming and not unhealthy per se
            return True, alpha
            
        # Is the scaling parameter close to expected?
        diff = abs(alpha - self.expected_alpha)
        is_healthy = diff <= self.tolerance
        
        return is_healthy, alpha
