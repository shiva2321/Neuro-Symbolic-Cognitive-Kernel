import unittest
from causal_reasoning import CausalDiscovery, CausalRelation

class TestCausalDiscovery(unittest.TestCase):
    def test_strong_causality(self):
        """Test detection of strong necessary and sufficient cause."""
        discovery = CausalDiscovery()
        ctx = "test"
        
        # Scenario: Action SWITCH_ON causes LIGHT_ON
        # 10 times: Switch ON -> Light ON
        for _ in range(10):
            discovery.observe(ctx, ["SWITCH_ON"], ["LIGHT_ON"])
            
        # 10 times: Switch OFF -> Light OFF
        for _ in range(10):
            discovery.observe(ctx, ["SWITCH_OFF"], ["LIGHT_OFF"])
            
        graph = discovery.induce_graph(ctx, min_confidence=0.5)
        
        # We expect SWITCH_ON -> LIGHT_ON
        # P(Light|Switch) = 10/10 = 1.0
        # P(Light|~Switch) = 0/10 = 0.0
        # Delta-P = 1.0
        
        links = graph.forward_chain("SWITCH_ON", max_depth=1)
        found = False
        for chain in links:
            if chain.end == "LIGHT_ON" and chain.total_strength > 0.9:
                found = True
                break
                
        self.assertTrue(found, "Should discover SWITCH_ON -> LIGHT_ON")
        
    def test_spurious_correlation(self):
        """Test rejection of ambient effects (correlation != causation)."""
        discovery = CausalDiscovery()
        ctx = "test"
        
        # Scenario: BIRD_CHIRPS happens all the time.
        # Action CLAP happens sometimes.
        
        # 10 times: CLAP -> BIRD_CHIRPS
        for _ in range(10):
            discovery.observe(ctx, ["CLAP", "SUNNY"], ["BIRD_CHIRPS"])
            
        # 10 times: WAIT -> BIRD_CHIRPS
        for _ in range(10):
            discovery.observe(ctx, ["WAIT", "SUNNY"], ["BIRD_CHIRPS"])
            
        # Analysis:
        # P(Chirp|Clap) = 1.0
        # P(Chirp|~Clap) = P(Chirp|Wait) = 1.0
        # Delta-P = 0.0
        
        graph = discovery.induce_graph(ctx, min_confidence=0.5)
        
        links = graph.forward_chain("CLAP", max_depth=1)
        # Should NOT find CLAP -> BIRD_CHIRPS
        found = False
        for chain in links:
            if chain.end == "BIRD_CHIRPS":
                found = True
        
        self.assertFalse(found, "Should NOT discover CLAP -> BIRD_CHIRPS (Delta-P is 0)")

if __name__ == '__main__':
    unittest.main()
