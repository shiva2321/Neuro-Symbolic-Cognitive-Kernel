"""
Quick verification that RuleLearner implements WorkspaceModule interface correctly.
"""
import sys

from python.core.reasoning.rule_learner import RuleLearner
from python.core.reasoning.global_workspace import WorkspaceModule, Coalition
from python.core.perception.grounding_verifier import GroundingVerifier
from python.core.integration.persistence import BrainStore
import numpy as np

def test_rule_learner_interface():
    """Verify RuleLearner implements all required WorkspaceModule methods."""
    
    # Create instance
    verifier = GroundingVerifier()  # No args needed
    learner = RuleLearner(verifier=verifier, min_support=3, min_confidence=0.3)
    
    # Check inheritance
    assert isinstance(learner, WorkspaceModule), "RuleLearner must inherit from WorkspaceModule"
    
    # Check required methods exist
    assert hasattr(learner, 'receive_broadcast'), "Missing receive_broadcast()"
    assert hasattr(learner, 'propose'), "Missing propose()"
    assert hasattr(learner, 'update'), "Missing update()"
    assert hasattr(learner, 'get_telemetry'), "Missing get_telemetry()"
    
    # Test method signatures
    learner.set_current_task("snake")
    
    # Test receive_broadcast
    learner.receive_broadcast({"winner": "OtherModule", "action": "move_up"})
    
    # Test propose
    state_hv = np.random.randint(0, 2, 10240, dtype=np.uint8)
    proposal = learner.propose(state_hv)
    assert proposal is None or isinstance(proposal, Coalition), "propose() must return Coalition or None"
    
    # Test update
    feedback_hv = np.random.randint(0, 2, 10240, dtype=np.uint8)
    learner.update(feedback_hv, reward=1.0, info={'winner': 'RuleLearner', 'action_taken': 'move_up'})
    
    # Test telemetry
    telemetry = learner.get_telemetry()
    assert isinstance(telemetry, dict), "get_telemetry() must return dict"
    assert 'active' in telemetry, "Telemetry must include 'active'"
    assert 'proposals_count' in telemetry, "Telemetry must include 'proposals_count'"
    assert 'wins_count' in telemetry, "Telemetry must include 'wins_count'"
    assert 'win_rate' in telemetry, "Telemetry must include 'win_rate'"
    assert 'confidence' in telemetry, "Telemetry must include 'confidence'"
    assert 'memory_size' in telemetry, "Telemetry must include 'memory_size'"
    
    print("✅ RuleLearner implements WorkspaceModule interface correctly!")
    print(f"✅ Telemetry: {telemetry}")

if __name__ == "__main__":
    test_rule_learner_interface()
