
import pytest
import os
import torch
import shutil
import sys
from pathlib import Path

# Add python/ directory to sys.path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from intelligent_buffer import IntelligentReplayBuffer, Experience
from persistence import BrainStore

# Mock DB path for testing
TEST_DB_PATH = "test_brain_archive.db"

@pytest.fixture
def clean_db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_ram_limit_eviction(clean_db):
    """Verify that RAM buffer does not exceed capacity and evicts oldest."""
    capacity = 5
    buffer = IntelligentReplayBuffer(ram_capacity=capacity, archival_threshold=10.0, db_path=TEST_DB_PATH)
    
    # Fill buffer
    for i in range(capacity):
        exp = Experience(
            state=torch.zeros(1), action_idx=i, reward=0, next_state=torch.zeros(1), 
            done=False, task_name="test", priority=1.0 # Low priority
        )
        buffer.add(exp)
        
    assert len(buffer.ram_buffer) == capacity
    assert buffer.ram_buffer[0].action_idx == 0 # Oldest is 0
    
    # Add one more (Low Priority) -> Should evict 0, and NOT archive (priority 1.0 < threshold 10.0)
    exp_new = Experience(
        state=torch.zeros(1), action_idx=99, reward=0, next_state=torch.zeros(1), 
        done=False, task_name="test", priority=1.0
    )
    buffer.add(exp_new)
    
    assert len(buffer.ram_buffer) == capacity
    assert buffer.ram_buffer[0].action_idx == 1 # Oldest should now be 1
    assert buffer.ram_buffer[-1].action_idx == 99 # Newest is 99
    
    # Check Disk - should be empty because priority was low
    assert buffer.store.count_episodes() == 0

def test_archival_on_eviction(clean_db):
    """Verify that high priority eviction leads to archival."""
    capacity = 2
    buffer = IntelligentReplayBuffer(ram_capacity=capacity, archival_threshold=5.0, db_path=TEST_DB_PATH)
    
    # Add High Priority item that will be evicted
    # Oldest item: Priority 10.0 (High)
    exp1 = Experience(
        state=torch.zeros(1), action_idx=101, reward=0, next_state=torch.zeros(1), 
        done=False, task_name="test", priority=10.0
    )
    buffer.add(exp1)
    
    # Add another
    buffer.add(Experience(
        state=torch.zeros(1), action_idx=102, reward=0, next_state=torch.zeros(1), 
        done=False, task_name="test", priority=1.0
    ))
    
    assert len(buffer.ram_buffer) == 2
    
    # Add 3rd item -> Evicts exp1 (Priority 10.0)
    # Since 10.0 > 5.0, it should Archive.
    buffer.add(Experience(
        state=torch.zeros(1), action_idx=103, reward=0, next_state=torch.zeros(1), 
        done=False, task_name="test", priority=1.0
    ))
    
    # Flush store to ensure write
    buffer.store.flush_episodes()
    
    # Check Disk
    count = buffer.store.count_episodes("test")
    assert count == 1
    
    stored_episodes = buffer.store.load_recent_episodes("test")
    assert stored_episodes[0].action == "101" # The action of exp1
    assert stored_episodes[0].impact_score == 10.0
    
    # Verify Data Integrity
    saved_sketch = stored_episodes[0].state_sketch
    assert "data" in saved_sketch
    assert saved_sketch["data"].shape == (1,)
    assert saved_sketch["priority"] == 10.0

