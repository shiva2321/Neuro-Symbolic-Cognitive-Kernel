import unittest
from global_workspace import GlobalWorkspace, WorkspaceModule

class MockModule(WorkspaceModule):
    def __init__(self, name):
        self.name = name
        self.received = []
        
    def receive_broadcast(self, content):
        print(f"[{self.name}] Received: {content}")
        self.received.append(content)

class TestGlobalWorkspace(unittest.TestCase):
    def test_competition_and_broadcast(self):
        """Test that the highest salience proposal wins and is broadcasted."""
        gw = GlobalWorkspace(attention_threshold=0.6)
        
        # 1. Setup modules
        vision = MockModule("Vision")
        memory = MockModule("Memory")
        gw.register_module("Vision", vision)
        gw.register_module("Memory", memory)
        
        # 2. Round 1: Vision wins (0.9 > 0.4)
        proposals_1 = {
            "Vision": ("Object detected: Snake", 0.9),
            "Memory": ("Recall: null", 0.4)
        }
        winner = gw.compete(proposals_1)
        
        self.assertEqual(winner, "Vision")
        self.assertEqual(gw.workspace_content, "Object detected: Snake")
        
        # Verify broadcast
        self.assertEqual(vision.received[-1], "Object detected: Snake")
        self.assertEqual(memory.received[-1], "Object detected: Snake") # Memory hears Vision
        
        # 3. Round 2: No one wins (both < 0.6)
        proposals_2 = {
            "Vision": ("Background noise", 0.2),
            "Memory": ("Faint memory", 0.3)
        }
        winner = gw.compete(proposals_2)
        
        self.assertIsNone(winner)
        # Content remains from previous or strictly implies "no update"?
        # Implementation just updates if threshold passed. so content is STALE or NONE?
        # Current impl: `self.workspace_content` is NOT cleared if no winner.
        # But `compete` returns None.
        
        # 4. Round 3: Memory wins (0.8 > 0.1)
        proposals_3 = {
            "Vision": ("Nothing new", 0.1),
            "Memory": ("Danger pattern recognized!", 0.8)
        }
        winner = gw.compete(proposals_3)
        
        self.assertEqual(winner, "Memory")
        self.assertEqual(vision.received[-1], "Danger pattern recognized!")
        
        print("\n[TEST] Global Workspace verification passed!")

if __name__ == '__main__':
    unittest.main()
