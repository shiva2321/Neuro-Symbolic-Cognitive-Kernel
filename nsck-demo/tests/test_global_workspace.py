import unittest
from global_workspace import GlobalWorkspace, WorkspaceModule, Coalition

class MockModule(WorkspaceModule):
    def __init__(self, name):
        self.name = name
        self.received = []
        
    def receive_broadcast(self, content):
        print(f"[{self.name}] Received: {content}")
        self.received.append(content)

class TestGlobalWorkspace(unittest.TestCase):
    def test_competition_and_broadcast(self):
        """Test that the highest activation Coalition wins and is broadcasted."""
        gw = GlobalWorkspace(attention_threshold=0.6)
        
        # 1. Setup modules
        vision = MockModule("Vision")
        memory = MockModule("Memory")
        gw.register_module("Vision", vision)
        gw.register_module("Memory", memory)
        
        # 2. Round 1: Vision wins (higher activation)
        proposals_1 = [
            Coalition(source="Vision", content="Object detected: Snake", base_salience=0.9),
            Coalition(source="Memory", content="Recall: null", base_salience=0.4)
        ]
        winner = gw.compete(proposals_1)
        
        self.assertIsNotNone(winner)
        self.assertEqual(winner.source, "Vision")
        self.assertEqual(gw.workspace_content, "Object detected: Snake")
        
        # Verify broadcast
        self.assertEqual(vision.received[-1], "Object detected: Snake")
        self.assertEqual(memory.received[-1], "Object detected: Snake")
        
        # 3. Round 2: No one wins (both below threshold)
        proposals_2 = [
            Coalition(source="Vision", content="Background noise", base_salience=0.05),
            Coalition(source="Memory", content="Faint memory", base_salience=0.1)
        ]
        winner = gw.compete(proposals_2)
        
        self.assertIsNone(winner)
        
        # 4. Round 3: Memory wins
        proposals_3 = [
            Coalition(source="Vision", content="Nothing new", base_salience=0.1),
            Coalition(source="Memory", content="Danger pattern recognized!", base_salience=0.8)
        ]
        winner = gw.compete(proposals_3)
        
        self.assertIsNotNone(winner)
        self.assertEqual(winner.source, "Memory")
        self.assertEqual(vision.received[-1], "Danger pattern recognized!")
        
        print("\n[TEST] Global Workspace verification passed!")

if __name__ == '__main__':
    unittest.main()
