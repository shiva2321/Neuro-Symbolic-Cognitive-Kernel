
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from global_workspace import GlobalWorkspace, Coalition
from homeostasis import HomeostaticMonitor

def test_consciousness_competition():
    print("Initializing Global Workspace (LIDA-Lite)...")
    workspace = GlobalWorkspace(attention_threshold=0.3)
    
    # 1. Setup Drives (Proto-Self)
    proto_self = HomeostaticMonitor()
    
    # Scene 1: Well-fed, looking for fun.
    # Energy = 1.0 -> Hunger = 0.0
    print("\n[Scene 1] Agent is Full (Energy=1.0)")
    proto_self.energy = 1.0
    proto_self.update()
    drives = proto_self.drives
    print(f"Drives: {drives}")
    
    # Proposals
    # Coalition A: "Play Snake" (High Base Salience, Low Relevance to Hunger)
    c_snake = Coalition(
        source="planner",
        content="ACTION_PLAY_SNAKE",
        base_salience=0.8,
        relevance=0.5,
        affect_match=0.0  # Does not satisfy any high drive
    )
    
    # Coalition B: "Eat Apple" (Low Base Salience, High Relevance to Hunger)
    c_eat = Coalition(
        source="perception",
        content="ACTION_EAT_APPLE",
        base_salience=0.2,
        relevance=0.1,
        # Affect match depends on hunger
        affect_match=drives['hunger'] * 2.0  # Simple heuristic: food satisfies hunger
    )
    
    winner = workspace.compete([c_snake, c_eat])
    print(f"Winner: {winner.content if winner else 'None'} (Snake Act: {c_snake.activation:.2f}, Eat Act: {c_eat.activation:.2f})")
    
    if winner.content == "ACTION_PLAY_SNAKE":
        print("[SUCCESS] Agent correctly chose to Play when full.")
    else:
        print("[FAILURE] Agent made wrong choice when full.")
        
    # Scene 2: Starving
    print("\n[Scene 2] Agent is Starving (Energy=0.1)")
    proto_self.energy = 0.1
    proto_self.update() # Updates drives
    drives = proto_self.drives
    print(f"Drives: {drives}")
    
    # Update Affect Match based on new drives
    c_eat.affect_match = drives['hunger'] * 2.0 # Huge bonus from Hunger
    
    winner = workspace.compete([c_snake, c_eat])
    print(f"Winner: {winner.content if winner else 'None'} (Snake Act: {c_snake.activation:.2f}, Eat Act: {c_eat.activation:.2f})")
    
    if winner.content == "ACTION_EAT_APPLE":
        print("[SUCCESS] Agent correctly chose to Eat when starving (Affect override).")
    else:
        print("[FAILURE] Agent ignored hunger.")

if __name__ == "__main__":
    test_consciousness_competition()
