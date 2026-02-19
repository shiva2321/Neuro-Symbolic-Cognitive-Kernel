import torch
import torch.nn.functional as F
from python.core.neural.snn_qat import TaskAwareSNN
import numpy as np

def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1, v2).item()

def run_probe(model_path="snn_task_aware.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize and Load Model
    # Note: We must call model first to initialize LazyLayers if we don't load state_dict immediately,
    # but since we are loading, we can just use the safe_globals fix or load with weights_only=False.
    model = TaskAwareSNN()
    
    print(f">> Loading model from {model_path}...")
    try:
        # We use weights_only=False here as a fallback if safe_globals didn't catch everything, 
        # but the fix in python_server.py should usually handle it.
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(">> Model loaded successfully.")
    except Exception as e:
        print(f">> [ERROR] Could not load model: {e}")
        print(">> Proceeding with randomly initialized model for demonstration.")

    model.to(device)
    model.eval()

    # 2. Define "Probe Concepts"
    print("\n>> DEFINING CONCEPTS FOR LATENT PROBING...")
    
    # Concept A: Visual stimulus (Snake Head near Food on Right)
    # [1, 4, 10, 10]
    visual_food_right = torch.zeros((1, 4, 10, 10), device=device)
    visual_food_right[0, 0, 5, 5] = 1.0 # Head at center
    visual_food_right[0, 1, 5, 6] = 1.0 # Food at right
    
    # Concept B: Visual stimulus (Snake Head near Wall on Left)
    visual_wall_left = torch.zeros((1, 4, 10, 10), device=device)
    visual_wall_left[0, 0, 5, 5] = 1.0 # Head
    visual_wall_left[0, 3, 5, 4] = 1.0 # Wall at left
    
    # Concept C: Conceptual stimulus (Text: "GO RIGHT")
    # In our UniversalEncoder, 2D input [B, Dim] is projected.
    # We'll use a dummy 256-dim vector for text embedding for now.
    text_right = torch.zeros((1, 256), device=device)
    text_right[0, :10] = 1.0 # Arbitrary pattern for "Right"
    
    # Concept D: Conceptual stimulus (Text: "DANGER")
    text_danger = torch.zeros((1, 256), device=device)
    text_danger[0, -10:] = 1.0 # Arbitrary pattern for "Danger"

    # 3. Extract Latent Representations
    with torch.no_grad():
        # latent = model.encoder(x)
        lat_visual_right = model.encoder(visual_food_right)
        lat_visual_danger = model.encoder(visual_wall_left)
        lat_text_right = model.encoder(text_right)
        lat_text_danger = model.encoder(text_danger)

    # 4. Compute Similarity Matrix
    concepts = {
        "Visual: Food Right": lat_visual_right,
        "Visual: Wall Left": lat_visual_danger,
        "Text: GO RIGHT": lat_text_right,
        "Text: DANGER": lat_text_danger
    }

    print("\n" + "="*60)
    print("LATENT SIMILARITY MATRIX (Mind's Eye Alignment)")
    print("="*60)
    
    names = list(concepts.keys())
    header = " " * 20
    for name in names:
        header += f"{name[:12]:>15}"
    print(header)

    for name1 in names:
        row = f"{name1[:18]:<20}"
        for name2 in names:
            sim = cosine_similarity(concepts[name1], concepts[name2])
            row += f"{sim:15.4f}"
        print(row)
    
    print("="*60)
    print("\n>> ANALYSIS:")
    
    # Check if Vision/Text are aligning
    v1 = lat_visual_right
    v2 = lat_text_right
    dist = cosine_similarity(v1, v2)
    print(f"[*] Visual(Food Right) <-> Text(GO RIGHT) Alignment: {dist:.4f}")
    
    v1 = lat_visual_danger
    v2 = lat_text_danger
    dist = cosine_similarity(v1, v2)
    print(f"[*] Visual(Wall Left) <-> Text(DANGER) Alignment: {dist:.4f}")

    if dist > 0.5:
        print("\n[RESULT] SUCCESS: Brain shows evidence of Multi-Modal Conceptual Transfer!")
    else:
        print("\n[RESULT] NOTE: Alignment is low. This is expected if the model hasn't seen paired data yet.")

if __name__ == "__main__":
    run_probe()
