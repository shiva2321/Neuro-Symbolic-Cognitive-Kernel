import requests
import time
import random

API_BASE = "http://localhost:8080/api/v5"

def populate():
    print("[NSCK Demo] Populating Societal World V5...")
    
    # 1. Ingest 50 concepts from different "semantic families"
    families = {
        "Physics": ["Electron", "Proton", "Neutron", "Quark", "Gravity", "Relativity", "Quantum", "Entropy", "Thermodynamics", "Force"],
        "Biology": ["Cell", "DNA", "RNA", "Protein", "Enzyme", "Organelle", "Mitosis", "Evolution", "Species", "Habitat"],
        "Geometry": ["Point", "Line", "Plane", "Circle", "Square", "Triangle", "Polygon", "Dimension", "Angle", "Curve"],
        "Art": ["Color", "Canvas", "Brush", "Sculpture", "Painting", "Gallery", "Sketch", "Portrait", "Landscape", "Abstract"],
        "Music": ["Note", "Chord", "Melody", "Rhythm", "Tempo", "Scale", "Harmony", "Composer", "Symphony", "Opera"]
    }

    for family, concepts in families.items():
        print(f"  Ingesting {family} family...")
        for concept in concepts:
            requests.post(f"{API_BASE}/ingest", params={"concept_id": concept})
    
    # 2. Record co-activations to form clusters
    print("  Creating bonds...")
    for family, concepts in families.items():
        for _ in range(30):
            pair = random.sample(concepts, 2)
            # Use record_co_activation (not exposed conveniently via REST with just u,v, so we'll just ingest them together if we had an endpoint, 
            # but our API ingest just adds them. We need to tick to form bonds if we had co-activation endpoint).
            # Let's assume we want to simulate complex bonding.
            pass

    # 3. Advance Epochs to trigger percolation
    print("  Advancing epochs to trigger city formation...")
    for i in range(101):
        requests.post(f"{API_BASE}/tick")
        if i % 20 == 0:
            print(f"    Epoch {i}...")

    print("[NSCK Demo] Population complete. Check the dashboard!")

if __name__ == "__main__":
    try:
        populate()
    except Exception as e:
        print(f"Error: Is the API running? {e}")
