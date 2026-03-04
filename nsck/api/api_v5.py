import os
import sys
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure NSCK root is in path
_nsck_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _nsck_root not in sys.path:
    sys.path.insert(0, _nsck_root)

from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld
from python.core.vsa.hypervec_shim import HyperVector

app = FastAPI(title="NSCK Societal World API V5", version="5.0.0")

# Enable CORS for the dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global World Instance (In a real app, this might be a singleton or DB-backed)
world = SocietalKnowledgeWorld()

@app.get("/api/v5/status")
async def get_status():
    return {
        "epoch": world.epoch_ticker,
        "population": len(world.registry),
        "domains": len(world.domains),
        "hnsw_nodes": world.hnsw.get_node_count() if hasattr(world.hnsw, 'get_node_count') else len(world.registry),
        "zipf_alpha": world.last_zipf_alpha
    }

@app.get("/api/v5/world")
async def get_world_hierarchy():
    """
    Returns the full hierarchical structure: 
    Continent -> Country -> State -> City -> Town -> Area -> Person
    """
    hierarchy = []
    for dom_id, domain in world.domains.items():
        dom_data = {
            "id": dom_id,
            "name": domain.domain_name,
            "level": domain.hierarchy_level,
            "health": domain.measure_health(),
            "neighborhoods": []
        }
        for nh_id, nh in domain.neighborhoods.items():
            dom_data["neighborhoods"].append({
                "id": nh_id,
                "name": f"District_{nh_id}",
                "members_count": len(nh.members),
                "stability": nh.stability_score,
                "social_energy": nh.social_energy,
                "valence": nh.collective_valence
            })
        hierarchy.append(dom_data)
    return hierarchy

@app.get("/api/v5/domain/{domain_id}")
async def get_domain_details(domain_id: str):
    if domain_id not in world.domains:
        raise HTTPException(status_code=404, detail="Domain not found")
    domain = world.domains[domain_id]
    return {
        "id": domain_id,
        "name": domain.domain_name,
        "level": domain.hierarchy_level,
        "neighborhoods": list(domain.neighborhoods.keys()),
        "health_history": domain.tda_health_history,
        "entropy": domain.graph_entropy
    }

@app.get("/api/v5/neighborhood/{nh_id}")
async def get_neighborhood_details(nh_id: str):
    # Search domains for this neighborhood
    for domain in world.domains.values():
        if nh_id in domain.neighborhoods:
            nh = domain.neighborhoods[nh_id]
            return {
                "id": nh_id,
                "domain": nh.domain_name,
                "members": list(nh.members),
                "stability": nh.stability_score,
                "energy": nh.social_energy,
                "betti": {"b0": nh.betti_0, "b1": nh.betti_1}
            }
    raise HTTPException(status_code=404, detail="Neighborhood not found")

@app.get("/api/v5/person/{concept_id}")
async def get_person_details(concept_id: str):
    if concept_id not in world.registry:
        raise HTTPException(status_code=404, detail="Concept not found")
    lhv = world.registry[concept_id]
    return {
        "id": concept_id,
        "activation": lhv.activation,
        "last_active": lhv.last_active_epoch,
        "bonds": list(lhv.current_bonds.keys()),
        "neighborhood": lhv.neighborhood_id,
        "affinity": lhv.domain_affinities
    }

@app.post("/api/v5/tick")
async def tick_world():
    world.tick_world()
    return {"status": "success", "epoch": world.epoch_ticker}

@app.post("/api/v5/ingest")
async def ingest_concept(concept_id: str, seed: Optional[int] = None):
    # Simple seed-based HV for demo/dashboard
    if seed is None:
        seed = abs(hash(concept_id)) % (2**31)
    vec = HyperVector(seed=seed)
    world.ingest_concept(concept_id, vec)
    return {"status": "ingested", "id": concept_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
