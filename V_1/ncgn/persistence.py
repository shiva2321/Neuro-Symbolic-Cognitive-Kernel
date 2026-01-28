"""
NCGN v7 Brain Persistence

Save and load brain state to/from disk for continuity across sessions.
Supports JSON format for human readability and NumPy arrays for efficiency.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

from .config import Config
from .topology import GraphTopology
from .state import CognitiveState
from .brain import Brain


class BrainPersistence:
    """
    Save and load Brain state to disk.
    
    Storage format:
    - brain_config.json: Configuration and metadata
    - brain_topology.json: Graph structure (nodes, edges)
    - brain_state.npz: NumPy arrays (activations, thresholds, etc.)
    """
    
    @staticmethod
    def save(brain: Brain, save_dir: str, name: str = "brain") -> str:
        """
        Save brain state to disk.
        
        Args:
            brain: Brain instance to save
            save_dir: Directory to save files
            name: Base name for files
            
        Returns:
            Path to saved directory
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save config and metadata
        meta = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "version": "7.0",
            "config": {
                "initial_capacity": brain.config.initial_capacity,
                "decay_delta": brain.config.decay_delta,
                "flow_alpha": brain.config.flow_alpha,
                "default_threshold": brain.config.default_threshold,
                "refractory_period": brain.config.refractory_period,
                "learning_rate": brain.config.learning_rate,
                "weight_min": brain.config.weight_min,
                "weight_max": brain.config.weight_max,
                "norm_beta": brain.config.norm_beta,
                "seizure_threshold": brain.config.seizure_threshold,
                "seizure_damping": brain.config.seizure_damping,
            },
            "stats": brain.get_stats(),
            "properties": brain._properties,
        }
        
        with open(save_path / f"{name}_config.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        
        # 2. Save topology (nodes and edges)
        topology_data = {
            "nodes": [],
            "edges": []
        }
        
        # Extract all labels and their indices
        for label in brain.topology.registry.all_labels():
            idx = brain.topology.registry.get_index(label)
            topology_data["nodes"].append({
                "label": label,
                "index": idx
            })
        
        # Extract all edges with weights
        for source in brain.topology.registry.all_labels():
            neighbors = brain.topology.get_neighbors(source)
            for target in neighbors:
                weight = brain.topology.get_edge_weight(source, target)
                topology_data["edges"].append({
                    "source": source,
                    "target": target,
                    "weight": weight
                })
        
        with open(save_path / f"{name}_topology.json", "w") as f:
            json.dump(topology_data, f, indent=2)
        
        # 3. Save state arrays (NumPy)
        n = brain.topology.num_nodes
        np.savez_compressed(
            save_path / f"{name}_state.npz",
            activations=brain.state.activations[:n],
            thresholds=brain.state.thresholds[:n],
            refractory_counters=brain.state.refractory_counters[:n],
            resting_potentials=brain.state.resting_potentials[:n],
            novelty_scores=brain.state.novelty_scores[:n],
        )
        
        # 4. Save embeddings if available
        if brain.state.embeddings is not None and n > 0:
            np.savez_compressed(
                save_path / f"{name}_embeddings.npz",
                embeddings=brain.state.embeddings[:n]
            )
        
        return str(save_path)
    
    @staticmethod
    def load(save_dir: str, name: str = "brain", use_embeddings: bool = True) -> Brain:
        """
        Load brain state from disk.
        
        Args:
            save_dir: Directory containing saved files
            name: Base name of files
            use_embeddings: Whether to load semantic layer
            
        Returns:
            Restored Brain instance
        """
        save_path = Path(save_dir)
        
        # 1. Load config and metadata
        with open(save_path / f"{name}_config.json", "r") as f:
            meta = json.load(f)
        
        # Reconstruct config
        cfg_data = meta.get("config", {})
        config = Config(
            initial_capacity=cfg_data.get("initial_capacity", 10000),
            decay_delta=cfg_data.get("decay_delta", 0.05),
            flow_alpha=cfg_data.get("flow_alpha", 0.9),
            default_threshold=cfg_data.get("default_threshold", 0.75),
            refractory_period=cfg_data.get("refractory_period", 3),
            learning_rate=cfg_data.get("learning_rate", 0.02),
            weight_min=cfg_data.get("weight_min", 0.0),
            weight_max=cfg_data.get("weight_max", 1.0),
            norm_beta=cfg_data.get("norm_beta", 0.1),
            seizure_threshold=cfg_data.get("seizure_threshold", 50.0),
            seizure_damping=cfg_data.get("seizure_damping", 0.5),
        )
        
        # Create brain
        brain = Brain(config=config, use_embeddings=use_embeddings)
        
        # 2. Load topology
        with open(save_path / f"{name}_topology.json", "r") as f:
            topology_data = json.load(f)
        
        # Add nodes
        for node in topology_data["nodes"]:
            brain.add_concept(node["label"])
        
        # Add edges
        for edge in topology_data["edges"]:
            brain.connect(edge["source"], edge["target"], weight=edge["weight"])
        
        # 3. Load state arrays
        state_path = save_path / f"{name}_state.npz"
        if state_path.exists():
            with np.load(state_path) as data:
                n = len(data["activations"])
                brain.state.activations[:n] = data["activations"]
                brain.state.thresholds[:n] = data["thresholds"]
                brain.state.refractory_counters[:n] = data["refractory_counters"]
                brain.state.resting_potentials[:n] = data["resting_potentials"]
                brain.state.novelty_scores[:n] = data["novelty_scores"]
        
        # 4. Load embeddings if available
        emb_path = save_path / f"{name}_embeddings.npz"
        if emb_path.exists() and brain.state.embeddings is not None:
            with np.load(emb_path) as data:
                n = len(data["embeddings"])
                brain.state.embeddings[:n] = data["embeddings"]
        
        # 5. Restore properties
        brain._properties = meta.get("properties", {})
        
        return brain
    
    @staticmethod
    def list_saved_brains(save_dir: str) -> list:
        """List all saved brains in a directory."""
        save_path = Path(save_dir)
        if not save_path.exists():
            return []
        
        brains = []
        for f in save_path.glob("*_config.json"):
            name = f.stem.replace("_config", "")
            with open(f, "r") as file:
                meta = json.load(file)
            brains.append({
                "name": name,
                "saved_at": meta.get("saved_at"),
                "nodes": meta.get("stats", {}).get("topology", {}).get("num_nodes", 0),
                "edges": meta.get("stats", {}).get("topology", {}).get("num_edges", 0),
            })
        
        return sorted(brains, key=lambda x: x.get("saved_at", ""), reverse=True)
    
    @staticmethod
    def delete_brain(save_dir: str, name: str) -> bool:
        """Delete a saved brain."""
        save_path = Path(save_dir)
        files = [
            f"{name}_config.json",
            f"{name}_topology.json",
            f"{name}_state.npz",
            f"{name}_embeddings.npz",
        ]
        
        deleted = False
        for f in files:
            path = save_path / f
            if path.exists():
                path.unlink()
                deleted = True
        
        return deleted
