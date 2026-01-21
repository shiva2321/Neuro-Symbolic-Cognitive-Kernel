"""
NCGN Graph Visualizer - Real-time Brain State Visualization

Provides multiple visualization backends:
- NetworkX + Matplotlib for static/saved images
- JSON export for web-based D3.js visualization
- Terminal-based using rich library (optional)

Visualizes:
- Node energy levels (color gradient)
- Synapse weights/confidence (edge thickness/color)
- Active/firing nodes (highlighting)
- System state metrics
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import colorsys

# Try to import optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ColorScheme(Enum):
    """Color schemes for visualization."""
    ENERGY = "energy"      # Blue (low) -> Red (high)
    NOVELTY = "novelty"    # Green (old) -> Yellow (new)
    SURPRISE = "surprise"  # White (low) -> Purple (high)


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    width: int = 1200
    height: int = 800
    node_size_base: int = 300
    node_size_scale: float = 500  # Energy multiplier
    edge_width_base: float = 0.5
    edge_width_scale: float = 3.0  # Weight multiplier
    show_labels: bool = True
    show_weights: bool = False
    show_energy: bool = True
    color_scheme: ColorScheme = ColorScheme.ENERGY
    layout: str = "spring"  # spring, circular, kamada_kawai, shell
    background_color: str = "#1a1a2e"  # Dark background
    active_node_glow: bool = True


class GraphState:
    """
    Captures the current state of the graph for visualization.
    
    Serializable to JSON for web visualization.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.tick: int = 0
        self.active_nodes: Set[str] = set()
        self.firing_nodes: Set[str] = set()
    
    def add_node(
        self,
        node_id: str,
        energy: float = 0.0,
        threshold: float = 0.75,
        novelty: float = 1.0,
        is_active: bool = False,
        is_firing: bool = False,
        properties: Optional[Dict] = None
    ):
        """Add a node to the state."""
        self.nodes[node_id] = {
            "id": node_id,
            "energy": energy,
            "threshold": threshold,
            "novelty": novelty,
            "is_active": is_active,
            "is_firing": is_firing,
            "properties": properties or {}
        }
        if is_active:
            self.active_nodes.add(node_id)
        if is_firing:
            self.firing_nodes.add(node_id)
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 0.5,
        confidence: float = 0.5,
        edge_type: str = "associates",
        is_active: bool = False
    ):
        """Add an edge to the state."""
        self.edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "confidence": confidence,
            "type": edge_type,
            "is_active": is_active
        })
    
    def set_metadata(self, **kwargs):
        """Set visualization metadata."""
        self.metadata.update(kwargs)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tick": self.tick,
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "activeNodes": list(self.active_nodes),
            "firingNodes": list(self.firing_nodes),
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save state to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GraphState':
        """Load state from dictionary."""
        state = cls()
        state.tick = data.get("tick", 0)
        state.metadata = data.get("metadata", {})
        
        for node in data.get("nodes", []):
            state.add_node(
                node_id=node["id"],
                energy=node.get("energy", 0.0),
                threshold=node.get("threshold", 0.75),
                novelty=node.get("novelty", 1.0),
                is_active=node.get("is_active", False),
                is_firing=node.get("is_firing", False),
                properties=node.get("properties", {})
            )
        
        for edge in data.get("edges", []):
            state.add_edge(
                source=edge["source"],
                target=edge["target"],
                weight=edge.get("weight", 0.5),
                confidence=edge.get("confidence", 0.5),
                edge_type=edge.get("type", "associates"),
                is_active=edge.get("is_active", False)
            )
        
        return state


class GraphVisualizer:
    """
    Main visualization class for NCGN brain state.
    
    Supports multiple output formats:
    - Static images (PNG, SVG) via Matplotlib
    - JSON for web visualization
    - GraphML for external tools
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.history: List[GraphState] = []
        self.max_history = 100
    
    def capture_state(
        self,
        memory,  # GraphMemory
        engine,  # System1Engine
        firing_set: Optional[Set[str]] = None
    ) -> GraphState:
        """
        Capture current graph state from memory and engine.
        
        Args:
            memory: GraphMemory instance
            engine: System1Engine instance
            firing_set: Set of nodes that fired this tick
        
        Returns:
            GraphState snapshot
        """
        state = GraphState()
        state.tick = engine.current_tick
        
        active_nodes = memory.get_active_nodes()
        firing = firing_set or engine.firing_set
        
        # Add all nodes
        for node_id, node in memory.nodes.items():
            state.add_node(
                node_id=node_id,
                energy=node.energy,
                threshold=node.threshold,
                novelty=node.novelty_score,
                is_active=node_id in active_nodes,
                is_firing=node_id in firing
            )
        
        # Add all edges
        for source_id, synapses in memory.forward_edges.items():
            for synapse in synapses:
                # Edge is active if source is firing
                is_active = source_id in firing
                state.add_edge(
                    source=source_id,
                    target=synapse.target_id,
                    weight=synapse.weight,
                    confidence=synapse.confidence,
                    edge_type=synapse.type,
                    is_active=is_active
                )
        
        # Add metadata
        state.set_metadata(
            surprise=engine.surprise_level,
            system2_triggered=engine.system2_triggered,
            paused=engine.paused,
            total_nodes=memory.node_count(),
            total_edges=memory.edge_count(),
            active_count=len(active_nodes),
            firing_count=len(firing)
        )
        
        # Store in history
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return state
    
    def _energy_to_color(self, energy: float) -> str:
        """Convert energy level to hex color (blue -> red gradient)."""
        # Clamp energy to [0, 1]
        energy = max(0.0, min(1.0, energy))
        
        # HSV interpolation: blue (240°) -> red (0°)
        hue = (1.0 - energy) * 0.66  # 0.66 = blue in HSV
        saturation = 0.8
        value = 0.6 + energy * 0.4  # Brighter when more energy
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    def _confidence_to_color(self, confidence: float) -> str:
        """Convert confidence to edge color."""
        # Gray (low) -> White (high)
        gray = int(100 + confidence * 155)
        return f"#{gray:02x}{gray:02x}{gray:02x}"
    
    def render_matplotlib(
        self,
        state: GraphState,
        filepath: Optional[str] = None,
        show: bool = True
    ) -> Optional[str]:
        """
        Render graph state using Matplotlib.
        
        Args:
            state: GraphState to render
            filepath: Optional path to save image
            show: Whether to display the plot
        
        Returns:
            Filepath if saved, None otherwise
        """
        if not HAS_NETWORKX or not HAS_MATPLOTLIB:
            print("Warning: NetworkX and Matplotlib required for rendering")
            return None
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in state.nodes.items():
            G.add_node(node_id, **node_data)
        
        # Add edges
        for edge in state.edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"],
                confidence=edge["confidence"]
            )
        
        # Calculate layout
        if self.config.layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif self.config.layout == "circular":
            pos = nx.circular_layout(G)
        elif self.config.layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.shell_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(self.config.width / 100, self.config.height / 100),
            facecolor=self.config.background_color
        )
        ax.set_facecolor(self.config.background_color)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        for edge in state.edges:
            color = self._confidence_to_color(edge["confidence"])
            if edge["is_active"]:
                color = "#ffff00"  # Yellow for active
            edge_colors.append(color)
            edge_widths.append(
                self.config.edge_width_base + 
                edge["weight"] * self.config.edge_width_scale
            )
        
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node_id in G.nodes():
            node_data = state.nodes[node_id]
            color = self._energy_to_color(node_data["energy"])
            if node_data["is_firing"]:
                color = "#ffffff"  # White for firing
            node_colors.append(color)
            
            size = (
                self.config.node_size_base + 
                node_data["energy"] * self.config.node_size_scale
            )
            node_sizes.append(size)
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=1
        )
        
        # Draw labels
        if self.config.show_labels:
            labels = {}
            for node_id in G.nodes():
                if self.config.show_energy:
                    energy = state.nodes[node_id]["energy"]
                    labels[node_id] = f"{node_id}\n({energy:.2f})"
                else:
                    labels[node_id] = node_id
            
            nx.draw_networkx_labels(
                G, pos, labels, ax=ax,
                font_size=8,
                font_color='white'
            )
        
        # Add title with metadata
        title = f"NCGN Brain State - Tick {state.tick}"
        if state.metadata:
            title += f"\nSurprise: {state.metadata.get('surprise', 0):.2f}"
            title += f" | Active: {state.metadata.get('active_count', 0)}"
        ax.set_title(title, color='white', fontsize=12)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save if filepath provided
        if filepath:
            plt.savefig(
                filepath,
                facecolor=self.config.background_color,
                edgecolor='none',
                dpi=100
            )
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def export_json(self, state: GraphState, filepath: str) -> str:
        """Export graph state to JSON for web visualization."""
        state.save(filepath)
        return filepath
    
    def export_graphml(self, state: GraphState, filepath: str) -> str:
        """Export graph to GraphML format for external tools."""
        if not HAS_NETWORKX:
            raise RuntimeError("NetworkX required for GraphML export")
        
        G = nx.DiGraph()
        
        for node_id, node_data in state.nodes.items():
            G.add_node(
                node_id,
                energy=node_data["energy"],
                threshold=node_data["threshold"],
                novelty=node_data["novelty"],
                active=node_data["is_active"],
                firing=node_data["is_firing"]
            )
        
        for edge in state.edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"],
                confidence=edge["confidence"],
                edge_type=edge["type"]
            )
        
        nx.write_graphml(G, filepath)
        return filepath
    
    def get_d3_compatible_json(self, state: GraphState) -> dict:
        """
        Get JSON structure compatible with D3.js force-directed graph.
        
        Returns format expected by D3.js:
        {
            "nodes": [{"id": "...", "group": 1, ...}],
            "links": [{"source": "...", "target": "...", "value": ...}]
        }
        """
        nodes = []
        for node_id, node_data in state.nodes.items():
            nodes.append({
                "id": node_id,
                "group": 1 if node_data["is_active"] else 0,
                "energy": node_data["energy"],
                "radius": 10 + node_data["energy"] * 20,
                "color": self._energy_to_color(node_data["energy"]),
                "firing": node_data["is_firing"]
            })
        
        links = []
        for edge in state.edges:
            links.append({
                "source": edge["source"],
                "target": edge["target"],
                "value": edge["weight"],
                "confidence": edge["confidence"],
                "active": edge["is_active"]
            })
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": state.metadata
        }


def quick_visualize(memory, engine, filepath: Optional[str] = None):
    """
    Quick visualization helper function.
    
    Usage:
        from ui.graph_visualizer import quick_visualize
        quick_visualize(memory, engine, "brain_state.png")
    """
    viz = GraphVisualizer()
    state = viz.capture_state(memory, engine)
    
    if filepath:
        if filepath.endswith('.json'):
            return viz.export_json(state, filepath)
        elif filepath.endswith('.graphml'):
            return viz.export_graphml(state, filepath)
        else:
            return viz.render_matplotlib(state, filepath, show=False)
    else:
        return viz.render_matplotlib(state, show=True)
